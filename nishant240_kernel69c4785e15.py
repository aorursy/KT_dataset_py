import pandas as pd

from sklearn import preprocessing

import numpy as np

import numpy as np

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

import torch

import torch.nn as nn

from torch.autograd import Variable

import torch.utils.data as data_utils

import torch.nn.functional as F

from sklearn.model_selection import train_test_split

from tqdm import tqdm

from sklearn.preprocessing import LabelBinarizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# from google.colab import drive

# drive.mount('/content/drive')
#read dataset

path = "../input/nishant-attention-notebook/data.csv"

df = pd.read_csv(path)



#label encoder

le = preprocessing.LabelEncoder()

le.fit(df.category)

df['label'] = le.transform(df.category)



#train test split

x_train, x_test, y_train, y_test = train_test_split(df.text,df.label, test_size=0.1, random_state=1)



#tokenize text

tokenizer = Tokenizer(num_words = 10000)

tokenizer.fit_on_texts(x_train)

word_index = tokenizer.word_index
#using glove embeddings

embeddings_index = {}

glove_path = "../input/glove-global-vectors-for-word-representation/glove.6B.100d.txt"

with open(glove_path,encoding="utf8") as f:

    for line in tqdm(f):

        values = line.split()

        word = values[0]

        coefs = np.asarray(values[1:], dtype='float32')

        embeddings_index[word] = coefs



print('Found %s word vectors.' % len(embeddings_index))
#setting parameters

MAX_NUM_WORDS = 400000

MAX_SEQUENCE_LENGTH = 1000

EMBEDDING_DIM = 100



#creating embedding matrix

num_words = min(MAX_NUM_WORDS, len(word_index)) + 1

embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))

for word, i in word_index.items():

    if i > MAX_NUM_WORDS:

        continue

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None:

        # words not found in embedding index will be all-zeros.

        embedding_matrix[i] = embedding_vector

embedding_matrix=torch.Tensor(embedding_matrix)
#label binarizer

lb = LabelBinarizer()

y_train=lb.fit_transform(y_train)



#preparing input

x_train = tokenizer.texts_to_sequences(x_train)

x_train = pad_sequences(x_train, maxlen = 500)



#input torch

x_torch_train = torch.from_numpy(np.array(x_train)).to(device)

y_torch_train = torch.from_numpy(np.array(y_train)).to(device)



#Data Loader

train = data_utils.TensorDataset(x_torch_train, y_torch_train)

train_loader = data_utils.DataLoader(train, batch_size=32, shuffle=True)
class SelfAttention(nn.Module):

    def __init__(self, embedding_matrix,batch_size, output_size, hidden_size,vocab_size, embedding_length,r):

        super(SelfAttention, self).__init__()

        # We will use da = 350, r = 30 & penalization_coeff = 1 as per given in the self-attention original ICLR paper

        self.batch_size = batch_size

        self.output_size = output_size

        self.hidden_size = hidden_size

        self.r = r

        self.vocab_size = vocab_size

        self.embedding_length = embedding_length

        self.word_embeddings = nn.Embedding(self.vocab_size, self.embedding_length)

        self.word_embeddings.weights = nn.Parameter(embedding_matrix, requires_grad=False)

        self.dropout = 0

        self.hidden = self.init_hidden()

        self.bilstm = nn.LSTM(embedding_length, hidden_size, dropout=self.dropout, bidirectional=True,batch_first =True)

        self.W_s1 = nn.Linear(2*hidden_size, 350)

        self.W_s2 = nn.Linear(350, self.r)

        self.label = nn.Linear(2*hidden_size, output_size)

        

    def init_hidden(self,features=400):

        return (Variable(torch.zeros(2,features,self.hidden_size,device=device)),

                Variable(torch.zeros(2,features,self.hidden_size,device=device)))



    def attention_net(self, lstm_output):

        

        #attention score

        attn_weight_matrix = self.W_s2(F.tanh(self.W_s1(lstm_output)))

        

        #attention weights

        attn_weight_matrix = F.softmax(attn_weight_matrix, dim=1)

        

        return attn_weight_matrix

    

    def forward(self, input_sentences):

        

        #Embedding

        inp = self.word_embeddings(input_sentences)         #input shape: (batch_size, max_len, dim)

        

        #BiLSTM

        output, self.hidden = self.bilstm(inp, self.hidden)

        output = output.to(device)                          #output:  (batch_size, max_len, 2*hidden_dim)

        

        #attention Weight matrix

        attn_weight_matrix = self.attention_net(output)     #attn wts: (batch_size,maxlen,self.r)

        

        #context vector 

        hidden_matrix = torch.bmm(attn_weight_matrix.permute(0,2,1), output.to(device))    #hidden_matrix: (batch_size,self.r,2*hidden_dim)

        

        #sentence embedding

        sen_emebedding = torch.sum(hidden_matrix,1)/self.r      #sen_embedding shape: (batch_size,2*hidden_dim)

        

        #FC layer

        logits = self.label(sen_emebedding)

        logits = F.softmax(logits)

        

        return logits,attn_weight_matrix

    

    

#model

selfattn = SelfAttention(embedding_matrix,32,5,64,embedding_matrix.shape[0],embedding_matrix.shape[1],30)

selfattn = selfattn.to(device)



#loss & optimizer

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(selfattn.parameters(), lr=0.01)
iter = 0

epochs=5

for epoch in range(epochs):

    correct = 0

    total_count = 0

    for i, (features, labels) in enumerate(train_loader):

        

        # features = Variable(features)

        features = torch.tensor(features,dtype=torch.long)

        # features = features.to(device)



        labels = torch.tensor(labels, dtype = torch.float).to(device)

        # labels = labels.to(device)

        # labels = Variable(labels,requires_grad=True)



        #Clear the gradients

        optimizer.zero_grad()

        selfattn.hidden = selfattn.init_hidden(features.shape[0])

        

        #Forward propagation 

        outputs,wts = selfattn(features.to(device))

        

        #loss

        loss = criterion(outputs, labels.to(device))

        

        #Backward propation

        loss.backward()

        

        #Updating gradients

        optimizer.step()

        iter += 1

    

        predicted = abs(torch.round(outputs))



        #Total number of labels

        total = labels.size(0)

        

        #Calculating the number of correct predictions

        for j in range(total):

            if np.array_equal(predicted.cpu().detach().numpy()[j],labels.detach().cpu().numpy()[j]) == True:

                correct+=1

        total_count += len(labels)

        

    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'

                  .format(epoch + 1, epochs, i + 1, len(train_loader), loss.item(),

                          (correct / total_count) * 100))
#label binarizer

#lb = LabelBinarizer()

y_test=lb.fit_transform(y_test)



#preparing input

x_test = tokenizer.texts_to_sequences(x_test)

x_test = pad_sequences(x_test, maxlen = 500)



#input torch

x_torch_test = torch.from_numpy(np.array(x_test)).to(device)

y_torch_test = torch.from_numpy(np.array(y_test)).to(device)



#for visualization

tmp = x_test



#Data Loader

# train = data_utils.TensorDataset(x_torch_train, y_torch_train)

# train_loader = data_utils.DataLoader(train, batch_size=32, shuffle=True)
test = data_utils.TensorDataset(x_torch_test, y_torch_test)

test_loader = data_utils.DataLoader(test, batch_size=32, shuffle=True)
with torch.no_grad():

    testcorrect = 0

    totaltest = 0

    for testfeatures,testlabels in test_loader:

        testfeatures = Variable(testfeatures)

        testfeatures = torch.tensor(testfeatures,dtype=torch.long).to(device)

        testlabels = torch.tensor(testlabels, dtype = torch.float).to(device)

        testlabels = Variable(testlabels,requires_grad=True)

        selfattn.hidden = selfattn.init_hidden(testfeatures.shape[0])

        testoutputs,testwts = selfattn(testfeatures)

        testpredicted = abs(torch.round(testoutputs))

        for j in range(testlabels.size(0)):

            if np.array_equal(testpredicted.detach().cpu().numpy()[j],testlabels.detach().cpu().numpy()[j]) == True:

                testcorrect+=1

        totaltest+=len(testlabels)

        # print(testcorrect)

    print("Test Accuracy: {:.2f}%".format((testcorrect/totaltest)*100))
def visualize_attention_new(wts,x_test_pad,i2w,filename):

    wts_add = torch.sum(wts,1)

    wts_add = wts_add.data.cpu().numpy()

    wts_add = wts_add.tolist()

#     id_to_word = {v:k for k,v in word_to_id.items()}

#     print(id_to_word)

    text= []

    print(len(x_test_pad))

    

    for test in x_test_pad:       

        # for i in test:

        #     print(i)

        #     print("word:",i2w.get(i))

        text.append(" ".join([str(i2w.get(i)) for i in test]))

    print(text)

    createHTML(text, wts_add, filename)

    print("Attention visualization created for {} samples".format(len(x_test_pad)))

    return
def createHTML(texts, weights, fileName):

    

    """

    Creates a html file with text heat.

    weights: attention weights for visualizing

    texts: text on which attention weights are to be visualized

    """

    fileName =fileName

    fOut = open(fileName, "w", encoding="utf-8")

    part1 = """

    <html lang="en">

    <head>

    <meta http-equiv="content-type" content="text/html; charset=utf-8">

    <style>

    body {

    font-family: Sans-Serif;

    }

    </style>

    </head>

    <body>

    <h3>

    Heatmaps

    </h3>

    </body>

    <script>

    """

    part2 = """

    var color = "255,0,0";

    var ngram_length = 3;

    var half_ngram = 1;

    for (var k=0; k < any_text.length; k++) {

    var tokens = any_text[k].split(" ");

    var intensity = new Array(tokens.length);

    var max_intensity = Number.MIN_SAFE_INTEGER;

    var min_intensity = Number.MAX_SAFE_INTEGER;

    for (var i = 0; i < intensity.length; i++) {

    intensity[i] = 0.0;

    for (var j = -half_ngram; j < ngram_length-half_ngram; j++) {

    if (i+j < intensity.length && i+j > -1) {

    intensity[i] += trigram_weights[k][i + j];

    }

    }

    if (i == 0 || i == intensity.length-1) {

    intensity[i] /= 2.0;

    } else {

    intensity[i] /= 3.0;

    }

    if (intensity[i] > max_intensity) {

    max_intensity = intensity[i];

    }

    if (intensity[i] < min_intensity) {

    min_intensity = intensity[i];

    }

    }

    var denominator = max_intensity - min_intensity;

    for (var i = 0; i < intensity.length; i++) {

    intensity[i] = (intensity[i] - min_intensity) / denominator;

    }

    if (k%2 == 0) {

    var heat_text = "<p><br><b>Example:</b><br>";

    } else {

    var heat_text = "<b>Example:</b><br>";

    }

    var space = "";

    for (var i = 0; i < tokens.length; i++) {

    heat_text += "<span style='background-color:rgba(" + color + "," + intensity[i] + ")'>" + space + tokens[i] + "</span>";

    if (space == "") {

    space = " ";

    }

    }

    //heat_text += "<p>";

    document.body.innerHTML += heat_text;

    }

    </script>

    </html>"""

    putQuote = lambda x: "\"%s\""%x

    textsString = "var any_text = [%s];\n"%(",".join(map(putQuote, texts)))

    weightsString = "var trigram_weights = [%s];\n"%(",".join(map(str,weights)))

#     print(textsString)

#     print(weightsString)

    fOut.write(part1)

    fOut.write(textsString)

    fOut.write(weightsString)

    fOut.write(part2)

    fOut.close()

  

    return
id_to_word = {v:k for k,v in word_index.items()}
visualize_attention_new(testwts,tmp,id_to_word,'../input/nishant-attention-notebook/attn.html')