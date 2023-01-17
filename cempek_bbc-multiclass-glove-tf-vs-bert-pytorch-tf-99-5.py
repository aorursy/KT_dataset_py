import os
import math
import pandas as pd
import numpy as np
import tensorflow as tf
import itertools
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
stopw = stopwords.words('english')

import seaborn as sns
import matplotlib.pyplot as plt
from plotly.offline import iplot
import cufflinks
cufflinks.go_offline()
cufflinks.set_config_file(world_readable = True , offline = False)


from transformers import AdamW
import transformers

from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from transformers import BertTokenizer, BertConfig, BertForSequenceClassification
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case = True)

from tqdm import tqdm
df = pd.read_csv('../input/bbc-fulltext-and-category/bbc-text.csv')

from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()
df['label'] = enc.fit_transform(df['category'])

class_names = df.groupby(['category', 'label']).count().reset_index().loc[:,['category', 'label']]
class_name_tokenizers = {}
for class_name in class_names.category:
    class_name_tokenizers[class_name] = [tokenizer.encode(class_name)[1],class_names[class_names['category'] == class_name]['label'].iloc[0]]
#Class names are encoded one hot encoding method and encoded with Bert tokenizer.
class_name_tokenizers
category_counts = df['category'].value_counts()
categories = category_counts.index

fig = plt.figure(figsize = (12,5))
ax = fig.add_subplot(111)
sns.barplot(x = category_counts.index , y = category_counts)
for a, p in enumerate(ax.patches):
    ax.annotate(f'{categories[a]}\n' + format(p.get_height(), '.0f'), xy = (p.get_x() + p.get_width() / 2.0, p.get_height()), xytext = (0,-25), size = 13, color = 'white' , ha = 'center', va = 'center', textcoords = 'offset points', bbox = dict(boxstyle = 'round', facecolor='none',edgecolor='white', alpha = 0.5) )
plt.xlabel('Categories', size = 15)
plt.ylabel('The Number of News', size= 15)
plt.xticks(size = 12)

plt.title("The number of News by Categories" , size = 18)
plt.show()
def clean(text, punctuation = False, stopword = False):
    # filter to allow only alphabets
#    text = re.sub(r'[^a-zA-Z\']', ' ', text)
    # Deleting Stopwords
    if stopword == True:
        text = re.sub(r'[^a-zA-Z\']', ' ', text)
    
        text = text.split()
        text = [word for word in text if word not in stopw]
        text = " ".join(text)
    # Seperating the punctuations to tokenize them
    if punctuation == True:
        punc = '@#!?+&*[]-%.:/();$£=><|{}^'
        for p in punc:
            text = text.replace(p, f' {p} ')    

    # Urls
    text = re.sub(r"https?:\/\/t.co\/[A-Za-z0-9]+", "", text)
    text = re.sub(r"  ", " ", text)
    text = re.sub(r"   ", " ", text)
    return text
      
df['cleaned_from_stopw'] = df['text'].apply(lambda x : clean(x, stopword = True))
df['cleaned'] = df["text"].apply(lambda x : clean(x,punctuation = True))
df["count"] = df["cleaned"].apply(lambda x: len(x.split()))
plt.figure(figsize = (8,8))
sns.distplot(df['count'] )
plt.xlim(0,1000)
plt.xlabel('The number of words', fontsize = 16)
plt.title("The Number of Words Distribution", fontsize = 18)
plt.show()
df =df.iloc[:-64,:].reset_index(drop= True)
df_test = df.iloc[-64:,:].reset_index(drop= True)
def create_n_gram(df, category= '',text_column = 'cleaned_from_stopw' ,n_gram = 1):
    n_gram_dict = {}
    if category != '':                                              # This condition is created in case of filtering the label.
        df = df[df['category'] == category]

    for k in tqdm(df['cleaned_from_stopw']):

        for i in range(len(k)):
            words = k.split()[i:i+n_gram]
            words = " ".join(words)
            
            if (len(words.split()) % n_gram) > 0 or words == '' :       # This condition is created to drop last words of text.  
                continue
            elif words in n_gram_dict.keys():                           
                n_gram_dict[words] +=1
            else:                                                      # We add new word into dictionary. If the word is already in the dictionary, it adds 1 in values.
                n_gram_dict[words] =1

    results = pd.DataFrame.from_dict([n_gram_dict]).T.reset_index()
    results.columns = [category+'_n_grams',category+ '_counts']

    return results
all_words = create_n_gram(df)
print("There are {} unique words in the dataset.".format(len(all_words)))
n_gram_dict = {}
for class_name in class_name_tokenizers.keys():
    for i in range(1,4):
        temp_result =create_n_gram(df,category = class_name, n_gram = i)
        temp_result = temp_result.sort_values(by = class_name + "_counts", ascending = False).head(30).reset_index(drop = True)
        temp_result.columns = [class_name + '_n_grams'+ str(i),class_name + '_counts_'+str(i)]
        n_gram_dict[class_name + str(i)] = temp_result
np_result = np.ones((30,1))
n_gram_result = pd.DataFrame(np_result)
n_gram_result.drop(columns = [0], inplace= True)
for key in n_gram_dict.keys():
    n_gram_result = n_gram_result.join(n_gram_dict[key])
n_gram_result.iloc[:,:6]
n_gram_result.iloc[:,6:12]
n_gram_result.iloc[:,12:18]
n_gram_result.iloc[:,18:24]
n_gram_result.iloc[:,24:30]
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
glove_embeddings = np.load('../input/pickled-glove840b300d-for-10sec-loading/glove.840B.300d.pkl', allow_pickle=True)
print("There are {} words and {} dimensions in Glove Dictionary. And, the number of dimenson is {}. I used the word 'sister' as an example".format(len(glove_embeddings.keys()),len(glove_embeddings['sister']),len(glove_embeddings['sister'])))

count = 0
uncovered_words = {}
covered_words = {}
embedding_matrix = {}
for text in df.cleaned:
    text = text.split()
    for word in text:
        
        if word not in glove_embeddings.keys():
            count += 1
            if word not in uncovered_words:
                uncovered_words[word] = 1
            else:
                uncovered_words[word] += 1
        
        else:
            if word not in covered_words:
                covered_words[word] = 1
            else:
                covered_words[word] += 1
print("---There are {} words in the whole dataset, and {:.2f}% of the words aren't covered by Glove---".format((len(uncovered_words) + len(covered_words)),len(uncovered_words) / (len(uncovered_words)+len(covered_words))*100))
print('---Top 20 most commong uncovered words---')
print(pd.DataFrame([uncovered_words]).T.reset_index().sort_values(by = 0, ascending = False).head(20))
df['cleaned'][0]
tokenizer_keras = Tokenizer(num_words = 29479, oov_token = "<OOV>")
tokenizer_keras.fit_on_texts(df['cleaned'])
word_index = tokenizer_keras.word_index                # After tokenization, we get all the words and characters in the dataset. 
vocab_size_keras = len(word_index)
embedding_dim = len(glove_embeddings['the'] )          # All of words in glove have same dimmensions(300), so we choose one example "the".
list(word_index.items())[0:10]
tokenizer_keras = Tokenizer(num_words = 29479, oov_token = "<OOV>")
tokenizer_keras.fit_on_texts(df['text'])
word_index = tokenizer_keras.word_index                # After tokenization, we get all the words and characters in the dataset. 
vocab_size_keras = len(word_index)


# Creating embedding matrix for all words in dataset
embedding_matrix = np.zeros((vocab_size_keras+1,embedding_dim))       # We added 1 to vocab size because tokenizer starts with 1, so 0th is not gonna used
for word, i in word_index.items():
    if word in glove_embeddings.keys():
        embedding_vector = glove_embeddings[word]
        embedding_matrix[i] = embedding_vector
tokenized = pd.DataFrame([word_index]).T.reset_index()
tokenized.columns = ['words','index']
temp_emd_matrix = pd.DataFrame(embedding_matrix).reset_index()
temp_emd_matrix = temp_emd_matrix.drop(0, axis = 0)
df_embedding_matrix = pd.merge(tokenized, temp_emd_matrix, on = 'index')
df_embedding_matrix.rename(columns = {"index": "tokens"})
df_embedding_matrix
def prepare_data(df,tokenizer, max_len= 64):
    sequences = tokenizer.texts_to_sequences(df['cleaned'])
    padded = pad_sequences(sequences, maxlen = max_len, padding = 'post', truncating = 'post')
    labels = tf.keras.utils.to_categorical(df['label'])
    return padded, labels
padded, labels = prepare_data(df_test,tokenizer_keras, max_len = 512)
training_portion =0.75
training_size = int(len(df) * training_portion)
padded_training = padded[:training_size]
labels_training = labels[:training_size]
padded_val = padded[training_size:]
labels_val = labels[training_size:]

model_glove = tf.keras.Sequential()
model_glove.add(tf.keras.layers.Embedding( vocab_size_keras+1,embedding_dim, input_length = max_len, weights = [embedding_matrix], trainable = False))
model_glove.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences = True)))
model_glove.add(tf.keras.layers.Dropout(0.5))
model_glove.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)))
model_glove.add(tf.keras.layers.Dropout(0.5))
model_glove.add(tf.keras.layers.Dense(64, activation = 'relu'))
model_glove.add( tf.keras.layers.Dense(5 , activation = 'softmax'))

model_glove.compile(loss = 'categorical_crossentropy', optimizer = 'Adam', metrics = ['accuracy'])
model_glove.summary()


model_glove.fit(padded_training,labels_training,epochs= 10, verbose =1 , validation_data = (padded_val, labels_val))
test_data, test_label = prepare_data(df_test,tokenizer_keras, max_len = 512)
test_prediction = model_glove.predict(test_data)

output_flat = np.argmax(test_label, axis=1).flatten()
prediction_flat = np.argmax(test_prediction, axis=1).flatten()
test_accuracy = np.sum(prediction_flat == output_flat) / len(output_flat)
print("The test set includes {} texts and the accuracy is {}".format(len(output_flat), test_accuracy))
# This is for tensorflow
from numba import cuda 
device = cuda.get_current_device()
device.reset()
# This is for pytorch
torch.cuda.empty_cache()
print(tokenizer.encode(df.cleaned[7], max_length = 128 ) + [0] * 128)    # This is example of input_ids token and max length is 256
class process_dataset:
    '''
    This class is created to to prepare the dataset for data loader.
    Bert needs 3 token vecors, so processing_data function or processing_data_encode_plus function will prepare these tokens. Both returns same vectors. Both can be used.
    This getitem method returns data dictionary that includes tokens and labels.
    Labels are encoded to 5 dimensions vectors.
    
    '''
    def __init__(self, df, token_ids_with_label,  max_len, tokenizer, text_column , label_column):
        self.df = df
        self.text = self.df[text_column]
        self.label = self.df[label_column]
       # self.encoded_label = self.df['label']
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.token_ids_with_label = token_ids_with_label
    def __len__(self):
        return len(self.df)
    def __getitem__(self,index):
        
        row = self.text[index]
        row_label = self.label[index]
        convert_token_ids = self.token_ids_with_label
        input_ids, attention_masks, token_type_ids, label = processing_data(row, row_label, self.max_len, convert_token_ids)

        y =torch.LongTensor(label)

        emb = nn.Embedding(5, len(y))
        emb.weight.data = torch.eye(5)
        label_one_hot = emb(torch.LongTensor(y))
        
        data = {
            'input_ids' : torch.tensor(input_ids),
            'attention_masks' : torch.tensor(attention_masks),
            'token_type_ids' : torch.tensor(token_type_ids),
            'labels' : label_one_hot
        }
        
        return data
    
def processing_data( row, row_label, max_len, convert_token_ids = False):
    if convert_token_ids == False:

        label = [class_name_tokenizers[row_label][1]]
    
        temp_input_ids = tokenizer.encode(row, max_length = max_len)

        pad_len = max_len - len(temp_input_ids)  

        input_ids =temp_input_ids + [0] * pad_len

        attention_masks= [1] * len(temp_input_ids)+ [0] * pad_len

        token_type_ids =  [0] * max_len
        
        return np.array(input_ids), np.array(attention_masks), np.array(token_type_ids), np.array(label)

    else:     
        
        
        label_token = tokenizer.encode(row_label)
        
        label = [class_name_tokenizers[row_label][1]]

        max_len = max_len - (len(label_token)-1)

        temp_input_ids = tokenizer.encode(row, max_length = max_len)

        pad_len = max_len - len(temp_input_ids) 

        input_ids = label_token  + temp_input_ids[1:] + [0] * pad_len                                    

        attention_masks= [1] *len(label_token) + [1] * len(temp_input_ids[1:]) + [0] * pad_len

        token_type_ids = [0] *len(label_token) + [1] * len(temp_input_ids[1:])+  [0] * pad_len

        return np.array(input_ids), np.array(attention_masks), np.array(token_type_ids), np.array([label])

# Second way to create input_ids, attention_masks, and token_type_ids

def processing_data_encode_plus(df_text, df_label, max_len , convert_token_ids = False):

    input_ids= []
    attention_masks = []
    token_type_ids = []
    if convert_token_ids == False:

        encoded = tokenizer.encode_plus(df_text,                                   
                                   add_special_tokens = True,
                                   max_length = max_len,
                                   pad_to_max_length = True,
                                   return_token_type_ids = True,
                                   return_attention_mask = True
                                   )
        input_ids = encoded['input_ids']
        attention_masks = encoded['attention_mask']
        token_type_ids = encoded['token_type_ids']
        
        label = tokenizer.encode(df_label)[1]

        return input_ids,attention_masks, token_type_ids, label
    
    else:

        encoded = tokenizer.encode_plus(df_label,
                                    df_text,                                   
                                   add_special_tokens = True,
                                   max_length = max_len,
                                   pad_to_max_length = True,
                                   return_token_type_ids = True,
                                   return_attention_mask = True
                                   )
        input_ids = encoded['input_ids']
        attention_masks = encoded['attention_mask']
        token_type_ids = encoded['token_type_ids']
        
        label = tokenizer.encode(df_label)[1]
        
        return input_ids,attention_masks, token_type_ids, label
#df["input_ids"],df["attention_masks"], df['token_type_ids'],  df['label']= map(list, zip(*df[['text', 'category']].apply(lambda x: processing_data(x.text, x.category,convert_token_ids = False, max_len = 512), axis = 1)))

#df["input_ids"],df["attention_masks"], df['token_type_ids'],  df['label']= map(list, zip(*df[['text', 'category']].apply(lambda x: processing_data_encode_plus(x.text, x.category,convert_token_ids = True, max_len = 512), axis = 1)))
def get_data_loader(df,train_index, val_index,tokenizer, batch_size = 16, max_len = 128, num_workers = 0,text_column = 'text', label_column = 'category'):
    df_train = df.iloc[train_index].reset_index(drop= True)
    df_val = df.iloc[val_index].reset_index(drop= True)
    train_loader = torch.utils.data.DataLoader(process_dataset(df_train, tokenizer = tokenizer, token_ids_with_label = False, max_len = max_len, text_column = text_column, label_column = label_column),
                                        batch_size = batch_size,
                                        shuffle = False,
                                        drop_last=True,
                                          pin_memory=False,
                                        num_workers = num_workers)
                                              
    
    test_loader = torch.utils.data.DataLoader(process_dataset(df_val, tokenizer = tokenizer, token_ids_with_label = False,max_len = max_len, text_column = text_column, label_column = label_column),
                                             batch_size= batch_size,
                                             shuffle = False,
                                            drop_last=True,
                                              pin_memory=False,
                                             num_workers = num_workers)
    return {'train' : train_loader, "val": test_loader}

def get_test_loader(df,tokenizer, batch_size = 16, max_len = 128, num_workers = 0,text_column = 'text', label_column = 'category'):
    
    test_loader = torch.utils.data.DataLoader(process_dataset(df, tokenizer = tokenizer, token_ids_with_label = False, max_len = max_len, text_column = text_column, label_column = label_column),
                                             batch_size = batch_size,
                                             shuffle = False,
                                             num_workers = num_workers)
    return test_loader
class NewsModel(nn.Module):
    def __init__(self):
        super(NewsModel, self).__init__()
        conf = transformers.BertConfig()
        conf.output_hidden_states = True
        self.model = transformers.BertModel.from_pretrained('bert-base-uncased', config = conf)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(conf.hidden_size, 5)
        nn.init.xavier_normal_(self.classifier.weight)
    def forward(self, input_ids, attention_masks , token_type_ids):
        out = self.model(input_ids,attention_masks, token_type_ids)
        last_hidden_state_cls = out[0][:,0,:]                                 # This is the output of bert layer like pictures

        output = self.dropout(last_hidden_state_cls)
            
        logits = self.classifier(output)
                
        return logits
            
def loss_fn(output, predicted ):
    ce_loss = nn.BCEWithLogitsLoss()
    loss = ce_loss(output, predicted)
    return loss
def flat_accuracy(output, prediction):
    prediction_flat = np.argmax(prediction, axis=1).flatten()
    output_flat = np.argmax(output, axis=1).flatten()
    return np.sum(prediction_flat == output_flat) / len(output_flat)
def train_model(model, data_loaders, criterion, optimizer, num_epochs, device, flat_accuracy):

    for epoch in range(num_epochs):

        for phase in ['train', 'val']:

            if phase == 'train':
                model.train()
            else:
                model.eval()
            losses = []        
            epoch_loss = 0.0
            accuracy = 0.0
            counter = 0.0
            best_accuracy = 0
            dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

            for data in data_loaders[phase]:

                input_ids = data['input_ids'].to(device)
                attention_masks = data['attention_masks'].to(device)
                token_type_ids = data['token_type_ids'].to(device)
                output = data['labels'].to(device, dtype = torch.long)
                

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    



                    prediction = model( input_ids, attention_masks,  token_type_ids)
                    output = torch.max(output.float(), 1)[0]

                    loss = criterion(output,prediction)
                    if phase == 'train':

                        loss.backward()
                        optimizer.step()
                    losses.append(loss.item())

                    prediction = prediction.detach().cpu().numpy()
                    output = output.detach().cpu().numpy()

                    accuracy += flat_accuracy(output, prediction)
                    counter +=1
                    epoch_loss += loss.item() * len(input_ids)
            epoch_loss = epoch_loss /len(data_loaders[phase].dataset)
            epoch_accuracy =accuracy / counter
            if (phase == 'val' and epoch_accuracy > best_accuracy):
                best_model_weights = model.state_dict()
                best_accuracy = epoch_accuracy
            print("Epoch {} - {}- Loss {:.3f} - Accuracy: {:.3f} , number of examples :{}".format(epoch,phase,epoch_loss,epoch_accuracy,len(data_loaders[phase].dataset)))

    return best_model_weights, losses , best_accuracy
data_loaders = get_data_loader(df, range(100),range(100),text_column = 'text', label_column = 'category',tokenizer = tokenizer)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
stop_cnt = 0
model = NewsModel()

for i in data_loaders['train']:
    
    output = i['labels']
    print(i["input_ids"])
    print(i["attention_masks"])
    print(i["token_type_ids"])
    prediction = model(i["input_ids"],i["attention_masks"],i["token_type_ids"]) 
    stop_cnt+=1
    if stop_cnt == 1:
        break
        
output = torch.max(output.float(),1)[0]
prediction = prediction.detach().cpu().numpy()
output = output.detach().cpu().numpy()

print(output)
print(prediction)
skf = StratifiedKFold(n_splits = 5, shuffle = True)
best_accuracy = 0
for fold, (train_index, val_index) in enumerate(skf.split(df, df.category),start = 1):

    print(f"fold :{fold}")
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    model = NewsModel()
    model.to(device)
    optimizer = AdamW(model.parameters(), lr = 3e-5)
    data_loaders = get_data_loader(df,train_index, val_index ,tokenizer = tokenizer, batch_size = 64, max_len = 128,text_column = 'cleaned', label_column = 'category')
    criterion = loss_fn
    

    model_weights, losses, accuracy = train_model(model, data_loaders, criterion, optimizer,flat_accuracy=flat_accuracy, num_epochs = 4, device = device)
    
    if accuracy > best_accuracy:
        final_model_weights = model_weights        # This keeps the best model's weights.
        best_accuracy = accuracy
print('The best accuracy of the validation dataset is {:.2f}'.format(best_accuracy*100))
def prediction_class(final_model_weights, text):
    batch_test_losses = []
    batch_test_accuracies = []
    best_test_accuracy = 0
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')
    model = NewsModel()

    model.load_state_dict(final_model_weights)
    class_names = pd.DataFrame(class_name_tokenizers).T
    test_data_loaders = get_test_loader(text, tokenizer, batch_size = 1, max_len = 128, num_workers = 0,text_column = 'cleaned', label_column = 'category')
    for i, data in enumerate(test_data_loaders):

        prediction = model(data['input_ids'], data['attention_masks'], data['token_type_ids'])
        output = data["labels"]
        output =torch.max(output.float(),1)[0]

        prediction = prediction.detach().cpu().numpy()
        output = output.detach().cpu().numpy()
#        print(output)
#        print(prediction)
#        print(text.loc[i,'text'])
#        print("Real output is {}, Prediction is :{}\n".format(text.loc[i,'category'], class_names[class_names[1] == np.argmax(output)].index[0]))
        batch_test_accuracy = flat_accuracy(output, prediction)
        batch_test_accuracies.append(batch_test_accuracy)
        test_accuracy = sum(batch_test_accuracies) / len(batch_test_accuracies)
    return print("Test dataset includes {} texts and the test set accuracy is {:.2f}.".format(len(batch_test_accuracies),test_accuracy))
prediction_class(final_model_weights, df_test)
torch.cuda.empty_cache()
tf.keras.backend.clear_session()
import tensorflow_hub as hub
def build_model(max_len = 128):
    ids = tf.keras.layers.Input(shape = (max_len, ), dtype = tf.int32)
    masks = tf.keras.layers.Input(shape =(max_len,), dtype = tf.int32)
    token_ids = tf.keras.layers.Input(shape =(max_len,), dtype = tf.int32)

    bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",  trainable=True)
    pooled_output, sequence_output = bert_layer([ids, masks ,token_ids])

    output = sequence_output[:,0,:]

    out = tf.keras.layers.Dropout(0.5)(output)

    out = tf.keras.layers.Dense(5, activation = 'softmax')(out)

    model = tf.keras.models.Model(inputs = [ids,masks,token_ids], outputs= out)

    optimizer = tf.optimizers.Adam(learning_rate = 3e-5)

    model.compile(loss ='categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy'])
    
    return model
def encode(df, max_len = 128):

    ids = np.zeros((len(df), max_len), dtype = 'float32')
    masks = np.zeros((len(df), max_len), dtype = 'float32')
    token_ids = np.zeros((len(df), max_len), dtype = 'float32')
    labels = np.zeros((len(df), max_len), dtype = 'float32')

    ids, masks, token_ids, labels=map(list, zip(*df[['text', 'category']].apply(lambda x: processing_data(x.text, x.category,convert_token_ids = False, max_len = max_len), axis = 1)))

    ids = np.array(ids, dtype ='float32')
    masks = np.array(masks, dtype ='float32')
    token_ids = np.array(token_ids, dtype ='float32')
    labels = tf.keras.utils.to_categorical(np.array(labels))
    return ids,masks, token_ids, labels
skf = StratifiedKFold(n_splits = 4, shuffle = True)

ids, masks, token_ids, labels = encode(df, max_len = 128)

for k , (train_index,val_index) in enumerate(skf.split(ids, labels.argmax(1)), start =1):
    ids_train = ids[train_index,:],
    masks_train = masks[train_index,:]
    token_train = token_ids[train_index,:]
    labels_train = labels[train_index,:]
    ids_val = ids[train_index,:],
    masks_val = masks[train_index,:]
    token_val = token_ids[train_index,:]
    labels_val = labels[train_index,:]
    print("fold :{}".format(k))
    model = build_model()
    history = model.fit((ids_train, masks_train, token_train), labels_train, epochs = 3, verbose = 1, batch_size = 32, validation_data = ((ids_val, masks_val, token_val), labels_val))
ids_test, masks_test, token_ids_test, labels_test = encode(df_test, max_len = 128)
prediction = model.predict((ids_test, masks_test, token_ids_test))
output = tf.keras.utils.to_categorical(df_test['label'])
test_result = flat_accuracy(output, prediction)
print("The test set includes {} texts and the accuracy is {}".format(len(output), test_result))