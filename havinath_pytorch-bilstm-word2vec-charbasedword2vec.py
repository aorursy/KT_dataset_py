# Code to download file into Colaboratory:
!pip install -U -q PyDrive

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials
# Authenticate and create the PyDrive client.
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)

id = '1vF3FqgBC1Y-RPefeVmY8zetdZG1jmHzT'
downloaded = drive.CreateFile({'id':id}) 
downloaded.GetContentFile('imdb_train.csv')

id = '1XhaV8YMuQeSwozQww8PeyiWMJfia13G6'
downloaded = drive.CreateFile({'id':id}) 
downloaded.GetContentFile('imdb_test.csv')

import pandas as pd
df_train = pd.read_csv("imdb_train.csv")
df_test = pd.read_csv("imdb_test.csv")

reviews_train = df_train['review'].tolist()
sentiments_train = df_train['sentiment'].tolist()
reviews_test = df_test['review'].tolist()
sentiments_test = df_test['sentiment'].tolist()

print("Training set number:",len(reviews_train))
print("Testing set number:",len(reviews_test))
from sklearn.preprocessing import LabelEncoder
import numpy as np

labels = np.unique(sentiments_train)


lEnc = LabelEncoder()
lEnc.fit(labels)
label_train_n = lEnc.transform(sentiments_train)
label_test_n = lEnc.transform(sentiments_test)
numClass = len(labels)

print(labels)
print(lEnc.transform(labels))
reviews_train = [s.lower() for s in reviews_train]
reviews_test = [s.lower() for s in reviews_test]

import re
#There are some sentences like "<br /><br />_____________________________________<br /><br />" which will be replaced by .
reviews_train = [re.sub('<br /><br />_+<br /><br />','. ', s) for s in reviews_train]
reviews_test = [re.sub('<br /><br />_+<br /><br />','. ', s)  for s in reviews_test]

#There are a lot of <br /><br /> which indicate end of line
reviews_train = [s.replace("<br /><br />",". ") for s in reviews_train]
reviews_test = [s.replace("<br /><br />",". ") for s in reviews_test]


import re

#Removing Hyperlinks starting with http, https and satisfying other requirements
reviews_train = [re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+','', s) for s in reviews_train]
reviews_test = [re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+','', s) for s in reviews_test]

#Removing Hyperlinks starting with www and satisfying other requirements
reviews_train = [re.sub('www.(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+','', s) for s in reviews_train]
reviews_test = [re.sub('www.(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+','', s) for s in reviews_test]


#Removing one specific url - ScheduleServlet?ACTION_DETAIL=DETAIL&FOCUS_ID=598947
reviews_train = [s.replace("scheduleservlet?action_detail=detail&focus_id=598947","") for s in reviews_train]
reviews_test = [s.replace("scheduleservlet?action_detail=detail&focus_id=598947","") for s in reviews_test]


longestwords_dict = {"It'saboutaguylookingforthewomanwhosavedhisgrandpafromtheNazis".lower(): "It's about a guy looking for the woman who saved his grandpa from the Nazis".lower(),
                     "LoveHateDreamsLifeWorkPlayFriends".lower():"Love Hate Dreams Life Work Play Friends".lower(),
                     "anothermaniacherewegoagain":"another maniac here we go again",
                     "heyijustleftmycoatbehind":"hey i just leftmy coat behind ",
                     "brianjonestownmassacre":"brian jonestown massacre",
                     "cough2Fast2Furiouscough".lower():"cough 2 Fast 2 Furious cough".lower(),
                     "realityshowfictionnal":"reality show fictional",
                     "hongkongmovieshootouts":"hongkong movie shootouts",
                     "worlddestructionthemes":"world destruction themes",
                     "redoredoredocopycopycopy":"redo redo redo copy copy copy",
                     "whateverherlastnameis":"what ever her last name is",
                     "specialagentfoxmulder":"special agent fox mulder",
                     "ammmmmbbbererrrrrrrrrgerrrrrrrssss":"amber gas"}
                     


for i, j in longestwords_dict.items():
  reviews_train = [s.replace(i, j) for s in reviews_train]
  reviews_test = [s.replace(i, j) for s in reviews_test]



import re

def remove_punctuation_re(x):

    tempwords = x.split()
    bool_longword = False #flag to join the modified words
    #Checking to see if multiple words are seperated by symbols other than space and handling accordingly
    for i in range(len(tempwords)):
      if len(tempwords[i]) > 20:      
        tempwords[i] = re.sub(r'[^\w\s]',' ',tempwords[i])
        if tempwords[i].count("_") > 1:  #Identifying multiple words which are seperated by _ and not by space. Example: - elizabeth_perkins_in_miracle_on
          tempwords[i] = tempwords[i].replace("_"," ")  
        bool_longword = True # Updating flag to join the modified words 
    if bool_longword == True:
      x = " ".join(tempwords)


    x = re.sub(r'[^\w\s]','',x)
    #y = "".join(filter(lambda char: char in string.printable, x))
    x = x.replace("½","")
    x = x.replace("¾","")
    x = x.replace("º","") #appended to timeº
    x = x.replace("ª","")
    x = x.replace(""," ")
    x = x.replace("³"," 3") #2 instances of alien³
    x = x.replace("1ç","1") # one mention of "1ç is to expensive" and hence dealing with it so that it does not give a different intent for other words like nouns
    x=x.replace("יגאל","")   
    return x

reviews_train = [remove_punctuation_re(s) for s in reviews_train]
reviews_test = [remove_punctuation_re(s) for s in reviews_test]
reviews_train = [re.sub('\d+','', s) for s in reviews_train]
reviews_test = [re.sub('\d+','', s) for s in reviews_test]
import nltk
nltk.download('punkt')

from nltk.tokenize import TweetTokenizer
tknzr = TweetTokenizer()

reviews_train = [tknzr.tokenize(s) for s in reviews_train]
reviews_test  = [tknzr.tokenize(s) for s in reviews_test]
text_train_nsu=[]
for tokens in reviews_train:
  filtered_sentence = [w.strip("_") for w in tokens]
  text_train_nsu.append(filtered_sentence)



text_test_nsu=[]
for tokens in reviews_test:
    filtered_sentence =  [w.strip("_") for w in tokens] 
    text_test_nsu.append(filtered_sentence)

nltk.download('stopwords')
from nltk.corpus import stopwords as sw
stop_words = sw.words('english')  #Only English Stopwords

# Some stop words might indicate a certain sentiment and hence whitelisting them
whitelist = ["n't", "not", "no","nt"]

text_train_ns=[]
for tokens in text_train_nsu:
  filtered_sentence = [w for w in tokens if (not w in stop_words) or (w in whitelist)]
  text_train_ns.append(filtered_sentence)
text_test_ns=[]
for tokens in text_test_nsu:
    filtered_sentence =  [w for w in tokens if (not w in stop_words) or (w in whitelist)] 
    text_test_ns.append(filtered_sentence)

nltk.download('wordnet')

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

text_train_le1 = []
for tokens in text_train_ns:
    lemma_sentence = [lemmatizer.lemmatize(w) for w in tokens ]
    text_train_le1.append(lemma_sentence)

text_test_le1 = []
for tokens in text_test_ns:
    lemma_sentence = [lemmatizer.lemmatize(w) for w in tokens ]
    text_test_le1.append(lemma_sentence)


text_train_le=[]
for tokens in text_train_le1:
  filtered_sentence = [w for w in tokens if (not w.replace(" ","").isdecimal()) and ( w.count("_") != len(w))]  #We are removing tokens with only numbers and _
  text_train_le.append(filtered_sentence)



text_test_le=[]
for tokens in text_test_le1:
    filtered_sentence = [w for w in tokens if (not w.replace(" ","").isdecimal()) and ( w.count("_") != len(w))]
    text_test_le.append(filtered_sentence)
import re
import datetime

#print("Start Time:"+datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S"))


for x in [text_train_le, text_test_le]:
  for sentence in x:
    #for word in sentence:
    for i in range(len(sentence)):
      sentence[i] = re.sub(r'^boo+$','boo',sentence[i]) #handle  booooooooooooooooooooooooooooooooooooooooooooooo
      sentence[i] = re.sub(r'^[a]h+a+(h+a+)+h*a*$','haha',sentence[i]) #handle  ahahahahahhahahahahahahahahahhahahahahahahah
      sentence[i] = re.sub(r'^zzz+$','zzz',sentence[i]) #handle  zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz
      sentence[i] = re.sub(r'^da+d+$','dad',sentence[i]) #handle  daaaaaaaaaaaaaaaaaddddddddd
      sentence[i] = re.sub(r'^s+o+$','so',sentence[i]) #handle  sssssssssssooooooooooooo
      sentence[i] = re.sub(r'^ca+ligula+$','caligula',sentence[i]) #handle  CAAAAAAAAAAAAAAAAAAAAAALIGULAAAAAAAAAAAAAAAAAAAAAAA, caligulaaaaaaaaaaaaaaaaa
      sentence[i] = re.sub(r'^no+[so]o*$','no',sentence[i]) #handle nooooooooooooooooooooo, nooooooooooooooooooooso
      sentence[i] = re.sub(r'^co+ff+i+n+$','coffin',sentence[i]) #handle coooofffffffiiiiinnnnn
      sentence[i] = re.sub(r'^ye+s+h*$','yes',sentence[i]) #handle yeeshhhhhhhhhhhhhhhhh
      sentence[i] = re.sub(r'^(blah)+$','blah',sentence[i]) #handle blahblahblahblahblahblahblahblahblahblahblahblahblahblahblahblahblahblah
      sentence[i] = re.sub(r'^s+t+u+p+i+d+$','stupid',sentence[i]) #handle sssssssttttttttttuuuuuuuuuuuuuuuuuupppppiiiddd
      sentence[i] = re.sub(r'^ter+i+ble$','terrible',sentence[i]) #handle terrrrrrrrrrrrrrrriiiiiiiiiiiible
      sentence[i] = re.sub(r'^all+$','all',sentence[i]) #handle       allllllllllllllllllllllllllllll 
      sentence[i] = re.sub(r'^a+hh+$','ahh',sentence[i]) #handle       ahhhhhhhhhhhhhhhhhhhhhhhhhhh, aaaaaaaaaaaahhhhhhhhhhhhhh 
      sentence[i] = re.sub(r'^i+v+y+$','ivy',sentence[i]) #handle       iiiiiiiivvvvyyyyyyyyyyyyyyyy
      sentence[i] = re.sub(r'^uh+k+u+h+k+$','uhkuhk',sentence[i]) #handle       Uhhhhkkkkkkuuuuuhhhhhkkkkkk 
      sentence[i] = re.sub(r'^bo+r+in+g+$','boring',sentence[i]) #handle       boooooooorrrrrinngggggggg
      sentence[i] = re.sub(r'^goo+d$','good',sentence[i]) #handle       goooooooooooooooooooood
      sentence[i] = re.sub(r'^ho+t+$','hot',sentence[i]) #handle       hottttttttttttttttttttt
      sentence[i] = re.sub(r'^wo+nderful$','wonderful',sentence[i]) #handle       wooooooooooooooonderful
      sentence[i] = re.sub(r'^a+r+g+h+$','argh',sentence[i]) #handle       aaaaarrrrrrgggggghhhhhh   
      sentence[i] = re.sub(r'^je+sus+$','jesus',sentence[i]) #handle       jeeeeeeeesussssssssss
      sentence[i] = re.sub(r'^(spoilers)+$','spoliers',sentence[i]) #handle       spoilersspoilersspoilersspoilers 
      sentence[i] = re.sub(r'^(fantastic)+$','fantastic',sentence[i]) #handle       FANTASTICFANTASTICFANTASTIC 
      sentence[i] = re.sub(r'^sup[eu]+r+b+$','superb',sentence[i]) #handle       supurrrrb
      sentence[i] = re.sub(r'^aa+[hg]+$','aah',sentence[i]) #handle       aaaaah 
      sentence[i] = re.sub(r'^a+nd$','and',sentence[i]) #handle       aaaand 
      sentence[i] = re.sub(r'^all+r+i+ght$','alright',sentence[i]) #handle       alllriiiiight 
      sentence[i] = re.sub(r'^bh*a+d+$','bad',sentence[i]) #handle       baaaaaad 
      sentence[i] = re.sub(r'^coo+l$','cool',sentence[i]) #handle       coool
      sentence[i] = re.sub(r'^doo+m$','doom',sentence[i]) #handle       dooooom  
      sentence[i] = re.sub(r'^du+h+$','duh',sentence[i]) #handle       duhhh  
      sentence[i] = re.sub(r'^e+vil$','evil',sentence[i]) #handle       eeeevil 
      sentence[i] = re.sub(r'^funn+y+$','funny',sentence[i]) #handle       funnny
      sentence[i] = re.sub(r'^grrr+l*$','grr',sentence[i]) #handle       grrrrrr
      sentence[i] = re.sub(r'^he+a+r+t$','heart',sentence[i]) #handle       heeaaaaaaaaaart 
      sentence[i] = re.sub(r'^he+re$','here',sentence[i]) #handle       heeeeeere 
      sentence[i] = re.sub(r'^hell+o+$','hello',sentence[i]) #handle       helloooo          
      sentence[i] = re.sub(r'^[hu]mm+$','hmm',sentence[i]) #handle       hmmmmm          
      sentence[i] = re.sub(r'^hu+ge$','huge',sentence[i]) #handle       huuuuuge        
      sentence[i] = re.sub(r'^jee+z$','jeez',sentence[i]) #handle       jeeez     
      sentence[i] = re.sub(r'^li+fe$','life',sentence[i]) #handle       liiiiiiiiife  
      sentence[i] = re.sub(r'^lo+n+g+$','long',sentence[i]) #handle       looong  
      sentence[i] = re.sub(r'^lo+v+e+$','love',sentence[i]) #handle       loooooove  
      sentence[i] = re.sub(r'^mm+$','mm',sentence[i]) #handle       mmmmm  
      sentence[i] = re.sub(r'^mo+re$','more',sentence[i]) #handle       mooooooooooooore     
      sentence[i] = re.sub(r'^na+h*$','no',sentence[i]) #handle       naaaa  
      sentence[i] = re.sub(r'^o[oh]+h+$','ohh',sentence[i]) #handle       ohhhh  
      sentence[i] = re.sub(r'^oo+p+s$','oops',sentence[i]) #handle       oooops  
      sentence[i] = re.sub(r'^pla+in$','plain',sentence[i]) #handle       plaaaain  
      sentence[i] = re.sub(r'^ple+a[sz]e+$','please',sentence[i]) #handle       pleaseee  
      sentence[i] = re.sub(r'^re+a+ll+y+$','really',sentence[i]) #handle       reaaaaallly             
      sentence[i] = re.sub(r'^ri+ght$','right',sentence[i]) #handle       right  
      sentence[i] = re.sub(r'^slo+w$','slow',sentence[i]) #handle       sloooow    
      sentence[i] = re.sub(r'^shh+$','shh',sentence[i]) #handle       shhhhh        
      sentence[i] = re.sub(r'^shi+t+y*$','shit',sentence[i]) #handle       shittttttttttttttty 
      sentence[i] = re.sub(r'^st[uo]+pid+$','stupid',sentence[i]) #handle       stooooopid  
      sentence[i] = re.sub(r'^thing+$','thing',sentence[i]) #handle       thinggg    
      sentence[i] = re.sub(r'^too+$','too',sentence[i]) #handle       toooo        
      sentence[i] = re.sub(r'^u[gh]+h+$','ugh',sentence[i]) #handle       uggghh                    
      sentence[i] = re.sub(r'^ve+r+y+$','very',sentence[i]) #handle       verrrrry       
      sentence[i] = re.sub(r'^wa+y+$','way',sentence[i]) #handle       waaaaaaaaay                    
      sentence[i] = re.sub(r'^we+ll+$','well',sentence[i]) #handle       welll    
      sentence[i] = re.sub(r'^wha+[at]+$','what',sentence[i]) #handle       whaaaaatttt                    
      sentence[i] = re.sub(r'^why+$','why',sentence[i]) #handle       whyyyyyyyyyyyyyyyy
      sentence[i] = re.sub(r'^wo+[ho]+$','woho',sentence[i]) #handle       wooohooo    
      sentence[i] = re.sub(r'^ye+a+h+$','yeah',sentence[i]) #handle       yeahhhh                    
      sentence[i] = re.sub(r'^ya+y+$','yay',sentence[i]) #handle       yaaayy    
      sentence[i] = re.sub(r'^yippee+$','yippee',sentence[i]) #handle       yippeee
      sentence[i] = re.sub(r'^yum+$','yum',sentence[i]) #handle       yummm
 

#print("End Time:"+datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S"))

import copy
import datetime

#Adding all words into a list called word_sequence
word_sequence = []
for corpus_temp in [text_train_le,text_test_le]:
  for sentence in corpus_temp:
    word_sequence.extend(sentence)


print("Total unique words:",len(set(word_sequence)))

#Using counter to count the number of occurences of each word
from collections import Counter
vocab_cnt = Counter()
vocab_cnt.update(word_sequence)
vocab_cnt = Counter({w:c for w,c in vocab_cnt.items() if c > 5})  #Ignoring words which have occured less than 6 times

print("Total unique words after ignoring words occuring less than 6 times:",len(vocab_cnt))

word_list = []  #List to store unique words which have occured more than 6 times


for k,v in vocab_cnt.items():
  word_list.append(k)


#print(len(word_list))
word_list = list(set(word_list))
#print(len(word_list))
word_list.sort()

# make dictionary so that we can reference each index of unique word
word_dict = {w: i for i, w in enumerate(word_list)}

voc_size = len(word_list)

skip_grams = []

#Creating skipgrams
for corpus in [text_train_le,text_test_le]:
  for sentence in corpus:
    for i in range(1, len(sentence) - 1):
    #for i in range(2, len(sentence) - 2):  #Uncomment for window size - 2 
    #for i in range(5, len(sentence) - 5):   #Uncomment for window size - 5 
      try:
        target = word_dict[sentence[i]]
        #context = [word_dict[sentence[i - 5]],word_dict[sentence[i - 4]],word_dict[sentence[i - 3]],word_dict[sentence[i - 2]],word_dict[sentence[i - 1]], word_dict[sentence[i + 1]],word_dict[sentence[i + 2]],word_dict[sentence[i + 3]],word_dict[sentence[i + 4]],word_dict[sentence[i + 5]]] #Uncomment for window size - 5
        #context = [word_dict[sentence[i - 2]],word_dict[sentence[i - 1]], word_dict[sentence[i + 1]],word_dict[sentence[i + 2]]]  #Uncomment for window size - 2 
        context = [word_dict[sentence[i - 1]], word_dict[sentence[i + 1]]]
        for w in context:
          #skip_grams.append([w,target])  #Uncomment for CBOW
          skip_grams.append([target, w])  #Skip Gram
      except:
        continue

print("Total number of skipgrams :",len(skip_grams))



   
def prepare_batch(data, size):
    random_inputs = []
    random_labels = []
    random_index = np.random.choice(range(len(data)), size, replace=False)
    for i in random_index:
        input_temp = [0]*voc_size   #length of the one hot vector will be the total number of unique words.
        input_temp[data[i][0]] = 1
        random_inputs.append(input_temp)  # target
        random_labels.append(data[i][1])  # context word
    return np.array(random_inputs), np.array(random_labels)
#hyperparameter declaration

learning_rate = 0.01
batch_size = 4000 
embedding_size = 200
total_epoch = 750 
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  #Using GPU if available
class SkipGram(nn.Module):
    def __init__(self):
        super(SkipGram, self).__init__()
        self.linear1 = nn.Linear(voc_size, embedding_size,bias=False)
        self.linear2 = nn.Linear(embedding_size, voc_size,bias=False)

    def forward(self, x):
        hidden = self.linear1(x)
        out = self.linear2(hidden)
        return out   

skip_gram_model = SkipGram().to(device)
criterion = nn.CrossEntropyLoss() #please note we are using "CrossEntropyLoss" here
optimizer = optim.Adam(skip_gram_model.parameters(), lr=learning_rate)

for epoch in range(total_epoch):
    #print("Epoch "+str(epoch)+" Start Time:"+datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")) #uncomment during debug
    inputs,labels = prepare_batch(skip_grams, batch_size)
    inputs_torch = torch.from_numpy(np.array(inputs)).float().to(device)
    outputs_torch = torch.from_numpy(np.array(labels)).to(device)
    skip_gram_model.train()
    # 1. zero grad
    optimizer.zero_grad()
    # 2. forword propagation
    predoutput_torch = skip_gram_model(inputs_torch)
    # 3. calculate loss
    loss = criterion(predoutput_torch, outputs_torch)
    # 4. back propagation
    loss.backward()
    optimizer.step()
    #print("Epoch "+str(epoch)+" End Time:"+datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")) #uncomment during debug
    if epoch % 10 == 9:
      print('Epoch: %d, loss: %.4f' %(epoch + 1, loss))


#Extracting weights i.e. vectors for words
weight1 = skip_gram_model.linear1.weight
trainedword2vec_embeddings = weight1.detach().T.cpu().numpy()

from google.colab import drive
drive.mount('/content/gdrive')
Word2VecModelName = "wordEmbeddingAssignment1-"+datetime.datetime.now().strftime("%d-%m-%Y-%H-%M")+".pt" #Adding current time to filename so that old models are not lost
path = F"/content/gdrive/My Drive/"+Word2VecModelName 
torch.save(skip_gram_model, path)
drive.flush_and_unmount()
from google.colab import drive
drive.mount('/content/gdrive')
path = F"/content/gdrive/My Drive/"+Word2VecModelName   #Load above saved model
#path = F"/content/gdrive/My Drive/wordEmbeddingAssignment1-B4000_E750_LR0.01.pt" #Uncomment this for loading a different model then above saved one
try:
  modelword2vec = torch.load(path)
except:
  modelword2vec = torch.load(path, map_location=torch.device('cpu')) #Lod into CPU if GPU is not available

drive.flush_and_unmount()
modelword2vec.eval()
#Extract weights
weight1 = modelword2vec.linear1.weight
trainedword2vec_embeddings = weight1.detach().T.cpu().numpy()
chars = []
wordlen = []

for word in word_list:
  chars+=list(word)  #adding all characters to chars list for words used in word2vec embedding
  wordlen.append(len(word))  #adding the length of each word to wordlen list

char_arr = list(set(chars))  #adding unique characters in char_arr

char_dic = {n: i for i, n in enumerate(char_arr)}  #Creating a dictionary for indexing characters when required

char_dic_len = len(char_dic) #This will be the length of one hot vector for every character

maxword_len = max(wordlen)  
print("Maximum word length : ",maxword_len)
def make_char_batch(wordlist,size):
    input_batch = []
    output_batch = []

    random_index = np.random.choice(range(len(wordlist)), size, replace=False)

    for i in random_index:
        input_data = [char_dic[n] for n in wordlist[i]] #Getting the index of every character from char dictionary for very letter in the word.

        #Adding 0 to the vector in case the word is smaller than the longest word
        diff = maxword_len - len(input_data)
        for x in range(diff):
          input_data.append(0)

        input_batch.append(np.eye(char_dic_len)[input_data])

        output_data  = trainedword2vec_embeddings[i]         # Output data - Vector of the word generated from word2vec model
        output_batch.append(output_data)
    return input_batch, output_batch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score

class CharNet(nn.Module):
    def __init__(self):
        super(CharNet, self).__init__()
        self.lstm = nn.LSTM(n_input, n_hidden, batch_first =True,bidirectional=True, dropout=0.2)
        self.linear = nn.Linear(n_hidden*2,n_class)

    def forward(self, sentence):
        
        #h_n of shape (num_layers * num_directions, batch, hidden_size): tensor containing the hidden state for t = seq_len.
        lstm_out, (h_n,c_n) = self.lstm(sentence)
        #concat the last hidden state from two direction
        hidden_out =torch.cat((h_n[0,:,:],h_n[1,:,:]),1)
        z = self.linear(hidden_out)
        log_output = F.log_softmax(z, dim=1)
        return log_output,hidden_out
learning_rate = 0.05
n_hidden = 50
total_epoch = 2000
char_batch_size =4000

n_input = char_dic_len
n_class = embedding_size

# Move the model to GPU
charnet = CharNet().to(device)
# Loss function and optimizer
#criterion = nn.NLLLoss()
criterion = nn.L1Loss() #Mean Absolute Error loss function
optimizer = optim.Adam(charnet.parameters(), lr=learning_rate)

#Uncomment while debugging
#print("learning_rate",learning_rate)
#print("n_hidden",n_hidden)
#print("total_epoch",total_epoch)
#print("char_batch_size",char_batch_size)


for epoch in range(total_epoch):
    #if epoch < 10: 
    #  print("Epoch "+str(epoch)+" Start Time:"+datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S"))  #Uncomment while debugging
    # Preparing input
    input_batch, target_batch = make_char_batch(word_list,char_batch_size)
    # Convert input into tensors and move them to GPU by uting tensor.to(device)
    input_batch_torch = torch.from_numpy(np.array(input_batch)).float().to(device)
    target_batch_torch = torch.from_numpy(np.array(target_batch)).to(device)  
    
    # Set the flag to training
    charnet.train()
    
    # forward + backward + optimize
    outputs,_ = charnet(input_batch_torch)
    #print(outputs[0].size())
    #print(target_batch_torch[0].size()) 
    loss = criterion(outputs, target_batch_torch)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # Set the flag to evaluation, which will 'turn off' the dropout
    charnet.eval()
    outputs,_ = charnet(input_batch_torch) 
    
    # Evaluation loss
    loss = criterion(outputs, target_batch_torch)
    #if epoch < 10:
    #  print("Epoch "+str(epoch)+" End Time:"+datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S"))  #Uncomment while debugging
    if epoch % 50 == 49:
      print('Epoch: %d, loss: %.5f' %(epoch + 1, loss.item()))
print('Finished Training')



charnet.eval()
_,hidden_state = charnet(input_batch_torch)

import datetime

from google.colab import drive
drive.mount('/content/gdrive')
CharModelName = "CharEmbeddingAssignment1-"+datetime.datetime.now().strftime("%d-%m-%Y-%H-%M")+"_LearningRate"+str(learning_rate)+"_n_hidden"+str(n_hidden)+"_Epoch"+str(total_epoch)+"_batch_size"+str(char_batch_size)+".pt"

charpath = F"/content/gdrive/My Drive/"+CharModelName 
torch.save(charnet, charpath)
drive.flush_and_unmount()
from google.colab import drive
drive.mount('/content/gdrive')
#CharModelName= "CharEmbeddingAssignment1-21-04-2020-08-30_LearningRate0.05_n_hidden50_Epoch2000_batch_size4000.pt"
charmodel_path = F"/content/gdrive/My Drive/"+CharModelName  #Load above saved model

CharEmbedModel= torch.load(charmodel_path)

drive.flush_and_unmount()
CharEmbedModel.eval()
import matplotlib.pyplot as plt
len_list = [len(s) for s in text_train_le]  #Storing length of reviews in training set.

#Creating a box plot for length of reviews
fig = plt.figure(1)
fig.suptitle('Box Plot for length of reviews', fontsize=14, fontweight='bold')
ax = fig.add_subplot(111)
ax.boxplot(len_list,vert=False,meanline=True,showmeans=True)
ax.set_xlabel('Number of words')

#Creating a histogram for length of reviews
fig = plt.figure(2)
fig.suptitle('Histogram for length of reviews', fontsize=14, fontweight='bold')
bx = fig.add_subplot(111)
bx.hist(len_list)
bx.set_xlabel('Number of words')
bx.set_ylabel('Number of reviews')


plt.show()
print("\nMaximum no. of words in reviews :",max(len_list))

#plt.hist(len_list)

# Below list are to plot histogram
xval=[]
yval=[]

for review_len in range(200,610,10):  #Creating a list of values from 200 to 600
  counter_r = 0
  for s in text_train_le:
    if len(s) < review_len:
      counter_r+=1  #counting number of reviews less then review_len
  xval.append(review_len)
  yval.append(counter_r)
  if review_len % 50 == 0: #Printing periodic details
    print("Number of reviews less than",review_len,"words :",counter_r)

#Plotting a histogram for visualization
plt.figure(figsize=(16,10))  #Increasing the figure size for easy visualization
plt.bar(xval,yval,align='center') #Histogram
plt.ylim(20500,25500)  #Limiting y axis so that we can only focus on number of reviews more than 21000
plt.xlabel('No. of words')
plt.ylabel('No. of reviews less than those words')
for i in range(len(yval)):
    plt.hlines(yval[i],0,xval[i]) # Drawing the horizontal lines
plt.show()


seq_length = 200

def add_padding(oldcorpus, seq_length):
    corpus = copy.deepcopy(oldcorpus) #Performing a  deepcopy so that text_train_le, text_test_le does not get effected
    output = []
    for sentence in corpus:
      if len(sentence)>seq_length:
        output.append(sentence[:seq_length])  #Truncating the sentence if it is longer than seq_length
      else:
        for j in range(seq_length-len(sentence)): #Padding the sentence if it is smaller than seq_length
          sentence.append("<PAD>")
        output.append(sentence)
    return output

text_train_pad = add_padding(text_train_le,seq_length )
text_test_pad = add_padding(text_test_le,seq_length )


CharEmbedModel.eval()

trained_char_based_word_embeddings= []  #list to store character based word embeddings of all words in the dictionary

for word1 in word_list:
  char_input = []
  input_data = [char_dic[n] for n in word1]  #Getting index of each character

  #Adding empty rows in case the word is smaller than the longest word
  diff = maxword_len - len(input_data)
  for x in range(diff):
        input_data.append(0)

  char_input.append(np.eye(char_dic_len)[input_data]) #converting to one-hot encoded vector

  char_input_batch_torch = torch.from_numpy(np.array(char_input)).float().to(device)
  charword2vec,_ = CharEmbedModel(char_input_batch_torch)  #Get the char based embedding of the word
  trained_char_based_word_embeddings.append(charword2vec[0].detach().cpu().numpy())  #Adding to the list for future use
def make_embed_forseq(corpus,labels,worddict,word2vecembed,charbasedembed,size,epoch):

    emb_dim = 2 * embedding_size # Twice - one for word2vec length and another for character based word embedding length
    input_batch_seq = []
    output_batch_seq = []

    indexes = range(epoch*size,(epoch+1)*size)
    indexes = [r if r < len(corpus) else (r % len(corpus)) for r in indexes] #In case the range is outside the length of the dataset then we start from beginining of the data set

    for i in indexes:
        sentence = corpus[i]
        out_temp = []
        for word in sentence:
            bool_Wordfound = False
            wordindex = 0
            try:  
                wordindex = worddict[word]
                bool_Wordfound = True
            except:  #To handle scenarios where the word is not present in the dictionary 
                bool_Wordfound = False
            if bool_Wordfound == True:
              concat_embed = [*word2vecembed[wordindex] ,*charbasedembed[wordindex]]  #Unpack both word2vec embedding and character based embedding and add to a new list
              out_temp.append(concat_embed)
            else:
                out_temp.append([0]*emb_dim) #If the word is not found in word_dict then create a matrix with all zeros
        input_batch_seq.append(out_temp)
        output_batch_seq.append(labels[i])
    return np.array(input_batch_seq),np.array(output_batch_seq)

n_input = 2 * embedding_size 
n_hidden = 50 
n_class = len(np.unique(label_train_n)) #number of unique labels
total_epoch = 240
learning_rate = 0.02
seq_batch_size = 250 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #self.lstm = nn.LSTM(n_input, n_hidden, batch_first =True,bidirectional=True, dropout=0.2)
        self.lstm = nn.LSTM(n_input, n_hidden, num_layers=2, batch_first =True, dropout=0.2)
        self.linear = nn.Linear(n_hidden,n_class)

    def forward(self, x):
        x,_ = self.lstm(x)
        x = self.linear(x[:,-1,:])
        x = F.log_softmax(x, dim=1)
        return x
import torch
import shutil
def save_ckp(state, is_best, checkpoint_name, best_model_name):
    f_path = checkpoint_name
    torch.save(state, f_path)
    if is_best:
        best_fpath = best_model_name
        shutil.copyfile(f_path, best_fpath)

def load_ckp(checkpoint_fpath, model, optimizer):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer, checkpoint['epoch']
from sklearn.metrics import f1_score

epoch_values = [] #Lists to store epoch values for later plotting
f1_scores = []  #Lists to store f1-values for later plotting

net = Net().to(device)
criterion = nn.NLLLoss()
optimizer = optim.Adam(net.parameters(), lr=learning_rate)

start_epoch = 0

#ckp_path = "/content/SeqCheckPoint_59.pt"  #Uncomment to load a check point 
#net, optimizer, start_epoch = load_ckp(ckp_path, net, optimizer) #Uncomment to load a check point 

#Uncomment for debugging
#print("n_hidden",n_hidden)
#print("total_epoch",total_epoch)
#print("learning_rate",learning_rate)
#print("seq_batch_size",seq_batch_size)


for epoch in range(start_epoch,total_epoch):
    #if epoch < 10:  #Uncomment during debugging
    #  print("Epoch "+str(epoch)+" Start Time:"+datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S"))  
    input_seq_batch, target_seq_batch = make_embed_forseq(text_train_pad,label_train_n,word_dict,trainedword2vec_embeddings,trained_char_based_word_embeddings,seq_batch_size,epoch)
    
    input_batch_torch = torch.from_numpy(input_seq_batch).float().to(device)
    target_batch_torch = torch.from_numpy(target_seq_batch).view(-1).to(device)

    net.train()
    outputs = net(input_batch_torch) 
    loss = criterion(outputs, target_batch_torch)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    net.eval()
    outputs = net(input_batch_torch)
   # if epoch < 10: #Uncomment during debugging
   #   print("Epoch "+str(epoch)+" End Time:"+datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S"))
    
    
    if epoch%10 == 9:
        loss = criterion(outputs, target_batch_torch)
        _, predicted = torch.max(outputs, 1)
        acc= accuracy_score(predicted.cpu().numpy(),target_batch_torch.cpu().numpy())

        #Below lines of code to calculate F1 score on test data for plotting.
        epoch_values.append(epoch+1) #Store epoch value for plotting
        #Create batch from test data only for validation and plotting and not for training
        test_seq_batch, testtarget_seq_batch = make_embed_forseq(text_test_pad,label_test_n,word_dict,trainedword2vec_embeddings,trained_char_based_word_embeddings,seq_batch_size,epoch)
        testinput_batch_torch = torch.from_numpy(test_seq_batch).float().to(device)
        test_outputs = net(testinput_batch_torch)   #Perform prediction of test batch
        _, test_predictions = torch.max(test_outputs, 1) 
        tempf1_score = f1_score(testtarget_seq_batch, test_predictions.cpu().numpy(), average='weighted') #calculate f1-score of the test batch
        #print('Epoch: %d, f1-score: %.5f' %(epoch + 1, tempf1_score))
        print('Epoch: %d, loss: %.5f, train_acc: %.2f, f1-score (on test dataset): %.3f' %(epoch + 1, loss.item(), acc, tempf1_score))
        f1_scores.append(tempf1_score)  #Store epoch value for plotting
    #Uncomment to save a check point
    #if epoch % 60 == 59:
    #  print("saving model check point")
    #  checkpoint = {
    #      'epoch': epoch + 1,
    #      'state_dict': net.state_dict(),
    #      'optimizer': optimizer.state_dict()
    #  }
    #  save_ckp(checkpoint, False, "/content/SeqCheckPoint_"+str(epoch)+".pt","/content/SeqCheckPoint_"+str(epoch)+".pt")  


print('Finished Training of Sequence Model')

#Uncomment during debugging
#print(epoch_values)  
#print(f1_scores)

import datetime

from google.colab import drive
drive.mount('/content/gdrive')
SeqModelName = "SeqModelAssignment1V2-"+datetime.datetime.now().strftime("%d-%m-%Y-%H-%M")+"_LearningRate"+str(learning_rate)+"_n_hidden"+str(n_hidden)+"_total_epoch"+str(total_epoch)+"_batch_size"+str(seq_batch_size)+".pt"

seqpath = F"/content/gdrive/My Drive/"+SeqModelName 
torch.save(net, seqpath)
drive.flush_and_unmount()
from google.colab import drive
drive.mount('/content/gdrive')

#SeqModelName = "SeqModelAssignment1-16-04-2020-10-01_LearningRate0.01_n_hidden50_total_epoch240_batch_size250_droput_0.2.pt" #comment this to use above saved model

seqpath = F"/content/gdrive/My Drive/"+SeqModelName 
SeqEmbedModel= torch.load(seqpath)
drive.flush_and_unmount()
SeqEmbedModel.eval()
test_targets_all = [] #List to store labels of test reviews
test_predictions_all = [] #List to store predictions of test reviews


for k in range(25):  #As we have 1000 in each batch, we are looping 25 times to cover the entire test set. The below function will not randomly pick items.
  test_seq_batch, test_target_seq_batch = make_embed_forseq(text_test_pad,label_test_n,word_dict,trainedword2vec_embeddings,trained_char_based_word_embeddings,1000,k)
  testinput_batch_torch = torch.from_numpy(test_seq_batch).float().to(device)

  test_outputs = SeqEmbedModel(testinput_batch_torch)  #Perform predictions for test set
  _, test_predictions = torch.max(test_outputs, 1)

  test_targets_all.extend(test_target_seq_batch)   #Store labels
  test_predictions_all.extend(test_predictions.cpu().numpy()) #Store predictions


from sklearn.metrics import classification_report
print(classification_report(test_targets_all,test_predictions_all)) #Calculating precision, recall, f1 for test set.
# Please comment your code
import matplotlib.pyplot as plt

epochs = [120,240,480,600,720,960]   #These values are captured from previous precitions on test data
f1_scores_testset = [0.33,0.43,0.34,0.43,0.33,0.43] #These values are captured from previous precitions on test data

plt.plot(epochs,f1_scores_testset, label="Learning Rate = 0.01")
plt.xticks(epochs)
plt.xlabel('Epochs')
plt.ylabel('Weighted F1 Score')
plt.title('Epochs vs Weighted F1 Score on entire Test Set')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.show()
# Please comment your code
import matplotlib.pyplot as plt

epochs = [120,240,480,600,720,960]   #These values are captured from previous precitions on test data
f1_scores_testset = [0.33,0.43,0.34,0.33,0.33,0.43] #These values are captured from previous precitions on test data

plt.plot(epochs,f1_scores_testset, label="Learning Rate = 0.02")
plt.xticks(epochs)
plt.xlabel('Epochs')
plt.ylabel('Weighted F1 Score')
plt.title('Epochs vs Weighted F1 Score on entire Test Set')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.show()
# Please comment your code
import matplotlib.pyplot as plt

epochs = [120,240,480,600,720,960]   #These values are captured from previous precitions on test data
f1_scores_testset = [0.33,0.43,0.43,0.33,0.33,0.43] #These values are captured from previous precitions on test data

plt.plot(epochs,f1_scores_testset, label="Learning Rate = 0.05")
plt.xticks(epochs)
plt.xlabel('Epochs')
plt.ylabel('Weighted F1 Score')
plt.title('Epochs vs Weighted F1 Score on entire Test Set')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.show()

plt.plot(epoch_values,f1_scores)
plt.xlabel('Epochs')
plt.ylabel('Weighted F1 Score')
plt.title('Epochs vs Weighted F1 Score for one batch of Test Set')
plt.show()