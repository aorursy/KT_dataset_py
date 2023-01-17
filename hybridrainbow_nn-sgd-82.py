import re
import tensorflow as tf
import torch
from torch import nn
from torch import functional as F
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from scipy.special import softmax
from sklearn.feature_extraction import text
from scipy.sparse import csr_matrix
nltk.download('punkt')
nltk.download('stopwords')
stemming = PorterStemmer()
stops = set(stopwords.words("english"))
df = pd.read_csv('train.csv')
df.rename(columns={' Review': 'Review', ' Prediction':
                   'Prediction'}, inplace=True)
df = df.dropna()
feature_columns = 'Review'
target_column = 'Prediction'

df1 = pd.read_csv('test.csv')

def apply_cleaning_function_to_list(text_to_clean):
    cleaned_text = []
    for raw_text in text_to_clean:
        cleaned_text.append(clean_text(raw_text))
    return cleaned_text


def clean_text(raw_text):
    tokens = nltk.word_tokenize(raw_text)
    phrase_list = [w.lower() for w in tokens if w.isalpha()]
    meaningful_words = [w for w in phrase_list if not w in stops]
    stemmed_words = [stemming.stem(w) for w in meaningful_words]
    phrase = ""
    for words in stemmed_words:
      phrase += words + ' '
    return phrase
text_to_clean = list(df['Review'])
cleaned_text = apply_cleaning_function_to_list(text_to_clean)
df['Review'] = cleaned_text
df['Review'].replace('', np.nan, inplace=True)
df = df.dropna()
df = df.loc[(df['Review'].str.len() != 0), :]
del cleaned_text
del text_to_clean

test_text_to_clean = list(df1['Review'])
test_cleaned_text = apply_cleaning_function_to_list(test_text_to_clean)
df1['Review'] = test_cleaned_text
df1['Review'].replace('bad', np.nan, inplace=True)
test_text = list(df1['Review'])
for i in range(len(test_text)):
  if len(test_text[i]) == 0:
    test_text[i] = "good"
df1['Review'] = test_text
#df1.to_csv(r'/content/df1.csv')
del test_cleaned_text
del test_text_to_clean
del test_text

y = list(df['Prediction'])
my_stop_words = text.ENGLISH_STOP_WORDS.union(["book"])
vectorizer = TfidfVectorizer(stop_words=my_stop_words, ngram_range=(1, 2),max_features = 5000,norm ='l2')


X_test_train = vectorizer.fit_transform(list(df['Review']) + list(df1['Review'])).toarray()
X = X_test_train[:len(list(df['Review']))]
test_X = X_test_train[len(list(df['Review'])):] 
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=42, test_size = 0.30)
X_train = torch.FloatTensor(X_train)
y_train = torch.LongTensor(y_train)
X_test = torch.FloatTensor(X_test)
y_test = torch.LongTensor(y_test)
def batch_generator(X, y, batch_size):
    np.random.seed(42)
    perm = np.random.permutation(len(X))
    num_samples = X.shape[0]
    num_batches = num_samples // batch_size
    flag = num_samples % batch_size
    X = X[perm]
    y = y[perm]
    for i in range(num_batches):
        yield X[batch_size * i:batch_size * (i + 1)], y[batch_size * i : batch_size * (i+1)]
    if flag != 0:
      yield X[-flag:], y[-flag:]
torch.manual_seed(42) 
np.random.seed(42)
def stack_layers(N):
  torch.manual_seed(42)
  np.random.seed(42)
  head = nn.Sequential(
    nn.Linear(len(X_train[0]), 1200),
    nn.Sigmoid(),
    nn.Dropout(p=0.1),
    nn.BatchNorm1d(1200)
  )
  blocks = []
  for i in range(N):
    torch.manual_seed(42)   
    np.random.seed(42)
    block = nn.Sequential(
      nn.Linear(500, 500),
      nn.ReLU6(),
      nn.Dropout(p=0.3),
      nn.BatchNorm1d(500)
      )
    blocks.append(block)
  torch.manual_seed(42)   
  np.random.seed(42)
  tail = nn.Sequential(nn.Linear(1200,len(target_column)))
  return nn.Sequential(head, *blocks, tail)

model = stack_layers(0)    
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.3, momentum=0.3, nesterov=True)


def train(X_train, y_train, X_test, y_test, num_epoch):
    train_losses = []
    test_losses = []
    for i in range(num_epoch):
        epoch_train_losses = []
        for X_batch, y_batch in batch_generator(X_train, y_train, 128):
            model.train(True)
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            
            optimizer.zero_grad()
            
            
          
            loss.backward()
            
            
            optimizer.step()
            
            
            epoch_train_losses.append(loss.item())        
        train_losses.append(np.mean(epoch_train_losses))
        model.train(False)
        with torch.no_grad():
            # Сюда опять же надо положить именно число равное лоссу на всем тест датасете
            test_losses.append(loss_fn(model(X_test),y_test).item())
            
    return train_losses, test_losses
train_losses, test_losses = train(X_train,y_train,X_test,y_test,num_epoch=33) #Подберите количество эпох так, чтобы график loss сходился
plt.plot(range(len(train_losses)), train_losses, label='train')
plt.plot(range(len(test_losses)), test_losses, label='test')
plt.legend()
plt.show()
model.eval()
train_pred_labels = (model.forward(X_train)).max(1)[1]
test_pred_labels = (model.forward(X_test)).max(1)[1]#YOUR CODE: use forward
train_acc = accuracy_score(train_pred_labels,y_train)
test_acc = accuracy_score(test_pred_labels, y_test)
print("Train accuracy: {}\nTest accuracy: {}".format(train_acc, test_acc))
df1 = pd.read_csv('test.csv')
print(df1.head())
df1 = df1.dropna()
test_X = torch.FloatTensor(test_X)
model.eval()
test_y = model(test_X)
print(test_y[0])
#= test_y.detach().numpy()"""
prob = torch.nn.functional.softmax(test_y, dim=1)
bb = prob.detach().numpy()
print(bb[0].argmax())
Predictions = []
for i in prob:
  Predictions.append(i.detach().numpy().argmax())
print(Predictions)
csvfilelist = [['Id', 'Prediction']]
for k in range(len(Predictions)):
  mylist = []
  mylist.append(k+1)
  mylist.append(Predictions[k])
  csvfilelist.append(mylist)
mylist = []
for k in range(len(Predictions)):
  mylist.append(k+1)
"""preddf = pd.DataFrame(csvfilelist)
print(preddf)"""
preddf = pd.DataFrame(data={"Id": mylist, "Prediction": Predictions})
preddf.to_csv("prediction.csv", sep=',',index=False)
#preddf.to_csv('prediction.csv', index=False)
print(preddf)