# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import nltk
import random
import os
from os import path
from PIL import Image

# Visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from subprocess import check_output
from wordcloud import WordCloud, STOPWORDS

# Set Plot Theme
sns.set_palette([
    "#30a2da",
    "#fc4f30",
    "#e5ae38",
    "#6d904f",
    "#8b8b8b",
])

from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import re
from nltk.stem import PorterStemmer

# Modeling
import statsmodels.api as sm
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.sentiment.util import *
from nltk.util import ngrams
from collections import Counter
from gensim.models import word2vec

# Warnings
import warnings
warnings.filterwarnings('ignore')


import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import tensorflow as tf
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense , Dropout, Bidirectional,SpatialDropout1D,Flatten
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

df = pd.read_csv("/kaggle/input/womens-ecommerce-clothing-reviews/Womens Clothing E-Commerce Reviews.csv",index_col=0)
df.head()
plt.figure(figsize=(10,6))
sns.heatmap(df.isnull(),cmap='viridis')
plt.show()
print("Dataframe Dimension: {} Rows, {} Columns".format(*df.shape))
df[["Title", "Division Name","Department Name","Class Name"]].describe(include=["O"]).T.drop("count",axis=1)
sns.countplot(df['Recommended IND'])
plt.title("Count of recommended vs non recommended items")
plt.show()
# Continous Distributions
fig = plt.figure(figsize=(20,14))
ax1 = plt.subplot2grid((2,2),(0,0))
ax1 = sns.distplot(df["Positive Feedback Count"])
ax1 = plt.title("Positive Feedback Count Distribution")

ax2 = plt.subplot2grid((2,2),(0,1))
ax2 = sns.distplot(df['Age'])
ax2 = plt.title("Age distribution")

ax3 = plt.subplot2grid((2,2),(1,0),colspan=2)
ax3 = sns.distplot(np.log10((df["Positive Feedback Count"][df["Positive Feedback Count"].notnull()]+1)))
ax3 = plt.title("Log Positive Feedback count")

plt.show()
def percentage_accumulation(series, percentage):
    return (series.sort_values(ascending=False)
            [:round(series.shape[0]*(percentage/100))]
     .sum()/series
     .sum()*100)

# Gini Coefficient- Inequality Score
# Source: https://planspace.org/2013/06/21/how-to-calculate-gini-coefficient-from-raw-data-in-python/
def gini(list_of_values):
    sorted_list = sorted(list_of_values)
    height, area = 0, 0
    for value in sorted_list:
        height += value
        area += height - value / 2.
    fair_area = height * len(list_of_values) / 2.
    return (fair_area - area) / fair_area

inequality = []
for x in list(range(100)):
    inequality.append(percentage_accumulation(df["Positive Feedback Count"], x))
plt.plot(inequality)
plt.title("Percentage of Positive Feedback by Percentage of Reviews")
plt.xlabel("Review Percentile starting with Feedback")
plt.ylabel("Percent of Positive Feedback Received")
plt.axvline(x=20, c = "r")
plt.axvline(x=53, c = "g")
plt.axhline(y=78, c = "y")
plt.axhline(y=100, c = "b", alpha=.3)
plt.show()
print("{}% of Positive Feedback belongs to the top 20% of Reviews".format(
    round(percentage_accumulation(df["Positive Feedback Count"], 20))))

# Gini
print("\nGini Coefficient: {}".format(round(gini(df["Positive Feedback Count"]),2)))
fig = plt.figure(figsize=(14,8))
ax1 = plt.subplot2grid((2,2),(0,0))
ax1 = sns.heatmap(pd.crosstab(df['Division Name'],df['Department Name']),cmap='Purples',annot=True, linewidths=.5,fmt='g',
                cbar_kws={'label': 'Count'})
ax1 = plt.title('Division Name Count by Department Name - Crosstab\nHeatmap Overall Count Distribution')

ax2 = plt.subplot2grid((2,2),(0,1))
ax2 = sns.heatmap(pd.crosstab(df['Division Name'],df['Department Name'],normalize=True).mul(100).round(0),cmap='Purples',annot=True,fmt='g',cbar_kws={'label': 'Percentage %'})
ax2 = plt.title("Division Name Count by Department Name - Crosstab\nHeatmap Overall Percentage Distribution")
plt.tight_layout(pad=0)
plt.show()

fig = plt.figure(figsize=(14,8))
ax1 = plt.subplot2grid((2,2),(0,0))
ax1 = sns.heatmap(pd.crosstab(df['Division Name'], df["Department Name"], normalize='columns').mul(100).round(0),
            annot=True, linewidths=.5,fmt='g', cmap="Purples",cbar_kws={'label': 'Percentage %'})
ax1 = plt.title('Division Name Count by Department Name - Crosstab\nHeatmap % Distribution by Columns')

ax2 = plt.subplot2grid((2,2),(0,1))
ax2 = sns.heatmap(pd.crosstab(df['Division Name'], df["Department Name"], normalize='index').mul(100).round(0),
            annot=True, linewidths=.5,fmt='g', cmap="Purples",cbar_kws={'label': 'Percentage %'})
ax2 = plt.title("Division Name Count by Department Name - Crosstab\nHeatmap % Distribution by Index")
plt.tight_layout(pad=0)
plt.show()

fig = plt.figure(figsize=(14,8))
ax1 = plt.subplot2grid((2,2),(0,0))
ax1 = sns.heatmap(pd.crosstab(df['Class Name'],df['Department Name']),cmap='inferno_r',annot=True, linewidths=.5,fmt='g',
                cbar_kws={'label': 'Count'})
ax1 = plt.title('Class Name Count by Department Name - Crosstab\nHeatmap Overall Count Distribution')

ax2 = plt.subplot2grid((2,2),(0,1))
ax2 = sns.heatmap(pd.crosstab(df['Class Name'],df['Department Name'],normalize=True).mul(100).round(0),cmap='inferno_r',annot=True,fmt='g',cbar_kws={'label': 'Percentage %'})
ax2 = plt.title("Class Name Count by Department Name - Crosstab\nHeatmap Overall Percentage Distribution")
plt.tight_layout(pad=0)
plt.show()
fig = plt.figure(figsize=(14,8))
ax1 = plt.subplot2grid((2,2),(0,0))
ax1 = sns.heatmap(pd.crosstab(df['Class Name'], df["Department Name"], normalize='columns').mul(100).round(0),
            annot=True, linewidths=.5,fmt='g', cmap="nipy_spectral_r",cbar_kws={'label': 'Percentage %'})
ax1 = plt.title('Class Name Count by Department Name - Crosstab\nHeatmap % Distribution by Columns')

ax2 = plt.subplot2grid((2,2),(0,1))
ax2 = sns.heatmap(pd.crosstab(df['Class Name'], df["Department Name"], normalize='index').mul(100).round(0),
            annot=True, linewidths=.5,fmt='g', cmap="nipy_spectral_r",cbar_kws={'label': 'Percentage %'})
ax2 = plt.title("Class Name Count by Department Name - Crosstab\nHeatmap % Distribution by Index")
plt.tight_layout(pad=0)
plt.show()

fig = plt.figure(figsize=(14,8))
ax1 = plt.subplot2grid((2,2),(0,0))
ax1 = sns.heatmap(pd.crosstab(df['Class Name'],df['Division Name']),cmap='cubehelix_r',annot=True, linewidths=.5,fmt='g',
                cbar_kws={'label': 'Count'})
ax1 = plt.title('Class Name Count by Division Name - Crosstab\nHeatmap Overall Count Distribution')

ax2 = plt.subplot2grid((2,2),(0,1))
ax2 = sns.heatmap(pd.crosstab(df['Class Name'],df['Division Name'],normalize=True).mul(100).round(0),cmap='cubehelix_r',annot=True,fmt='g',cbar_kws={'label': 'Percentage %'})
ax2 = plt.title("Class Name Count by Division Name - Crosstab\nHeatmap Overall Percentage Distribution")
plt.tight_layout(pad=0)
plt.show()
fig = plt.figure(figsize=(14,8))
ax1 = plt.subplot2grid((2,2),(0,0))
ax1 = sns.heatmap(pd.crosstab(df['Class Name'], df["Division Name"], normalize='columns').mul(100).round(0),
            annot=True, linewidths=.5,fmt='g', cmap="ocean_r",cbar_kws={'label': 'Percentage %'})
ax1 = plt.title('Class Name Count by Division Name - Crosstab\nHeatmap % Distribution by Columns')

ax2 = plt.subplot2grid((2,2),(0,1))
ax2 = sns.heatmap(pd.crosstab(df['Class Name'], df["Division Name"], normalize='index').mul(100).round(0),
            annot=True, linewidths=.5,fmt='g', cmap="ocean_r",cbar_kws={'label': 'Percentage %'})
ax2 = plt.title("Class Name Count by Division Name - Crosstab\nHeatmap % Distribution by Index")
plt.tight_layout(pad=0)
plt.show()
def minmaxscaler(df):
    return (df-df.min())/(df.max()-df.min())
def zscorenomalize(df):
    return (df - df.mean())/df.std()


df.describe()
print(df.info())
sns.set(rc={'figure.figsize':(11,4)})
pd.isnull(df).sum().plot(kind='bar',color='royalblue')
plt.ylabel('count')
plt.title('Missing values')
plt.show()
sns.set(rc={'figure.figsize':(16,6)})
plt.hist(df['Age'],bins=50,color='royalblue')
plt.xlabel("Age")
plt.ylabel("Count")
plt.title("Age Distribution")
plt.show()
sns.set(rc={"figure.figsize":(14,6)})
sns.boxplot(x='Rating',y='Age',data=df)
plt.title('Rating Distribution per Age')
plt.show()
sns.countplot(df['Rating'])
plt.title("Rating count")
plt.show()
fig = plt.figure(figsize=(20,14))
ax1 = plt.subplot2grid((2,2),(0,0))
# ax1 = plt.xticks(rotation=90)
ax1 = sns.countplot(df['Division Name'])
ax1 = plt.title("Review in each division")

ax2 = plt.subplot2grid((2,2),(0,1))
ax2 = plt.xticks(rotation=90)
ax2 = sns.countplot(df['Department Name'])
ax2 = plt.title("Review in each department")

ax3 = plt.subplot2grid((2,2),(1,0),colspan=2)
ax3 = plt.xticks(rotation=90)
ax3 = sns.countplot(df['Class Name'])
ax3 = plt.title("Reviews in each Class")

df.head(2)
recommended = df[df['Recommended IND']==1]
not_recommended = df[df['Recommended IND']==0]
recommended
fig = plt.figure(figsize=(20, 14))
ax1 = plt.subplot2grid((2, 2), (0, 0))
ax1 = sns.countplot(recommended['Division Name'],color='red',alpha=1,label="recommended")
ax1 = sns.countplot(not_recommended['Division Name'],color='black',alpha=1,label="non-recommended")
ax1 = plt.title("Recommended Items in each Division")
ax1 = plt.legend(loc='best')

ax2 = plt.subplot2grid((2, 2), (0, 1))
ax2 = sns.countplot(recommended['Department Name'],color='blue',alpha=1,label="recommended")
ax2 = sns.countplot(not_recommended['Department Name'],color='black',alpha=1,label="non-recommended")
ax2 = plt.title("Recommended Items in each Department")
ax2 = plt.legend(loc='best')

ax3 = plt.subplot2grid((2, 2), (1, 0),colspan=2)
ax3 = sns.countplot(recommended['Class Name'],color='orange',alpha=1,label="recommended")
ax3 = sns.countplot(not_recommended['Class Name'],color='black',alpha=1,label="non-recommended")
ax3 = plt.title("Recommended Items for each class")
ax3 = plt.legend(loc='best')

def percentstandardize_barplot(x,y,hue, data, ax=None, order= None):
    sns.barplot(x= x, y=y, hue=hue, ax=ax, order=order,
    data=(data[[x, hue]]
     .reset_index(drop=True)
     .groupby([x])[hue]
     .value_counts(normalize=True)
     .rename('Percentage').mul(100)
     .reset_index()
     .sort_values(hue)))
    plt.title("Percentage Frequency of {} by {}".format(hue,x))
    plt.ylabel("Percentage %")
hue = "Recommended IND"
fig = plt.figure(figsize=(20, 14))

ax1 = plt.subplot2grid((2, 2), (0, 0))
ax1 = percentstandardize_barplot(x="Department Name",y="Percentage",hue=hue,data=df)
ax1 = plt.title("Recommended Items in each Department")
ax1 = plt.legend(loc='best')

ax2 = plt.subplot2grid((2, 2), (0, 1))
ax2 = percentstandardize_barplot(x="Division Name",y="Percentage", hue=hue,data=df)
ax2 = plt.title("Recommended Items in each Division")
ax2 = plt.legend(loc='best')

ax3 = plt.subplot2grid((2, 2), (1, 0),colspan=2)
ax3 = percentstandardize_barplot(x="Class Name",y="Percentage", hue=hue,data=df)
ax3 = plt.title("Recommended Items for each class")
ax3 = plt.legend(loc='best')
plt.show()
xvar = ["Department Name","Division Name","Class Name"]
hue = "Rating"
fig = plt.figure(figsize=(20, 14))

ax1 = plt.subplot2grid((2, 2), (0, 0))
ax1 = percentstandardize_barplot(x=xvar[0],y="Percentage", hue=hue,data=df)
ax1 = plt.title("Percentage Frequency of {}\nby {}".format(hue, xvar[0]))
ax1 = plt.ylabel("Percentage %")

ax2 = plt.subplot2grid((2, 2), (0,1))
ax2 = percentstandardize_barplot(x=xvar[1],y="Percentage", hue="Rating",data=df)
ax2 = plt.title("Percentage Frequency of {}\nby {}".format(hue, xvar[1]))
ax2 = plt.ylabel("Percentage %")

ax3 = plt.subplot2grid((2, 2), (1,0),colspan=2)
ax2 = plt.xticks(rotation=45)
ax3 = percentstandardize_barplot(x=xvar[2],y="Percentage", hue="Rating",data=df)
ax3 = plt.title("Percentage Frequency of {}\nby {}".format(hue, xvar[2]))
ax3 = plt.ylabel("Percentage %")

plt.show()
hue = "Rating"
fig = plt.figure(figsize=(20, 14))

ax1 = plt.subplot2grid((2, 2), (0, 0))
ax1 = sns.countplot(x="Rating", hue="Recommended IND",data=df)
ax1 = plt.title("Occurrence of {}\nby {}".format(hue, "Recommended IND"))
ax1 = plt.ylabel("Count")

ax2 = plt.subplot2grid((2,2),(0,1))
ax2 = percentstandardize_barplot(x="Rating",y="Percentage", hue="Recommended IND",data=df)
ax2 = plt.title("Percentage Normalized Occurrence of {}\nby {}".format(hue, "Recommended IND"))
ax1 = plt.ylabel("% Percentage by Rating")
plt.show()
fig = plt.figure(figsize=(18,8))
plt.xticks(rotation=90)
plt.xlabel('Clothing ID')
plt.ylabel("Popularity")
plt.title("ID of Top 50 Clothing Items")
df['Clothing ID'].value_counts()[:30].plot(kind='bar',color='royalblue')
plt.show()
g = sns.jointplot(x= df["Positive Feedback Count"], y=df["Age"], kind='reg', color='royalblue')
g.fig.suptitle("Scatter Plot for Age and Positive Feedback Count")
plt.show()
fig = plt.figure(figsize=(20,14))
ax1 = plt.subplot2grid((2,2),(0,0))
ax1 = sns.boxplot(x="Division Name",y='Rating',data=df)
ax1 = plt.title('Rating Distribution per Division')

ax2 = plt.subplot2grid((2,2),(0,1))
ax2 = sns.boxplot(x="Department Name",y='Rating',data=df)
ax2 = plt.title('Rating Distribution per Department')

ax3 = plt.subplot2grid((2,2),(1,0),colspan=2)
ax3 = plt.xticks(rotation=45)
ax3 = sns.boxplot(x="Class Name",y='Rating',data=df)
ax3 = plt.title('Rating Distribution per Class')
df.head()
df.columns
df['Review Text'][0]
data = df[['Review Text','Rating']]
data.head()
data.isnull().sum()
data.dropna(inplace=True)
data.isnull().sum()
data.shape
X = data['Review Text']
y = pd.get_dummies(data['Rating']).values
# y = data['Rating']
messages = X.copy()
messages = list(messages)

type(messages)
voc_size = len(X)

from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
corpus = []
for i in range(0, len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages[i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)

corpus[1]
onehot_repr=[one_hot(words,voc_size)for words in corpus] 
sent_length=40
embedded_docs=pad_sequences(onehot_repr,padding='post',maxlen=sent_length)
print(embedded_docs[0:5])

X = np.array(embedded_docs)
y = np.array(y)
print(X.shape,y.shape)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)
EMBEDDING_DIM=100
model = Sequential()
model.add(Embedding(voc_size, EMBEDDING_DIM, input_length=sent_length))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2,return_sequences=True))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100,dropout=0.2,recurrent_dropout=0.2))
model.add(Dense(5, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print("\n")
print("-------------MODEL SUMMARY--------------")
print("\n")
model.summary()
epochs = 100
batch_size = 64
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto',restore_best_weights=True)
print("\n")
print("-------------STARTING TRAINING--------------")
print("\n")
history = model.fit(X_train,y_train,validation_split=0.2,epochs=epochs,batch_size=batch_size,callbacks=[es])
print("\n")
print("-------------TRAINING COMPLETED--------------")
print("\n")
accr = model.evaluate(X_test,y_test)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))


from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report
# Import and evaluate each test batch using Matthew's correlation coefficient
from sklearn.metrics import accuracy_score,matthews_corrcoef

from tqdm import tqdm, trange,tnrange,tqdm_notebook
import random
import os
import io

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader,RandomSampler,TensorDataset,SequentialSampler
from transformers import BertTokenizer, BertConfig,AdamW, BertForSequenceClassification,get_linear_schedule_with_warmup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
torch.cuda.get_device_name(0)
SEED = 10
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if device == torch.device("cuda"):
    torch.cuda.manual_seed_all(SEED)
device = torch.device("cuda")
data.head()
sentences = data['Review Text'].values
MAX_LEN = 256
# importing bert tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',do_lower_case=True)
input_ids = [tokenizer.encode(sent, add_special_tokens=True,max_length=MAX_LEN,pad_to_max_length=True) for sent in sentences]
labels = data['Rating'].values

print("Actual sentence before tokenization: ",sentences[1])
print("Encoded Input from dataset: ",input_ids[1])
attention_mask=[]
attention_mask = [[float(i>0) for i in seq] for seq in input_ids]
print(attention_mask[1])
X_train,X_test,y_train,y_test = train_test_split(input_ids,labels,random_state=41,test_size=0.1)
train_masks, test_masks,_,_  = train_test_split(attention_mask,input_ids,random_state=41,test_size=0.1)
X_train = torch.tensor(X_train)
X_test = torch.tensor(X_test)
y_train = torch.tensor(y_train)
y_test = torch.tensor(y_test)
train_masks = torch.tensor(train_masks)
test_masks = torch.tensor(test_masks)
X_train.shape, y_train.shape, X_test.shape,y_test.shape, train_masks.shape,test_masks.shape
batch_size=32
train_data = TensorDataset(X_train,train_masks,y_train)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data , sampler=train_sampler, batch_size=batch_size)
test_data = TensorDataset(X_test,test_masks,y_test)
test_sampler = RandomSampler(test_data)
test_dataloader = DataLoader(test_data,sampler=test_sampler,batch_size=batch_size)

train_data[0]
len(train_dataloader)
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=6).to(device)
# Parameters:
lr = 2e-5
adam_epsilon = 1e-8
# Number of training epochs (authors recommend between 2 and 4)
epochs = 3
num_warmup_steps = 0
num_training_steps = len(train_dataloader)*epochs
optimizer = AdamW(model.parameters(),lr=lr,eps=adam_epsilon,correct_bias=False)  # To reproduce BertAdam specific behavior set correct_bias=False
scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
## Store our loss and accuracy for plotting
train_loss_set = []
learning_rate = []

# Gradients gets accumulated by default
model.zero_grad()

# tnrange is a tqdm wrapper around the normal python range
for _ in tnrange(1,epochs+1,desc='Epoch'):
  print("<" + "="*22 + F" Epoch {_} "+ "="*22 + ">")
  # Calculate total loss for this epoch
  batch_loss = 0

  for step, batch in enumerate(train_dataloader):
    # Set our model to training mode (as opposed to evaluation mode)
    model.train()
    
    # Add batch to GPU
    batch = tuple(t.to(device) for t in batch)
    # Unpack the inputs from our dataloader
    b_input_ids, b_input_mask, b_labels = batch

    # Forward pass
    outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
    loss = outputs[0]
    
    # Backward pass
    loss.backward()
    
    # Clip the norm of the gradients to 1.0
    # Gradient clipping is not in AdamW anymore
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    # Update parameters and take a step using the computed gradient
    optimizer.step()
    
    # Update learning rate schedule
    scheduler.step()

    # Clear the previous accumulated gradients
    optimizer.zero_grad()
    
    # Update tracking variables
    batch_loss += loss.item()

  # Calculate the average loss over the training data.
  avg_train_loss = batch_loss / len(train_dataloader)

  #store the current learning rate
  for param_group in optimizer.param_groups:
    print("\n\tCurrent Learning rate: ",param_group['lr'])
    learning_rate.append(param_group['lr'])
    
  train_loss_set.append(avg_train_loss)
  print(F'\n\tAverage Training loss: {avg_train_loss}')
    
  # Validation

  # Put model in evaluation mode to evaluate loss on the validation set
  model.eval()

  # Tracking variables 
  eval_accuracy,eval_mcc_accuracy,nb_eval_steps = 0, 0, 0

  # Evaluate data for one epoch
  for batch in test_dataloader:
    # Add batch to GPU
    batch = tuple(t.to(device) for t in batch)
    # Unpack the inputs from our dataloader
    b_input_ids, b_input_mask, b_labels = batch
    # Telling the model not to compute or store gradients, saving memory and speeding up validation
    with torch.no_grad():
      # Forward pass, calculate logit predictions
      logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
    
    # Move logits and labels to CPU
    logits = logits[0].to('cpu').numpy()
    label_ids = b_labels.to('cpu').numpy()

    pred_flat = np.argmax(logits, axis=1).flatten()
    labels_flat = label_ids.flatten()
    
    df_metrics=pd.DataFrame({'Epoch':epochs,'Actual_class':labels_flat,'Predicted_class':pred_flat})
    
    tmp_eval_accuracy = accuracy_score(labels_flat,pred_flat)
    tmp_eval_mcc_accuracy = matthews_corrcoef(labels_flat, pred_flat)
    
    eval_accuracy += tmp_eval_accuracy
    eval_mcc_accuracy += tmp_eval_mcc_accuracy
    nb_eval_steps += 1

  print(F'\n\tValidation Accuracy: {eval_accuracy/nb_eval_steps}')
  print(F'\n\tValidation MCC Accuracy: {eval_mcc_accuracy/nb_eval_steps}')
df_metrics
df_metrics['Actual_class'].unique() , df_metrics['Predicted_class'].unique() 
data[['Review Text','Rating']].drop_duplicates(keep='first')
df.head()
## emotion labels
label2int = {
  "bad": 2,
  "neutral": 3,
  "good": 4,
    "excellent":5
}
print(classification_report(df_metrics['Actual_class'].values, df_metrics['Predicted_class'].values, target_names=label2int.keys(), digits=len(label2int)))
# saving the model
# model_save_folder = 'model/'
# tokenizer_save_folder = 'tokenizer/'

# path_model = F'/kaggle/working/{model_save_folder}'
# path_tokenizer = F'/kaggle/working/{tokenizer_save_folder}'

# #create the dir

# !mkdir -p {path_model}
# !mkdir -p {path_tokenizer}

# ## Now let's save our model and tokenizer to a directory
# model.save_pretrained(path_model)
# tokenizer.save_pretrained(path_tokenizer)

# model_save_name = 'fineTuneModel.pt'
# path = path_model = F'/kaggle/working/{model_save_folder}/{model_save_name}'
# torch.save(model.state_dict(),path);
