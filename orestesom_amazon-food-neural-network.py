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
df = pd.read_csv('../input/amazon-fine-food-reviews/Reviews.csv')
df.head()
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12,8), dpi=90)


g = sns.countplot(x='Score', data=df, order=df['Score'].value_counts().index)
plt.xlabel("Stars", color='green')
plt.ylabel("Count", color='green')

plt.title('Reviews rate');

#change the score to a binary mode.

df['score_binary'] = df['Score'].apply(lambda x: 1  if x >=4 else 0)
df.head()[['Summary','score_binary']]
plt.figure(figsize=(12,8), dpi=90)

g = sns.countplot(x='score_binary', data=df)
plt.xlabel("Type or review", color='green')
plt.ylabel("Count", color='green')

plt.xticks((0,1),('Negative','Positive'))
plt.title('Types or Reviews in data');

#Commentarist with more reviews

commentarist = df['ProfileName'].value_counts().sort_values(ascending=False)
commentarist.head(10)

nan_values = df[['ProfileName', 'HelpfulnessNumerator', 'Score', 'Time', 'Summary', 'Text']].isna().sum().sort_values(ascending=False)
nan_values
summary_nan = df.loc[df['Summary'].isna()]

summary_nan[['HelpfulnessNumerator', 'Score', 'Time', 'Summary', 'Text']].head(10)
duplicates = df[df.duplicated(['Text'])]
duplicates = duplicates.sort_values(by='Text')
duplicates[['Score','Text']].head(10)
print('Rows in data before removing duplicates: ' + str(df.shape[0]))
df = df.drop_duplicates(subset=['Score','Text'], keep='last')
print('-'*20)
print('Rows in data after removed the duplicates: ' + str(df.shape[0]))
#shuffle the data
df = df.sample(frac=1)

#size of split
split_train_size = 0.9

#make the int of split
split = int(len(df) * split_train_size)

#-----------------------------------------------------#
#take the text
text = df.Text.values

#convert the text into train and val
train_text, val_text = text[:split], text[split:]

#-----------------------------------------------------#
#create the labels for the reviews
labels = [{'POSITIVE':bool(y), 'NEGATIVE':not bool(y) }
         for y in df.score_binary.values]


train_labels = [{'cats':labels} for labels in labels[:split]]
val_labels = [{'cats':labels} for labels in labels[split:]]


print(train_text[:2], train_labels[:2])
print('-'*25)
print(val_text[:2], val_labels[:2])
#delete the df to save space in memory
del df
import spacy

#nlp is a empty model
nlp = spacy.blank('en')

#create the text categorizer
textcat = nlp.create_pipe(
              'textcat',
              config={
                  'exclusive_classes':True,
                  'architecture': 'bow'})

#adding the pipe to the model
nlp.add_pipe(textcat)


#add the labels to the text categorizer
textcat.add_label('NEGATIVE')
textcat.add_label('POSITIVE')

print(nlp.meta)
from spacy.util import minibatch, compounding
import random


def train_model(model, train_data, optimizer):
    losses = {}
    random.shuffle(train_data)
    
    batches = minibatch(train_data, size=compounding(4.0, 32.0, 1.001))
    for batch in batches:
        text, labels = zip(*batch)
        model.update(text, labels,  drop=0.10, sgd=optimizer, losses=losses)
        
    return losses

optimizer = nlp.begin_training()
train_data = list(zip(train_text, train_labels))
losses = train_model(nlp, train_data, optimizer)
print(losses['textcat'])
def predictions(model, texts):
    docs = [model.tokenizer(texts) for texts in texts]
    
    textcat = model.get_pipe('textcat')
    scores, _ = textcat.predict(docs)
    
    predicted_class = scores.argmax(axis=1)
    
    return predicted_class


text = val_text[10:15]
predictions_test = predictions(nlp, text)

for p, t in zip(predictions_test, text):
    print(f"{textcat.labels[p]}: {t} \n")
    
 
def evaluation_model(model, text, labels):
    #predict the classes   
    predicted_class = predictions(model, text)
    #get the true classes
    true_classes = [int(each['cats']['POSITIVE']) for each in labels]
    #get the one that are truly positive
    correct_classes = predicted_class == true_classes
    #and get the score (mean value)
    accuracy = correct_classes.mean()
    
    return accuracy

evaluation_bow_model =  evaluation_model(nlp, val_text, val_labels)

print('Score in evaluation bow model: ' + str(evaluation_bow_model))
#get the total predictions of the validation set
total_predictions = predictions(nlp, val_text)
val_labels_predicted = [textcat.labels[p] for p in total_predictions]
validations = [each['cats']['POSITIVE'] for each in val_labels]
val_labels_ = ['POSITIVE' if i==True else 'NEGATIVE' for i in validations]

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score 


font = {'family': 'serif',
        'color':  'blue',
        'weight': 'normal',
        'size': 16,
        }

#and make a confusion matrix to analisys the model
labels = ['POSITIVE', 'NEGATIVE']

plt.figure(figsize=(12,8), dpi=90)

cm = confusion_matrix(val_labels_, val_labels_predicted, labels=labels)

sns.heatmap(cm,
           annot=True,
           fmt='d',
           xticklabels=labels,
           yticklabels=labels,
           cmap='Greens')
plt.xlabel('Actual', fontdict=font, labelpad=20)
plt.ylabel('Predicted', fontdict=font, labelpad=20)


print(accuracy_score(val_labels_,val_labels_predicted))
print(classification_report(val_labels_,val_labels_predicted))