import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import re
import string
from string import digits
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
print(os.listdir("../input"))
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
print("\nTrain data: \n",train.head())
print("\nTest data: \n",test.head())
train_data=train.drop(train.columns[0], axis=1) 
test_data=test
print(train_data.head())
print(test_data.head())
train_comments=train_data.iloc[:,0]
test_comments=test_data.iloc[:,1]

#saving index to separate them later
train_comments_index=train_comments.index
test_comments_index=test_comments.index

frames = [train_comments, test_comments]
comments = pd.concat(frames, ignore_index=True)


labels=train_data.iloc[:,1:]

print("Train Comments Shape: ",train_comments.shape)
print("Test Comments Shape: ",test_comments.shape)
print("Comments Shape after Merge: ",comments.shape)
print("Comments are: \n",comments.head())
print("\nLabels are: \n", labels.head())
c=comments.str.translate(str.maketrans(' ', ' ', string.punctuation))
c.head()
c=c.str.translate(str.maketrans(' ', ' ', '\n'))
c=c.str.translate(str.maketrans(' ', ' ', digits))
c.head()
c=c.apply(lambda tweet: re.sub(r'([a-z])([A-Z])',r'\1 \2',tweet))
c.head()
c=c.str.lower()
c.head()
c=c.str.split()
c.head()
stop = set(stopwords.words('english'))
c=c.apply(lambda x: [item for item in x if item not in stop])
c.head()    
from tqdm import tqdm
lemmatizer = WordNetLemmatizer()
com=[]
for y in tqdm(c):
    new=[]
    for x in y:
        z=lemmatizer.lemmatize(x)
        z=lemmatizer.lemmatize(z,'v')
        new.append(z)
    y=new
    com.append(y)
clean_data=pd.DataFrame(np.array(com), index=comments.index,columns={'comment_text'})
clean_data['comment_text']=clean_data['comment_text'].str.join(" ")
print(clean_data.head())
train_clean_data=clean_data.loc[train_comments_index]
test_clean_data=clean_data.drop(train_comments_index,axis=0).reset_index(drop=True)
print("PreProcessed Train Data : ",train_clean_data.head(5))
print("PreProcessed Test Data : ",test_clean_data.head(5))
frames=[train_clean_data,labels]
train_result = pd.concat(frames,axis=1)
frames=[test.iloc[:,0],test_clean_data]
test_result = pd.concat(frames,axis=1)
print(train_result.head())
print(test_result.head())
train_result.to_csv('train_data.csv', index = False)
test_result.to_csv('test_data.csv', index = False)
