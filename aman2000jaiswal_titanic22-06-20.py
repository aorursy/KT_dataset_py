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
import re

import matplotlib.pyplot as plt

import seaborn as sns

from collections import defaultdict
df = pd.read_csv('/kaggle/input/titanic/train.csv')
df.head(5)
df.info()
df.describe()
df[['Sex','Cabin','Embarked','Name','Ticket']].describe()
df['sex'] = df['Sex']=='male'

df['sex'] = df['sex'].astype('int')
df['Cabintype'] = df.Cabin.apply(lambda x: str(x)[0])

df['Cabintype'].unique()
def lastname(name):

    name = str(name)

    x = name.split()

    x.reverse()

    i = 0

    if(x[i][-1]==')'):

        while(x[i][0]!='('):

            i+=1

        i+=1

    return x[i]

        

        

df['lastName']=df['Name'].apply(lambda x: lastname(x))
for i,j in zip(df.Name.values[0:10],df.lastName.values[0:10]):

    print(i,"{}".format(" "*(60 - len(i))),j)
def ticket_letter(ticket):

    ticket = str(ticket)

    letter = re.findall('[A-Za-z]+',ticket)

    if(letter==[]):

        letter = 'Num'

    else:

        letter=letter[0]

    return letter

df['Ticket_letter'] = df['Ticket'].apply(lambda x: ticket_letter(x))
df.head()
cols = ['Pclass','Age','SibSp','Parch','lastName','Fare','Embarked','sex','Cabintype','Ticket_letter']

X = df[cols]

y = df['Survived']
X.info()
sns.boxplot(X.Age)
medianAge = X.Age.median()

print(medianAge)

X.Age.fillna(medianAge,inplace=True)
modeEmbarked = X.Embarked.mode()

print(modeEmbarked[0])

X.Embarked = X.Embarked.fillna(modeEmbarked[0])
X.info()
from sklearn.model_selection import train_test_split

X_train,X_valid,y_train,y_valid = train_test_split(X,y,stratify=y,test_size=0.2,random_state = 31)
from sklearn.preprocessing import LabelEncoder

embarked_le = LabelEncoder()

X_train.Embarked = embarked_le.fit_transform(X_train.Embarked)

X_valid.Embarked = embarked_le.transform(X_valid.Embarked)
cabin_le = LabelEncoder()

X_train['Cabintype'] = cabin_le.fit_transform(X_train.Cabintype)

X_valid['Cabintype'] = cabin_le.transform(X_valid.Cabintype)
dt = defaultdict(int)

for num,i in enumerate(set(X_train.Ticket_letter)):

    dt[i] = num+1

def get_ticket(ticket):

    return dt[ticket]

X_train['Ticket_letter'] = X_train.Ticket_letter.apply(get_ticket)

X_valid['Ticket_letter'] = X_valid.Ticket_letter.apply(get_ticket)
dd = defaultdict(int)

for num,i in enumerate(set(X_train.lastName)):

    dd[i] = num+1

def getlast(name):

    return dd[name]



X_train['lastName'] = X_train['lastName'].apply(getlast)

X_valid['lastName'] = X_valid['lastName'].apply(getlast)
X_train.head()
X_valid.head()
plt.figure(figsize=(10,8))

sns.heatmap(X_train.corr(),cmap='coolwarm_r',annot=True)
X_train = X_train[['Pclass','Age','Fare','Embarked','sex','Ticket_letter','Parch']]

X_valid = X_valid[['Pclass','Age','Fare','Embarked','sex','Ticket_letter','Parch']]
X_train = np.array(X_train,dtype = 'float')

X_valid = np.array(X_valid,dtype = 'float')
y_train = np.array(y_train,dtype='float')

y_train = np.array(y_train,dtype='float')
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(X_train,y_train)

print(lr.score(X_train,y_train))

print(lr.score(X_valid,y_valid))
from sklearn.decomposition import PCA

pca = PCA(n_components=2)

pcavalue = pca.fit_transform(X_train)
plt.figure(figsize=(12,8))

sns.scatterplot(x=pcavalue[:,1],y=pcavalue[:,0],hue=y_train,palette = 'rocket')

plt.xlim([-40,100])

plt.grid()
plt.figure(figsize=(12,8))

sns.scatterplot(x=pcavalue[:,1],y=pcavalue[:,0],hue=y_train,palette = 'rocket')

plt.xlim([-35,0])

plt.grid()
from lightgbm import LGBMClassifier

from xgboost import XGBClassifier

from sklearn.ensemble import RandomForestClassifier
model_lgb = LGBMClassifier()

model = XGBClassifier()

model.fit(X_train,y_train)
model.score(X_train,y_train)
model.score(X_valid,y_valid)
full_X,full_y = np.append(X_train,X_valid,axis=0),np.append(y_train,y_valid,axis=0)
full_X.shape,full_y.shape,X_train.shape,X_valid.shape,
full_y_reshape = full_y.reshape(-1,1)

full = np.append(full_X,full_y_reshape,axis=1)
from sklearn.model_selection import cross_val_score

score = cross_val_score(model,full_X, full_y, cv=5)

score
test_df = pd.read_csv('/kaggle/input/titanic/test.csv')



test_df['sex'] = test_df['Sex']=='male'

test_df['sex'] = test_df['sex'].astype('int')



test_df['Cabintype'] = test_df.Cabin.apply(lambda x: str(x)[0])



test_df['lastName']=test_df['Name'].apply(lambda x: lastname(x))



test_df['Ticket_letter'] = test_df['Ticket'].apply(lambda x: ticket_letter(x))



X_test = test_df[cols]



X_test.Age.fillna(medianAge,inplace=True)



X_test.Embarked = X_test.Embarked.fillna(modeEmbarked[0])



X_test.fillna(0,inplace=True)



X_test.Embarked = embarked_le.transform(X_test.Embarked)



X_test['Cabintype'] = cabin_le.transform(X_test.Cabintype)



X_test['Ticket_letter'] = X_test.Ticket_letter.apply(get_ticket)



X_test['lastName'] = X_test['lastName'].apply(getlast)



X_test = X_test[['Pclass','Age','Fare','Embarked','sex','Ticket_letter','Parch']]

X_test = np.array(X_test,dtype = 'float')
from sklearn.model_selection import KFold

kf = KFold(n_splits=5)



prediction1 = []

valid_prediction = np.array([0.]*891)

for train_index,valid_index in kf.split(full):

    model = XGBClassifier()

    model.fit(full_X[train_index],full_y[train_index])

    valid_prediction+=model.predict(full_X)

    print("train:  ",model.score(full_X[train_index],full_y[train_index]),"     valid: ",model.score(full_X[valid_index],full_y[valid_index]))

    prediction1.append(model.predict(X_test))
c=0

ic =0

for i,j in zip(valid_prediction,full_y):

    if i==0 or i==1:

        if(i==j):

            c+=1

        else:

            ic+=1
c,ic
from sklearn.model_selection import KFold

kf = KFold(n_splits=5)



prediction2 = []

valid_prediction = np.array([0.]*891)

for train_index,valid_index in kf.split(full):

    model = LGBMClassifier()

    model.fit(full_X[train_index],full_y[train_index])

    valid_prediction+=model.predict(full_X)

    print("train:  ",model.score(full_X[train_index],full_y[train_index]),"     valid: ",model.score(full_X[valid_index],full_y[valid_index]))

    prediction2.append(model.predict(X_test))



c=0

ic =0

for i,j in zip(valid_prediction,full_y):

    if i==0 or i==1:

        if(i==j):

            c+=1

        else:

            ic+=1

c,ic
from sklearn.model_selection import KFold

kf = KFold(n_splits=5)



prediction3 = []

valid_prediction = np.array([0.]*891)

for train_index,valid_index in kf.split(full):

    model = RandomForestClassifier()

    model.fit(full_X[train_index],full_y[train_index])

    valid_prediction+=model.predict(full_X)

    print("train:  ",model.score(full_X[train_index],full_y[train_index]),"     valid: ",model.score(full_X[valid_index],full_y[valid_index]))

    prediction3.append(model.predict(X_test))



c=0

ic =0

for i,j in zip(valid_prediction,full_y):

    if i==0 or i==1:

        if(i==j):

            c+=1

        else:

            ic+=1

c,ic
prediction1 = np.array(prediction1)

prediction1 = np.mean(prediction1,axis=0)

prediction1 = (prediction1>0.5)

prediction1  =np.array(prediction1,dtype='int64')



prediction2 = np.array(prediction2)

prediction2 = np.mean(prediction2,axis=0)

prediction2 = (prediction2>0.5)

prediction2  =np.array(prediction2,dtype='int64')



prediction3 = np.array(prediction3)

prediction3 = np.mean(prediction3,axis=0)

prediction3 = (prediction3>0.5)

prediction3  =np.array(prediction3,dtype='int64')
from scipy import stats
prediction = []

for i,j,k in zip(prediction1,prediction2,prediction3):

    prediction.append(stats.mode([i,j,k])[0][0])
prediction[0:10]
from sklearn.model_selection import KFold

kf = KFold(n_splits=5)



prediction = []

for train_index,valid_index in kf.split(full):

    model = LGBMClassifier()

    model.fit(full_X[train_index],full_y[train_index])

    print("train:  ",model.score(full_X[train_index],full_y[train_index]),"     valid: ",model.score(full_X[valid_index],full_y[valid_index]))

    prediction.append(model.predict(X_test))
prediction = np.array(prediction)

prediction = np.mean(prediction,axis=0)

prediction = (prediction>0.5)

prediction  =np.array(prediction,dtype='int64')
submission = pd.read_csv('../input/titanic/gender_submission.csv')

submission.head()
submission['Survived'] = prediction

submission.to_csv('submission.csv',index=False)