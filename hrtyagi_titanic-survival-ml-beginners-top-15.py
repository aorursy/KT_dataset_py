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
import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
train=pd.read_csv("/kaggle/input/titanic/train.csv")

test=pd.read_csv("/kaggle/input/titanic/test.csv")
train_id=train["PassengerId"]

test_id=test["PassengerId"]

# storing unique id for future refrence
train.describe()
train.head()
print(train.Pclass.value_counts(dropna=False))

print("--"*50)

print(train.Embarked.value_counts(dropna=False))

print("--"*50)

print(train.SibSp.value_counts(dropna=False))

print("--"*50)

print(train.Parch.value_counts(dropna=False))

print("--"*50)
train.isnull().sum().sort_values(ascending = False)
mat=train.corr()

fig,ax = plt.subplots(figsize = (10,10))

sns.heatmap(mat,annot = True, annot_kws={'size': 12})
test.describe()
test.head()
print(train.Pclass.value_counts(dropna=False))

print("--"*50)

print(train.Embarked.value_counts(dropna=False))

print("--"*50)

print(train.SibSp.value_counts(dropna=False))

print("--"*50)

print(train.Parch.value_counts(dropna=False))

print("--"*50)
test.isnull().sum().sort_values(ascending=False)
del train["Cabin"]

train["Age"].fillna(train.Age.mean(),inplace=True)
train.describe()
train.isnull().sum().sort_values(ascending=False)
#droping the 2 missing values of embarked

train.dropna(inplace=True)

# as we know  we only have 2 missing values in training data set.
del test["Cabin"]
test["Age"].fillna(test.Age.mean(),inplace=True)
test.describe()
test.isnull().sum().sort_values(ascending=False)
# As you can not remove test data we have to fill that missing value of fare

test.fillna(test.Fare.median(),inplace=True)

test.isnull().sum().sort_values(ascending=False)
survived = 'survived'

not_survived = 'not survived'

fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(10, 4))

women = train[train['Sex']=='female']

men = train[train['Sex']=='male']

ax = sns.distplot(women[women['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[0], kde =False)

ax = sns.distplot(women[women['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[0], kde =False)

ax.legend()

ax.set_title('Female')

ax = sns.distplot(men[men['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[1], kde = False)

ax = sns.distplot(men[men['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[1], kde = False)

ax.legend() 

ax.set_title('Male')
Y=train["Survived"]
del train["PassengerId"]

del test["PassengerId"]
del train["Survived"]
train.head()
train.shape
test.shape
test.head()
final=pd.concat([train,test],axis =0)
final.shape
final.head()
def One_hot_encoding(columns):

    final_df=final

    i=0

    for fields in columns:

        df1=pd.get_dummies(final[fields],drop_first=True)#applying get_dummies and droping first column which is created

        

        final.drop([fields],axis=1,inplace=True)

        if i==0:

            final_df=df1.copy()

        else:           

            final_df=pd.concat([final_df,df1],axis=1)

        i=i+1

       

        

    final_df=pd.concat([final,final_df],axis=1)

        

    return final_df
columns=["Sex","Embarked","Pclass","Parch"]
df_final = One_hot_encoding(columns)
df_final.head()
df_final.drop("Name",axis=1,inplace=True)

df_final.drop("Ticket",axis=1,inplace=True)

df_final.head()
from sklearn import preprocessing

# Get column names first

names = df_final.columns

# Create the Scaler object

scaler = preprocessing.StandardScaler()

# Fit your data on the scaler object

scaled_df = scaler.fit_transform(df_final)

df_final = pd.DataFrame(scaled_df, columns=names)
cols = []

count = 1

for column in df_final.columns:

    cols.append(count)

    count+=1

    continue

    

df_final.columns = cols
df_final.head()
df_train=df_final.iloc[:889,:]

df_test=df_final.iloc[889:,:]
X=df_train
df_test.shape
from sklearn.model_selection import train_test_split



X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
from sklearn.ensemble import RandomForestClassifier
model_rforest = RandomForestClassifier()
model_rforest.fit(X_train,Y_train)
print("R-Squared Value for Training Set: {:.3f}".format(model_rforest.score(X_train,Y_train)))

print("R-Squared Value for Test Set: {:.3f}".format(model_rforest.score(X_test,Y_test)))
#predictions_01 = model_rforest.predict(df_test)
#output_01 = pd.DataFrame({'PassengerId': test_id, 'Survived': predictions_01})

#output_01.to_csv('my_submission_05.csv', index=False)

#print("Your submission was successfully saved!")
from sklearn.tree import DecisionTreeClassifier
model_dec = DecisionTreeClassifier()

model_dec.fit(X_train, Y_train)
print("R-Squared Value for Training Set: {:.3f}".format(model_dec.score(X_train,Y_train)))

print("R-Squared Value for Test Set: {:.3f}".format(model_dec.score(X_test,Y_test)))
predictions_02 = model_dec.predict(df_test)
output_02 = pd.DataFrame({'PassengerId': test_id, 'Survived': predictions_02})

output_02.to_csv('my_submission_06.csv', index=False)

print("Your submission was successfully saved!")