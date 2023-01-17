# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt

import seaborn as sns

import warnings

%matplotlib inline



warnings.filterwarnings('ignore')





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train=pd.read_csv("../input/train.csv")

test=pd.read_csv("../input/test.csv")

train.info()



train.describe()
trainint=train.select_dtypes(include='int64')

trainint

trainobj=train.select_dtypes(include='object')

obj_col=trainobj.columns

int_col=trainint.columns

int_col



for i in trainobj:

    print(i,": ",len(trainobj[i].unique()))

    

for i in trainint:

    print(i,": ",len(trainint[i].unique()))



train.head(30)
train.shape

ShapeSplit=train.shape[0]

ShapeSplit
all_data=pd.concat([train,test],axis=0)

all_data.shape

all_data['Embarked'].fillna(all_data['Embarked'].mode()[0], inplace = True)

all_data = pd.concat((train, test))



got = all_data.Name.str.split(',').str[1]

all_data['Title'] = pd.DataFrame(got).Name.str.split('\s+').str[1]

all_data.head()

ax = plt.subplot()

ax.set_ylabel('Average Age')

all_data.groupby('Title').mean()['Age'].plot(kind='bar', figsize=(13, 8), ax=ax)
AgeInput=pd.DataFrame(columns=["Id","Age","Titl"])



AgeInput['Age']=all_data['Age']

AgeInput['Titl']=all_data['Title']



AgeInput=AgeInput.groupby('Titl').mean()['Age']

AgeInput=pd.DataFrame(data=AgeInput)

AgeInput['Id']=range(0,len(AgeInput),1)

AgeInput["Title"]=AgeInput.index.values

AgeInput=AgeInput.set_index("Id")

AgeInput

all_data=all_data.set_index("PassengerId")
all_data.shape

for i in range(1, all_data.shape[0]+1):

    if np.isnan(all_data["Age"][i]):

        for j in range(0, AgeInput.shape[0]):

            if all_data.Title[i] == AgeInput["Title"][j]:

                all_data.Age[i] = AgeInput["Age"][j]

                
all_data.head()
all_data['Embarked'].fillna(all_data['Embarked'].mode()[0], inplace = True)



from sklearn.preprocessing import OneHotEncoder, LabelEncoder

label = LabelEncoder()

all_data['Sex_Code']      = label.fit_transform(all_data['Sex'])

all_data['Embarked_Code']      = label.fit_transform(all_data['Embarked'])



Emb_dum=pd.get_dummies(all_data["Embarked"])

all_data=pd.concat([all_data,Emb_dum],axis=1)


all_data=all_data.drop(["Cabin"],axis=1)

all_data=all_data.drop(["Name"],axis=1)

all_data=all_data.drop(["Ticket"],axis=1)

all_data=all_data.drop(["Sex"],axis=1)

#all_data=all_data.drop(["Ticket"],axis=1)

all_data=all_data.drop(["Embarked"],axis=1)



all_data.head()

all_data["FamilyS"]=all_data["Parch"]+all_data["SibSp"]

all_data.head()
#all_data=all_data.drop(["Sex"],axis=1)

all_data=all_data.drop(["Embarked_Code"],axis=1)

#all_data=all_data.drop(["Embarked"],axis=1)

#all_data=all_data.drop(["Title"],axis=1)

all_data.head()
all_data["FamilyS"]=all_data["Parch"]+all_data["SibSp"]

all_data['isAlone']=(all_data['FamilyS']==0)

all_data['isAlone']= label.fit_transform(all_data['isAlone'])



title_dum=pd.get_dummies(all_data["Title"])

all_data=pd.concat([all_data,title_dum],axis=1)

all_data.head()

#sns.distplot(all_data["Age"])

all_data["Age"].isna().count()

all_data.info()

#all_data[all_data['Age'] == isnan()].head(5)

all_data[all_data['Age'].isnull()]
train=all_data[:len(train)]

test=all_data[len(train):]
titanic_features = ['Age','Fare','Pclass','Sex_Code','C','Q','S','FamilyS',"isAlone","Capt.","Col.","Don.","Dona.","Dr.","Jonkheer.",

                    "Lady.","Major.","Master.","Miss.","Mlle.","Mme.","Mr.","Mrs.","Ms.","Rev.","Sir.","the"]

y= train["Survived"]

X = train[titanic_features]



from sklearn.model_selection import train_test_split



train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
from sklearn.linear_model import LogisticRegression



logmodel= LogisticRegression()

logmodel.fit(train_X,train_y)

predictions = logmodel.predict(val_X)
from sklearn.metrics import classification_report,confusion_matrix,mean_absolute_error,accuracy_score





print(classification_report(val_y,predictions))

print(confusion_matrix(val_y,predictions))

print(accuracy_score(val_y,predictions))

from sklearn.ensemble import RandomForestClassifier



rf = RandomForestClassifier()

rf.fit(train_X,train_y)

pred=rf.predict(val_X)

print(classification_report(val_y,pred))

print(confusion_matrix(val_y,pred))

print(accuracy_score(val_y,pred))





#print("%.4f" % rf.oob_score_)
from sklearn.neighbors import KNeighborsClassifier



knn= KNeighborsClassifier(n_neighbors=1)

knn.fit(train_X,train_y)

pred= knn.predict(val_X)



print(confusion_matrix(val_y,pred))

print(classification_report(val_y,pred))
test_X= test[titanic_features]

test_X[test_X['S'].isna()]
test_X.to_csv('xx.csv',index=False)


null_data = pd.DataFrame(test_X.isnull().sum().sort_values(ascending=False))[:50]



null_data.columns = ['Null Count']

null_data.index.name = 'Feature'

null_data.head()

test_X['Fare'][1044]=25

#test_X[test_X['Fare'].isna()]
predictions = logmodel.predict(test_X)
predictions=predictions.astype("int")

predictions
output = pd.DataFrame({'PassengerId': test.index,

                       'Survived': predictions.astype('int')})

output.to_csv('titanic_subm2.csv', index=False)
print(os.listdir())