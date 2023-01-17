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
#imports 

import numpy as np

import pandas as pd

import os

import io 

import seaborn as sns

import matplotlib.pyplot as plt
train=pd.read_csv("/kaggle/input/titanic/train.csv",index_col='PassengerId')

test=pd.read_csv("/kaggle/input/titanic/test.csv",index_col='PassengerId')

type(train)
train.info()
test.info()
train.head()
test.head()
tt=pd.concat((train,test),axis=0)

tt.info()
tt.describe()
tt[tt.Embarked.isnull()]
tt.Embarked.value_counts()
pd.crosstab(tt[tt.Survived!=-888].Survived,tt[tt.Survived!=-888].Embarked)
tt.Embarked.fillna("C",inplace=True)
tt[tt.Embarked.isnull()]
tt[tt.Fare.isnull()]
median_fare=tt.loc[(tt.Pclass==3)&(tt.Embarked=="S"),"Fare"].median()

print(median_fare)
tt.Fare.fillna(median_fare,inplace=True)
tt[tt.Fare.isnull()]
tt[tt.Age.isnull()]
tt.Name
#function 

def GetTitle(name):

    first_name=name.split(",")[1]

    title=first_name.split(".")[0]

    title=title.strip().lower()

    return title

#use map function to apply the function on each name value 

tt.Name.map(lambda x:GetTitle(x)) #alternatively u can use tt.Name.map(GetTitle)
tt.Name.map(lambda x:GetTitle(x)).unique()
def GetTitle(name):

    title_groups={

        "mr":"Mr",

        "mrs":"Mrs",

        "miss":"Miss",

        "master":"Master",

        "don":"Sir",

        "rev":"Sir",

        "dr":"officer",

        'mme':"Mrs", 

        'ms':"Mr",

        'major':"officer", 

        'lady':"Lady",

        'sir':"Sir", 

        'mlle':"Miss",

        'col':"officer",

        'capt':"officer",

        'the countess':"Lady",

        'jonkheer':"Sir",

        'dona':"Lady"

    }

    first_name=name.split(",")[1]

    title=first_name.split(".")[0]

    title=title.strip().lower()

    return title_groups[title]
tt['Title']=tt.Name.map(lambda x: GetTitle(x))
tt.head()
#boxplot of age with titles 

tt[tt.Age.notnull()].boxplot('Age','Title');
#finally filling the age values 

title_age_median=tt.groupby('Title').Age.transform("median")

tt.Age.fillna(title_age_median,inplace=True)
tt.info()
logfare=np.log(tt.Fare+1)
#binning using qcut 

pd.qcut(tt.Fare,4,labels=['very low','low','high','very high'])#discretisation = converting continous variable to discrete ones 
tt["Fare_Bins"]=pd.qcut(tt.Fare,4,labels=['very low','low','high','very high'])
tt.head()
tt.info()
corr1_new_tt=tt.corr()

plt.figure(figsize=(5,20))

sns.heatmap(corr1_new_tt[['Age']].sort_values(by=['Age'],ascending=False).head(10),vmin=-1, cmap='bone_r', annot=True)
tt["AgeState"]=np.where(tt["Age"]>=18,"Adult","Child")
tt["FamSize"]=tt.Parch+tt.SibSp+1


tt["IsMale"]= np.where(tt["Sex"]=="male",1,0)
tt=pd.get_dummies(tt,columns=["Pclass","Title","Fare_Bins","Embarked","AgeState"])
tt.info()
tt.head()
tt.info()
tt.drop(["Cabin","Name","SibSp","Ticket","Parch","Sex"],axis=1,inplace=True)
# Splitting dataset into train

train_processed=tt[:len(train)]

# Splitting dataset into test

test_processed= tt[len(train):]
train_processed.info()
test_processed.info()
test_processed= test_processed.drop('Survived', axis=1)
test_processed.info()
#uppercase for matric and lower case for vectors 

X=train_processed.loc[:,"Age":].to_numpy().astype("float")

y=train_processed["Survived"].ravel()



print(X.shape,y.shape)
from sklearn.model_selection import train_test_split 

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)
#average survival in train and test 

print("Mean survival rate in train:{0:3f}".format(np.mean(y_train)))

print("Mean survival rate in test:{0:3f}".format(np.mean(y_test)))
from sklearn.linear_model import LogisticRegression

model_1=LogisticRegression(random_state=0)

model_1.fit(X_train,y_train)

print("Score for LOGISTIC REGRESSION model: {0:.2f}".format(model_1.score(X_test,y_test)))
#performance metrics

from sklearn.metrics import accuracy_score, precision_score,recall_score,confusion_matrix 

print("Accuracy Score:{0:.2f}".format(accuracy_score(y_test,model_1.predict(X_test))))

print("Confusion matrix:\n {0}".format(confusion_matrix(y_test,model_1.predict(X_test))))

print("Precision Score:{0:.2f}".format(precision_score(y_test,model_1.predict(X_test))))

print("Recall Score:{0:.2f}".format(recall_score(y_test,model_1.predict(X_test))))
model_1.coef_
from sklearn.preprocessing import MinMaxScaler,StandardScaler
scaler=MinMaxScaler()

X_train_scaled=scaler.fit_transform(X_train)
X_train_scaled[:,0].min(),X_train_scaled[:,0].max()
X_test_scaled=scaler.fit_transform(X_test)
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)

X_test_scaled=scaler.fit_transform(X_test)
X_train_scaled[:,0].mean()
model_2s=LogisticRegression(random_state=0)

model_2s.fit(X_train_scaled,y_train)

print("Score for LOGISTIC REGRESSION model SCALED: {0:.2f}".format(model_1.score(X_test,y_test)))

from sklearn.ensemble import RandomForestClassifier
model_3 = RandomForestClassifier(criterion='gini', n_estimators=1100,max_depth=5,min_samples_split=4,min_samples_leaf=5,max_features='auto',oob_score=True,random_state=0,n_jobs=-1,verbose=1)

model_3.fit(X_train, y_train)

print("Accuracy: {0:.2f}".format(model_3.score(X_test,y_test)))
model_4 = RandomForestClassifier(criterion='gini', n_estimators=1100,max_depth=5,min_samples_split=4,min_samples_leaf=5,max_features='auto',oob_score=True,random_state=0,n_jobs=-1,verbose=1)

model_4.fit(X_train_scaled, y_train)

print("Accuracy: {0:.2f}".format(model_4.score(X_test_scaled,y_test)))
print("Accuracy Score:{0:.2f}".format(accuracy_score(y_test,model_3.predict(X_test))))

print("Confusion matrix:\n {0}".format(confusion_matrix(y_test,model_3.predict(X_test))))

print("Precision Score:{0:.2f}".format(precision_score(y_test,model_3.predict(X_test))))

print("Recall Score:{0:.2f}".format(recall_score(y_test,model_3.predict(X_test))))
def get_submission_file(model,filename):

    test_X=test_processed.to_numpy().astype(float)

    predictions=model.predict(test_X)

    submission=pd.DataFrame({"PassengerId":test_processed.index,"Survived":predictions})

    submission.to_csv(filename,index=False)

get_submission_file(model_3,"Random_forest_sub.csv")
