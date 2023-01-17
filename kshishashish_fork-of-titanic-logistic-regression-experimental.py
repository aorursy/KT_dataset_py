import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

trainset=pd.read_csv('../input/titanic/train.csv')
testset=pd.read_csv('../input/titanic/test.csv')
trainset['family_aboard']=trainset['Parch']+trainset['SibSp']
testset['family_aboard']=testset['Parch']+testset['SibSp']
trainset.drop(['Parch'],axis=1,inplace=True)
testset.drop(['Parch'],axis=1,inplace=True)
trainset.drop(['SibSp'],axis=1,inplace=True)
testset.drop(['SibSp'],axis=1,inplace=True)

trainset.head()
def name_set(trainset):
    new = trainset['Name'].str.split(",", n = 1, expand = True) 

    # making separate first name column from new data frame 
    trainset['Family Name']= new[0]
    new1=new[1].str.split(".",n=1,expand=True)
    trainset['Title']=new1[0]
    trainset['Name']= new1[1]
    return trainset
name_set(trainset)
name_set(testset)
def title_set (trainset):
    trainset.loc[trainset["Title"] == " Miss", "Title"] = 'Ms'
    trainset.loc[trainset["Title"] == " Mlle", "Title"] = 'Ms'
    trainset.loc[trainset["Title"] == " Mme", "Title"] = 'Mrs'

    trainset.loc[trainset["Title"] == " Mr", "Title"] = 'Mr'
    trainset.loc[trainset["Title"] == " Ms", "Title"] = 'Ms'
    trainset.loc[trainset["Title"] == " Mrs", "Title"] = 'Mrs'
    trainset.loc[trainset["Title"] == " Master", "Title"] = 'Master'

    trainset.loc[trainset["Title"] == " Dr", "Title"] = 'X'
    trainset.loc[trainset["Title"] == " Rev", "Title"] = 'X'
    trainset.loc[trainset["Title"] == " Major", "Title"] = 'X'
    trainset.loc[trainset["Title"] == " Col", "Title"] = 'X'
    trainset.loc[trainset["Title"] == " Lady", "Title"] = 'X'
    trainset.loc[trainset["Title"] == " Don", "Title"] = 'X'
    trainset.loc[trainset["Title"] == " Jonkheer", "Title"] = 'X'
    trainset.loc[trainset["Title"] == " the Countess", "Title"] = 'X'
    trainset.loc[trainset["Title"] == " Capt", "Title"] = 'X'
    trainset.loc[trainset["Title"] == " Sir", "Title"] = 'X'
    return trainset
title_set(trainset)
title_set(testset)
sns.pairplot(trainset)
print(trainset.head())
print(testset.head())
corr_matrix=trainset.corr()
plt.figure(figsize=(12,12))

sns.heatmap(corr_matrix,annot=True)
plt.show()
from sklearn.preprocessing import LabelEncoder

enc=LabelEncoder()


trainset['Name'] = enc.fit_transform(trainset['Name'])
testset['Name'] = enc.fit_transform(testset['Name'])

trainset['Family Name'] = enc.fit_transform(trainset['Family Name'])
testset['Family Name'] = enc.fit_transform(testset['Family Name'])

trainset['Title'] = enc.fit_transform(trainset['Title'])
testset['Title'] = enc.fit_transform(testset['Title'])

trainset['Sex'] = enc.fit_transform(trainset['Sex'])
testset['Sex'] = enc.fit_transform(testset['Sex'])

trainset['Cabin'] = enc.fit_transform(trainset['Cabin'].astype('str'))
testset['Cabin'] = enc.fit_transform(testset['Cabin'].astype('str'))

trainset['Embarked'] = enc.fit_transform(trainset['Embarked'].astype('str'))
testset['Embarked'] = enc.fit_transform(testset['Embarked'].astype('str'))

trainset['Ticket'] = enc.fit_transform(trainset['Ticket'].astype('category'))
testset['Ticket'] = enc.fit_transform(testset['Ticket'].astype('category'))
trainset["Age"].fillna(trainset.groupby("Title")["Age"].transform("median"), inplace= True)
testset["Age"].fillna(testset.groupby('Title')['Age'].transform("median"), inplace= True)
testset["Fare"].fillna(testset.groupby('Title')['Fare'].transform("median"), inplace= True)
y_train=trainset['Survived']

X_train=trainset
X_test=testset
X_train.drop(['Survived'],axis=1,inplace=True)
X_train.drop(['Ticket'],axis=1,inplace=True)
X_test.drop(['Ticket'],axis=1,inplace=True)

X_train.head()
X_test.head()
PID=X_test['PassengerId']
PID.head()
X_train.set_index(['PassengerId'],inplace = True)
X_test.set_index(['PassengerId'],inplace = True)

X_train.isnull().sum()
X_test.isnull().sum()
from sklearn.model_selection import train_test_split
X_train_1, X_CV, y_train_1, y_CV= train_test_split(X_train,y_train, test_size=0.1, random_state=100)
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()
X_train_1= sc.fit_transform(X_train_1)
X_CV= sc.transform(X_CV)
X_test=sc.transform(X_test)
X_train
(np.isfinite(X_train_1)==0).sum()
from sklearn.linear_model import LogisticRegressionCV

clf=LogisticRegressionCV(cv=5,
                        scoring='accuracy',
                        random_state=0,
                        n_jobs=-1,
                        verbose=3,
                        max_iter=300).fit(X_train_1,y_train_1)
y_predict=clf.predict(X_CV)
clf.score(X_CV,y_CV)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_CV,y_predict)
cm
cm_df = pd.DataFrame(cm,
                     index = ['yes','no'], 
                     columns = ['yes','no'])

plt.figure(figsize=(6,6))
sns.heatmap(cm_df, annot=True)

plt.title('Logistic Regression for Survival')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()


y_test=clf.predict(X_test)
y_test=y_test.reshape(-1,1)
y_test.shape
df=pd.DataFrame({'Survived':y_test[:,0]})
df.head()
df.shape
df['PassengerId']=PID
df=df[["PassengerId","Survived"]]
df.head()
df.to_csv('results.csv',index = False)
'''test_data['Survived'] = prediction
submission = pd.DataFrame(test['PassengerId'],test_data['Survived'])
submission.to_csv("Submission.csv")'''