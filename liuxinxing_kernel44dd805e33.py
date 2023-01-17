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
%matplotlib inline
import matplotlib.pyplot as plt
traindata=pd.read_csv("/kaggle/input/titanic/train.csv")
traindata.head()
traindata.info()
traindata.describe()
#各变量和存活的函数
def visualplot(feature,stacked=False):
    if stacked:
        ratio=traindata.groupby([feature,'Survived'])['Survived'].count().unstack().plot(kind='bar',stacked='True')
    else:
        ratio=(traindata.groupby([feature]).sum()/traindata.groupby([feature]).count())['Survived']
        ratio.plot(kind='bar')
    plt.title(feature+' and Survival') 
print("女性生存率:", traindata["Survived"][traindata["Sex"] == "female"].value_counts(normalize=True)[1])
print("男性生存率:", traindata["Survived"][traindata["Sex"] == "male"].value_counts(normalize=True)[1])
visualplot('Sex',('male','female'))
visualplot('Pclass',(1,2,3))
meanage=traindata['Age'].mean()
traindata['Age'].fillna(meanage,inplace=True)
bins=np.arange(0,90,10)
traindata['age_cut']=pd.cut(traindata.Age,bins)
visualplot('age_cut',True)
visualplot('SibSp',True)
visualplot('Parch',True)
visualplot('Embarked',True)
plt.plot(traindata['Fare'])
centerfare=traindata.Fare[traindata.Fare<=182]
farebins=np.arange(0,190,10) 
traindata['cats']=pd.cut(centerfare,farebins)
visualplot('cats',True)
#构建清洗后使用的训练集
traindf=traindata[['Pclass','Sex','Age','SibSp','Parch','Fare','Survived']]
agerange=max(traindf['Age'])-min(traindf['Age'])
frange=max(traindf['Fare'])-min(traindf['Fare'])
traindf.loc[traindf['Fare']>182,'Fare']=np.mean(traindf.Fare)
traindf.loc[traindf['Sex']=='female','Sex']=0
traindf.loc[traindf['Sex']=='male','Sex']=1
traindf['Age']=(traindf['Age']-min(traindf['Age']))/agerange
traindf['Fare']=(traindf['Fare']-min(traindf['Fare']))/frange  
traindf
traindf=traindf.dropna()
traindf
X_train=traindf.iloc[:,:6]
y_train=traindf.iloc[:,-1]
#SVM
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.svm import SVC
CScale = [i for i in range(10,101,10)];
gammaScale = [i/10 for i in range(1,11)];
cv_scores = 0
for i in CScale:
    for j in gammaScale:
        model = SVC(kernel = 'rbf', C = i,gamma=j)
        scores = cross_val_score(model,X_train, y_train,cv =5,scoring = 'accuracy')
        if scores.mean()>cv_scores:
            cv_scores = scores.mean()
            savei = i
            savej = j*100
print(cv_scores)
CScale = [i for i in range(savei-5,savei+5)];
gammaScale = [i/100+0.01 for i in range(int(savej)-5,int(savej)+5)];
cv_scores = 0
for i in CScale:
    for j in gammaScale:
        model = SVC(kernel = 'rbf', C = i,gamma=j)
        scores = cross_val_score(model,X_train, y_train,cv =5,scoring = 'accuracy')
        if scores.mean()>cv_scores:
            cv_scores = scores.mean()
            savei = i
            savej = j
model = SVC(kernel = 'rbf', C=savei,gamma=savej)
print(model.fit(X_train, y_train))
print(cv_scores)
testdata=pd.read_csv("/kaggle/input/titanic/test.csv")
testdata
testdf=testdata[['Pclass','Sex','Age','SibSp','Parch','Fare']]
meanage=traindata['Age'].mean()
testdata['Age'].fillna(meanage,inplace=True)
testdata['Fare'].fillna(np.mean(traindf.Fare),inplace=True)
testdf=testdata[['Pclass','Sex','Age','SibSp','Parch','Fare']]
testdf.loc[testdf['Sex']=='female','Sex']=0
testdf.loc[testdf['Sex']=='male','Sex']=1
testdf['Age']=(testdf['Age']-min(traindf['Age']))/agerange
testdf.loc[testdf['Fare']>182,'Fare']=np.mean(traindf.Fare)
testdf['Fare']=(testdf['Fare']-min(traindf['Fare']))/frange  
X_test=testdf.dropna()
X_test
model = SVC(kernel = 'rbf', C=savei,gamma=savej)
model.fit(X_train, y_train)
pre = model.predict(X_test)
print(pre)
gender_submission=pd.read_csv("/kaggle/input/titanic/gender_submission.csv")
gender_submission
gender_submission["Survived"]=pre
gender_submission.to_csv("submission.csv", index=False)
