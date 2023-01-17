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

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import preprocessing



data = pd.read_csv(r"C:\Users\bibhu\Desktop\Datasets\adult.data",delimiter = ' *, *',header = None,engine = 'python')



data.columns=['age','workclass','fnlwgt','education','education_num','marital_status','occupation',

              'relationship','race','sex','capital_gain','capital_loss','hours_per_week','native_country','income']



data.isnull().sum()

data2=data.describe(include='all')

data.dtypes

# data cleaning

for i in data.columns:

    if data[i].dtype=="O":

        print(i,":",sum(data[i]=="?"))

    

for i in data.columns:

    if data[i].dtype=="O":

        data[i].replace("?",data.describe(include='all')[i][2],inplace=True)

    

for i in data.columns:

    if data[i].dtype=="O":

        print(i,":",sum(data[i]=="?"))

#Lable encoding

from sklearn.preprocessing import LabelEncoder

le1=LabelEncoder()

data.income=le1.fit_transform(data.income)



le2=LabelEncoder()

data.workclass=le2.fit_transform(data.workclass)



le3=LabelEncoder()

data.education=le3.fit_transform(data.education)



le4=LabelEncoder()

data.marital_status=le4.fit_transform(data.marital_status)



le5=LabelEncoder()

data.occupation=le5.fit_transform(data.occupation)



le6=LabelEncoder()

data.relationship=le6.fit_transform(data.relationship)



le7=LabelEncoder()

data.race=le7.fit_transform(data.race)



le8=LabelEncoder()

data.sex=le8.fit_transform(data.sex)



le9=LabelEncoder()

data.native_country=le9.fit_transform(data.native_country)

# plotting Heatmap

cor=data.corr()

plt.figure(figsize=(10,5))

sns.heatmap(cor,annot=True,cmap='coolwarm')

plt.show()

#data analysis





sns.countplot(data.income)

sns.countplot(data.sex,hue=data.income)



plt.figure(figsize=(10,5))

sns.distplot(data.age[data.income==0])

sns.distplot(data.age[data.income==1])

plt.legend(['lessthan 50k','greterthan 50k'])



plt.figure(figsize=(10,5))

sns.distplot(data.workclass[data.income==0])

sns.distplot(data.workclass[data.income==1])

plt.legend(['lessthan 50k','greterthan 50k'])







plt.figure(figsize=(10,5))

sns.distplot(data.education[data.income==0])

sns.distplot(data.education[data.income==1])

plt.legend(['lessthan 50k','greterthan 50k'])



plt.figure(figsize=(10,5))

sns.distplot(data.race[data.income==0])

sns.distplot(data.race[data.income==1])

plt.legend(['lessthan 50k','greterthan 50k'])



plt.figure(figsize=(10,5))

sns.distplot(data.capital_loss[data.income==0])

sns.distplot(data.capital_loss[data.income==1])

plt.legend(['lessthan 50k','greterthan 50k'])



# Other parameters have effect on the data set

#input output selection

ip=data.drop(['income','capital_loss','fnlwgt'],axis=1)

op=data['income']



from sklearn.model_selection import train_test_split

xtr,xts,ytr,yts=train_test_split(ip,op,test_size=0.1)



from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

sc.fit(xtr)



from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

sc.fit(xtr)

xtr=sc.transform(xtr)

xts=sc.transform(xts)



#  logistic regression



from sklearn.linear_model import LogisticRegression

alg=LogisticRegression()



#train the algorithm with the training data

alg.fit(xtr,ytr)

#checking accuracy of the model

#accuracy=alg.score(xts,yts)

#print(accuracy)



yp=alg.predict(xts)



from sklearn import metrics

cm=metrics.confusion_matrix(yts,yp)



accuracy=metrics.accuracy_score(yts,yp)

print(accuracy)



recall=metrics.recall_score(yts,yp)

print(recall)





#applying naive bayes

from sklearn.naive_bayes import GaussianNB



df = GaussianNB()

df.fit(xtr, ytr)

#checking accuracy

accuracy=df.score(xts,yts)

print(accuracy)



recall=metrics.recall_score(yts,yp)

print(recall)



# implementing KNN method

from sklearn.model_selection import train_test_split

xtr,xts,ytr,yts = train_test_split(ip,op,test_size=0.4,random_state=42, stratify=op)

from sklearn.neighbors import KNeighborsClassifier



neighbors=np.arange(2,9)

train_accuracy =np.empty(len(neighbors))

test_accuracy = np.empty(len(neighbors))



for i,k in enumerate(neighbors):

    knn = KNeighborsClassifier(n_neighbors=k)

    knn.fit(xtr, ytr)

    train_accuracy[i] = knn.score(xtr, ytr)

    test_accuracy[i] = knn.score(xts, yts) 

    

plt.title('k-NN Varying number of neighbors')

plt.plot(neighbors, test_accuracy, label='Testing Accuracy')

plt.plot(neighbors, train_accuracy, label='Training accuracy')

plt.legend()

plt.xlabel('Number of neighbors')

plt.ylabel('Accuracy')

plt.show()



knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(xtr,ytr)

yp=knn.predict(xts)



from sklearn import metrics

accuracy=metrics.accuracy_score(yts,yp)

print(accuracy)



recall=metrics.recall_score(yts,yp,average)

print(recall)

#SVM

from sklearn import svm



svc=svm.SVC(kernel='linear',C=100,gamma=0.01)

svc.fit(xtr,ytr)

yp = svc.predict(xts)



#NAIVE BAYESS

from sklearn.naive_bayes import GaussianNB

clf = GaussianNB()

clf.fit(xtr, ytr)

yp = clf.predict(xts)

#Decission Tree

from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier()



#train

clf = clf.fit(xtr,ytr)



#Predict

y_pred = clf.predict(xts)

#Random forest

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(criterion = 'entropy',random_state = 0)

clf.fit(xtr,ytr)

yp = clf.predict(xts)



from sklearn import metrics

cm=metrics.confusion_matrix(yts,yp)



accuracy=metrics.accuracy_score(yts,yp)

print(accuracy)



recall=metrics.recall_score(yts,yp)

print(recall)




