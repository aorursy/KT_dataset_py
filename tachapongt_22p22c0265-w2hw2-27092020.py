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

import pylab as py

import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier, export_graphviz ,DecisionTreeRegressor

from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB, CategoricalNB, ComplementNB

from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn.preprocessing import MinMaxScaler

from sklearn.neural_network import MLPClassifier,MLPRegressor

from sklearn.base import clone
#url ='https://raw.githubusercontent.com/birdtbird/for_superAI_file/main/train%20(1).csv'

Titanic = pd.read_csv('../input/titanic/train.csv')#training data

Titanic.head()
DropList = ['Name','Ticket','Fare','Embarked','Fare']

trainingD =Titanic.drop(columns=DropList)

trainingD.head()
trainingD.isnull().values.any()
trainingD = trainingD.dropna()
trainingD.isnull().values.any()
sort="Survived"

trainingD_Id = trainingD[[sort,"PassengerId"]].groupby(sort).count()

print(trainingD_Id)

trainingD.Survived.isnull().values.any()
sort="Pclass"

trainingD_Id = trainingD[[sort,"PassengerId"]].groupby(sort).count()

print(trainingD_Id)

trainingD.Pclass.isnull().values.any()
sort="Age"

trainingD_Id = trainingD[[sort,"PassengerId"]].groupby(sort).count()

print(trainingD_Id)

trainingD.Age.isnull().values.any()
plot = pd.Series(Titanic.Age)

plot.plot(kind='bar')
sort="SibSp"

trainingD_Id = trainingD[[sort,"PassengerId"]].groupby(sort).count()

print(trainingD_Id)

trainingD.SibSp.isnull().values.any()
sort="Parch"

trainingD_Id = trainingD[[sort,"PassengerId"]].groupby(sort).count()

print(trainingD_Id)

trainingD.Parch.isnull().values.any()
sort="Cabin"

trainingD_Id = trainingD[[sort,"PassengerId"]].groupby(sort).count()

print(trainingD_Id)

trainingD.Cabin.isnull().values.any()
#url ='https://raw.githubusercontent.com/birdtbird/for_superAI_file/main/test.csv'

TestD = pd.read_csv('../input/titanic/test.csv')#Test data

TestD.head()
DTC = DecisionTreeClassifier()

DTR = DecisionTreeRegressor()



NB=GaussianNB()

MNB = MultinomialNB()

BNB = BernoulliNB()

Cat_NB = CategoricalNB()

ComNB = ComplementNB()



MLPC = MLPClassifier()

MLPR = MLPRegressor()



ModelList = [DTC, DTR, NB, MNB, BNB, Cat_NB, ComNB, MLPC, MLPR]

ModelName = ['DecisionTreeClassifier', 'DecisionTreeRegressor', 'GaussianNB',

             'MultinomialNB', 'BernoulliNB', 'CategoricalNB', 'ComplementNB',

             'MLPClassifier','MLPRegressor']
def TestModel(model,Y_Output,Y_Test,Name):

  Y_Output=model.predict(X_T)

  print(Name," Accuracy:",metrics.accuracy_score(Y_Test, Y_Output.astype(int)))

  #print(Name," Accuracy:",metrics.accuracy_score(Y_Test, Y_Output))

  #print(Y_Output)

  #print(Y_Test.to_numpy())
EXcepFeature = ['Sex','Cabin','Survived']



X = trainingD.drop(EXcepFeature,axis =1 )

Y = trainingD.Survived



# No 5-fold cross validation

X,X_T,Y,Y_T =train_test_split(X, Y, test_size=0.2, random_state=1)

#  pass()



for i in range(len(ModelList)):

  model = clone(ModelList[i])

  Name  = ModelName[i]

  model.fit(X,Y)

  Y_Output = model.predict(X_T)

  TestModel(model,Y_Output,Y_T,Name)
DTC = DecisionTreeClassifier()

DTR = DecisionTreeRegressor()



NB=GaussianNB()

MNB = MultinomialNB()

BNB = BernoulliNB()

ComNB = ComplementNB()



MLPC = MLPClassifier()

MLPR = MLPRegressor()



ModelList = [DTC, DTR, NB, MNB, BNB, ComNB, MLPC, MLPR]

ModelName = ['DecisionTreeClassifier', 'DecisionTreeRegressor', 'GaussianNB',

             'MultinomialNB', 'BernoulliNB','ComplementNB',

             'MLPClassifier','MLPRegressor']
def TestModelWithFiveFold(model,i,Name):

  test = listFold[i]

  X_test=test.drop( EXcepFeature,axis =1 )

  Y_Test=test.Survived

  Y_Output=model.predict(X_test)

  #Y_Output = np.where(Y_Output.astype(int) <0.8,Y_Output,0)

  precisionLO, recallLO, thresholdsLO = cal_precision_recall_f1(Y_Output,Y_Test)

  acc = metrics.accuracy_score(Y_Test, Y_Output.astype(int))



  precision.append(precisionLO)

  recall.append(recallLO)

  thresholds.append(thresholdsLO)

  Accuracy.append(acc)



  print(Name," Accuracy : ",acc)

  #print('precision : ',precisionLO,)

  #print('recall : ',recallLO)

  #print('thresholds : ',thresholdsLO)



  #print(Name," Accuracy:",metrics.accuracy_score(Y_Test, Y_Output))

  #print(Y_Output)

  #print(Y_Test.to_numpy())
def cal_precision_recall_f1(lab,prob):

  slab = lab[np.argsort(-prob)]

  rlist = []

  prlist = []

  relist = []

  f1list = []

  tplist = []

  fplist = []

  tnlist = []

  fnlist = []

  for i in range(len(slab)):

    s = slab[i]

    rlist.append(s)

    pr = sum(rlist)/len(rlist)

    prlist.append(pr)

    re = sum(rlist)/sum(slab)

    relist.append(re)

    f1 = 2*((pr*re)/(pr+re))

    f1list.append(f1)

  return prlist,relist,f1list

def cal_roc(lab,prob):

  slab = lab[np.argsort(-prob)]

  tpr_list = []

  fpr_list = []

  pre_list = []

  rec_list = []

  f1_list = []

  preres = np.zeros(len(lab))

  for i in range(len(slab)):

    s = slab[i]

    preres[i] = 1

    tp = sum((slab == preres) & (slab==1))

    tn = sum((slab == preres) & (slab==0))

    fp = sum((slab != preres) & (slab==0))

    fn = sum((slab != preres) & (slab==1))

    tprate = tp/(tp+fn)

    fprate = fp/(fp+tn)



    pre = tp/(tp+fp)

    rec = tp/(tp+fn)



    tpr_list.append(tprate)

    fpr_list.append(fprate)

    pre_list.append(pre)

    rec_list.append(rec)



    f1 = 2*((pre*rec)/(pre+rec))

    f1_list.append(f1)

  



  ap = np.mean(pre_list)

  auc = np.mean(tpr_list)



  return tpr_list,fpr_list,pre_list,rec_list,f1_list,ap,auc
import warnings

warnings.filterwarnings("ignore")



EXcepFeature = ['Sex','Cabin','Survived']



global precision, recall, thresholds, Accuracy

precision = []

recall = []

thresholds = []

Accuracy = []



#with 5-fold cross validation



a,_ =trainingD.shape

XFold_1=trainingD.loc[np. array([i for i in range(1,a+1)])%5==0,:]

XFold_2=trainingD.loc[np. array([i for i in range(1,a+1)])%5==1,:]

XFold_3=trainingD.loc[np. array([i for i in range(1,a+1)])%5==2,:]

XFold_4=trainingD.loc[np. array([i for i in range(1,a+1)])%5==3,:]

XFold_5=trainingD.loc[np. array([i for i in range(1,a+1)])%5==4,:]

listFold=[XFold_1,XFold_2,XFold_3,XFold_4,XFold_5]

##############################################################



for i in range(0,5):

  print('XFold_%s' % (i+1))

  for model,Name in zip(ModelList,ModelName):

    for j in range(0,5):

      if j==i:

        continue

      train = listFold[j]

      X = train.drop(EXcepFeature,axis =1 )

      Y = train.Survived

      model=model.fit(X,Y)

    TestModelWithFiveFold(model,i,Name)

    print('------------------------------------------------------------------------------------------------------------')

  print('oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo')
P=[]

G=[]

listA=[]

c=0

A=0

for c in range(8):

  print(ModelName[c])

  plt.plot(recall[c],precision[c])

  plt.plot(recall[c+1*8],precision[c+1*8])

  plt.plot(recall[c+2*8],precision[c+2*8])

  plt.plot(recall[c+3*8],precision[c+3*8])

  plt.plot(recall[c+4*8],precision[c+4*8])

  plt.show()

  

P=[]

G=[]

listA=[]

c=0

A=0

for c in range(8):

  print(ModelName[c])

  plt.plot(thresholds[c])

  plt.plot(thresholds[c+1*8])

  plt.plot(thresholds[c+2*8])

  plt.plot(thresholds[c+3*8])

  plt.plot(thresholds[c+4*8])

  plt.show()
C=0

A=0

for i,j in zip(recall,precision):

  plt.plot(i,j)

  plt.show()