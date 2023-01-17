import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
train=pd.read_csv("../input/titanic/train.csv")

train.head(10)
test=pd.read_csv("../input/titanic/test.csv")

test.head(10)
train.info()
test.info()
plt.figure(figsize=(15,10))

plt.title('heatgraph of null-values')

sns.heatmap(train.isnull(),yticklabels=False,cbar=False) # there mamy of NaN in 'Cabin' and 'age' in the train dataset
plt.figure(figsize=(15,10))

plt.title('heatgraph of null-values')

sns.heatmap(test.isnull(),yticklabels=False,cbar=False) # as same as train dataset it also have many NaN in the 'Cabin' and 'age' column 
train1=train.drop("Cabin",axis=1)



train1.head(10)
test1=test.drop("Cabin",axis=1)

test1.head(10)
train.describe()
test.describe()
train1.Age.replace([None],[29.699118],inplace=True)
plt.figure(figsize=(15,10))

plt.title('heatgraph of null-values')

sns.heatmap(train1.isnull(),yticklabels=False,cbar=False) # now we can see that there is no null-value
test1.Age.replace([None],[30.272590	],inplace=True)
plt.figure(figsize=(15,10))

plt.title('heatgraph of null-values')

sns.heatmap(test1.isnull(),yticklabels=False,cbar=False) #similarly here also
test1.Fare.replace([None],[35.627188],inplace=True)
corr=train1.corr()

plt.figure(figsize=(15,10))

sns.heatmap(corr,annot=True,annot_kws={"size":15}) # for train dataset

plt.title('heatgraph of correlation')
corr=test1.corr()

plt.figure(figsize=(15,10))

sns.heatmap(corr,annot=True,annot_kws={"size":15}) # for test dataset

plt.title('heatgraph of correlation')
from sklearn.linear_model import LogisticRegression
train1.head(5)
train1.Sex.replace(["male", "female"], [0, 1], inplace=True)
train1.head(5)
print("number of rows in train data -> ",len(train1))

print("number of rows in test data -> ",len(test1))
train2=train1
#now we choose the independent part

Train_IndepentVars=train1.drop("PassengerId",axis=1)

Train_IndepentVars=Train_IndepentVars.drop("Survived",axis=1)

Train_IndepentVars=Train_IndepentVars.drop("Name",axis=1)

Train_IndepentVars=Train_IndepentVars.drop("Ticket",axis=1)

Train_IndepentVars=Train_IndepentVars.drop("Embarked",axis=1)

Train_IndepentVars.head(5)

Train_IndepentVars=np.array(Train_IndepentVars)

print(Train_IndepentVars)
#now we choose the dependent/target part

Train_TargetVar = train2.values[:,1]

print(Train_TargetVar)
Train_TargetVar.dtype
logmodel=LogisticRegression()

Train_TargetVar=Train_TargetVar.astype('int')

logmodel.fit(Train_IndepentVars,Train_TargetVar)
predictions=logmodel.predict(Train_IndepentVars)

predictions,len(predictions)
from sklearn.metrics import classification_report
print(classification_report(Train_TargetVar,predictions))
from sklearn.metrics import confusion_matrix
print (pd.Series(Train_TargetVar).value_counts())

print(confusion_matrix(Train_TargetVar,predictions))

# confusion_matrix(Train_TargetVar,predictions)

confusion_df = pd.DataFrame(confusion_matrix(Train_TargetVar,predictions),

             columns=["Predicted Class " + str(class_name) for class_name in [0,1]],

             index = ["Class " + str(class_name) for class_name in [0,1]])



print(confusion_df)



#                 predicted     

# actual     No           YES

# NO         TN           FP   

# YES        FN           TP
print(logmodel.coef_)
print(logmodel.intercept_)
c=logmodel.score(Train_IndepentVars,Train_TargetVar)

c