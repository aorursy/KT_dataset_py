#!pip install -U scikit-learn

import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

%matplotlib inline
df=pd.read_csv('/kaggle/input/pima-indians-diabetes-database/diabetes.csv')
df
df.describe()
df.hist(bins=20,figsize=(10,15))
fig, axes = plt.subplots(nrows=2,ncols=4, figsize=(20,15), sharey=True)

df.boxplot(by='Outcome', return_type='axes', ax=axes)
corr_matrix=df.corr()

print(corr_matrix['Outcome'])
from sklearn.model_selection import StratifiedShuffleSplit



split=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)

for train_index,test_index in split.split(df,df['Outcome']):

    strat_train_set=df.iloc[train_index]

    strat_test_set=df.iloc[test_index]



print("test set\n",strat_test_set['Outcome'].value_counts()/len(strat_test_set) )

print("train set\n",strat_train_set['Outcome'].value_counts()/len(strat_train_set) )

df['Outcome'].value_counts()/len(df)  
fig, axes = plt.subplots(nrows=2,ncols=4, figsize=(20,15), sharey=True)

strat_train_set.boxplot(by='Outcome', return_type='axes', ax=axes)
Q1 = strat_train_set.quantile(0.25)

Q3 = strat_train_set.quantile(0.75)

IQR = Q3[:-1] - Q1[:-1]

#print("IQR",IQR)

new_train = strat_train_set[~((strat_train_set < (Q1[:-1] - 1.5 * IQR)) |(strat_train_set > (Q3[:-1] + 1.5 * IQR))).any(axis=1)]

new_train.shape

Q1[:-1]
fig, axes = plt.subplots(nrows=2,ncols=4, figsize=(20,15), sharey=True)

new_train.boxplot(by='Outcome', return_type='axes', ax=axes)
from sklearn.model_selection import train_test_split

from sklearn.utils import resample



# separate minority and majority classes

negative = new_train[new_train.Outcome==0]

positive = new_train[new_train.Outcome==1]

# upsample minority

pos_upsampled = resample(positive,replace=True, # sample with replacement

                         n_samples=len(negative), # match number in majority class

                         random_state=27) # reproducible results

# combine majority and upsampled minority

upsampled = pd.concat([negative, pos_upsampled])



# check new class counts

upsampled.Outcome.value_counts()





upsampled.reset_index(inplace=True)

upsampled.drop('index',axis=1,inplace=True)

scaled_features=upsampled.drop("Outcome",axis=1)

labels=upsampled["Outcome"].copy()
# from sklearn.preprocessing import StandardScaler

# scaler = StandardScaler()

# scaled_features=scaler.fit_transform(features)



new_train

scaled_features

scaled_features
from sklearn.neighbors import KNeighborsClassifier



neigh = KNeighborsClassifier(n_neighbors=5)

neigh.fit(scaled_features,labels)

from sklearn.metrics import accuracy_score



def accu_score(c,features,label):

      y_pred=c.predict(features)

      accuracy=accuracy_score(label, y_pred)

      print('accuracy:',accuracy)

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import cross_val_predict

from sklearn.metrics import classification_report

from sklearn.metrics import precision_recall_curve

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score





def con_matrix(c,features,label):

    y_train_pred = cross_val_predict(c,features,label,cv=5)

    print("confusion_matrix\n",confusion_matrix(label, y_train_pred))

    target_names = ['class 0', 'class 1']

    precision=precision_score(label, y_train_pred)

    print('precision',precision)

    recall=recall_score(label, y_train_pred)

    print('recall', recall)

    precisions,recalls,thresholds=precision_recall_curve(label, y_train_pred)

    print('precisions',precisions)

    print('recalls',recalls)

    print('thresholds',thresholds)







accu_score(neigh,scaled_features,labels)

con_matrix(neigh,scaled_features,labels)

from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(random_state=0)

clf.fit(scaled_features,labels)

accu_score(clf,scaled_features,labels)

con_matrix(clf,scaled_features,labels)





from sklearn.pipeline import make_pipeline

from sklearn.svm import SVC



svm=SVC(gamma='auto')

svm.fit(scaled_features,labels)



accu_score(svm,scaled_features,labels)

con_matrix(svm,scaled_features,labels)
strat_test_set.reset_index(inplace=True)

strat_test_set.drop('index',axis=1,inplace=True)

test_features=strat_test_set.drop("Outcome",axis=1)

test_labels=strat_test_set["Outcome"].copy()

strat_test_set
test_features
print('KNN\n')

accu_score(neigh,test_features,test_labels)

print('Deciion Tree\n')

accu_score(clf,test_features,test_labels)

print('SVM\n')

accu_score(svm,test_features,test_labels)