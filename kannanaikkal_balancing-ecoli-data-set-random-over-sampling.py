#importing required libraries



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn import tree

from sklearn.neighbors import NearestCentroid

from matplotlib.colors import ListedColormap

from sklearn.metrics import confusion_matrix

from imblearn.over_sampling import SMOTE

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import recall_score

from imblearn.over_sampling import RandomOverSampler

from sklearn.model_selection import train_test_split

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier
data_file = pd.read_csv('../input/ecoli-uci-dataset/ecoli.csv')  #Loading Data



df1 = pd.DataFrame(data_file)                    #Data frame created and cleaned

df1_cleaned = df1.drop('SEQUENCE_NAME',axis=1)



df1_cleaned.SITE.replace(('cp','im','imS','imL','imU','om','omL','pp'),(1,2,3,4,5,6,7,8),inplace=True) #Data encoding

#print("Correlation",df1_cleaned.corr(method='pearson'))  #Correlation between each measures
dataset = df1_cleaned.values

X=dataset[:,0:7]

y=dataset[:,7]

clf1 = tree.DecisionTreeClassifier()

clf1 = clf1.fit(X,y)      #Fitting data set to decision tree

#tree.plot_tree(clf1)   #Plotting tree

#Percentage of distribution of protein sequence



print('There is cp',round(df1_cleaned['SITE'].value_counts()[1]/len(df1_cleaned) * 100,2),'% of the data set')

print('There is im',round(df1_cleaned['SITE'].value_counts()[2]/len(df1_cleaned) * 100,2),'% of the data set')

print('There is imS',round(df1_cleaned['SITE'].value_counts()[3]/len(df1_cleaned) * 100,2),'% of the data set')

print('There is imL',round(df1_cleaned['SITE'].value_counts()[4]/len(df1_cleaned) * 100,2),'% of the data set')

print('There is imU',round(df1_cleaned['SITE'].value_counts()[5]/len(df1_cleaned) * 100,2),'% of the data set')

print('There is om',round(df1_cleaned['SITE'].value_counts()[6]/len(df1_cleaned) * 100,2),'% of the data set')

print('There is omL',round(df1_cleaned['SITE'].value_counts()[7]/len(df1_cleaned) * 100,2),'% of the data set')

print('There is pp',round(df1_cleaned['SITE'].value_counts()[8]/len(df1_cleaned) * 100,2),'% of the data set')
#Confusion matrix construction

X_train, X_test, y_train, y_test = train_test_split(X, y)

model = XGBClassifier()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)



accuracy = accuracy_score(y_test, y_pred)

print("Accuracy: %.2f%%" % (accuracy * 100.0))

recall = recall_score(y_test, y_pred, average='micro')

print("Recall Score: %.2f%%" % (recall * 100.0))

conf_mat = confusion_matrix(y_true=y_test, y_pred=y_pred)

print('Confusion matrix:\n', conf_mat)
print("Number transactions X_train dataset: ", X_train.shape)

print("Number transactions y_train dataset: ", y_train.shape)

print("Number transactions X_test dataset: ", X_test.shape)

print("Number transactions y_test dataset: ", y_test.shape)
print("Before OverSampling, counts of label 'cp': {}".format(sum(y_train==1)))

print("Before OverSampling, counts of label 'im': {} \n".format(sum(y_train==2)))

print("Before OverSampling, counts of label 'imS': {}".format(sum(y_train==3)))

print("Before OverSampling, counts of label 'imL': {} \n".format(sum(y_train==4)))

print("Before OverSampling, counts of label 'imU': {}".format(sum(y_train==5)))

print("Before OverSampling, counts of label 'om': {} \n".format(sum(y_train==6)))

print("Before OverSampling, counts of label 'omL': {}".format(sum(y_train==7)))

print("Before OverSampling, counts of label 'pp': {} \n".format(sum(y_train==8)))
lr = LogisticRegression()

lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

rec = recall_score(y_test, y_pred,average='micro')

acc = accuracy_score(y_test,y_pred)

con = confusion_matrix(y_test, y_pred)



print("Recall Score",rec*100)

print("\nAccuracy Score", acc*100)

print("\n",con)
ros = RandomOverSampler(random_state=8)

X_res, y_res = ros.fit_resample(X_train, y_train)

p = y_res

df23 = pd.DataFrame(X_res,p)

df23.to_csv('ecoli_sampled.csv')
