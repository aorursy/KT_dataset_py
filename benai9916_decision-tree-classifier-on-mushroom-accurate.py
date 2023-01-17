import numpy as np 
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, cross_val_score
from sklearn.metrics import (accuracy_score, roc_auc_score, roc_curve, roc_auc_score ,recall_score, 
                             precision_score, confusion_matrix, classification_report, f1_score, auc)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# setting the size of the dataframe to disply
pd.set_option('max_columns', 25)
# loading the data

mushroom_df = pd.read_csv('../input/mushroom-classification/mushrooms.csv')

# printing the first five row of the dataframe
mushroom_df.head()
# no of rows and columns

mushroom_df.shape
# checking for any null values

mushroom_df.isnull().sum()
# check information about dataset 

mushroom_df.info()
# detail about dataset

mushroom_df.describe()

# function to print the value coutn go each columm

cols = mushroom_df.columns.to_list()
def value_count(cols):
    each_cols = mushroom_df[cols]
    for i in each_cols:
        print('Number of unique value in column "{}" is : {} -->  {} \n'.format(i.upper(), len(dict(each_cols[i].value_counts())) ,dict(each_cols[i].value_counts())))
        #print(dict(each_cols['class'].value_counts()))

     
value_count(cols)
# Convert categories to numbers using one hot encoder

x = pd.get_dummies(mushroom_df[mushroom_df.columns[1:]])
x.head()
# converting output value to numberic
labe_encode = LabelEncoder()
y = labe_encode.fit_transform(mushroom_df['class'])
# checking the no or columns after one hot encoding

x.shape[1]

# there are 117 columsn, excluding output column class
# spliting data into train and test

x_train,x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state=1, stratify = y)
print('x_train:', x_train.shape)
print('y_train:', y_train.shape)
print('x_test:', x_test.shape)
print('y_train:', y_test.shape)
# create an instance of decision tree

tree_classifier = DecisionTreeClassifier(criterion='gini', random_state=42)

# fit the data to model

tree_classifier.fit(x_train, y_train)
# make a prediction with testing data

y_predict = tree_classifier.predict(x_test)
y_predict_train = tree_classifier.predict(x_train)
# checking model accuracy

print('Model accuracy: ', accuracy_score(y_test, y_predict))
# cross validation to evaluate model

cross_metrix = cross_val_score(tree_classifier, x, y, scoring='accuracy')

print(cross_metrix)
print(cross_metrix.mean())
print(cross_metrix.std())
# confusion matrix

confusion_score = confusion_matrix(y_test, y_predict)
confusion_score
# plotting confusion matrix

sns.heatmap(confusion_score, annot=True, annot_kws={'size':16})
# slice confusion matrix into four pieces

TP = confusion_score[1, 1]
TN = confusion_score[0, 0]
FP = confusion_score[0, 1]
FN = confusion_score[1, 0]

# classification Erroe rate

print('Error rate: ', 1 - accuracy_score(y_test, y_predict))

# 0.0 shwos that there is no error, out model is perfect
# True positive (Recall or sensitivity)
print('True positive (Recall or sensitivity)', recall_score(y_test, y_predict))


# True Negative (sensitivity)
print('True positive (specificity)', TN/ (TN + FP))


# False Positive 
print('False positive (Recall or sensitivity)', FP/ (FP + TN))


# Precision
print('Precision', precision_score(y_test, y_predict))
# print classification report

print('classification report: \n', classification_report(y_test, y_predict))
#  predicted responses

fpr,tpr,thresholds=roc_curve(y_test,y_predict)

# calculate acu curv
roc_auc=auc(fpr,tpr)
# we pass y_test and y_pred_prob
# we do not use y_pred_class, because it will give incorrect results without generating an error
# roc_curve returns 3 objects fpr, tpr, thresholds
# fpr: false positive rate
# tpr: true positive rate

plt.figure(figsize=(10,10))
plt.title('Receiver Operating Characteristic')
plt.plot(fpr,tpr, color='red',label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],linestyle='--')
plt.axis('tight')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')

