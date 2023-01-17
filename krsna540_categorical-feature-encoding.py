# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # for data visualization

import seaborn as sns

from sklearn import preprocessing

from sklearn.preprocessing import MinMaxScaler

sns.set(style='darkgrid')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv("/kaggle/input/cat-in-the-dat/train.csv")

test_data = pd.read_csv("/kaggle/input/cat-in-the-dat/test.csv")

subm_data = pd.read_csv("/kaggle/input/cat-in-the-dat/sample_submission.csv")

train_data.head()
train_data.shape
#To CHeck 

train_data.isnull().sum().sum()
test_data.isnull().sum().sum()
#Id column is not necessary 

train_data=train_data.drop(['id'], axis = 1)

test_data=test_data.drop(['id'], axis = 1)
All_features=train_data.columns.tolist()

Numerical_features=['bin_0','bin_1','bin_2','ord_0','day','month','target']

categorical_features=list(set(All_features) - set(Numerical_features))

Numerical_features.remove('target')

print(categorical_features)

print(Numerical_features)
# to get description of numerical data in dataset

train_data.describe()
fig, ax = plt.subplots(2, 3, figsize=(15, 10))

for variable, subplot in zip(Numerical_features, ax.flatten()):

    sns.boxplot(train_data[variable], ax=subplot, color='black')

    for label in subplot.get_xticklabels():

        label.set_rotation(0)
for cname in categorical_features:

    print(cname+" : "+str(len(train_data[cname].unique())))
train_data.head()
binary = {'T': 1,'F': 0}

train_data["bin_3"]= [binary[item] for item in train_data["bin_3"]]

test_data["bin_3"]= [binary[item] for item in test_data["bin_3"]]

binary = {'Y': 1,'N': 0}

train_data["bin_4"]= [binary[item] for item in train_data["bin_4"]]

test_data["bin_4"]= [binary[item] for item in test_data["bin_4"]]

train_data.head()
nominal_col = ['nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4']

ordinal_col = ['ord_0', 'ord_1', 'ord_2', 'ord_3']
# keeping the ord_5 features aside as it has higher amount of cardinality

# Importing categorical options of pandas

from pandas.api.types import CategoricalDtype 



# seting the orders of our ordinal features

ord_1 = CategoricalDtype(categories=['Novice', 'Contributor','Expert', 

                                     'Master', 'Grandmaster'], ordered=True)

ord_2 = CategoricalDtype(categories=['Freezing', 'Cold', 'Warm', 'Hot',

                                     'Boiling Hot', 'Lava Hot'], ordered=True)

ord_3 = CategoricalDtype(categories=['a', 'b', 'c', 'd', 'e', 'f', 'g',

                                     'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o'], ordered=True)

ord_4 = CategoricalDtype(categories=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',

                                     'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',

                                     'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'], ordered=True)
# Transforming ordinal Features

train_data.ord_1 = train_data.ord_1.astype(ord_1)

train_data.ord_2 = train_data.ord_2.astype(ord_2)

train_data.ord_3 = train_data.ord_3.astype(ord_3)

train_data.ord_4 = train_data.ord_4.astype(ord_4)

train_data.ord_1 = train_data.ord_1.cat.codes

train_data.ord_2 = train_data.ord_2.cat.codes

train_data.ord_3 = train_data.ord_3.cat.codes

train_data.ord_4 = train_data.ord_4.cat.codes

train_data.head()



test_data.ord_1 = test_data.ord_1.astype(ord_1)

test_data.ord_2 = test_data.ord_2.astype(ord_2)

test_data.ord_3 = test_data.ord_3.astype(ord_3)

test_data.ord_4 = test_data.ord_4.astype(ord_4)

test_data.ord_1 = test_data.ord_1.cat.codes

test_data.ord_2 = test_data.ord_2.cat.codes

test_data.ord_3 = test_data.ord_3.cat.codes

test_data.ord_4 = test_data.ord_4.cat.codes
print(str(train_data.day.unique())+" "+str(train_data.month.unique()))
train_data['ord_5_ot'] = 'Others'

train_data.loc[train_data['ord_5'].isin(train_data['ord_5'].value_counts()[:25].sort_index().index), 'ord_5_ot'] = train_data['ord_5']



test_data['ord_5_ot'] = 'Others'

test_data.loc[test_data['ord_5'].isin(test_data['ord_5'].value_counts()[:25].sort_index().index), 'ord_5_ot'] = test_data['ord_5']
train_data.head()
plt.figure(figsize=(20,5))

sns.countplot(x='ord_5_ot', data=train_data,

                   order=list(train_data['ord_5_ot'].value_counts().sort_index().index) ,

                   color='black') 
ord_5_count = train_data['ord_5'].value_counts().reset_index()['ord_5'].values

plt.figure(figsize=(20,5))

g = sns.distplot(ord_5_count, bins= 50,color='black')

g.set_title("Frequency", fontsize=22)

g.set_xlabel("Total", fontsize=18)

g.set_ylabel("Density", fontsize=18)

plt.show()
### Credit of this features to: 

## https://www.kaggle.com/gogo827jz/catboost-baseline-with-feature-importance

import string

# Then encode 'ord_5' using ACSII values

# Add up the indices of two letters in string.ascii_letters

train_data['ord_5_new'] = train_data['ord_5_ot'].apply(lambda x:sum([(string.ascii_letters.find(letter)+1) for letter in x]))

test_data['ord_5_new'] = test_data['ord_5_ot'].apply(lambda x:sum([(string.ascii_letters.find(letter)+1) for letter in x]))

#train_data['ord_5_new']= train_data['ord_5_new'].astype('float64')

                                                    
train_data=train_data.drop(['ord_5_ot','ord_5'], axis = 1) 

test_data=test_data.drop(['ord_5_ot','ord_5'], axis = 1) 

train_data.head()
nominal_col = ['nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4']

fig, ax = plt.subplots(2, 3, figsize=(20, 10))

for variable, subplot in zip(nominal_col, ax.flatten()):

    sns.countplot(train_data[variable], ax=subplot, color='black')

    for label in subplot.get_xticklabels():

        label.set_rotation(0)
high_card_feats = ['nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9']

for x in high_card_feats:

    print(x+"-"+str(len(train_data[x].unique())))
for col in high_card_feats:

    train_data[f'hash_{col}'] = train_data[col].apply( lambda x: hash(str(x)) % 5000 )

    test_data[f'hash_{col}'] = test_data[col].apply( lambda x: hash(str(x)) % 5000 )
for col in high_card_feats:

    enc_nom_1 = (train_data.groupby(col).size()) / len(train_data)

    train_data[f'freq_{col}'] = train_data[col].apply(lambda x : enc_nom_1[x])

    #df_test[f'enc_{col}'] = df_test[col].apply(lambda x : enc_nom_1[x])
from sklearn.preprocessing import LabelEncoder



# Label Encoding

for f in ['nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9']:

    if train_data[f].dtype=='object' or test_data[f].dtype=='object': 

        lbl = preprocessing.LabelEncoder()

        lbl.fit(list(train_data[f].values) + list(test_data[f].values))

        train_data[f'le_{f}'] = lbl.transform(list(train_data[f].values))

        test_data[f'le_{f}'] = lbl.transform(list(test_data[f].values))   
plt.figure(figsize=(25,5))

sns.countplot(x='le_nom_5', data=train_data,

                   order=list(train_data['le_nom_5'].value_counts().sort_index().index) ,

                   color='black') 
train_data.columns
test_data.columns
train_data.drop([ 'hash_nom_5','hash_nom_6', 'hash_nom_7', 'hash_nom_8', 'hash_nom_9','freq_nom_5','freq_nom_6', 'freq_nom_7', 'freq_nom_8', 'freq_nom_9',

                'nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9'], axis=1, inplace=True)



test_data.drop([ 'hash_nom_5','hash_nom_6', 'hash_nom_7', 'hash_nom_8', 'hash_nom_9','nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9'], axis=1, inplace=True)
train_data.head()
# credits to eda-feat-engineering-encode-conquer kernal

test_data['target'] = 'test'

df = pd.concat([train_data, test_data], axis=0, sort=False )
print(f'Shape before dummy transformation: {df.shape}')

df = pd.get_dummies(df, columns=['nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4'],\

                          prefix=['nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4'], drop_first=True)

print(f'Shape after dummy transformation: {df.shape}')
train_data, test_data = df[df['target'] != 'test'], df[df['target'] == 'test'].drop('target', axis=1)

del df
train_data.head()
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

x = train_data.drop(["target"], axis=1)

y = train_data["target"]

y = y.astype(bool)

test_X = test_data.drop([],axis=1)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()

from sklearn.feature_selection import RFE

rfe = RFE(logreg, 15)             # running RFE with 10 variables as output

rfe = rfe.fit(x_train,y_train)

print(rfe.support_)           # Printing the boolean results

print(rfe.ranking_)           # Printing the ranking
col = x_train.columns[rfe.support_]

print(col)

UpdatedTrain_X=x_train[col]

print(UpdatedTrain_X.shape)

UpdatedTest_X=x_test[col]

print(UpdatedTest_X.shape)
import statsmodels.api as sm

df_train_rfe = sm.add_constant(UpdatedTrain_X)

log_mod_rfe = sm.GLM(y_train,df_train_rfe,family = sm.families.Binomial())

mod_res_rfe = log_mod_rfe.fit()

log_mod_rfe.fit().summary()
#Predicting the Test Data

UpdatedTestCoef_X = sm.add_constant(UpdatedTest_X[col])

predictions = mod_res_rfe.predict(UpdatedTestCoef_X)

Y_pred= predictions.map(lambda x: 1 if x > 0.5 else 0)

Y_pred.head()
from sklearn import metrics

print(metrics.confusion_matrix(y_test, Y_pred), "\n")

print("accuracy", metrics.accuracy_score(y_test, Y_pred))

print("precision", metrics.precision_score(y_test,Y_pred))

print("recall", metrics.recall_score(y_test,Y_pred))

confusion=confusion_matrix(y_test,Y_pred)    

TP = confusion[1,1] # true positive 

TN = confusion[0,0] # true negatives

FP = confusion[0,1] # false positives

FN = confusion[1,0] # false negatives

# Let's see the sensitivity of our logistic regression model

print("Sensitivity",TP / float(TP+FN))

# positive predictive value 

print ("Positive Predection Rate",TP / float(TP+FP))

# Negative predictive value

print ("Negative Predection rate",TN / float(TN+ FN))

# Calculate false postive rate - predicting churn when customer does not have churned

print("False positive Predection Rate",FP/ float(TN+FP))
k_range = [1,3,5,7,9,10,15]

scores=[]

for k in k_range:

    knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')

    knn.fit(UpdatedTrain_X, y_train)

    y_pred = knn.predict(UpdatedTest_X)

    scores.append(metrics.accuracy_score(y_test, y_pred))
plt.plot(k_range, scores)

plt.xlabel('Value of K for KNN')

plt.ylabel('Testing Accuracy')
knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')

knn.fit(UpdatedTrain_X, y_train)

y_pred = knn.predict(UpdatedTest_X)

print(classification_report(y_test, y_pred))
from sklearn import metrics

print("accuracy", metrics.accuracy_score(y_test, y_pred))

print("precision", metrics.precision_score(y_test,y_pred))

print("recall", metrics.recall_score(y_test,y_pred))

confusion=confusion_matrix(y_test,y_pred)    

TP = confusion[1,1] # true positive 

TN = confusion[0,0] # true negatives

FP = confusion[0,1] # false positives

FN = confusion[1,0] # false negatives

# Let's see the sensitivity of our logistic regression model

print("Sensitivity",TP / float(TP+FN))

# positive predictive value 

print ("Positive Predection Rate",TP / float(TP+FP))

# Negative predictive value

print ("Negative Predection rate",TN / float(TN+ FN))

# Calculate false postive rate - predicting churn when customer does not have churned

print("False positive Predection Rate",FP/ float(TN+FP))
sub_data=test_data[col]

y_pred = knn.predict(sub_data)
y_pred
subm_data['target'] = y_pred

subm_data.to_csv('submission.csv')