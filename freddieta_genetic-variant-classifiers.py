import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Limit floats output to 3 decimal points
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Increase default figure and font sizes for easier viewing.
plt.rcParams['figure.figsize'] = (8, 6)
plt.rcParams['font.size'] = 14

"""
import os
print(os.listdir("../input"))
"""

dataframe = pd.read_csv('../GeneticVariantClassification/clinvar_conflicting.csv')
dataframe.head()
fig = plt.figure(figsize = (20, 20))
sns.countplot(x = 'CLASS', data = dataframe, hue = 'CHROM', palette='icefire')
fig = plt.figure(figsize = (15,15))
sns.heatmap(dataframe.isnull(), cmap = 'seismic', cbar = False)
dataframe.head(2)
considered = ['CHROM', 'POS', 'REF', 'ALT', 'AF_ESP', 'AF_EXAC', 'AF_TGP',
       'CLNDISDB', 'CLNDN', 'CLNHGVS', 'CLNVC','MC', 'ORIGIN', 'CLASS',
       'Allele', 'Consequence', 'IMPACT', 'SYMBOL', 'Feature_type',
       'Feature', 'BIOTYPE', 'STRAND','CADD_PHRED', 'CADD_RAW']

dataframe2 = dataframe[considered]
dataframe2 = dataframe2.dropna()
dataframe2['CHROM'] = dataframe2['CHROM'].astype(str)

print('Number of Null Values: ', dataframe2.isnull().sum().sum())
trimDown = []
for i in dataframe2.columns.values:
    if dataframe2[i].nunique() < 1000:
        trimDown.append(i)
        
print("Columns selected for training are: ", trimDown)
dataframe_final = dataframe2[trimDown]
dataframe_final.info()
dataframe_final['CHROM'] = dataframe_final['CHROM'].astype(str)
dataframe_final.info()
dataframe_final
from sklearn.feature_extraction import FeatureHasher
FH = FeatureHasher(n_features = 5, input_type = 'string')

hash1 = FH.fit_transform(dataframe_final['REF'])
hash1 = hash1.toarray()
hashedFeatures1 = pd.DataFrame(hash1)
nameList = {}
for i in hashedFeatures1.columns.values:
    nameList[i] = "REF"+str(i+1)

hashedFeatures1.rename(columns = nameList, inplace = True)
print("The Hashed REF table is something like this : \n",hashedFeatures1.head())
hash2 = FH.fit_transform(dataframe_final['ALT'])
hash2 = hash2.toarray()
hashedFeatures2 = pd.DataFrame(hash2)

nameList2 = {}
for i in hashedFeatures2.columns.values:
    nameList[i] = "ALT"+str(i+1)

hashedFeatures2.rename(columns = nameList, inplace = True)
print("The Hashed ALT table is something like this : \n",hashedFeatures2.head())
binaryFeature1 = pd.get_dummies(dataframe_final['CLNVC'])

print("While the One hot encoded matrix of CLNVC Columns is like this : \n")

binaryFeature1.head()
dataframe_final.columns
dataframe_final = dataframe_final.drop(columns = ['MC'], axis = 1)
dataframe_final.columns
hash0 = FH.fit_transform(dataframe_final['CHROM'])
hash0 = hash0.toarray()
hashedFeatures0 = pd.DataFrame(hash0)

nameList0 = {}
for i in hashedFeatures0.columns.values:
    nameList0[i] = "CHROM"+str(i+1)


hashedFeatures0.rename(columns = nameList0, inplace = True)
hashedFeatures0.head()
hash3 = FH.fit_transform(dataframe_final['Allele'])
hash3 = hash3.toarray()
hashedFeatures3 = pd.DataFrame(hash3)

nameList3 = {}
for i in hashedFeatures3.columns.values:
    nameList3[i] = "Allele"+str(i+1)


hashedFeatures3.rename(columns = nameList3, inplace = True)
hashedFeatures3.head()
hash4 = FH.fit_transform(dataframe_final['Consequence'])
hash4 = hash4.toarray()
hashedFeatures4 = pd.DataFrame(hash4)

nameList4 = {}
for i in hashedFeatures4.columns.values:
    nameList4[i] = "Consequence"+str(i+1)


hashedFeatures4.rename(columns = nameList4, inplace = True)
hashedFeatures4.head()
dataframe_final['IMPACT'].nunique()
dataframe_final = dataframe_final.drop(columns=['Feature_type'], axis = 1)
binaryFeature3 = pd.get_dummies(dataframe_final['IMPACT'])
binaryFeature3.head()
binaryFeature4 = pd.get_dummies(dataframe_final['BIOTYPE'], drop_first=True)
binaryFeature4.head()
binaryFeature5 = pd.get_dummies(dataframe_final['STRAND'], drop_first=True)
binaryFeature5.head()
dataframe3 = pd.concat([binaryFeature1, binaryFeature3, binaryFeature4, binaryFeature5, hashedFeatures1 , hashedFeatures2, hashedFeatures3, hashedFeatures4,hashedFeatures0, dataframe_final['CLASS']], axis = 1)
dataframe3 = dataframe3.dropna()
dataframe3.rename(columns={1 : "one", 16 : "sixteen"}, inplace = True)


print(dataframe3.columns.values)
dataframe3.head()
y = dataframe3['CLASS']
X = dataframe3.drop(columns = ['CLASS'], axis = 1)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
# from xgboost import XGBClassifier
# XGBoost in different directory

import sys
sys.path.append("/usr/local/lib/python3.7/site-packages")

from xgboost import XGBClassifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)
logReg = LogisticRegression()
logReg.fit(X_train, y_train)
pred_logReg = logReg.predict(X_test)

print('Classification Report: \n', classification_report(y_test, pred_logReg))
decisionTree = DecisionTreeClassifier(max_depth = 6)
decisionTree.fit(X_train, y_train)
pred_decisionTree = decisionTree.predict(X_test)

print('Classification Report: \n', classification_report(y_test, pred_decisionTree))
randomForest = RandomForestClassifier()
randomForest.fit(X_train, y_train)
pred_randomForest = randomForest.predict(X_test)

print( "Classification Report :\n ", classification_report(y_test, pred_randomForest))
gbc = GradientBoostingClassifier()
gbc.fit(X_train, y_train)
pred_gbc = gbc.predict(X_test)

print('Classification Report :\n ', classification_report(y_test, pred_gbc))
from collections import OrderedDict
important_features = {}
for i in zip(X.columns, logReg.coef_[0]):
    important_features[i[0]] = i[1]
important_final = OrderedDict(important_features)

dataframe_features = pd.DataFrame(important_final, index = range(1)).T
dataframe_features.rename(columns = {0: 'Importance_LogReg'}, inplace = True)

dataframe_features.plot(kind = 'bar', figsize = (20,5))
important_features2 = {}
for i in zip(X.columns, randomForest.feature_importances_):
    important_features2[i[0]] = i[1]
    
important_final2 = OrderedDict(important_features2)
print(important_final2)

dataframe_features2 = pd.DataFrame(important_final2, index = range(1)).T
dataframe_features2.rename(columns = {0: 'Importance_RandomForest'}, inplace = True)

dataframe_features2.plot(kind = 'bar', figsize = (15,5))
dataframe_compare = pd.concat([dataframe_features, dataframe_features2], axis = 1)
dataframe_compare.plot(kind = 'bar', figsize = (20,5))
# pip install keras
from keras.models import Sequential
from keras.layers import (Dense, Flatten, Dropout, BatchNormalization)
model = Sequential()

model.add(Dense(128 , input_dim = 38, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.33))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.33))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(1, activation = 'sigmoid'))
model.summary()
model.fit(X, y, batch_size=64, epochs = 20, verbose=1)
prediction = model.predict(X_test)
def finalPredictions(x):
    if x<0.5 : 
        return 0
    else:
        return 1
pred_deep = []
for i in prediction:
    pred_deep.append(finalPredictions(i))
print(classification_report(y_test, pred_deep))

