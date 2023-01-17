# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/indian_liver_patient.csv')
df.head()
df.describe().T
#It is clear from the above described data that, Albumin_and_Globulin_Ratio columns has some missing rows.**
#As Albumin_and_Globulin_Ratio count is 579, where as other columns rows are 583.
print( "Albumin_and_Globulin_Ratio: ", df['Albumin_and_Globulin_Ratio'].isna().sum() )
print("-----------------------------------------------------")
print( "From entire dataset: ", df.isna().sum() )
sns.countplot(data=df, x='Dataset')
sns.countplot(data=df, x='Gender')
#As there are only four rows which are null in Albumin_and_Globulin_Ratio.
#We can remove those four rows from dataset, It impact on information loss is least
df.dropna(inplace=True)
#all the null valued rows are dropped now.
df.isna().sum()

#Get dummy integer values for the String Values
df['Gender'] = pd.get_dummies(df['Gender'])['Female']
#Female=1 and Male = 0
df.head()
# draw correlation heatmap
fig, ax = plt.subplots(figsize=(15,10)) # Sample figsize in inches
sns.heatmap(df.corr(), annot=True, linewidths=.5, ax=ax)

# draw scatter matrix of all features
pd.plotting.scatter_matrix(df, figsize=(15,10))
df = df
#Take all features in features variable
features = df.drop('Dataset',axis=1)
labels = df['Dataset']
features.head()
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
def pipeline(clfs, X_train, y_train, X_test, y_test):
    accuracys = []
    for i,clf in enumerate(clfs):
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        accuracys.append(accuracy_score(y_test, preds))
    return accuracys
#Create test train set
X_train, X_test, y_train, y_test = train_test_split(features,labels,test_size=0.25, random_state=45)

clfs = [DecisionTreeClassifier(), RandomForestClassifier(), MLPClassifier(), LogisticRegression()]
for i in range(3):
    print(pipeline(clfs, X_train, y_train, X_test, y_test))

