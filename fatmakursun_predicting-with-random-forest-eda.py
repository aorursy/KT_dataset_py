# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
adult_data = pd.read_csv('../input/adult_train.csv')
adult_data.head()
adult_data.info()
sns.pairplot(adult_data, hue = 'Target', markers= ['o','s'])
plt.figure(figsize=(9,6))

sns.countplot(x='Target', hue='Sex', data=adult_data)
adult_data['Workclass'].value_counts()
plt.figure(figsize=(9,6))

sns.countplot(x='Target', hue='Education', data=adult_data ,palette='rainbow')
adult_data.groupby(['Country','Target'])[['Target']].count().head(20)
adult_data['Target']=[0 if i==' <=50K' else 1 for i in adult_data['Target']]
#assing X as a Dataframe of features and y as a Series of outcome variable



X = adult_data.drop('Target', axis = 1)

y = adult_data.Target
X.info()
# I will decide which categorical data variables I want to use in model 

for col_name in X.columns:

    if X[col_name].dtypes == 'object':

        unique_categorical = len(X[col_name].unique())

        print("Feature '{col_name}' has {unique_categorical} unique categories".format(col_name=col_name, unique_categorical = unique_categorical))
X['Country'] = ['United-States' if i == ' United-States' else 'Other' for i in X['Country']]

X['Country'].value_counts().sort_values(ascending = False)
X.columns
X.isnull().sum().sort_values(ascending=False)
#create a list of features to dummy

todummy_list = ['Workclass', 'Education','Martial_Status', 'Occupation', 'Relationship', 'Race', 'Sex', 'Country']
# Function to dummy all the categorical variables used for modeling



def dummy_df(df,todummy_list):

    for x in todummy_list:

        dummies = pd.get_dummies(df[x],prefix=x, dummy_na=False)

        df = df.drop(x,1)

        df = pd.concat([df,dummies],axis = 1)

    return df
X = dummy_df(X,todummy_list)
X.isnull().sum().sort_values(ascending = False)
# Handling missing data

from sklearn.preprocessing import Imputer

imp = Imputer(missing_values = 'NaN', strategy = 'median',axis = 0)

imp.fit(X)

X = pd.DataFrame(data = imp.transform(X),columns = X.columns)



X.isnull().sum().sort_values(ascending = False)
def find_outliers_tukey(x):

    q1 = np.percentile(x,25)

    q3 = np.percentile(x,75)

    iqr = q3 - q1

    floor = q1 - 1.5*iqr

    ceiling = q3 + 1.5*iqr

    outlier_indices = list(x.index[(x<floor) | (x>ceiling)])

    outlier_values = list(x[outlier_indices])

    

    return outlier_indices,outlier_values
tukey_indices,tukey_values = find_outliers_tukey(X['Age'])

np.sort(tukey_values)
from itertools import combinations

from sklearn.preprocessing import PolynomialFeatures



def add_interactions(df):

    combos = list(combinations(list(df.columns),2))

    colnames = list(df.columns) + ['_'.join(x) for  x in combos]

    

    poly = PolynomialFeatures(interaction_only=True, include_bias=False)

    df = poly.fit_transform(df)

    df = pd.DataFrame(df)

    df.columns = colnames

    

    noint_indicies = [i for i, x in enumerate(list((df == 0).all())) if x]

    df = df.drop(df.columns[noint_indicies], axis= 1)

    

    return df
X = add_interactions(X)
X.head() # as you can se there are may many features now.
from sklearn.decomposition import PCA



pca = PCA(n_components=10)

X_pca = pd.DataFrame(pca.fit_transform(X))
X_pca.head()
from sklearn.model_selection import train_test_split
Xtrain,Xtest,ytrain,ytest = train_test_split(X,y, test_size=0.1,random_state=101)
X.shape
import sklearn.feature_selection



select = sklearn.feature_selection.SelectKBest(k=20)

selected_features = select.fit(Xtrain,ytrain)

indices_selected = selected_features.get_support(indices=True)

colnames_selected = [X.columns[i] for i in indices_selected] 





Xtrain_selected = Xtrain[colnames_selected]

Xtest_selected = Xtest[colnames_selected]
colnames_selected
Xtrain_selected
from sklearn.ensemble import RandomForestClassifier 
rf= RandomForestClassifier(n_estimators=100)

rf.fit(Xtrain_selected,ytrain)
rf_prediction = rf.predict(Xtest_selected)
from sklearn.metrics import roc_auc_score

print(roc_auc_score(ytest, rf_prediction))
from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(ytest, rf_prediction))
print(classification_report(ytest, rf_prediction))