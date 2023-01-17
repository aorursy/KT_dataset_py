# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt



# Tools

from sklearn.model_selection import train_test_split

from sklearn.model_selection import RandomizedSearchCV

from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import StandardScaler



# Model

from sklearn.ensemble import RandomForestClassifier



# Metrics

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report,confusion_matrix

import warnings

warnings.filterwarnings("ignore")



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session



# Configs

pd.options.display.float_format = '{:,.4f}'.format

sns.set(style="whitegrid")

plt.style.use('seaborn')

seed = 42

np.random.seed(seed)
def evalua(y_test,pred):

    print('Confusion matrix:\n',confusion_matrix(y_test,pred),'\n')

    print('Classification report:\n',classification_report(y_test,pred),'\n')

    print('Accuracy:',accuracy_score(y_test,pred),'\n')
# Load data

file_path = '/kaggle/input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv'

df = pd.read_csv(file_path)

print("DataSet = {:,d} rows and {} columns".format(df.shape[0], df.shape[1]))



print("\nAll Columns:\n=>", df.columns.tolist())



quantitative = [f for f in df.columns if df.dtypes[f] != 'object']

qualitative = [f for f in df.columns if df.dtypes[f] == 'object']

del qualitative[-1]



print("\nStrings Variables:\n=>",qualitative ,

      "\n\nNumerics Variables:\n=>", quantitative)



df.head(3)
# Transform data



df['Churn'].replace(to_replace='Yes', value=1, inplace=True)

df['Churn'].replace(to_replace='No',  value=0, inplace=True)



df = df.dropna()

df.isnull().sum()
# String to numeric (TCA)

df1=df.Churn

for i in qualitative:

    spd1=pd.DataFrame(df.groupby(df[i]).mean().Churn)

    auxiliar=spd1.to_dict()

    spd2=df[i].map(auxiliar.get('Churn'))

    df1=pd.DataFrame(df1).join(pd.DataFrame(spd2))
# Create numeric data frame

df2=df[quantitative]

df=pd.DataFrame(df1).join(pd.DataFrame(df2))
# Correlation analysis plot

plt.figure(figsize=(12, 6))

df.drop(['customerID'], axis=1, inplace=True)

corr = df.apply(lambda x: pd.factorize(x)[0]).corr()

ax = sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, 

                 linewidths=.2, cmap="YlGnBu")
corr['Churn'].sort_values(ascending=True)
# Samples

x = df.drop(['Churn'], axis=1)

y = df['Churn']

x_train, x_test, y_train, y_test = train_test_split(x, y.values, test_size=0.20, random_state=42)
# Search model

random_search = {

               'criterion': ['entropy', 'gini'],

               'max_depth': [2,3,4,5],

               'max_features': ['auto', 'sqrt'],

               'min_samples_leaf': [10,30,50],

               'n_estimators': [60,80,100]}



clf = RandomForestClassifier()

model = RandomizedSearchCV(estimator = clf, param_distributions = random_search, n_iter = 50, 

                               cv = 6, verbose= 1, random_state= 101, n_jobs = 10)

model.fit(x_train,y_train)



# Training 

print('Training:')

evalua(y_train,model.predict(x_train))
# the best estimator

print('Our best model:',model.best_estimator_)
# Test

print('Testing:')

pred=model.predict(x_test)

evalua(y_test,pred)
# Target for marketing

churn_scoring=model.predict_proba(x)

cuts=np.percentile(churn_scoring[:,1],[0,50,60,70,80,100])

df['Churn scoring']=pd.cut(np.array(churn_scoring[:,1]),cuts)

dfs=df.groupby('Churn scoring').mean()['Churn']

dfs.plot(kind='barh',title='Churn Rates',color=['r','b','y','g'],figsize=(20,10),grid=bool)
# Model performance

df['Scoring']=churn_scoring[:,1]

plt.figure(figsize=(15, 8))

plt.title("KDE for scoring (RFO)")

ax0 = sns.kdeplot(df[df['Churn'] == 0]['Scoring'].dropna(), color= 'navy', label= 'Churn: No')

ax1 = sns.kdeplot(df[df['Churn'] == 1]['Scoring'].dropna(), color= 'orange', label= 'Churn: Yes')