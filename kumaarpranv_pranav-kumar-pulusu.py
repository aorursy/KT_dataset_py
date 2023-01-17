# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt 

import seaborn as sns

import warnings

warnings.filterwarnings("ignore")
traindf=pd.read_csv('/kaggle/input/predict-the-churn-for-customer-dataset/Train File.csv')

testdf=pd.read_csv('/kaggle/input/predict-the-churn-for-customer-dataset/Test File.csv')
testdf.head()
traindf.head()
traindf.isnull().sum()
testdf.isnull().sum()
for i in traindf.columns:

    print(i," ", type(i))

    print(traindf[i].unique())


trainsz=len(traindf)

testsz=len(testdf)

fulldf=pd.concat([traindf.drop('Churn',axis=1),testdf])

fulldf[['MonthlyCharges','TotalCharges']]=fulldf[['MonthlyCharges','TotalCharges']].apply(pd.to_numeric, errors='coerce')

len(fulldf)
fulldf['TotalCharges']=fulldf['TotalCharges'].fillna(fulldf['TotalCharges'].median())

fulldf['total_charges_to_tenure_ratio'] = fulldf['TotalCharges'] / fulldf['tenure']

fulldf['monthly_charges_diff'] = fulldf['MonthlyCharges'] - fulldf['total_charges_to_tenure_ratio']

plt.figure(figsize=(16, 8))

df=fulldf[:trainsz]

df['Churn']=traindf['Churn']

df.drop(['customerID'],

        axis=1, inplace=True)

corr = df.apply(lambda x: pd.factorize(x)[0]).corr()

ax = sns.heatmap(corr, xticklabels=corr.columns, annot = True,yticklabels=corr.columns, 

                 linewidths=.2, cmap="YlGnBu")
fulldf.isnull().sum()
fulldf=fulldf.replace([np.inf,-np.inf], 0)

df=df.replace([np.inf,-np.inf], 0)

#fulldf=fulldf.replace([np.inf,-np.inf], np.nan)

#df=df.replace([np.inf,-np.inf], np.nan)

#print(fulldf.isna().sum())

#infl=len(fulldf[fulldf.isna().any(axis=1)])

#trainsz-=infl

#fulldf[fulldf.isna().any(axis=1)]
print(trainsz)
#traindf.drop(fulldf[fulldf.isna().any(axis=1)].index, axis=0,inplace=True)



#df.dropna(inplace=True,axis=0)

#fulldf.dropna(inplace=True,axis=0)

#df.isna().sum()

from sklearn import preprocessing

df_ex=df.copy()



df_ex.columns
le = preprocessing.LabelEncoder()

le_map={}

cat_lst=['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',

       'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',

       'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',

       'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod','Churn']

for i in cat_lst:

    le.fit(df_ex[i])

    df_ex[i]=le.transform(df_ex[i])

    le_map[i] = dict(zip(le.classes_, le.transform(le.classes_)))
for i in df_ex.columns:

    if i in le_map:

        print(i,": ",le_map[i])

    else:

        print(i)

    sns.distplot(df_ex[i])

    plt.show()

  
#converting to normal

for i in ['MonthlyCharges','TotalCharges']:

    print(i)

    df_ex[i]=sns.distplot(df_ex.TotalCharges.rank(method='min').apply(lambda x: (x-1)/len(df_ex.TotalCharges)-1))

    df[i]=df_ex[i]

    plt.show()

#sns.distplot(np.log(df_ex['TotalCharges']))
df_ex['total_charges_to_tenure_ratio']=(df_ex['total_charges_to_tenure_ratio'])**(1/3)

sns.distplot(df_ex['total_charges_to_tenure_ratio'].rank(method='min').apply(lambda x: (x-1)/len(df_ex['total_charges_to_tenure_ratio'])-1))

df['total_charges_to_tenure_ratio']=df_ex['total_charges_to_tenure_ratio']

plt.show()
corr['Churn']

from sklearn.model_selection import train_test_split

from xgboost import XGBClassifier

import xgboost as xgb



params = {'random_state': 0, 'n_jobs': 4, 'n_estimators': 5000, 'max_depth': 8}



drop = [ 'PhoneService_No',

        'Dependents_No', 'PaperlessBilling_No']  #'gender_Female','Partner_No',

#df=fulldf.drop(["customerID","tenure","gender","MultipleLines","PaymentMethod"],axis=1) #

df=fulldf.drop(["customerID"],axis=1)

#fulldf = fulldf[['tenure','MonthlyCharges','TotalCharges','Contract','OnlineSecurity','InternetService']]

#for i in ['MonthlyCharges','TotalCharges','total_charges_to_tenure_ratio']:

#df['total_charges_to_tenure_ratio']=np.log(df['total_charges_to_tenure_ratio'])

df = pd.get_dummies(df)

df.drop(drop,axis=1,inplace=True)

df.head()




X=df[:trainsz]

y=pd.get_dummies(traindf['Churn'])['Yes']

X_test=df[trainsz:]



X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3)
#clf=XGBClassifier()

#clf=LogisticRegression()

#clf = RandomForestClassifier()
X_train.head()
model=XGBClassifier()

model.fit(X_train,y_train)



imp = model.feature_importances_

ind = np.argsort(imp)[::-1]



names = [X_train.columns[i] for i in ind]

plt.figure()

plt.bar(range(X_train.shape[1]), imp[ind])

plt.xticks(range(X_train.shape[1]), names, rotation=90)



plt.show()

print(ind[:10])

#labels=X_train.columns

#X_train_new = X_train.loc[:, [labels[i] for i in ind[:30]]]

#X_val_new = X_val.loc[:, [labels[i] for i in ind[:30]]]

#X_test_new = X_test.loc[:, [labels[i] for i in ind[:30]]]
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression



param_test = {

    

    'gamma': [0.5, 1, 1.5, 2, 5],

    'max_depth': [3, 4, 5]

  

}



clf = GridSearchCV(estimator = 

XGBClassifier(learning_rate =0.1,

              objective= 'binary:logistic',

              nthread=4,

              seed=27), 

              param_grid = param_test,

              scoring= 'accuracy',

              n_jobs=4,

              iid=False,

              verbose=10)



#clf=LogisticRegression()
clf.fit(X_train,y_train)

from sklearn.metrics import accuracy_score, average_precision_score

y_pred= clf.predict(X_val)

print(y_pred)

accuracy_score(y_val,y_pred)

y_test=clf.predict(X_test)
testdf=testdf[['customerID']]

testdf['Churn']=y_test
testdf.replace(0,'No',inplace=True)

testdf.replace(1,'Yes',inplace=True)
testdf.head()
testdf.to_csv('xgsol.csv',index=False)
from IPython.display import FileLink

FileLink(r'xgsol.csv')