import numpy as np 

import pandas as pd 

import seaborn as sns

from sklearn import decomposition, preprocessing, svm 

import matplotlib.pyplot as plt

from scipy import stats

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import LinearSVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

sns.set(color_codes=True)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

print("Setup Complete")
heart_attack_data=pd.read_csv('../input/heart-attack-prediction/data.csv')
heart_attack_data.head()
heart_attack_data.describe()
heart_attack_data.shape
#check how writed columns name. Are there some spaces between words. 

heart_attack_data.columns
#replace column name with surplus space to correct

heart_attack_data.rename({'num       ':'num'},axis=1,inplace=True)
heart_attack_data.columns
#check are missing data in DF

heart_attack_data.isnull().sum()
#delete rows where all data missing

heart_attack_data.dropna(axis=0,how='all')
heart_attack_data.replace('?', np.nan, inplace=True)
#replace Nan median 

median=heart_attack_data['chol'].median()

heart_attack_data['chol'].fillna(median,inplace=True)

median=heart_attack_data['slope'].median()

heart_attack_data['slope'].fillna(median,inplace=True)

median=heart_attack_data['ca'].median()

heart_attack_data['ca'].fillna(median,inplace=True)

median=heart_attack_data['thal'].median()

heart_attack_data['thal'].fillna(median,inplace=True)
heart_attack_data.head()
heart_attack_data = heart_attack_data.reset_index()
def clean_dataset(df):

    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"

    df.dropna(inplace=True)

    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)

    return df[indices_to_keep].astype(np.float64)

clean_dataset(heart_attack_data)
#visualization of the relationship between cholesterol and pressure

sns.jointplot(x=heart_attack_data['chol'], y=heart_attack_data['trestbps'], data=heart_attack_data, kind="kde")

#Build predictive model using split data method

features=['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','slope','ca','thal']

label=['num']

X=heart_attack_data[features]

y=heart_attack_data[label]

X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=1)

gnb = GaussianNB()

pred_model = gnb.fit(X_train, y_train)

target_pred = gnb.predict(X_test)

accuracy_score(y_test, target_pred, normalize = True)
#This model has not good accuracy - only 53%. Let's try SVC algorithm.
svc_model = LinearSVC(random_state=0)

pred = svc_model.fit(X_train, y_train).predict(X_test)
accuracy_score(y_test, pred, normalize = True)
#SVC has better accuracy, but still not so good. Let's try 
neigh = KNeighborsClassifier(n_neighbors=3)

neigh.fit(X_train, y_train)

pred = neigh.predict(X_test)

accuracy_score(y_test, pred)
#I built 3 algorithms but all of them have loe accuracy - no more 64%. In my opinion, the reason is the small amount of data. So, to build more accurate model, need to add additional data to our DataFrame.