# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv',delimiter=',', header=None, skiprows=1, names=['customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents',
       'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
       'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
       'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
       'PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'Churn'])
df.sample(10)
df = df.drop(['customerID'], axis = 1)
df.info()
df ["TotalCharges"]= pd.to_numeric(df["TotalCharges"], errors='coerce')
df.describe()
df['Churn'].value_counts()
import matplotlib.pyplot as plt
import seaborn as sns
count = ["Churn","PhoneService","MultipleLines","StreamingTV","Contract","PaymentMethod"]
for i in count:
    plt.figure( figsize=(14, 4) )
    sns.countplot(i, data=df)
    plt.xticks(rotation=90)
    plt.title('Distribution of Columns')
    plt.tight_layout()
plt.show()
def rescale(data, new_min=0, new_max=1):
    return (data - data.min()) / (data.max() - data.min()) * (new_max - new_min) + new_min


columns = ['tenure','MonthlyCharges','TotalCharges','SeniorCitizen','Churn']
obj_cols = df[columns].select_dtypes(include=['object'])
num_cols = df[columns].select_dtypes(exclude=['object'])
num_cols = rescale(num_cols)
num_cols

df_final = pd.concat([obj_cols, num_cols], axis=1,sort=False)
df_final.groupby('Churn').mean().plot.bar()
plt.show()
df["Monthly"] = df["TotalCharges"] / df ["tenure"]
drop_list1 = ["tenure","TotalCharges","MonthlyCharges"]
df = df.drop(drop_list1, axis = 1)
df
df['Churn'] = df['Churn'].astype("category")
df['Churn'] = df['Churn'].cat.codes
obj_cols1 = df.select_dtypes(include=['object'])
num_cols1 =df.select_dtypes(exclude=['object'])

obj = pd.get_dummies(obj_cols1, columns=obj_cols1.columns)
obj
df = pd.concat([obj, num_cols1], axis=1,sort=False)
cor = df.corr()
a = cor["Churn"].sort_values(ascending=False)
a
droplist2 = ['gender_Female','gender_Male','MultipleLines_No phone service', 'MultipleLines_No','MultipleLines_Yes','PhoneService_Yes','PhoneService_No']
df.drop(droplist2 ,axis = 1,inplace = True)
df
df.isna().sum()
df['Monthly'].fillna(value=df['Monthly'].mean(), inplace=True)
df.isna().sum()
cor = df.corr()
a = cor["Churn"].sort_values(ascending=False)
a
X = df.drop(['Churn'], axis = 1)
Y = df['Churn']
X = rescale(X)
X
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.20, random_state = 14)
x_train.isna().sum()
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
models = []
models.append(('Logistic Regression', LogisticRegression(max_iter=250)))
models.append(('Naive Bayes', GaussianNB()))
models.append(('Decision Tree (CART)',DecisionTreeClassifier())) 
models.append(('K-NN', KNeighborsClassifier()))
models.append(('SVM', SVC()))
models.append(('AdaBoostClassifier', AdaBoostClassifier()))
models.append(('BaggingClassifier', BaggingClassifier()))
models.append(('RandomForestClassifier', RandomForestClassifier()))
models.append(('XGB',xgb.XGBClassifier()))
for name, model in models:
    model = model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    from sklearn import metrics
    print("Model -> %s -> ACC: %%%.2f" % (name,metrics.accuracy_score(y_test, y_pred)*100))
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
grid={'max_iter': [100,125, 150,190],
         'C':[200, 400,800,900,950,1000,1010,1200], 'penalty':['l1', 'l2', 'elasticnet', 'none'], 'solver' :['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']}
logreg=LogisticRegression()
logreg_cv=GridSearchCV(logreg,grid,cv=10,  n_jobs=-1, verbose=1)
logreg_cv.fit(x_train,y_train)

print("tuned hpyerparameters :(best parameters) ",logreg_cv.best_params_)
print("accuracy :",logreg_cv.best_score_)
logregfinal=LogisticRegression(C= 1000, max_iter = 150, solver = 'lbfgs')
logregfinal.fit(x_train,y_train)
score = logregfinal.score(x_test, y_test)
prediction_test = logregfinal.predict(x_test)
print (metrics.accuracy_score(y_test, prediction_test))