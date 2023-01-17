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


import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # to make visualizations 

import seaborn as sns # to make visualization

pd.set_option('display.max_columns', 21)



df = pd.read_csv("/kaggle/input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv",na_values=np.nan)

df.head()
# to see dimensions of data

df.shape
#Data types of data and no of rows in each feature

df.info()
df.gender = [1 if each == "Male" else 0 for each in df.gender]



columns_to_convert = ['Partner', 

                      'Dependents', 

                      'PhoneService', 

                      'MultipleLines',

                      'OnlineSecurity',

                      'OnlineBackup',

                      'DeviceProtection',

                      'TechSupport',

                      'StreamingTV',

                      'StreamingMovies',

                      'PaperlessBilling', 

                      'Churn']



for item in columns_to_convert:

    df[item] = [1 if each == "Yes" else 0 if each == "No" else -1 for each in df[item]]

    

    

df.head()
#Drop customer id column becasue it is unuseful in our project idea.

df.drop(columns=['customerID'],inplace=True)
df['TotalCharges']=pd.to_numeric(df['TotalCharges'],errors='coerce')
df.isna().sum()
df.dropna(inplace=True)
df.isna().sum().sum()
# see some stat for numerical features 



df.describe()
# see some stat for categorical features 



df.describe(include=['O'])

print(df.shape)

df.drop_duplicates(inplace=True)

print(df.shape)
df.head()
df['InternetService'].value_counts()
pd.pivot_table(data=df,index='InternetService',values=['TotalCharges'],aggfunc='mean').sort_values(by='TotalCharges',ascending=False)
plt.style.use('fivethirtyeight')

sns.countplot(df['InternetService'])
sns.countplot(df['gender'],hue=df['Churn'])


sns.countplot(df['PaymentMethod'])

plt.xticks(rotation=90)

df.Contract.value_counts()


sns.countplot(df['Contract'])
plt.hist((df.MonthlyCharges),bins=10)

plt.title('the distribution of monthly charge')

plt.show()
df = pd.get_dummies(data=df)

df.head()
X=df.drop(columns=['Churn'])

y=df['Churn']
df.corr()['Churn'].sort_values()
x=df[['tenure','Contract_Two year','InternetService_No','TotalCharges','Contract_Month-to-month','InternetService_Fiber optic','PaymentMethod_Electronic check']]
y=df['Churn']
from sklearn.preprocessing import MinMaxScaler

stander=MinMaxScaler()

x=stander.fit_transform(x)
#Split data into Train and Test 

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,random_state =1)
# %%KNN Classification

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 8) #set K neighbor as 8

knn.fit(x_train,y_train)

predicted_y = knn.predict(x_test)

accuracy_knn=knn.score(x_test,y_test)

print("KNN accuracy according to K=8 is :",accuracy_knn)
# %%Logistic regression classification

from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression()

lr_model.fit(x_train,y_train)

accuracy_lr = lr_model.score(x_test,y_test)

print("Logistic Regression accuracy is :",accuracy_lr)

# %%SVM Classification

from sklearn.svm import SVC

svc_model = SVC(random_state = 42)

svc_model.fit(x_train,y_train)

accuracy_svc = svc_model.score(x_test,y_test)

print("SVM accuracy is :",accuracy_svc)
# %%Decision Tree Classification

from sklearn.tree import DecisionTreeClassifier

dt_model = DecisionTreeClassifier()

dt_model.fit(x_train,y_train)

accuracy_dt = dt_model.score(x_test,y_test)

print("Decision Tree accuracy is :",accuracy_dt)
# %%Random Forest Classification

from sklearn.ensemble import RandomForestClassifier

rf_model_initial = RandomForestClassifier(n_estimators = 2, random_state = 1)

rf_model_initial.fit(x_train,y_train)

print("Random Forest accuracy for 5 trees is :",rf_model_initial.score(x_test,y_test))


from sklearn.metrics import recall_score, confusion_matrix, precision_score, f1_score, accuracy_score, classification_report



#for Logistic Regression

cm_lr = confusion_matrix(y_test,lr_model.predict(x_test))



# %% confusion matrix visualization

import seaborn as sns

f, ax = plt.subplots(figsize = (5,5))

sns.heatmap(cm_lr, annot = True, linewidths = 0.5, color = "red", fmt = ".0f", ax=ax)

plt.xlabel("y_predicted")

plt.ylabel("y_true")

plt.title("Confusion Matrix of Logistic Regression")

plt.show()
from sklearn.metrics import classification_report
report = classification_report(y_test, lr_model.predict(x_test))

print(report)
