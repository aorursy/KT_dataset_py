import pandas as pd

import numpy as np

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score

from sklearn.metrics import mean_absolute_error

from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestRegressor



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_path = '/kaggle/input/mini-flight-delay-prediction/flight_delays_train.csv'

test_path = '/kaggle/input/mini-flight-delay-prediction/flight_delays_test.csv'



df_train = pd.read_csv(train_path)

df_test = pd.read_csv(test_path)
df_train.info()
df_train.head()
df_test.head()
print(df_train.dep_delayed_15min.unique())

df_train.dep_delayed_15min.value_counts()
x = df_train['Month'].str.split('-')

df_train['New-Month']=x.apply(lambda x:x[1])



y = df_train['DayofMonth'].str.split('-')

df_train['New-DayOfMonth']=y.apply(lambda x:x[1])



z = df_train['DayOfWeek'].str.split('-')

df_train['New-DayOfWeek']=z.apply(lambda x:x[1])
x = df_test['Month'].str.split('-')

df_test['New-Month']=x.apply(lambda x:x[1])



y = df_test['DayofMonth'].str.split('-')

df_test['New-DayOfMonth']=y.apply(lambda x:x[1])



z = df_test['DayOfWeek'].str.split('-')

df_test['New-DayOfWeek']=z.apply(lambda x:x[1])
labelenconder = LabelEncoder()
df_train['UniqueCarrier_ENC'] = labelenconder.fit_transform(df_train['UniqueCarrier'])
df_test['UniqueCarrier_ENC'] = labelenconder.fit_transform(df_test['UniqueCarrier'])
features = ['New-Month', 'New-DayOfMonth', 'New-DayOfWeek', 'UniqueCarrier_ENC']



predictors = df_train[features]

target = df_train['dep_delayed_15min'].map({'Y': 1, 'N': 0}).values



x_train, x_val, y_train, y_val = train_test_split(predictors, target, test_size = 0.22, random_state = 0)
from sklearn.tree import DecisionTreeClassifier



decisiontree = DecisionTreeClassifier()

decisiontree.fit(x_train, y_train)



y_pred = decisiontree.predict(x_val)



acc_decisiontree = round(accuracy_score(y_pred, y_val) * 100, 2)



print(acc_decisiontree)
from sklearn.ensemble import RandomForestClassifier



randomforest = RandomForestClassifier()

randomforest.fit(x_train, y_train)



y_pred = randomforest.predict(x_val)



acc_randomforest = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_randomforest)
from sklearn.linear_model import LogisticRegression



logreg = LogisticRegression()

logreg.fit(x_train, y_train)



y_pred = logreg.predict(x_val)



acc_logreg = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_logreg)
from sklearn.svm import SVC



svc = SVC()

svc.fit(x_train, y_train)



y_pred = svc.predict(x_val)



acc_svc = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_svc)
from sklearn.ensemble import GradientBoostingClassifier



gbk = GradientBoostingClassifier()

gbk.fit(x_train, y_train)



y_pred = gbk.predict(x_val)



acc_gbk = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_gbk)
models = pd.DataFrame({

    'Model': ['Support Vector Machines', 'Logistic Regression', 

              'Random Forest', 

              'Decision Tree', 'Gradient Boosting Classifier'],

    'Score': [acc_svc, acc_logreg, 

              acc_randomforest, acc_decisiontree, acc_gbk]})

models.sort_values(by='Score', ascending=False)
df_test.head()
predictions = gbk.predict(df_test[features])



#set the output as a dataframe and convert to csv file named submission.csv

output = pd.DataFrame({ 'UniqueCarrier' : df_test['UniqueCarrier'], 'Origin': df_test['Origin'], 'Dest': df_test['Dest'], 'dep_delayed_15min': predictions })

output.to_csv('submission.csv', index=False)