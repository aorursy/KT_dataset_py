import pandas as pd

import plotly.express as px

import numpy as np

import statsmodels.api as sm

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.preprocessing import scale, StandardScaler

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error, r2_score, roc_auc_score, roc_curve, classification_report

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.neural_network import MLPClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

import warnings



import lightgbm



warnings.filterwarnings("ignore")



%matplotlib inline



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Import Data 

data = pd.read_csv("/kaggle/input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv")
data.head()
data.shape
#Correlation Part.

corr = data.corr()

sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns , cmap='hot')
fig = px.box(data, x='sex', y='age', points="all")

fig.update_layout(title_text="Age Spread with Gender -> Male = 1 Female =0")

fig.show()



#I took from : https://www.kaggle.com/nayansakhiya/heart-fail-analysis-and-quick-prediction-95-rate
corr[abs(corr['DEATH_EVENT']) > 0.1]['DEATH_EVENT']
# The effect of death situation on sex

# Male = 1, Female = 0

# It shows Male mortality is higher than Female mortality



fig = px.pie(data, values='DEATH_EVENT',names='sex', title='GENDER',

      width=680, height=480)

fig.show()
# I will put one part independent variables called 'piece_of_data'

piece_of_data = data[['age', 'ejection_fraction', 'serum_creatinine', 'serum_sodium', 'time']]

#piece_of_data.head()
x = piece_of_data

y = data['DEATH_EVENT']
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=1,test_size=0.2)
from sklearn.svm import SVC

svc = SVC(random_state=0, kernel = 'rbf')

svc.fit(x_train, y_train)
y_pred = svc.predict(x_test)

print(y_pred)
# I reached 90% RATE right now.

con_matrix = confusion_matrix(y_test, y_pred)

svm_accuracy = accuracy_score(y_test, y_pred)

print(svm_accuracy)
from xgboost import XGBClassifier

xgb_model = XGBClassifier().fit(x_train, y_train)

y_pred = xgb_model.predict(x_test)

accuracy_score(y_test, y_pred)

xgb_before =accuracy_score(y_test, y_pred) 

xgb_before 
rf = RandomForestClassifier(n_estimators = 17, criterion='gini', random_state=0)

rf.fit(x_train,y_train)
y_pred = rf.predict(x_test)

conf_matr = confusion_matrix(y_test, y_pred)

rf_accuracy = accuracy_score(y_test, y_pred)

print(rf_accuracy)
from catboost import CatBoostClassifier

catb_model = CatBoostClassifier().fit(x_train,y_train)

y_pred = catb_model.predict(x_test)

accuracy_score(y_test, y_pred)*100