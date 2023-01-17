import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import plotly.express as px
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
sns.set()
warnings.filterwarnings('ignore')
df = pd.read_csv('../input/health-insurance-cross-sell-prediction/train.csv')
df.head()
df.drop('id',axis=1,inplace=True) #Dropping the ID column
df['Vehicle_Age'].value_counts()
cols_to_label=[]
for i in df.columns:
    if df[i].dtypes == 'O':
        cols_to_label.append(i)

cols_to_label
df[cols_to_label] = df[cols_to_label].apply(LabelEncoder().fit_transform) 
df.head()
sns.countplot(df['Response'])
sns.distplot(df['Age'])
sns.distplot(df['Vintage'])
sns.distplot(df['Annual_Premium'])
sns.scatterplot(x='Age', y='Annual_Premium', data=df)
sns.countplot(df['Gender'])
for i in df.columns:
    print(i)
    print(df[i].value_counts())
df.isnull().sum()
cols_to_scale = ['Age','Annual_Premium','Policy_Sales_Channel','Vintage']

scaler = StandardScaler().fit(df[cols_to_scale])
df[cols_to_scale] = pd.DataFrame(scaler.transform(df[cols_to_scale]), columns=cols_to_scale)
df.head()
sns.heatmap(df.corr() ,cmap='coolwarm', vmax=0.7, vmin=-0.7)
X_train, X_test, y_train, y_test = train_test_split(df.drop('Response', axis=1), df['Response'], test_size=0.2)
model1 = LogisticRegression(random_state = 365).fit(X_train, y_train)
preds = model1.predict(X_test)
print(f'The accuracy score of Logistic Regression model is: {accuracy_score(preds, y_test)}')
model2 = XGBClassifier().fit(X_train, y_train)
preds = model2.predict(X_test)
print(f'The accuracy score of XGBClassifier model is: {accuracy_score(preds, y_test)}')