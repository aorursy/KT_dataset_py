import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier as rf
import os
path = "/kaggle/input/caravan-insurance-challenge"
os.chdir(path)
os.listdir(path)
df = pd.read_csv("caravan-insurance-challenge.csv")
df.head()
df.shape
df.describe()
df.info()
pd.options.display.max_rows = 100
df.nunique()
df['ORIGIN'].value_counts()
df_train = df[df['ORIGIN'] == 'train']
df_test = df[df['ORIGIN'] == 'test']
df_train.shape
df_test.shape
df_train.drop(['ORIGIN'], axis = 'columns', inplace = True)
df_test.drop(['ORIGIN'], axis= 'columns', inplace = True)
df_train.shape
df_test.shape
y_train = df_train.pop('CARAVAN')
df_train.shape
y_test = df_test.pop('CARAVAN')
df_test.shape
cat_cols = df_train.columns[df_train.nunique() < 5]
cat_columns = cat_cols.tolist()
cat_columns
num_cols = df_train.columns[df_train.nunique() >= 5]
num_columns = num_cols.tolist()
num_columns
len(cat_columns)
len(num_columns)
import matplotlib.pyplot as plt    
import seaborn as sns
plt.figure(figsize=(15,18))
for i,j in enumerate(df_train[num_columns]):
   plt.subplot(9,7,i+1)
   sns.boxplot(y = df_train[j])
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([
                        ('abc', RobustScaler(),num_columns),
                        ('cde', OneHotEncoder(), cat_columns)    
                        ],
                        remainder = 'passthrough'
                        )
X_train = df_train
X_test = df_test
from sklearn.pipeline import Pipeline
pipe = Pipeline([
                ('ct', ct),
                ('rf', rf())
                ])
pipe.fit(X_train,y_train)
y_pred = pipe.predict(X_test)
df.nunique()
X_train.nunique()
X_test.nunique()
y_train.nunique()
y_test.nunique()
df.nunique() - X_train.nunique()
