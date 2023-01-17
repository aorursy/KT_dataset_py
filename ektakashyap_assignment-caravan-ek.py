#Clear memory

%reset -f



# Calling Libraries



# Warnings

import warnings

warnings.filterwarnings("ignore")



# Data manipulation library

import pandas as pd

import numpy as np

import re

from scipy.stats import kurtosis, skew



# Plotting library

import seaborn as sns

import plotly.express as px

import plotly.graph_objects as go 

from plotly.subplots import make_subplots

import matplotlib.pyplot as plt

from matplotlib.colors import LogNorm



# Data transformation libraries

from sklearn.preprocessing import StandardScaler as ss

from sklearn.preprocessing import OneHotEncoder as onehot

from sklearn.preprocessing import RobustScaler as rs

from category_encoders import TargetEncoder

from sklearn.model_selection import train_test_split



# Pipelines libraries

from sklearn.pipeline import make_pipeline

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer 

from sklearn.metrics import accuracy_score



# 1.5 RandomForest modeling

from sklearn.ensemble import RandomForestClassifier as rf



# os related

import os
# Display multiple outputs from a jupyter cell

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"
# Set numpy options to display wide array

np.set_printoptions(precision = 3,          # Display upto 3 decimal places

                    threshold=np.inf        # Display full array

                    )
# Seting display options

pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows', 100)
# os.chdir("E:\HPS\Finance & Accounts\One time activity\SS\Python for analytics\Class Notes & Recordings\Class-18_26-07-20\Assignment")

# os.listdir()
# Reading dataset

df_caravan = pd.read_csv("../input/caravaninsurancechallenge/caravan-insurance-challenge.csv")
df_caravan.shape
# Checking Null values in data set

df_caravan.info()
df_caravan.dtypes.value_counts()
df_caravan.nunique()
# Dropping object from the data set

df1 = df_caravan.drop(["ORIGIN"], axis = 1)

df1.head()
#checking for column wise standard deviation and finding if std dev of any column is zero

# And droping that column in case stddev is zero

s = []

for i in df1.columns:

    s.append(df1[i].std())  

    

s

0 in s

# None of the column has std dev zero
ct = pd.crosstab(df_caravan.ORIGIN, df_caravan.CARAVAN)

ct
ct.sum(0) # sum(0) for column wise sum and sum(1) for row wise sum 
# Splitting dataset



#Training Data

df_train = df_caravan[df_caravan["ORIGIN"]=="train"]

#drop ORIGIN col from train_data

df_train.drop(['ORIGIN'],axis=1,inplace=True)



#Testing Data

df_test = df_caravan[df_caravan["ORIGIN"]=="test"]

#drop ORIGIN col from test_data

df_test.drop(['ORIGIN'],axis=1,inplace=True)
df_train.head()

df_test.head()
df_train['CARAVAN'].value_counts()
# Copy the target Column to a new variable and drop from the dataframe



y = df_train.pop('CARAVAN')

df_train.head()

y.head()
df_train.describe()
df_test.describe()
# Both training & testing data set have matching properties
# Create of numerical and Categorical columns



num_columns = df_train.columns[df_train.nunique() > 6]

cat_columns = df_train.columns[df_train.nunique() <= 6]

len(num_columns)

len(cat_columns)
# Observing distribution

plt.figure(figsize=(15,15))

sns.distributions._has_statsmodels=False



for i in range(len(num_columns)):

    abc = plt.subplot(11,5,i+1)

    out = sns.distplot(df_train[num_columns[i]])

    

plt.tight_layout()
# Column Transformer

ct = ColumnTransformer([

                        ('rs',rs(),num_columns),

                        ('onehot',onehot(handle_unknown='ignore'),cat_columns),

                     ],

    remainder="passthrough"

                    )

dx = ct.fit_transform(df_train)

dx[:5,:5]

X = df_train
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.30)
pipe = Pipeline([('ct',ct),

                 ('rf',rf())

                 ])

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

np.sum(y_pred == y_test)/len(y_test)