
"""
Created on Sun Aug  2 09:31:33 2020

@author: Rupesh Roy
"""

# 1.1 Call data manipulation libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
# 1.2 Data transformation classes
from sklearn.preprocessing import RobustScaler
from category_encoders import TargetEncoder
from sklearn.model_selection import train_test_split
# 1.3  Pipelines libraries
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
# 1.5 RandomForest modeling
from sklearn.ensemble import RandomForestClassifier 
#2.1 Set the path of the data file and import dataset in dataframe
path = "/kaggle/input"
os.chdir(path)
os.listdir(path)
df_carvan = pd.read_csv("../input/caravan-insurance-challenge/caravan-insurance-challenge.csv")
df_carvan.shape
# 2.2 Check null values in original dataset

df_carvan.info() #there is no null values 


# 2.3 Remove target column

target = df_carvan.iloc[:,[0,-1]]
df_carvan = df_carvan.drop(columns=['CARAVAN'])
target.info()
# 2.4 Check unique values in dataset to decide numerical and categorical columns

uniqueValues = df_carvan.nunique()
uniqueValues[uniqueValues<=4].index

cat_cols = uniqueValues[uniqueValues<=4].index[1:]
num_cols = uniqueValues[uniqueValues>4].index

# 3.0 Split dataframe in train/test based on ORIGIN column value 
X_train = df_carvan[df_carvan["ORIGIN"]=="train"].copy()
X_test = df_carvan[df_carvan["ORIGIN"]=="test"].copy()


X_train = X_train.drop(columns=['ORIGIN'])
X_test = X_test.drop(columns=['ORIGIN'])
y_train = target[target["ORIGIN"]=="train"].iloc[:,1]
y_test = target[target["ORIGIN"]=="test"].iloc[:,1]

X_train.shape
X_test.shape
# 4.0 Find distribution of dataset 
df_carvan[num_cols].columns[:20]
for i,j in enumerate(df_carvan[num_cols].columns[:20]):
    plt.subplot(5,4,i+1)
    sns.distplot(df_carvan[num_cols][j],kde_kws={'bw':0.1})
# 5.0 Implementing Pipeline
# As there are outliners in good numbers we use RobustScaler for scaling 
# 5.1
ct = ColumnTransformer([
                        ('robust_scalar', RobustScaler(), num_cols),
                        ('ohe', TargetEncoder(),cat_cols),
                        ],remainder="passthrough",
    )
pipe = Pipeline([
     ('ct',ct),
     ('rf',RandomForestClassifier())
     ])
# 5.2 Use Pipeline to fit train data

pipe.fit(X_train,y_train)

# 5.2 Pedict test data to find accuracy score of Model
y_predict = pipe.predict(X_test)
accuracy_score(pipe.predict(X_test), y_test)