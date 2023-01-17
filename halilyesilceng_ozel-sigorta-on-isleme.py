import numpy as np

import pandas as pd 

from sklearn.model_selection import train_test_split, GridSearchCV,cross_val_score, cross_val_predict

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

import matplotlib.pyplot as plt

from sklearn.preprocessing import scale 

from sklearn import model_selection

from sklearn.tree import DecisionTreeRegressor

from sklearn.linear_model import LinearRegression

from sklearn.neighbors import KNeighborsRegressor

from sklearn.ensemble import BaggingRegressor

from argparse import Namespace

import seaborn as sns 



from sklearn import preprocessing

from warnings import filterwarnings

filterwarnings('ignore')



sns.set(rc={'figure.figsize':(10,8)})
args = Namespace(

    target="expenses",

    data_file="0_dataset.csv",

    cv = 10

)
premium =  pd.read_csv("../input/insurance-premium-prediction/insurance.csv")

df = premium.copy()
df.head(10)

df.info()
df.isna().sum()
df["age"].unique()
df["sex"].unique()
df["bmi"].unique()
df["children"].unique()
df["smoker"].unique()
df["region"].unique()
df["expenses"].unique()
df.sample(30)
df.describe()
le = preprocessing.LabelEncoder()
df["sex"] = le.fit_transform(df["sex"])  # 0 is female, 1 is male

df.head()
df["smoker"] = le.fit_transform(df["smoker"])  # 1 is yes, 0 is no

df.head()
df_region = pd.get_dummies(df["region"])

df_region.head()
df_region.columns = ['Bursa', 'Ankara', 'Istanbul', 'Izmir']

df_region.head()
df = pd.concat([df, df_region], axis = 1)

df.drop(["Bursa", "region"], axis = 1, inplace = True)

df.head()
df.head()
age = df[['age']].values.astype(float)



min_max_scaler = preprocessing.MinMaxScaler()

age_scaled = min_max_scaler.fit_transform(age)

df['age_scaled'] = pd.DataFrame(age_scaled)
bmi = df[['bmi']].values.astype(float) 



min_max_scaler = preprocessing.MinMaxScaler() 

bmi_scaled = min_max_scaler.fit_transform(bmi) 

df['bmi_scaled'] = pd.DataFrame(bmi_scaled) 
df.head()
df.drop(["age", "bmi"], axis = 1, inplace = True)

df.head()
export_csv = df.to_csv (r'1_preprocessed_data.csv', index = None, header = True)