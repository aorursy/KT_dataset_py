import numpy as np
import pandas as pd 
import missingno as msno
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
from sklearn import metrics
from sklearn.metrics import mean_absolute_error
from sklearn import preprocessing
from warnings import filterwarnings
filterwarnings('ignore')

sns.set(rc={'figure.figsize':(10,8)})
dünya=pd.read_csv("../input/dunyacsv/dnya.csv", sep = ",")
df=dünya.copy()
df.info
df.head()
df.isnull().sum()
df.shape
df.describe().T
df.dtypes
df["Ülke"].unique()
df.tail()
df.columns
df["Gelir"]
df["Gelir TL"]=df["Gelir"]*6.8
df
df["KITA"].value_counts()
df.describe().T
corr = df.corr()
corr
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values);
df.isnull().sum()
df.isnull().sum().sum()