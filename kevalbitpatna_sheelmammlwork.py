import xgboost
import numpy as np
import pandas as pd
from math import sqrt
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn import cross_validation, metrics
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score
train = pd.read_csv("../input/bank.csv")

train.head()
train.head(15)
train.info()
train.describe()
train.columns
train.values
train.job.value_counts()
train.marital.value_counts()
train.default.value_counts()
train.housing.value_counts()
train.loan.value_counts()
train.contact.value_counts()    
train.month.value_counts() 
train.day_of_week.value_counts() 
train.poutcome.value_counts() 
train.shape
train.head()

