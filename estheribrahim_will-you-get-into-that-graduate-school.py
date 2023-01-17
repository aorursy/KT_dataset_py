import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import sys

import os

from sklearn.model_selection import train_test_split



#Feature Scaling (Scaling using minmaxscaler)

from sklearn.preprocessing import MinMaxScaler



#Regression libraries used in this kernel

from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import AdaBoostRegressor

from sklearn.ensemble import GradientBoostingRegressor



#Metrics for regression model performance

from sklearn.metrics import r2_score

from sklearn.metrics import mean_squared_error



#Reading the data into pandas dataframe



df = df = pd.read_csv("../input/Admission_Predict.csv",sep = ",")






