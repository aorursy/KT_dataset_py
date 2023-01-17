%matplotlib inline

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score,StratifiedKFold

from sklearn.metrics import r2_score

from sklearn.preprocessing import StandardScaler,MinMaxScaler,LabelEncoder





from sklearn.linear_model import Lasso,Ridge,ElasticNet,BayesianRidge,ARDRegression,RANSACRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.svm import SVR

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor, RandomForestClassifier

from sklearn.decomposition import PCA

from sklearn.neural_network import MLPRegressor

movie_data = pd.read_csv ("../movie_metadata.csv")

movie_data.columns