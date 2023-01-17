# Importing libraries
import numpy as np
import pandas as pd
import seaborn as sns

# Supress warnings
import warnings
warnings.filterwarnings("ignore")

# Classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis , QuadraticDiscriminantAnalysis

# Regression
from sklearn.linear_model import LinearRegression,Ridge,Lasso,RidgeCV, ElasticNet
from sklearn.ensemble import RandomForestRegressor,BaggingRegressor,GradientBoostingRegressor,AdaBoostRegressor 
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

# Modelling Helpers :
from sklearn.preprocessing import Imputer , Normalizer , scale
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFECV
from sklearn.model_selection import GridSearchCV , KFold , cross_val_score

# Preprocessing :
from sklearn.preprocessing import MinMaxScaler , StandardScaler, Imputer, LabelEncoder

# Metrics :
# Regression
from sklearn.metrics import mean_squared_log_error,mean_squared_error, r2_score,mean_absolute_error 
# Classification
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

print("Setup complete...")
# Loading dataset
yt = pd.read_csv("../input/data.csv")
print("Dataset loaded...")
yt.head(5)
yt.drop(['Rank', 'Channel name'], axis = 1, inplace = True)
le = LabelEncoder()
yt['Grade'] = le.fit_transform(yt['Grade'])
yt.head(5)
yt.isnull().sum()
yt.dtypes
# Changing to numeric datatypes for plotting
yt['Video Uploads'] = pd.to_numeric(yt['Video Uploads'], errors = 'coerce')
yt['Subscribers'] = pd.to_numeric(yt['Subscribers'], errors = 'coerce')
yt.dtypes
corr = yt.corr()
sns.heatmap(data = corr, annot = True, square = True, cbar = True)
# above heatmap shows that -
# 1) Subscribers and views have a positive corr
# 2) Grade and subscribers also have a positive corr
sns.lineplot(x = 'Subscribers', y = 'Video views', hue = 'Grade', data = yt)
sns.lineplot(x = 'Grade', y = 'Subscribers', data = yt)
