import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
# importing libraries 
from sklearn.ensemble import VotingClassifier ,BaggingClassifier, ExtraTreesClassifier, RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier,GradientBoostingClassifier
from sklearn.metrics import accuracy_score 
from numpy import mean,std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score,RepeatedStratifiedKFold,train_test_split
from sklearn.linear_model import LogisticRegression,RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from matplotlib import pyplot
from sklearn.datasets import load_wine,load_iris
from matplotlib.pyplot import figure
figure(num=2, figsize=(16, 12), dpi=80, facecolor='w', edgecolor='k')
import xgboost as xgb
from sklearn.feature_selection import SelectKBest,f_regression
from sklearn.linear_model import LinearRegression,BayesianRidge,ElasticNet,Lasso,SGDRegressor,Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,RobustScaler,StandardScaler
from sklearn.pipeline import make_pipeline,Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA,KernelPCA
from sklearn.ensemble import ExtraTreesRegressor,GradientBoostingRegressor,RandomForestRegressor,VotingClassifier
from sklearn.model_selection import cross_val_score,KFold,GridSearchCV,RandomizedSearchCV,StratifiedKFold,train_test_split
from sklearn.base import BaseEstimator,clone,TransformerMixin,RegressorMixin
from sklearn.svm import LinearSVR,SVR
#import xgboost 
from xgboost import XGBRegressor
#Import Pandas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from scipy.stats import skew
from scipy.stats.stats import pearsonr
%matplotlib inline
seed = 1075
np.random.seed(seed)

exercise_data = pd.read_csv( '../input/exercise.csv' )
calories_data = pd.read_csv( '../input/calories.csv' )
exercise_data.head() 
calories_data.head()
df = pd.merge(exercise_data,calories_data,on='User_ID', how='left')
df.head()
df.info()
sns.pairplot(df,kind = "scatter")
# in the scatter plot of duration vs calories and heart rate vs calories the relationship
# was curved upward (not linear)
# feature engineering:  add squared duration and heart rate to try a better fit with calories
df = df.assign( squared_duration = df[ 'Duration' ] ** 2 )
df = df.assign( squared_heart_rate = lambda x: x[ 'Heart_Rate' ] ** 2 )

df.head()
sns.pairplot(df,kind = "scatter")
# since we don't want the prediction to be negative calories, 
# convert calories to natural logarithm to always get a positive number
import numpy as np
df = df.assign( log_Calories = lambda x: 
                                 np.log( x[ 'Calories' ] ) )
df.head()
# scale numbers with normal distribution using z-score
from scipy.stats import zscore

df = df.assign( zscore_body_temp = zscore( df[ 'Body_Temp' ] ) )
df = df.assign( zscore_height = zscore( df[ 'Height' ] ) )
df = df.assign( zscore_weight = zscore( df[ 'Weight' ] ) )
df = df.assign( zscore_squared_heart_rate = zscore( df[ 'squared_heart_rate' ] ) )

df.head()
# scale non-normal columns (age, squared_duration) using Min-Max 
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

# NOTE:  joined_data[ ['Age', 'squared_duration'] ] produces a copy, loc doesn't
minMaxData = pd.DataFrame( scaler.fit_transform( df.loc[ :, ['Age','squared_duration'] ] )
                         , columns = [ 'minMaxAge', 'minMaxSquaredDuration' ] )
df = pd.concat( [ df, minMaxData ], axis = 1, join = 'inner' )
df.head()
# what to do with Gender (string binary categorical variable)?
# convert to zero (male) and one (female)
# trick:  first convert to boolean (Gender==female) , then to int by adding 0
df = df.assign( numeric_gender = 0 + ( df[ 'Gender' ] == 'female' ) )
df.head()
# exclude User_ID and log_Calories from the prediction model (they're not features)
del df[ 'User_ID' ]
ageDF = df[ 'Age' ]
heartRateDF = df[ 'Heart_Rate' ]

# remove unneeded columns

# remove Duration and Heart_Rate
del df[ 'Duration' ]
del df[ 'Heart_Rate' ]
del df[ 'Calories' ]




df.pop( 'Body_Temp' )
df.pop( 'Height' )
df.pop( 'Weight' )
df.pop( 'squared_heart_rate' )
df.pop( 'Age' )
df.pop( 'squared_duration' )
df.pop( 'Gender' )
df.info()
# split data into test and training

from sklearn.model_selection import train_test_split
X = df.drop('log_Calories',axis = 1)
y = df['log_Calories']
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.6, random_state=42)
X_train.shape, X_test.shape

#train_X,test_X,train_Y,test_Y = train_test_split( df, test_size = 0.3 )
X.head()
from sklearn.tree import DecisionTreeRegressor
dt_model = DecisionTreeRegressor(random_state=10)
dt_model.fit(X_train, y_train)
dt_model.score(X_train, y_train)
dt_model.score(X_test, y_test)
y_pred = dt_model.predict(X_test)
predctn =pd.DataFrame({'Actual':y_test, 'Predicted':y_pred})
predctn
from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor()
rf_model.fit(X_train,y_train)
rf_model.score(X_train,y_train)
rf_model.score(X_test,y_test)
y_pred_rf = rf_model.predict(X_test)
rf_pred =pd.DataFrame({'Actual':y_test, 'Predicted':y_pred_rf})
rf_pred
from sklearn.metrics import mean_squared_error, r2_score
# The mean squared error, The MSE is a measure of the quality of an estimator—it is always non-negative, and values closer to zero are better.
print("Mean squared error Random Forest: %.2f"% mean_squared_error(y_test, y_pred_rf))
print("Mean squared error Decision Tree: %.2f"% mean_squared_error(y_test, y_pred))

# Explained variance score: 1 is perfect prediction
print('Test Variance score Random Forest: %.2f' % r2_score(y_test, y_pred_rf))
print('Test Variance score Decision Tree: %.2f' % r2_score(y_test, y_pred))