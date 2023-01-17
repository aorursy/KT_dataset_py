#data preprocessing
import pandas as pd

#Linear Algebra
import numpy as np

#Data Visualization
import seaborn as sns
%matplotlib inline
import matplotlib.pyplot as plt
from matplotlib import style
#plotly
!pip install chart_studio
import plotly.express as px
import chart_studio.plotly as py
import plotly.graph_objs as go
from plotly.offline import iplot
import cufflinks
cufflinks.go_offline()
cufflinks.set_config_file(world_readable=True, theme='pearl')

#Algorithms
from sklearn.svm import SVR
from sklearn import linear_model
from sklearn.linear_model import Ridge,Lasso,ElasticNet
from pandas import Series, DataFrame
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.tree import DecisionTreeRegressor 
from sklearn.ensemble import RandomForestRegressor 
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.ensemble import GradientBoostingRegressor 
from sklearn import metrics  
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.preprocessing import StandardScaler,LabelEncoder


import warnings
warnings.filterwarnings('ignore')
flight=pd.read_excel('../input/data-train/Data_Train.xlsx')
flight.head()
flight.describe()
flight.info()
flight.shape
flight['Date']=flight['Date_of_Journey'].str.split('/').str[0]
flight['Month']=flight['Date_of_Journey'].str.split('/').str[1]
flight['Year']=flight['Date_of_Journey'].str.split('/').str[2]

flight.head()
flight.dtypes
flight['Date']=flight['Date'].astype(int)
flight['Month']=flight['Month'].astype(int)
flight['Year']=flight['Year'].astype(int)
flight.dtypes
#droping Date-of-journey column
flight.drop('Date_of_Journey',axis=1,inplace=True)

flight.head()
flight['Arrival_time']=flight['Arrival_Time'].str.split(' ').str[0]
flight.head()
#dropping Arrival_time
flight.drop('Arrival_Time',axis=1,inplace=True)
flight.head()
flight[flight['Total_Stops'].isnull()]
flight['Total_Stops']=flight['Total_Stops'].fillna('1 stop')
flight['Total_Stops']=flight['Total_Stops'].replace('non-stop','0 stop')
flight.head()
flight['Stop']=flight['Total_Stops'].str.split(' ').str[0]
flight.head()
flight.drop('Total_Stops',axis=1,inplace=True)
flight.head()
flight['Stop']=flight['Stop'].astype(int)
flight.dtypes
flight['Arrival_hour']=flight['Arrival_time'].str.split(':').str[0]
flight['Arrival_minutes']=flight['Arrival_time'].str.split(':').str[1]
flight.head()
#Dropping Arrival_time feature from data
flight.drop('Arrival_time',axis=1,inplace=True)
#converting data type from string to float
flight['Arrival_hour']=flight['Arrival_hour'].astype(int)
flight['Arrival_minutes']=flight['Arrival_minutes'].astype(int)
flight['Dep_hour']=flight['Dep_Time'].str.split(':').str[0]
flight['Dep_minutes']=flight['Dep_Time'].str.split(':').str[1]
flight['Dep_hour']=flight['Dep_hour'].astype(int)
flight['Dep_minutes']=flight['Dep_minutes'].astype(int)
flight.drop('Dep_Time',axis=1,inplace=True)
flight.head()
flight['Route_1']=flight['Route'].str.split('→').str[0]
flight['Route_2']=flight['Route'].str.split('→').str[1]
flight['Route_3']=flight['Route'].str.split('→').str[2]
flight['Route_4']=flight['Route'].str.split('→').str[3]
flight['Route_5']=flight['Route'].str.split('→').str[4]

flight.head()
flight['Price'].fillna((flight['Price'].mean()),inplace=True)
flight['Route_1'].fillna('None',inplace=True)
flight['Route_2'].fillna('None',inplace=True)
flight['Route_3'].fillna('None',inplace=True)
flight['Route_4'].fillna('None',inplace=True)
flight['Route_5'].fillna('None',inplace=True)
flight.head()
flight.drop(['Route','Duration'],axis=1,inplace=True)
flight.head()
from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
flight['Airline']=encoder.fit_transform(flight['Airline'])
flight['Source']=encoder.fit_transform(flight['Source'])
flight['Destination']=encoder.fit_transform(flight['Destination'])
flight['Additional_Info']=encoder.fit_transform(flight['Additional_Info'])
flight['Route_1']=encoder.fit_transform(flight['Route_1'])
flight['Route_2']=encoder.fit_transform(flight['Route_2'])
flight['Route_3']=encoder.fit_transform(flight['Route_3'])
flight['Route_4']=encoder.fit_transform(flight['Route_4'])
flight['Route_5']=encoder.fit_transform(flight['Route_5'])
flight.head()
#dropping year column
flight.drop('Year',axis=1,inplace=True)
flight.isnull().sum()
#check still any missing values present or not
sns.heatmap(flight.isnull())
flight['Stop'].value_counts().iplot(kind='bar',
                                              yTitle='Counts', 
                                              linecolor='black', 
                                              opacity=0.7,
                                              color='blue',
                                              theme='pearl',
                                              bargap=0.5,
                                              gridcolor='white',
                                              title='Distribution of classes column ')

fig = px.scatter(flight, x="Arrival_hour", y="Dep_hour", color='Price')
fig.show()
#check cor-relation
corr_hmap=flight.corr()
plt.figure(figsize=(8,7))
sns.heatmap(corr_hmap,annot=True)
plt.show()
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
x=flight.drop('Price',axis=1)
x.head()
y=flight['Price']
y.head()
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)
x_train.shape
y_train.shape
x_test.shape
y_test.shape
model=SelectFromModel(Lasso(alpha=0.005,random_state=0))
model.fit(x_train,y_train)
model.get_support()
selected_features=x_train.columns[(model.get_support())]
selected_features
from sklearn.model_selection import RandomizedSearchCV
#Randomized Search CV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
# max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 100]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

print(random_grid)
# Use the random grid to search for best hyperparameters
# First create the base model to tune
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation, 
# search across 50 different combinations
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 50, cv = 5, verbose=2, random_state=42, n_jobs = 1)

rf_random.fit(x_train,y_train)
y_pred=rf_random.predict(x_test)
rsquare=metrics.r2_score(y_test,y_pred)
print('R-square',rsquare)
sns.distplot(y_test-y_pred)
plt.scatter(y_test,y_pred)

pred_rfr=rf_random.predict(x_test)
print("predicted price",pred_rfr)
print("actual price",y_test)