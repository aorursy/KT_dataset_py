import numpy as np 
import pandas as pd
pd.set_option('display.max_rows', 800)
pd.set_option('display.max_columns', 500)

import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# import all libraries and dependencies for machine learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV 

import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv("../input/melbourne-housing-market/MELBOURNE_HOUSE_PRICES_LESS.csv")
df.head()
df.info()
df.describe()  
df.head()
num_col = df.select_dtypes(include=np.number).columns
print("Numerical columns: \n",num_col)

cat_col = df.select_dtypes(exclude=np.number).columns
print("Categorical columns: \n",cat_col)
df.drop(["Address","Date","Postcode"], axis=1,inplace=True)
df.dropna(inplace=True)

# Import label encoder 
from sklearn import preprocessing 
  
# label_encoder object knows how to understand word labels. 
label_encoder = preprocessing.LabelEncoder() 
  
# Encode labels in column . 

df['Suburb']= label_encoder.fit_transform(df['Suburb'])
df['Type']= label_encoder.fit_transform(df['Type'])
df['Method']= label_encoder.fit_transform(df['Method'])
df['SellerG']= label_encoder.fit_transform(df['SellerG'])
df['Regionname']= label_encoder.fit_transform(df['Regionname'])
df['CouncilArea']= label_encoder.fit_transform(df['CouncilArea'])
  
df.head()

# Let's check the distribution of y variable
plt.figure(figsize=(10,10), dpi= 80)
sns.boxplot(df['Price'])
plt.title('Price')
plt.show()
plt.figure(figsize=(8,8))
plt.title('Price Distribution Plot')
sns.distplot(df['Price'])
num_col = df.select_dtypes(include=np.number).columns
print("Numerical columns: \n",num_col)

cat_col = df.select_dtypes(exclude=np.number).columns
print("Categorical columns: \n",cat_col)
# Let's check the multicollinearity of features by checking the correlation matric

plt.figure(figsize=(15,15))
p=sns.heatmap(df[num_col].corr(), annot=True,cmap='RdYlGn',center=0)
# Train test split
X = df.drop(['Price'], axis = 1)
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=500)
def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
gbr = GradientBoostingRegressor(learning_rate = 0.05, random_state = 100)
gbr.fit(X_train,y_train)
y_pred = gbr.predict(X_test)

print("r2 score : ",r2_score(y_test,y_pred))
print("MAPE     : ",mean_absolute_percentage_error(y_test,y_pred))
gbr = GradientBoostingRegressor(learning_rate = 0.1, random_state = 100)
gbr.fit(X_train,y_train)
y_pred = gbr.predict(X_test)

print("r2 score : ",r2_score(y_test,y_pred))
print("MAPE     : ",mean_absolute_percentage_error(y_test,y_pred))
gbr = GradientBoostingRegressor(random_state = 100)

# defining parameter range 
param_grid={'n_estimators':[100,200], 
            'learning_rate': [0.15,0.2,0.3,0.5],
            'max_depth':[2,3,5], 
            'min_samples_leaf':[1,3,5]}   
  
grid = GridSearchCV(gbr, param_grid, refit = True, verbose = 3, n_jobs = -1) 
  
# fitting the model for grid search 
grid.fit(X_train, y_train)
# Best parameter after hyper parameter tuning 
print(grid.best_params_) 
  
# Moel Parameters 
print(grid.best_estimator_)
# Prediction using best parameters
grid_predictions = grid.predict(X_test) 
  
print("r2 score : ",r2_score(y_test,grid_predictions))
print("MAPE     : ",mean_absolute_percentage_error(y_test,grid_predictions))
#You can still decrease the mape by trying out different values for estimators ,learning depth and other factors,
#but be mindful that trying out of more values means it will lead to pressure on your RAM and the process will take a lot of time
#Maybe hours as well and your computer might get hanged in between, so do it only if you have powerful gpu and good ram.