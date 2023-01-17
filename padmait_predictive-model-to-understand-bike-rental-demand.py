import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# To split into training-test data set
from sklearn.model_selection import train_test_split

# To scale the dataset
from sklearn.preprocessing import MinMaxScaler

# To get statistical information of the model
import statsmodels.api as sm

# Check for the VIF values of the feature variables. 
from statsmodels.stats.outliers_influence import variance_inflation_factor

# To calculate the R-squared score, RMSE, MAE on the test set.
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from math import sqrt

# Importing RFE and LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        file_name = os.path.join(dirname, filename)
        print(file_name)
        
bike_df = pd.read_csv(file_name)
bike_df.head() # Checking the top 5 rows of the dataframe
# checking botton 5 rows of the dataframe
bike_df.tail()
# Checking the shape of the dataframe
bike_df.shape
# Checking the size of the dataframe
bike_df.size
# Inspecting type
print(bike_df.dtypes)
# How many types of each data type column exists and total memory usage
bike_df.info()
# Looking for any null value in any column 
print(bike_df.isnull().any())
# Checking the numerical columns data distribution
bike_df.describe()
# Checking the number of unique values each column possess to identify categorical columns
bike_df.nunique().sort_values()
categorical_columns = ['season','mnth','weekday','weathersit']
for col in categorical_columns:
    bike_df[col] =pd.Categorical(bike_df[col])
bike_df.info() # Observe the data frame after conversion
sns.pairplot(bike_df[['temp','atemp','hum','windspeed','cnt']])
plt.show()
# Let's check the correlation coefficients to see which variables are highly correlated
plt.figure(figsize = (16, 10))
sns.heatmap(bike_df.corr(), annot = True, cmap="YlGnBu")
plt.show()
drop_cols = ['instant', 'dteday', 'casual', 'registered','atemp']
bike_df.drop(labels=drop_cols,axis=1,inplace=True)
bike_df.info()
plt.figure(figsize=(20, 12))

features = ['yr','mnth','season','weathersit','holiday','workingday','weekday']

for i in enumerate(features):
    plt.subplot(2,4,i[0]+1)
    sns.boxplot(x = i[1], y = 'cnt', data = bike_df)
    plt.title(i[1])
plt.figure(figsize=(20, 6))

features = ['temp','hum','windspeed','cnt']

for i in enumerate(features):
    plt.subplot(1,4,i[0]+1)
    sns.distplot(bike_df[i[1]])
    plt.title(i[1])
plt.show()
# Replacing the numbers with strings from data dictionary so that the column names will be meaningful

bike_df[['season']]=bike_df[['season']].apply(lambda x: x.map({1: "spring", 2: "summer", 3:"fall", 4:"winter"}))
bike_df[['mnth']]=bike_df[['mnth']].apply(lambda x: x.map({1:"jan",2:"feb",3:"mar",4:"apr",5:"may",6:"jun",7:"jul",8:"aug",9:"sep",10:"oct",11:"nov",12:"dec"}))
bike_df[['weekday']]=bike_df[['weekday']].apply(lambda x: x.map({1:"mon",2:"tue",3:"wed",4:"thur",5:"fri",6:"sat",0:"sun"}))
bike_df[['weathersit']]=bike_df[['weathersit']].apply(lambda x: x.map({1:"clear",2:"mist",3:"snow",4:"rain"}))

bike_df.head()
# Let's drop the first column from corresponding dummy variables in df using 'drop_first = True'
season = pd.get_dummies(bike_df['season'], drop_first = True)
mnth = pd.get_dummies(bike_df['mnth'], drop_first = True)
weekday = pd.get_dummies(bike_df['weekday'], drop_first = True)
weathersit = pd.get_dummies(bike_df['weathersit'], drop_first = True)

# Add the results to the original bike sharing dataframe
bike_df = pd.concat([bike_df, season,mnth,weekday,weathersit], axis = 1)

# Drop the categorical variables as we have created the dummies for it
bike_df.drop(['season','mnth','weekday','weathersit'], axis = 1, inplace = True)
bike_df.head()
bike_df.info() # Observe the dataframe after creation of dummy variables and dropping categorical variables
df_train, df_test = train_test_split(bike_df, train_size = 0.70, test_size = 0.30, random_state = 10) 
df_train.shape # Observe the train dataset shape
df_test.shape# Observe the train dataset shape
df_train.isnull().any()
df_test.isnull().any()
scaler = MinMaxScaler()
# Apply scaler() to all the columns(only for continuous variables) except the 'yes-no' and 'dummy' variables
num_vars = ['temp','hum','windspeed','cnt']

df_train[num_vars] = scaler.fit_transform(df_train[num_vars])
df_train.head()
df_train.describe()
y_train = df_train.pop('cnt') # target variable 
X_train_initial = df_train # target variable should not be in predictor list
print(X_train_initial.head(),"\nY train:\n",y_train.head())
# Use linear regression as the model and RFE to select 15 features
lr = LinearRegression()
#rank all features, i.e continue the elimination until the last one
rfe = RFE(lr, n_features_to_select=15)
rfe = rfe.fit(X_train_initial, y_train)

print("All features by their rank:")
print(sorted(list(zip(X_train_initial.columns,rfe.support_,rfe.ranking_))))
col = X_train_initial.columns[rfe.support_]
print("RFE selected columns:\n", col)
def build_lr_model(feature_list):
    X_train = X_train_initial[feature_list] # get feature list 
    X_train_lm = sm.add_constant(X_train) # required by statsmodels 
    lr = sm.OLS(y_train, X_train_lm).fit() # build model and learn coefficients
    print("Co-efficients:\n",lr.params) # OLS coefficients
    print(lr.summary()) # OLS summary with R-squared, adjusted R-squared, p-value etc.
    calculate_vif(X_train) # Calculate VIF for features
    return(lr, X_train_lm) # return the model and the X_train fitted with constant

def calculate_vif(X_train):
    vif = pd.DataFrame()
    vif['Features'] = X_train.columns # Read the feature names
    vif['ViF'] = [variance_inflation_factor(X_train.values,i) for i in range(X_train.shape[1])] # calculate VIF
    vif['ViF'] = round(vif['ViF'],2)
    vif.sort_values(by='ViF', ascending = False, inplace=True)  
    print(vif) # prints the calculated VIFs for all the features
features = list(col) #  Use RFE selected variables
lr_model1, X_train_lm1 = build_lr_model(features) # Call the function and get the model and the X_train_lm for prediction
features = list(col) #  Use RFE selected variables
features.remove('may') # Remove 'may' from RFE features list
lr_model2, X_train_lm2 = build_lr_model(features) # Call the function and get the model and the X_train_lm for prediction
features = list(col) #  Use RFE selected variables
features.remove('may') # Remove 'may' from RFE features list as per model 2
features.remove('temp') # Remove 'temp' from RFE features list
lr_model3, X_train_lm3 = build_lr_model(features) # Call the function and get the model and the X_train_lm for prediction
features = list(col) #  Use RFE selected variables
features.remove('may') # Remove 'may' from features list as per model 2
lr_model2, X_train_lm2 = build_lr_model(features) # Call the function and get the model and the X_train_lm for prediction
features = list(col) #  Use RFE selected variables
features.remove('may') # Remove 'may' from features list as per model 2
features.remove('hum')  # Remove 'hum' from features list
lr_model4, X_train_lm4 = build_lr_model(features) # Call the function and get the model and the X_train_lm for prediction
features = list(col) #  Use RFE selected variables
features.remove('may') # Remove 'may' from features list as per model 2
features.remove('hum')  # Remove 'hum' from features list as per model 4
features.remove('aug') # Remove 'aug' from features list
lr_model5, X_train_lm5 = build_lr_model(features) # Call the function and get the model and the X_train_lm for prediction
features = list(col) #  Use RFE selected variables
features.remove('may') # Remove 'may' from features list as per model 2
features.remove('hum') # Remove 'hum' from features list as per model 4 
features.remove('aug') # Remove 'aug' from features list as per model 5
features.remove('fall') # Remove 'aug' from features list
lr_model6, X_train_lm6 = build_lr_model(features) # Call the function and get the model and the X_train_lm for prediction
features = list(col) #  Use RFE selected variables
features.remove('may') # Remove 'may' from features list as per model 2
features.remove('hum') # Remove 'hum' from features list as per model 4 
features.remove('aug') # Remove 'aug' from features list as per model 5
features.remove('fall') # Remove 'aug' from features list as per model 6
features.remove('mar') # Remove 'fall' from features list
lr_model7, X_train_lm7 = build_lr_model(features) # Call the function and get the model and the X_train_lm for prediction
features = list(col) #  Use RFE selected variables
features.remove('may') # Remove 'may' from features list as per model 2
features.remove('hum') # Remove 'hum' from features list as per model 4 
features.remove('aug') # Remove 'aug' from features list as per model 5
features.remove('fall') # Remove 'aug' from features list as per model 6
features.remove('mar') # Remove 'fall' from features list as per model 7
features.remove('oct') # Remove 'fall' from features list
lr_model8, X_train_lm8 = build_lr_model(features) # Call the function and get the model and the X_train_lm for prediction
features = list(col) #  Use RFE selected variables
features.remove('may') # Remove 'may' from features list as per model 2
features.remove('hum') # Remove 'hum' from features list as per model 4 
features.remove('aug') # Remove 'aug' from features list as per model 5
features.remove('fall') # Remove 'aug' from features list as per model 6
features.remove('mar') # Remove 'fall' from features list as per model 7
features.remove('oct') # Remove 'fall' from features list as per model 8
features.remove('mist') # Remove 'mist' from features list
lr_model9, X_train_lm9 = build_lr_model(features) # Call the function and get the model and the X_train_lm for prediction
features = list(col) #  Use RFE selected variables
features.remove('may') # Remove 'may' from features list as per model 2
features.remove('hum') # Remove 'hum' from features list as per model 4 
features.remove('aug') # Remove 'aug' from features list as per model 5
features.remove('fall') # Remove 'fall' from features list as per model 6
features.remove('mar') # Remove 'mar' from features list as per model 7
features.remove('oct') # Remove 'oct' from features list as per model 8
lr_model, X_train_lm = build_lr_model(features) # Call the function and get the model and the X_train_lm for prediction
lr_model.params
lr_model.summary()
y_train_predicted = lr_model.predict(X_train_lm) # get predicted value on training dataset using statsmodels predict()
residual_values = y_train - y_train_predicted # difference in actual Y and predicted value
plt.figure(figsize=[10,5])
plt.subplot(121)
sns.distplot(residual_values, bins = 15) # Plot the histogram of the error terms
plt.title('Residuals follow normal distribution', fontsize = 18)
plt.subplot(122) 
plt.scatter(y_train_predicted, residual_values) # Residual vs Fitted Values
plt.plot([0,0],'r') # draw line at 0,0 to show that residuals have constant variance
plt.title('Residual vs Fitted Values: No Pattern Seen')
plt.show()
num_vars = ['temp','hum','windspeed','cnt']

df_test[num_vars] = scaler.transform(df_test[num_vars]) # Use the scaler of training data set and transform test dataset
y_test = df_test.pop('cnt') # actual target values 
X_test = df_test # remove target variable from the features
# Creating X_test_model dataframe by selecting features of the model
print("Model features are ", features)
X_test_model = X_test[features] # features have the list of variables in the model
print("Checking test data set features: " , X_test_model.columns)

X_test_model = sm.add_constant(X_test_model) # Adding constant variable as required by statsmodels

# Making predictions using the final model
y_predicted_model = lr_model.predict(X_test_model)
rmse = sqrt(mean_squared_error(y_test,y_predicted_model))
print('Root mean square error :',rmse)
mae=mean_absolute_error(y_test,y_predicted_model)
print('Mean absolute error :',mae)
train_r2 = round(r2_score(y_train, y_train_predicted),3)
train_r2
n = df_train.shape[0]
p = len(features)
print(n,p)
round(1-(1-train_r2)*(n-1)/(n-p-1),3) 
test_r2 = round(r2_score(y_test, y_predicted_model),4)
test_r2
n = df_test.shape[0]
p = len(features)
print(n,p)
round(1-(1-test_r2)*(n-1)/(n-p-1),4)
features
lr_model.params