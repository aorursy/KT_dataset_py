# Importing the Libraries
import pandas as pd                                  # For managing Data Structures
import numpy as np                                   # For mathematical functions
import matplotlib.pyplot as plt                      # For Data visualization
import seaborn as sns                                # For Data visualization
from mpl_toolkits.mplot3d import Axes3D              # For 3D graphs
from sklearn.impute import SimpleImputer             # For handeling the missing data (Categorical)
from sklearn.preprocessing import LabelEncoder       # For Label encoding
from sklearn.preprocessing import OneHotEncoder      # For One Hot Encoding
from sklearn.compose import ColumnTransformer        # Fro using OneHotEncoder to transform columns
from sklearn.linear_model import LinearRegression    # For linear regression model
from sklearn.tree import DecisionTreeRegressor       # For Decision tree regression model
from sklearn.ensemble import RandomForestRegressor   # For Random Forest Regression Model
from sklearn import metrics                          # For Evaluation of the regression models
train_df = pd.read_csv("../input/big-mart-sales-prediction/Train.csv")
test_df = pd.read_csv("../input/big-mart-sales-prediction/Test.csv")
y_df = pd.read_csv("../input/big-mart-sales-prediction/Submission.csv")
# Changing the Column Names
train_df.columns = ['Item_ID','Weight','Fat_Content','Visibility','Item_Type',
                    'MRP','Out_ID', 'Out_year','Out_Size','Out_Loc','Out_Type', 'Sales']
test_df.columns = ['Item_ID','Weight','Fat_Content','Visibility','Item_Type',
                    'MRP','Out_ID', 'Out_year','Out_Size','Out_Loc','Out_Type']
# getting basic information of training and test datasets
train_df.info()
test_df.info()
# Checking the Unique values for categorical dat
print("Fat_Content\n ",train_df.Fat_Content.unique())
print("Item_Type\n ",train_df.Item_Type.unique())
print("Out_ID\n ",train_df.Out_ID.unique())
print("Out_Size\n ",train_df.Out_Size.unique())
print("Out_Loc\n ",train_df.Out_Loc.unique())
print("Out_Type\n ",train_df.Out_Type.unique())
# Handling categories in "Fat_Content"
# Training Set
train_df['Fat_Content'] = train_df['Fat_Content'].replace('low fat', 'Low Fat')
train_df['Fat_Content'] = train_df['Fat_Content'].replace('LF', 'Low Fat')
train_df['Fat_Content'] = train_df['Fat_Content'].replace('reg', 'Regular')
# Test Set
test_df['Fat_Content'] = test_df['Fat_Content'].replace('low fat', 'Low Fat')
test_df['Fat_Content'] = test_df['Fat_Content'].replace('LF', 'Low Fat')
test_df['Fat_Content'] = test_df['Fat_Content'].replace('reg', 'Regular')

print("New Categories: Fat_Content (Training Set)\n ",train_df.Fat_Content.unique())
print("New Categories: Fat_Content (Test Set)\n ",test_df.Fat_Content.unique())
# Checking the Missing values
print(pd.concat([train_df.isnull().sum(), (train_df.isnull().sum()/train_df.isnull().count()*100)],
                    axis = 1,
                    keys = ['Missing values (Train Set)','%']))
print(pd.concat([test_df.isnull().sum(), (test_df.isnull().sum()/test_df.isnull().count()*100)],
                    axis = 1,
                    keys = ['Missing values (Test Set)','%']))
#--------------------------------- Handling the Missing Data-------------------------------------------
# ---------------- Training Set
# Weight
si1 = SimpleImputer(missing_values = np.nan, strategy = 'mean')
arr = train_df.iloc[:,1].values.reshape(-1,1)
si1 = si1.fit(arr)
arr = si1.transform(arr)
train_df['Weight'] = arr[:,0]
# Out Size
si2 = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent')
arr = train_df.iloc[:,8].values.reshape(-1,1)
si2 = si2.fit(arr)
arr = si2.transform(arr)
train_df['Out_Size'] = arr[:,0]
# Checking values after Imputing - Training set
print(pd.concat([train_df.isnull().sum(), (train_df.isnull().sum()/train_df.isnull().count()*100)],
                    axis = 1,
                    keys = ['Missing values (Train Set)','%']))

#---------------- Test Set
# Weight
arr = test_df.iloc[:,1].values.reshape(-1,1)
si1 = si1.fit(arr)
arr = si1.transform(arr)
test_df['Weight'] = arr[:,0]
# Out_Size
arr = test_df.iloc[:,8].values.reshape(-1,1)
si2 = si2.fit(arr)
arr = si2.transform(arr)
test_df['Out_Size'] = arr[:,0]
# Checking values after Imputing - Training set
print(pd.concat([test_df.isnull().sum(), (test_df.isnull().sum()/test_df.isnull().count()*100)],
                    axis = 1,
                    keys = ['Missing values (Test Set)','%']))
#--------------- EDA -----------------
# No of times different types of Items has been purchased
sns.set_style('darkgrid')
plt.figure(figsize=(15,10))
sns.countplot(train_df.Item_Type, hue=train_df.Fat_Content)
plt.xticks(rotation=90)
plt.legend(loc = 'upper right', bbox_to_anchor=(1.1, 1), title = 'Fat Content')
# Average sales from different Item Category
plt.figure(figsize=(15,10))
sns.barplot(x = 'Item_Type', y = 'Sales', data = train_df)
plt.xticks(rotation = 90)
# Visibility of different Item Type
plt.figure(figsize = (15,10))
sns.barplot(x = 'Item_Type', y = 'Visibility', data = train_df)
plt.xticks(rotation = 90)
# Sales from different Outlet Location
plt.figure(figsize=(15,10))
sns.barplot(x = 'Out_Loc', y = 'Sales', hue = 'Out_Size', data = train_df)
plt.xlabel("Outlet Location")
plt.legend(loc = 'upper right', bbox_to_anchor=(1.1, 1), title = 'Outlet Size')
# Sales from differenr type of Outlets
plt.figure(figsize=(15,10))
sns.barplot(x = 'Out_Type', y = 'Sales', hue = 'Out_Size', data = train_df)
plt.xlabel("Outlet Type")
plt.legend(loc = 'upper right', bbox_to_anchor=(1.1, 1), title = 'Outlet Size')
# Relation between Visibility and Sales
plt.figure(figsize=(15,10))
plt.scatter(train_df.Visibility, train_df.Sales, marker = '.', edgecolors = 'Black')
plt.xlabel("Visibility")
plt.ylabel("Sales")
# Relation between the price of the item and sales
plt.figure(figsize=(15,10))
plt.scatter(train_df.MRP, train_df.Sales, marker = '.', edgecolors = 'black')
plt.xlabel("MPR")
plt.ylabel("Sales")
# 3D representation between Visibility, MRP and Sales
fig = plt.figure(figsize=(15,10))
ax = Axes3D(fig)
ax.scatter(train_df.Visibility,
           train_df.MRP, 
           train_df.Sales, 
           marker = 'o', edgecolors = 'black')
ax.set_xlabel('Visibility')
ax.set_ylabel('MRP')
ax.set_zlabel('Sales')
ax.legend()
ax.grid(linestyle='-', linewidth='0.5', color='red')
# ------------------- Encoding the categorical variables using Label encoder

lencoder = LabelEncoder()
# Training Set
for i in (2,4,6,7,8,9,10):
    train_df.iloc[:,i] = lencoder.fit_transform(train_df.iloc[:,i])
# Test set
for i in (2,4,6,7,8,9,10):
    test_df.iloc[:,i] = lencoder.fit_transform(test_df.iloc[:,i])
# Checking the Unique values for categorical data after label encoding
print("Fat_Content\n ",train_df.Fat_Content.unique())
print("Item_Type\n ",train_df.Item_Type.unique())
print("Out_ID\n ",train_df.Out_ID.unique())
print("Out_Size\n ",train_df.Out_Size.unique())
print("Out_Loc\n ",train_df.Out_Loc.unique())
print("Out_Type\n ",train_df.Out_Type.unique())
# Plotting a heatmap to visualise the correlation between different variables
plt.figure(figsize = (15,10))
sns.heatmap(train_df.corr(), annot= True)

# Selecting the appropriate factors from training and test set
# this dataset will be later used in building the Machine Learning Model
X_train = train_df.iloc[:, 1:11]
X_test = test_df.iloc[:, 1:11]
y_train = train_df.iloc[:, 11].values
y_test = y_df.iloc[:,3].values
X_train.head()
# Encoding usinng One Hot Encoder
ohe = ColumnTransformer([('onehotencoder',OneHotEncoder(),[1,3,5,6,7,8,9])], remainder = 'passthrough')
X_train = ohe.fit_transform(X_train).toarray()
X_test = ohe.fit_transform(X_test).toarray()
# Dropping dummy columns to evade dummy variable trap
X_train = np.delete(X_train, [0,2,18,28,37,40,43], axis = 1)
X_test = np.delete(X_test, [0,2,18,28,37,40,43], axis = 1)
# ----------------------------------------- Building Regression Models -------------------------------------------
# Linear Regression
LR_regressor = LinearRegression(normalize=True)
LR_regressor.fit(X_train, y_train)
y_pred_LR = LR_regressor.predict(X_test)

# Model Evaluation (Linear Regression)
mse_LR = metrics.mean_squared_error(y_test, y_pred_LR)
r2_LR = metrics.r2_score(y_test, y_pred_LR)
RMSE_LR = np.sqrt(mse_LR)

print("---------------------- Linear Regression ----------------------\n",
     "Mean Squared Error: ", mse_LR, "\n",
     "R Squared: ", r2_LR, "\n",
     "Root Mean Squared Error: ", RMSE_LR)
# Decision Tree Regression
DT_regressor = DecisionTreeRegressor(random_state=0)
DT_regressor.fit(X_train, y_train)
y_pred_DTR = DT_regressor.predict(X_test)

# Model Evaluation
mse_DTR = metrics.mean_squared_error(y_test, y_pred_DTR)
r2_DTR = metrics.r2_score(y_test, y_pred_DTR)
RMSE_DTR = np.sqrt(mse_DTR)

print("---------------------- Decision Tree Regression ----------------------\n",
     "Mean Squared Error: ", mse_DTR, "\n",
     "R Squared: ", r2_DTR, "\n",
     "Root Mean Squared Error: ", RMSE_DTR)
# Random Forest Regression
RF_regressor = RandomForestRegressor(n_estimators = 300, random_state = 0)
RF_regressor.fit(X_train, y_train)
y_pred_RF = RF_regressor.predict(X_test)

# Model Evaluation
mse_RF = metrics.mean_squared_error(y_test, y_pred_RF)
r2_RF = metrics.r2_score(y_test, y_pred_RF)
RMSE_RF = np.sqrt(mse_RF)

print("---------------------- Random Forest Regression ----------------------\n",
     "Mean Squared Error: ", mse_RF, "\n",
     "R Squared: ", r2_RF, "\n",
     "Root Mean Squared Error: ", RMSE_RF)

# Exporting results to csv File
results = {
            'Item_Identifier': test_df.Item_ID,
            'Outlet_Identifier': test_df.Out_ID,
            'Item_Outlet_Sales': y_pred_LR
        }
results = pd.DataFrame(results)
results.to_csv('Submission_Sid.csv')
