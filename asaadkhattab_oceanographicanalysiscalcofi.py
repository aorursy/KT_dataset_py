from IPython.display import IFrame
embed = "https://docs.google.com/presentation/d/e/2PACX-1vTFUKkshFjS3SpFm392ru6L5CVemXbfU2Op1NaaEFiW16x4Je70wKbiR0u_TcR0UyqOiINeGCNVUquK/embed?"
IFrame(embed,frameborder="0", width="900", align="center", height="569", allowfullscreen="true", mozallowfullscreen="true", webkitallowfullscreen="true")
#Import necessary libraries
import pandas as pd
import numpy as np

# Plotting modules
import matplotlib.pyplot as plt
import seaborn as sns

# Analysing datetime
from datetime import datetime as dt
from datetime import timedelta

# File system manangement
import os,sys

# Suppress warnings 
import warnings
warnings.filterwarnings('ignore')

#Interactive Shell
from IPython.core.interactiveshell import InteractiveShell  
InteractiveShell.ast_node_interactivity = "all"

#Pandas profiling
from pandas_profiling import ProfileReport

import missingno as msno
import re 

%matplotlib inline
cwd = os.getcwd()
print(cwd)
os.listdir(cwd)
#os.listdir( os.getcwd() )
#KAGGLE.com
path = '/kaggle/input/calcofi/bottle.csv'
bottle = pd.read_csv(path) 
# Import CSV file and read the dataset
#path = '../data/calcofi/bottle.csv'
#bottle = pd.read_csv(path, encoding='latin-1') 
# Show all columns
pd.set_option('max_columns', None)
bottle.head()
bottle.tail(3)
# Convert Celcius to Fahr
def cel_to_fahr(x):
    x = x * 1.8 + 32
    return float(x)
# Dimensions of the dataset. #(samples,features)
print("There are", bottle.shape[0], "Rows(Observations).")
print("There are", bottle.shape[1], "Columns(Features).")
bottle.shape
newfolder = "../OceanographicAnalysisCalCOFI/data"

try:
    os.mkdir(newfolder)
except OSError:
    print ("Creation of the directory %s failed" % newfolder)
else:
    print ("Successfully created the directory %s " % newfolder)
newfolder = "../OceanographicAnalysisCalCOFI/figures"

try:
    os.mkdir(newfolder)
except OSError:
    print ("Creation of the directory %s failed" % newfolder)
else:
    print ("Successfully created the directory %s " % newfolder)
newfolder = "../OceanographicAnalysisCalCOFI/models"

try:
    os.mkdir(newfolder)
except OSError:
    print ("Creation of the directory %s failed" % newfolder)
else:
    print ("Successfully created the directory %s " % newfolder)
# Get DataFrame Information
bottle.info()
bottle.dtypes.value_counts()
print(bottle.columns)

#Counts and percentage of null values 
dictionary = {
    "NullCount":bottle.isnull().sum().sort_values(ascending=False),
    "NullPercent":bottle.isnull().sum().sort_values(ascending=False)/len(bottle)*100
}

na_df = pd.DataFrame(dictionary)
na_df.columns = ['NullCount','NullPercent']
na_df[(na_df['NullCount'] > 0)].reset_index()

pct_null = bottle.isnull().sum() / len(bottle)
missing_features = pct_null[pct_null > 0.19].index
bottle.drop(missing_features, axis=1, inplace=True)
df = bottle
# Visualize Missingness
msno.matrix(df)
plt.show()
print ( df.nunique() / df.shape[0] * 100 )
df.head()
df.columns
df = df.rename(columns = { 
    "Cst_Cnt": "CastCount",
    "Btl_Cnt": "BottleCount",
    "Depthm": "DepthMeters",
    "T_degC": "TempDegC",
    "Salnty": "Salinity",    
    "STheta": "PDensity"
    
})
df = df.set_index('BottleCount')
# Extract Year
search = []    

for values in df['Depth_ID']:
    search.append(re.search(r'\d{2}-\d{2}', values).group())
    
df['Year'] = search
df['Year'] = df['Year'].replace(to_replace='-',value='', regex = True) 

df['Year'] = pd.to_datetime(df['Year']).values.astype('datetime64[Y]')
df['Year'] =  pd.DatetimeIndex(df['Year']).year
# Extract Month 
search = []    

for values in df['Depth_ID']:
    search.append(re.search(r'-\d+', values).group())
    
df['Month'] = search
df['Month'] = df['Month'].str[-2:]

df['Month'] = df['Month'].astype('int64')
df['TempDegF'] = df['TempDegC'].apply(cel_to_fahr)
df = df.drop("TempDegC", axis = 1)
print('Salinity:', df.Salinity.unique() ) 
print('TempDegF:', df.TempDegF.unique() ) 
df.describe(include="all").T
range = df.aggregate([min, max])
print(range)
sns.boxplot(x='Year',data=df)
df['Year'] = df['Year'].drop(df[df['Year']>2020].index)
df['Month'] = df['Month'].drop(df[df['Month']>12].index)
df["Salinity"].describe(include="all").T
df["TempDegF"].describe(include="all").T
duplicateRowsDF = df.duplicated() 
df[duplicateRowsDF]
df.drop_duplicates(inplace=True)
df.shape
dfo = df.select_dtypes(include=['object'], exclude=['datetime'])
dfo.shape
#get levels for all variables
vn = pd.DataFrame(dfo.nunique()).reset_index()
vn.columns = ['VarName', 'LevelsCount']
vn.sort_values(by='LevelsCount', ascending = False)
df = df.drop(['Depth_ID','Sta_ID'],axis=1)
corr = df.corr()

plt.figure(figsize=(20,10))
sns.heatmap(corr,
            linecolor='blue',linewidths=.1, 
            cmap="YlGnBu", annot=True)
plt.yticks(rotation=0);
df = df.drop(['CastCount','R_Depth','R_TEMP', 'R_SALINITY', 'C14A1q', 'C14A2q', 'DarkAq' ], axis=1)
from sklearn.preprocessing import StandardScaler

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
numeric_transformer = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
])

numeric_features = df.select_dtypes(include=['int64', 'float64']).columns

preprocessor = ColumnTransformer(
    remainder='passthrough',
    transformers=[
        ('num', numeric_transformer, numeric_features),
    ]
)

steps = [('preprocessor', preprocessor)]

pipeline = Pipeline(steps)

pipeline.fit(df[:])
df_pipe = pipeline.transform(df[:])
df_pipe = pd.DataFrame(df_pipe)
df_pipe
#Change the name of columns back to original names.
df_pipe.columns = df.columns

df_pipe.sample()
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
column_names = ["Salinity","TempDegF"]
sal_temp = df.reindex(columns=column_names)
sal_temp['Salinity'].fillna((sal_temp['Salinity'].mean()), inplace=True)
sal_temp['TempDegF'].fillna((sal_temp['TempDegF'].mean()), inplace=True)
sal_temp.head()
X = sal_temp.Salinity.values
y = sal_temp.TempDegF.values

# Print the dimensions of X and y before reshaping
print("Dimensions of y before reshaping: {}".format(y.shape))
print("Dimensions of X before reshaping: {}".format(X.shape))

# Reshape X and y
y = y.reshape(-1,1)
X = X.reshape(-1,1)

# Print the dimensions of X and y after reshaping
print("Dimensions of y after reshaping: {}".format(y.shape))
print("Dimensions of X after reshaping: {}".format(X.shape))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 12)
model = LinearRegression()
model.fit(X_train, y_train)

#Predict the Test set results
y_pred = model.predict(X_test)
_= plt.scatter(X_train, y_train, color = 'red')
_= plt.plot(X_train, model.predict(X_train), color = 'blue')
_= plt.title('Temperature vs Salinity (Training set)')
_= plt.xlabel('Salinity')
_= plt.ylabel('Temperature')
_= plt.show()
_= plt.scatter(X_test, y_test, color = 'red')
_= plt.plot(X_train, model.predict(X_train), color = 'blue')
_= plt.title('Temperature vs Salinity (Training set)')
_= plt.xlabel('Salinity')
_= plt.ylabel('Temperature')
_= plt.show()
X = df_pipe.drop(['TempDegF'],axis=1).values
y = df_pipe.TempDegF.values

SEED = 42
TS = 0.30

# Create training and test sets
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size = TS, random_state=SEED)

#Feature Scaling to prevent information leakage
#sc = StandardScaler()
#X_train = sc.fit_transform(X_train)
#X_test = sc.transform(X_test)

print (X_train.shape)
print (y_train.shape)

print (X_test.shape)
print (y_test.shape)
# Create logistic regression model
linreg = LinearRegression()

# Train the model using the training sets
linreg.fit(X_train, y_train)

# Make predictions using the testing set
y_pred = linreg.predict(X_test)

linreg.score(X_test,y_test)

linreg_training_score = round(linreg.score(X_train, y_train) * 100, 2)
linreg_test_score = round(linreg.score(X_test, y_test) * 100, 2)

print('Linear Regression Training Score: \n', linreg_training_score)
print('Linear Regression Test Score: \n', linreg_test_score)

# Compute and print R^2 and RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error: {}".format(rmse))
# 5-fold cross-validation:
cv_scores_5 = cross_val_score(linreg, X, y, cv=5)
print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores_5)))


# 15-fold cross-validation:
cv_scores_15 = cross_val_score(linreg, X, y, cv=15)
print("Average 15-Fold CV Score: {}".format(np.mean(cv_scores_15)))

# 25-fold cross-validation:
cv_scores_25 = cross_val_score(linreg, X, y, cv=25)
print("Average 25-Fold CV Score: {}".format(np.mean(cv_scores_25)))
dtr = DecisionTreeRegressor(random_state=42)
model = dtr.fit(X_train, y_train)

y_pred = model.predict(X_test)


dtr_training_score = round(model.score(X_train, y_train) * 100, 2)
dtr_test_score = round(model.score(X_test, y_test) * 100, 2)

print('Decision Tree Training Score: \n', dtr_training_score)
print('Decision Test Score: \n', dtr_test_score)
rfr = RandomForestRegressor(random_state=0, n_jobs=-1)
model = rfr.fit(X_train, y_train)
y_pred = model.predict(X_test)

rfr_training_score = round(model.score(X_train, y_train) * 100, 2)
rfr_test_score = round(model.score(X_test, y_test) * 100, 2)

print('Random Forest Training Score: \n', rfr_training_score)
print('Random Forest Test Score: \n', rfr_test_score)

# We will look at the predicted prices to ensure we have something sensible.

print(y_pred)

models = pd.DataFrame({
    
    'Model': [ 
        'Linear Regression',
        'Decision Tree',
        'Random Forest',   
    ],
             
    
    'Training Score': [ 
        linreg_training_score,
        dtr_training_score, 
        rfr_training_score,
    ],
    
    'Test Score': [ 
        linreg_test_score,
        dtr_test_score,
        rfr_test_score,
    ]})


models.sort_values(by='Test Score', ascending=False)
df.aggregate([min, max])
#df.to_csv('../data/calcofi/wrangle_csv.csv', index=True)
my_submission = pd.DataFrame({'Id': df_pipe.index, 'Temperature': print(y_pred)})
# you could use any filename. We choose submission here
my_submission.to_csv('submission.csv', index=False)