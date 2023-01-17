import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
import matplotlib.pyplot as plt
import seaborn as sns

# Any results you write to the current directory are saved as output.
column_headers = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
boston = pd.read_csv('../input/housingdata.csv', names = column_headers)
print('Dataset Shape', boston.shape)
boston.head()
sns.set_style('whitegrid')
MEDV = sns.distplot(boston['MEDV'])
MEDV.set(title = "Distribution of MEDV column")
boston['MEDV'].describe()
#Calculate all missing values in the dataset.
missing_values = boston.isnull().sum().sum()
print("Missing values in dataset: ", missing_values)
boston.info()
f, ax = plt.subplots(nrows = 7, ncols = 2, figsize=(16,16))
columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX',
       'PTRATIO', 'B', 'LSTAT']
row = 0
col = 0
for i, column in enumerate(columns):
    g = sns.distplot(boston[column], ax=ax[row][col])
    col += 1
    if col == 2:
        col = 0
        row += 1

plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=.5)

boston.iloc[:,:-1].describe()
rad_out = boston.copy()
rad_out['OUTLIER'] = rad_out['RAD'].apply(lambda x: 1 if x > 15 else 0)

sns.boxplot(x='OUTLIER', y='MEDV', data=rad_out)
rad_out.groupby('OUTLIER').mean()['MEDV']
tax_out = boston.copy()
tax_out['OUTLIER'] = boston['TAX'].apply(lambda x: 1 if x > 600 else 0)
sns.boxplot(x='OUTLIER', y='MEDV', data=tax_out)
tax_out.groupby('OUTLIER').mean()['MEDV']


boston.groupby('ZN').count()
zn = boston.copy()
zn['BINNED'] = pd.cut(zn['ZN'], bins = 4)
zn.head()
zn_grouped = zn.groupby('BINNED').mean()['MEDV']
zn_grouped
plt.figure(figsize=(8,8))
plt.bar(zn_grouped.index.astype(str), zn_grouped)
#Sort correlations
correlations = boston.corr()['MEDV'].sort_values()
correlations
sns.lmplot(x='RM', y='MEDV', data=boston)
sns.lmplot(x='LSTAT', y='MEDV', data=boston)
sns.pairplot(data=boston)
plt.figure(figsize=(8,8))
sns.heatmap(boston.corr())
#Imports to calculate VIF's for each predictor
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor 
from patsy import dmatrices
#gather features
features = "+".join(boston.columns[:-1])

# get y and X dataframes based on this regression:
y, X = dmatrices('MEDV ~' + features, boston, return_type='dataframe')
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns
vif.sort_values(by='VIF Factor', ascending=False).iloc[1:,:]
tax_rad = boston.copy()
tax_rad['taxrad'] = tax_rad['TAX'] + tax_rad['RAD']
tax_rad = tax_rad.drop(['TAX', 'RAD'], axis=1)
#gather features
features = "+".join(['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'taxrad',
       'PTRATIO', 'B', 'LSTAT'])

# get y and X dataframes based on this regression:
y, X = dmatrices('MEDV ~' + features, tax_rad, return_type='dataframe')
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns
vif.sort_values(by='VIF Factor', ascending=False).iloc[1:,:]
##Dataframe to capture polynomial features
poly_features = boston.copy()

#Capture target variable
poly_target = poly_features['MEDV']
poly_features = poly_features.drop(columns=['MEDV'])

#Import polynomial feature module
from sklearn.preprocessing import PolynomialFeatures

#Create polynomial object with degree of 2
poly_transformer = PolynomialFeatures(degree = 2)

#Train the polynomial features
poly_transformer.fit(poly_features)

#Transform the features
poly_features = poly_transformer.transform(poly_features)

print('Polynomial Features Shape: ', poly_features.shape)
#Create dataframe of features.
poly_features = pd.DataFrame(poly_features, columns = poly_transformer.get_feature_names(boston.columns[:-1]))

#Add target back in to poly_features
poly_features['MEDV'] = poly_target

#Find correlations within target
poly_corrs = poly_features.corr()['MEDV'].sort_values()

print(poly_corrs.head(10))
print(poly_corrs.tail(10))
manual_features = boston.copy()
manual_features['TAX_OUT'] = manual_features['TAX'].apply(lambda x: 1 if x > 600 else 0)
manual_features['RAD_OUT'] = manual_features['RAD'].apply(lambda x: 1 if x > 15 else 0)
manual_features.head()
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
#Function to fit, train, and test linear regression model.
def basicLR(data):
    #X Set
    X = data.drop(columns='MEDV')
    
    #Y set
    y = data['MEDV']
    
    #Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
    
    #Create linear model object
    lm = LinearRegression()
    
    #Fit linear object model to training data
    lm.fit(X_train, y_train)
    
    #Make predictions using lm.predict
    predictions = lm.predict(X_test)
    
    #Print model quality of fit scores.
    print('r^2: ', r2_score(y_test, predictions))
    print("MSE: ", mean_squared_error(y_test, predictions))
    
    return r2_score(y_test, predictions), mean_squared_error(y_test, predictions)
#original boston data results
bostonr2, bostonMSE = basicLR(boston)
#Polynomial feature results
polyr2, polyMSE = basicLR(poly_features)
basicLR_frame = pd.DataFrame(data=[[bostonr2, polyr2], [bostonMSE, polyMSE]], columns=['Baseline', 'Polynomial'], index=['r^2', 'MSE'])
basicLR_frame
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
#Function to fit, train, and test linear regression model.
def RidgeLR(data):
    #X Set
    X = data.drop(columns='MEDV')
    
    #Y set
    y = data['MEDV']
    
    #Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
    
    #Alphas to tune
    alphas = {'alpha':[.001, .01, .1, 10, 100]}
    
    #Create Ridge object
    ridge = Ridge(random_state = 101)
    
    #Create ridge model
    clf = GridSearchCV(ridge, alphas)
    
    #Fit linear object model to training data
    clf.fit(X_train, y_train)
    
    #Make predictions using lm.predict
    predictions = clf.predict(X_test)
    
    #Print model quality of fit scores.
    print('r^2: ', r2_score(y_test, predictions))
    print("MSE: ", mean_squared_error(y_test, predictions))
    
    return r2_score(y_test, predictions), mean_squared_error(y_test, predictions)
bridger2, bridgemse = RidgeLR(boston)
polyridger2, polyridgemse = RidgeLR(poly_features)
RidgeLR = pd.DataFrame(data=[[bridger2, polyridger2], [bridgemse, polyridgemse]], columns=['boston', 'poly features'], index=['r^2', 'MSE'])
RidgeLR
