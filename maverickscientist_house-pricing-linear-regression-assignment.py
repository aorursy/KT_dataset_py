import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
import datetime 
from sklearn.feature_selection import RFE
from sklearn.metrics import r2_score
from sklearn.impute import SimpleImputer
import os

# Hide warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning) 
# Reading the dataset
houseFilePath = '../input/train.csv'
house_df = pd.read_csv(houseFilePath,encoding = "ISO-8859-1", low_memory=False)
house_df.head()
# Let's check the dataset for null values in columns
house_df.info()
# For the year columns, let's convert them into present age to make them numeric and more easily understandable

current_year = int(datetime.datetime.now().year)
house_df['YearBuilt_Age'] = current_year-house_df.YearBuilt
house_df['YearRemodAdd_Age'] = current_year - house_df.YearRemodAdd
house_df['GarageYrBlt_Age'] = current_year - house_df.GarageYrBlt
house_df['YrSold_Age'] = current_year - house_df.YrSold

house_df[['YearBuilt','YearRemodAdd','GarageYrBlt','YrSold','YearBuilt_Age','YearRemodAdd_Age',
             'GarageYrBlt_Age','YrSold_Age']].sample(10)
# Let's drop these columns as we have got their Ages now.
house_df.drop(['YearBuilt', 'YearRemodAdd','GarageYrBlt','YrSold'], axis = 1, inplace = True)

house_df.shape
house_df['MSSubClass'].astype('object')
house_df['OverallQual'].astype('object')
house_df['OverallCond'].astype('object')
# Let's map the values in these columns to their string counterparts.
MSSubClassMapping = {
    20:"1-STORY 1946 & NEWER ALL STYLES",
    30:"1-STORY 1945 & OLDER",
    40:"1-STORY W/FINISHED ATTIC ALL AGES",
    45:"1-1/2 STORY - UNFINISHED ALL AGES",
    50:"1-1/2 STORY FINISHED ALL AGES",
    60:"2-STORY 1946 & NEWER",
    70:"2-STORY 1945 & OLDER",
    75:"2-1/2 STORY ALL AGES",
    80:"SPLIT OR MULTI-LEVEL",
    85:"SPLIT FOYER",
    90:"DUPLEX - ALL STYLES AND AGES",
    120:"1-STORY PUD (Planned Unit Development) - 1946 & NEWER",
    150:"1-1/2 STORY PUD - ALL AGES",
    160:"2-STORY PUD - 1946 & NEWER",
    180:"PUD - MULTILEVEL - INCL SPLIT LEV/FOYER",
    190:"2 FAMILY CONVERSION - ALL STYLES AND AGES"
}
house_df = house_df.replace({"MSSubClass": MSSubClassMapping })
house_df.MSSubClass.value_counts()
# Mappings for OverallCond & OverallQual are exactly same. Let's map them too.
OverallMapping = {
    
      10:"Very Excellent",
       9:"Excellent",
       8:"Very Good",
       7:"Good",
       6:"Above Average",
       5:"Average",
       4:"Below Average",
       3:"Fair",
       2:"Poor",
       1:"Very Poor"
}
house_df = house_df.replace({"OverallQual": OverallMapping })
house_df = house_df.replace({"OverallCond": OverallMapping })
house_df.OverallQual.value_counts()

house_df.OverallCond.value_counts()
sns.set(style="ticks", color_codes=True)
g = sns.pairplot(house_df,vars=["MSSubClass", "SalePrice"])
sns.set(style="ticks", color_codes=True)
g = sns.pairplot(house_df,vars=["OverallQual", "SalePrice"])
sns.set(style="ticks", color_codes=True)
g = sns.pairplot(house_df,vars=["OverallCond", "SalePrice"])
#Let's analyze columns like Street &  Utilities for their value occurances
house_df.Street.value_counts()
house_df.Utilities.value_counts()
# Clearly, from analyses done above, 'Utilities' & 'Street' have no significance as just one value is present in maximum number of records.
# Let's drop these columns and the column 'Id'.
house_df.drop(['Street', 'Utilities','Id'], axis = 1, inplace = True)
# We also see that the column "Alley" has only 91 non null values. Let's analyze all such columns where
# the number of missing values per column is more than 80 %. In such cases, it is better to drop them.
round(house_df.isnull().sum()/len(house_df.index),2)[round(house_df.isnull().sum()/
                                                                 len(house_df.index),2).values>0.80]
# Let's remove the ablove columns since they are hardly going to have any impact on our modelling
house_df.drop(['Alley', 'PoolQC','Fence','MiscFeature'], axis = 1, inplace = True)
house_df.info()
categorical_columns = []
numeric_columns = []
for c in house_df.columns:
    if house_df[c].map(type).eq(str).any(): #check if there are any strings in column
        categorical_columns.append(c)
    else:
        numeric_columns.append(c)
# Let's list down all categorical columns
categorical_columns
# Let's list down all numerial columns
numeric_columns
#create two DataFrames, one for each data type
data_numeric = house_df[numeric_columns]
data_categorical = house_df[categorical_columns]

# fill the numerical columns with the mean
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
data_numeric = pd.DataFrame(imp.fit_transform(data_numeric), columns = data_numeric.columns)

# and the categorical columns with 'NA'
imp = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value='NA')
data_categorical = pd.DataFrame(imp.fit_transform(data_categorical), columns = data_categorical.columns) 

#join the two masked dataframes back together
house_df = pd.concat([data_numeric, data_categorical], axis = 1)

# we will plot all the numeric variables one by one through a loop

for feature, column in enumerate (data_numeric.columns):
    plt.figure(feature)
    sns.scatterplot(x=data_numeric[column],y=data_numeric['SalePrice'])
corr=data_numeric.corr()

plt.figure(figsize=(20,15))
sns.heatmap(corr,annot=True,cmap="YlGnBu")
house_df.info()
house_df.head(20)
# Below function is used for hadling the outliers where I am taking the lower and upper quantile as 0.25 & 0.99 respectively
def drop_outliers(x):
    list = []
    for col in data_numeric:
        Q1 = x[col].quantile(.25)
        Q3 = x[col].quantile(.99)
        IQR = Q3-Q1
        x =  x[(x[col] >= (Q1-(1.5*IQR))) & (x[col] <= (Q3+(1.5*IQR)))] 
    return x   

house_df = drop_outliers(house_df)

house_df.shape
# Let's create X & y datasets for further analysis.
X=house_df.drop(columns=['SalePrice'])
y=house_df['SalePrice']
X.info()
# Let's check the distribution of SalePrice column
plt.figure(figsize=(16,6))
sns.distplot(house_df.SalePrice)
plt.show()
# We see that its not perfectly normalized, rather, it's skewed. Let's transform it to use the log value.
y_log = np.log(house_df.SalePrice + 1)
y_log.describe()
# Lets normalize it now.
def normalize(column):
    upper = column.max()
    lower = column.min()
    y = (column - lower)/(upper-lower)
    return y

y_log_normalized = normalize(y_log)
y_log_normalized.describe()
# Let's plot it to see if it has improved on a log scale.
plt.figure(figsize=(16,6))
sns.distplot(y_log_normalized)
plt.show()
y_log_normalized.describe()
# creating dummy variables from the list of categorical variables just created.
dummy_vars_df = pd.get_dummies(data_categorical, drop_first=True)
dummy_vars_df.head()
# since we have the dummy variables now, get can drop the categorical variables from X.
X=X.drop(columns=data_categorical)
X.head()
# add the dummy variables to X.
X=pd.merge(X,dummy_vars_df, left_on=X.index, right_on=dummy_vars_df.index)
X.head()
X.columns
# get rid of the 'key_0' column after the merge.
X.drop(columns='key_0',inplace=True)
X.info()
# scaling the features
from sklearn.preprocessing import scale

# storing column names in cols, since column names are (annoyingly) lost after 
# scaling (the df is converted to a numpy array)
cols = X.columns
X = pd.DataFrame(scale(X))
X.columns = cols
X.columns
# split into train and test

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y_log_normalized, 
                                                    train_size=0.7,
                                                    test_size = 0.3, random_state=100)
# We will use numpy.linspace to create the alpha values with equal steps
alphaArray = np.linspace(1, 1000, num=50)
# list of alphas to tune

params = {'alpha': alphaArray}


ridge = Ridge()

# cross validation
folds = 5
model_cv = GridSearchCV(estimator = ridge, 
                        param_grid = params, 
                        scoring= 'neg_mean_absolute_error', 
                        cv = folds, 
                        return_train_score=True,
                        verbose = 1)            
model_cv.fit(X_train, y_train) 
cv_results = pd.DataFrame(model_cv.cv_results_)
cv_results = cv_results[cv_results['param_alpha']<=1000]
cv_results.head()
# plotting mean test and train scoes with alpha 
cv_results['param_alpha'] = cv_results['param_alpha'].astype('int32')

# plotting
plt.plot(cv_results['param_alpha'], cv_results['mean_train_score'])
plt.plot(cv_results['param_alpha'], cv_results['mean_test_score'])
plt.xlabel('alpha')
plt.ylabel('Negative Mean Absolute Error')
plt.title("Negative Mean Absolute Error and alpha")
plt.legend(['train score', 'test score'], loc='upper right')
plt.show()
# Let's loop through different values of alpha to see the affect on r2 scores
for a in alphaArray:
    ridge = Ridge(alpha=a)
    ridge.fit(X_train, y_train)
    # Let's predict the train & test datasets now to see the 'r2 scores'.
    y_pred_train = ridge.predict(X_train)
    y_pred_test = ridge.predict(X_test)
    print('Alpha:{:1} |train_R2:{:2} |test_R2:{:3}'.format(a, r2_score(y_train,y_pred_train), r2_score(y_test,y_pred_test)))
# Above analysis puts the optimum value of alpha as 225.26. Let's go ahead and use it to see how the coefficients change.
alpha = 225.26
ridge = Ridge(alpha=alpha)
ridge.fit(X_train, y_train)
ridge.coef_
# Let's predict the train & test datasets now to see the 'r2 scores'.
y_pred_train = ridge.predict(X_train)
print(r2_score(y_train,y_pred_train))

y_pred_test = ridge.predict(X_test)
print(r2_score(y_test,y_pred_test))
# Let's see the coefficients in descending order of their values obtained after Ridge Regression.
model_parameter = list(ridge.coef_)
model_parameter.insert(0,ridge.intercept_)
cols = X_train.columns
cols.insert(0,'constant')
ridge_coef = pd.DataFrame(list(zip(cols,model_parameter)))
ridge_coef.columns = ['Feature','Coef']
ridge_coef.sort_values(by='Coef',ascending=False).head(10)
# We will use numpy.linspace to create the alpha values with equal steps.
# Note, for Lasso we have reduced the range for Alpha as it reduces coefficient value to 0 as we increase
# and we do not want over-penalization.

alphaArray = np.linspace(0.001, 1, num=50)

# list of alphas to tune

params = {'alpha': alphaArray}

lasso = Lasso()

# cross validation
model_cv = GridSearchCV(estimator = lasso, 
                        param_grid = params, 
                        scoring= 'neg_mean_absolute_error', 
                        cv = folds, 
                        return_train_score=True,
                        verbose = 1)            

model_cv.fit(X_train, y_train) 
cv_results = pd.DataFrame(model_cv.cv_results_)
cv_results.head()
# plotting mean test and train scoes with alpha 
cv_results['param_alpha'] = cv_results['param_alpha'].astype('float32')

# plotting
plt.plot(cv_results['param_alpha'], cv_results['mean_train_score'])
plt.plot(cv_results['param_alpha'], cv_results['mean_test_score'])
plt.xlabel('alpha')
plt.ylabel('Negative Mean Absolute Error')
plt.xscale('log')
plt.title("Negative Mean Absolute Error and alpha")
plt.legend(['train score', 'test score'], loc='upper right')
plt.show()
# Let's loop through different values of alpha to see the affect on r2 scores
for a in alphaArray:
    lasso = Lasso(alpha=a)
    lasso.fit(X_train, y_train)
    y_train_pred = lasso.predict(X_train)
    y_test_pred = lasso.predict(X_test)
    print('Alpha:{:1} |train_R2:{:2} |test_R2:{:3}'.format(a, r2_score(y_train,y_train_pred), r2_score(y_test,y_test_pred)))
# This puts alpha at 0.001, let's use this value.
alpha =0.001

lasso = Lasso(alpha=alpha)
        
lasso.fit(X_train, y_train)
lasso.coef_
from sklearn.feature_selection import RFE
from sklearn.metrics import r2_score

y_train_pred = lasso.predict(X_train)
y_test_pred = lasso.predict(X_test)

print(r2_score(y_true=y_train,y_pred=y_train_pred))
print(r2_score(y_true=y_test,y_pred=y_test_pred))
model_param = list(lasso.coef_)
model_param.insert(0,lasso.intercept_)
cols = X_train.columns
cols.insert(0,'const')
lasso_coef = pd.DataFrame(list(zip(cols,model_param)))
lasso_coef.columns = ['Featuere','Coef']
lasso_coef.sort_values(by='Coef',ascending=False).head(10)
