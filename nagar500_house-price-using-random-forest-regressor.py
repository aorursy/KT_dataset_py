import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns


import os
print(os.listdir("../input"))


# Any results you write to the current directory are saved as output.
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')

test_df_original = test_df.copy()
test_df.head()
print("training data", train_df.shape)
print("training data", test_df.shape)
train_df.info()
train_df['SalePrice'].describe()
def explore(df):
    numerical = ['GrLivArea' , 'TotalBsmtSF' ]
    for feature in  numerical:
        data = pd.concat([df['SalePrice'], df[feature]], axis=1)
        data.plot.scatter(x=feature, y='SalePrice', ylim=(0,800000));
    
explore(train_df)

def explore(df):
    categorical = ['OverallQual' , 'YearBuilt', 'BsmtCond','BsmtCond']
    for feature in  categorical:
        data = pd.concat([df['SalePrice'], df[feature]], axis=1)
        f, ax = plt.subplots(figsize=(16, 8))
        fig = sns.boxplot(x=feature, y="SalePrice", data=data)
        fig.axis(ymin=0, ymax=800000);
    
explore(train_df)
def heatMap(df):

    corr = df.corr()
    plt.subplots(figsize=(20,12))
    sns.heatmap(corr, vmax=0.9, square=True)
    

    
heatMap(train_df)
# least correlated features
corr = train_df.corr()
least_corr_features = corr.index[abs(corr["SalePrice"])<0.25]
plt.figure(figsize=(10,10))
g = sns.heatmap(train_df[least_corr_features].corr(),annot=True,cmap="RdYlGn")

## We will be deleting these features from our data set as they do not help much in making predictions
least_corr_features
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', '1stFlrSF', 'YearBuilt', 'FullBath']
sns.pairplot(train_df[cols], size = 2.0)
plt.show();
plt.figure(figsize = (12, 6))
sns.boxplot(x = 'Neighborhood', y = 'SalePrice',  data = train_df)
xt = plt.xticks(rotation=45)
#missing data

def missing_data_total(df):
    total = df.isnull().sum().sort_values(ascending=False)
    percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    return missing_data.head(20)

print("missing training data",missing_data_total(train_df))
print("missing testing data" ,missing_data_total(test_df))    
    

# defining a function to perform transformation

def delete(df):
    
    # we will be deleting useless and least correlated features from our dataset
    useless = [ 'MasVnrType', 'GarageYrBlt','GarageArea' ,'TotalBsmtSF' , 'TotRmsAbvGrd','MSSubClass', 'OverallCond', 'BsmtFinSF2', 'BsmtUnfSF',
       'LowQualFinSF', 'BsmtFullBath', 'BsmtHalfBath', 'BedroomAbvGr',
       'KitchenAbvGr', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea',
       'MiscVal', 'MoSold', 'YrSold']
    
    for feature in useless: 
        df.drop([feature], axis = 1, inplace = True)
        
    return df

def fill_categorical(df):
    
    fill = ['PoolQC', 'MiscFeature','Alley', 'Fence','FireplaceQu','GarageType', 'GarageFinish', 'GarageQual', 'GarageCond','BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
    
    for feature in fill:
        #filling these features with "None"
        
        df[feature] = df[feature].fillna("None")
        
    return df

def fill_numerical(df):
    
    
    fill = ['GarageCars','MasVnrArea','BsmtFinSF1']
    
    for feature in fill:
        
        #filling these features with "None"
        
        df[feature] = df[feature].fillna(0)
        
    return df

def fill_mode(df):
    
    fill=['KitchenQual', 'Electrical','MSZoning',"Utilities", "Functional","Exterior1st", "Exterior2nd"]
    
    for feature in fill:
        df[feature] = df[feature].fillna(df[feature].mode()[0])
        
    return df

# now we will fill lot frontage

def fill_lot(df):
    
    df["LotFrontage"] = df.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))
    return df
    
def transform(df):
    
    df = delete(df)
    df = fill_categorical(df)
    df = fill_numerical(df)
    df = fill_mode(df)
    df = fill_lot(df)
    return df
    
    
train_df = transform(train_df)
test_df = transform(test_df)


train_df.head()

train_df.describe(include=['O'])

from sklearn.preprocessing import LabelEncoder

def encoder(df):
    cols = ( 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC',  'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'CentralAir','PoolQC')
    for feature in cols:
        label = LabelEncoder() 
        label.fit(list(df[feature].values)) 
        df[feature] = label.transform(list(df[feature].values))
    
    print('Shape data: {}'.format(df.shape))
  
    return df

train_df = encoder(train_df)
test_df = encoder(test_df)
 
######### Total area of the house   ######## As basement was correlated to 1stfloor area, it was deleted earlier. 

def new_feature(df):
    df['TotalSF'] = df['1stFlrSF'] + df['2ndFlrSF']
    df.drop(['1stFlrSF', '2ndFlrSF'], axis=1, inplace=True)
    return df
train_df= new_feature(train_df)
test_df = new_feature(test_df)

train_df.head()

# We will drop these 2 features from data frame and include a new feature instead
from scipy import stats
from scipy.stats import norm, skew 

ntrain = train_df.shape[0]
ntest = test_df.shape[0]
y_train = train_df.SalePrice.values

#train_df = train_df.drop

all_data = pd.concat([train_df, test_df]).reset_index(drop=True)
all_data.drop(['SalePrice'], axis=1, inplace=True)
print("all_data size is : {}".format(all_data.shape))


skewed_feature = all_data.dtypes[all_data.dtypes != "object"].index

# Check the skew of all numerical features
skewed_feats = all_data[skewed_feature].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
print("\nSkew in numerical features for all the data (training + testing): \n")
skewness = pd.DataFrame({'Skew' :skewed_feats})
print(skewness.head(10))

from scipy.special import boxcox1p

# We are using Box Cox and not Logarithmic transformation, because we have imputes some features to 0 vale. So logarithmic 
#transformation of these features will not be possible 
skewness = skewness[abs(skewness) > 0.75]
print("There are {} skewed numerical features to  transform".format(skewness.shape[0]))
skewed_features = skewness.index
  
lam = 0.15 
for transform in skewed_features:
    all_data[transform] = boxcox1p(all_data[transform], lam)
      

    
all_data = pd.get_dummies(all_data)
print(all_data.head())
### Splitting features
train = all_data[:ntrain]
test = all_data[ntrain:]

print("Training shape", train.shape)
print("Testing shape", test.shape)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train, y_train, test_size = 0.3, random_state = 200)


from sklearn.ensemble import RandomForestRegressor
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import make_scorer, mean_squared_error

scorer = make_scorer(mean_squared_error, False)
parameters = {'n_estimators' : [100,150,200,250,300] , 'max_features' : [0.2, 0.1,0.3,0.05], 'min_samples_split' :[2,3,4]}

clf = RandomForestRegressor(random_state = 42, n_jobs = -1)
grid_obj = GridSearchCV(clf, parameters, scoring =  scorer)

#Fit the grid search object to the training data and find the optimal parameters using fit()
grid_fit = grid_obj.fit(X_train, y_train)

# Get the estimator
best_clf = grid_fit.best_estimator_

print(best_clf)

# Make predictions using the unoptimized and model
predictions = (clf.fit(X_train, y_train)).predict(X_test)
best_predictions = best_clf.predict(X_test)

print("\nUnOptimized Model\n------")
print("Final rmse score on the testing data: {:.4f}".format(mean_squared_error(y_test, predictions)))

print("\nOptimized Model\n------")
print("Final rmse score on the testing data: {:.4f}".format(mean_squared_error(y_test, best_predictions)))




pred_test = best_clf.predict(test)
submission = pd.read_csv('../input/sample_submission.csv')
submission['SalePrice']=pred_test
submission['Id']=test_df_original['Id']

#converting to csv

pd.DataFrame(submission, columns=['Id','SalePrice']).to_csv('randomforesthouse.csv')