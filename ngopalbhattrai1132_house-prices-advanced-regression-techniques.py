# Importing all the Dependencies

import pandas as pd

import numpy as np     

import matplotlib.pyplot as plt

import seaborn as sns



pd.options.display.max_columns = 81
# Loading the DataSet

Train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

Test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

Train.head()
# Understanding about the dataset

Train.info()

Train.describe()
# Lets Analyze the Integer columns

Train.select_dtypes(np.int64).nunique().value_counts().sort_index().plot.bar(color = 'red',figsize = (15,8),edgecolor = 'black')

plt.xlabel('No. Of Unique Values')

plt.ylabel('Count')

plt.title('Count number of Unique Values')
# Lets Analyze the Float Columns



plt.figure(figsize = (15,8))

Color = {1:'blue',2:'red',3:'green'}

for i,col in enumerate(Train.select_dtypes(np.float64)):

    ax = plt.subplot(1,3,i+1)

    sns.kdeplot(Train.loc[:,col],color = Color[i+1],ax = ax)

    plt.xlabel(f'{col}')

    plt.ylabel('Density')

    plt.title(f'{col.capitalize()} Distribution')

    
Object_Col = Train.select_dtypes('object')





# Merge The Dataset so that the Operation that we do happens in both

Test['SalePrice'] = np.nan

Data = Train.append(Test,ignore_index = True)



Object_Col
# Lets Check the NULL Values

Null_Values = pd.DataFrame(Data.isnull().sum()/Data.shape[0]).rename(columns = {0:'Total'}).sort_values('Total',ascending = False)

Null_Values.head(50)



# Filling the NULL VALUES



# These Two Lines Shows that PoolQC null values can be filled by NA

Data.PoolQC.isnull().sum()

Data.PoolArea.value_counts().plot.bar(color = 'Red',edgecolor = 'black')

plt.xlabel('Frequency')

plt.ylabel('Count')

plt.title('PoolArea')



Data.PoolQC.fillna('NA',inplace = True)

# These Two Lines Shows that MiscFeature null values can be filled by NA



print(Data.MiscFeature.isnull().sum(),end = '\n\n')

print(Data.MiscFeature.value_counts())



Data.MiscFeature.value_counts().plot.bar(color = 'Red',edgecolor = 'black')

plt.xlabel('Frequency')

plt.ylabel('Count')

plt.title('MiscFeature')





print(Data.MiscVal.value_counts().head())

Data.MiscFeature.fillna('NA',inplace = True)
# Here I am taking a Hunch and filling null values with NA. As there is no information provied in Documentation



print(Data.Alley.isnull().sum(),end = '\n\n')

print(Data.Alley.value_counts())



Data.Alley.value_counts().plot.bar(color = 'Red',edgecolor = 'black')

plt.xlabel('Frequency')

plt.ylabel('Count')

plt.title('Alley')



Data.Alley.fillna('NA',inplace = True)

# Here I am taking a Hunch and filling null values with NA. As there is no information provied in Documentation



print(Data.Fence.isnull().sum(),end = '\n\n')

print(Data.Fence.value_counts())



Data.Fence.value_counts().plot.bar(color = 'Red',edgecolor = 'black')

plt.xlabel('Frequency')

plt.ylabel('Count')

plt.title('Fence')



Data.Fence.fillna('NA',inplace = True)

# Filling with Gd here



print(Data.FireplaceQu.isnull().sum(),end = '\n\n')

print(Data.FireplaceQu.value_counts())



Data.FireplaceQu.value_counts().plot.bar(color = 'Red',edgecolor = 'black')

plt.xlabel('Frequency')

plt.ylabel('Count')

plt.title('FireplaceQu')



Data.FireplaceQu.fillna('Gd',inplace = True)
for col in Data.select_dtypes('object'):

    Data.loc[:,col] = pd.factorize(Data.loc[:,col])[0]

    Data.loc[Data[col] == -1,col] = np.nan



# Rest of the NULL Values will be filled during Imputation.
# Lets do Feature Engineering

# First Let's create the Feature Subset and study every subset in detail to do Feature Engineering



from collections import Counter



Id_ = ['Id']



Property_ = ['MSSubClass', 'MSZoning','LotFrontage', 'LotArea','LotShape','LotConfig','LandContour','LandSlope','Neighborhood','BldgType'

            ,'HouseStyle','RoofStyle','RoofMatl','Condition1','Condition2']

Basement_ = ['BsmtQual','BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1',

             'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF','BsmtFullBath', 'BsmtHalfBath']



Exterior_ = ['Exterior1st', 'Exterior2nd', 'ExterQual', 'ExterCond']



Vantilation_ = ['Heating','HeatingQC', 'CentralAir', 'Electrical']



Floor_ = ['1stFlrSF', '2ndFlrSF','LowQualFinSF', 'GrLivArea']

    

Bath_ = ['FullBath','HalfBath']



Kitchen_ = ['KitchenAbvGr', 'KitchenQual']



Fire_ = ['Fireplaces', 'FireplaceQu']

    

Garage_ = ['GarageType','GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual','GarageCond']



Porch_ = ['OpenPorchSF','EnclosedPorch', '3SsnPorch', 'ScreenPorch']



Pool_ = ['PoolArea', 'PoolQC']



Misc_ = ['MiscFeature', 'MiscVal']



Sold_ = ['MoSold', 'YrSold','YearBuilt','YearRemodAdd']



Sale_ = ['SaleType','SaleCondition']



MasVnr_ = ['MasVnrType','MasVnrArea']



Overall_ = ['OverallQual', 'OverallCond']



Others_ = ['Fence','PavedDrive', 'WoodDeckSF','Functional','TotRmsAbvGrd','BedroomAbvGr','Foundation','Utilities','Street','Alley']



Target_ = ['SalePrice']



Total = Id_ + Property_ + Basement_ + Exterior_ + Vantilation_ + Floor_ + Bath_ + Kitchen_ + Fire_ + Garage_ + Porch_ + Pool_+ Misc_ + Sold_ + Sale_ + MasVnr_ +  Overall_ + Others_ + Target_

 

Subset = [Property_, Basement_,Exterior_, Vantilation_, Floor_, Bath_, Kitchen_, Fire_, Garage_, Porch_,Pool_, Misc_, Sold_, Sale_, MasVnr_, Overall_, Others_]



print('There are no repeats:',np.all(np.array(list(Counter(Total).values())) == 1))

print('We Covered Every Attribute:',len(Total) == Data.shape[1])  # Because One is Target Variable



# Step 1: Identifying and removing redundant features in every Subset



Redundant_Attribute = []



for subset in Subset:

    Temp_Data = Data[subset]

    Correlation = Temp_Data.corr()

    Col = Correlation.where(np.triu(np.ones(Correlation.shape),k = 1).astype(np.bool))

    Temp = [C for C in Col if np.any(Col[C].abs() > 0.90)]

    #print(f'Redundant Features in {subset} are: {Temp}',end = '\n\n')

    Redundant_Attribute.extend(Temp)

    

    

print('Total Number of Redundant Features are : {0} in Subset\nThose Features are: {1}'.format(len(Redundant_Attribute),Redundant_Attribute))



# And Lets check Redundant as in Whole Data Matrix



Big_Corr = Data.corr()

Upper = Big_Corr.where(np.triu(np.ones(Big_Corr.shape),k = 1).astype(np.bool))

Redundant = [C for C in Upper if np.any(Upper[C].abs() > 0.90)]

print('Redundant Features as a whole are :',Redundant)



# There are not any Redundant Features :( so we need to keep 'em all'
# Lets Study every Subset and See what we can do with respect to Target



# Lets start with Basement_ attribute

Temp = Data[Basement_ + Target_].copy()

Temp['Total/sf1'] = Temp['TotalBsmtSF'] / Temp['BsmtFinSF1']

Temp['Total/sf2'] = Temp['TotalBsmtSF'] / Temp['BsmtFinSF2']

Temp['Total/sf1 and sf2'] = Temp['TotalBsmtSF'] / (Temp['BsmtFinSF1'] * Temp['BsmtFinSF2'])

Temp['Total/sf1 + sf2'] = Temp['TotalBsmtSF'] / (Temp['BsmtFinSF1'] + Temp['BsmtFinSF2'])

Temp['sf1 + sf2'] = Temp['BsmtFinSF1'] + Temp['BsmtFinSF2']

Temp['BsmtQual+BsmtCond'] = Temp['BsmtQual'] + Temp['BsmtCond']

sns.lmplot('BsmtQual','BsmtCond',data = Temp,fit_reg = False)



print(Temp.corr()[Target_[0]])

# By this Data I can say that I could remove BsmtFinSF2 and add sf1 + sf2

Data['sf1 + sf2'] = Temp['BsmtFinSF1'] + Temp['BsmtFinSF2']

Data.drop(['BsmtFinSF2'],inplace = True,axis = 1)



# Lets Study Exterior_ Subset

Temp = Data[Exterior_ + Target_].copy()



#sns.lmplot('ExterQual','ExterCond',data = Temp,fit_reg = False)

sns.lmplot('Exterior1st','Exterior2nd',data = Temp,fit_reg = True)

Temp['Exterior 3'] = Temp['Exterior1st'] + Temp['Exterior2nd']

Temp.corr()[Target_[0]]



# I could remove Exterior 1st and add Exterior 3

Data['Exterior 3'] =Temp['Exterior1st'] + Temp['Exterior2nd']

Data.drop(['Exterior1st'],inplace = True,axis = 1)



# Lets start with Property_ attribute

Temp = Data[Property_ + Target_].copy()

Temp['Condition 3']  = Temp['Condition1'] + Temp['Condition2']

Temp['Roof Style + Roof Material'] = Temp['RoofStyle'] + Temp['RoofMatl']

Temp.corr()[Target_[0]]

# So we could remove Condition 2 and Roof Material

Data['Condition 3']  = Temp['Condition1'] + Temp['Condition2']

Data['Roof Style + Roof Material'] = Temp['RoofStyle'] + Temp['RoofMatl']

Data.drop(['Condition2','RoofMatl'],inplace = True,axis = 1)
# Lets Start with Vantilation_ Category



Temp = Data[Vantilation_ + Target_].copy()

Temp['HeatingQC/Heating'] = Temp['HeatingQC']/Temp['Heating']



print(Temp.corr()[Target_[0]])

# Nothing much to do here
# Lets look at Floor_ Category

Temp = Data[Floor_ + Target_].copy()

Temp['GrLivArea/1st'] = Temp['GrLivArea'] / Temp['1stFlrSF']

Temp['GrLivArea/2nd'] = Temp['GrLivArea'] / Temp['2ndFlrSF']

Temp['GrLivArea/1st + 2nd'] = Temp['GrLivArea'] / (Temp['1stFlrSF'] + Temp['2ndFlrSF'])



Temp['LowQualFinSF/1st'] = Temp['LowQualFinSF'] / Temp['1stFlrSF']

Temp['LowQualFinSF/2nd'] = Temp['LowQualFinSF'] / Temp['2ndFlrSF']

Temp['LowQualFinSF/1st and 2nd'] = Temp['LowQualFinSF'] / (Temp['1stFlrSF'] * Temp['2ndFlrSF'])

Temp['LowQualFinSF/1st + 2nd'] = Temp['LowQualFinSF'] / (Temp['1stFlrSF'] + Temp['2ndFlrSF'])

Temp['1st + 2nd'] = Temp['1stFlrSF'] + Temp['2ndFlrSF']

Temp['LowQualFinSF+GrLivArea'] = Temp['LowQualFinSF'] + Temp['GrLivArea']

#sns.lmplot('BsmtQual','BsmtCond',data = Temp,fit_reg = False)

print(Temp.corr()[Target_[0]])



# By this We can remove attribute LowQualFinSF and Add 1st _ 2nd and LowQualFinSF+GrLivArea and GrLivArea/1st and 2nd 



Data['1st + 2nd'] = Temp['1stFlrSF'] + Temp['2ndFlrSF']

Data['LowQualFinSF+GrLivArea'] = Temp['LowQualFinSF'] + Temp['GrLivArea']

Data.drop(['LowQualFinSF'],inplace = True,axis = 1)
# Lets Study Bath_ Category

Temp = Data[Bath_ + Target_].copy()

Temp['Total'] = Temp['FullBath'] + Temp['HalfBath']

sns.lmplot('FullBath','HalfBath',data = Temp)

# Lets Keep Total and Remove Both Attribute

print(Temp.corr()[Target_[0]])

Data['TotalBath'] = Temp['FullBath'] + Temp['HalfBath']

Data.drop(['FullBath','HalfBath'],axis = 1,inplace = True)

# Lets Study Kitchen_ Category

Temp = Data[Kitchen_ + Target_].copy()

# Lets Try to Create Some attribute

Temp['TotalKitchen'] = Temp['KitchenAbvGr'] + Temp['KitchenQual']

#print(Temp[['KitchenAbvGr','KitchenQual']].head(50))

print(Temp.corr()[Target_[0]])

# Lets Keep Total and Remove Others

Data['TotalKitchen'] = Temp['KitchenAbvGr'] + Temp['KitchenQual']

Data.drop(['KitchenAbvGr','KitchenQual'],axis = 1,inplace = True)
# Lets Study Fire_ Category

Temp = Data[Fire_ + Target_].copy()

# Total Fireplace Qual

print(Temp[['Fireplaces','FireplaceQu']].head(10))

Temp['TotalFire'] = Temp['Fireplaces'] + Temp['FireplaceQu']

print(Temp.corr()[Target_[0]])

Data['TotalFire'] = Temp['Fireplaces'] + Temp['FireplaceQu']

Data.drop(['FireplaceQu'],axis = 1,inplace = True)
# Lets Study Garage_ Category

Temp = Data[Garage_ + Target_].copy()

Temp['GarageArea/Cars'] = Temp['GarageArea'] / Temp['GarageCars']

Temp['GarageType+GarageFinish'] = Temp['GarageType'] / Temp['GarageFinish']

Temp['GarageArea+Qual'] = Temp['GarageArea'] + Temp['GarageQual']

Temp['GarageQual+Cond'] = Temp['GarageQual'] + Temp['GarageCond']

print(Temp.corr()[Target_[0]])

Data['GarageArea+Qual'] = Temp['GarageArea'] + Temp['GarageQual']

Data.drop(['GarageQual'],axis = 1,inplace = True)
# Lets Study Porch_ CATEGORY

Temp = Data[Porch_ + Target_].copy()





Temp['3SsnPorch/OpenPorchSF'] = Temp['3SsnPorch'] / Temp['OpenPorchSF']

Temp['3SsnPorch/EnclosedPorch'] = Temp['3SsnPorch'] / Temp['EnclosedPorch']

Temp['3SsnPorch/OpenPorchSF and EnclosedPorch'] = Temp['3SsnPorch'] / (Temp['OpenPorchSF'] * Temp['EnclosedPorch'])

Temp['3SsnPorch/OpenPorchSF + EnclosedPorch'] = Temp['3SsnPorch'] / (Temp['OpenPorchSF'] + Temp['EnclosedPorch'])



Temp['ScreenPorch/OpenPorchSF'] = Temp['ScreenPorch'] / Temp['OpenPorchSF']

Temp['ScreenPorch/EnclosedPorch'] = Temp['ScreenPorch'] / Temp['EnclosedPorch']

Temp['ScreenPorch/OpenPorchSF and EnclosedPorch'] = Temp['ScreenPorch'] / (Temp['OpenPorchSF'] * Temp['EnclosedPorch'])

Temp['ScreenPorch/OpenPorchSF + EnclosedPorch'] = Temp['ScreenPorch'] / (Temp['OpenPorchSF'] + Temp['EnclosedPorch'])



Temp['3SsnPorch+ScreenPorch'] = Temp['3SsnPorch'] + Temp['ScreenPorch']

Temp['OpenPorchSF+EnclosedPorch'] = Temp['OpenPorchSF'] + Temp['EnclosedPorch']

print(Temp.corr()[Target_[0]])

# Nothing Good so not doing anything here
# Lets Study Sold_ Attribute

Temp = Data[Sold_ + Target_].copy()

Temp['YRBuilt/YrSold'] = Temp['YearBuilt'] / Temp['YrSold']

Temp['YRBuilt/MoSold'] = Temp['YearBuilt'] / Temp['MoSold']

Temp['YearRemodAdd/YrSold'] = Temp['YearRemodAdd'] / Temp['YrSold']

Temp['YearRemodAdd/MoSold'] = Temp['YearRemodAdd'] / Temp['MoSold']

Temp['YRBuilt/YearRemodAdd'] = Temp['YearBuilt'] / Temp['YearRemodAdd']

Temp['YRBuilt+YearRemodAdd'] = Temp['YearBuilt'] + Temp['YearRemodAdd']

print(Temp.corr()[Target_[0]])

Data['YRBuilt/YrSold'] = Temp['YearBuilt'] / Temp['YrSold']

Data['YearRemodAdd/YrSold'] = Temp['YearRemodAdd'] / Temp['YrSold']

Data['YRBuilt+YearRemodAdd'] = Temp['YearBuilt'] + Temp['YearRemodAdd']

Data.drop(['YrSold'],axis = 1,inplace = True)
# Lets look at Overall_ Category

Temp = Data[Overall_ + Target_].copy()

Temp['OverallQual+Cond'] = Temp['OverallQual'] + Temp['OverallCond']

Data['OverallQual+Cond'] = Temp['OverallQual'] + Temp['OverallCond']

print(Temp.corr()[Target_[0]])

Data.drop(['OverallCond'],axis = 1,inplace = True)
# We can also Try Something else at Feature ENgineering but now lets Start with Machine Learning

# Lets Start by Seperating traing and Test Set and Labels

Train_Labels = np.array(Data.loc[Data[Target_[0]].notnull(),Target_[0]])

train_set = Data.loc[Data[Target_[0]].notnull(),:].drop(['SalePrice','Id'],axis = 1)

test_set = Data.loc[Data[Target_[0]].isnull(),:].drop(['SalePrice'],axis = 1)

Features = train_set.columns

Id = test_set['Id']

test_set.drop(['Id'],axis = 1,inplace = True)

from sklearn.model_selection import cross_val_score

from sklearn.impute import SimpleImputer as Imputer

from sklearn.preprocessing import MinMaxScaler

from sklearn.pipeline import Pipeline

from sklearn.metrics import make_scorer

from sklearn.metrics import mean_squared_error,r2_score

pipeline = Pipeline([('imputer', Imputer(strategy = 'median')),

('scaler', MinMaxScaler())])

# Fit and transform training data

train_set = pipeline.fit_transform(train_set)

test_set = pipeline.transform(test_set)

scorer = make_scorer(r2_score,greater_is_better = True)  # A simple scorer function for regression part to check how good model is
# This is very helpful code snippet from Kaggler @Will koehrsen and I find it very useful everything is self explanatory



def plot_feature_importances(df, n = 10, threshold = None):

    """Plots n most important features. Also plots the cumulative importance if

    threshold is specified and prints the number of features needed to reach threshold cumulative importance.

    Intended for use with any tree-based feature importances. 

    

    Args:

        df (dataframe): Dataframe of feature importances. Columns must be "feature" and "importance".

    

        n (int): Number of most important features to plot. Default is 15.

    

        threshold (float): Threshold for cumulative importance plot. If not provided, no plot is made. Default is None.

        

    Returns:

        df (dataframe): Dataframe ordered by feature importances with a normalized column (sums to 1) 

                        and a cumulative importance column

    

    Note:

    

        * Normalization in this case means sums to 1. 

        * Cumulative importance is calculated by summing features from most to least important

        * A threshold of 0.9 will show the most important features needed to reach 90% of cumulative importance

    

    """

    plt.style.use('fivethirtyeight')

    

    # Sort features with most important at the head

    df = df.sort_values('importance', ascending = False).reset_index(drop = True)

    

    # Normalize the feature importances to add up to one and calculate cumulative importance

    df['importance_normalized'] = df['importance'] / df['importance'].sum()

    df['cumulative_importance'] = np.cumsum(df['importance_normalized'])

    

    plt.rcParams['font.size'] = 12

    

    # Bar plot of n most important features

    df.loc[:n, :].plot.barh(y = 'importance_normalized', 

                            x = 'feature', color = 'darkgreen', 

                            edgecolor = 'k', figsize = (12, 8),

                            legend = False, linewidth = 2)



    plt.xlabel('Normalized Importance', size = 18); plt.ylabel(''); 

    plt.title(f'{n} Most Important Features', size = 18)

    plt.gca().invert_yaxis()

    

    

    if threshold:

        # Cumulative importance plot

        plt.figure(figsize = (8, 6))

        plt.plot(list(range(len(df))), df['cumulative_importance'], 'b-')

        plt.xlabel('Number of Features', size = 16); plt.ylabel('Cumulative Importance', size = 16); 

        plt.title('Cumulative Feature Importance', size = 18);

        

        # Number of features needed for threshold cumulative importance

        # This is the index (will need to add 1 for the actual number)

        importance_index = np.min(np.where(df['cumulative_importance'] > threshold))

        

        # Add vertical line to plot

        plt.vlines(importance_index + 1, ymin = 0, ymax = 1.05, linestyles = '--', colors = 'red')

        plt.show();

        

        print('{} features required for {:.0f}% of cumulative importance.'.format(importance_index + 1, 

                                                                                  100 * threshold))

    

    return df

from sklearn.ensemble import GradientBoostingRegressor

model = GradientBoostingRegressor(n_estimators = 100,random_state = 10)

cv_Score = cross_val_score(model,train_set,Train_Labels,cv = 10,scoring = scorer)



print(f'10 Fold Cross Validation R2 Score = {round(cv_Score.mean(),4)} with std of {(cv_Score.std(),4)}')



model.fit(train_set, Train_Labels)

# Feature importances into a dataframe

feature_importances = pd.DataFrame({'feature': Features, 'importance': model.feature_importances_})

plot_feature_importances(feature_importances, n = 20, threshold = 0.95)
# A Simple self explanatory function to test All the models that you will use





import warnings 

from sklearn.exceptions import ConvergenceWarning



# Filter out warnings from models

warnings.filterwarnings('ignore', category = ConvergenceWarning)

warnings.filterwarnings('ignore', category = DeprecationWarning)

warnings.filterwarnings('ignore', category = UserWarning)



# Dataframe to hold results

model_results = pd.DataFrame(columns = ['model', 'cv_mean', 'cv_std'])



def cv_model(train, train_labels, model, name, model_results=None):

    """Perform 10 fold cross validation of a model"""

    

    cv_scores = cross_val_score(model, train, train_labels, cv = 10, scoring=scorer, n_jobs = -1)

    print(f'10 Fold CV Score: {round(cv_scores.mean(), 5)} with std: {round(cv_scores.std(), 5)}')

    

    if model_results is not None:

        model_results = model_results.append(pd.DataFrame({'model': name, 

                                                           'cv_mean': cv_scores.mean(), 

                                                            'cv_std': cv_scores.std()},

                                                           index = [0]),

                                             ignore_index = True)



        return model_results
from sklearn.linear_model import LinearRegression

from sklearn import linear_model

from sklearn.linear_model import Ridge



model = LinearRegression()

model_results = cv_model(train_set,Train_Labels,model,'Linear Regression',model_results)



model = linear_model.Lasso()

model_results = cv_model(train_set,Train_Labels,model,'Lasso Regression',model_results)



model = Ridge()

model_results = cv_model(train_set,Train_Labels,model,'Ridge Regression',model_results)



model = GradientBoostingRegressor(n_estimators = 100,random_state = 10)

model_results = cv_model(train_set,Train_Labels,model,'Gradient Boosting',model_results)



# Clearly GradientBoosting does a good Job

# Lets Try Dimensionality Reduction

from sklearn.decomposition import PCA



Correlation = np.corrcoef(train_set,rowvar = False)

U,Sigma,V = np.linalg.svd(Correlation)

   

for K in range(1,train_set.shape[1]):

    if (1 - np.sum(Sigma[:K])/np.sum(Sigma) <= 0.001):

        break

pca = PCA(n_components = K)

train_pca_selected = pca.fit_transform(train_set)

test_pca_selected = pca.transform(test_set)
model = LinearRegression()

model_results = cv_model(train_pca_selected,Train_Labels,model,'Linear Regression',model_results = None)



model = linear_model.Lasso()

model_results = cv_model(train_pca_selected,Train_Labels,model,'Lasso Regression',model_results = None)



model = Ridge()

model_results = cv_model(train_pca_selected,Train_Labels,model,'Ridge Regression',model_results = None)



model = GradientBoostingRegressor(n_estimators = 100,random_state = 10)

model_results = cv_model(train_pca_selected,Train_Labels,model,'Gradient Boosting',model_results = None)

from sklearn.feature_selection import RFECV



# Create a model for feature selection

model = GradientBoostingRegressor(n_estimators = 100,random_state = 10)

# Create the object

selector = RFECV(model, step = 1, cv = 3, scoring= scorer, n_jobs = -1)

selector.fit(train_set, Train_Labels)



plt.plot(selector.grid_scores_);



plt.xlabel('Number of Features'); plt.ylabel('R2 Score'); plt.title('Feature Selection Scores');

selector.n_features_
rankings = pd.DataFrame({'feature': Features, 'rank': list(selector.ranking_)}).sort_values('rank')

rankings.head(10)
train_selected = selector.transform(train_set)

test_selected = selector.transform(test_set)



# Convert back to dataframe



selected_features = Features[np.where(selector.ranking_ ==1)]

train_selected = pd.DataFrame(train_selected, columns = selected_features)

test_selected = pd.DataFrame(test_selected, columns = selected_features)
model = LinearRegression()

model_results = cv_model(train_selected,Train_Labels,model,'Linear Regression',model_results = None)



model = linear_model.Lasso()

model_results = cv_model(train_selected,Train_Labels,model,'Lasso Regression',model_results = None)



model = Ridge()

model_results = cv_model(train_selected,Train_Labels,model,'Ridge Regression',model_results = None)



model = GradientBoostingRegressor(n_estimators = 100,random_state = 10)

model_results = cv_model(train_selected,Train_Labels,model,'Gradient Boosting',model_results = None)

from sklearn.model_selection import GridSearchCV

params_grid = [

    {'n_estimators' : [10,100,500,1000] , 'learning_rate' : [1,0.1,0.3,0.01]}

]

model = GradientBoostingRegressor()

grid_search = GridSearchCV(model,params_grid,cv = 5,scoring = scorer)

grid_search.fit(train_selected,Train_Labels)

grid_search.best_params_



# We got the parameters. Feel free to play with it
model = GradientBoostingRegressor(n_estimators = 1000,random_state = 10,learning_rate = 0.01)

model.fit(train_selected,Train_Labels)

Prediction = model.predict(test_selected)

Result = pd.DataFrame({'Id' : Id, 'SalePrice' : Prediction})

Result.head()
