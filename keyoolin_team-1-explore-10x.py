# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # data visualisation
import seaborn as sns # data visualisation
from sklearn.model_selection import train_test_split #train_test_split
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, Lasso# Regression Models
from sklearn.preprocessing import StandardScaler #Scaler
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error #reports
from sklearn.metrics import mean_absolute_error, r2_score, explained_variance_score #reports
from scipy import stats #Imports statistics
from yellowbrick.regressor import AlphaSelection

%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#importing the train dataset
train = pd.read_csv('../input/train.csv')
#importing the test dataset
test = pd.read_csv('../input/test.csv')
#Checking the description of all variables
f = open("../input/data_description.txt", "r")
print(f.read())
f.close()
#Converting Numeric variables which are actually Categorical variables into categories
def cat(data):
    
    data = data.replace({"MSSubClass" : {20 : "SC20", 30 : "SC30", 40 : "SC40", 45 : "SC45", 
                                       50 : "SC50", 60 : "SC60", 70 : "SC70", 75 : "SC75", 
                                       80 : "SC80", 85 : "SC85", 90 : "SC90", 120 : "SC120", 
                                       150 : "SC150", 160 : "SC160", 180 : "SC180", 190 : "SC190"},
                       "MoSold" : {1 : "Jan", 2 : "Feb", 3 : "Mar", 4 : "Apr", 5 : "May", 6 : "Jun",
                                   7 : "Jul", 8 : "Aug", 9 : "Sep", 10 : "Oct", 11 : "Nov", 12 : "Dec"}
                      })
    return data
train = cat(train)
test = cat(test)
#Function to create graphs easier. Creates multiple graphs based on inputted columns to allow for better visualisation and comparing of visuals
def graph(data, x, y = '', graph_type = 'scatter', **kwargs):
    
    """
    Version 1
    ---------

    This function allows for the creation of similar graphs for multiple columns in a DataFrame. 
    Advised to limit column size to 20 to reduce memory usage.

    Parameters
    ----------    
    data: of type DataFrame, required.
        Input data variables may contain numeric and categorical features/series.
        
    graph_type: of type string. Must be an element of ['scatter','hist','dist', 'histogram']
        Optional, 'scatter' taken as default
    
    x: of type list. Columns to be visualised.
        Input data variables must be of type list, even if a single variable entered.
        Columns/features entered may be numeric or categorical.
        
    y: of type string. Y column to be plotted against.
        Input data variables, needs to be numeric. Must be type string. Single column entry.
        Optional, only required for scatter plots

    Returns
    -------
    ax : matplotlib Axes
        Returns the Axes object with the plot drawn onto it. 
        
    Raises
    ------
    ValueError() : Unacceptable graph type
    
    Example
    -------
    
    graph(data = train, graph_type = 'scatter', x = ['LotArea'], y = 'SalePrice')
    
    """

    if graph_type == "scatter": 
        for i, col in enumerate(x): #Allows for the counting of number of plots
            if data[col].dtypes != 'object': #Allows only numeric variables
                plt.figure(i, figsize = (15,5)) #Creates a figure with i being the plot number stored in memory
                sns.scatterplot(data = data,x = col, y = y) 
            else:
                plt.figure(i + len(x), figsize = (15,5)) #i + len(x) because errors were occuring with i
                sns.catplot(x=col, y=y, data=data, aspect = 3, jitter = True); #Categorical scatter plot

    elif graph_type in ["hist", 'dist', 'histogram']:
        for i, col in enumerate(x):
            if data[col].dtypes != 'object': #Cannot plot a distribution of categorical features
                plt.figure(i, figsize = (15,5))
                sns.distplot(data[col])
        
    else:
        raise ValueError('{} not an accepted graph_type'.format(graph_type)) #ValueError if incorrect graph_type entered
#Simple distribution of the Sale Price. 
sns.distplot(train['SalePrice'], bins = 40, kde = False, hist = True)
#As the sale price is not normally distributed, we are going to log the distribution to increase predictiveness and normality
sns.distplot(np.log(train['SalePrice']), bins = 40, kde = False, hist = True)
graph(train, graph_type = 'scatter', x = train.columns, y = 'SalePrice')
numeric = []
for  col in train.columns:
    if train[col].dtypes != 'object':
        numeric.append(col)
        
z = np.abs(stats.zscore(train[numeric]))
print(np.where(z > 3))
train.shape
train[(z < 3).all(axis=1)].shape
where = np.where(z>3) # Checks the location of all outliers
where = where[1] #Gets only the columns of the outliers
where = pd.Series(where) #Converts the array to a pandas Series
where = where.unique() #Gets the unique columns in integar value
where = list(where) #Gets a list of the column indexes
where = sorted(where) #Sorts the column indexes
where = train.iloc[:, where].columns #Gets the column names in an index format
where = list(where)
where
graph(data = train, graph_type = 'scatter', x = where, y = 'SalePrice')
train.drop(train[train['SalePrice']>700000].index, axis = 0, inplace = True)
train[train['SalePrice']>700000]
train.reset_index(inplace = True)
train.drop('index', axis = 1, inplace = True)
#Function to transform outliers. These would need to be found before entering into function
def outlier_transform(data, column, value, against):
    
    """
    Version 1
    ---------

    This function allows for the transformation of outliers in the dataset. Unfortunately, 
    version 1's capabilities are limited and thus not recommended unless the dataset is well known.
    Transformed based on location of datapoints in the against column.

    Parameters
    ----------    
    data: of type DataFrame, required.
        Input data variable may contain numeric and categoric features/series.
        
    column: of type string. Columns for outliers to be transformed
        Input data variable must be of type string. 
        Columns/features entered must be numeric in value.
        
    value: of type int/float. Input data variable, needs to be numeric. 
        value represents the upper limit of which any values past this, will be considered an outlier and transformed.
    
    against: of type string. 
        Input data variable must be of type string. 
        Columns/features entered must be numeric in value.
        column will be transformed based on it's location in the against feature.

    Returns
    -------
    pandas DataFrame : DateFrame with column outliers transformed to be more representive of the sample characteristics
    
    Example
    -------
    
    outlier_transform(data = train, column = , value, against = 'SalePrice')

    """
    
    
    replace_values_index = list(data[data[column]>value].index)
    against_values = list(data[data[column]>value][against])
    
    #values = {}
    for i in range(len(against_values)):
        df = data[(data[against] > against_values[i] - 10000) & (data[against] < against_values[i] + 10000)]
        mean = df[column].mean()
        data.at[replace_values_index[i], column] = mean
    
    return data
train = outlier_transform(train, 'LotFrontage', 200, against = 'SalePrice')
train = outlier_transform(train, 'LotArea', 100000, against = 'SalePrice')
train = outlier_transform(train, 'MasVnrArea', 1300, against = 'SalePrice')
graph(train, graph_type = 'scatter', x = ['LotFrontage', 'LotArea', 'MasVnrArea'], y ='SalePrice')
zero = []
for i in train:
    temp = train[i].value_counts()
    print(temp.head(2))
    if temp.index[0] == 0 or temp.index[0] == 'None':
        zero.append(i)
zero
#graph(train, graph_type = 'dist', x = zero)
#Function to create a dataframe with statistics on the missing data
def understand_missing(data):
    
    """
    Version 1
    ---------

    This function searches the dataframe for missing values and returns statistics of missing values

    Parameters
    ----------    
    data: of type DataFrame, required.
        Input data variable may contain numeric and categoric features/series.

    Returns
    -------
    pandas DataFrame : DateFrame with missing values and percentages missing
    
    Example
    -------
    
    understand_missing(train)

    """
    missing = pd.DataFrame(data.isnull().sum().sort_values(ascending = False), columns = ['Total Missing'])
    missing = missing[missing['Total Missing']>0]
    
    missing['Percent'] = (missing["Total Missing"]/len(data))*100
    
    return missing

#Function to fill missing data
def fill_missing(data):
    
    def garage(cols):
        YearBuilt = cols[0]
        GarageYrBuilt = cols[1]
        
        if pd.isnull(GarageYrBuilt):
            return YearBuilt
        else: 
            return GarageYrBuilt
    
    data['GarageYrBlt'] = data[['YearBuilt','GarageYrBlt']].apply(garage, axis = 1)

    data['LotFrontage'].fillna(data.LotFrontage.median(), inplace = True)
    
    
    for i in data[list(understand_missing(data).index)]:
        if data[i].dtypes != 'object':
            data[i].fillna(0, inplace = True)
        elif data[i].dtypes == 'object':
            data[i].fillna('None', inplace = True)
    
    return data
def sorting(series):
    return series.sort_values(kind="quicksort", ascending = False)

def top_corr(data):
    corr = data.corr().abs().unstack()
    return sorting(corr)

#Function to check correlations
def correlate(data, column1='', column2=''):
    
    """
    Version 1
    ---------

    This function checks the correlation in a DataFrame

    Parameters
    ----------    
    data: of type DataFrame, required.
        Input data variable may contain numeric and categoric features/series.
        
    column1, column2: of type string. Columns to be correlated must be of dtype int or float.
        Only numeric columns can be correlated using this function.
        
    Returns
    -------
    pandas DataFrame : DateFrame with two columns and correlations
    pandas Series: One index column and correlations
    Int: Single correlation value
    
    Example
    -------
    
    correlate(data = train)
    correlate(data = train, column1='SalePrice')
    correlate(data = train, column1='LotFrontage', column2='SalePrice')

    """
                
    #Returns correlations based on columns entered
    if column1 != '' and column2 != '':
        return data.corr()[column1][column2]
    elif column1 != '' and column2 == '':
        return sorting(data.corr()[column1])[1:]
    elif column1 == '' and column2 != '':
        return sorting(data.corr()[column2])[1:]
    
    #Getting a list of the top correlated columns and their values
    sort = top_corr(data)

    #Deleting duplicates
    data_sort = pd.DataFrame(sort.reset_index()) #Creates a dataframe from sorted series
    count = len([i for i in sort if i == 1.0]) #Checks the number of self correlations

    data_sort.drop(range(0,count), inplace = True) #Removes all self correlations
    lis = np.arange(0, len(data_sort), 2) #Creates index of alternating values. eg. [0,2,4,6,2n]
   
    return data_sort.iloc[lis] #returns without duplicates

#Function which drops all columns which have high multicollinearity
def drop_columns(data, drop_level = 0.65, print_output = True, column = 'SalePrice'):
    
    """
    Version 1
    ---------

    This function checks drops columns in a DataFrame based on high correlations. 
    Function 'correlate' needs to have been initialised.

    Parameters
    ----------    
    data: of type DataFrame, required.
        Input data variable may contain numeric and categoric features/series.
        
    drop_level: of type int, optional.
        Default drop level of 0.65. drop_level indicated the threshold of which higher correlations 
        will be dropped. I.e. 0.65 and above.
        
    print_output: of type Boolean, optional.
        If True, prints analysis of each correlation compared and dropped.
        
    column: of type String, optional.
        Default column of SalePrice. High correlations with this column will be kept.
        
    Returns
    -------
    list: droplist of all columns with high correlations which should be dropped.
    
    Example
    -------
    drop_columns(data = train, drop_level = 0.65, print_output = True, column = 'SalePrice')

    """
    
    corr = correlate(data)
    
    droplist = []

    # Drops all columns which have a high correlation value with each other removing multicollinearity
    for i in range(len(corr[corr[0]>drop_level])):

        level_0 = corr.iloc[i]['level_0']
        level_1 = corr.iloc[i]['level_1']
        value = corr.iloc[i][0] #Prints correlation Value
        if print_output == True:
            print('Correlation order number: '+str(i+1)) #Prints the correlation column names
            print('{}   {}    {}'.format(level_0, level_1, value))
            print('')

        if (level_0 != column) & (level_1 != column): #Ensures that high correlations with SalePrice are not dropped

            correlation1 = correlate(df, level_0, column)  #Checks correlation with column and Sale Price

            correlation2 = correlate(df, level_1, column) #Checks correlation with column and Sale Price

            if print_output == True:
                print('Correlation 1')
                print(level_0," : ", correlation1)
                print('')
                print('Correlation 2')
                print(level_1," : ", correlation2)
                print('')

            #print(float(correlation1[0]))

            #Test to determine which column to delete
            #Column with the highest correlation with SalePrice will be kept while the column with the lowest correlation will be deleted.
            if (float(correlation1) > float(correlation2)):
                if print_output == True:
                    print('Correlation 2 is lower, drop {}'.format(level_1))
                droplist.append(level_1)
            elif(float(correlation1) < float(correlation2)):
                if print_output == True:
                    print('Correlation 1 is lower, drop {}'.format(level_0))
                droplist.append(level_0)

            
        if print_output == True:
            print('Droplist :')
            print(droplist)
            print('\n')
            
    return droplist #Returns the dataframe with all appropriate columns dropped.

#Function which runs an ANOVA test to determine whether categorical variables are worth keeping in the model
def cat_corr(data, column):
    
    """
    Version 1
    ---------

    This function runs an ANOVA to determine correlations of elements within features. 
    Prints out features and the statistics associated with them.

    Parameters
    ----------    
    data: of type DataFrame, required.
        Input data variable may contain numeric and categoric features/series.
        
    column: of type String, required.
        Column in DataFrame which the ANOVA is to be run on.
        
    Returns
    -------
    None
    
    Example
    -------
    cat_corr(data = train, column = 'MasVnrType')

    """
    
    values = list(data[column].value_counts().index)
    if len(values) == 2:
        F, p = stats.f_oneway(data[data[column]==values[0]].SalePrice,
                              data[data[column]==values[1]].SalePrice)
        
    elif len(values) == 3:
        F, p = stats.f_oneway(data[data[column]==values[0]].SalePrice,
                              data[data[column]==values[1]].SalePrice,
                              data[data[column]==values[2]].SalePrice)
        
    elif len(values) == 4:
        F, p = stats.f_oneway(data[data[column]==values[0]].SalePrice,
                              data[data[column]==values[1]].SalePrice,
                              data[data[column]==values[2]].SalePrice,
                              data[data[column]==values[3]].SalePrice)
    elif len(values) == 5:
        F, p = stats.f_oneway(data[data[column]==values[0]].SalePrice,
                              data[data[column]==values[1]].SalePrice,
                              data[data[column]==values[2]].SalePrice,
                              data[data[column]==values[3]].SalePrice,
                              data[data[column]==values[4]].SalePrice)
        
    elif len(values) == 6:
        F, p = stats.f_oneway(data[data[column]==values[0]].SalePrice,
                              data[data[column]==values[1]].SalePrice,
                              data[data[column]==values[2]].SalePrice,
                              data[data[column]==values[3]].SalePrice,
                              data[data[column]==values[4]].SalePrice,
                              data[data[column]==values[5]].SalePrice)
        
    elif len(values) == 7:
        F, p = stats.f_oneway(data[data[column]==values[0]].SalePrice,
                              data[data[column]==values[1]].SalePrice,
                              data[data[column]==values[2]].SalePrice,
                              data[data[column]==values[3]].SalePrice,
                              data[data[column]==values[4]].SalePrice,
                              data[data[column]==values[5]].SalePrice,
                              data[data[column]==values[6]].SalePrice)
        
    elif len(values) == 8:
        F, p = stats.f_oneway(data[data[column]==values[0]].SalePrice,
                              data[data[column]==values[1]].SalePrice,
                              data[data[column]==values[2]].SalePrice,
                              data[data[column]==values[3]].SalePrice,
                              data[data[column]==values[4]].SalePrice,
                              data[data[column]==values[5]].SalePrice,
                              data[data[column]==values[6]].SalePrice,
                              data[data[column]==values[7]].SalePrice)
        
    elif len(values) == 9:
        F, p = stats.f_oneway(data[data[column]==values[0]].SalePrice,
                              data[data[column]==values[1]].SalePrice,
                              data[data[column]==values[2]].SalePrice,
                              data[data[column]==values[3]].SalePrice,
                              data[data[column]==values[4]].SalePrice,
                              data[data[column]==values[5]].SalePrice,
                              data[data[column]==values[6]].SalePrice,
                              data[data[column]==values[7]].SalePrice,
                              data[data[column]==values[8]].SalePrice)
    else:
        print(column, 'has too many values', '\n' )
        
    if len(values) < 7:    
        print('The ANOVA output for feature', column, 'is:')    
        print(' The F value of the ANOVA is : ',F,'\n','The p value of the ANOVA is : ', p)
        
        if p > 0.05:
            print(' Recommended that feature',column,'is dropped as the feature values are too similar and thus have low inference ability')
        print('\n')
understand_missing(train)
understand_missing(test)
graph(data = train, graph_type = 'scatter', x = ['PoolQC', 'MiscFeature', 'Alley', 'Fence'], y = 'SalePrice')
train[['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'SalePrice']].corr()
for i in train.columns:
    if train[i].dtypes == object:
        cat_corr(train, i)
train.drop(['Street', 'Utilities', 'LandSlope', 'PoolQC','MiscFeature'], axis = 1, inplace = True)
#visualising the null values as a heatmap
fig, ax = plt.subplots(figsize=(15,8))
sns.heatmap(train.isnull(), cmap='YlGnBu')
#This calls the function to fill in all missing values
train = fill_missing(train)
test = fill_missing(test)
understand_missing(train)
train[['OpenPorchSF', 'EnclosedPorch', '3SsnPorch','ScreenPorch']].head()
graph(train, graph_type = 'scatter', x = ['OpenPorchSF', 'EnclosedPorch', '3SsnPorch','ScreenPorch'], y = 'SalePrice')   
train['Porch'] = train['OpenPorchSF'] + train['EnclosedPorch'] + train['3SsnPorch'] + train['ScreenPorch']
test['Porch'] = test['OpenPorchSF'] + test['EnclosedPorch'] + test['3SsnPorch'] + test['ScreenPorch']
sns.scatterplot(data = train,x = 'Porch', y= 'SalePrice')
train.drop(['OpenPorchSF', 'EnclosedPorch', '3SsnPorch','ScreenPorch'], axis = 1, inplace = True)
test.drop(['OpenPorchSF', 'EnclosedPorch', '3SsnPorch','ScreenPorch'], axis = 1, inplace = True)
train[['BsmtQual', 'BsmtCond',
       'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2',
       'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath']].head(5)
graph(train, graph_type = 'scatter', x = ['BsmtQual', 'BsmtCond',
       'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2',
       'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'], y = 'SalePrice')
def bsmtbath(cols):
    full = cols[0]
    half = cols[1]
    
    if cols[0] == 0 and cols[1] == 0:
        return 0
    else: return 1
    
train['BsmtBath'] = train[['BsmtFullBath', 'BsmtHalfBath']].apply(bsmtbath, axis = 1)
test['BsmtBath'] = test[['BsmtFullBath', 'BsmtHalfBath']].apply(bsmtbath, axis = 1)
train[['BsmtBath', 'BsmtFullBath', 'BsmtHalfBath']].head()
graph(data = train, graph_type='scatter', x=['BsmtBath', 'BsmtFullBath', 'BsmtHalfBath'], y='SalePrice')
train.drop(['BsmtFullBath', 'BsmtHalfBath'], axis = 1, inplace = True)
test.drop(['BsmtFullBath', 'BsmtHalfBath'], axis = 1, inplace = True)
graph(data = train, graph_type='scatter', x=['BsmtQual', 'BsmtCond'], y='SalePrice')
sns.scatterplot(data = train,x = 'BsmtFinSF2', y = 'SalePrice')
graph(train, x = ['GarageType', 'GarageYrBlt', 'GarageFinish',
       'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond'], y = 'SalePrice')
train.drop(['GarageQual', 'GarageCond'], axis = 1, inplace = True)
test.drop(['GarageQual', 'GarageCond'], axis = 1, inplace = True)
#Function to standardize the dataframe
def standard(data):
    standard_list = []
    
    for i in data:
        if i != 'SalePrice' and i != 'Id':
            if data[i].dtypes != 'object': #check if column is an object
                standard_list.append(i)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data[standard_list])
    X_standard = pd.DataFrame(X_scaled, columns = data[standard_list].columns)
    
    data.drop(standard_list, axis = 1, inplace = True)
    
    return pd.concat([data, X_standard], axis = 1)
train = standard(train)
test = standard(test)
#Function to create dummy variables for 
def create_dummies(dataframe):
    temp_dict = {}
    drop = []
    
    for i in dataframe:
        if dataframe[i].dtypes == 'object': #check if column is an object
            drop.append(i) #Append column name to a drop list
            if dataframe[i].isnull().sum() > 0: #Check if any missing values
                temp_df = pd.get_dummies(dataframe[i]).reset_index() #Create dummies
                if 'None' in list(temp_df.columns): #Check if any None values in dataframe
                    temp_df.drop('None', axis = 1, inplace = True) #Deletes none column
                #print('Check, temp_df - missing = True \n',temp_df)
            else:
                temp_df = pd.get_dummies(dataframe[i], drop_first = True).reset_index() #Creates dummies, drops first to avoid dummy trap
                #print('Check, temp_df - missing = False \n',temp_df)
            temp_dict[i] = temp_df #Stores dummy dataframes in a dictionary
            #print('Check, temp_dict \n',temp_dict[i])
            
    temp_dict = { k: v.set_index("index") for k, v in temp_dict.items()} #Sets the index up for concatenation
    df = pd.concat(temp_dict, axis=1)
    df.columns = pd.Index([i[0] + '_' + i[1] for i in df.columns.tolist()]) #Converts columns from multi-index to single index
    
    return df, drop
dummies, drop = create_dummies(train)
train.drop(drop, axis = 1, inplace = True)
train = pd.concat([train, dummies], axis = 1)

dummies, drop = create_dummies(test)
test.drop(drop, axis = 1, inplace = True)
test = pd.concat([test, dummies], axis = 1)

trc = list(train.columns)
tc = list(test.columns)
diff = list(frozenset(tc).difference(trc))

test.drop(diff, axis = 1, inplace = True)
# Creates a regression model
def model(X, y, test_size = 0.3, random_state = 101, model_type = ' '):
    #Splits the data into train and test data, not needed for official hand-in
    
    """
    Version 1
    ---------

    This function creates a regression model

    Parameters
    ----------    
    X: of type DataFrame, required.
        Input data variable must contain numeric features/series. 
        Categoric features would not be accepted. Independent variable in the regression model.
        
    y: of type pandas Series, required.
        Input data variable must contain numeric features/series.
        Dependent or Target variable in the regression model.
        
    model_type: of type string, required.
        Must be either Linear, RidgeCV, Lasso or LassoCV
        
    test_size: of type int, optional.
        Train_test_split test size. Default value of 30% test split.
        Determines how the variables would be split into a training and testing dataset. 
    
    random_state: of type int, optional.
        Train_test_split random state. Default value of 101.
        The random state determines the random seed for the splitting of the data.
        
        
    Returns
    -------
    pandas Series: A series of predictions for the test features. 
    
    Example
    -------
    model(X = train["LotArea","LotFrontage"], y = train['SalePrice'], 
    ...   test_size = 0.3, random_state = 101, model_type = "Lasso")

    """
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state) 
    
    def metrics(model, X, predictions, y_test):
        print(pd.DataFrame(model.coef_, X.columns, columns=["Coefficient"]))
        print("\n")

        #Outputing the Predictive Metrics
        print("Assessment of Predictive Analytics")
        print("The Residual Sum of Squares is: ", round(((y_test - predictions)**2).sum(),2))
        print('The Mean Absolute Error is: ', mean_absolute_error(y_test,predictions))
        print('The Mean Squared Error is: ', mean_squared_error(y_test,predictions))
        print('The Root Mean Squared Error is: ', np.sqrt(mean_absolute_error(y_test,predictions)))
        print('The r^2 value is: ', r2_score(y_test, predictions))
        print('The explained variance score is: ', explained_variance_score(y_test, predictions))
    
    if 'Linear' in model_type:
        #Fitting a Linear Regression Model
        model_linear = LinearRegression()        
        model_linear.fit(X_train, y_train)
        predictions = model_linear.predict(X_test)

        metrics(model_linear, X, predictions, y_test)
        
    if 'Ridge' in model_type:
        #Fitting a Ridge Regression Model
        model_ridge = Ridge()
        model_ridge.fit(X_train, y_train)
        predictions = model_ridge.predict(X_test)

        metrics(model_ridge, X, predictions, y_test)
    
    if 'Lasso' in model_type:
        model_lasso = Lasso(alpha=0.01)
        model_lasso.fit(X_train, y_train)
        predictions = model_lasso.predict(X_test)
        
    return predictions
"""Submission Test 1

columns = list(test.columns)
columns.append('SalePrice')

train['SalePrice'] = np.log(train['SalePrice'])

droplist = drop_columns(train[columns], drop_level = 0.6, print_output = True)

X = train[list(test.columns)].drop(droplist, axis = 1)
y = train['SalePrice']

model_lasso = Lasso(alpha=0.01)
model_lasso.fit(X, y)
predictions = model_lasso.predict(test.drop(droplist, axis = 1))

my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predictions})
my_submission.to_csv('submission.csv', index=False)

"""
"""

train['SalePrice'] = np.log(train['SalePrice'])

X = train[list(test.columns)]
y = train['SalePrice']

model_lasso = Lasso(alpha=0.01)
model_lasso.fit(X, y)
predictions = model_lasso.predict(test)

predictions = np.exp(predictions)

#my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predictions})
#my_submission.to_csv('submission2.csv', index=False)

"""
X = train[list(test.columns)].drop('Id', axis = 1)
y = train['SalePrice']

# Create a list of alphas to cross-validate against
alphas = np.logspace(-10, 1, 400)

# Instantiate the linear model and visualizer
model2 = RidgeCV(alphas=alphas)
visualizer2 = AlphaSelection(model2)

visualizer2.fit(X, y)
alpha_plot = visualizer2.poof()
alpha_plot
train['SalePrice'] = np.log(train['SalePrice'])

X = train[list(test.columns)].drop('Id', axis = 1)
y = train['SalePrice']

model_Ridge = Ridge(alpha=10)
model_Ridge.fit(X, y)
predictions = model_Ridge.predict(test.drop('Id', axis =1))

predictions = np.exp(predictions)

my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predictions})
my_submission.to_csv('submission.csv', index=False)
