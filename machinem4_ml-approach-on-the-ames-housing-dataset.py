import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from scipy.stats import norm

from scipy import stats

from scipy.stats import norm, skew

import warnings

warnings.filterwarnings('ignore')

import missingno as msno

%matplotlib inline
def response_variable_distribution(response_variable, train_df):

    '''

    Gets the distribution, mean and standard deviation of the response variable. Also provides a probability plot of sample data against the quantiles of a 

    specified theoretical distribution as well as providing a best-fit line for the data.

    

    Parameters

    ----------

    

    response_variable: str

        str object containing the response variable in the analysis.

    

    train_df: dataframe

        dataframe object containing the original values of the response variable

    

    Returns

    -------

    

    distplot: plot

        distplot of y variable as well as its mu and sigma values.

    

    probplot: plot

        probplot of y variable as well quantiles and least squares fit.

    '''

    

    # set up the axes figure layout and total size for the plot

    fig, ax = plt.subplots(nrows=1,ncols=2)

    fig.set_size_inches(12, 12)



    # plot the distribution and fitted parameters on the first subplot

    plt.subplot(1,2,1)

    sns.distplot(train_df[response_variable] , fit=norm);

    (mu, sigma) = norm.fit(df_train[response_variable])

    plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

                loc='best')

    plt.ylabel('Frequency')

    plt.title(response_variable + ' distribution')



    # plot the the QQ-plot with quantiles and least squares fit on the second subplot

    plt.subplot(1,2,2)

    res = stats.probplot(train_df[response_variable], plot=plt)

    plt.show()
def percent_missing_data(df):

    '''

    Gets the percentage of data missing (where data missing is greater than 0) for every column in the dataframe and provides this information in a barplot 

    

    Parameters

    ----------

    

    df: dataframe

        dataframe object containing the dataframe to be cheacked for missing values.

    

    Returns

    -------

    

    barplot: plot

        barplot of percent of all missing values that are greater than 0.

    '''

    # calculate total isnull values then work out their percentage of total values then concat these into a dataframe

    total = df.isnull().sum().sort_values(ascending=False)

    percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)

    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

    

    # create a mask to remove any columns from dataframe that dont have missing values

    mask = missing_data['Percent'] > 0

    missing_data = missing_data[mask]

    

    

    # plot the the barplot with missing value percentage, and x labels rotated by 90 degrees, on the created axis

    f, ax = plt.subplots(figsize=(15, 12))

    plt.xticks(rotation='90')

    sns.barplot(x=missing_data.index, y=missing_data.Percent)

    plt.xlabel('Features', fontsize=15)

    plt.ylabel('Percent of missing values', fontsize=15)

    plt.title('Percent missing data by feature', fontsize=15)
def variable_imputation_check(string,df,train_df,response_value,subplots='N',ncols=1,add_rows=1,fig_size=(12,12)):

    '''

    Gets the counts of the categories present, within the various columns, containing the given string. For each of these a countplot or regplot is then displayed.

    

    Parameters

    ----------

    

    string: str

        str object containing the string to select columns from the dataframe.

        

    df: dataframe

        dataframe object containing the datapoints that need to be counted.

        

    train_df: dataframe

        dataframe object containing the original values of the variable that needs to be predicted datapoints that need to be counted.

        

    response_value: str

        str object containing the response varaible from the tainset as a string.

        

    subplots: str (default='N')

        str object containing 'Y' or 'N'. Used if multiple countplots need to be drawn

        

    ncols: int (default=1)

        int value specifiying the amount of columns to create within the sub plot

        

    add_rows: int (default=1)

        int value specifiying the amount of rows to add to the subplot, should they be needed

        

    fig_size: array (default=(12,12))

        array-like object containing the lenght and width of the figure to be drawn in inches

    

    Returns

    -------

    

    countplots: plot

        countplots of column values specified within string.

        

    regplots: plot

        regplots of column values, that have 20 > unique values, against the response variable.

    '''

    

    #create the base dataframe to work with

    String_cols = list([col for col in df.columns if string in col])

    df_string = df[String_cols]



    # if there are going to be subplots

    if subplots == 'Y':

        

        # define number of rows(based on number of items in String_cols), create plot axis and their size

        nrows = int(round((len(String_cols)/ncols),0)+add_rows)

        fig, ax = plt.subplots(nrows=nrows,ncols=ncols)

        fig.set_size_inches(fig_size)

        

        # instantiate count parameter(used in subplot positioning)

        count = 1

        

        # loop through the elements of String_cols

        for x in String_cols:

            

            # if number of categories of x > 20 create a regplot of this category on a given subplot and increase count by 1 

            if len(df[x].unique()) > 16:

                df_string[response_value] = train_df[response_value]

                plt.subplot(nrows,ncols,count)

                sns.regplot(y=df_string[response_value], x=df_string[x])

                count += 1

                

            # if number of categories of x < 20 create a countplot of this category on a given subplot and increase count by 1 

            else:

                plt.subplot(nrows,ncols,count)

                sns.countplot(df_string[x])

                count += 1

                

    # if there are not going to be subplots create a countplot

    else:

        sns.countplot(df[string])  
def variable_group_check(string,df):

    '''

    Generates a subsetted dataframe containing only columns that match the given string as well as the description of that dataframe.

    

    Parameters

    ----------

        

    string: str

        str object containing the string to select columns from the dataframe.

        

    df: dataframe

        dataframe object to be subsetted by string.

    

    Returns

    -------

    

    df_string_description: dataframe

        dataframe containing the description of all elements present within the dataframe(categorical and numerical)

    

    df_string: dataframe

        dataframe containing the datapoints of the subsetted orginal dataframe.

    '''

    

    # create a dataframe containing only columns that match the string parameter

    String_cols = list([col for col in df.columns if string in col])

    df_string = df[String_cols]

    

    # return both the dataframe as well as the description of its contents

    return df_string.describe(include='all'), df_string
def variable_correlation_check(string,df,train_df,response_variable,ordinal_variables=[],categorical_variables=[],color='PiYG',

                               fig_size=(12,12)):

    '''

    Gets the correlation between all columns containing the string and the response variable and displays this as a heatmap. 

    It also labels ordinal variables based on the measurement scale and value scale provided as well as dummy encoding all categorical variables.

    

    Parameters

    ----------

    

    string: str

        str object containing the string to select columns from the dataframe.

    

    df: dataframe

        dataframe object to be subsetted by string.

        

    train_df: dataframe

        dataframe object containing the original values of the response variable.

        

    response_variable: str

        str object containing the response variable.

        

    ordinal_variables: list (default=[])

        list-like object containing the ordinal variables to encode.

        

    categorical_variables: list

        list-like object containing the categorical variables to dummy encode.

        

    color: str (default='PiYG')

        str object containing the cmap for the heatmap color scheme.

        

    fig_size: array (default=(12,12))

        array-like object containing the lenght and width of the figure to be drawn in inches

    

    Returns

    -------

    

    sns.heatmap: plot

        annotated heatmap of all variables.

    '''

    

    #create the base dataframe to work with

    String_cols = list([col for col in df.columns if string in col])

    df_string = df[String_cols]

    

    # if there are no categorical variables

    if categorical_variables == []:

        

        # loop through list of ordinal variables and replace values with user defined input

        for x in ordinal_variables: 

            

            # strip unnessecary characters

            x = str(x)    

            x = x.replace('"' , '')

            x = x.replace('[' , '')

            x = x.replace("'" , '')

            x = x.replace(']' , '')

            

            # ask for user input for the measurement scale and value scale and replace their values

            measurement_scale = input('Input the measurement scale to be used for ' + x + ':')

            measurement_scale = measurement_scale.split()

            value_scale = input('Input the value scale to be used for ' + x + ':')

            value_scale = value_scale.split()

            df_string[x] = df_string[x].replace(to_replace=measurement_scale, value=value_scale)

            

            # convert replaced values into int type within dataframe

            df_string[x] = df_string[x].astype(int)



        # get response variable from train data and plot heatmap

        df_string[response_variable] = train_df[response_variable]

        corrmat = df_string.corr()

        f, ax = plt.subplots(figsize=fig_size)

        sns.heatmap(corrmat, vmax=.8, square=True, cmap=color, annot=True)

    

    # if there are no ordinal variables

    elif ordinal_variables == []:

        

        # loop through and dummy encode all categorical variable values    

        for x in categorical_variables:

            dummies = pd.get_dummies(df_string[str(x)], prefix=str(x))

            df_string = pd.concat([df_string, dummies], axis=1)

            

        # get response variable from train data and plot heatmap

        df_string[response_variable] = train_df[response_variable]

        corrmat = df_string.corr()

        f, ax = plt.subplots(figsize=fig_size)

        sns.heatmap(corrmat, vmax=.8, square=True, cmap=color, annot=True)

    

    # if there are both ordinal and categorical variables

    else:

        

        #loop through list of ordinal variables and replace values with user defined input

        for x in ordinal_variables:

            

            # strip unnessecary characters

            x = str(x)    

            x = x.replace('"' , '')

            x = x.replace('[' , '')

            x = x.replace("'" , '')

            x = x.replace(']' , '')

            

            # ask for user input for the measurement scale and value scale and replace their values

            measurement_scale = input('Input the measurement scale to be used for ' + x + ':')

            measurement_scale = measurement_scale.split()

            value_scale = input('Input the value scale to be used for ' + x + ':')

            value_scale = value_scale.split()

            df_string[x] = df_string[x].replace(to_replace=measurement_scale, value=value_scale)

            

            # convert replaced values into int type within dataframe

            df_string[x] = df_string[x].astype(int)

            

        # loop through and dummy encode all categorical variable values  

        for x in categorical_variables:

            dummies = pd.get_dummies(df_string[str(x)], prefix=str(x))

            df_string = pd.concat([df_string, dummies], axis=1)

            

        # get response variable from train data and plot heatmap

        df_string[response_variable] = train_df[response_variable]

        corrmat = df_string.corr()

        f, ax = plt.subplots(figsize=fig_size)

        sns.heatmap(corrmat, vmax=.8, square=True, cmap=color, annot=True)
def outlier_detection(corrmat,correlator,corr_score,df,train_df,ncols):

    '''

    Gets the regplots of the top correlated variables to the specified correlator

    

    Parameters

    ----------

    

    corrmat: dataframe

        dataframe object containing the correlation matrix to be subsetted. 

    

    correlator: str

        str object containing the variable to check for correlations with in the analysis.

        

    corr_score: int

        int value specifiying the cutoff correlation value for varaibles to make the plot

        

    df: dataframe

        dataframe object containing the dataframe to be subsetted.

        

    train_df: dataframe

        dataframe object containing the original values of the variable that needs to be predicted datapoints that need to be counted.

        

    ncols: int 

        int value specifiying the amount of coloumns to create within the sub plot

        

    Returns

    -------

        

    regplots: plot

        regplots of variables meeting the corr_score requirements.

    '''

    

    # create a mask that selects, based on corr_score, columns from the orginal corrmat

    mask = corrmat[correlator] >= corr_score

    corrmat = corrmat[mask]

    corrmat_cols = list(corrmat[correlator].index)

    

    # define number of rows(based on number of items in corrmat_cols)

    nrows = int(round(len(corrmat_cols)/3,0))+1

    

    # create plot axis and their size and subset to contain only train data

    fig, ax = plt.subplots(nrows=nrows,ncols=ncols)

    fig.set_size_inches(32, 40)

    variable = df[:ntrain]

    variable[correlator] = train_df[correlator]

    

    # instantiate count

    count = 0

    

    # loop through columns in corrmat_cols and create regplots for each who's index is based on count

    for col in corrmat_cols:

        count+=1

        plt.subplot(nrows,ncols,count)

        sns.regplot(y=variable[correlator], x=variable[col])

        #plt.title('Exterior1st Countplot', fontsize=15)
def ordered_label_encoder(df, col_names, to_replace, replace_values):

    '''

    Replaces the values of data, for the columns in col_list, in the to_replace list with the values present in the replace_values list. These changes are done to the base dataframe.

    

    Parameters

    ----------

    

    df: dataframe

        dataframe object containing the dataframe to be subsetted.

    

    col_names: list

        list_like object containing the list of columns that the replacement must be applied to

        

    to_replace: list

        list_like object containing the measurement scale of the ordinal variable

        

    replace_values: list

        list_like object containing the value scale of the replacement elements

    

    Returns

    -------

    

    df: dataframe

        altered original dataframe.

    '''

    # loop through columns present in col_names

    for col in col_names:

        

        # replace to_replace values with replace_values

        df[col] = df[col].replace(to_replace=to_replace, value=replace_values)
# import from kaggle dataset

df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')



#Save the 'Id' column for later use in model predictions

df_train_Id = df_train['Id']

df_test_Id = df_test['Id']



#Now drop the  'Id' column from the base dataframe as it interferes with the missing number calculations

df_train.drop("Id", axis = 1, inplace = True)

df_test.drop("Id", axis = 1, inplace = True)
response_variable_distribution('SalePrice',df_train)
# Use log transformation (log(1+x)) via the the numpy fuction log1p to SalePrice

df_train["SalePrice"] = np.log1p(df_train["SalePrice"])
response_variable_distribution('SalePrice',df_train)
# split off sales price from train data into y_train for later use in the modelling section

y_train = df_train.SalePrice.values



# create mask varaibles for test and train subsetting later on

ntrain = df_train.shape[0]

ntest = df_test.shape[0]



# concatenate train and test dataframes, drop SalePrice column and print the shape of the new all_data dataframe

all_data = pd.concat((df_train, df_test)).reset_index(drop=True)

all_data.drop(['SalePrice'], axis=1, inplace=True)

print("all_data size is : {}".format(all_data.shape))
percent_missing_data(all_data)
variable_imputation_check('Pool',all_data,df_train,'SalePrice',subplots='Y',ncols=2,add_rows=0)
all_data['PoolQC'] = all_data['PoolQC'].fillna('None')
variable_imputation_check('MiscFeature',all_data,df_train,'SalePrice')
all_data['MiscFeature'] = all_data['MiscFeature'].fillna('None')
variable_imputation_check('MiscFeature',all_data,df_train,'SalePrice')
# was initially dropped

#all_data = all_data.drop(['MiscFeature'], axis=1)
variable_imputation_check('Alley',all_data,df_train,'SalePrice')
all_data['Alley'] = all_data['Alley'].fillna('None')
variable_imputation_check('Fence',all_data,df_train,'SalePrice')
all_data['Fence'] = all_data['Fence'].fillna('None')
variable_imputation_check('Fireplace',all_data,df_train,'SalePrice',subplots='Y', ncols=2, add_rows=0, fig_size=(12, 12))
all_data['FireplaceQu'] = all_data['FireplaceQu'].fillna('None')
variable_correlation_check(string='Fireplace',df=all_data,train_df=df_train,response_variable='SalePrice',ordinal_variables=['FireplaceQu'],color='YlGn')
# subset orginal dataframe by train (to prevent missing SalePrice values from affecting the plots) and then concatenate SalePrice to this new dataframe

lot_frontage = all_data[:ntrain]



#concatenate SalePrice to this new dataframe and then subset it to contain only SalePrice and LotFrontage

lot_frontage = pd.concat([lot_frontage, df_train[['SalePrice']]], axis=1)

lot_frontage = lot_frontage[['SalePrice','LotFrontage']]



# subset orginal dataframe by train (to prevent missing SalePrice values from affecting the plots) and then concatenate SalePrice to this new dataframe

lot_frontage_0fill = all_data[:ntrain]



#concatenate SalePrice to this new dataframe, fill NA values with 0 and then subset it to contain only SalePrice and LotFrontage

lot_frontage_0fill = pd.concat([lot_frontage_0fill, df_train[['SalePrice']]], axis=1).fillna(value=0)

lot_frontage_0fill = lot_frontage_0fill[['SalePrice','LotFrontage']]



# subset orginal dataframe by train (to prevent missing SalePrice values from affecting the plots) and then concatenate SalePrice to this new dataframe

lot_frontage_grouped_medianfill = all_data[:ntrain]



#concatenate SalePrice to this new dataframe, fill NA values wiht the median by grouped neighborhood and then subset it to contain only SalePrice and LotFrontage

lot_frontage_grouped_medianfill = pd.concat([lot_frontage_grouped_medianfill, df_train[['SalePrice']]], axis=1)

lot_frontage_grouped_medianfill = df_train.groupby("Neighborhood")['LotFrontage', 'SalePrice'].transform(

    lambda x: x.fillna(x.median()))
# set up the axes, figure layout and total size for the plot

fig, ax = plt.subplots(nrows=1,ncols=3)

fig.set_size_inches(12, 12)



# plot the first subplot (Lot Frontage Without Fill)

plt.subplot(1,3,1)

sns.regplot(y=lot_frontage['SalePrice'], x=lot_frontage['LotFrontage'])

plt.title('Lot Frontage Without Fill', fontsize=12)



# plot the second subplot (Lot Frontage With 0 Fill)

plt.subplot(1,3,2)

sns.regplot(y=lot_frontage_0fill['SalePrice'], x=lot_frontage_0fill['LotFrontage'])

plt.title('Lot Frontage With 0 Fill', fontsize=12)



# plot the third subplot (Lot Frontage Grouped Median Fill)

plt.subplot(1,3,3)

sns.regplot(y=lot_frontage_grouped_medianfill['SalePrice'], x=lot_frontage_grouped_medianfill['LotFrontage'])

plt.title('Lot Frontage With Grouped Median Fill', fontsize=12)
all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(

    lambda x: x.fillna(x.median()))
variable_imputation_check('Lot',all_data,df_train,'SalePrice',subplots='Y', ncols=2, add_rows=0, fig_size=(12,12))
variable_correlation_check(string='Lot',df=all_data,train_df=df_train,response_variable='SalePrice',ordinal_variables=['LotShape'],categorical_variables=['LotConfig'])
variable_imputation_check('Garage',all_data,df_train,response_value='SalePrice',subplots='Y', ncols=3, fig_size=(16, 16))
all_data[['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']] = all_data[['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']].fillna('None')
all_data[['GarageYrBlt', 'GarageArea', 'GarageCars']] = all_data[['GarageYrBlt', 'GarageArea', 'GarageCars']].fillna(0)
variable_correlation_check(string='Garage',df=all_data,train_df=df_train,response_variable='SalePrice',ordinal_variables=['GarageQual','GarageCond','GarageFinish'],categorical_variables=['GarageType'],fig_size=(16,16))
variable_imputation_check('Bsmt',all_data,df_train,response_value='SalePrice',subplots='Y', ncols=3, add_rows=0, fig_size=(18, 18))
all_data[['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath']] = all_data[['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath']].fillna(0)
all_data[['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']] = all_data[['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']].fillna('None')
variable_correlation_check(string='Bsmt',df=all_data,train_df=df_train,response_variable='SalePrice',ordinal_variables=['BsmtQual','BsmtCond','BsmtExposure'],categorical_variables=['BsmtFinType1','BsmtFinType2'],fig_size=(16,16))
variable_imputation_check('Mas',all_data,df_train,response_value='SalePrice',subplots='Y', ncols=2, add_rows=0, fig_size=(12, 12))
all_data[['MasVnrType']] = all_data[['MasVnrType']].fillna('None')
all_data[['MasVnrArea']] = all_data[['MasVnrArea']].fillna(0)
variable_correlation_check(string='Mas',df=all_data,train_df=df_train,response_variable='SalePrice',categorical_variables=['MasVnrType'])
variable_imputation_check('MS',all_data,df_train,response_value='SalePrice',subplots='Y', ncols=2, add_rows=0, fig_size=(12, 12))
all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])
all_data['MSSubClass'] = all_data['MSSubClass'].fillna("None")
# convert MSSubclass to string as it is currently represented as an integer

all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)
variable_correlation_check(string='MS',df=all_data,train_df=df_train,response_variable='SalePrice',categorical_variables=['MSSubClass','MSZoning'],fig_size=(18,18))
variable_imputation_check('Utilities',all_data,df_train,response_value='SalePrice')
all_data['Utilities'] = all_data['Utilities'].fillna("None")
variable_imputation_check('Functional',all_data,df_train,'SalePrice')
all_data['Functional'] = all_data['Functional'].fillna('Typ')
variable_imputation_check('Electrical',all_data,df_train,response_value='SalePrice')
all_data['Electrical'] = all_data['Electrical'].fillna('SBrkr')
variable_imputation_check('Kitchen',all_data,df_train,response_value='SalePrice',subplots='Y', ncols=2, add_rows=0, fig_size=(12, 12))
all_data['KitchenQual'] = all_data['KitchenQual'].fillna('TA')
variable_correlation_check(string='Kitchen',df=all_data,train_df=df_train,response_variable='SalePrice',ordinal_variables=['KitchenQual'], color='YlGn')
all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])

all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])
variable_imputation_check('SaleType',all_data,df_train,response_value='SalePrice')
all_data['SaleType'] = all_data['SaleType'].fillna('WD')
# create total and percent measures of missing data and then concatenate these into a dataframe

total = all_data.isnull().sum().sort_values(ascending=False)

percent = (all_data.isnull().sum()/all_data.isnull().count()).sort_values(ascending=False)*100

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(1)
variable_imputation_check('SF',all_data,df_train,response_value='SalePrice',subplots='Y', ncols=3, add_rows=0, fig_size=(18, 18))
variable_correlation_check(string='SF',df=all_data,train_df=df_train,response_variable='SalePrice')
# feature enginner a TotalSF measure to see if it correlates better with SalePrice

all_data['TotalSF'] = all_data['1stFlrSF'] + all_data['2ndFlrSF'] +  all_data['TotalBsmtSF']
variable_imputation_check('Qual',all_data,df_train,response_value='SalePrice',subplots='Y', ncols=3, add_rows=0, fig_size=(18, 18))
variable_correlation_check(string='Qual',df=all_data,train_df=df_train,response_variable='SalePrice',ordinal_variables=['BsmtQual','GarageQual','ExterQual','KitchenQual'])
variable_imputation_check('Cond',all_data,df_train,response_value='SalePrice',subplots='Y', ncols=3, add_rows=1, fig_size=(18, 18))
variable_correlation_check(string='Cond',df=all_data,train_df=df_train,response_variable='SalePrice',ordinal_variables=['BsmtCond','GarageCond','ExterCond',],categorical_variables=['SaleCondition','Condition1','Condition2'],fig_size=(18,18))
variable_imputation_check('Street',all_data,df_train,response_value='SalePrice')
variable_imputation_check('Land',all_data,df_train,response_value='SalePrice',subplots='Y', ncols=2, add_rows=0)
variable_correlation_check(string='Land',df=all_data,train_df=df_train,response_variable='SalePrice',ordinal_variables=['LandSlope'],categorical_variables=['LandContour'])
variable_imputation_check('Neighborhood',all_data,df_train,response_value='SalePrice',fig_size=(18, 18))
variable_correlation_check(string='Neighborhood',df=all_data,train_df=df_train,response_variable='SalePrice',categorical_variables=['Neighborhood'],fig_size=(18,18),color='YlGn')
variable_imputation_check('BldgType',all_data,df_train,response_value='SalePrice')
variable_correlation_check(string='BldgType',df=all_data,train_df=df_train,response_variable='SalePrice',categorical_variables=['BldgType'])
variable_imputation_check('HouseStyle',all_data,df_train,response_value='SalePrice')
variable_correlation_check(string='HouseStyle',df=all_data,train_df=df_train,response_variable='SalePrice',categorical_variables=['HouseStyle'])
fig, ax = plt.subplots(nrows=1,ncols=2)

fig.set_size_inches(16, 16)



plt.subplot(1,2,1)

sns.countplot(all_data['RoofStyle'])

plt.title('RoofStyle Countplot', fontsize=15)



plt.subplot(1,2,2)

sns.countplot(all_data['RoofMatl'])

plt.title('RoofStyle Countplot', fontsize=15)
variable_correlation_check(string='Roof',df=all_data,train_df=df_train,response_variable='SalePrice',categorical_variables=['RoofStyle','RoofMatl'],fig_size=(16,16))
variable_imputation_check(string='Heating',df=all_data,train_df=df_train,response_value='SalePrice',subplots='Y',ncols=2,add_rows=0,fig_size=(12,12))
variable_correlation_check(string='Heating',df=all_data,train_df=df_train,response_variable='SalePrice',ordinal_variables=['HeatingQC'],categorical_variables=['Heating'])
variable_imputation_check(string='CentralAir',df=all_data,train_df=df_train,response_value='SalePrice')
variable_correlation_check(string='CentralAir',df=all_data,train_df=df_train,response_variable='SalePrice',ordinal_variables=['CentralAir'],color='YlGn')
#Year and month sold are transformed into categorical features.

all_data['YrSold'] = all_data['YrSold'].astype(str)

all_data['MoSold'] = all_data['MoSold'].astype(str)
ordered_label_encoder(all_data,['FireplaceQu','BsmtQual','GarageFinish','GarageQual','ExterQual','KitchenQual','HeatingQC','ExterCond','BsmtCond','GarageCond','PoolQC'],['None','Po','Fa','TA','Gd','Ex'],[0,1,2,3,4,5])

ordered_label_encoder(all_data,['LotShape'],['IR3','IR2','IR1','Reg'],[0,1,2,3])

ordered_label_encoder(all_data,['GarageFinish'],['None','Unf','RFn','Fin'],[0,1,2,3])

ordered_label_encoder(all_data,['CentralAir','PavedDrive'],['N','Y'],[0,1])

ordered_label_encoder(all_data,['Street'],['Grvl','Pave'],[0,1])

ordered_label_encoder(all_data,['Alley'],['None','Grvl','Pave'],[0,1,2])

ordered_label_encoder(all_data,['LandSlope'],['Sev','Mod','Gtl'],[0,1,2])

ordered_label_encoder(all_data,['BsmtExposure'],['None','No','Mn','Av','Gd'],[0,1,2,3,4])

ordered_label_encoder(all_data,['Functional'],['Sal','Sev','Maj2','Maj1','Mod','Min2','Min1','Typ'],[0,1,2,3,4,5,6,7])
# subset the original dataframe by the train mask created earlier and add SalePrice

numerical_heatmap = all_data[:ntrain]

numerical_heatmap['SalePrice'] = df_train['SalePrice']



# plot heatmap of correlation values

corrmat = numerical_heatmap.corr()

f, ax = plt.subplots(figsize=(32, 32))

sns.heatmap(corrmat, vmax=.8, square=True, cmap="PiYG", annot=True);
outlier_detection(corrmat=corrmat, correlator='SalePrice', corr_score=0.30, df=all_data, train_df=df_train, ncols=3)
all_data = pd.get_dummies(all_data)

print(all_data.shape)
train = all_data[:ntrain]

print(train.shape)

test = all_data[ntrain:]

print(train.shape)
train = all_data[:ntrain]

train['SalePrice'] = df_train['SalePrice']



# outliers were dropped using the previous graphs as visual aids for value criteria

train = train.drop(train[(train['1stFlrSF']>4000) & (df_train['SalePrice']<13)].index)

train = train.drop(train[(train['BsmtFinSF1']>4000) & (df_train['SalePrice']<13)].index)

train = train.drop(train[(train['GarageArea']>1220) & (df_train['SalePrice']<13)].index)

train = train.drop(train[(train['GrLivArea']>4000) & (df_train['SalePrice']<13)].index)

train = train.drop(train[(train['LotFrontage']>300) & (df_train['SalePrice']<13)].index)

train = train.drop(train[(train['OpenPorchSF']>500) & (df_train['SalePrice']<11)].index)

train = train.drop(train[(train['TotalBsmtSF']>6000) & (df_train['SalePrice']<13)].index)

y_train = train.SalePrice.values



# SalePrice was dropped from the train data as it is the response variable

train = train.drop(['SalePrice'], axis=1)

print(train.shape)

print(y_train.shape)
from sklearn.linear_model import ElasticNet, Lasso, Ridge, BayesianRidge

from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor

from sklearn.kernel_ridge import KernelRidge

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.preprocessing import StandardScaler

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

from sklearn.model_selection import KFold, cross_val_score, train_test_split

from sklearn.metrics import mean_squared_error

import optuna
#Validation function

n_folds = 5



# 

def rmsle_cv(model):

    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)

    rmse= np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv = kf))

    return(rmse)
# Define an objective function to be minimized.

def objective(trial):



    # Invoke suggest methods of a Trial object to generate hyperparameters.

    alpha = trial.suggest_loguniform('alpha', 1e-10, 1)

    fit_intercept = trial.suggest_categorical('fit_intercept', [True,False])

    normalize=trial.suggest_categorical('normalize', [True,False])

    precompute=trial.suggest_categorical('precompute', [True,False])

    max_iter=trial.suggest_int('max_iter', 1000, 10000)

    tol=trial.suggest_loguniform('tol', 1e-10, 1)

    warm_start=trial.suggest_categorical('warm_start', [True,False])

    positive=trial.suggest_categorical('positive', [True,False])

    random_state=trial.suggest_int('random_state', 1, 10)

    selection=trial.suggest_categorical('selection', ['cyclic','random'])

    

    # create a variable containing the model and a set of selected hyperparameter values

    classifier_obj = Lasso(alpha=alpha,

                           fit_intercept=fit_intercept,

                           normalize=normalize,

                           precompute=precompute,

                           max_iter=max_iter,

                           tol=tol,

                           warm_start=warm_start,

                           positive=positive,

                           random_state=random_state,

                           selection=selection)



    # define x and y variables

    x, y = train,y_train

    

    # check cross validation score of the model based on x and y values

    score = cross_val_score(classifier_obj, x, y)

    accuracy = score.mean()

    

    # A objective value linked with the Trial object.

    return 1.0 - accuracy  



# Create a new study and invoke optimization of the objective function

study = optuna.create_study() 

study.optimize(objective, n_trials=1000)
# used to print the optimal hyperparameters found by the objective function

study.best_params
# run the model using the the optimised hyperparameters

lasso = make_pipeline(RobustScaler(), Lasso(alpha=0.0007699830238031739,

                                            fit_intercept=True,

                                            normalize=False,

                                            precompute=False,

                                            max_iter=8117,

                                            tol=1.203470716648057e-07,

                                            warm_start=False,

                                            positive=True,

                                            random_state=7,

                                            selection='cyclic'))
# check the cross-validation score of the model from the train data

score = rmsle_cv(lasso)

print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
# scale the data using RobustScaler then fit the model to our dataframe

scaler = RobustScaler()

train2 = scaler.fit_transform(train)

lasso2 = Lasso(alpha=0.0007699830238031739,

               fit_intercept=True,

               normalize=False,

               precompute=False,

               max_iter=8117,

               tol=1.203470716648057e-07,

               warm_start=False,

               positive=True,

               random_state=7,

               selection='cyclic')

lasso2.fit(train, y_train)



# get coeffecients from the lasso model and use them to remove all columns that have a coeffecient of 0

coeff = pd.DataFrame(lasso2.coef_, all_data.columns, columns=['Coefficient']).reset_index()

Cols_to_remove = coeff['Coefficient'] == 0 

Cols_to_remove = coeff[Cols_to_remove]

Cols_to_remove = list(Cols_to_remove['index'])

all_data = all_data.drop(Cols_to_remove, axis=1)
train = all_data[:ntrain]

print(train.shape)

test = all_data[ntrain:]

print(train.shape)
train = all_data[:ntrain]

train['SalePrice'] = df_train['SalePrice']

#train = train.drop(train[(train['1stFlrSF']>4000) & (df_train['SalePrice']<13)].index)

train = train.drop(train[(train['BsmtFinSF1']>4000) & (df_train['SalePrice']<13)].index)

train = train.drop(train[(train['GarageArea']>1220) & (df_train['SalePrice']<13)].index)

train = train.drop(train[(train['GrLivArea']>4000) & (df_train['SalePrice']<13)].index)

train = train.drop(train[(train['LotFrontage']>300) & (df_train['SalePrice']<13)].index)

train = train.drop(train[(train['OpenPorchSF']>500) & (df_train['SalePrice']<11)].index)

#train = train.drop(train[(train['TotalBsmtSF']>6000) & (df_train['SalePrice']<13)].index)

y_train = train.SalePrice.values

train = train.drop(['SalePrice'], axis=1)

print(train.shape)

print(y_train.shape)
# Define an objective function to be minimized.

def objective(trial):



    # Invoke suggest methods of a Trial object to generate hyperparameters.

    alpha = trial.suggest_loguniform('alpha', 1e-10, 1)

    fit_intercept = trial.suggest_categorical('fit_intercept', [True,False])

    normalize=trial.suggest_categorical('normalize', [True,False])

    precompute=trial.suggest_categorical('precompute', [True,False])

    max_iter=trial.suggest_int('max_iter', 1000, 10000)

    tol=trial.suggest_loguniform('tol', 1e-10, 1)

    warm_start=trial.suggest_categorical('warm_start', [True,False])

    positive=trial.suggest_categorical('positive', [True,False])

    random_state=trial.suggest_int('random_state', 1, 10)

    selection=trial.suggest_categorical('selection', ['cyclic','random'])

    

    # create a variable containing the model and a set of selected hyperparameter values

    classifier_obj = Lasso(alpha=alpha,

                           fit_intercept=fit_intercept,

                           normalize=normalize,

                           precompute=precompute,

                           max_iter=max_iter,

                           tol=tol,

                           warm_start=warm_start,

                           positive=positive,

                           random_state=random_state,

                           selection=selection)



    # define x and y variables

    x, y = train,y_train

    

     # check cross validation score of the model based on x and y values

    score = cross_val_score(classifier_obj, x, y)

    accuracy = score.mean()

    

    # A objective value linked with the Trial object.

    return 1.0 - accuracy



# Create a new study and invoke optimization of the objective function

study = optuna.create_study() 

study.optimize(objective, n_trials=1000)
# used to print the optimal hyperparameters found by the objective function

study.best_params
# run the model using the the optimised hyperparameters

lasso = make_pipeline(RobustScaler(), Lasso(alpha=3.0340354779991275e-05,

                                            fit_intercept=True,

                                            normalize=True,

                                            precompute=True,

                                            max_iter=5941,

                                            tol=9.623735660996391e-10,

                                            warm_start=False,

                                            positive=True,

                                            random_state=3,

                                            selection='cyclic'))
# check the cross-validation score of the model from the train data

score = rmsle_cv(lasso)

print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
# Define an objective function to be minimized.

def objective(trial):



    # Invoke suggest methods of a Trial object to generate hyperparameters.

    alpha = trial.suggest_loguniform('alpha', 1e-10, 1)

    l1_ratio = trial.suggest_uniform('l1_ratio', 0.0, 1.0)

    fit_intercept = trial.suggest_categorical('fit_intercept', [True,False])

    normalize=trial.suggest_categorical('normalize', [True,False])

    precompute=trial.suggest_categorical('precompute', [True,False])

    max_iter=trial.suggest_int('max_iter', 1000, 10000)

    tol=trial.suggest_loguniform('tol', 1e-10, 1)

    warm_start=trial.suggest_categorical('warm_start', [True,False])

    positive=trial.suggest_categorical('positive', [True,False])

    random_state=trial.suggest_int('random_state', 1, 10)

    selection=trial.suggest_categorical('selection', ['cyclic','random'])

    

    # create a variable containing the model and a set of selected hyperparameter values

    classifier_obj = ElasticNet(alpha=alpha,

                                l1_ratio=l1_ratio,

                                fit_intercept=fit_intercept,

                                normalize=normalize,

                                precompute=precompute,

                                max_iter=max_iter,

                                tol=tol,

                                warm_start=warm_start,

                                positive=positive,

                                random_state=random_state,

                                selection=selection)



    # define x and y variables

    x, y = train,y_train

    

     # check cross validation score of the model based on x and y values

    score = cross_val_score(classifier_obj, x, y)

    accuracy = score.mean()

    

    # A objective value linked with the Trial object.

    return 1.0 - accuracy



# Create a new study and invoke optimization of the objective function

study = optuna.create_study() 

study.optimize(objective, n_trials=1000)
# used to print the optimal hyperparameters found by the objective function

study.best_params
# run the model using the the optimised hyperparameters

ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=6.869814939303578e-05,

                                                l1_ratio=0.15676425376728362,

                                                fit_intercept=True,

                                                normalize=True,

                                                precompute=True,

                                                max_iter=9216,

                                                tol=0.0220818991822515,

                                                warm_start=False,

                                                positive=True,

                                                random_state=7,

                                                selection='random'))
# check the cross-validation score of the model from the train data

score = rmsle_cv(ENet)

print("\nENet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
# Define an objective function to be minimized.

def objective(trial):



    # Invoke suggest methods of a Trial object to generate hyperparameters.

    n_iter=trial.suggest_int('n_iter', 1, 1000)

    tol=trial.suggest_loguniform('tol', 1e-10, 1)

    alpha_1=trial.suggest_loguniform('alpha1', 1e-10, 10)

    alpha_2=trial.suggest_loguniform('alpha2', 1e-10, 10)

    lambda_1=trial.suggest_loguniform('lambda_1', 1e-10, 10)

    lambda_2=trial.suggest_loguniform('lambda_2', 1e-10, 10)

    compute_score=trial.suggest_categorical('compute_score', [True,False])

    fit_intercept=trial.suggest_categorical('fit_intercept', [True,False])

    normalize=trial.suggest_categorical('normalize', [True,False])

    verbose=trial.suggest_categorical('verbose', [True,False])

    

    # create a variable containing the model and a set of selected hyperparameter values

    classifier_obj = BayesianRidge(n_iter=n_iter,

                                   tol=tol,

                                   alpha_1=alpha_1,

                                   alpha_2=alpha_2,

                                   lambda_1=lambda_1,

                                   lambda_2=lambda_2,

                                   compute_score=compute_score,

                                   fit_intercept=fit_intercept,

                                   normalize=normalize,

                                   verbose=verbose)



    # define x and y variables

    x, y = train,y_train

    

     # check cross validation score of the model based on x and y values

    score = cross_val_score(classifier_obj, x, y)

    accuracy = score.mean()

    

    # A objective value linked with the Trial object.

    return 1.0 - accuracy



# Create a new study and invoke optimization of the objective function

study = optuna.create_study() 

study.optimize(objective, n_trials=1000)
# used to print the optimal hyperparameters found by the objective function

study.best_params
# run the model using the the optimised hyperparameters

BayesianRidgeRegression = make_pipeline(RobustScaler(), BayesianRidge(n_iter=879,

                                                                      tol=2.99590539046086e-10,

                                                                      alpha_1=0.019444194793005434,

                                                                      alpha_2=4.7042655856621804,

                                                                      lambda_1=0.08808043997430755,

                                                                      lambda_2=0.045167765758061834,

                                                                      compute_score=True,

                                                                      fit_intercept=True,

                                                                      normalize=False,

                                                                      verbose=False,))
# check the cross-validation score of the model from the train data

score = rmsle_cv(BayesianRidgeRegression)

print("\nBayesianRidgeRegression score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):

    def __init__(self, models):

        self.models = models

        

    # we define clones of the original models to fit the data in

    def fit(self, X, y):

        self.models_ = [clone(x) for x in self.models]

        

        # Train cloned base models

        for model in self.models_:

            model.fit(X, y)



        return self

    

    #Now we do the predictions for cloned models and average them

    def predict(self, X):

        predictions = np.column_stack([

            model.predict(X) for model in self.models_

        ])

        return np.mean(predictions, axis=1)
averaged_models = AveragingModels(models = (lasso,ENet,BayesianRidgeRegression))

score = rmsle_cv(averaged_models)

print(" Averaged base models score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
# new rmsle for predicted values not the pipeline objects

def rmsle(y, y_pred):

    return np.sqrt(mean_squared_error(y, y_pred))
averaged_models.fit(train.values, y_train)

averaged_train_pred = averaged_models.predict(train.values)

averaged_models_pred = np.expm1(averaged_models.predict(test.values))

print(rmsle(y_train, averaged_train_pred))
sub = pd.DataFrame()

sub['Id'] = df_test_Id

sub['SalePrice'] = averaged_models_pred

sub.to_csv('submission.csv',index=False)