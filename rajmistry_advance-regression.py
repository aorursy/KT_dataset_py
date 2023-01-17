# Supress Warnings

import warnings

warnings.filterwarnings('ignore')



from warnings import simplefilter

simplefilter(action='ignore', category=FutureWarning)



#for data import and basic oprtaion

import pandas as pd

import numpy as np

 

#for visulization and plotting

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



#cross validation

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV



#linear regression

from sklearn.feature_selection import RFE

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Ridge

from sklearn.linear_model import Lasso



#Feature Scaling or normalization

from sklearn.preprocessing import StandardScaler



from sklearn.metrics import r2_score,mean_squared_error 

from sklearn.model_selection import train_test_split



from sklearn import metrics

from scipy.stats import skew



#Misc.

from IPython.core.display import display, HTML

display(HTML("<style>.container { width:90% !important; }</style>"))

pd.set_option('display.max_columns', None)  

pd.set_option('display.expand_frame_repr', False)

pd.set_option('max_colwidth', -1)
# Read columns for previous Leads Data Dictionary source file.

print(open('../input/data_description.txt', 'r').read()) 
def drop_columns(dataframe, axis =1, percent=0.3):

    '''

    * drop_columns function will remove the rows and columns based on parameters provided.

    * dataframe : Name of the dataframe  

    * axis      : axis = 0 defines drop rows, axis =1(default) defines drop columns    

    * percent   : percent of data where column/rows values are null,default is 0.3(30%)

    '''

    df = dataframe.copy()

    ishape = df.shape

    if axis == 0:

        rownames = df.transpose().isnull().sum()

        rownames = list(rownames[rownames.values > percent*len(df)].index)

        df.drop(df.index[rownames],inplace=True) 

        print("\nNumber of Rows dropped\t: ",len(rownames))

    else:

        colnames = (df.isnull().sum()/len(df))

        colnames = list(colnames[colnames.values>=percent].index)

        df.drop(labels = colnames,axis =1,inplace=True)        

        print("Number of Columns dropped\t: ",len(colnames))

        

    print("\nOld dataset rows,columns",ishape,"\nNew dataset rows,columns",df.shape)



    return df



def column_univariate(df,col,type,nrow=None,ncol=None,hue =None):

    

    '''

    * column_univariate will plot the graphs based on the parameters.

    * df      : dataframe name

    * col     : Column name

    * nrow    : no of rows in sub plot

    * ncol    : no of cols in sub plot

    * type : variable type : continuos or categorical

                Continuos(0)   : Distribution, Violin & Boxplot will be plotted.

                Categorical(1) : Countplot will be plotted.

                Categorical(2) : Subplot-Countplot will be plotted.

    * hue     : It's only applicable for categorical analysis.

    

    '''

    sns.set(style="darkgrid")

    

    if type == 0:

        

        fig, axes =plt.subplots(nrows =1,ncols=3,figsize=(12,6))

        axes [0].set_title(" Distribution Plot")

        sns.distplot(df[col],ax=axes[0])

        axes [1].set_title("Violin Plot")

        sns.violinplot(data =df, x=col,ax=axes[1], inner="quartile")

        axes [2].set_title(" Box Plot")

        sns.boxplot(data =df, x=col,ax=axes[2],orient='v')

        

        for ax in axes:

            ax.set_xlabel('Common x-label')

            ax.set_ylabel('Common y-label')

        

    if  type == 1:

        total_len = len(df[col])

        percentage_labels = round((df[col].value_counts()/total_len)*100,4)

    

        temp = pd.Series(data = hue)

        

        fig, ax=plt.subplots(nrows =1,ncols=1,figsize=(12,4))

        ax.set_title("Count Plot")

        width = len(df[col].unique()) + 6 + 4*len(temp.unique())

        fig.set_size_inches(width , 7)

        sns.countplot(data = df, x= col,

                           order=df[col].value_counts().index,hue = hue)  

        mystring = col.replace("_", " ").upper()

        plt.xlabel(mystring)

          

        

        if len(temp.unique()) > 0:

            for p in ax.patches:

                ax.annotate('{:1.1f}%'.format((p.get_height()*100)/float(len(df))),

                            (p.get_x()+0.05, p.get_height()+20))  

        else:

            for p in ax.patches:

                height = p.get_height()

                ax.text(p.get_x() + p.get_width()/2.,

                height + 2,'{:.2f}%'.format(100*(height/total_len)),

                        fontsize=14, ha='center', va='bottom')

        del temp

        

    elif type == 2:

        fig, ax = plt.subplots(nrow, ncol, figsize=(24, 10))

        for variable, subplot in zip(col, ax.flatten()):

            total = float(len(df[variable]))

            ax=sns.countplot(data = df, x= variable,ax=subplot,

                           order=df[variable].value_counts().index) 

            for p in ax.patches:    

                height = p.get_height()

                ax.text(p.get_x()+p.get_width()/2., height + 3, '{:1.2f}%'.format((height/total)*100),

                        ha="center") 

    else:

        exit

    

    plt.tight_layout() 



def get_column_dummies(col,df_in):

    '''

    * get_column_dummies will get/map column dummies values based on the parameters.

    * col   : column name

    * df_in  : dataframe

    '''

    temp = pd.get_dummies(df_in[col], drop_first = True,prefix=col)

    df_out = pd.concat([df_in, temp], axis = 1)

    df_out.drop([col], axis = 1, inplace = True)  

    return df_out

  



def column_describe(df,col):

    '''

    * column_describe: Describe stastical based on the parameters.

    * df    : dataframe input

    * col   : column name

    '''

    print(df[col].describe())

    



def box_plot(df_in,numerical_cols):

    '''

    * box_plot: Plot box plot  based on the parameters.

    * df_in    : dataframe input

    * numerical_cols   : numerical column name

    '''

    plt.figure(figsize=(25,10))

    for var in numerical_cols:

        plt.subplot(5,8,numerical_cols.index(var)+1)

        sns.boxplot(y=var,palette='BuGn_r', data=df_in)

     

    plt.tight_layout()

    plt.show()



 

    

def performance_metric(n,k,y_pred,y_true):

    p_materic=[]

    resid=np.subtract(y_pred,y_true)

    rss=round(np.sum(np.power(resid,2)),2)

    p_materic.append(rss)

     

    aic=round(n*np.log(rss/n)+2*k,2)

    p_materic.append(aic)

    

    

    bic=round(n*np.log(rss/n)+k*np.log(n),2)

      

    p_materic.append(bic)

    

    r_square_score=round(r2_score(y_true,y_pred),4)

    p_materic.append(r_square_score)

     

    # adjusted r2 using formula adj_r2 = 1 - (1- r2) * (n-1) / (n - k - 1)

    # k = number of predictors = data.shape[1] - 1

    adj_r2 = round(1 - (1-r_square_score)*(n- 1) / (n - k - 1),2)

    p_materic.append(adj_r2)

    

    mae=round(metrics.mean_absolute_error(y_true, y_pred),2)

    p_materic.append(mae)

    mse=round(metrics.mean_squared_error(y_true, y_pred),2)

    p_materic.append(mse)

    rmse=round(np.sqrt(metrics.mean_squared_error(y_true, y_pred)),2)

    p_materic.append(rmse)

    cols=['RSS','AIC','BIC','Rˆ2 score','Adjust Rˆ2','MAE','MSE','RMSE']

    list_of_tuples = list(zip( cols,p_materic))

    pm = pd.DataFrame(list_of_tuples, columns = ['Metrics', 'Value'])

    print(pm.transpose())



     
#load the dataset

df = pd.DataFrame(pd.read_csv('../input/train.csv'))

#check top records

df.head() 
#check size of dataset 

df.shape
#check dataset columns

df.columns
#check info about the data

df.info()
#check info about data types

df.dtypes.value_counts()
#check number of unique classes in dataset

df.select_dtypes('object').apply(pd.Series.nunique, axis = 0)
#check first few records

df.head(3)
#check last few records

df.tail(3)
#describe the dataset for stastical view

df.describe().transpose()
#list numerical variables/features

numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

numerical_cols
#list categorical vriables/features

categorical_cols = df.select_dtypes(include=[np.object]).columns.tolist()

categorical_cols
f, ax = plt.subplots(figsize=(12, 9))

k = 15 #number of variables for heatmap

corr_matrix = df.corr()

cols = corr_matrix.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(df[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
plt.figure(figsize = (18,12))  

result_corr=df.corr(method='pearson')

sns.heatmap(result_corr[(result_corr >= 0.7) | (result_corr <= -0.7)], 

            cmap='YlGnBu', vmax=1.0, vmin=-1.0, linewidths=0.1,

            annot=True, annot_kws={"size": 10}, square=True);

 
#Check duplicate values/row in dataset

sum(df.duplicated(subset = 'Id')) == 0
df.isnull().sum()
#Compute percentage missing values 

missing_counts= round((df.isnull().sum() * 100/ len(df)),2).sort_values(ascending=False)

missing_counts[missing_counts > 0]
#Calculate and print percentage of missing values greater than 30% from each column.

percent_missing = round(df.isnull().sum() * 100 / len(df),2)

missing_value_df = pd.DataFrame({'column_name': df.columns,'percent_missing': percent_missing})

missing_value_df.sort_values('percent_missing', inplace=True)

missing_value_df=missing_value_df[missing_value_df['percent_missing'] >30]

missing_value_df
#Plot percentage column wise missing values.

fig, ax = plt.subplots(figsize=(8,5))    

x = missing_value_df.column_name

y = missing_value_df.percent_missing 



sns.set()

sns.barplot(x,y)

ax = plt.gca()

for p in ax.patches:

    height = p.get_height()

    ax.text(p.get_x() + p.get_width()/2.,

            height + 2,

            int(height),

            fontsize=12, ha='center', va='bottom')

sns.set(font_scale=1.3)

ax.set_xlabel("Column Name")

ax.set_ylabel("Frequency of missing values [in %ge] ")

plt.title("Distribution of column wise missing values")

plt.xticks(rotation=90)

plt.show()
#Drop missing values more than 70% from column

df = df.drop(df.loc[:,list(round(100*(df.isnull().sum()/len(df.index)), 2)>70)].columns, 1)
#Re-Computing percentage missing values 

missing_counts= round((df.isnull().sum() * 100/ len(df)),2).sort_values(ascending=False)

missing_counts[missing_counts > 0]
#compute column missing values percentage 

print('Percent of missing "Fireplace quality is %.2f%%' %((df['FireplaceQu'].isnull().sum()/df.shape[0])*100))
#load the describe value

column_univariate(df,'FireplaceQu',1)
#plot countplot for column

column_univariate(df,'FireplaceQu',1)
#check unque value for column

df['FireplaceQu'].unique()
#Replace Lead quality missing values with "No Fireplace"

df['FireplaceQu'] = df['FireplaceQu'].replace(np.nan,'No Fireplace')

#Also,we will impute all other values 

#Imputing level values of FireplaceQu

df['FireplaceQu'].replace({'Fa':'Has Fireplace'},inplace=True)

df['FireplaceQu'].replace({'TA':'Has Fireplace'},inplace=True)

df['FireplaceQu'].replace({'Gd':'Has Fireplace'},inplace=True)

df['FireplaceQu'].replace({'Ex':'Has Fireplace'},inplace=True)

df['FireplaceQu'].replace({'Po':'Has Fireplace'},inplace=True)
#Plot Countplot for column missing values

column_univariate(df,'FireplaceQu',1)
#compute column missing values percentage 

print('Percent of missing "LotFrontage is %.2f%%' %((df['LotFrontage'].isnull().sum()/df.shape[0])*100))
#load the describe value

column_describe(df,'LotFrontage')
#plot countplot for column

column_univariate(df,'LotFrontage',1)
#We will replace NA values with 0 since there is no linear feet of street connected to property.

df['LotFrontage'].replace({np.nan:'0'},inplace=True)

df['LotFrontage']=df['LotFrontage'].values.astype(np.int64)
# As it is same as Year Built

df=df.drop(['GarageYrBlt'],axis=1) 
#compute column missing values percentage 

print('Percent of missing "GarageType is %.2f%%' %((df['GarageType'].isnull().sum()/df.shape[0])*100))
#load the describe value

column_describe(df,'GarageType')
#plot countplot for column

column_univariate(df,'GarageType',1)
#Replace Lead quality missing values with "No Fireplace"

df['GarageType'] = df['GarageType'].replace(np.nan,'Others')
#Replacing CarPort, No Garage, Basement,and 2Types  GarageType values to Others

df['GarageType'].replace({'CarPort':'No Garage'},inplace=True)

df['GarageType'].replace({'Basment':'No Garage'},inplace=True)

df['GarageType'].replace({'No Garage':'No Garage'},inplace=True)

df['GarageType'].replace({'2Types':'No Garage'},inplace=True)
#plot countplot for column

column_univariate(df,'GarageType',1)
#compute column missing values percentage 

print('Percent of missing "GarageFinish is %.2f%%' %((df['GarageFinish'].isnull().sum()/df.shape[0])*100))
#load the describe value

column_describe(df,'GarageFinish')
#plot countplot for column

column_univariate(df,'GarageFinish',1)
#Wll replace missing values with "No Garage" instead of blank/Na.

df['GarageFinish'].replace({np.nan:'No Garage'},inplace=True)
#plot countplot for column

column_univariate(df,'GarageFinish',1)
#compute column missing values percentage 

print('Percent of missing "GarageQual is %.2f%%' %((df['GarageQual'].isnull().sum()/df.shape[0])*100))
#load the describe value

column_describe(df,'GarageQual')
#plot countplot for column

column_univariate(df,'GarageQual',1)
#will replace Na values with No Garage

df['GarageQual'].replace({np.nan:'No Garage'},inplace=True)
#plot countplot for column

column_univariate(df,'GarageQual',1)
#compute column missing values percentage 

print('Percent of missing "GarageCond is %.2f%%' %((df['GarageCond'].isnull().sum()/df.shape[0])*100))
#load the describe value

column_describe(df,'GarageCond')
#plot countplot for column

column_univariate(df,'GarageCond',1)
#will replace missing values with No Garage  

df['GarageCond'].replace({np.nan:'No Garage'},inplace=True)
#Replacing eachlevel values of GarageCond with Okay/No Garage.

df['GarageCond'].replace({'TA':'OK'},inplace=True)

df['GarageCond'].replace({'Fa':'OK'},inplace=True)

df['GarageCond'].replace({'Gd':'OK'},inplace=True)

df['GarageCond'].replace({'Ex':'OK'},inplace=True)

df['GarageCond'].replace({'Po':'No Garage'},inplace=True)
#plot countplot for column

column_univariate(df,'GarageCond',1)
plt.figure(figsize=(20, 8))

plt.subplot(1,4,1)

sns.boxplot(x = 'GarageCond', y = 'SalePrice', data = df)

plt.subplot(1,4,2)

sns.boxplot(x = 'GarageQual', y = 'SalePrice', data = df)

plt.subplot(1,4,3)

sns.boxplot(x = 'GarageFinish', y = 'SalePrice', data = df)

plt.subplot(1,4,4)

sns.boxplot(x = 'GarageType', y = 'SalePrice', data = df)
df=df.drop(['GarageQual'],axis=1)
#compute column missing values percentage 

print('Percent of missing "BsmtExposure is %.2f%%' %((df['BsmtExposure'].isnull().sum()/df.shape[0])*100))
#load the describe value

column_describe(df,'BsmtExposure')
#plot countplot for column

column_univariate(df,'BsmtExposure',1)
#Will replace NaN values with "No Basement"

df['BsmtExposure'].replace({np.nan:'No Basement'},inplace=True)
#plot countplot for column

column_univariate(df,'BsmtExposure',1)
#compute column missing values percentage 

print('Percent of missing "BsmtFinType1 is %.2f%%' %((df['BsmtFinType1'].isnull().sum()/df.shape[0])*100))
#load the describe value

column_describe(df,'BsmtFinType1')
#plot countplot for column

column_univariate(df,'BsmtFinType1',1)
#Replacing missing values with No Basement

df['BsmtFinType1'].replace({np.nan:'No Basement'},inplace=True)
#plot countplot for column

column_univariate(df,'BsmtFinType1',1)
#compute column missing values percentage 

print('Percent of missing "BsmtFinType2 is %.2f%%' %((df['BsmtFinType2'].isnull().sum()/df.shape[0])*100))
#load the describe value

column_describe(df,'BsmtFinType2')
#plot countplot for column

column_univariate(df,'BsmtFinType2',1)
#will replace BsmtFinType2 missing values with No Basement.

df['BsmtFinType2'].replace({np.nan:'No Basement'},inplace=True)
#plot countplot for column

column_univariate(df,'BsmtFinType2',1)
#compute column missing values percentage 

print('Percent of missing "BsmtCond is %.2f%%' %((df['BsmtCond'].isnull().sum()/df.shape[0])*100))
#load the describe value

column_describe(df,'BsmtCond')
#plot countplot for column

column_univariate(df,'BsmtCond',1)
#Will replace missing values with No Basement

df['BsmtCond'].replace({np.nan:'No Basement'},inplace=True)
# We will combined levels of fair/good quality to OK instead of multiple.

df['BsmtCond'].replace({'Fa':'OK'},inplace=True)

df['BsmtCond'].replace({'TA':'OK'},inplace=True)

df['BsmtCond'].replace({'Gd':'OK'},inplace=True)

# Poor quality Level can be combined as Not OK (Po, No Basement)

df['BsmtCond'].replace({'Po':'Not OK'},inplace=True)

df['BsmtCond'].replace({'No Basement':'Not OK'},inplace=True)
#plot countplot for column

column_univariate(df,'BsmtCond',1)
#compute column missing values percentage 

print('Percent of missing "BsmtQual is %.2f%%' %((df['BsmtQual'].isnull().sum()/df.shape[0])*100))
#load the describe value

column_describe(df,'BsmtQual')
#plot countplot for column

column_univariate(df,'BsmtQual',1)
#Will replace missing values with No basement 

df['BsmtQual'].replace({np.nan:'No Basement'},inplace=True)
# Combined levels:FA/Ex/No to others.

df['BsmtQual'].replace({'Fa':'Others'},inplace=True)

df['BsmtQual'].replace({'Ex':'Others'},inplace=True)

df['BsmtQual'].replace({'No Basement':'Others'},inplace=True)
#plot countplot for column

column_univariate(df,'BsmtQual',1)
cols=['BsmtQual','BsmtExposure','BsmtFinType1','BsmtFinType2','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF']

plt.figure(figsize=(20, 10))

sns.pairplot(df[cols])

plt.show()
df=df.drop(['BsmtFinSF1','BsmtFinSF2','BsmtUnfSF'],axis=1)
plt.figure(figsize=(12, 8))

plt.subplot(2,2,1)

sns.boxplot(x = 'BsmtQual', y = 'SalePrice', data = df)

plt.subplot(2,2,2)

sns.boxplot(x = 'BsmtExposure', y = 'SalePrice', data = df)

plt.subplot(2,2,3)

sns.boxplot(x = 'BsmtFinType1',y = 'SalePrice', data = df)

plt.subplot(2,2,4)

sns.boxplot(x = 'BsmtFinType2',y = 'SalePrice', data = df)
#Dropping variables BsmtFinType1 and BsmtFinType2 as two do not seem to have a strong influence on sale price

df=df.drop(['BsmtFinType1','BsmtFinType2'],axis=1)
result=df[['OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch']]

plt.figure(figsize=(20, 10))

sns.pairplot(result)

plt.show()
df=df.drop(['ScreenPorch','3SsnPorch'],axis=1)
#compute column missing values percentage 

print('Percent of missing "Electrical is %.2f%%' %((df['Electrical'].isnull().sum()/df.shape[0])*100))
#load the describe value

column_describe(df,'Electrical')
#plot countplot for column

column_univariate(df,'Electrical',1)
#Will replace missing values with UnKnown

df['Electrical'].replace({np.nan:'Unknown'},inplace=True)

#Imputing the level : FuseA,FuseF,FuseP and Mix to Other

df['Electrical'].replace({'FuseA':'Other'},inplace=True)

df['Electrical'].replace({'FuseF':'Other'},inplace=True)

df['Electrical'].replace({'FuseP':'Other'},inplace=True)

df['Electrical'].replace({'Mix':'Other'},inplace=True)

df['Electrical'].replace({'Unknown':'Other'},inplace=True)
#plot countplot for column

column_univariate(df,'Electrical',1)
#compute column missing values percentage 

print('Percent of missing "MasVnrType is %.2f%%' %((df['MasVnrType'].isnull().sum()/df.shape[0])*100))
#load the describe value

column_describe(df,'MasVnrType')
#plot countplot for column

column_univariate(df,'MasVnrType',1)
#Will drop null Rows in MasVnrType 

df=df.dropna(how='any',axis=0)
# checking whether some rows have more than 1 missing values

len(df[df.isnull().sum(axis=1) > 1].index)
#Converting the binned year columns as object datatype

df['YearBuilt']=df['YearBuilt'].values.astype(np.object)

df['YearRemodAdd']=df['YearRemodAdd'].values.astype(np.object)
#Dropping column MasVnrarea and LotFrontage as these are not adding value

df=df.drop(['MasVnrArea','LotFrontage'],axis=1)
# Some of the non-numeric predictors are stored as numbers; convert them into strings 

df['MSSubClass'] = df['MSSubClass'].apply(str)

df['YrSold'] = df['YrSold'].astype(str)

df['MoSold'] = df['MoSold'].astype(str)
round(100*(df.isnull().sum()/len(df.index)), 2)
#Numerica column list

numerical_cols=df.iloc[:, (np.where((df.dtypes == np.int64) | (df.dtypes == np.float64)))[0]].columns.tolist()

numerical_cols
#Plot outliers using box plot

cols = [  'LotArea',      'OverallQual',   'OverallCond', 'TotalBsmtSF',

         '1stFlrSF',      '2ndFlrSF',      'LowQualFinSF', 'GrLivArea',

         'BsmtFullBath',  'BsmtHalfBath',  'FullBath',     'HalfBath',

         'BedroomAbvGr',  'KitchenAbvGr',  'TotRmsAbvGrd', 'Fireplaces',

         'GarageCars',    'GarageArea',    'WoodDeckSF',   'OpenPorchSF',

         'EnclosedPorch', 'PoolArea',      'MiscVal',      'SalePrice'

]

box_plot(df,cols)
#Remove the upper outlier(s) with the 95th percentile and the lower one(s) with the 5th percentile.

for col in numerical_cols:

    percentiles = df[col].quantile([0.05,0.95]).values

    df[col][df[col] <= percentiles[0]] = percentiles[0]

    df[col][df[col] >= percentiles[1]] = percentiles[1]
#Plot outliers using box plot

box_plot(df,cols)
column_univariate(df,'SalePrice',0)

df.SalePrice.describe() 
#skewness and kurtosis

print("Skewness: %f" % df['SalePrice'].skew())

print("Kurtosis: %f" % df['SalePrice'].kurt())
summary=df.SalePrice.describe(percentiles = [0.25,0.50,0.75,0.85,0.90,1])

plt.rcParams['figure.figsize'] = (14,6)

prices = pd.DataFrame({"SalePrice":df["SalePrice"], "log(SalePrice + 1)":np.log1p(df["SalePrice"])})

prices.hist(bins = 20)
column_univariate(df,'LotArea',0)

column_describe(df,'LotArea')
column_univariate(df,'GrLivArea',0)

column_describe(df,'GrLivArea')
column_univariate(df,'TotalBsmtSF',0)

column_describe(df,'TotalBsmtSF')
plt.figure(figsize=(14, 7))

plt.subplot(3,2,1)

sns.distplot(df['1stFlrSF']) 

plt.subplot(3,2,2)

sns.boxplot(df['1stFlrSF']) 

plt.subplot(3,2,3)

sns.distplot(df['2ndFlrSF']) 

plt.subplot(3,2,4)

sns.boxplot(df['2ndFlrSF']) 

plt.figure(figsize=(14, 7))

plt.subplot(1,2,1)

fig = sns.boxplot(x=df['OverallQual'], y=df["SalePrice"] )

fig.axis(ymin=0, ymax=400000);



plt.subplot(1,2,2)

fig = sns.scatterplot(x=df['OverallQual'], y=df["SalePrice"] )

fig.axis(ymin=0, ymax=400000);

plt.show()

 
plt.figure(figsize=(14, 7))

plt.subplot(1,2,1)

sns.scatterplot(x=df['1stFlrSF'], y=df['SalePrice']);

plt.subplot(1,2,2)

sns.scatterplot(x=df['2ndFlrSF'], y=df['SalePrice']);

f, ax = plt.subplots(figsize=(26, 6))

fig = sns.boxplot(x=df['YearBuilt'], y=df["SalePrice"] )

fig.axis(ymin=0, ymax=400000);

plt.xticks(rotation=90);
plt.figure(figsize=(16, 7))

plt.subplot(1,3,1)

sns.scatterplot(x=df['TotalBsmtSF'], y=df['SalePrice']);

plt.subplot(1,3,2)

sns.scatterplot(x=df['GrLivArea'], y=df['SalePrice']);

plt.subplot(1,3,3)

sns.scatterplot(x=df['LotArea'], y=df['SalePrice']);
plt.figure(figsize=(16, 7))

plt.subplot(1,2,1)

sns.scatterplot(x=df['GarageArea'], y=df['SalePrice']);

plt.subplot(1,2,2)

sns.scatterplot(x=df['GarageCars'], y=df['SalePrice']);
plt.figure(figsize=(16, 5))

plt.subplot(1,3,1)

sns.scatterplot(x=df['WoodDeckSF'], y=df['SalePrice']);

plt.subplot(1,3,2)

sns.scatterplot(x=df['OpenPorchSF'], y=df['SalePrice']);

plt.subplot(1,3,3)

sns.scatterplot(x=df['EnclosedPorch'], y=df['SalePrice']);
cols =[  'OverallCond',  'LowQualFinSF',  'BsmtFullBath',  'BsmtHalfBath',

         'FullBath',     'HalfBath',      'BedroomAbvGr',  'KitchenAbvGr', 

         'TotRmsAbvGrd', 'Fireplaces',    'PoolArea',      'MiscVal' ]

fig, ax = plt.subplots(2, 6, figsize=(35, 12))

for variable, subplot in zip(cols, ax.flatten()):

    sorted_value = df.groupby([variable])['SalePrice'].median().sort_values()

    sns.boxplot(df[variable],df["SalePrice"], order=list(sorted_value.index), ax=subplot)
cols= ['MSSubClass',    'MSZoning',  'Street',  'LotShape',

        'LandContour',  'Utilities',  'LotConfig',   'LandSlope',

          'BldgType','HouseStyle' ,'Condition1','Condition2'

          ]

column_univariate(df,cols,2,2,6)  
cols=['Neighborhood','YearBuilt', 'YearRemodAdd']

 

fig, ax = plt.subplots(3, 1, figsize=(26, 12))

for variable, subplot in zip(cols, ax.flatten()):

    total = float(len(df[variable]))

    ax=sns.countplot(data = df, x= variable,ax=subplot,order=df[variable].value_counts().index) 

    ax.set_xticklabels(ax.get_xticklabels(), rotation=50, ha="right")

     

                          

    for p in ax.patches:    

        height = p.get_height()

        ax.text(p.get_x()+p.get_width()/2., height + 3, '{:1.2f}%'.format((height/total)*100),ha="center") 

        

plt.tight_layout()

plt.show()

                         
cols=[ 'RoofStyle',  'RoofMatl', 'Exterior1st',  'Exterior2nd',

 'MasVnrType',  'ExterQual',  'ExterCond',  'Foundation',

 'BsmtQual',  'BsmtCond',  'BsmtExposure',  'Heating',

 'HeatingQC',  'CentralAir'

]



fig, ax = plt.subplots(2, 7, figsize=(26, 10))

for variable, subplot in zip(cols, ax.flatten()):

    total = float(len(df[variable]))

    ax=sns.countplot(data = df, x= variable,ax=subplot,order=df[variable].value_counts().index) 

    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right")

     

                          

    for p in ax.patches:    

        height = p.get_height()

        ax.text(p.get_x()+p.get_width()/2., height + 3, '{:1.2f}%'.format((height/total)*100),ha="center") 

        

plt.tight_layout()

plt.show()

cols=[ 'KitchenQual',  'Functional',  'FireplaceQu',  'GarageType',

       'GarageFinish', 'GarageCond',  'PavedDrive',  'MoSold', 

       'YrSold' , 'SaleType',     'SaleCondition','Electrical']



fig, ax = plt.subplots(2, 6, figsize=(20, 12))

for variable, subplot in zip(cols, ax.flatten()):

    total = float(len(df[variable]))

    ax=sns.countplot(data = df, x= variable,ax=subplot,order=df[variable].value_counts().index) 

    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right")

     

                          

    for p in ax.patches:    

        height = p.get_height()

        ax.text(p.get_x()+p.get_width()/2., height + 3, '{:1.2f}%'.format((height/total)*100),ha="center") 

        

plt.tight_layout()

plt.show()
variables =['MSSubClass',  'MSZoning',  'Street',   'LotShape',

             'LandContour','Utilities', 'LotConfig', 'LandSlope',

             'Condition1', 'Condition2', 'BldgType','HouseStyle']



        

fig, ax = plt.subplots(2, 6, figsize=(40, 15))

for variable, subplot in zip(variables, ax.flatten()):

    sorted_value = df.groupby([variable])['SalePrice'].median().sort_values()

    sns.boxplot(df[variable],df["SalePrice"], order=list(sorted_value.index), ax=subplot)

    for label in subplot.get_xticklabels():

        label.set_rotation(0)
variables=[ 'YearBuilt',  'YearRemodAdd',  'Neighborhood' ]



fig, ax = plt.subplots(3, 1, figsize=(30, 15))

for variable, subplot in zip(variables, ax.flatten()):    

    sorted_value = df.groupby([variable])['SalePrice'].median().sort_values()

    sns.boxplot(df[variable],df["SalePrice"], order=list(sorted_value.index), ax=subplot)

    for label in subplot.get_xticklabels():        label.set_rotation(90)
variables=[  'RoofStyle',   'RoofMatl',  'Exterior1st',    'Exterior2nd',

             'MasVnrType',  'ExterQual', 'ExterCond',       'Foundation',

             'BsmtQual',    'BsmtCond',   'BsmtExposure',  'Heating'

            ]

fig, ax = plt.subplots(2, 6, figsize=(36, 14))

for variable, subplot in zip(variables, ax.flatten()):

    sorted_value = df.groupby([variable])['SalePrice'].median().sort_values()

    sns.boxplot(df[variable],df["SalePrice"], order=list(sorted_value.index), ax=subplot)

    for label in subplot.get_xticklabels():

        label.set_rotation(90)
variables=[ 'HeatingQC',  'CentralAir',  'Electrical',  'KitchenQual', 

            'Functional', 'FireplaceQu', 'GarageType',  'GarageFinish',

            'GarageCond', 'PavedDrive',  'MoSold',      'YrSold',

             'SaleType',  'SaleCondition']

fig, ax = plt.subplots(2, 6, figsize=(35, 10))

for variable, subplot in zip(variables, ax.flatten()):

    sorted_value = df.groupby([variable])['SalePrice'].median().sort_values()

    fig=sns.boxplot(df[variable],df["SalePrice"], order=list(sorted_value.index), ax=subplot)

    fig.axis(ymin=0, ymax=400000);

    

    for label in subplot.get_xticklabels():

        label.set_rotation(0)
#Applying equation to feature to build the new feature.

df['YrSold'] = df['YrSold'].astype(int)

df['OverallGrade'] = (df['OverallCond'] * df['OverallQual']) / 100.0

df['Home_age_when_sold'] = df['YrSold'] - df['YearBuilt']

df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF'] 

df['Total_Bath_rooms'] = (df['FullBath'] + (0.5 * df['HalfBath']) +df['BsmtFullBath'] + (0.5 * df['BsmtHalfBath']))

df['GarageArea'] = df['GarageArea'].apply(lambda x: np.exp(6) if x <= 0.0 else x)

df['GarageCars'] = df['GarageCars'].apply(lambda x: 0 if x <= 0.0 else x)

df['HasPool'] = df['PoolArea'].apply(lambda x: 1 if x > 0 else 0)

df['YearsSinceRemodel'] = df['YrSold'].astype(int) - df['YearRemodAdd'].astype(int)
plt.figure(figsize=(30, 6))

plt.subplot(1,8,1)

sns.scatterplot(x=df['OverallGrade'], y=df['SalePrice']);

plt.subplot(1,8,2)

sns.scatterplot(x=df['Home_age_when_sold'], y=df['SalePrice']);

plt.subplot(1,8,3)

sns.scatterplot(x=df['TotalSF'], y=df['SalePrice']);

plt.subplot(1,8,4)

sns.scatterplot(x=df['Total_Bath_rooms'], y=df['SalePrice']);

plt.subplot(1,8,5)

sns.scatterplot(x=df['YearsSinceRemodel'], y=df['SalePrice']);

plt.subplot(1,8,6)

sns.scatterplot(x=df['GarageArea'], y=df['SalePrice']);

plt.subplot(1,8,7)

sns.scatterplot(x=df['GarageCars'], y=df['SalePrice']);

plt.subplot(1,8,8)

sns.scatterplot(x=df['HasPool'], y=df['SalePrice']);

categorical = [  'MSSubClass',  'MSZoning',  'Street',      'LotShape',    'LandContour','Utilities',

                 'LotConfig',   'LandSlope', 'Neighborhood','Condition1',  'Condition2',  'BldgType',   'HouseStyle',  'YearBuilt', 

                 'YearRemodAdd','RoofStyle', 'RoofMatl',    'Exterior1st', 'Exterior2nd','MasVnrType',  'ExterQual',  'ExterCond',  

                 'Foundation', 'BsmtQual',   'BsmtCond',    'BsmtExposure','Heating','HeatingQC',   'CentralAir','Electrical',  

                 'KitchenQual','Functional', 'FireplaceQu',  'GarageType',  'GarageFinish','GarageCond',  'PavedDrive','MoSold',  

                 'YrSold', 'SaleType','SaleCondition' 

                 ]



for col in categorical:

    df=get_column_dummies(col,df)

df.head()
#We will transform skewed numeric features using numply log method.

skewness = df[numerical_cols].apply(lambda x: skew(x))

left_skewed_cols = skewness[skewness > 0.5].index

right_skewed_cols = skewness[skewness < -0.5].index

df[left_skewed_cols] = np.log1p(df[left_skewed_cols])

df[right_skewed_cols] = np.exp(df[right_skewed_cols])

 
 # Putting feature variable to X and outcmoe variable to y

X = df.drop(['Id','SalePrice'], axis=1)

y = df['SalePrice']
#Check first few records

X.head()
#Split the dataset 

np.random.seed(0) 

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
#Create StandardScaler instance

scaler = StandardScaler()
#Fit and transform scaler

numerical = X_train.select_dtypes(include=[np.number]).columns.tolist()

X_train[numerical] = scaler.fit_transform(X_train[numerical])

X_train.head()
#Fit and transform scaler

numerical = X_test.select_dtypes(include=[np.number]).columns.tolist()

X_test[numerical] = scaler.transform(X_test[numerical])

X_test.head()
# Create correlation matrix

corr_matrix = X_train.corr().abs()



# Select upper triangle of correlation matrix

upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))



# Find index of feature columns with correlation greater than 0.7 and less than -0.7

to_drop = [column for column in upper.columns if any(upper[column] > 0.7) | any(upper[column] < -0.7)]

to_drop
plt.figure(figsize = (18,12))  

result_corr=X_train.corr(method='pearson')

sns.heatmap(result_corr[(result_corr >= 0.7) | (result_corr <= -0.7)], 

            cmap='YlGnBu', vmax=1.0, vmin=-1.0, linewidths=0.1,

            annot=True, annot_kws={"size": 10}, square=True);
# drop categorical variables 

X_train = X_train.drop(to_drop, axis=1)

X_test = X_test.drop(to_drop, axis=1)
# will use for metric calculation.

n=X_train.shape[0]

k=X_train.shape[1]
#Train/Test data overfitting checking

def overfit_check(x):

    overfit = []

    for i in x.columns:

        counts = x[i].value_counts()

        zeros = counts.iloc[0]

    if zeros / len(X) * 100 >99.94:

        overfit.append(i)



    overfit = list(overfit)

    x.drop(overfit,axis=1,inplace=True)



overfit_check(X_train)

overfit_check(X_test)
#instantiate the model

lr = LinearRegression(fit_intercept=True)
# fit the linear regression on the features and target variable.

model = lr.fit(X_train, y_train)
# predict prices of X_train

lr_y_train_pred = lr.predict(X_train)

# predict prices of X_test

lr_y_test_pred = lr.predict(X_test)
performance_metric(n,k,lr_y_train_pred,y_train)
performance_metric(n,k,lr_y_test_pred,y_test)
actual_values = y_test

predictions=lr_y_test_pred

# predicted versus actual on log and non-log scale

figure, axes = plt.subplots(nrows=1, ncols=2,figsize=(16, 6))

# both test data and predictions on log scale

ax = sns.regplot(x = y_test, y = predictions, ax=axes[0], color = 'blue')



ax.set(xlabel='Actual Log (SalePrice): Test Set', ylabel = 'Predicted Log (SalePrice): Test Set')



# both test data and predictions on actual (anti-logged) scale

ax = sns.regplot(x = np.log(y_test), y = np.log(predictions), ax=axes[1], color = 'grey')



ax = ax.set(xlabel='Actual SalePrice: Test Set', ylabel = 'Predicted SalePrice: Test Set')

import warnings

warnings.filterwarnings("ignore")



# list of alphas to tune

params = {'alpha': [ 0.00001,0.0001,0.0005, 0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5,

                     0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,

                     8.0, 9.0, 10.0, 20, 50, 100, 500, 1000 ]}

                     



# cross-validation

KFolds = 10



# instantiate the model

lasso = Lasso()

 

# cross-validation With Parameter Tuning Using Grid Search

lasso_model_cv = GridSearchCV(estimator = lasso, 

                        param_grid = params, 

                        scoring= 'neg_mean_absolute_error', 

                        cv = KFolds, 

                        return_train_score=True,

                        verbose = 1) 

# fit the cross-validation model

lasso_model_cv.fit(X_train, y_train) 
cv_results = pd.DataFrame(lasso_model_cv.cv_results_)

cv_results.head()
# plotting mean test and train scores with alpha 

plt.figure(figsize = (8,6)) 

cv_results['param_alpha'] = cv_results['param_alpha'].astype('float32')



# plotting

plt.plot(cv_results['param_alpha'], cv_results['mean_train_score'])

plt.plot(cv_results['param_alpha'], cv_results['mean_test_score'])

plt.xlabel('alpha')

plt.ylabel('Negative Mean Absolute Error')



plt.title("Negative Mean Absolute Error and alpha")

plt.legend(['train score', 'test score'], loc='upper right')

plt.show()
lasso_model_cv.best_params_   
alpha = lasso_model_cv.best_params_['alpha']

lasso = Lasso(alpha=alpha)

lasso.fit(X_train, y_train) 
#Extracting the coefficients and model equation from lasso regression

#lasso.coef_

  

lasso_coef = pd.DataFrame(np.round_(lasso.coef_, decimals=3), 

X_train.columns, columns = ["penalized_regression_coefficients"])

# remove the non-zero coefficients

lasso_coef = lasso_coef[lasso_coef['penalized_regression_coefficients'] != 0]

# sort the values from high to low

lasso_coef = lasso_coef.sort_values(by = 'penalized_regression_coefficients', 

ascending = False)

# plot the sorted dataframe

plt.figure(figsize = (16,30))

ax = sns.barplot(x = 'penalized_regression_coefficients', y= lasso_coef.index , 

data=lasso_coef)

ax.set(xlabel='Penalized Regression Coefficients')

plt.title("Coefficients in the Lasso Model")

coef = pd.Series(lasso.coef_, index = X_train.columns)

print("Lasso picked " + str(sum(coef != 0)) + " features  and eliminated the other " +  str(sum(coef == 0)) + " variables")

# plotting feature importances!

imp_coef = pd.concat([coef.sort_values().head(10),coef.sort_values().tail(10)])

plt.rcParams['figure.figsize'] = (8.0, 10.0)

imp_coef.plot(kind = "barh")

plt.title("Most Important Coefficients in the Lasso Model")

imp_coef
# lasso regression model with optimal alpha

# predict train/test

y_train_pred = lasso.predict(X_train) 

y_test_pred  = lasso.predict(X_test)
print("lasso Train Score:", 100*round(lasso.score(X_train, y_train),4))

print("lasso Test  Score:", 100*round(lasso.score(X_test, y_test),4))
performance_metric(n,k,y_train_pred,y_train)
performance_metric(n,k,y_test_pred,y_test)
# predicted versus actual on log and non-log scale



figure, axes = plt.subplots(nrows=1, ncols=2,figsize=(16, 6))



# both test data and predictions on log scale

ax = sns.regplot(x = np.exp(y_test), y = np.exp(y_test_pred), ax=axes[0], color = 'olivedrab')



ax.set(xlabel='Actual SalePrice: Test Set', ylabel = 'Predicted SalePrice: Test Set')



# both test data and predictions on actual (anti-logged) scale

ax = sns.regplot(x = y_test , y = y_test_pred, ax=axes[1], color = 'brown')

ax = ax.set(xlabel='Actual Log(SalePrice): Test Set',ylabel = 'Predicted Log(SalePrice): Test Set') 
final_predictions = np.exp(y_test_pred)

orignal_predictions = np.exp(y_test)



list_of_tuples = list(zip(orignal_predictions[:10], final_predictions[:10]))

Lasso_predictions = pd.DataFrame(list_of_tuples,columns = ['Original predictions Value', 'Final predictions Value']) 

Lasso_predictions.head(10)
import warnings

warnings.filterwarnings("ignore")



# list of alphas to tune

params = {'alpha': [ 0.0001,0.0005, 0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5,

                     0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 

                     4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 20, 50, 100, 500, 1000 ]}



#instantiate the model

ridge = Ridge()



# Cross validation and GridSearch

KFolds = 10

ridge_model_cv = GridSearchCV(estimator = ridge, 

                        param_grid = params,

                        scoring= 'neg_mean_absolute_error', 

                        cv = KFolds, 

                        return_train_score=True,

                        verbose = 1) 

#fit the cross-validation model

ridge_model_cv.fit(X_train, y_train) 
#store cross validation results to dataframe

cv_results = pd.DataFrame(ridge_model_cv.cv_results_)



cv_results.head()
# plotting mean test and train scoes with alpha 

cv_results['param_alpha'] = cv_results['param_alpha'].astype('int32')



# plotting alpha and mean train score

figure, axes = plt.subplots(nrows=1, ncols=1,figsize=(8, 6))

plt.plot(cv_results['param_alpha'], cv_results['mean_train_score'])

plt.plot(cv_results['param_alpha'], cv_results['mean_test_score'])

plt.xlabel('alpha')

plt.ylabel('Negative Mean Absolute Error')

plt.title("Negative Mean Absolute Error and alpha")

plt.legend(['train score', 'test score'], loc='upper right')

plt.show()
#Report best alpha value based on train/test score.

ridge_model_cv.best_params_
alpha = ridge_model_cv.best_params_['alpha']



ridge = Ridge(alpha=alpha)

ridge.fit(X_train, y_train)

ridge_coef = pd.DataFrame(np.round_(ridge.coef_, decimals=3), 

X_train.columns, columns = ["penalized_regression_coefficients"])

# remove the non-zero coefficients

ridge_coef = ridge_coef[ridge_coef['penalized_regression_coefficients'] != 0]

# sort the values from high to low

ridge_coef = ridge_coef.sort_values(by = 'penalized_regression_coefficients', 

ascending = False)

# plot the sorted dataframe

plt.figure(figsize = (18,50))

ax = sns.barplot(x = 'penalized_regression_coefficients', y= ridge_coef.index , 

data=ridge_coef)

ax.set(xlabel='Penalized Regression Coefficients')

plt.title("Coefficients in the Ridge Model")
coef = pd.Series(ridge.coef_, index = X_train.columns)

print("Ridge picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")

# plotting feature importances!

imp_coef = pd.concat([coef.sort_values().head(10),

                    coef.sort_values().tail(10)])

plt.rcParams['figure.figsize'] = (8.0, 10.0)

imp_coef.plot(kind = "barh")

plt.title("Most Important Coefficients in the Ridge Model")

imp_coef

# Ridge regression with with optimal alpha

rrm = Ridge(alpha=alpha)

rrm.fit(X_train, y_train)

 

# predict for train/test data

y_train_pred = rrm.predict(X_train)

y_test_pred = rrm.predict(X_test)
print("Ridge Train Score:", 100*round(ridge.score(X_train, y_train),4))

print("Ridge Test  Score:", 100*round(ridge.score(X_test, y_test),4))
performance_metric(n,k,y_train_pred,y_train)
performance_metric(n,k,y_test_pred,y_test)
# predicted versus actual on log and non-log scale

 

figure, axes = plt.subplots(nrows=1, ncols=2,figsize=(16, 6))



# both test data and predictions on log scale

ax = sns.regplot(x = np.exp(y_test), y = np.exp(y_test_pred), ax=axes[0], color = 'olivedrab')



ax.set(xlabel='Actual SalePrice: Test Set', ylabel = 'Predicted SalePrice: Test Set')



# both test data and predictions on actual (anti-logged) scale

ax = sns.regplot(x = y_test , y = y_test_pred, ax=axes[1], color = 'brown')

ax = ax.set(xlabel='Actual Log(SalePrice): Test Set',ylabel = 'Predicted Log(SalePrice): Test Set') 
final_predictions = np.exp(y_test_pred)

orignal_predictions = np.exp(y_test)



list_of_tuples = list(zip(orignal_predictions[:10], final_predictions[:10]))

Ridge_predictions = pd.DataFrame(list_of_tuples,columns = ['Original predictions Value', 'Final predictions Value']) 

Ridge_predictions.head(10)