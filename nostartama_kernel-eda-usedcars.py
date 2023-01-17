# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



#viz

import matplotlib.pyplot as plt 

import seaborn as sns 



#Statistics Module 

from scipy.stats import chi2

from scipy.stats import chi2_contingency

from scipy.stats import anderson, shapiro

import statsmodels.api as sm 

from statsmodels.formula.api import ols

from scipy import stats

import pylab

from collections import Counter



import os

dir=[]

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        dir.append(os.path.join(dirname, filename))







%matplotlib inline
#548k row x 22 column dataset

df_red=pd.read_csv(dir[0])



#1.72m row x 26 column dataset

df=pd.read_csv(dir[1])
#df = all data

#df = dataset



#dataset = data after drop

#dataset = df
#preview dataset 

def preview_df(df, options = None):

    '''

    Preview dataset is one of exploratory data analysis part, which is 

    we will know what the dataset is. 



    Paramaters :

    ------------

    df      :   object, DataFrame

            Dataset that will be used  

    option  :   Optional(default = 'top record data')

            1. top_record data  : Showing top record data(default = 10)

            2. shape_data       : showing how many rows and column of dataset

            3. info_data        : showing how many columns that includes missing value

                                  and knowing what the data type is of each column. 

    '''

    #default option is Top record 

    if options == None :

        options = 'top_record'

    

    if options == 'top_record':

        print('=>> Top 10 Record Data : ','\n')

        df = df.head(10)

        return df 



    if options == 'shape_data':

        print('=>> Data shape : ','\n')

        df = df.shape

        return df



    if options == 'info_data':

        print('=>> Data Info : ','\n')

        df = df.info()

        return df

        

    if options == 'isnull':

        print('=>> Data Info : ','\n')

        df = df.isnull().sum()

        return df

    

    if options == 'duplrows':

        duplicate_rows_df = df[df.duplicated()]

        return print('number of duplicate rows: ', duplicate_rows_df.shape)

        

def Describe(df,col=None):

    '''

    Documentation :

    --------------

    * df  : Dataframe Name

    * col : Columns Name    

    '''

    if col is None :

        describe = df.describe()

    else:

        describe = df[col].describe()

        

    return describe



    
preview_df(df)
preview_df(df, 'info_data')
Describe(df)
preview_df(df, 'duplrows')
# Make a list for both of the data type 





def check_dtypes_1(df):

    '''

    Parameters :

    ------------

    df : Dataframe name 



    Step :

    ------

    > 1. Do iteration for each feature to define which one categorical and nummerical feature. 

    > 2. Columns in dataframe will be seperated based on the dtypes

    > 3. All of the column will be entered to the list that have been created



    result :

    --------

    The result will be formed as dataframe

    '''

    

    categorical_list = []

    numerical_list = []

    

    #Looping 

    for col in df.columns.tolist():

        if df[col].dtype=='object':

            categorical_list.append(col)

        else:

            numerical_list.append(col)

    

    #make dataframe that have two feature, that is categorical and numerical feature

    categorical = pd.Series(categorical_list, name='Categorical Feature')

    numerical = pd.Series(numerical_list, name='Numerical Feature')

    df_dtypes = pd.concat([categorical,numerical], axis=1)

    

    return df_dtypes
check_dtypes_1(df)
def unique_columns(df):

    '''

    Parameter 

    ---------

    df : array

        the data frame that will be checked the unique the entities 

    '''

    for col in df.columns:

        if df[col].dtypes == 'object':

            unique_cat = len(df[col].unique())

            print("Feature '{col}' has {unique_cat} unique categories".format(col=col, unique_cat=unique_cat))
unique_columns(df)
def missing_value(df):

    '''

    Documentation :

    --------------

    * df : Dataframe Name

    '''

    #count the number of missing value 

    total = df.isnull().sum().sort_values(ascending = False)

    percent = round(df.isnull().sum()/len(df)*100,2).sort_values(ascending = False)

    missing  = pd.concat([total, percent], axis=1, keys=['Total_Missing', 'Percent(%)'])

    

    return missing
missing_value(df)
def Drop_Missing_value(df, threshold = None):

    '''

    Parameters :

    ------------

    df              :  Object, Dataframe

                    the dataframe which want to dropped     

    threshold       :   float, default (0.75)

                    the number of threshold was determined by user 

    '''

    #default number of threshold 

    if threshold == None : 

        threshold = 0.75



    # Define variable that we need 

    threshold = threshold

    size_df   = df.shape[0]



    # Define Column list that will we removed

    dropcol   = []



    #looping to take the number of null of every feature 

    for col in df.columns :

        if (df[col].isnull().sum()/size_df >= threshold):

            dropcol.append(col)

    print('Columns that have been removed : ')

    print ('_'*29)

    

    # Make df using pd.concat

    drop_col = pd.Series(dropcol, name='Features')

    drop_col = pd.concat([drop_col], axis = 1)

    

    print(drop_col)

    

    df = df.drop(dropcol, axis =1)



    return df
dataset = Drop_Missing_value(df, threshold = 0.60)
categorical_list = []

numerical_list = []



def list_dtypes(df):

    for col in df.columns.tolist():

        if df[col].dtype=='object':

            categorical_list.append(col)

        else:

            numerical_list.append(col)

    print('Number of categorical features:', str(len(categorical_list)))

    print('Number of numerical features:', str(len(numerical_list)))



    return categorical_list, numerical_list
list_dtypes(dataset)
def fill_missing(df, feature_list = None , vartype = None ):

    '''

    Documentation :

    ---------------

    df              : object, dataframe

    feature_list    : feature list is the set of numerical or categorical features 

                      that have been seperated before

    vartype         : variable type : continuos or categorical, default (numerical)

                        (0) numerical   : variable type continuos/numerical

                        (1) categorical : variable type categorical

    Note :

    ------

    > if numerical variable will be filled by median 

    > if categorical variabe will filled by modus

    > if have been made variebles based on the dtypes list before, 

      insert it into feature list in the function.     



    Example :

    ---------

    # 1. Define feature that will be filled in 

      num_feature = numeric_list

      

    # 2. Input Dataframe

      dataframe = df

      

    # 3. Vartype

      var_type = 0

      

    # 4. Filling Value

      Fill_missing(dataframe, num_feature, var_type)

    '''

    #default vartype 

    if vartype == None :

        vartype = 'numerical'



    # filling numerical data with median 

    if vartype == 'numerical' :

        for col in feature_list:

            df[col] = df[col].fillna(df[col].median())

    

    # filling categorical data with modus  

    if vartype == 'categorical' :

        for col in feature_list:

            df[col] = df[col].fillna(df[col].mode().iloc[0])
# 1. define feature that will be filled in 

num_feature = numerical_list

cat_feature = categorical_list



# 2. Vartype

num_type = 'numerical'

cat_type = 'categorical'



# 3. Filling Value

fill_missing(df, num_feature, num_type)

fill_missing(df, cat_feature, cat_type)
#Univariat Analisys

def Univariate_plot(df, column, vartype, hue = None ):

    '''

    Documentation :

    Univariate function will plot the graphs based on the parameters.

    * df      : dataframe name

    * column  : Column name

    * vartype : variable type : continuos or categorical

                (0) Continuos/Numerical   : Distribution, Violin & Boxplot will be plotted.

                (1) Categorical           : Countplot will be plotted.

    * hue     : It's only applicable for categorical analysis.

    '''

    sns.set(style="darkgrid")

      

    if vartype == 0:

        fig, ax=plt.subplots(nrows = 3, ncols=1,figsize=(12,12))

        # Distribution Plot

        ax[0].set_title("Distribution Plot",fontsize = 10)

        sns.distplot(df[column], kde=False, fit=stats.gamma, color='darkblue', label = column, ax=ax[0])

        

        # Violinplot 

        ax[1].set_title("Violin Plot",fontsize = 10)

        sns.violinplot(data= df, x=column, color = 'limegreen', inner="quartile", orient='h', ax=ax[1])

        

        #Boxplot

        ax[2].set_title("Box Plot",fontsize = 10)

        sns.boxplot(data =df, x=col,color='cyan',orient="h",ax=ax[2])

        

        fig.tight_layout()

        

    if vartype == 1 :

        #Count plot 

        fig = plt.figure(figsize=(12,6))

        plt.title('Count Plot',fontsize = 20)

        ax=sns.countplot(data=df, x=column, palette="deep",order = df[col].value_counts(ascending = False).index[:10])

        

        plt.xticks(rotation=45)

        ax.set_xlabel(column, fontsize = 15)

        ax.tick_params(labelsize=12)
# 1. Define Dataframe

dataframe = df



# 2. Define feature

col='manufacturer'



# 3. Vartype 

var_type = 1   

 

#Visualization

Univariate_plot(df=dataframe,column = col, vartype = var_type)
df_viz = df[(df['manufacturer'] =='ford') | (df['manufacturer'] =='chevrolet') | (df['manufacturer'] =='toyota') | (df['manufacturer'] =='honda') | (df['manufacturer'] =='nissan')]
plt.figure(figsize = (15, 10))

ax = sns.countplot(x="year", hue='manufacturer',data=df_viz_year)
def bivariate_plot(df, xcol, ycol, plot_type, hue = None, title= None):

    '''

    Documentation :

    --------------

    Bivariate function will plot the graphs based on the parameters.

    * df        : dataframe name

    * xcol      : X Column name

    * ycol      : Y column name

    * plot_type : plot type : scatter plot, boxplot, and violin plot 

                (0) Scactter plot     : graph between xcol(numerical) and ycol(numerical) 

                (1) Boxplot           : graph between xcol(categorical) and ycol(numerical)

                (2) Violin plot       : graph between xcol(categorical) and ycol(numerical)

    * hue     : name of variables in ``data`` or vector data, optional Grouping variable that 

                will produce points with different colors.

    '''

    if title == None :

        title = 'Bivariate Plot'

        

    # Scatter plot 

    if plot_type == 0 :

        

        

        fig = plt.figure(figsize=(12,8))

        ax = sns.scatterplot(data=df, x=xcol, y=ycol, s=150)

        #title of graph

        ax.axes.set_title(title,fontsize = 20 )

        ax.set_xlabel(xcol, fontsize = 15)

        ax.set_ylabel(ycol, fontsize = 15)

        ax.tick_params(labelsize=12)

        

    #boxplot

    if plot_type == 1 : 

        fig = plt.figure(figsize = (12, 7))

        ax =sns.boxplot(data=df, x=xcol, y=ycol, hue = hue)

        plt.setp(ax.artists, alpha=.5, linewidth=2, edgecolor="k")

        plt.xticks(rotation=45)

        #title of graph

        ax.axes.set_title(title,fontsize = 20 )

        ax.set_xlabel(xcol, fontsize = 15)

        ax.set_ylabel(ycol, fontsize = 15)

        ax.tick_params(labelsize=12)

        

    #violinplot 

    if plot_type == 2 :

        fig =plt.figure(figsize = (12, 7))

        ax = sns.violinplot(data=df, x=xcol, y=ycol,  hue = hue)

        plt.xticks(rotation=45)

        #title of graph

        ax.axes.set_title(title,fontsize = 20 )

        ax.set_xlabel(xcol, fontsize = 15)

        ax.set_ylabel(ycol, fontsize = 15)

        ax.tick_params(labelsize=12)

    

    ax.legend()
plt.figure(figsize = (15, 10))

plt.xticks(rotation=45)

plt.yticks(rotation=45)



ax = sns.barplot(x="fuel", y="odometer",hue='manufacturer',data=df_viz)
plt.figure(figsize = (15, 10))



ax = sns.barplot(x="type", y="price", hue="manufacturer",data=df_viz)
plt.figure(figsize = (20, 10))



ax = sns.barplot(x="paint_color", y="price", hue="manufacturer",data=df_viz)

plt.xticks(rotation=45)

plt.yticks(rotation=45)
plt.figure(figsize = (15, 20))



df_viz_year = df_viz[(df_viz['year'] >2010)]



ax = sns.barplot(x="condition", y="price",hue="manufacturer", data=df_viz_year)
def multivariate(df, column, plot_type ):

    '''

    Documentation :

    --------------

    Multivvariate function will plot the graphs based on the parameters.

    * df      : dataframe name

    * column  : Column name (array)

    * plot : plot_type : hitmap and pairplot

                (0) Hitmap    : Hitmap graph will be plotted.

                (1) pairplot  : pairplot graph will be plotted.

    '''

    # hitmap plot

    if plot_type == 0 :

        corrMatt = df[column].corr()

        mask = np.array(corrMatt)

        mask[np.tril_indices_from(mask)] = False

        fig,ax= plt.subplots(figsize=(20,12))

        fig.set_size_inches(25,10)

        sns.heatmap(corrMatt, mask=mask,vmax=0.9, square=True,annot=True)

        

        

    # pairplot 

    if plot_type == 1 :

        pairplot = sns.pairplot(df[column], size=2, aspect=2,

                                plot_kws=dict(edgecolor="k", linewidth=0.5),

                                diag_kind="kde", diag_kws=dict(shade=True))

        fig = pairplot.fig 

        fig.subplots_adjust(top=0.90, wspace=0.2)

        fig.suptitle('Pairplot', fontsize=15)

        

    plt.xticks(rotation=45)

    plt.yticks(rotation=45)
# 1. Define Dataframe

dataframe = df



# 2. Define Column

colname = numerical_list



# 3. Plot_type

plot = 0



# 4. Visualization 

multivariate(dataframe, colname, plot)