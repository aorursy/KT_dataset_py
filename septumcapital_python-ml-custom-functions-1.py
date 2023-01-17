# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Use relevant imports as needed for running the functions



#Data Library Imports

import pandas as pd

import numpy as np



from collections import Counter



# Data Visualization imports

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set(style='white', context='notebook', palette='deep')
#Model Cross Validation Imports

from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
#Model Algo Imports

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.svm import SVC
# Multiple Outlier detection in a given dataset

from collections import Counter

def detect_outliers(df,n,features):

    """

    Takes a dataframe df of features and returns a list of the indices

    corresponding to the observations containing more than n outliers according

    to the Tukey (Tukey JW., 1977) method.

    """

    outlier_indices = []

    

    # iterate over features(columns)

    for col in features:

        # 1st quartile (25%)

        Q1 = np.percentile(df[col], 25)

        # 3rd quartile (75%)

        Q3 = np.percentile(df[col],75)

        # Interquartile range (IQR)

        IQR = Q3 - Q1

        

        # outlier step

        outlier_step = 1.5 * IQR

        

        # Determine a list of indices of outliers for feature col

        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step )].index

        

        # append the found outlier indices for col to the list of outlier indices 

        outlier_indices.extend(outlier_list_col)

        

    # select observations containing more than 2 outliers

    outlier_indices = Counter(outlier_indices)        

    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )

    

    return multiple_outliers   



"""

Reference : https://www.kaggle.com/yassineghouzam/titanic-top-4-with-ensemble-modeling

Sample Usage -  



Outliers_to_drop = detect_outliers(train_dataset,2,["Age","SibSp","Parch","Fare"]) # Call the multiple outlier detection function

train_dataset.loc[Outliers_to_drop] # Show the outliers rows

train_dataset = train_dataset.drop(Outliers_to_drop, axis = 0).reset_index(drop=True) # Drop outliers



Notes from author of the original function in Kaggle Kernel -

Since outliers can have a dramatic effect on the prediction (espacially for regression problems), i choosed to manage them.

I used the Tukey method (Tukey JW., 1977) to detect ouliers which defines an interquartile range comprised between the 1st and 3rd quartile of the distribution values (IQR). 

An outlier is a row that have a feature value outside the (IQR +- an outlier step).I decided to detect outliers from the numerical values features (Age, SibSp, Sarch and Fare). 

Then, i considered outliers as rows that have at least two (n) outlier numerical values.



"""
# Joining two dataframes with same number and type of columns and returning the combined dataframe with combined rows (axis=0)

def join_two_dataframes_having_same_features(df1,df2):

    joined_dataframe =  pd.concat(objs=[df1, df2], axis=0).reset_index(drop=True)

    return joined_dataframe

"""

Usage - 

Join train and test dataframe for feature engineering

combined_dataframe = join_two_dataframes_having_same_features(train_dataframe,test_dataframe)

"""
# High Level Review and Analysis of a given dataset

# Provides  decribe, info , data types , null summary, head, tail of the given dataframe.

# Usually used in the inital stage before visual and detailed exploration of dataset





def inital_review_of_dataframe(df):

    # Fill empty and NaNs values with NaN

    df = df.fillna(np.nan)

    print("Info of DataFrame : ", df.info())

    print("Is Null Summary : ", df.isnull().sum())

    print("Data Types : ", df.dtypes)

    print("Summary and Stats of Numerical Variables: ", df.describe(percentiles=[0.01,0.1,0.25,0.5,0.75,0.9,0.99]))

    print("Summary and Stats of Object type or Categorical Variables: ", df.describe(include=['O']))

    print("First few records : ", df.head())

    print("Last few records : ", df.tail())



"""

Usage - 

inital_review_of_dataframe(train_dataframe)

"""
# Visual Plot Analysis of correlations between 3 features and 1 target variable y in a dataframe df 



#import library if not done previously

import seaborn as sns

import warnings





def correl_plt_of_3features_with_1target(y,df,feature1, feature2, feature3):

    warnings.filterwarnings("ignore")

    sns.set(style='white', context='notebook', palette='deep')

    g = sns.heatmap(df[[feature1,feature2,feature3,y]].corr(), vmin = -1, annot=True, fmt = ".2f", cmap = "coolwarm")



"""

Usage - 

correl_plt_of_3features_with_1target(target_var_name, train_dataframe, feature1_name, feature2_name, feature3_name)

"""
# Visual Data Exploration Function for 3 category type variables and 1 numerical variable in a DataFrame using catplot of Seaborn



def catplots_with_3cat1num_vars(df,numvar,catvar1,catvar2,catvar3):

    

    import warnings

    warnings.filterwarnings("ignore")

    sns.set(style='ticks', context='notebook', palette='deep')

    

    fig, ax = plt.subplots(1,3,sharex = True,figsize=(25,5))

    sns.catplot(x=catvar1,data=df, kind = "count",ax=ax[0]) 

    plt.close()

    sns.catplot(x=catvar2,data=df, kind = "count",ax=ax[1])

    plt.close()

    sns.catplot(x=catvar3,data=df, kind = "count",ax=ax[2])

    plt.close()

    

    fig, ax = plt.subplots(1,2,sharex = True,figsize=(25,10))

    g1 = sns.catplot(x=catvar1, y=numvar, hue=catvar2, data=df, ax=ax[0])

    plt.close()

    g2 = sns.catplot(x=catvar1, y=numvar, hue=catvar2, data=df, kind = "swarm", ax=ax[1])

    plt.close()    

    

    fig, ax = plt.subplots(1,2,sharex = True,figsize=(25,10))

    g3 = sns.catplot(x=catvar1, y=numvar, hue=catvar2, data=df, kind = "bar", ax=ax[0])

    plt.close()    

    g4 = sns.catplot(x=catvar1, y=numvar, hue=catvar2, data=df, kind = "violin", ax=ax[1])

    plt.close()



    fig, ax = plt.subplots(1,2,sharex = True,figsize=(25,10))

    g5 = sns.catplot(x=catvar1, y=numvar, hue=catvar2, data=df, kind = "box", ax=ax[0])

    plt.close()

    g6 = sns.catplot(x=catvar1, y=numvar, hue=catvar2, data=df, kind = "boxen", ax=ax[1])

    plt.close()

    

    g7 = sns.catplot(x=catvar1, y=numvar, hue=catvar2, data=df, kind = "point")

    g8 = sns.catplot(x=catvar1, y=numvar, hue=catvar2, col=catvar3, col_wrap=4, data=df, height=5, aspect=.8)

    

    

"""

Usage - 

catplots_with_3cat1num_vars(train_dataframe,numerical_var_name,categorical_var_name1,categorical_var_name2,categorical_var_name3)

"""
# Visual Data Exploration Function for 2 variables in a DataFrame using regplot of Seaborn



import seaborn as sns

import warnings

import matplotlib.pyplot as plt

%matplotlib inline





def regplot_with_2vars(df,xvar,yvar,color_choice ='r', jitter = 0):

    sns.set(color_codes=True)

    ax = sns.regplot(x=xvar, y=yvar, data=df,color = color_choice, x_jitter = jitter, y_jitter = jitter)

    

"""

Usage - 

regplot_with_2vars(train_dataframe,xaxis_varname,yaxis_varname,'g',0.1)

"""
# Visual Data Exploration Function for a variable in a DataFrame to plot distribution using distplot of Seaborn

def plot_distribution_1var(df,varname,xsize = 25, ysize = 12, color_choice = 'r' ):

    

    df[varname] = df[varname].fillna(df[varname].median())

    x = df[varname]

   

    ax = sns.distplot(df[varname], rug=True, rug_kws={"color": color_choice},

                   kde_kws={"color": "k", "lw": 3, "label": "KDE-Kernel Density Estimate"},

                   label="Skewness : %.2f"%(x.skew()),

                   hist_kws={"histtype": "step", "linewidth": 3,

                             "alpha": 1, "color": color_choice})

    ax.figure.set_size_inches(xsize, ysize)

    ax = ax.legend(loc="best")

    

    

"""

Usage - 

plot_distribution_1var(train_datafeame,Varname,xaxis_size_opt,yaxis_size_opt,'g')

"""
#Log Transform the values of  a specific column in a dataframe and return the transformed set

def log_transform_dataframe_col(df,col_name):

    x = df[col_name].map(lambda i: np.log(i) if i > 0 else 0)

    return x



"""

Usage - 

transformed = log_transform_dataframe_col(train_dataframe,column_name)

"""
#Create a panda series with subset of a panda dataframe string column based on pattern match

def create_series_from_pattern_in_anothercol(df,col_name,start_pattern,end_pattern):

    patterns = [i.split(start_pattern)[1].split(end_pattern)[0].strip() for i in df[col_name]]

    return pd.Series(patterns)



"""

Usage - 

titles = []

column_name = "Name"

titles = create_newcol_from_pattern_in_anothercol(input_dataframe,column_name,',','.')

"""