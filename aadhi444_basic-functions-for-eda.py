# import all the necessary libraries/packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
%matplotlib inline

from collections import Counter

pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
def outlier_index(df,n, features):
    """
    Takes a dataframe df of features and returns a list of the indices
    corresponding to the observations containing more than n outliers according
    to the Tukey method.
    """
    index = [] 
    
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
        index.extend(outlier_list_col)
        
    # select observations containing more than 2 outliers
    outlier_indices = Counter(index)        
    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )
    
        
    return multiple_outliers 

# guide to use the function
# index = outlier_index(df,n,['f1','f2'])
# df.iloc[index,:].head(len(index))
def missing_values(df):
    total = df.isnull().sum().sort_values(ascending = False)
    percentage = (df.isnull().sum()/df.isnull().count()*100).sort_values(ascending = False) 
    missing_data = pd.concat([total, percentage, df.dtypes], axis=1, keys=['Total', 'Percentage', 'Type'])
    print(missing_data)
def plot_correlation(df,method = 'pearson'):
    """
    df is the data frame with numerical or ordinal values.
    method is the correlation method used as described above.
    method takes values - {‘pearson’, ‘kendall’, ‘spearman’}
    default value is 'pearson'
    """
    sns.heatmap(df.corr(method = method), annot=True, fmt = ".2f",cmap = sns.color_palette("magma"),
                linewidth=2, edgecolor="k", vmax=1., square=True)
    plt.title("CORRELATION PLOT", fontsize=15)
    plt.show()  
def clip_percentiles(cols,df,lower, upper):
    """
    cols - the columns for which the outliers need to be clipped
    df - the dataframe which has the data 
    lower and upper - the percentiles for which we clip the data
    """
    df2 = df.copy()
    
    for col in cols:
        ulimit = np.percentile(df2[col].values,upper)
        llimit = np.percentile(df2[col].values,lower)
        df2.loc[df2[col]>ulimit,col] = ulimit
        df2.loc[df2[col]<llimit,col] = llimit
        
    return df2