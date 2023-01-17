# Loading Libraries

import pandas as pd # for data analysis

import numpy as np # for scientific calculation

import seaborn as sns # for statistical plotting

import datetime # for working with date fields

import matplotlib.pyplot as plt # for plotting

%matplotlib inline

import math # for mathematical calculation
#Reading Mercedes-Benz given Data Set.

import os

for dirname, _, filenames in os.walk('/kaggle/input/benz_train.csv'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

benz_eda=pd.read_csv('/kaggle/input/benz_train.csv')
# Describe method is used to view some basic statistical details like percentile, mean, std etc. of a data frame of numeric values.

# Observation:Mostly given data set contains nearly 366 boolean, 8 categorical and 2 numerical features.

# All boolean features is having 0's and 1's value 

# Numberical features is one target variable and another one is unique ID's values.

# Categorical features contains different types of distinct values and difficult to understand what those are. In problem

# statement itself company mentioned it's a confidential dataand we will not be able to provide description 

# for each of the anonymized set of variables.

# Moving ahead with EDA and visualization to understand data better.

benz_eda.describe()
#Performed Pandas profiling to understand quick overview of columns

# Commented the below code which takes good amount of time to generate .html file. 

# It gives better insights to proceed with EDA and feature engineering.



#import pandas_profiling

#report = pandas_profiling.ProfileReport(benz_eda)

#covert profile report as html file

#report.to_file("benz_eda.html")
#Checking shape of data

#Observation: It contains 4209 data points with 378 columns (376 features, 1 Unique details and 1 target variable)

benz_eda.shape
# Generated HeatMap.

# Observation: From the below heatmap it is difficult to do correlation analysis between the variables.

# Let's see next observation after EDA and feature engineering.

corr = benz_eda.corr()

ax = sns.heatmap( corr,vmin=-1, vmax=1, center=0,  cmap=sns.diverging_palette(20, 220, n=200),square=True)

ax.set_xticklabels(ax.get_xticklabels(),rotation=45,horizontalalignment='right');
# Copy categorial variables and seperate it from main dataset to understand categorical variables data.

benz_cat=benz_eda.select_dtypes(include=['object']).copy()
# Verifying top 5 sample records of categorical data.

# Observation: It seems that, it contains different types of distinct values defined on each and every categorical variables.

benz_cat.head()
#Checking shape of data

benz_cat.shape
# Checking Null Values : We can see there are No Null Values 

benz_cat.isnull().sum()
# Copy numberical variables and seperate it from main dataset to understand output variable data.

benz_num=benz_eda.select_dtypes(include=['float64']).copy()
# Checking the shape of data

benz_num.shape
# Verifying top 1 sample records of numerical data.

benz_num.head(1)
# Checking Null Values : We can see there are No Null Values 

benz_num.isnull().sum()
# Explored data by using describe() method.

# Observation: Output variable seems having outliers and will dig more by on this with data visualization.

benz_num.describe()
# Concatenated both categorical and target variable data to do more analysis 

# and will dig more by on this with data visualization.

benz_cat_num = pd.concat([benz_cat, benz_num], axis=1, sort=False)
#re-dhecking shape of data after merging the 2 dataframes.

print(benz_cat.shape)

print(benz_num.shape)

print(benz_cat_num.shape)
# Re-checking the describe method

benz_cat_num.describe().T
# Re-checking the top 1 sample records.

benz_cat_num.head(1)
# Distribution plot for output variable.

# Observation: We can see it clearly that we have outliers and datapoints has been distributed mostly between around 

# 65 to 140.

sns.distplot(benz_cat_num['y'],kde = False)
# Count plot for categorical variables.

# Observation: we can see clearly how datapoints has been distributed across the distinct values 

# for all categorical features.

# X0,X1,X2,X5,X6,X8 : Some categorical values contains more datapoints and some contains very less. 

# It's having more distinct categorical values.

# X2,X3 : Some categorical values contains more datapoints and some contains very less. 

# It's having very less distinct categorical values.

plt.rcParams['figure.figsize'] = (20.0, 10.0)

plt.rcParams['font.family'] = "serif"

fig, ax =plt.subplots(4,2)

sns.countplot(benz_cat_num['X0'], ax=ax[0,0])

sns.countplot(benz_cat_num['X1'], ax=ax[0,1])

sns.countplot(benz_cat_num['X2'], ax=ax[1,0])

sns.countplot(benz_cat_num['X3'], ax=ax[1,1])

sns.countplot(benz_cat_num['X4'], ax=ax[2,0])

sns.countplot(benz_cat_num['X5'], ax=ax[2,1])

sns.countplot(benz_cat_num['X6'], ax=ax[3,0])

sns.countplot(benz_cat_num['X8'], ax=ax[3,1])

fig.show()
# Reffered encoding techniques from the below blog and selected best suitable encoding technique for this dataset.

#https://towardsdatascience.com/all-about-categorical-variable-encoding-305f3361fd02
# Re-verifying the sample top 1 record data

benz_cat_num.head(1)
# Mean Encoding technique has been choosen to ensure better encoding amoung other technique's based on the above plots.

# Verified the top 1 sample data after mean-encoding technique.

mean_encode_X0 = benz_cat_num.groupby('X0')['y'].mean()

#print(mean_encode)

mean_encode_X1 = benz_cat_num.groupby('X1')['y'].mean()

mean_encode_X2 = benz_cat_num.groupby('X2')['y'].mean()

mean_encode_X3 = benz_cat_num.groupby('X3')['y'].mean()

mean_encode_X4 = benz_cat_num.groupby('X4')['y'].mean()

mean_encode_X5 = benz_cat_num.groupby('X5')['y'].mean()

mean_encode_X6 = benz_cat_num.groupby('X6')['y'].mean()

mean_encode_X8 = benz_cat_num.groupby('X8')['y'].mean()

benz_cat_num.loc[:,'X0_mean_enc']=benz_cat_num['X0'].map(mean_encode_X0)

benz_cat_num.loc[:,'X1_mean_enc']=benz_cat_num['X1'].map(mean_encode_X1)

benz_cat_num.loc[:,'X2_mean_enc']=benz_cat_num['X2'].map(mean_encode_X2)

benz_cat_num.loc[:,'X3_mean_enc']=benz_cat_num['X3'].map(mean_encode_X3)

benz_cat_num.loc[:,'X4_mean_enc']=benz_cat_num['X4'].map(mean_encode_X4)

benz_cat_num.loc[:,'X5_mean_enc']=benz_cat_num['X5'].map(mean_encode_X5)

benz_cat_num.loc[:,'X6_mean_enc']=benz_cat_num['X6'].map(mean_encode_X6)

benz_cat_num.loc[:,'X8_mean_enc']=benz_cat_num['X8'].map(mean_encode_X8)

benz_cat_num.head(1)
# Describe method is used to view some basic statistical details like percentile, mean, std etc. of a data frame of numeric values.

# Observation: Now, you can see the below stats how categorical values has been converted with mean-encodings

# And no outliers found for any of the categorical variables after encoding.

benz_cat_num.describe().T
# Re-verifying the sample top 1 record data

benz_cat_num.head(1)
# Dropping main categorical variables after mean encoding by considering meanencoding categorical variables here.

benz_cat_num=benz_cat_num.drop(['X0','X1','X2','X3','X4','X5','X6','X8'],axis=1)
# Generated HeatMap.

# Observation: From the below heatmap you able to see correlation between the categorical variables 

# after mean-encoding technique.

corr = benz_cat_num.corr()

ax = sns.heatmap( corr,vmin=-1, vmax=1, center=0,  cmap=sns.diverging_palette(20, 220, n=200),square=True)

ax.set_xticklabels(ax.get_xticklabels(),rotation=45,horizontalalignment='right');
# After feature engineering on categorical variables applying those on main dataset.

# Note: However, am not using the concat method here because their is no common unique field having on both dataframe. 

# and haven't considered ID column while doing mean encoding.

benz_eda_final=benz_eda.copy()
# Re-checking the shape of the data.

benz_eda_final.shape
# Mean Encoding technique has been choosen to ensure better encoding amoung other technique's based on the above plots.

# Verified the top 1 sample data after mean-encoding technique.

mean_encode_X0 = benz_eda_final.groupby('X0')['y'].mean()

#print(mean_encode)

mean_encode_X1 = benz_eda_final.groupby('X1')['y'].mean()

mean_encode_X2 = benz_eda_final.groupby('X2')['y'].mean()

mean_encode_X3 = benz_eda_final.groupby('X3')['y'].mean()

mean_encode_X4 = benz_eda_final.groupby('X4')['y'].mean()

mean_encode_X5 = benz_eda_final.groupby('X5')['y'].mean()

mean_encode_X6 = benz_eda_final.groupby('X6')['y'].mean()

mean_encode_X8 = benz_eda_final.groupby('X8')['y'].mean()

benz_eda_final.loc[:,'X0_mean_enc']=benz_eda_final['X0'].map(mean_encode_X0)

benz_eda_final.loc[:,'X1_mean_enc']=benz_eda_final['X1'].map(mean_encode_X1)

benz_eda_final.loc[:,'X2_mean_enc']=benz_eda_final['X2'].map(mean_encode_X2)

benz_eda_final.loc[:,'X3_mean_enc']=benz_eda_final['X3'].map(mean_encode_X3)

benz_eda_final.loc[:,'X4_mean_enc']=benz_eda_final['X4'].map(mean_encode_X4)

benz_eda_final.loc[:,'X5_mean_enc']=benz_eda_final['X5'].map(mean_encode_X5)

benz_eda_final.loc[:,'X6_mean_enc']=benz_eda_final['X6'].map(mean_encode_X6)

benz_eda_final.loc[:,'X8_mean_enc']=benz_eda_final['X8'].map(mean_encode_X8)

benz_eda_final.head(1)
# Dropping main categorical variables after mean encoding by considering meanencoding categorical variables here.

# Observations: We are having outliers on only target variable.

benz_eda_final=benz_eda_final.drop(['X0','X1','X2','X3','X4','X5','X6','X8'],axis=1)

benz_eda_final.describe().T
# Checking Null Values : We can see there are No Null Values 

benz_eda_final.isnull().sum()
# Remove Outliers, As observed based on target variable. As per standards, In case, if I consider Q1-25% and Q3-75% 

# unfortunately I loose nearly 30% of datapoints. However I have only small volume of data. If you observe below distplot

# & boxplot plotted on target variable it seems only little amount of outliers we have.So, In my opinion 10% of 

# outliers is enough here.

print(benz_eda_final.shape)

Q1 = benz_eda_final.quantile(0.005)

#print(Q1)

Q3 = benz_eda_final.quantile(0.995)

#print(Q3)

IQR = Q3 - Q1

#print(IQR)

benz_eda_final_outliers_01 = benz_eda_final[~((benz_eda_final < (Q1 - 1.5 * IQR)) |(benz_eda_final > (Q3 + 1.5 * IQR))).any(axis=1)]

print(benz_eda_final_outliers_01.shape)
# Distribution plot for output variable.

# Observation: 

# a) Left Side Plot: We can see it clearly that we have outliers and datapoints has been distributed mostly between around 

# 65 to 140.

# b) Right Side Plot: After outlier removal, we can see how data has been distributed as similar as bi-nominal dist.

figure,ax=plt.subplots(nrows=1,ncols=2,figsize=(15,6))

sns.distplot(benz_eda.y,ax=ax[0],kde=False)

sns.distplot(benz_eda_final_outliers_01.y,ax=ax[1],kde=False)

plt.show()

plt.tight_layout()
# Box plot for output variable.

# Observation: 

# a) Top Side Plot: We can see it clearly that we have outliers and datapoints has been distributed mostly at top side

# above the 140.

# b) Down Side Plot: After outlier removal, we can see removal of data points which clearly shows the difference.

plt.boxplot(benz_eda['y'])

plt.show()

plt.boxplot(benz_eda_final_outliers_01['y'])

plt.show()
# Generated HeatMap after removal of outliers and you can see the difference clearly here.

# Observation: Compare the difference between above and this below heatmap.

corr = benz_eda_final_outliers_01.corr()

ax = sns.heatmap( corr,vmin=-1, vmax=1, center=0,  cmap=sns.diverging_palette(20, 220, n=200),square=True)

ax.set_xticklabels(ax.get_xticklabels(),rotation=45,horizontalalignment='right');
# Copying final dataset after outliers removal to the PCA dataframe.

benz_final_PCA = benz_eda_final_outliers_01.copy()
# re-checking the shape of data

benz_final_PCA.shape
# Verifying the datatypes are integer type or not.

benz_final_PCA.dtypes
# Converting newly created mean encoding variable data type to integer data types.

benz_final_PCA.X0_mean_enc=benz_final_PCA.X0_mean_enc.astype(np.int64)

benz_final_PCA.X1_mean_enc=benz_final_PCA.X1_mean_enc.astype(np.int64)

benz_final_PCA.X2_mean_enc=benz_final_PCA.X2_mean_enc.astype(np.int64)

benz_final_PCA.X3_mean_enc=benz_final_PCA.X3_mean_enc.astype(np.int64)

benz_final_PCA.X4_mean_enc=benz_final_PCA.X4_mean_enc.astype(np.int64)

benz_final_PCA.X5_mean_enc=benz_final_PCA.X5_mean_enc.astype(np.int64)

benz_final_PCA.X6_mean_enc=benz_final_PCA.X6_mean_enc.astype(np.int64)

benz_final_PCA.X8_mean_enc=benz_final_PCA.X8_mean_enc.astype(np.int64)

benz_final_PCA.y=benz_final_PCA.y.astype(np.int64)

# Drop ID column which is not required for PCA.

benz_final_PCA.drop(['ID'],axis=1,inplace=True)
# Splitted target variable for PCA.

target = benz_final_PCA['y']
# Re-verifying the samples of finalized PCA dataset to go with.

pd.set_option('display.max_columns', None)

benz_final_PCA.head()
from sklearn import datasets

from sklearn.decomposition import PCA
# PCA

# Observations: As per below results, I've considered 26 PCA component for ELBOW Method to proceed further.

pca = PCA(n_components=377)

pca.fit_transform(benz_final_PCA.values)

print (pca.explained_variance_ratio_)

print (pca.explained_variance_)

variance_ratio_cum_sum=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)

print(variance_ratio_cum_sum)

print (pca.components_)
# Minimum number of components for which the variance is 98.43% is 26

pca.explained_variance_ratio_[:26].sum()
# Plotting relation between components and variance

# Observation: From below ELBOW curve it shows clearly I've picked right PCA component = 26.

plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.xlabel('Number of components')

plt.ylabel('Cumulative explained variance')

plt.annotate('26',xy=(26, .98))
# consider first 26 components as they are explaining the 98% of variation in the data.

# Now, the final dataframe is ready to build ML model. 

# However, this notebook main intense is to apply upto EDA only.So, not proceeding further in building the ML model.

x_pca = PCA(n_components=26)

benz_final_PCA_final = x_pca.fit_transform(benz_final_PCA)

# correlation between the variables after transforming the data with PCA

# Observation: Final dataset is completely fit to build ML model.

correlation = pd.DataFrame(benz_final_PCA_final).corr()

sns.heatmap(correlation, vmax=1, square=True,cmap='viridis')

plt.title('Correlation between different features')
# Copying final dataset after outliers removal to the ICA dataframe.

benz_final_ICA = benz_eda_final_outliers_01.copy()
# Drop ID column which is not required for ICA.

benz_final_ICA.drop(['ID'],axis=1,inplace=True)
from sklearn.decomposition import FastICA
#ICA

# Observations: Considered all ncomponents for ICA and transformed data based on Fast ICA.

ica = FastICA(n_components=377, random_state=420)

benz_final_ICA_Final = ica.fit_transform(benz_final_ICA)

# Generated HeatMap after removal of outliers and you can see the difference clearly here.

# Observation: Compare the difference between above and this below heatmap.

correlation = pd.DataFrame(benz_final_ICA_Final).corr()

ax = sns.heatmap( correlation,vmin=-1, vmax=1, center=0,  cmap=sns.diverging_palette(20, 220, n=200),square=True)

ax.set_xticklabels(ax.get_xticklabels(),rotation=45,horizontalalignment='right');
#Remove Outliers by using standards.

print(benz_cat_num.shape)

Q1 = benz_cat_num.quantile(0.25)

#print(Q1)

Q3 = benz_cat_num.quantile(0.75)

#print(Q3)

IQR = Q3 - Q1

#print(IQR)

benz_cat_num_outliers_removal = benz_cat_num[~((benz_cat_num < (Q1 - 1.5 * IQR)) |(benz_cat_num > (Q3 + 1.5 * IQR))).any(axis=1)]

print(benz_cat_num_outliers_removal.shape)
#Describe method.

# Observation: Now you can see the target variable stats and their is no outliers in dataset.

benz_cat_num_outliers_removal.describe().T
# Distribution method: Clearly shows that how data has been distributed.

sns.distplot(benz_cat_num_outliers_removal['y'],kde = False)
# Count plot for categorical variables.

# Observation: we can see clearly how datapoints has been distributed across the distinct values 

# for all categorical features after outliers removal.

plt.rcParams['figure.figsize'] = (20.0, 10.0)

plt.rcParams['font.family'] = "serif"

fig, ax =plt.subplots(4,2)

sns.countplot(benz_cat_num_outliers_removal['X0_mean_enc'], ax=ax[0,0])

sns.countplot(benz_cat_num_outliers_removal['X1_mean_enc'], ax=ax[0,1])

sns.countplot(benz_cat_num_outliers_removal['X2_mean_enc'], ax=ax[1,0])

sns.countplot(benz_cat_num_outliers_removal['X3_mean_enc'], ax=ax[1,1])

sns.countplot(benz_cat_num_outliers_removal['X4_mean_enc'], ax=ax[2,0])

sns.countplot(benz_cat_num_outliers_removal['X5_mean_enc'], ax=ax[2,1])

sns.countplot(benz_cat_num_outliers_removal['X6_mean_enc'], ax=ax[3,0])

sns.countplot(benz_cat_num_outliers_removal['X8_mean_enc'], ax=ax[3,1])

fig.show()