import pandas as pd

import numpy as np



df = pd.read_csv('../input/ibm-sql-course-chicago-crime-and-public-schools/Census_Data_-_Selected_socioeconomic_indicators_in_Chicago__2008___2012-v2.csv')
df.dtypes
df_num = df.drop(['COMMUNITY_AREA_NAME'], axis = 1)

df_cat = df['COMMUNITY_AREA_NAME'].copy()
df_num = df_num.astype(np.float64).head()
df.rename(columns = {"PERCENT HOUSEHOLDS BELOW POVERTY":"household_BPL"},inplace = True )

df.rename(columns = {"COMMUNITY_AREA_NUMBER":'comm_area_no'}, inplace = True)
df.dtypes
df.rename(columns = {"COMMUNITY_AREA_NAME":'comm_area_name'}, inplace = True)

df.rename(columns = {"PERCENT OF HOUSING CROWDED":'percent_housing_crowded'}, inplace=True)

df.rename(columns = {"PERCENT AGED 16+ UNEMPLOYED":'per_16_unemp'}, inplace=True)

df.rename(columns = {"PERCENT AGED 25+ WITHOUT HIGH SCHOOL DIPLOMA":'per_25_without_highschool'}, inplace=True)

df.rename(columns = {"PERCENT AGED UNDER 18 OR OVER 64":'per_under18_over64'}, inplace=True)

df.rename(columns = {"PER_CAPITA_INCOME ":'per_capita_income'}, inplace = True)

df.rename(columns = {"HARDSHIP_INDEX":'hardship_index'}, inplace = True)
df.head()
#simple feature scaling on per_capita_income - Range 0 to 1

df_norm = df

df_norm['per_capita_income'] = df_norm['per_capita_income']/df_norm['per_capita_income'].max()
# min-max scaling

df_min_max = df

df_min_max['per_capita_income'] = (df_min_max['per_capita_income'] - df_min_max['per_capita_income'].min())/(

    df_min_max['per_capita_income'].max() - df_min_max['per_capita_income'].min() )
df_min_max.head()
# z scores range -3 to 3 usually

df_z = df

df_z["per_capita_income"] = ( df_z["per_capita_income"] - df_z["per_capita_income"].mean() )/df_z["per_capita_income"].std(ddof=1)

df_z.head()
# drop missing values - dropna()



#df_norm.dropna().info()
# Identify the column with missing value and replace the values there

df_z.info()
df_z[["comm_area_no"]].isna().tail()
df_z[["comm_area_no"]].tail()
df_z["comm_area_no"]  =  df_z["comm_area_no"].replace(np.nan, np.float(78))
df_z[["comm_area_no"]].tail()
df_z.head()
df_z['hardship_index'].isna().tail()
df_z['hardship_index'].tail()
print("The mean is", df_z['hardship_index'].mean(),"and median is", df_z['hardship_index'].median(),

     "and the max is", df_z['hardship_index'].max(), "and the min is", df_z['hardship_index'].min())
import matplotlib.pyplot as plt

%matplotlib inline

df_z.hist('hardship_index', bins=25)
mean = round(df_z['hardship_index'].mean(),1)

#df_z['hardship_index'].replace(np.nan, mean)

mean
df_z['hardship_index'] = df_z['hardship_index'].astype(np.float64)
df_z['hardship_index'].dtypes
df_z['hardship_index'].replace(np.nan, mean)
df_z.info()
bins = np.linspace(min(df_z['hardship_index']),max(df_z['hardship_index']), 4)

print(bins)

#bins = np.linspace(min(df_z['hardship_index']), max(df_z['hardship_index']), 4)
group_names = ["Easy", "Medium", "High"]

df_z['hardship_binned'] = pd.cut(df_z['hardship_index'], bins = bins, labels = group_names)
df_z.head()
# One hot encoded dataframe

df_demo = pd.get_dummies(df_z['hardship_binned'])
# Column binding it with the original data frame

df_onehot = pd.concat([df_z, df_demo], axis = 1)
df_onehot.head()
%%capture



! pip install seaborn
import seaborn as sns
df.columns
# Using group_by (Per capita income by community area name)

boxp = df.groupby(['comm_area_name'], as_index=False)[['comm_area_name','per_capita_income']].mean()
from scipy import stats
#Pearson correlation coefficient



df['hardship_index'] = df['hardship_index'].replace(np.nan,mean)

pearsonr, pvalue = stats.pearsonr(df['per_capita_income'], df['hardship_index'])

print("The Pearson r is:", pearsonr,"and the p-value is:", pvalue)
# pre-ANOVA grouping



bins = np.linspace(df['hardship_index'].min(), df['hardship_index'].max(), 4)

df.drop(['hardship_binned'], inplace=True, axis = 1)

df['hardship_binned'] = pd.cut(df['hardship_index'], bins = bins, labels = ["Low", "Medium", "high"], include_lowest=True)

grouped = df[['hardship_binned', 'per_capita_income']].groupby(['hardship_binned'])



# To get a value at 0 index of a column

df.at[0,'per_capita_income']
grouped.get_group('Low')['per_capita_income']
# ANOVA



fval, pval = stats.f_oneway(grouped.get_group('Low')['per_capita_income'], 

               grouped.get_group('Low')['per_capita_income'],

               grouped.get_group('Low')['per_capita_income'])

print("The f val is:", fval,"and the p-value is", pval)
# To find the number of rows and columns of a dataframe



df.shape[0] #rows

df.shape[1] #columns
# To find the column names as a list



df.columns.values