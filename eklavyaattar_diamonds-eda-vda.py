# import the necessary pacakges
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#read data file
df_diamondDataComplete = pd.read_csv("../input/diamonds.csv")
df_diamondDataComplete.head(10)
# Create a new dataframe so as to have a copy of the original csv imported dataframe
df_diamondData= df_diamondDataComplete
df_diamondData.head()
# Drop unwanted columns: Columns x, y & z
df_diamondData.drop(['x', 'y','z'], axis =1, inplace = True)
df_diamondData = df_diamondData.drop(df_diamondData.columns[0], axis=1) 
# View Data
print(df_diamondData.head())
# Check Column Names
print(df_diamondData.columns)
# Check datatypes of the columns
df_diamondData.dtypes
# Check summary
df_diamondData.describe()
#1. Checking fot NULLS
df_diamondData.isnull().sum()
#2. Checking fot Whitespaces
np.where(df_diamondData.applymap(lambda x: x == ' '))
#3. Checking for outlier values based on the given information for each of the desired columns
# Create a dummy datafarme as we are only checking the values
df_diamondData_dummy = df_diamondData
# -A. CARAT (values to be present between 0.2 to 5.01, as given in the handout)
print(df_diamondData_dummy[(df_diamondData_dummy['carat'] < 0.2) | (df_diamondData_dummy['carat'] > 5.01)])
# -B. CUT (values to be present Fair, Good, Very Good, Premium, Ideal)
value_list_cut = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
print(df_diamondData[~df_diamondData.cut.isin(value_list_cut)])
# -C. COLOR ( values to be present D ,E ,F ,G ,H ,I ,J)
value_list_color = ['D' ,'E' ,'F' ,'G' ,'H' ,'I' ,'J']
print(df_diamondData[~df_diamondData.color.isin(value_list_color)])
# -D. CLARITY ( values to be present I1 ,SI2 ,SI1 ,VS2 ,VS1 ,VVS2 ,VVS1 ,IF)
value_list_clarity = ['I1' ,'SI2' ,'SI1' ,'VS2' ,'VS1' ,'VVS2' ,'VVS1' ,'IF']
print(df_diamondData[~df_diamondData.clarity.isin(value_list_clarity)])
# -E. PRICE ( values to be present between $326 to $18,823)
print(df_diamondData_dummy[(df_diamondData_dummy['price'] < 326) | (df_diamondData_dummy['price'] > 18823)])
# For table and depth we dont have any accepted range of values given hence we will use an
# outlier function to detecrt outliers for table and depth columns
# Detect outlier fucntion
def outliers_iqr(ys):
    quartile_1, quartile_3 = np.percentile(ys, [25, 75])
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    return np.where((ys > upper_bound) | (ys < lower_bound))
# -F. TABLE
# Call outlier function and Print outlier' s indexed values
outliers_iqr(ys=df_diamondData_dummy['table'])
a = np.array([outliers_iqr(ys=df_diamondData_dummy['table'])])
a.size
# -G. DEPTH
outliers_iqr(ys=df_diamondData_dummy['depth'])
b = np.array([outliers_iqr(ys=df_diamondData_dummy['depth'])])
b.size
# Create a new dataframe to strore the imputed values
df_diamondData_mod = df_diamondData
#Imputation for Table and Depth columns using quantile method
#1. TABLE
# Creating the accepted range of values
down_quantiles_diamondtable = df_diamondData_mod.table.quantile(0.25)
up_quantiles_diamondtable   = df_diamondData_mod.table.quantile(0.75)
# Getting the minimum and maximum values
outliers_low_diamondtable = (df_diamondData_mod.table < down_quantiles_diamondtable)
outliers_high_diamondtable = (df_diamondData_mod.table > up_quantiles_diamondtable)
# Updating the column with the quantile values
df_diamondData_mod.table  = df_diamondData_mod.table.mask(outliers_low_diamondtable,down_quantiles_diamondtable)
df_diamondData_mod.table  = df_diamondData_mod.table.mask(outliers_high_diamondtable,up_quantiles_diamondtable)
# Call outlier function and check for the count of outlier values
a = np.array([outliers_iqr(ys=df_diamondData_mod['table'])])
a.size
#2. DEPTH
# Creating the accepted range of values
down_quantiles_diamonddepth = df_diamondData_mod.depth.quantile(0.25)
up_quantiles_diamonddepth = df_diamondData_mod.depth.quantile(0.75)
# Getting the minimum and maximum values
outliers_low_diamonddepth = (df_diamondData_mod.depth < down_quantiles_diamonddepth)
outliers_high_diamonddepth = (df_diamondData_mod.depth > up_quantiles_diamonddepth)
# Updating the column with the quantile values
df_diamondData_mod.depth  = df_diamondData_mod.depth.mask(outliers_low_diamonddepth,down_quantiles_diamonddepth)
df_diamondData_mod.depth  = df_diamondData_mod.depth.mask(outliers_high_diamonddepth,up_quantiles_diamonddepth)
# Call outlier function and check for the count of outlier values
b = np.array([outliers_iqr(ys=df_diamondData_mod['depth'])])
b.size
# Assign the imputed dataframe to a new dataframe
df_diamond = df_diamondData_mod
#1. Realtionship between clarity and price
pd.crosstab(df_diamond["clarity"], df_diamond["price"], margins= True)
pd.crosstab(df_diamond["clarity"], columns="count")
#2.Realtionship between cut and price
pd.crosstab(df_diamond["cut"], df_diamond["price"], margins= True)
pd.crosstab(df_diamond["cut"], columns="count")
#3. Realtionship between color and price
pd.crosstab(df_diamond["color"], df_diamond["price"], margins= True)
pd.crosstab(df_diamond["color"], columns="count")
#4. Realtionship between carat and price
pd.crosstab(df_diamond["carat"], df_diamond["price"], margins= True)
pd.crosstab(df_diamond["carat"], columns="count")
#5. Price table
pd.crosstab(index=df_diamond["price"], columns="count")
# Highest price value - 
pd.crosstab(index=df_diamond["price"], columns="count").nlargest(10, 'count')
#6. Realtionship between carat and cut
pd.crosstab(index=df_diamond["carat"], columns=df_diamond["cut"])
#7. Realtionship between Clarity and Color
pd.crosstab(df_diamond["clarity"],df_diamond["color"], margins = True)
# 1. To check realtionshiop between clarity and price using Violin plot
sns.violinplot(x='clarity', y='price', data=df_diamond)
plt.show()
# 2. To check realtionshiop between cut, clarity and price using factor plot
sns.factorplot(x='clarity', y='price', data=df_diamond, hue = 'cut', 
               kind = 'bar', size = 8) 
plt.show()
# 3. To check realtionshiop between clarity, carat using violin plot
sns.violinplot(x='clarity', y='carat', data=df_diamond)
plt.show()
# 4. To check realtionshiop between price and carat using Joint plot
sns.jointplot(x='carat', y='price', data=df_diamond)
plt.show()
# 5. Cut - Price relation
sns.factorplot(x='cut', y='price', data=df_diamond, hue = 'color', 
               kind = 'bar', size = 8) 
plt.show()
# 6. To check realtionshiop between cut, price w.r.t. calrity using factor plot
sns.factorplot(x='clarity', y='price', data=df_diamond, hue = 'cut', 
               kind = 'bar', size = 8) 
plt.show()
# 7. To check realtionshiop between color, price w.r.t. calrity using factor plot
sns.factorplot(x='clarity', y='price', data=df_diamond, hue = 'color', 
               kind = 'bar', size = 8)
plt.show()
# 8. To check realtionshiop between color and cut
sns.countplot(y="cut", hue="color", data=df_diamond)
plt.show()