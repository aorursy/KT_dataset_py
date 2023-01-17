#loading libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import missingno as msno
pd.pandas.set_option('display.max_columns',None)
pd.pandas.set_option('display.max_rows',None)
#loading data
dataset=pd.read_csv('/kaggle/input/home-data-for-ml-course/train.csv')
df=dataset.copy()
#number of rows and columns
df.shape
df.describe().T
#summary of a dataframe
df.info()
#creating list with feature belonging to respective datatype

numerical_feature=[feature for feature in df.columns if df[feature].dtype !='object']
print('There are {} numerical features.'.format(len(numerical_feature)))

year=[feature for feature in numerical_feature if "Yr" in feature or "Year" in feature]

discrete_feature=[feature for feature in numerical_feature if (df[feature].nunique())<25 and feature not in year+['Id']]
print('There are {} are discrete features'.format(len(discrete_feature)) )

continuous_feature=[feature for feature in numerical_feature if feature not in discrete_feature and feature not in year+['Id']]
print('There are {} are continuous features'.format(len(continuous_feature)) )

cat_feature=[feature for feature in df.columns if df[feature].dtypes=='object']
print('There are {} categorical features.'.format(len(cat_feature)))
df['SalePrice'].describe()
null_feature=[]
null_values=[]

for feature in df.columns:
    if df[feature].isnull().sum()>=1:
        null_feature.append(feature)
        null_values.append(df[feature].isnull().sum())
        
df_missing={'Feature':null_feature,'No.of missing values':null_values}
data_missing=pd.DataFrame(df_missing)

plt.figure(figsize=(10,6)) 
sns.barplot(x='Feature',y='No.of missing values',data=data_missing,palette='Set1')
plt.xticks(rotation=90)
plt.title("No. of missing values")
sns.set(style="whitegrid")
msno.heatmap(df)
df[continuous_feature].hist(bins=25, figsize=(22,8), layout=(2,8));
feat=[]
for feature in continuous_feature:
    data=df.copy()
    if 0 in data[feature].unique():
        pass
    else:
        feat.append(feature)
np.log(data[feat]).hist(bins=25, figsize=(20,5), layout=(1,5),color='green');
sns.pairplot(np.log(df[feat]))
corr=np.log(df[continuous_feature]).corr()
corr['SalePrice'].sort_values(ascending=False)
def outlier_function(df, col_name):
    first_quartile = np.percentile(np.array(df[col_name].tolist()), 25)
    third_quartile = np.percentile(np.array(df[col_name].tolist()), 75)
    IQR = third_quartile - first_quartile
                      
    upper_limit = third_quartile+(3*IQR)
    lower_limit = first_quartile-(3*IQR)
    outlier_count = 0
                      
    for value in df[col_name].tolist():
        if (value < lower_limit) | (value > upper_limit):
            outlier_count +=1
    return lower_limit, upper_limit, outlier_count
for feature in continuous_feature:
    if outlier_function(df,feature)[2] > 1:
        print('{}: {} outliers'.format(feature,outlier_function(df,feature)[2]))
corr=df[continuous_feature].corr()
sns.heatmap(corr)
sns.pairplot(df[year])
for feature in discrete_feature:
    if outlier_function(df,feature)[2] > 1:
        print('{}: {} outliers'.format(feature,outlier_function(df,feature)[2]))
f = plt.figure(figsize=(20,20))
for i in range(len(discrete_feature)):
    f.add_subplot(6, 3, i+1)
    g=df.groupby([discrete_feature[i]])['SalePrice'].median().sort_values()
    ax=sns.boxplot(x=discrete_feature[i],y='SalePrice',data=df,order=list(g.index),palette='Blues')
    ax=sns.stripplot(x=discrete_feature[i],y='SalePrice',data=df,color='red',jitter=0.2,size=2.5)
    plt.title(discrete_feature[i],fontsize=15)

plt.tight_layout()
plt.show()
corr=df[discrete_feature].corr()
sns.heatmap(corr)
plt.title('Correlation among discrete variable');
total_cat=[]
for feature in cat_feature:
        total_cat.append(df[feature].nunique())
        
total_categories={'Feature':cat_feature,'No.of categories':total_cat}
total_categories=pd.DataFrame(total_categories)

plt.figure(figsize=(14,6)) 
sns.barplot(x='Feature',y='No.of categories',data=total_categories,palette='Set1')
plt.xticks(rotation=90) 
plt.title("Number of Categories per feature", fontsize= 15)
sns.set(style="whitegrid")
plt.figure(figsize=(20,6))
g=df.groupby(['Neighborhood'])['SalePrice'].median().sort_values()
sns.boxplot(x='Neighborhood', y='SalePrice', data=df,order=list(g.index),palette='cubehelix')
sns.stripplot(x='Neighborhood', y='SalePrice', data=df,color='orange',jitter=0.2,size=2.5)
plt.title('Sale Price with respect to Neighbourhood',fontsize=20);
plt.figure(figsize=(18,6))
g=df.groupby(['Exterior1st'])['SalePrice'].median().sort_values()
sns.boxplot(x='Exterior1st', y='SalePrice', data=df,order=list(g.index),palette='cubehelix')
sns.stripplot(x='Exterior1st', y='SalePrice', data=df,color='orange',jitter=0.2,size=2.5)
plt.title('Sale Price with respect to Exterior covering on house',fontsize=20);
plt.figure(figsize=(18,6))
g=df.groupby(['Exterior2nd'])['SalePrice'].median().sort_values()
sns.boxplot(x='Exterior2nd', y='SalePrice', data=df,order=list(g.index),palette='cubehelix')
sns.stripplot(x='Exterior2nd', y='SalePrice', data=df,color='orange',jitter=0.2,size=2.5)
plt.title('Sale Price with respect to Exterior covering on house (if more than one material)',fontsize=20);
cat_f=cat_feature.copy()
cat_f.remove('Neighborhood')
cat_f.remove('Exterior1st')
cat_f.remove('Exterior2nd')
f = plt.figure(figsize=(20,40))
for i in range(len(cat_f)):
    f.add_subplot(14, 3, i+1)
    g=df.groupby([cat_f[i]])['SalePrice'].median().sort_values()
    ax=sns.violinplot(x=cat_f[i],y='SalePrice',data=df,order=list(g.index),palette='BuGn_r')
    ax=sns.stripplot(x=cat_f[i],y='SalePrice',data=df,jitter=0.1,size=2.5,color='magenta')
    plt.title(cat_f[i],fontsize=15)

plt.tight_layout()
plt.show()