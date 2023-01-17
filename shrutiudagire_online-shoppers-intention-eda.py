import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
df=pd.read_csv("../input/online-shoppers-intention/online_shoppers_intention.csv")
df.shape
df.head()
df.shape
Numeric=['Administrative',
 'Administrative_Duration',
 'Informational',
 'Informational_Duration',
 'ProductRelated',
 'ProductRelated_Duration',
 'BounceRates',
 'ExitRates',
 'PageValues',
 'SpecialDay']
Categorical=['Month',
 'OperatingSystems',
 'Browser',
 'Region',
 'TrafficType',
 'VisitorType',
 'Weekend',
 'Revenue']
df.corr()
# We can see multicollinearity
df.corr()['Revenue'].sort_values(ascending=False)
plt.figure(figsize=(20,10))
sns.heatmap(df.corr(),annot=True)
plt.figure(figsize=(15,5))
sns.scatterplot(df['ProductRelated'],df['ProductRelated_Duration'],hue=df['Revenue'])
plt.figure(figsize=(15,5))
sns.scatterplot(df['ExitRates'],df['BounceRates'],hue=df['Revenue'])
sns.scatterplot(df['Administrative_Duration'],df['Administrative'],hue=df['Revenue'])
sns.scatterplot(df['Informational_Duration'],df['Informational'],hue=df['Revenue'])
df.head()
df['Revenue'].value_counts()
list1=['Administrative_Duration','Informational','ProductRelated_Duration']
for i in list1:
    sns.barplot(df['Revenue'],df[i])
    plt.show()
col=['Administrative', 'Administrative_Duration', 'Informational',
       'Informational_Duration', 'ProductRelated', 'ProductRelated_Duration',
       'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay']
for i in col:
    df[i].plot(kind='kde')
    plt.title(i)
    plt.show()
plt.figure(figsize=(20,5))
sns.countplot(df['Weekend'],hue=df['Revenue'])
weekend_df=pd.DataFrame()
weekend_df['Weekend']=[True,False]
weekend_df['Revenue User']=[df[(df['Weekend']==True) & (df['Revenue']==True)].shape[0]/df[(df['Weekend']==True)].shape[0],df[(df['Weekend']==False) & (df['Revenue']==True)].shape[0]/df[(df['Weekend']==False)].shape[0]]
weekend_df['Non Revenue User']=[df[(df['Weekend']==True) & (df['Revenue']==False)].shape[0]/df[(df['Weekend']==True)].shape[0],df[(df['Weekend']==False) & (df['Revenue']==False)].shape[0]/df[(df['Weekend']==False)].shape[0]]

weekend_df.set_index('Weekend',inplace=True,drop=True)
weekend_df=weekend_df.sort_values(by='Revenue User')
weekend_df
weekend_df.plot(kind='bar',figsize=(20,7))
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper right')
plt.figure(figsize=(20,7))
sns.boxplot(df['Weekend'],df['SpecialDay'])
sns.countplot(df['Month'],hue=df['Weekend'])
weekend_month=pd.crosstab(df['Month'],df['Weekend'])
weekend_month=weekend_month.sort_values(by=True,ascending=False)
weekend_month
sns.countplot(df['TrafficType'],hue=df['Weekend'])
sns.boxplot(df['Revenue'],df['BounceRates'])
plt.figure(figsize=(20,7))
# sns.boxplot(df['Weekend'],df['ExitRates'],hue=df['Revenue'])
sns.boxplot(df['Revenue'],df['ExitRates'])
# ExiteRates for Revenue is Low as compared to exiteRate of nonRevenue user
sns.countplot(df['Browser'],hue=df['Revenue'])
sns.countplot(df['Month'],hue=df['Revenue'])
week=dict(list(df.groupby(['Month'])))
month_list=df['Month'].value_counts().index
special_day_weekends=[]
special_day_weekday=[]
final=[]
for i in month_list:
    list1=[]
    special_day_weekends.append(week[i]['SpecialDay'].count())
    list1.append(week[i]['SpecialDay'].value_counts().index)
    final.append(list1)
special_df=pd.DataFrame({"Month":month_list,"Special day":special_day_weekends,"Total":final})
special_df
plt.figure(figsize=(20,10))
sns.countplot(df['Month'],hue=df['Region'])
plt.figure(figsize=(20,10))
sns.countplot(df['Month'],hue=df['VisitorType'])
plt.figure(figsize=(20,9))
sns.countplot(df['Month'],hue=df['TrafficType'])
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper right')
sns.countplot(df['OperatingSystems'])
sns.countplot(df['Browser'])
plt.figure(figsize=(20,10))
sns.countplot(df['Region'],hue=df['TrafficType'])
plt.legend(loc="upper right")
plt.figure(figsize=(20,10))
sns.countplot(df['Region'],hue=df['VisitorType'])
sns.boxplot(df['VisitorType'],df['BounceRates'])
plt.figure(figsize=(20,10))
sns.boxplot(df['VisitorType'],df['BounceRates'],hue=df['Revenue'])
sns.boxplot(df['VisitorType'],df['ExitRates'],hue=df['Revenue'])
sns.countplot(df['VisitorType'],hue=df['Revenue'])
pd.crosstab(df['VisitorType'],df['Revenue'])
422/(1272+422),1470/(1470+9081)
sns.countplot(df['Weekend'],hue=df['VisitorType'])
sns.boxplot(df['VisitorType'],df['ExitRates'])
plt.figure(figsize=(20,10))
sns.boxplot(df['Revenue'],df['BounceRates'])

