#Importing the required modules
import numpy as np 
import pandas as pd 
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

df=pd.read_csv("../input/googleplaystore.csv")
print(df.columns)
print("="*80)
print(df.head())
print(df['Category'].unique())
print("="*80)
print(df['Type'].unique())
print("="*80)
print(df['Content Rating'].unique())
print("="*80)
print(df['Genres'].unique())
df.drop('Genres',axis=1,inplace=True)
df = df.drop(df[ df['Category'] == '1.9' ].index, axis=0)
data=df.drop_duplicates(subset=['App'])
data.info()
data['Rating'].fillna(0)
data['Content Rating'].fillna(method='ffill')
data['Current Ver'].fillna(1)
data['Android Ver'].fillna(method='bfill')
data['Type'].fillna('Free')
pass
#print(df['Type'].unique())
data.info()
sns.set_context({"figure.figsize": (20, 5)})
c=sns.countplot(x="Category",data=data, palette = "Set3",order=reversed(data['Category'].value_counts().index))
c.set_xticklabels(c.get_xticklabels(), rotation=-65, ha="left")
c.set_yticklabels(c.get_yticklabels(), rotation=0, ha="right") #just to check
plt.title('Count of app in different category',size = 40)
data.groupby('Category').mean()['Rating'].plot("barh",figsize=(10,13),title ="Rating in difft Category");
#print(data.groupby("Category").mean().max())

sdf=data.groupby("Category").mean()
sdf['Cat']=sdf.index

print("The category which got the Max rating :",sdf['Rating'].idxmax())
print("The category which got the Min rating :",sdf['Rating'].idxmin())

sx=sdf.loc[sdf['Rating'].idxmax()]               #This will retunr entire row of the max value
si=sdf.loc[sdf['Rating'].idxmin()] 

#print(type(sx))                   #Series

print("Mean rating for",sdf['Rating'].idxmax(),"is : ",sx['Rating'])
print("Mean rating for",sdf['Rating'].idxmin(),"is :",si['Rating'])


#df[df['Rating'] == df['Rating'].max()]       #This one will list all rows that have max Rating
sns.set_context({"figure.figsize": (20, 5)})
c=sns.countplot(x="Content Rating",hue='Type',data=data, palette = "Set1",order=reversed(data['Content Rating'].value_counts().index))
c.set_xticklabels(c.get_xticklabels(), rotation=0, ha="right")
plt.title('Apps by there content ratings',size = 40)
data['Installs']=data['Installs'].apply(lambda x : str(x).replace('+',''))
data['Installs']=data['Installs'].apply(lambda x : str(x).replace(',',''))
print(data['Installs'].unique())

data.groupby('Installs')['Rating'].mean().plot("barh",figsize=(10,13),title ="Rating vs Installs");

