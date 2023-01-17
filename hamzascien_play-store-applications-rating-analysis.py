import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import rcParams
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
%matplotlib inline
data=pd.read_csv('../input/google-play-store-apps/googleplaystore.csv')
data.head()
data.info()
data.shape
#handling missing data
totale=data.isnull().sum().sort_values(ascending=False)
percentage=(data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)
missing_data=pd.concat([totale,percentage],axis=1,keys=['totale','percent'])
missing_data.head(6)
data.dropna(how='any',inplace=True)
totale=data.isnull().sum().sort_values(ascending=False)
percentage=(data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)
missing_data=pd.concat([totale,percentage],axis=1,keys=['totale','percent'])
missing_data.head(6)
data.Rating.describe()
sns.set(style='darkgrid')
rcParams['figure.figsize']=11,9
g=sns.kdeplot(data.Rating,color='y',shade=True)
g.set_xlabel('Rating')
g.set_ylabel('Frequency')
plt.title("Distribution of Rating",size=30)
print('Dataset has a ', len(data['Category'].unique()) , "categories")

print("\n", data['Category'].unique())
## counting 
categories=data.groupby(['Category']).App.count().reset_index().sort_values(by='App',ascending=False)
a=sns.barplot(x='Category',y='App',data=categories,palette='nipy_spectral')
a.set_xticklabels(a.get_xticklabels(), rotation=90, ha="right")
plt.title('Count of app in each category',size=30)
a=sns.boxplot(x='Rating',y='Category',data=data,palette='magma')
plt.title('Raing of Apps in each category',size=25)
# converting Reviews's data type to numerical
data.Reviews=data.Reviews.astype(int)
data.Reviews.head()
# Reviews distribution
rcParams['figure.figsize']=11,9
a=sns.kdeplot(data.Reviews,color='y',shade=True)
a.set_xlabel('Reviews',size=17)
a.set_ylabel('Frequency',size=17)
plt.title('Distribution of Reviews',size=25)
data[data.Reviews<1000000].shape
data[data.Reviews>3000000].head()
a=sns.jointplot('Reviews','Rating',data=data,size=9,color='y')
rcParams['figure.figsize']=11,9
a=sns.regplot(x='Reviews',y='Rating',data=data[data.Reviews<1000000],color='g')
plt.title('Rating vs Reviews',size=25)
data.Installs.unique()
## It is preferable to encode it by numbs
data.Installs=data.Installs.replace(r'[\,\+]', '', regex=True).astype(int)
install_sorted=sorted(data.Installs.unique())
install_sorted
data.Installs.replace(install_sorted,range(0,len(install_sorted),1),inplace=True)
from scipy.stats import spearmanr
a=sns.jointplot(x='Installs',y='Rating',data=data,kind='kde',size=9,color='y',stat_func=spearmanr)
a=sns.regplot(x='Installs',y='Rating',data=data,color='pink')
plt.title('Relation Between Rating And Installs',size=20)
data.Size.unique()
data.Size.replace('Varies with device',np.nan,inplace=True)
## Change size values to the same units
data.Size=(data.Size.replace(r'[kM]+$', '', regex=True).astype(float) * data.Size.str.extract(r'[\d\.]+([KM]+)', expand=False)
            \
            .replace(['k','M'], [10**3, 10**6]).astype(float))
## filling null values by the mean of each category
data.Size.fillna(data.groupby('Category')['Size'].transform('mean'),inplace=True)
data.Size
a=sns.jointplot(x='Size',y='Rating',data=data,color='y',size=9,kind='kde',stat_func=spearmanr)

a=sns.regplot(x='Size',y='Rating',data=data,color='black')
plt.title('Rating vs Size',size=25)
data.Type.unique()
a=sns.countplot(x="Type",data=data)
percent=round(data.Type.value_counts(sort=True)/data.Type.count()*100,2).astype(str)+'%'
Type_values=pd.concat([data.Type.value_counts(sort=True),percent],axis=1,keys=['Totale','percent'])
Type_values
type_dum=pd.get_dummies(data['Type'])
type_dum.drop(['Paid'],axis=1,inplace=True)
data=pd.concat([data,type_dum],axis=1)

# now we drop Type column
data.drop(['Type'],axis=1,inplace=True)
data.Price.unique()
data.Price=data.Price.apply(lambda x:float(x.replace('$','')))
data.Price.describe()
a=sns.regplot(x='Price',y='Rating',data=data,color='m')
plt.title('Relation Between Price and Rating',size=25)
## let's us check little bit for more details
data[data.Price==0].shape
bins=[-1,0.98,1,3,5,16,30,401]
labels=['Free','Cheap','Not Cheap','Medium','Expensive','Very expensive','Extra expensive']
data['Price category']=pd.cut(data.Price,bins,labels=labels)
data.groupby(['Price category'], as_index=False)['Rating'].mean()
a=sns.catplot(x='Rating',y='Price category',data=data,kind='bar',height=10,palette='mako')
plt.title('Barplot of Rating\'s mean for each Price category',size=25)
data.Genres.unique()
data.Genres.value_counts()
data['Genres'] = data['Genres'].str.split(';').str[0]
data.Genres.value_counts()
## We can Group Music & Audio  as  Music
data['Genres'].replace('Music & Audio', 'Music',inplace = True)
data.groupby('Genres',as_index=False)['Rating'].mean().describe()
a=sns.catplot(x='Rating',y='Genres',data=data,kind='bar',height=10,palette='coolwarm')
plt.title('Barplot of Rating\'s mean for each Genre',size=25)
data['Content Rating'].unique()
data['Content Rating'].value_counts()
plt.figure(figsize=(13,10))
a=sns.boxenplot(x='Content Rating',y='Rating',data=data,palette='Accent')
plt.title('boxen plot of Rating Vs Content Rating',size=25)
