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
#Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
gps = pd.read_csv('../input/googleplaystore.csv')

gps.columns = [x.strip().replace(' ', '_') for x in gps.columns]
gps.rename(columns={'Price':'Price_in_Dollars'}, inplace=True)

gps.Size=[x.strip().replace('1,000','1000') for x in gps.Size]
gps.Size=[x.strip().replace('Varies with device','9999') for x in gps.Size]
def num_size(Size):
    if Size[-1] == 'k':
        return float(Size[:-1])*1000
    else:
        return float(Size[:-1])*1000000
gps['Size']=gps['Size'].map(num_size).astype(float)
def func(x):
    try:
        float(x)
        return True
    except ValueError:
        return False

a=gps.Size.apply(lambda x: func(x))
a.head()
#To find if there are any string values 
gps.Size[~a].value_counts()
gps.Installs=[x.strip().replace('+','') for x in gps.Installs]
gps.Installs=[x.strip().replace(',','') for x in gps.Installs]
gps[gps['Installs']=='Free']
gps.drop([10472], inplace=True)
gps.Installs[~a].value_counts()

gps['Android_Ver']=gps['Android_Ver'].astype('str')
gps['Android_Ver'] = gps.Android_Ver.apply(lambda x: x.replace(' and up', ''))
gps['Android_Ver'] = gps.Android_Ver.apply(lambda x: x.replace('Varies with device', '9999'))
a=gps.Android_Ver.apply(lambda x: func(x))
gps.Android_Ver[~a].value_counts()
#Considering the lowest value of Android version

gps['Android_Ver']=gps.Android_Ver.apply(lambda x: x.replace('nan', '9999'))
gps['Android_Ver']=gps.Android_Ver.apply(lambda x: x.replace('4.4W', '4.4'))
gps['Android_Ver']=gps.Android_Ver.apply(lambda x: x.replace('5.0 - 8.0', '5.0'))
gps['Android_Ver']=gps.Android_Ver.apply(lambda x: x.replace('4.0.3 - 7.1.1', '4.0.3'))
gps['Android_Ver']=gps.Android_Ver.apply(lambda x: x.replace('5.0 - 6.0', '5.0'))
gps['Android_Ver']=gps.Android_Ver.apply(lambda x: x.replace('5.0 - 7.1.1', '5.0'))
gps['Android_Ver']=gps.Android_Ver.apply(lambda x: x.replace('4.1 - 7.1.1', '4.1'))
gps['Android_Ver']=gps.Android_Ver.apply(lambda x: x.replace('2.2 - 7.1.1', '2.2'))
gps['Android_Ver']=gps.Android_Ver.apply(lambda x: x.replace('7.0 - 7.1.1', '7.0'))
#Considering all 'Varies with device' as 9999

gps['Current_Ver'] = gps.Android_Ver.apply(lambda x: x.replace('Varies with device', '9999'))
#A lot of null values were found in the dataset. These null values were replaced by mean of all the reatings.


plt.figure(figsize=(15,5))
sns.countplot(x='Rating',data=gps)
plt.figure(figsize=(5,5))
plt.tight_layout()
sns.boxplot(x='Type', y='Rating', data=gps)

gps.drop([9148], inplace=True)
#Demographics of columns with null values

sns.heatmap(gps.isnull(),yticklabels=False, cbar=False, cmap= 'viridis')
gps['Rating'].fillna(value=round(gps['Rating'].mean(),1), inplace= True)
#Filling the null values with mean of the column

sns.heatmap(gps.isnull(),yticklabels=False, cbar=False, cmap= 'viridis')
sns.heatmap(gps.corr())
gps['Reviews']=gps['Reviews'].astype('float')
sns.jointplot(x='Rating', y='Reviews', kind='hex', data=gps)
gps.groupby('Category')['Category'].agg({'Occ':len}).sort_values('Occ', ascending=False)
sns.countplot(x='Category', data=gps)
plt.xticks(rotation=90)
g=pd.DataFrame(gps.groupby('Content_Rating')['Content_Rating'].count())
cont = list(gps.Content_Rating.unique())
cont
g

plt.tight_layout()
plt.figure(figsize=(5,5))
plt.pie(g, labels=cont, startangle = -90, autopct = '%.2f%%')
plt.figure(figsize=(15,15))
sns.barplot(x='Rating', y='Category', data=gps)


gps['Installs'] = gps['Installs'].astype(int)
#Categories with hishest rating

rating5=pd.DataFrame(gps[gps['Rating']==5.0]).reset_index()
rating5.drop(columns=['index'], inplace=True)
rating5.head()
rating5.nlargest(5,'Reviews')
rating5.nlargest(5,'Installs')
n=gps.Installs
num=[]

for i in n:
    if i <=100:
        num.append('A')
    elif 101<i<100000:
        num.append('B')
    elif 100001<i<100000000:
        num.append('C')
    else:
        num.append('Highest')

        
gps['Group'] = num
installs=pd.DataFrame(gps.groupby('Group')['Group'].agg({'Count':len}).sort_values('Group', ascending=True))
sns.countplot(x='Group', data=gps, palette='husl', order=gps['Group'].value_counts().index)
plt.title("Grouping of Installed Apps")
plt.figure(figsize=(5,5))
sns.barplot(x='Group', y='Rating', data=gps, hue='Type', palette='husl')
plt.legend(loc=4)
plt.title('Number of applications installed and their ratings with respect to the type of application')
plt.figure(figsize=(5,5))
sns.barplot(x='Group', y='Reviews', data=gps, hue='Type', palette='husl')
plt.legend(loc=0)
plt.title('Group of installed applications with respect to Reviews and type')
gps['Reviews']=gps['Reviews'].astype(int)
top5reviews=gps.nlargest(15,'Reviews')
top5reviews = top5reviews.sort_values(by='Reviews', ascending=False).drop_duplicates('App')
top5reviews.plot(x='App',y='Reviews', kind='bar')
plt.xlabel('Applications')
plt.ylabel('Reviews')
plt.title('Top 5 Applications with highest Reviews')

plt.figure(figsize=(5,5))
ax=sns.barplot(x='Installs', y='Category', data=gps)
plt.xticks(rotation=90)
plt.figure(figsize=(5,15))
sns.barplot(x='Size', y='Category', data=gps)
cat_size=gps[(gps['Category']=='COMMUNICATION') & (gps['Size']<=50000000)]
large_cat_size=cat_size.nlargest(10, 'Installs')
large_cat_size=large_cat_size.sort_values('Installs', ascending = False).drop_duplicates('App')
large_cat_size.plot(x='App',y='Installs', kind='bar')
plt.xlabel('Applications')
plt.ylabel('Installs')
plt.title('Top 5 Communication Applications with highest Installs')
plt.xticks(rotation=75)
cat_soc=gps[(gps['Category']=='SOCIAL')&(gps['Size']<50000000)]
large_cat_soc=cat_soc.nlargest(6, 'Installs')
large_cat_soc=large_cat_soc.sort_values(by='Installs',ascending=False).drop_duplicates('App')
large_cat_soc.plot(x='App',y='Installs', kind='bar')
plt.xlabel('Applications')
plt.ylabel('Installs')
plt.title('Top 5 Social Applications with highest Installs')
plt.xticks(rotation=75)

cat_vp=gps[(gps['Category']=='VIDEO_PLAYERS')&(gps['Size']<50000000)]
large_cat_vp=cat_vp.nlargest(6, 'Installs')
large_cat_vp=large_cat_vp.sort_values(by='Installs',ascending=False).drop_duplicates('App')
large_cat_vp.plot(x='App',y='Installs', kind='bar')
plt.xlabel('Applications')
plt.ylabel('Installs')
plt.title('Top 5 Video Player Applications with highest Installs')
plt.xticks(rotation=75)

