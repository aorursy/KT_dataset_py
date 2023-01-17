# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# Any results you write to the current directory are saved as output.
def disable_pandas_warnings():
    import warnings
    warnings.resetwarnings()  # Maybe somebody else is messing with the warnings system?
    warnings.filterwarnings('ignore')  # Ignore everything
    # ignore everything does not work: ignore specific messages, using regex
    warnings.filterwarnings('ignore', '.*A value is trying to be set on a copy of a slice from a DataFrame.*')
    warnings.filterwarnings('ignore', '.*indexing past lexsort depth may impact performance*')
data=pd.read_csv('../input/googleplaystore.csv')
data.head()
data.info()
##NaN values is filled 0
data.Rating.fillna('0',inplace=True)
assert data.Rating.notnull().all() ## No errror 
data.info()
data.Reviews[data.Reviews=="3.0M"] ## wrong value '3.0M'
data.Category.unique() ## wrong category '1.9'
data[data.Category=="1.9"]
data.drop([10472,10472],inplace=True) ## dropped wrong tupple
data.Rating=data.Rating.astype('float') # Rating converted to float
data.Reviews=data.Reviews.astype('float') # Reviews converted to float
category=list(data.Category.unique()) ## unique vales of category
rate_rating=[]
for i in category:
                x=data[data.Category==i]
                avg=sum(x.Rating)/len(x)
                rate_rating.append(avg)           ## average rating append in the rate_rating
data2=pd.DataFrame({'Category':category,'Avarage_Rating':rate_rating})
data2.sort_values('Avarage_Rating',inplace=True)
plt.figure(figsize=(15,10))
sns.barplot(x=data2['Category'], y=data2['Avarage_Rating'])
plt.xticks(rotation= 90)
plt.xlabel('Categories')
plt.ylabel('Avarage Rating')
plt.title("Average rating by category")

data2
rate_reviews=[]
for i in category:
                x=data[data.Category==i]
                avg=sum(x.Reviews)/len(x)
                rate_reviews.append(avg)           ## average reviews append in the rate_reviews
data3=pd.DataFrame({'Category':category,'Avarage_Reviews':rate_reviews})
data3.sort_values('Avarage_Reviews',inplace=True,ascending=False)
plt.figure(figsize=(15,10))
sns.barplot(x=data3['Category'], y=data3['Avarage_Reviews'])
plt.xticks(rotation= 90)
plt.xlabel('Categories')
plt.ylabel('Avarage_Reviews')
plt.title("Average reviews by category")
data3
#normalization 
data2['Avarage_Rating']=data2['Avarage_Rating']/max(data2['Avarage_Rating']) #norm for avarage rating
data3['Avarage_Reviews']= data3['Avarage_Reviews']/max(data3['Avarage_Reviews']) #norm for avarage reviews
data4=pd.concat([data2,data3['Avarage_Reviews']],axis=1) ## horizantal concat 
data4.sort_values('Avarage_Rating',inplace=True)
# visualization
f,ax1 = plt.subplots(figsize =(20,10))
sns.pointplot(x='Category',y='Avarage_Rating',data=data2,color='blue',alpha=0.8)
sns.pointplot(x='Category',y='Avarage_Reviews',data=data3,color='red',alpha=0.8)
plt.text(20,0.6,'Avarage Reviews',color='red',fontsize = 17,style = 'italic')
plt.text(20,0.55,'Avarage Rating',color='blue',fontsize = 18,style = 'italic')
plt.xlabel('Category',fontsize = 15,color='blue')
plt.ylabel('Values',fontsize = 15,color='blue')
plt.xticks(rotation= 90)
plt.title('Avarage_Rating VS  Avarage_Reviews',fontsize = 20,color='green')
plt.grid()
data4.head()
disable_pandas_warnings()
from scipy import stats
g = sns.jointplot(data4.Avarage_Rating, data4.Avarage_Reviews, kind="kde", size=7)
g = g.annotate(stats.pearsonr) #pearsonr is positive 
plt.savefig('graph.png')
plt.show()
disable_pandas_warnings()
g = sns.jointplot("Avarage_Rating", "Avarage_Reviews", data=data4,size=5, ratio=3, color="r")
data["Content Rating"].value_counts(dropna=False)
labels=data["Content Rating"].value_counts().index ## Content Rating values names
sizes=data["Content Rating"].value_counts().values  ## each Content Rating counts
explode=[0,0,0,0,0,0] 
colors = ['green','blue','red','yellow','purple','brown']
print(labels)
print(sizes)
plt.figure(figsize = (8,8))
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%')
plt.title('Content Rating of Play Store Apps',color = 'red',fontsize = 15)
data4.head()
disable_pandas_warnings()
sns.lmplot(x="Avarage_Rating", y="Avarage_Reviews",data=data4)
plt.show()
pal = sns.cubehelix_palette(2, rot=-.5, dark=.3)
sns.violinplot(data=data4, palette=pal, inner="points") 
plt.show() 
f,ax = plt.subplots(figsize=(5, 5))
sns.heatmap(data4.corr(), annot=True, linewidths=0.5,linecolor="red", fmt= '.1f',ax=ax)
plt.show() 
data.Type.unique()
data.Type.dropna(inplace=True)
data.Type.unique()
##  rating with free or paid  by Content Rating
f,ax = plt.subplots(figsize=(9, 7))
sns.boxplot(x="Type",y="Rating",hue='Content Rating',data=data)
plt.show()
sns.swarmplot(x="Type", y="Rating",hue="Content Rating", data=data)
plt.show()
sns.pairplot(data4)
plt.show()
sns.countplot(data.Type)
plt.show()
liste=data.Genres.value_counts()  ## Genres value counts
print(liste)
plt.figure(figsize=(10,10))
sns.barplot(x=liste[:30].index,y=liste[:30].values)
plt.ylabel('Counts')
plt.xlabel('Genres')
plt.xticks(rotation= 90)
plt.title('Counts of Genres',color = 'red',fontsize=15)