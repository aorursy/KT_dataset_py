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

# close warning
import warnings
warnings.filterwarnings("ignore")

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
gps = pd.read_csv('../input/googleplaystore.csv')
gps.info()# Display the content of data
# shape gives number of rows and columns in a tuple
gps.shape
gps.columns
gps.columns = [each.replace(" ","_") if(len(each.split())>1) else each for each in gps.columns]
print(gps.columns)
gps.columns = [each.lower() for each in gps.columns]
print(gps.columns)
gps.rename(columns={"size":"Size"}, inplace=True)
gps.head(10)
gps.tail()
gps.sample(5)
gps.dtypes
gps[gps.duplicated(keep='first')]
gps.drop_duplicates(subset='app', inplace=True)
gps = gps[gps['installs'] != 'Free']
print('Number of apps in the dataset : ' , len(gps))
# - Installs : Remove + and ,

gps['installs'] = gps['installs'].apply(lambda x: x.replace('+', '') if '+' in str(x) else x)
gps['installs'] = gps['installs'].apply(lambda x: x.replace(',', '') if ',' in str(x) else x)
gps['installs'] = gps['installs'].apply(lambda x: int(x))
gps.head()
# - Size : Remove 'M', Replace 'k' and divide by 10^-3

gps['Size'] = gps['Size'].apply(lambda x: str(x).replace('Varies with device', 'NaN') if 'Varies with device' in str(x) else x)

gps['Size'] = gps['Size'].apply(lambda x: str(x).replace('M', '') if 'M' in str(x) else x)
gps['Size'] = gps['Size'].apply(lambda x: str(x).replace(',', '') if 'M' in str(x) else x)
gps['Size'] = gps['Size'].apply(lambda x: float(str(x).replace('k', '')) / 1000 if 'k' in str(x) else x)

#gps.Size = gps.Size.astype(float)
#OR
gps['Size'] = gps['Size'].apply(lambda x: float(x))
gps['installs'] = gps['installs'].apply(lambda x: float(x))

gps['price'] = gps['price'].apply(lambda x: str(x).replace('$', '') if '$' in str(x) else str(x))
gps['price'] = gps['price'].apply(lambda x: float(x))

gps['reviews'] = gps['reviews'].apply(lambda x: int(x))
gps.head()
gps.info()
#gps.rating.value_counts(dropna =False) Nan 1463
gps.rating.fillna(0.0,inplace=True)
# Display positive and negative correlation between columns
gps.corr()
#sorts all correlations with ascending sort.
gps.corr().unstack().sort_values().drop_duplicates()
#correlation map
plt.subplots(figsize=(10,10))
sns.heatmap(gps.corr(), annot=True, linewidths=1,linecolor="green", fmt=".2f")
plt.show()
#figsize - image size
#data.corr() - Display positive and negative correlation between columns
#annot=True -shows correlation rates
#linewidths - determines the thickness of the lines in between
#cmap - determines the color tones we will use
#fmt - determines precision(Number of digits after 0)
#if the correlation between the two columns is close to 1 or 1, the correlation between the two columns has a positive ratio.
#if the correlation between the two columns is close to -1 or -1, the correlation between the two columns has a negative ratio.
#If it is close to 0 or 0 there is no relationship between them.
gps.category.value_counts()
#Installs for each Category

category_list = list(gps.category.unique())
category_installs_ratio = []
for i in category_list:
    x = gps[gps.category == i]
    category_installs_rate = sum(x.installs)/len(x)
    category_installs_ratio.append(category_installs_rate)
data = pd.DataFrame({'category_list':category_list,'category_installs_ratio':category_installs_ratio})
data.sort_values('category_installs_ratio',ascending=False,inplace=True)

# visualization
plt.figure(figsize=(15,10))
sns.barplot(x=data.category_list, y=category_installs_ratio)
plt.xticks(rotation=90)
plt.xlabel('Category')
plt.ylabel('Rating')
plt.title('Rating Given Category')
# Most common 15 Category Name

#Counter(gps.category)
categoryName = Counter(gps.category)
mostCommonCatName = categoryName.most_common(20)
x,y = zip(*mostCommonCatName)
x,y = list(x),list(y)

# visualization
plt.figure(figsize=(15,20))
sns.barplot(x=x ,y=y , palette = sns.cubehelix_palette(len(x)))
plt.xticks(rotation=90)
plt.xlabel('Category')
plt.ylabel('Frequency')
plt.title('Most common 15 Category Name')
#Reviews for each Category

category_list = list(gps.category.unique())
category_reviews_ratio = []
for i in category_list:
    x = gps[gps.category == i]
    category_reviews_rate = sum(x.reviews)/len(x)
    category_reviews_ratio.append(category_reviews_rate)
data2 = pd.DataFrame({'category_list':category_list,'category_reviews_ratio':category_reviews_ratio})
data2.sort_values('category_reviews_ratio',ascending=True,inplace=True)

# visualization
plt.figure(figsize=(15,10))
sns.barplot(x=data2.category_list, y=category_reviews_ratio)
plt.xticks(rotation=90)
plt.xlabel('Category')
plt.ylabel('Reviews')
plt.title('Reviews Given Category')
# category rating ratio vs category reviews ratio of each Category
data['category_installs_ratio'] = data['category_installs_ratio']/max(data['category_installs_ratio'])
data2['category_reviews_ratio'] = data2['category_reviews_ratio']/max(data2['category_reviews_ratio'])
cnc_data = pd.concat([data,data2['category_reviews_ratio']],axis=1)
cnc_data.sort_values('category_installs_ratio',inplace=True)

# visualize
f,ax1 = plt.subplots(figsize =(20,20))
sns.pointplot(x='category_list',y='category_installs_ratio',data=cnc_data,color='lime',alpha=0.8)
sns.pointplot(x='category_list',y='category_reviews_ratio',data=cnc_data,color='red',alpha=0.8)
plt.text(40,0.6,'category installs ratio',color='lime',fontsize = 17,style = 'italic')
plt.text(40,0.55,'category reviews ratio',color='red',fontsize = 18,style = 'italic')
plt.xticks(rotation=90)
plt.xlabel('Category',fontsize = 15,color='blue')
plt.ylabel('Values',fontsize = 15,color='blue')
plt.title('category installs ratio  VS  category reviews ratio',fontsize = 20,color='blue')
plt.grid()
# joint kernel density
# pearsonr= if it is 1, there is positive correlation and if it is, -1 there is negative correlation.
# If it is zero, there is no correlation between variables

from scipy import stats
g = sns.jointplot(cnc_data.category_installs_ratio,cnc_data.category_reviews_ratio,kind="kde", height=5)
g = g.annotate(stats.pearsonr)
plt.savefig('graph.png')
plt.show()
#you can change parameters of joint plot
# kind : { “scatter” | “reg” | “resid” | “kde” | “hex” }
# Different usage of parameters but same plot with previous one
g = sns.jointplot('category_installs_ratio', 'category_reviews_ratio', data=cnc_data,height=7, ratio=3, color="r")
gps.content_rating.value_counts()
gps = gps[gps['content_rating'] != 'Adults only 18+']
gps = gps[gps['content_rating'] != 'Unrated']
#content rates according in category data 

labels = gps.content_rating.value_counts().index
colors = ['red','green','blue','pink']
explode = [0.2,0.2,0.2,0.2]
sizes = gps.content_rating.value_counts().values

# visualization
plt.figure(figsize = (7,7))
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%')
plt.title('Content Rating',color = 'blue',fontsize = 15)
# Visualization of category_installs_ratio rate vs category_reviews_ratio of each state with different style of seaborn code
# lmplot 
# Show the results of a linear regression within each dataset
sns.lmplot(x='category_installs_ratio',y='category_reviews_ratio',data=cnc_data)
plt.show()
# Visualization of category_installs_ratio rate vs category_reviews_ratio of each state with different style of seaborn code
sns.kdeplot(cnc_data.category_installs_ratio,cnc_data.category_reviews_ratio,shade=True,cut=3)
plt.show()
# Show each distribution with both violins and points
# Use cubehelix to get a custom sequential palette
pal = sns.cubehelix_palette(2, rot=-.5, dark=.3)
sns.violinplot(data=cnc_data, palette=pal, inner="points")
plt.show()
plt.figure(figsize=(10,10))
sns.boxplot(x="type", y="rating", hue="content_rating", data=gps, palette="PRGn")
plt.show()
plt.figure(figsize=(10,10))
sns.swarmplot(x="type", y="rating", hue="content_rating", data=gps.head(3000), palette="PRGn")
plt.show()
# pair plot
sns.pairplot(cnc_data)
plt.show()
#frequency of free and paid

plt.figure(figsize=(10,7))
sns.countplot(gps.type)
plt.title("Free-Paid",color = 'blue',fontsize=15)