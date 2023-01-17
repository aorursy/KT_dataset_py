import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

import os
print(os.listdir("../input"))
apps = pd.read_csv("../input/googleplaystore.csv")
apps.info()
apps.head()
apps.Rating.value_counts()
apps[apps.Rating == 19.0]
apps = apps.drop(apps.index[10472])
apps.iloc[10471:10475]
apps = apps.dropna()
apps.Category.nunique()
apps.Category.unique()
category_list = list(apps.Category.unique())
ratings = []

for category in category_list:
    x = apps[apps.Category == category]
    rating_rate = x.Rating.sum()/len(x)
    ratings.append(rating_rate)
data = pd.DataFrame({'Category':category_list, 'Rating':ratings})
new_index = (data['Rating'].sort_values(ascending=False)).index.values
sorted_data = data.reindex(new_index)

sorted_data
plt.figure(figsize=(25,15))
sns.barplot(x=sorted_data.Category, y=sorted_data.Rating)

plt.xticks(rotation = 45)
plt.xlabel('Application Category')
plt.ylabel('Ratings')
plt.title('Average Ratings by Category')
plt.show()
apps["Content Rating"].unique()
# list of categories
cat_list = list(apps.Category.unique())

# content rating lists
everyone = []
teen = []
everyone_10 = []
mature_17 = []
adults_only_18 = []
unrated = []

# the function which fills category's (temp) content rating counts into lists
def insert_counts(everyone, teen, everyone_10, mature_17, adults_only_18, unrated, temp):
    
    # everyone
    try:
        everyone.append(temp.groupby('Content Rating').size()['Everyone'])
    except:
        everyone.append(0)
    
    # teen
    try:
        teen.append(temp.groupby('Content Rating').size()['Teen'])
    except:
        teen.append(0)
    
    # everyone 10+
    try:
        everyone_10.append(temp.groupby('Content Rating').size()['Everyone 10+'])
    except:
        everyone_10.append(0)
        
    # mature 17+
    try:
        mature_17.append(temp.groupby('Content Rating').size()['Mature 17+'])
    except:
        mature_17.append(0)
        
    # adults only 18+
    try:
        adults_only_18.append(temp.groupby('Content Rating').size()['Adults only 18+'])
    except:
        adults_only_18.append(0)
        
    # unrated
    try:
        unrated.append(temp.groupby('Content Rating').size()['Unrated'])
    except:
        unrated.append(0)

# fill lists iteratively via function
for cat in cat_list:
    temp = apps[apps.Category == cat]
    insert_counts(everyone, teen, everyone_10, mature_17, adults_only_18, unrated, temp)
    
f,ax = plt.subplots(figsize = (25,25))
sns.barplot(x=everyone,y=cat_list,color='green',alpha = 0.5,label='Everyone')
sns.barplot(x=teen,y=cat_list,color='blue',alpha = 0.7,label='Teen')
sns.barplot(x=everyone_10,y=cat_list,color='pink',alpha = 0.6,label='Everyone 10+')
sns.barplot(x=mature_17,y=cat_list,color='yellow',alpha = 0.6,label='Mature 17+')
sns.barplot(x=adults_only_18,y=cat_list,color='red',alpha = 0.6,label='Adults Only 18+')
sns.barplot(x=unrated,y=cat_list,color='aqua',alpha = 0.6,label='Unrated')

ax.legend(loc='lower right',frameon = True)
ax.set(xlabel='Percentage of Content Ratings', ylabel='Categories',title = "Percentage of Categories According to Content Ratings ")
plt.figure(figsize = (10,8))
sns.countplot(apps['Content Rating'])
plt.show()
plt.figure(figsize = (15,15))
plt.pie(everyone, labels = cat_list, autopct = '%.1f%%', rotatelabels = True, startangle = -90.0)
plt.title('Distribution of the Everyone Content across Categories')

plt.show()
plt.figure(figsize = (15,15))
plt.pie(mature_17, labels = cat_list, autopct = '%.1f%%', rotatelabels = True)
plt.title('Distribution of the Mature 17+ Content across Categories')
plt.show()
apps2 = apps
apps2['Android Ver'].value_counts()
apps2['Android Ver'][apps2['Android Ver'] == 'Varies with device'] = '4.1 and up'

apps2['android_ver_int'] = apps2['Android Ver'].str[0:1].astype(int)

apps2['android_ver_int'].value_counts()
new_index2 = (apps2['android_ver_int'].sort_values(ascending=False)).index.values
sorted_apps2 = apps2.reindex(new_index2)

sorted_apps2.head(7)
f,ax = plt.subplots(figsize = (5,5))
sns.heatmap(sorted_apps2.corr(), annot = True, fmt = '.2f', ax = ax)
plt.show()
new_df = apps2.groupby('Category').mean()
new_df.sort_values('Rating', inplace = True)

new_df.head()
new_df['Category'] = new_df.index
f,ax2 = plt.subplots(figsize =(20,10))
sns.pointplot(x='Category',y='android_ver_int',data=new_df,color='magenta',alpha=0.8)
sns.pointplot(x='Category',y='Rating',data=new_df,color='aqua',alpha=0.8)
plt.text(x = 18, y = 4.3, s = 'Average Rating', color = 'aqua', fontsize = 17,style = 'italic')
plt.text(x = 18, y = 3.46, s = 'Average Min Supported Android Ver', color='magenta',fontsize = 18,style = 'italic')
plt.xlabel('Categories', fontsize = 15, color = 'black')
plt.ylabel('Ratings', fontsize = 15, color = 'black')
plt.xticks(rotation = 75)
plt.show()
g = sns.jointplot(new_df['android_ver_int'], new_df['Rating'], kind="kde", height=7, color='aqua')
plt.savefig('graph.png')
plt.show()
apps['Reviews_int'] = apps.Reviews.astype(int)
f,ax = plt.subplots(figsize = (5,5))
sns.heatmap(apps.corr(), annot = True, fmt = '.2f', ax = ax)
plt.show()
#Content Rating
#Type
#Rating

plt.figure(figsize = (12,7))
sns.boxplot(x='Content Rating', y='Rating', hue='Type', data=apps, palette='PRGn')
plt.show()
