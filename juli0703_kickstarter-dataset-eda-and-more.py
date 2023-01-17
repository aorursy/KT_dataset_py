import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
kick = pd.read_csv('../input/ks-projects-201801.csv')
kick.shape
kick.head()
plt.figure(figsize=(12,9))
sns.heatmap(kick.isnull(),cbar=False,cmap='viridis',yticklabels=False)
plt.show()
kick.duplicated().value_counts()
kick['main_category'].nunique()
print('The main_category column has ' + str(kick['main_category'].nunique()) + ' unique values.')
print('The category column has ' + str(kick['category'].nunique()) + ' unique values.')
kick['main_category'].value_counts().sort_values(ascending=True).tail(20).plot(kind='barh',
                                         figsize=(12,9),
                                         fontsize=15)
plt.title('Count of Main Categories Kickstarter Projects',fontsize=20)
plt.show()
print(kick['main_category'].value_counts())
kick['category'].value_counts().sort_values(ascending=True).tail(20).plot(kind='barh',
                                                                         figsize=(12,9),
                                                                         fontsize=15)
plt.title('Count of Top Category Kickstarter Projects.',fontsize=20)
plt.show()
print(kick['category'].value_counts().head(20))
kick['currency'].value_counts().plot(kind='bar',
                                    figsize=(12,9))
plt.title('Currency Count\n',fontsize=20)
plt.xticks(fontsize=15)
plt.show()
print('Currency Count:\n')
print(kick['currency'].value_counts())
kick['country'].value_counts().sort_values(ascending=False).plot(kind='bar',
                                                                figsize=(12,9))
plt.title('Number of Kickstarter by Country\n',fontsize=20)
plt.xlabel('Countries',fontsize=15)
plt.xticks(fontsize=15)
plt.show()
print('Country Count:\n')
print(kick['country'].value_counts().sort_values(ascending=False))
kick[kick['currency']=='GBP']['country'].value_counts()
kick['state'].value_counts().plot(kind='bar',
                                 figsize=(12,9),
                                 fontsize=15)
plt.title('Project State\n',fontsize=20)
plt.xticks(fontsize=15)
plt.show()
print(kick['state'].value_counts())
print('\nOnly {}% of Kickstarter projects turned out to be successful.'.format(str(round(len(kick[kick['state']=='successful']) / len(kick['state']),3)*100)))
type(kick['launched'][0])
kick['launched'] = pd.to_datetime(kick['launched'])
type(kick['launched'][0])
kick['launched_month'] = kick['launched'].apply(lambda x: x.month)
kick['launched_year'] = kick['launched'].apply(lambda x: x.year)

kick.groupby('launched_month')['launched_month'].count().plot(kind='bar',
                                                             figsize=(12,9))
plt.title('Kickstarters Launched by the Month\n',fontsize=20)
plt.xlabel('Month Launched',fontsize=15)
plt.show()
print(kick.groupby('launched_month')['launched_month'].count())
kick.groupby('launched_year')['launched_year'].count().plot(kind='bar',
                                                           figsize=(12,9))
plt.title('Kickstarters Launched by the Year\n',fontsize=20)
plt.xlabel('Year Launched',fontsize=15)
plt.xticks(fontsize=15)
plt.show()
print(kick.groupby('launched_year')['launched_year'].count())
kick[kick['launched_year']==1970]
plt.figure(figsize=(12,9))
sns.distplot(kick[kick['pledged']<=1000]['pledged'],kde=False,bins=60)
plt.title('Kickstarters with Under $1000 Dollars in Pledges\n',fontsize=20)
plt.xlabel('Pledged Dollars',fontsize=15)
plt.show()
print('{}% of Kickerstart projects made less than $1000 dollars worth in pledges.'.format(round(len(kick[kick['pledged']<=1000]) / (len(kick))*100),2))
plt.figure(figsize=(12,9))
sns.distplot(kick[kick['backers']<=200]['backers'],kde=False,bins=60)
plt.title('Number of Backers for Kickstarter Projects\n',fontsize=20)
plt.xlabel('Number of Backers',fontsize=15)
plt.show()
print('{}% of Kickstarter Projects got more than 200 backers.'.format(round(len(kick[kick['backers']>=200]) / len(kick) * 100)))
plt.figure(figsize=(10,7))
sns.heatmap(kick.drop(['launched_month','launched_year'],axis=1).corr(),cmap='coolwarm',annot=True)
plt.show()
success_cat = kick[kick['state']=='successful']['category'].value_counts().sort_values(ascending=True).tail(20)
success_cat.plot(kind='barh',
                figsize=(12,9),
                fontsize=15)
plt.title('Most Successful Categories by Count\n',fontsize=20)
plt.show()
notSuccess_cat = kick[kick['state']!='successful']['category'].value_counts().sort_values(ascending=True).tail(20)
notSuccess_cat.plot(kind='barh',
                figsize=(12,9),
                fontsize=15)
plt.title('Least Successful Categories by Count\n',fontsize=20)
plt.show()
fig,ax = plt.subplots(figsize=(12,9))
sns.countplot(kick['main_category'],hue=kick['state'],)
plt.title('State of Kickstarter Projects\n',fontsize=20)
plt.xticks(rotation=90,fontsize=15)
plt.xlabel('Main Category',fontsize=18)
plt.ylabel('Count',fontsize=18)
plt.legend(loc=1)
plt.show()
for i in kick['main_category'].unique():
    '''
    str round [('successful') & ('category')] / (number in category) * 100
    '''
    print(i + ' category success rate: {}%'.format(str(round(len(kick[(kick['state']=='successful') & (kick['main_category']==i)]) / len(kick[kick['main_category']==i])*100))))
highBacker = kick[kick['backers']>=60000]
highPledge = kick[kick['pledged']>=5000000]
highPledge['category'].value_counts().plot(kind='bar',
                                          figsize=(12,9))
plt.title('Category of High Pledgers (over $5M)\n',fontsize=20)
plt.xticks(fontsize=15)
plt.show()
print(highPledge['category'].value_counts())
highBacker['category'].value_counts().plot(kind='bar',
                                          figsize=(12,9))
plt.title('Category of Projects with High Number of Backers (over 60K)\n',fontsize=20)
plt.xticks(fontsize=15)
plt.show()
print(highBacker['category'].value_counts())
print('Project with the most backers: ' + highBacker['name'][187652])
print('Project with the most money pledged: ' + highPledge['name'][157270])
highBacker[highBacker['name']==highBacker['name'][187652]]
highPledge[highPledge['name']==highPledge['name'][157270]]
kickFood = kick[kick['main_category']=='Food']
kickFood['category'].value_counts().plot(kind='bar',fontsize=15,figsize=(12,9))
plt.title('Count of Different Food Categories\n',fontsize=20)
plt.xlabel('Category',fontsize=15)
plt.ylabel('Count',fontsize=15)
plt.show()
print(kickFood['category'].value_counts())
kickFood[kickFood['state']=='successful']['category'].value_counts()[1:].plot(kind='bar',
                                                                             figsize=(12,9))
plt.title('Successful Food Categories (Excluding the "Food" column)\n',fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel('Category',fontsize=15)
plt.ylabel('Count',fontsize=15)
plt.show()
print(kickFood[kickFood['state']=='successful']['category'].value_counts()[1:])
kickFood[kickFood['state']!='successful']['category'].value_counts()[1:].plot(kind='bar',
                                                                             figsize=(12,9))
plt.title('Unsuccessful Food Categories (Excluding the "Food" column)\n',fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel('Category',fontsize=15)
plt.ylabel('Count',fontsize=15)
plt.show()
print(kickFood[kickFood['state']!='successful']['category'].value_counts()[1:])
print('{}% of Kickstarter projects are successful.'.format(round(len(kick[kick['state']=='successful']) / len(kick)*100),3))
print('{}% of Food Kickstarter projects are successful.'.format(round(len(kickFood[kickFood['state']=='successful']) / len(kickFood)*100),3))
fig,ax = plt.subplots(figsize=(12,9))
sns.countplot(kickFood['category'],hue=kickFood['state'])
plt.title('State of Kickstarter Food Projects\n',fontsize=20)
plt.xticks(rotation=90,fontsize=15)
plt.xlabel('Category',fontsize=18)
plt.ylabel('Count',fontsize=18)
plt.legend(loc=1)
plt.show()
for i in kickFood['category'].unique():
    '''
    str round [('successful') & ('category')] / (number in category) * 100
    '''
    print(i + ' category success rate: {}%'.format(str(round(len(kickFood[(kickFood['state']=='successful') & (kickFood['category']==i)]) / len(kickFood[kickFood['category']==i])*100))))