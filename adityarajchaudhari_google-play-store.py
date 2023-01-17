import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

plt.style.use('seaborn-bright')

import seaborn as sbn

%matplotlib inline
apps = pd.read_csv('../input/google-play-store-apps/googleplaystore.csv')

apps.head(2)
rows,columns = apps.shape

print(f'The no.of rows are : {rows}')

print(f'The no.of columns are : {columns}')
apps.info()
apps.App.nunique()
apps.App.duplicated().any()
apps.drop_duplicates(subset='App',keep='first',inplace=True)
apps.shape
apps.isnull().sum()
apps['Rating']=apps['Rating'].fillna(0)
apps.Type.value_counts()
apps['Type']=apps['Type'].replace('0','Free')
apps['Type']=apps['Type'].replace(np.nan,'Free')
apps['Content Rating'].unique()
apps['Content Rating'] = apps['Content Rating'].replace(np.nan,'Everyone')
apps['Current Ver'] =apps['Current Ver'].replace(np.nan,'1.0.0')
apps['Android Ver'] = apps['Android Ver'].replace(np.nan,'1.0 and up')
apps.isna().sum()
unwanted=apps['App']=='Life Made WI-Fi Touchscreen Photo Frame'

apps.drop(index=apps[unwanted].index,inplace=True)

apps.head(2)
apps.info()
apps['Reviews']=apps.Reviews.astype('int64')
apps['Installs']=apps['Installs'].apply(lambda a :a.replace(',',''))

apps['Installs']=apps['Installs'].apply(lambda a :a.replace('+',''))

apps['Installs']=apps['Installs'].astype('int64')
apps['date'] = pd.to_datetime(apps['Last Updated'])

apps['year_of_update'] = apps['date'].dt.year

apps.drop('Last Updated',axis=1,inplace=True)
apps['Price'] = apps['Price'].apply(lambda x : x.replace('$',''))

apps['Price'] = apps['Price'].astype('float64')
apps.info()
apps['Category'].value_counts()
plt.figure(figsize=[21,9])



apps['Category'].value_counts().plot.bar(color=sbn.color_palette('rainbow'))

plt.xlabel('CATEGORY',fontsize=25)

plt.ylabel('COUNT',fontsize=25)

plt.tick_params(labelsize=15)

plt.title('CATEGORY OF APPS',fontsize=35)

plt.show()
category=round(apps.groupby(['Category']).mean('Rating').sort_values('Rating',ascending=0),2).reset_index()

category
plt.figure(figsize=[21,9])



sbn.barplot(x='Category',y='Rating',data=category,palette='Set1')

plt.xlabel('CATEGRORY',fontsize=25)

plt.xticks(rotation=90)

plt.ylabel('RATING',fontsize=25)

plt.tick_params(labelsize=15)

plt.title('CATEGORYWISE RATING',fontsize=35)

plt.show()
plt.figure(figsize=[21,9])

sbn.barplot(x='Category',y='Reviews',data=category.sort_values('Reviews',ascending=0),palette='prism')

plt.xlabel('CATEGRORY',fontsize=25)

plt.xticks(rotation=90)

plt.ylabel('REVIEWS (IN MILLIONS)',fontsize=25)

plt.tick_params(labelsize=15)

plt.title('CATEGORYWISE REVIEWS(IN MILLIONS)',fontsize=35)

plt.show()
plt.figure(figsize=[21,9])

sbn.barplot(x='Category',y='Installs',data=category.sort_values('Installs',ascending=0),palette='prism_r')

plt.xlabel('CATEGRORY',fontsize=25)

plt.xticks(rotation=90)

plt.ylabel('INSTALLS(IN CRORES)',fontsize=25)

plt.tick_params(labelsize=15)

plt.title('CATEGORYWISE INSTALLS(IN CRORES)',fontsize=35)

plt.show()

plt.figure(figsize=[21,9])

sbn.countplot(x='Category',data=apps,hue='Type',palette='Set1',hue_order=['Paid','Free'])

plt.xlabel('CATEGORY',fontsize=25)

plt.xticks(rotation=90)

plt.ylabel('COUNT',fontsize=25)

plt.tick_params(labelsize=15)

plt.title('NO.OF PAID AND FREE APPS',fontsize=35)

plt.show()
plt.figure(figsize=[17,7])

avg=round(apps['Rating'].mean(),2)

plt.hist(apps['Rating'],edgecolor='black',color='#ff3333')

plt.axvline(avg,color='black')

plt.xlabel('RATINGS',fontsize=25)

plt.ylabel('COUNT',fontsize=25)

plt.tick_params(labelsize=15)

plt.title('AVERAGE RATING',fontsize=35)

plt.show()

print(f'Considering Unrated Apps as 0 we found the average rating of applications on playstore valued {avg} which we displayed using axvline as average')
plt.figure(figsize=[17,7])

avg=round(apps['Rating'].median(),2)

plt.hist(apps['Rating'],color='#3333ff',edgecolor='black')

plt.axvline(avg,color='black')

plt.xlabel('RATINGS',fontsize=25)

plt.ylabel('COUNT',fontsize=25)

plt.tick_params(labelsize=15)

plt.title('AVERAGE RATING',fontsize=35)

plt.show()

print(f'We have considered ratings of unrated apps as 0 but in some cases it may not be true so for this reason plotting another histogram with totally excluding 0 ratings with average rating valued as {avg} which we displayed using axvline')
reviews = apps.nlargest(10,'Reviews')

reviews
plt.figure(figsize=[19,9])

sbn.barplot(x='Reviews',y='App',data=reviews)

plt.xlabel('REVIEWS (IN CRORES)',fontsize=25)

plt.ylabel('APP',fontsize=25)

plt.title('TOP REVIEWED APPS',fontsize=35)

plt.tick_params(labelsize=15)

plt.show()
popular = ((apps['Reviews']>50000000) & (apps['Rating']>4.0))

popular_apps = apps.loc[popular,['App','Rating','Reviews']].sort_values('Reviews',ascending=0)

popular_apps
plt.figure(figsize=[17,7])

sbn.barplot(x='Reviews',y='App',data=popular_apps,palette='Set2')

plt.xlabel('REVIEWS',fontsize=25)

plt.ylabel('APPS',fontsize=25)

plt.tick_params(labelsize=15)

plt.title('MOST POPULAR APPS',fontsize=35)

plt.show()
apps['Type'].value_counts()
lab=['Free','Paid']
plt.figure(figsize=[7,7])

apps['Type'].value_counts().plot.pie(labels=lab,colors=['#8a8a5c','#d6d6c2'],autopct='%1.2f%%',startangle=90,wedgeprops={'edgecolor':'#ff471a'},explode=[0.0,0.05],shadow=True,textprops={'color':'#ff471a'})

plt.title('TYPE OF APP',fontsize=35)

plt.show()
costly_apps = apps.nlargest(10,'Price')

costly_apps
plt.figure(figsize=[17,9])

sbn.barplot(x='Price',y='App',data=costly_apps,palette='Set1_r')

plt.xlabel('PRICE (IN DOLLARS)',fontsize=25)

plt.ylabel('APP',fontsize=25)

plt.title('MOST EXPENSIVE APPS',fontsize=35)

plt.tick_params(labelsize=15)

plt.show()
popular_paid = ((apps['Type']=='Paid') & (apps['Rating']>4.2) & (apps['Reviews']>100000))

popular_paid_apps = apps.loc[popular_paid,['App','Category','Rating','Reviews','Installs','Type','Price']].sort_values('Reviews',ascending=False)

popular_paid_apps
plt.figure(figsize=[15,7])

sbn.barplot(x='Reviews',y='App',data=popular_paid_apps,palette='rocket_r')

plt.xlabel('REVIEWS (IN MILLIONS)',fontsize=25)

plt.ylabel('APP',fontsize=25)

plt.tick_params(labelsize=15)

plt.title('POPULAR PAID APPS',fontsize=35)

plt.show()
content_rtg=apps['Content Rating'].value_counts()

content_rtg
lab=['Everyone','Teen','Mature 17+','Everyone 10+']
plt.figure(figsize=[7,7])

plt.pie(content_rtg[:4],wedgeprops={'edgecolor':'red'},textprops={'color':'#cc0000'},startangle=90,shadow=True,labels=lab,labeldistance=1.15,autopct='%1.1f%%',pctdistance=0.9,colors=sbn.color_palette('rocket'))

plt.title('CONTENT RATING',fontsize=35)

plt.show()
plt.rcParams['figure.figsize']=19,9

content_rating=apps.groupby(['Content Rating','Category']).size().reset_index().pivot(columns='Content Rating',index='Category',values=0)

content_rating.plot(kind='bar',stacked=True,color=sbn.color_palette('Oranges_r'))

plt.xlabel('CATEGORY',fontsize=25)

plt.ylabel('COUNT',fontsize=25)

plt.tick_params(labelsize=15)

plt.title('CATEGORY-WISE DISTRIBUTION OF CONTENT RATING')

plt.show()
plt.figure(figsize=[15,7])

sbn.countplot(x='Content Rating',data=apps,hue='Type')

plt.xlabel('CONTENT RATING',fontsize=25)

plt.ylabel('COUNT',fontsize=25)

plt.tick_params(labelsize=15)

plt.title('CONTENT RATING BASED ON TYPE OF APP',fontsize=35)

plt.show()
genres = apps['Genres'].value_counts()

genres
plt.figure(figsize=[15,7])

genres[:10].plot.bar(color=sbn.color_palette('Pastel1'))

plt.xlabel('GENRES',fontsize=25)

plt.ylabel('COUNT',fontsize=25)

plt.tick_params(labelsize=15)

plt.title('TOP 10 GENRES',fontsize=35)

plt.show()
apps['Android Ver'].nunique()
apps['Android Ver'].value_counts().plot.bar(color=sbn.color_palette('BuPu_r'))

plt.xlabel('ANDROID VERSIONS',fontsize=25)

plt.ylabel('COUNT',fontsize=25)

plt.tick_params(labelsize=15)

plt.title('NO OF APPS COMPATIBLE WITH ANDROID VERSIONS',fontsize=35)

plt.show()
apps['year_of_update'].value_counts()
apps.year_of_update.value_counts()[::-1].plot.barh(color=sbn.color_palette('cubehelix'))

plt.xlabel('COUNT OF APPS',fontsize=25)

plt.ylabel('YEARS',fontsize=25)

plt.tick_params(labelsize=15)

plt.title('APPS UPDATED PER YEAR',fontsize=35)

plt.show()