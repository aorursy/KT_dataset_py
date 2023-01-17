import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import re

import datetime

import squarify

import plotly

# plotly standard imports

import plotly.graph_objs as go

import chart_studio.plotly as py



# Cufflinks wrapper on plotly

import cufflinks as cf



# Options for pandas

#pd.options.display.max_columns = 30



# Display all cell outputs

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = 'all'



from plotly.offline import iplot, init_notebook_mode, plot

cf.go_offline()



init_notebook_mode(connected=True)
apps = pd.read_csv('../input/google-play-store-apps/googleplaystore.csv')

apps.head(2)
#lets see the total Rows and columns

apps.shape
apps.describe()
apps.info()
apps.isnull().sum()
# we can also see the null values by this

sns.heatmap(apps.isnull(),yticklabels=False, cbar=False, cmap= 'viridis')
# Checking For the Outliers



sns.boxplot(y=apps['Rating'])
#->Calculating q75 and q25



q75, q25 = np.nanpercentile(apps['Rating'], [75 ,25])
print("25% qunatile: " + str(q25),"75% quantile: " + str(q75))
#Inter-Quartile Range



iqr = q75-q25

print("IQR:" + str(iqr))



#We now create a benchmark using InterQuartile Range for Outlier Treatment.



bench = q75 + (iqr*1.5)

print("Benchmark: "+ str(bench))



#We use loc to identify the position of the outlier cell and replace it with our capping value.



apps.loc[apps['Rating'] > 5.25, 'Rating'] = np.nan
sns.boxplot(y=apps['Rating'])
sns.set_color_codes()

apps['Rating'].hist()
rat = apps[apps['Rating']>4]['App'].count()



print("Total number of Apps having ratings above 4 is: "+ str(rat))
# filling Numerical data by interpolation.

apps = apps.interpolate()
# filling categorical data with mode

apps.fillna(apps['Type'].mode().values[0],inplace=True)
apps.fillna(apps['Current Ver'].mode().values[0],inplace=True)

apps.fillna(apps['Android Ver'].mode().values[0],inplace=True)
apps.isna().sum()
sns.heatmap(apps.isnull(),yticklabels=False, cbar=False, cmap= 'viridis')
#droping the duplicate rows

apps = apps.drop_duplicates()

apps.shape
apps[apps['Price']=='Everyone']
# Fixing the Bad data

apps['Reviews'] = apps['Reviews'].replace('3.0M',3000000)

apps['Price'] = apps['Price'].replace('Everyone',0)

apps['Type'] = apps['Type'].replace('0',np.nan)

apps['Type'] = apps['Type'].interpolate()
#converting Price,Installs,Reviews into numeric

apps['Reviews'] = pd.to_numeric(apps['Reviews'])



apps['Size'] = apps['Size'].apply(lambda x: str(x).replace('+','') if '+' in str(x) else str(x))

apps['Size'] = apps['Size'].apply(lambda x: str(x).replace(',','') if ',' in str(x) else str(x))



apps['Installs'] = apps['Installs'].apply(lambda x: str(x).replace(',','') if ',' in str(x) else str(x))

apps['Installs'] = apps['Installs'].apply(lambda x: str(x).replace('+','') if '+' in str(x) else str(x))



apps['Price'] = apps['Price'].apply(lambda x: str(x).replace('$','') if '$' in str(x) else str(x))

apps['Price'] = apps['Price'].apply(lambda x: float(x))
#apps[apps['Installs']=='Free'].head(1)



apps['Installs'] = apps['Installs'].replace('Free',0)



apps['Installs'] = pd.to_numeric(apps['Installs'])
apps.info()
# Feature Transformation

apps2 = apps.copy()

apps2.skew()
#-> Decresing skewness of the variables

apps2['Reviews'] = np.sqrt(np.sqrt(np.sqrt(apps2['Reviews'])));

apps2['Installs'] = np.sqrt(np.sqrt(np.sqrt(np.sqrt(apps2['Installs']))));

apps2['Price'] = np.sqrt(np.sqrt(np.sqrt(np.sqrt(np.sqrt(apps2['Price'])))));

apps2.skew()



## We can see we have reduced the skewness to lowest level which is pretty good for Machine Learning Models
apps['Content Rating'].value_counts()
# Fixing the Similar values Issue

apps['Content Rating'] = apps['Content Rating'].replace('Everyone 10+','Everyone');

apps['Content Rating'] = apps['Content Rating'].replace('Adults only 18+','Mature 17+');
install_reviews = apps2[['Installs','Reviews']]

install_reviews2 = apps[['Installs','Reviews']]
install_reviews.head()

install_reviews2.head()
install_reviews.corr()
install_reviews2.corr()
plt.figure(figsize=(16,5))

plt.subplot(121)

sns.regplot(data = install_reviews2,x='Installs',y='Reviews')

plt.title(" Before Performing Feature Transformation (Correlation: 0.63)")



plt.subplot(122)

sns.regplot(data = install_reviews,x='Installs',y='Reviews');

plt.title(" After Performing Feature Transformation (Correlation: 0.94)");
top10_cateogries = apps.groupby(apps['Category'])['Category'].count().sort_values(ascending=False).head(10)
top10_cateogries
plt.figure(figsize=(20,10))

plt.subplot(121)

top10_cateogries.plot.pie(autopct='%1.f%%',

                                                             wedgeprops={'linewidth':10,'edgecolor':'white'},

                                                             shadow=True,

                                                             fontsize=20)

plt.title("% Wise Distribution of Categories")



plt.subplot(122)

top10_cateogries.plot(marker='o',markersize=10,color='red')

top10_cateogries.plot(kind='bar',color='green')

plt.title("Distribution of Categories");
Category_counts = apps['Category'].value_counts().head(15)

Category_counts.head()
plt.figure(figsize=(16,9))

squarify.plot(sizes=Category_counts.values,label=Category_counts.index,value=Category_counts.values,color=["#FF6138","#FFFF9D","#BEEB9F", "#79BD8F","#684656","#E7EFF3"], alpha=0.6)

plt.title('Distribution of Investors and Investments Done');
plt.figure(figsize=(12,6))

sns.countplot(data=apps,x='Category')

plt.ylabel("Number of Apps")

plt.title("Number of Apps per Category", size=20)

plt.xticks(rotation=90);
top10_Apps_Higest_Reviews = apps[['App','Reviews']].sort_values(by='Reviews',ascending=False).drop_duplicates('App').reset_index().drop('index',axis=1).head(10)

top10_Apps_Higest_Reviews['Reviews'] =round(top10_Apps_Higest_Reviews['Reviews']*100/top10_Apps_Higest_Reviews['Reviews'].sum())

top10_Apps_Higest_Reviews['Reviews'] = top10_Apps_Higest_Reviews['Reviews'].astype(int).astype(str)



top10_Apps_Higest_Reviews['Reviews'] = top10_Apps_Higest_Reviews['Reviews'].apply(lambda x: x+'%')

top10_Apps_Higest_Reviews
least5_Apps_lowest_Reviews = apps[['App','Reviews']].sort_values(by='Reviews').drop_duplicates('App').reset_index().drop('index',axis=1).head(5)

least5_Apps_lowest_Reviews
a1 = apps[apps['Size'].str.contains('.M',regex=True)]

#apps2 = apps[apps['Size']=='.M']

top10_Apps_Higest_Size = a1[['App','Size']].sort_values(by='Size',ascending=False).drop_duplicates('Size').reset_index().drop('index',axis=1).head(10)

top10_Apps_Higest_Size
ap = apps2.copy()

ap = ap.dropna()

a2 = ap[ap['Size'].str.contains('.M',regex=True)]



t = a2[['Category','App','Size']]

t = t.groupby('Category')['App','Size'].max().sort_values(by='Size',ascending=False).drop_duplicates().reset_index()

t
top10_most_Installed_Apps = apps[['App','Installs']].sort_values(by='Installs',ascending=False).reset_index().drop('index',axis=1).head(10)

top10_most_Installed_Apps = top10_most_Installed_Apps[top10_most_Installed_Apps['Installs']!='Free']

top10_most_Installed_Apps.head(10)
apps['Type'].value_counts()
Top10_Free_Paid_Apps = pd.crosstab(apps['Category'],apps['Type']).sort_values(by=['Free','Paid'],ascending=False).head(10)

Top10_Free_Paid_Apps = Top10_Free_Paid_Apps.reset_index()

Top10_Free_Paid_Apps
f,ax2 = plt.subplots(figsize =(20,10))

sns.pointplot(data=Top10_Free_Paid_Apps,x='Category',y='Free',color='green',alpha=0.8)

sns.pointplot(data=Top10_Free_Paid_Apps,x='Category',y='Paid',color='blue',alpha=0.8)

plt.text(x = 5, y = 1444.3, s = 'Free Apps', color = 'green', fontsize = 17,style = 'italic')

plt.text(x = 5, y = 1333.46, s = 'Paid Apps', color='blue',fontsize = 18,style = 'italic')

plt.xlabel('Categories', fontsize = 15, color = 'black')

plt.ylabel('Ratings', fontsize = 15, color = 'black')

plt.xticks(rotation = 60);

plt.title("Free Apps Vs Paid Apps wrt differnet Categories");
plt.figure(figsize=(14,6))

Top10_Free_Paid_Apps.plot.bar(subplots=True);
plt.figure(figsize=(14,6))

sns.countplot(data=apps,x='Category',hue='Type')

plt.xticks(rotation=90)

plt.title("Distribution of Category wrt Type");
top10_apps_MaxPrice = apps[['App','Category','Price']].sort_values(by='Price',ascending=False).drop_duplicates('Price').reset_index().drop('index',axis=1).head(10)

top10_apps_MaxPrice
sns.barplot(data=top10_apps_MaxPrice,x='Category',y='Price')

plt.xticks(rotation=90)

plt.title("Categoies wrt total Price");
top10_apps_MaxPrices = apps.groupby(['Category','App'])['Category','App','Price'].sum().sort_values(by='Price',ascending=False).drop_duplicates().head(10)

top10_apps_MaxPrices
plt.figure(figsize=(12,6))

top10_apps_MaxPrices.plot(kind='line',marker='o',markersize=5,color='blue')

plt.xticks(rotation=90)

plt.title("Top10 Apps MaxPrices");
apps['Content Rating'].value_counts()



# Category Rating with Free is wrong so we will remove that one.



apps['Content Rating'] = apps['Content Rating'].replace('Free',np.nan).dropna() 

apps['Content Rating'].value_counts()
apps['Content Rating'].value_counts().plot(kind='bar',color=['red','green','blue','purple','brown','yellow'])

plt.yscale('log')

plt.title(" Distribution of Content Ratings ");
plt.figure(figsize=(16,10))

plt.subplot(121)

apps['Content Rating'].value_counts().plot.pie(autopct='%1.f%%',

                                                             wedgeprops={'linewidth':10,'edgecolor':'white'},

                                                             shadow=True,

                                                             fontsize=20)

plt.title("% wise Contert Rating ")



plt.subplot(122)

apps['Content Rating'].value_counts().plot(color='red',marker='o',markersize='10',linestyle='dashed',linewidth=3)

apps['Content Rating'].value_counts().plot(kind='bar',color='green');

plt.title("Distribution of Contert Rating ");
appCategory_ContentRatings = pd.crosstab(apps['Category'],apps['Content Rating'])

appCategory_ContentRatings.sort_values(['Everyone','Teen'],ascending=False).head(8)
plt.figure(figsize=(14,8))

plt.tight_layout()

sns.heatmap(data=appCategory_ContentRatings,linewidths=0.1,linecolor='black',cmap='bone_r')

plt.title('App_Category Vs Ratings');
plt.figure(figsize=(14,8))

plt.tight_layout()

sns.clustermap(data=appCategory_ContentRatings,linewidths=0.1,linecolor='black',cmap='bone_r')

plt.title('App_Category Vs Ratings');
apps['Android Ver'].value_counts().head(5)
# need to replace Varies with device with 0 then we will make the data balance.

apps['Android Ver'] = apps['Android Ver'].replace("Varies with device",'0')
apps['Android Ver'].value_counts().head(5)
apps['Android Ver'] = apps['Android Ver'].apply(lambda x:x.split('W')[0])

apps = apps[apps['Android Ver']!='Free']

apps['Android Ver']=apps['Android Ver'].apply(lambda x:x[0:3])

apps['Android Ver'] = apps['Android Ver'].astype(float);

apps['Android Ver'] = round(apps['Android Ver'])

apps['Android Ver'].value_counts()
# Converting 0.0 a resonable value

apps['Android Ver'] = apps['Android Ver'].replace(0.0,np.nan)

apps['Android Ver'] = apps['Android Ver'].fillna(method='ffill')

apps['Android Ver'].value_counts()
## Distribution of People using different Android Versions wrt Type of App installed

Android_version_Type = pd.crosstab(apps['Type'],apps['Android Ver'])

Android_version_Type.reset_index()
plt.figure(figsize=(14,8))

plt.tight_layout()

sns.heatmap(data= Android_version_Type,linewidths=0.1,linecolor='black',cmap='tab20')

plt.title('Android Version Vs Types');
apps['Last Updated'].value_counts()
apps['Last Updated'] = pd.to_datetime(apps['Last Updated'])
def get_monthName(dt):

    var = str(dt.month_name()) + " " + str(dt.year)

    return var



apps['Last_Updated_Month'] = apps['Last Updated'].map(get_monthName)
def get_month(dt):

    return dt.month_name()



apps['Month_of_Updation'] = apps['Last Updated'].map(get_month)



def get_year(dt):

    return dt.year



apps['Year_of_Updation'] = apps['Last Updated'].map(get_year)
#Updatings of apps

updates = apps['Year_of_Updation'].value_counts()
plt.figure(figsize=(19,6))

plt.subplot(121)

sns.countplot(data=apps,x='Year_of_Updation')

plt.title("Count of Apps updated in Each Year ");



plt.subplot(122)

updates.plot(kind='line',marker='o',markersize=15);
### Which category has updated there apps most of the time



### top 5 apps updated most of the time
apps['Year_of_Updation'] = apps['Year_of_Updation'].astype(int)

sns.regplot(data=apps,x='Year_of_Updation',y='Rating',scatter_kws={'alpha':0.15})

plt.title("Ratings vs Year_of_Updation");
updating_Apps_year =  pd.crosstab(apps['Category'],apps['Year_of_Updation'])

updating_Apps_year
plt.figure(figsize=(14,8))

plt.tight_layout()

sns.heatmap(data=updating_Apps_year,linewidths=0.1,linecolor='black',cmap='cividis_r')

plt.title('App_Category Vs Year of Updation');
updating_Apps_month =  pd.crosstab(apps['Category'],apps['Month_of_Updation'])

updating_Apps_month
plt.figure(figsize=(14,8))

plt.tight_layout()

sns.heatmap(data=updating_Apps_month,linewidths=0.1,linecolor='black',cmap='Reds')

plt.title('App_Category Vs month of updation');
plt.figure(figsize=(14,8))

plt.tight_layout()

sns.clustermap(data=updating_Apps_month,linewidths=0.1,linecolor='black',cmap='bone_r')

plt.title('App_Category Vs month of Updation');
apps['Current Ver'].value_counts()
current_version = apps[['Category','Current Ver']].groupby('Category').max().sort_values(by='Current Ver',ascending=False).reset_index().head(20)

current_version
total_version = apps[['Category','Current Ver']].groupby('Category').count().sort_values(by='Current Ver',ascending=False).reset_index().head(10)

total_version
plt.figure(figsize=(12,4))

sns.catplot(data=total_version,x='Category',y='Current Ver',kind='bar')

plt.xticks(rotation=90);
# remove/ deselect the unwanted column 'Varies'

apps = apps[apps['Android Ver']!='Varies']

apps['Android Ver'] = apps['Android Ver'].replace('4.4W','4.4')



a = apps[['Category','Android Ver']].copy()

a
#Latest versions of all Categories of Apps

versions =  a.groupby('Category')['Android Ver'].max().sort_values(ascending=False)

versions = versions.reset_index()

versions = versions[(versions['Category']!='1.9') & (versions['Android Ver']!='1.9')]



versions['Android Ver'] = versions['Android Ver'].replace('Free',np.nan)

versions.dropna(inplace=True)
versions.head(10)
versions['Android Ver'] = pd.to_numeric(versions['Android Ver'])



plt.figure(figsize=(15,10))

sns.barplot(data=versions,y='Category',x='Android Ver')

plt.title("Categories working on Latest Android Version");
plt.figure(figsize=(13,5))

sns.countplot(data=versions,y='Android Ver')

plt.title(" Number of users using Android Version 5 or More");
apps = apps[apps['Category']!='1.9']
fg = apps2[['Category','Rating']].copy()

fg['Rating'] = round(fg['Rating'])

fg.groupby(apps['Category'])['Rating'].value_counts()
#Top 10 Cateogy which got highest Raatings

top10_rating_eachApp = fg.groupby(apps['Category'])['Rating'].value_counts()

top10_rating_eachApp = top10_rating_eachApp.unstack().reset_index()

top10_rating_eachApp.columns=['Category','1 star','2 star','3 star','4 star','5 star']



top10_rating_eachApp = top10_rating_eachApp.sort_values(['5 star','4 star'],ascending=False).head(10)

top10_rating_eachApp = top10_rating_eachApp.fillna(0)

top10_rating_eachApp = top10_rating_eachApp.reset_index().set_index('index').reset_index().drop('index',axis=1)

top10_rating_eachApp
#Melting the  top10_rating_each category



melted_top10_rating_eachApp = pd.melt(top10_rating_eachApp, 

                    id_vars=["Category"],

                    var_name=["1 star"])



melted_top10_rating_eachApp.columns=["Category","Ratings","Count"]

melted_top10_rating_eachApp.head()
plt.figure(figsize=(19,6))

sns.barplot(data=melted_top10_rating_eachApp,x='Category',y='Count',hue='Ratings')

plt.xticks(rotation=90)

plt.title("Top 10 Categories having higest Ratings");
Ratingscols = apps[['App','Category','Rating']].copy()
Ratingscols.sort_values('Rating', ascending=False).groupby(['Category'])['Rating'].value_counts()
# Relation between Ratings and Reviews

x = sns.regplot(data=apps,x='Rating',y='Reviews',scatter_kws={'alpha':0.15})

x.set(yscale="log");
apps[['Rating','Reviews']].corr()
top5_install = apps.sort_values(by='Installs',ascending=False).groupby(['Category','App'])['Installs'].value_counts()

top5_install
Rating_ContentRating = pd.crosstab(round(apps['Rating']),apps['Content Rating'])

Rating_ContentRating
#percentage of paid and free apps in Play Store?

plt.figure(figsize=(16,5))

plt.subplot(121)

sns.countplot(data=apps,x='Type')

plt.title("Distribution of Free vs Paid Apps");



plt.subplot(122)

labels = apps.Type.unique() 

sizes = [len(apps[apps.Type == "Free"]), len(apps[apps.Type == "Paid"])]

explode = (0, 0.2)



plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90, colors=["red","orange"]) 

plt.axis('equal');  

plt.title("Percentage of Paid Vs Free Apps");
Content_Type_Year_Updation = pd.crosstab(apps['Content Rating'],[apps['Type'], apps['Year_of_Updation']])

Content_Type_Year_Updation
plt.figure(figsize=(14,8))

plt.tight_layout()

sns.heatmap(data=Content_Type_Year_Updation,linewidths=0.1,linecolor='black',cmap='gist_stern')

plt.title('Rating_Content WRT Rating and Year');
Content_Type_Installs = pd.crosstab(apps['Content Rating'],[apps['Type'], apps['Installs']])

Content_Type_Installs
plt.figure(figsize=(14,9))

plt.tight_layout()

sns.heatmap(data=Content_Type_Installs,linewidths=0.1,linecolor='black',cmap='gist_stern')

plt.title('Rating_Content WRT Rating and Installs');
bins = [0.01,75,150,225,300,375,400]

labels =["0-75 $","75-150 $","150-225 $","225-300 $","300-375 $","375-400 $"]

apps['Price_range'] = pd.cut(apps['Price'],bins=bins, labels=labels)
apps.head(2)
# Filling the Empty Categorical values

apps['Price_range'] = apps['Price_range'].cat.add_categories('Free')
apps['Price_range'].fillna('Free', inplace =True) 
apps.head(2)
Content_Type_Year_PriceRange = pd.crosstab(apps['Content Rating'],[apps['Type'], apps['Price_range']])

Content_Type_Year_PriceRange
plt.figure(figsize=(12,6))

Content_Type_Year_PriceRange.plot(kind='line',marker='o')

plt.xticks(rotation=60)

plt.title('Content Ratings wrt Type and Price');
plt.figure(figsize=(14,9))

plt.tight_layout()

sns.heatmap(data=Content_Type_Year_PriceRange,linewidths=0.1,linecolor='black',cmap='gist_stern')

plt.title('Rating_Content WRT Rating and Installs');
ab = apps[['Category','Type','Content Rating','Android Ver']].copy()
# remove/ deselect the unwanted column 'Varies'

ab = ab[ab['Android Ver']!='Varies']

ab.head()
# creating bins for dividing Android Version



bins = [1.0,3.0,6.0,8.0]

labels =["1.0 - 3.0","3.0 - 6.0","6.0 - 8.0"]

ab['Android Ver'] = pd.cut(ab['Android Ver'],bins=bins, labels=labels);
ab.tail()
Content_Type_Year_AndroidVersion = pd.crosstab(ab['Content Rating'],[ab['Type'], ab['Android Ver']])

Content_Type_Year_AndroidVersion
Content_Type_Year_AndroidVersion.plot()

plt.title("Content type wrt Android Version using")

plt.xticks(rotation=60);
plt.figure(figsize=(14,9))

plt.tight_layout()

sns.heatmap(data=Content_Type_Year_AndroidVersion,linewidths=0.1,linecolor='black',cmap='gist_stern')

plt.title('Rating_Content WRT Rating and Andoid Version');
Content_Type_Year_Month = pd.crosstab(apps['Content Rating'],[apps['Type'], apps['Month_of_Updation']])

Content_Type_Year_Month
plt.figure(figsize=(12,6))

sns.lineplot(data=apps,x='Month_of_Updation',y='Reviews',hue='Type',marker='o',markersize=10)

plt.xticks(rotation=60);
plt.figure(figsize=(20,5))

fig = sns.countplot(x=apps['Installs'], palette="hls")

fig.set_xticklabels(fig.get_xticklabels(),rotation=90)

plt.title("Distribution of Installations by count");

plt.show(fig);
plt.figure(figsize=(15,3))

fig = sns.barplot(y=apps['Genres'].value_counts().reset_index()[:10]['Genres'], x=apps['Genres'].value_counts().reset_index()[:10]['index'], palette="hls")

fig.set_xticklabels(fig.get_xticklabels(),rotation=90)

plt.title("Genres wrt Different Categories");
g = apps.sort_values(by='Reviews',ascending=False);
plt.figure(figsize=(10,5))

fig = sns.barplot(x=g['App'][:10], y=g['Reviews'][:10], palette="hls")

fig.set_xticklabels(fig.get_xticklabels(),rotation=90)

plt.title("Top 5 Apps which got highest ratings")

plt.show(fig);
apps['Size'].value_counts()
temp=apps['Price'].apply(lambda x: True if x>350 else False)

apps[temp].head(3)
apps['Pri_Genres'] = apps['Genres'].apply(lambda x: x.split(';')[0])

apps['Pri_Genres'].head()
apps['Sec_Genres'] = apps['Genres'].apply(lambda x: x.split(';')[-1])

apps['Sec_Genres'].head()
grouped = apps.groupby(['Pri_Genres','Sec_Genres'])

grouped = grouped.size().reset_index()

grouped.head()
twowaytable = pd.crosstab(apps["Pri_Genres"],apps["Sec_Genres"])

twowaytable.head()
twowaytable.plot(kind="barh", figsize=(15,15),stacked=True);

plt.legend(bbox_to_anchor=(1.0,1.0))

plt.title("Pri Genres Vs Sec Genres");
apps['Last Updated'].head()
sns.set(font_scale=1.5, style = "whitegrid") #set the font size and background

plt.figure(figsize=(8,6)) #set the plotsize

sns.distplot(apps['Rating'], hist=False, color="orange", kde_kws={"shade": True})

plt.ylabel("Frequency")

plt.title("Distribution of App Ratings");
plt.figure(figsize=(8,6))

plt.hist(apps.Rating, range=(1,5), bins=16)

plt.axvline(x=apps.Rating.mean(), linewidth=4, color='g', label="mean")

plt.axvline(x=apps.Rating.median(), linewidth=4, color='r', label="median")

plt.xlabel("App Ratings")

plt.ylabel("Count")

plt.title("Histogram of App Ratings")

plt.legend(["mean", "median"])

plt.show();
avg_category_rating = apps.groupby("Category")['Rating'].mean().sort_values(ascending=False).reset_index()

avg_category_rating.head(10)
plt.figure(figsize=(17,6))

x = sns.barplot(x="Category", y="Rating", data=avg_category_rating, palette="Blues") 

plt.xticks(rotation=90)

x.set(ylim=(3.5,5))

plt.axhline(apps['Rating'].mean(),color='green');

plt.text(x = 27, y = 4.25, s = 'Mean App Rating', color='green',fontsize = 18,style = 'italic');
sns.lineplot(data=install_reviews,x='Installs',y='Reviews');

plt.title("Reviews vs Installs");
apps['Review_to_Install_Ratio'] = apps['Reviews'] / apps['Installs']

apps['Review_to_Install_Ratio'].head()
f, axes = plt.subplots(1, 3, figsize=(35, 10), sharex=True) #set the plotsize, divide p





g1 = sns.kdeplot(apps.Review_to_Install_Ratio[apps.Installs == 1000000000], shade=True, ax=axes[0], color="blue")

g1.title.set_text("Distriution of Reviews per Download in 1 Billion Installed Apps")



g2 = sns.kdeplot(apps.Review_to_Install_Ratio[apps.Installs == 500000000], shade=True, ax=axes[1], color="green")

g2.title.set_text("Distriution of Reviews per Download in 500 Million Installed Apps")



g3 = sns.kdeplot(apps.Review_to_Install_Ratio[apps.Installs == 100000000], shade=True, ax=axes[2],color="red")

g3.title.set_text("Distriution of Reviews per Download in 100 Million Installed Apps")
#Do we have a correlation between price of the app and rating?

plt.figure(figsize=(12,6))

plt.axhline(y=apps.Rating.mean(), linewidth=4, color='g', label="mean")

sns.regplot(x="Price", y="Rating", data=apps,scatter_kws={'alpha':0.15})

plt.title("Price VS Rating", size=20)

plt.legend();
#Chagning MB and GB to overall KB's

def num_size(Size):

    if Size[-1] == 'k':

        return float(Size[:-1])*1024

    else:

        return float(Size[:-1])*1024*1024
apps['Size'].value_counts().head()
apps['Size'] = apps['Size'].replace('Varies with device',0.0)
apps3 = apps[apps['Size']!=0.0].copy()

apps3['Size'].value_counts()
apps3['Size']=apps3['Size'].map(num_size).astype(float)
apps3['Size'].head()
#apps['Android_Ver']=apps.Android_Ver.apply(lambda x: x.replace('nan', '9999'))
#Size vs reviews

plt.figure(figsize=(15,5))

sns.scatterplot(data=apps3,x='Size',y='Rating')

plt.axhline(apps3['Rating'].mean(),color='red')

plt.title("Size vs Ratings");
ab = apps3[(apps3['Category']== 'GAME') | (apps3['Category']== 'FAMILY')]

plt.figure(figsize=(15,6))

sns.scatterplot(data=ab,x='Size',y='Rating',hue='Category')

plt.axhline(apps3['Rating'].mean(),color='brown')

plt.title(" Size vs Ratings wrt Category");
f = np.log(apps3['Size'])

g = np.log(apps3['Reviews'])



ac = apps3[(apps3['Category']== 'GAME') | (apps3['Category']== 'FAMILY') | (apps3['Category']== 'DATING') | (apps3['Category']== 'TOOLS')]



plt.figure(figsize=(16,7))

sns.scatterplot(f,g,hue=ac['Category'])

plt.title("Size vs Reviews wrt Category");
plt.figure(figsize=(16,5))

sns.boxplot(data=apps3,y='Price',x='Category')

plt.xticks(rotation=90);

plt.title("Category Vs Price");
plt.figure(figsize=(16,5))

sns.boxplot(data=apps3[apps3['Price']<250],y='Price',x='Category')

plt.xticks(rotation=90);

plt.title("Category Vs Price");
medi = apps3[(apps3['Category']=='MEDICAL') & (apps3['Price']>0)]

medi.head(2)
sns.countplot(data=medi,x='Content Rating');

plt.title("Medical Apps Content Ratings");
plt.figure(figsize=(16,5))

sns.scatterplot(data=apps3,y='Rating',x='Size',hue='Type')

plt.axhline(apps3['Rating'].mean(),color='red');

plt.title("Size Vs Ratings");
plt.figure(figsize=(16,5))

sns.scatterplot(data=apps3,y='Rating',x='Size',hue='Content Rating')

plt.axhline(apps3['Rating'].mean(),color='red');
plt.figure(figsize=(19,5))

plt.subplot(121)

x = np.log(apps2['Installs'])

y = np.log(apps2['Reviews'])

sns.scatterplot(x,y,hue= apps3['Content Rating']);

plt.title("Installs vs Reviews");

plt.legend(loc='best');



plt.subplot(122)

sns.heatmap(apps2[['Installs','Reviews']].corr(),annot=True)

plt.title("Correlation between both");
g = sns.kdeplot(np.sqrt(apps['Reviews']), color="Green", shade = True)

g.set_xlabel("Reviews")

g.set_ylabel("Frequency")

plt.title('Distribution of Reveiw');
apps3.loc[(apps3['Price'] > 0) & (apps3['Price'] <= 0.99), 'PriceBand'] = '1 cheap'

apps3.loc[(apps3['Price'] > 0.99) & (apps3['Price'] <= 2.99), 'PriceBand']   = '2 not cheap'

apps3.loc[(apps3['Price'] > 2.99) & (apps3['Price'] <= 4.99), 'PriceBand']   = '3 normal'

apps3.loc[(apps3['Price'] > 4.99) & (apps3['Price'] <= 14.99), 'PriceBand']   = '4 expensive'

apps3.loc[(apps3['Price'] > 14.99) & (apps3['Price'] <= 29.99), 'PriceBand']   = '5 too expensive'

apps3.loc[(apps3['Price'] > 29.99), 'PriceBand']  = 'Highly Expensive'

apps3[['PriceBand', 'Rating']].groupby(['PriceBand'], as_index=False).mean()
g = sns.catplot(x="PriceBand",y="Rating",data=apps3, kind="boxen", height = 10 ,palette = "Pastel1")

g.despine(left=True)

g.set_xticklabels(rotation=90)

g = g.set_ylabels("Rating")

plt.title('Boxen plot Rating VS PriceBand',size = 20);
#Days from last updations



apps3['new'] = pd.to_datetime(apps3['Last Updated'])



apps3['lastupdate'] = (apps3['Last Updated'] -  apps3['Last Updated'].max()).dt.days

apps3['lastupdate'].head()



plt.figure(figsize = (10,10))

sns.regplot(x="lastupdate", y="Rating", color = 'lightgreen',data=apps3);

plt.title('Rating  VS Last Update( days ago )',size = 20);
apps3['Total_money_generated'] = apps3['Installs'] * apps3['Price']
top_10_category_revenue = apps3.groupby('Category')['Total_money_generated'].sum().astype(int).sort_values(ascending=False).head(10)

top_10_category_revenue
plt.figure(figsize=(16,10))

plt.subplot(121)

top_10_category_revenue.head().plot.pie(autopct='%1.f%%',

                                                             wedgeprops={'linewidth':10,'edgecolor':'white'},

                                                             shadow=True,

                                                             fontsize=20)

plt.title("Percentage of top 5 Revenue generating Apps")



plt.subplot(122)

top_10_category_revenue.head().plot(color='brown',marker='o',markersize='10',linestyle='dashed',linewidth=3)

top_10_category_revenue.head().plot(kind='bar',color='pink');

plt.title("Top 5 Revenue generating Apps");
# Creating Tree Map



plt.figure(figsize=(16,9))

squarify.plot(sizes=top_10_category_revenue.values,label=top_10_category_revenue.index,value=top_10_category_revenue.values,color=["#FF6138","#FFFF9D","#BEEB9F", "#79BD8F","#684656","#E7EFF3"], alpha=0.6)

plt.title('Revenue Generated by Each App');
plt.figure(figsize=(16,5))

sns.countplot(data=apps3,x='Content Rating',hue='Category')

#pltlegend().remove()

plt.legend(loc='right');

plt.title("Content type wrt Category");
plt.figure(figsize=(15,5))

sns.heatmap(apps3.corr(),annot=True);
apps3.nlargest(5,'Reviews')
apps3.nsmallest(5,'Reviews')
#Grouping the number of reviews in 4 groups -A, B, C, Highest

bins = [0,100,100000,1000000,100000000]

labels =["A","B","C","Highest"]

apps3['Review_category'] = pd.cut(apps3['Reviews'],bins=bins, labels=labels)
apps3['Review_category'] = pd.cut(apps3['Reviews'],bins=bins, labels=labels)
apps3['Reviews'].max()
ax = sns.countplot(data=apps3,x='Review_category')

for p in ax.patches:

    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),

               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=20);

plt.title("Distribution of Review Category");
plt.figure(figsize=(5,5))

x = sns.barplot(x='Review_category', y='Rating', data=apps3, hue='Type', palette='husl')

plt.legend(loc=4)

plt.title('Number of applications installed and their ratings with respect to the type of application')

x.set(ylim=(1,5));
plt.figure(figsize=(5,5))

sns.barplot(x='Review_category', y='Reviews', data=apps3, hue='Type', palette='husl')

plt.legend(loc=0)

plt.title('Group of installed applications with respect to Reviews and type');
aas = apps3[(apps3['Size']<51200)]

aas = aas.sort_values(by=['Installs','Reviews'], ascending=False).reset_index().drop('index',axis=1)

aas.head(3)
sns.barplot(data=aas.head(),y='App',x='Installs')

plt.xticks(rotation=90);
sns.pointplot(data=aas,y='Reviews',x='Installs',marker='o')

plt.xticks(rotation=90);
ass2 = apps3[(apps3['Category']=='SOCIAL')&(apps3['Size']<50000000)]

ass2 = ass2.sort_values(by='Installs',ascending=False)

ass2.drop_duplicates('App',inplace=True);
sns.barplot(data=ass2.head(5),y='App',x='Installs',hue='Review_category');

plt.title("Apps in Social Cateogory that are small size but have highest Installations");
sns.heatmap(apps2[['Rating','Reviews','Installs','Size','Price']].corr(), annot=True, fmt='.2f', cmap='YlGnBu_r');
sns.pairplot(apps,hue = 'Type', palette='Set2');
#for label in ax.get_xticklabels():

#    label.set_rotation(90)



sns.catplot(x = 'Content Rating', y = 'Rating',hue='Type',data=apps,

            kind = 'violin', inner = 'stick', split = True,

            height=8, aspect=1.5, palette='Set3')

plt.title("Content Rating vs Rating wrt Type");
f,ax=plt.subplots(1,2,figsize=(14,10))

a = apps[apps['Type']=='Free']['Category'].value_counts()

a.plot(kind='barh',ax=ax[0])

ax[0].set_title('Category based on Free and Paid')

b = apps[apps['Type']=='Paid']['Category'].value_counts()

b.plot(kind='barh',ax=ax[1]);

#plt.xticks(rotation=90)
apps[apps['Type']=='Free']['Category'].value_counts().reset_index().head()
plt.figure(figsize=(16,6))

sns.boxplot(data=apps,x='Category',y='Rating')

plt.xticks(rotation=90)

plt.title("Category vs Rating");
top_10_apps_mostRating_2018_mostDownloads = apps3[(apps3['Rating'] >= 4.5) & (apps3['Year_of_Updation']== 2018) & (apps3['Installs'] >= 100000000)]
top_10_apps_mostRating_2018_mostDownloads = top_10_apps_mostRating_2018_mostDownloads.drop_duplicates('App')

top_10_apps_mostRating_2018_mostDownloads.head()
ap1 = top_10_apps_mostRating_2018_mostDownloads[top_10_apps_mostRating_2018_mostDownloads['Category']=="GAME"]

ap1 = ap1.sort_values(by='Installs',ascending=False)

ap1.head()
sns.lineplot(data=ap1.head(10),x='App',y='Reviews',marker='o')

plt.xticks(rotation=90)

##plt.title("Game Category Apps wrt Reviews");
sns.barplot(data=ap1.head(10),x='App',y='Reviews',hue='Content Rating')

plt.xticks(rotation=90)

plt.title("Games with overall Reviews wrt Content Ratings");
new_app = apps[['Category','Rating','Android Ver']].copy()
ab = new_app.groupby('Category').mean().reset_index()

ab.head()
f,ax2 = plt.subplots(figsize =(20,10))

sns.pointplot(x='Category',y='Android Ver',data=ab,color='magenta',alpha=0.8)

sns.pointplot(x='Category',y='Rating',data=ab,color='aqua',alpha=0.8)

plt.text(x = 18, y = 4.3, s = 'Average Rating', color = 'aqua', fontsize = 17,style = 'italic')

plt.text(x = 18, y = 3.46, s = 'Average Min Supported Android Ver', color='magenta',fontsize = 18,style = 'italic')

plt.xlabel('Categories', fontsize = 15, color = 'black')

plt.ylabel('Ratings', fontsize = 15, color = 'black')

plt.xticks(rotation =90)

plt.title(" Avg Ratings vs Ang minimum Supported Android Version");
g = sns.jointplot(ab['Android Ver'], ab['Rating'], kind="kde", height=7, color='aqua')

plt.savefig('graph.png')

plt.show()
plt.figure(figsize = (12,7))

sns.boxplot(x='Content Rating', y='Rating', hue='Type', data=apps, palette='PRGn')

plt.title("Content Rating Vs Rating");
plt.figure(figsize=(15,5))

plt.axhline(y=apps.Rating.mean(), linewidth=4, color='g', label="mean")

sns.scatterplot(data=apps3,x='Size',y='Rating',alpha = 0.5)

plt.xlabel('Size')              

plt.ylabel('Rating')

plt.title('How do Sizes impact the app rating?');
plt.figure(figsize=(20,10))

ax=sns.countplot('Installs',data=apps)

for p in ax.patches:

    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),

               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=20)

plt.xticks(rotation=90,fontsize=20)

plt.yticks(fontsize=20)

plt.xlabel('Installs',fontsize=20)

plt.ylabel('Count',fontsize=20)

plt.title('Installation Count of google apps',fontsize=20,color='red');
p_price=apps.groupby('Category')['Price'].sum().sort_values(ascending=False).reset_index()

plt.figure(figsize=(17,5))

sns.barplot(x='Category',y='Price',data=p_price)



plt.xlabel('Category',fontsize=15)

plt.ylabel('Price',fontsize=15)

plt.xticks(rotation=90,fontsize=15)

plt.yticks(fontsize=15)

plt.title('Total Price according to the category in google apps',fontsize=15);
content_price=apps.groupby('Content Rating')['Price'].sum().sort_values(ascending=False).reset_index()

plt.figure(figsize=(17,5))

sns.barplot(x='Content Rating',y='Price',data=content_price)

plt.ylabel('Content Rating',fontsize=15)

plt.xlabel('Price',fontsize=15)

plt.xticks(fontsize=15)

plt.yticks(fontsize=15)

plt.title('Total Price of content rating of google apps',fontsize=15);
plt.figure(figsize=(30,20))

plt.subplot(121)

apps.groupby('Content Rating')['Genres'].nunique().plot.pie(autopct='%1.f%%',

                                                             wedgeprops={'linewidth':10,'edgecolor':'white'},

                                                             shadow=True,

                                                             fontsize=20)

plt.title(" % Distribution based on Generes")



plt.subplot(122)

apps.groupby('Type')['Genres'].nunique().plot.pie(autopct='%1.f%%',

                                                             wedgeprops={'linewidth':5,'edgecolor':'white'},

                                                             shadow=True,

                                                             fontsize=20)

plt.title(" % Distribution based on Generes Type");
### Same using Pie plot



plt.figure(figsize=(30,20))

plt.subplot(121)

apps.groupby('Content Rating')['Genres'].nunique().plot.pie(autopct='%1.f%%',

                                                             wedgeprops={'linewidth':10,'edgecolor':'white'},

                                                             shadow=True,

                                                             fontsize=20)

plt.ylabel('')

circle=plt.Circle((0,0),0.5,color='white')

plt.gca().add_artist(circle)

plt.title('Content rating according to genres',fontsize=20)



plt.subplot(122)

apps.groupby('Type')['Genres'].nunique().plot.pie(autopct='%1.f%%',

                                                             wedgeprops={'linewidth':5,'edgecolor':'white'},

                                                             shadow=True,

                                                             fontsize=20)

plt.ylabel('')

circle=plt.Circle((0,0),0.5,color='white')

plt.gca().add_artist(circle)

plt.title('Google app type according to genres',fontsize=20);
plt.figure(figsize=(6,12))

g = sns.FacetGrid(apps3,col='Content Rating', hue='Type', col_wrap=3)

g.map(sns.regplot, 'Installs', 'Reviews');
apps[apps['App'].str.startswith('G')].head(2)
apps[apps['App'].str.endswith('E')].head(2)
apps['Installs'] = apps['Installs'].replace('Free',np.nan)

apps['Installs'] = apps['Installs'].fillna(apps['Installs'].interpolate())

apps['Installs'] = pd.to_numeric(apps['Installs'])



apps.groupby('Category')['Installs'].agg(['mean', 'min', 'max']).astype(int).head()
capitalizer = apps['Last_Updated_Month'].apply(lambda x: x.capitalize())

capitalizer
apps.loc[apps['Reviews'].idxmax(),]['App']
dummy = pd.get_dummies(apps3['Type'])

dummy
apps2 = pd.concat([apps2, dummy], axis=1)
apps2.drop('Type',axis=1,inplace=True)
apps2.head(2)
apps['Content Rating'].value_counts()
dummy1 = pd.get_dummies(apps3['Content Rating'])

dummy1
apps2 = pd.concat([apps2, dummy1], axis=1)

apps2.drop('Content Rating',axis =1,inplace=True)

apps2.head(2)
## Let's find the top 10 most frequent categories for the variable Category



apps3['Category'].value_counts().sort_values(ascending=False).head(10)
# let's make a list with the most frequent categories of the variable



top_10 = [x for x in apps3['Category'].value_counts().sort_values(ascending=False).head(10).index]

top_10
## Creating function to perform One Hot Encoding



def one_hot_top_x(df, variable, top_x_labels):

        

    for label in top_x_labels:

        df[variable+'_'+label] = np.where(apps2[variable]==label, 1, 0)
one_hot_top_x(apps2, 'Category',top_10)
# Removing the Category Column

apps2.drop('Category',axis=1,inplace=True)

apps2.head(2)
import statistics as sts

from statistics import mode
apps4 = apps3.copy()

print("Mean of Rating: " + str(round(apps4.Rating.mean(),2)) + " Median of Rating:" + str(round(apps4.Rating.median(),2)))
sns.distplot(apps4['Rating'],norm_hist=True)

plt.title("Distribution of Ratings");
plt.figure(figsize=(8,6))

plt.hist(apps4.Rating, range=(1,5), bins=16)

plt.axvline(x=apps4.Rating.mean(), linewidth=4, color='g', label="mean")

plt.axvline(x=apps4.Rating.median(), linewidth=4, color='r', label="median")

plt.axvline(x=mode(apps4['Rating']), linewidth=4, color='yellow', label="mode")

plt.xlabel("App Ratings")

plt.ylabel("Count")

plt.title("Histogram of App Ratings")

plt.legend();
sns.boxplot(data=apps4,x='Price')

plt.title("Price Variation")
# Seeing the variance



va1 = apps4['Rating'].var()



print("Variance: " + str(round(va1,2)))



#-> We can see the variance is too large to compare the dispersion of age, we have to use standard deviation



#Cal standard deviation and coff of variation (sd/mean) of age

staDev = apps4['Rating'].std()



print("Standard Deviation: " + str(round(staDev,2)))



#To Compare 2 variable we need coff of variance



coffOfVariation = apps4['Rating'].std()/apps4['Rating'].mean()

coffOfVariation



print("Coffecient of Varaition: " + str(round(coffOfVariation,2)))
cov = apps4['Rating'].cov(apps4['Price'])



print("Covarience: " + str(cov))



# cal corelation coeff (corelation adjust covariance, so that relationship b/w 2 variables become easy to understand)



#(corelation coeff = cov(x,y)/sd(x).sd(y), -1 to 1 range)



sdMul = apps4['Rating'].std()*apps4['Price'].std()



corelation_coeff_Rating_Price = apps4['Rating'].cov(apps4['Price'])/sdMul

print("Correlation Coefficient: " + str(corelation_coeff_Rating_Price))
top10_rating_eachApp.head(3)



# These are the Top 3 apps Category so we will compare these
family = apps4[(apps4['Category'] == 'FAMILY') & (apps4['Price'] > 0)]['Price'].head(78)

game = apps4[(apps4['Category'] == 'GAME') & (apps4['Price'] > 0)]['Price'].head(78)

tools = apps4[(apps4['Category'] == 'TOOLS') & (apps4['Price'] > 0)]['Price'].head(78)
family = family.reset_index()

family.columns = ['index','Price']

family.drop('index',axis=1,inplace=True)

#family['Price'] = round(family['Price'],1)



game = game.reset_index()

game.columns = ['index','Price']

game.drop('index',axis=1,inplace=True)



tools = tools.reset_index()

tools.columns = ['index','Price']

tools.drop('index',axis=1,inplace=True)



price = pd.concat([family,game,tools],axis=1)

price.columns=['Family_price','Game_price','Tool_price']

price.head()
# Comparing variance, Standard deviation and cofficient of variation of 3 Category Price



v1_Family = price['Family_price'].var()



print("Variance of Family Category: " + str(round(v1_Family,2)))



v2_Game = price['Game_price'].var()



print("Variance of Game Category: " + str(round(v2_Game,2)))



v3_Tool = price['Tool_price'].var()



print("Variance of Tool Category: " + str(round(v3_Tool,2)))







#-> Since var is has a square unit we cant compare so we calculate sd



sd1_Family = price['Family_price'].std()



print("Standard Deviation of Family Category: " + str(round(sd1_Family,2)))



sd2_Game = price['Game_price'].std()



print("Standard Deviation of Game Category: " + str(round(sd2_Game,2)))



sd3_Tool = price['Tool_price'].std()



print("Standard Deviation of Tool Category: " + str(round(sd3_Tool,2)))







#-> We need to cal coff of variation of comparision b/w the ratings



coff_family = price['Family_price'].std()/price['Family_price'].mean()

print("Coff of variance of Family Category: " + str(round(coff_family,2)))



coff_game = price['Game_price'].std()/price['Game_price'].mean()

print("Coff of variance of Game Category: " + str(round(coff_game,2)))



coff_tool = price['Tool_price'].std()/price['Tool_price'].mean()

print("Coff of variance of Tool Category: " + str(round(coff_tool,2)))
# Statistical Analysis



# Taking sample



sample_data =  apps4.sample(700)

sample_data.shape
# Confidence Interval = (Sample Mean + Margin of Error , Sample Mean - Margin of Error)



# Margin of Error = (critical value) * (SD)/(sqrt(Sample Size))    (Standard_Error = (SD)/(sqrt(Sample Size))

  



#Step1) Calculating mean, sd and SE

m1 = sample_data['Rating'].mean()



sd1 =sample_data['Rating'].std()



sample_size = len(sample_data)



SE =  sd1/np.sqrt(sample_size)



print("Mean is: "+ str(round(m1,2)) + " Standard Deviation is: " + str(round(sd1,2)) + " Standard Error is: " + str(round(SE,2)))
#Step2) Finding T statistic 

# Degree of Freedom = n-1 = 700-1 = 699 and for 95% confidence i.e alpha is 5% so alpha/2 is .25



import scipy.stats as stats



t_statistic = stats.t.ppf(q=.95,df=699)

t_statistic
#Step3) using the formula to find confidence interval 



#### [mean(ages2) - 1.65*SE , mean(ages2) + 1.65*SE]



CI_lowerLimit = m1 - t_statistic * SE

CI_upperLimit = m1 + t_statistic * SE



print("We are 95% confident that the average Rating of Apps is between: " + "(" + str(round(CI_lowerLimit,2)) + " , " + str(round(CI_upperLimit,2)) + ")")
# Confidence Interval = (Sample Mean + Margin of Error , Sample Mean - Margin of Error)



# Margin of Error = (critical value) * (SD)/(sqrt(Sample Size))    (Standard_Error = (SD)/(sqrt(Sample Size))

  

sample_data2 =  apps4[apps4['Category']=='FAMILY'].sample(700)

sample_data2.shape



#Step1) Calculating mean, sd and SE

m1 = sample_data2['Reviews'].mean()



sd1 =sample_data2['Reviews'].std()



sample_size = len(sample_data2)



SE =  sd1/np.sqrt(sample_size)



print("Mean is: "+ str(round(m1,2)) + " Standard Deviation is: " + str(round(sd1,2)) + " Standard Error is: " + str(round(SE,2)))
#Step2) Finding T statistic 

# Degree of Freedom = n-1 = 700-1 = 699 and for 95% confidence i.e alpha is 5% so alpha/2 is .25



import scipy.stats as stats



t_statistic = stats.t.ppf(q=.95,df=699)

t_statistic
#Step3) using the formula to find confidence interval 



#### [mean(ages2) - 1.65*SE , mean(ages2) + 1.65*SE]



CI_lowerLimit = m1 - t_statistic * SE

CI_upperLimit = m1 + t_statistic * SE



print("We are 95% confident that the average Reviews of Apps is between: " + "(" + str(round(CI_lowerLimit,2)) + " , " + str(round(CI_upperLimit,2)) + ")")
family_paid = apps[(apps['Category'] == 'FAMILY') & (apps['Price'] > 0)]['Installs']



family_free = apps[(apps['Category'] == 'FAMILY') & (apps['Price'] == 0)]['Installs'].head(191)
family_paid = family_paid.reset_index()

family_paid.columns = ['index','Installs']

family_paid.drop('index',axis=1,inplace=True)



family_free = family_free.reset_index()

family_free.columns = ['index','Installs']

family_free.drop('index',axis=1,inplace=True)



installs = pd.concat([family_paid,family_free],axis=1)

installs.columns=['Paid_Apps_Installs','Free_Apps_Installs']

installs.head()
installs['Paid_Apps_Installs'] = pd.to_numeric(installs['Paid_Apps_Installs'])

installs['Free_Apps_Installs'] = pd.to_numeric(installs['Free_Apps_Installs'])
# Step1)Cal mean and sd



m1_free = installs['Free_Apps_Installs'].mean()

m2_paid = installs['Paid_Apps_Installs'].mean()



sd1_free = installs['Free_Apps_Installs'].std()

sd2_paid = installs['Paid_Apps_Installs'].std()



print("Mean of Free Apps: " + str(round(m1_free,2)) + " Standard deviation of Free Apps:" + str(round(sd1_free)))

print("Mean of Paid Apps: " + str(round(m2_paid,2)) + " Standard deviation of Paid Apps:" + str(round(sd2_paid)))



print("Total number of Installation: " + str(len(installs)))
#Step2) Calculate unbiased estimator i.e pooled sample variance

pooled_variance = 190*20586486*20586486  + 190*1029624*1029624 /380

polled_variance = 21243176570478

polled_sd = np.sqrt(21243176570478)

polled_sd
# Step3) Calculating Student's T statistic



#Degree of freedom is 191 + 191 -2 =380 and confidence level is 95%

import scipy.stats as stats



t_stat = stats.t.ppf(q=.95,df=380)

t_stat



#appling formula

lower_bound = (9539900.52 - 163726.77) - t_stat*np.sqrt(212431765704786/24 + 212431765704786/24)

upper_bound = (9539900.52 - 163726.77) + t_stat*np.sqrt(212431765704786/24 + 212431765704786/24)



print("We are 95% confident that Free Apps Installs for Family Category is greater than Paid Apps by margin of :")

print("(" + str(round(lower_bound,2)) + " , " + str(round(upper_bound,2))+ ")")
#Test at 5% significance. Calculate the p-value of the test



#Taking sample of 200 ages

tools_size = apps3[apps3['Category']=='TOOLS']['Size'].sample(200)



# Step1) Writing null and alternative hypothesis 



# H0:mean of tools App Size is > 9000000

# H1:mean of tools App Size is < 9000000



# Its a one-sided tail test



# Step2) Cal the mean,sd and SE

m1_size = tools_size.mean()



sd1_size = tools_size.std()



SE = sd1_size/np.sqrt(200) # SE-> Standard Error



print("Mean of tools: " + str(round(m1_size,2)) + " Std of tools: " + str(round(sd1_size,2)) + " SE is:" + str(round(SE,2)))
# Step3) Finding T Score



# T=  x - hyphosis_mean(x) / SE



T_score , p_value = stats.ttest_1samp(a=tools_size,popmean=9196363.03)



# Step4) Finding Critical T value and comapring

# as degree of freedom is 200 -1 = 199 and we are caluclation at alpha=0.05 (95%)



t_critical = stats.t.ppf(q=.95,df=199)

print("Critical t value: " + str(round(t_critical,3)) + ", T score is: " + str(T_score))
# Step4) Finding P value



T_score , p_value = stats.ttest_1samp(a=tools_size,popmean=9196363.03)



print("P value: " + str(round(p_value,2)))
#Test at 5% significance. Calculate the p-value of the test



#Taking sample of 350 ages

game_ratings = apps3[apps3['Category']=='GAME']['Rating'].sample(350)



# Step1) Writing null and alternative hypothesis 



# H0:mean of game Category rating < 4

# H1:mean of game Category rating > 4



# Its a one-sided tail test



# Step2) Cal the mean,sd and SE

m1_rating = game_ratings.mean()



sd1_rating = game_ratings.std()



SE_rating = sd1_rating/np.sqrt(350) # SE-> Standard Error



print("Mean of Ratings: " + str(round(m1_rating,2)) + " Std of Ratings: " + str(round(sd1_rating,2)) + " SE is:" + str(round(SE_rating,2)))
# Step3) Finding T Score



# T=  x - hyphosis_mean(x) / SE

population_mean = apps3[apps3['Category']=='GAME']['Rating'].mean()



T_score = (population_mean - 4)/ SE_rating

T_score = 5.71



# Step4) Finding Critical T value and comapring

# as degree of freedom is 350 -1 = 349 and we are caluclation at alpha=0.05 (95%)



t_critical = stats.t.ppf(q=.95,df=349)

print("Critical t value: " + str(round(t_critical,3)) + ", T score is: " + str(T_score))
# Step4) Finding P value



#As per the P value calculator wrt T Score ,DF ,Significance Level ,One-tailed

p_value = 0.00001



print("P value: " + str(p_value))
tools_paid = apps4[(apps4['Category'] == 'TOOLS') & (apps4['Type'] == 'Paid')]['Installs']



game_paid = apps4[(apps4['Category'] == 'GAME') & (apps4['Type'] == 'Paid')]['Installs'].head(78)



tools_paid = tools_paid.reset_index()

tools_paid.columns = ['index','Installs']

tools_paid.drop('index',axis=1,inplace=True)



game_paid = game_paid.reset_index()

game_paid.columns = ['index','Installs']

game_paid.drop('index',axis=1,inplace=True)



installs_all = pd.concat([tools_paid,game_paid],axis=1)

installs_all.columns=['Tools_Paid_Apps_Installs','Game_Paid_Apps_Installs']

installs_all.head(10)
# Step1) Writing null and alternative hypothesis 



# H0: mean(tools) = mean(games)  i.e mean(tools) - mean(games) = 0

# H1: mean(tools) != mean(games) i.e mean(tools) - mean(games) != 0



# Step2) Cal mean and sd



m1_tools = installs_all['Tools_Paid_Apps_Installs'].mean()

m2_games = installs_all['Game_Paid_Apps_Installs'].mean()



sd1_tools = installs_all['Tools_Paid_Apps_Installs'].std()

sd2_games = installs_all['Game_Paid_Apps_Installs'].std()



print("Mean of Tools Category: " + str(round(m1_tools,2)) + " Standard deviation of Tools Category:" + str(round(sd1_tools)))

print("Mean of Games Category: " + str(round(m2_games,2)) + " Standard deviation of Games Category:" + str(round(sd2_games)))



print("Total number of Installation: " + str(len(installs_all)))



SE1 = sd1_tools/np.sqrt(156)

SE2 = sd2_games/np.sqrt(156)



SE = SE1+SE2 # SE-> Standard Error

print("Standard Error is: "+ str(round(SE,2)))
#Step3) Calculate unbiased estimator i.e pooled sample variance



#degree_freedom<- 78+78-2 

degree_freedom= 154



# using formula of Pooled variance -> (n1-1)*sd1^2 + (n2-1)*sd2^2 /(n1+n2-2) 

pooled_variance = 77*114450*114450 + 77*1152922*1152922 /156  

polled_variance = 671163970292.0

polled_sd = np.sqrt(671163970292.0)

polled_sd
# Step4) Finding T Score

#t_score <- diff of sample mean - diff of hyphothsis mean / SE



#t_score <- (m2 - m1) - 0 /SE

T_score = (m2_games - m1_tools) - 0 /SE

T_score = 2.307



# Step5) Finding Critical T value and comapring



#Degree of freedom is 78 + 78 -2 = 156 and confidence level is 95%



t_critical = stats.t.ppf(q=.95,df=156)

print("Critical t value: " + str(round(t_critical,3)) + ", T score is: " + str(T_score))
# Step6) Finding P value



#As per the P value calculator wrt T Score ,DF ,Significance Level ,One-tailed

p_value = .02237



print("P value: " + str(p_value))
personization_price = apps4[(apps4['Category'] == 'PERSONALIZATION') & (apps4['Price']>0) & (apps4['Content Rating']=='Everyone')]['Price']

medical_price = apps4[(apps4['Category'] == 'MEDICAL') & (apps4['Price']>0) & (apps4['Content Rating']=='Everyone')]['Price'].head(83)



personization_price = personization_price.reset_index()

personization_price.columns = ['index','price']

personization_price.drop('index',axis=1,inplace=True)



medical_price = medical_price.reset_index()

medical_price.columns = ['index','price']

medical_price.drop('index',axis=1,inplace=True)



price_all = pd.concat([personization_price,medical_price],axis=1)

price_all.columns=['Personaization_Apps_Price','Medical_Apps_Price']

price_all.tail(5)
# Step1) Writing Null and Alternative Hyphothesis 



# H0: mean(personalization) = mean(medical)  i.e mean(personalization) - mean(medical) = 0

# H1: mean(personalization) != mean(medical) i.e mean(personalization) - mean(medical) != 0



# Step2) Cal mean and sd



m1_personalization = price_all['Personaization_Apps_Price'].mean()

m2_medical = price_all['Medical_Apps_Price'].mean()



sd1_personalization = price_all['Personaization_Apps_Price'].std()

sd2_medical = price_all['Medical_Apps_Price'].std()



print("Mean of personalization Category: " + str(round(m1_personalization,2)) + " Standard deviation of personalization Category:" + str(round(sd1_personalization,2)))

print("Mean of medical Category: " + str(round(m2_medical,2)) + " Standard deviation of medical Category:" + str(round(sd2_medical,2)))



print("Total number of Installation: " + str(len(price_all)))



SE1 = sd1_personalization/np.sqrt(166)

SE2 = sd2_medical/np.sqrt(166)



SE = SE1+SE2 # SE-> Standard Error

print("Standard Error is: "+ str(round(SE,2)))
# Step3) Cal Pooled Sample variance



#degree_freedom = 83+83-2 

degree_freedom = 164



# using formula of Pooled variance -> (n1-1)*sd1^2 + (n2-1)*sd2^2 /(n1+n2-2)

pooled_variance = 82*sd1_personalization*sd1_personalization + 82*sd2_medical*sd2_medical /164

polled_variance = 125.99

polled_sd = np.sqrt(125.99)

polled_sd
# Step4) Finding T score



# T_score <- (m2 - m1) - 0 /SE

T_score = (m2_medical - m1_personalization) - 0 

T_score = T_score/SE



# Step5) Finding Critical t value and comapring



#Degree of freedom is 83 + 83 -2 = 166 and confidence level is 95%



t_critical = stats.t.ppf(q=.95,df=166)

print("Critical t value: " + str(round(t_critical,3)) + ", T score is: " + str(T_score))
# Step 6) Calculate P value



#As per the P value calculator wrt T Score ,DF ,Significance Level ,One-tailed

p_value = .000001



# calculating using T statistic

t_critical,p_value2 = stats.ttest_ind(a=price_all['Personaization_Apps_Price'],b=price_all['Medical_Apps_Price'],equal_var=False)



print("P value: " + str(p_value))