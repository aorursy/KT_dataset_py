import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



import warnings

warnings.filterwarnings("ignore")

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df1=pd.read_csv('../input/world-happiness/2015.csv')
df1.head()
df1.columns=['country','region','happiness_rank','happiness_score','standard_error','economy','family','health',

            'freedom','trust','generosity','dystopia_residual']
df1.head()
df1.tail()
df1.isnull().sum()
df1.info()
df1.describe().T
df1.corr()
df1 = df1.sort_values(by=["happiness_score"], ascending=False)

df1['rank']=tuple(zip(df1.happiness_score))

df1['rank']=df1.groupby('happiness_score',sort=False)['rank'].apply(lambda x : pd.Series(pd.factorize(x)[0])).values

df1.head()
df1.drop(["rank"],axis=1,inplace=True)
df1.reset_index(inplace=True,drop=True)

df1.head()
df1.nunique()
#World Happiness Score Freedom - 2015

plt.figure(figsize=(18,5))

sns.set(style='whitegrid')

ax=sns.barplot(x=df1['region'].value_counts().index,y=df1['region'].value_counts().values,palette="Blues_d")

plt.legend(loc=8)

plt.xlabel('region')

plt.xticks(rotation=75)

plt.ylabel('Frequency')

plt.title('Show of region Bar Plot')

plt.show()
f,ax=plt.subplots(figsize=(9,10))

sns.barplot(x=df1['country'].value_counts().values,y=df1['country'].value_counts().index,alpha=0.5,color='red',label='country')

sns.barplot(x=df1['region'].value_counts().values,y=df1['region'].value_counts().index,color='blue',alpha=0.7,label='region')

ax.legend(loc='upper right',frameon=True)

ax.set(xlabel='country , region',ylabel='Groups',title="country vs region ")

plt.show()
df1['region'].unique()

len(df1[(df1['region']=='Sub-Saharan Africa')].happiness_score)

f,ax1=plt.subplots(figsize=(15,10))

sns.pointplot(x=np.arange(1,41),y=df1[(df1['region']=='Sub-Saharan Africa')].health,color='lime',alpha=0.8)

sns.pointplot(x=np.arange(1,41),y=df1[(df1['region']=='Sub-Saharan Africa')].economy,color='red',alpha=0.5)

plt.xlabel('Sub-Saharan Africa index State')

plt.ylabel('Frequency')

plt.title('Sub-Saharan Africa health & economy')

plt.xticks(rotation=90)

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

plt.grid()

plt.show()
labels=df1['region'].value_counts().index

colors=['blue','red','yellow','green','brown']

explode=[0.3,0.3,0.2,0.1,0,0,0,0,0,0]

values=df1['region'].value_counts().values



#visualization

plt.figure(figsize=(7,7))

plt.pie(values,explode=explode,labels=labels,colors=colors,autopct='%1.1f%%')

plt.title('region According Analysis',color='black',fontsize=10)

plt.show()
plt.figure(figsize=(15,7))

sns.lmplot(x='happiness_score',y='economy',data=df1)

plt.xlabel('happiness_score')

plt.ylabel('economy')

plt.title('happiness_score vs economy')

plt.show()
plt.figure(figsize=(15,7))

sns.lmplot(x='happiness_score',y='health',data=df1)

plt.xlabel('happiness_score')

plt.ylabel('health')

plt.title('happiness_score vs health')

plt.show()
plt.figure(figsize=(15,7))

sns.lmplot(x='happiness_score',y='trust',data=df1)

plt.xlabel('happiness_score')

plt.ylabel('trust')

plt.title('happiness_score vs trust')

plt.show()
plt.figure(figsize=(15,7))

sns.lmplot(x='happiness_score',y='freedom',data=df1)

plt.xlabel('happiness_score')

plt.ylabel('freedom')

plt.title('happiness_score vs freedom')

plt.show()
plt.figure(figsize=(18,7))

sns.lmplot(x='economy',y='health',hue='region',data=df1)

plt.xlabel('happiness_score')

plt.ylabel('trust')

plt.title('happiness_score vs trust')

plt.show()
sns.kdeplot(df1['happiness_score'])

plt.xlabel('Values')

plt.ylabel('Frequency')

plt.title('happiness_score Kde Plot System Analysis')

plt.show()
plt.figure(figsize=(18,5))

sns.kdeplot(df1['health'],shade=True,color='r')

sns.kdeplot(df1['economy'],shade=True,color='g')

sns.kdeplot(df1['freedom'],shade=True,color='black')

sns.kdeplot(df1['family'],shade=True,color='orange')

sns.kdeplot(df1['generosity'],shade=True,color='yellow')

sns.kdeplot(df1['dystopia_residual'],shade=True,color='brown')

sns.kdeplot(df1['trust'],shade=True,color='b')

plt.xlabel('Values')

plt.ylabel('Frequency')

plt.title('Health, Economy, Freedom, Family, Generosity,Dystopia Residual, Trust Kde Plot System Analysis')

plt.show()
sns.violinplot(df1['happiness_score'])

plt.xlabel('happiness_score')

plt.ylabel('Frequency')

plt.title('Violin happiness_score Show')

plt.show()
plt.figure(figsize=(18,5))

sns.violinplot(x=df1['region'],y=df1['economy'])

plt.xticks(rotation=75)

plt.show()
sns.heatmap(df1.corr())

plt.show()
sns.heatmap(df1.corr(),vmin=0,vmax=1)

plt.show()
plt.figure(figsize=(18,5))

sns.heatmap(df1.corr(),annot=True,fmt=".0%")

plt.show()
# Compute the correlation matrix

corr = df1.corr()



# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})

plt.show()
sns.heatmap(df1.corr(),cmap='YlGnBu')

plt.show()
plt.figure(figsize=(18,5))

sns.set(style='whitegrid')

sns.boxplot(df1['happiness_score'])

plt.show()
plt.figure(figsize=(18,5))

sns.boxplot(x=df1['region'],y=df1['happiness_score'])

plt.xticks(rotation=75)

plt.show()
plt.figure(figsize=(18,5))

sns.boxenplot(x="region", y="health",

              color="b",

              scale="linear", data=df1)

plt.xticks(rotation=75)

plt.show()
plt.figure(figsize=(18,5))

sns.boxplot(x=df1['region'],y=df1['happiness_score'])

plt.xticks(rotation=75)

sns.swarmplot(x=df1['region'],y=df1['happiness_score'],color=".25")

plt.xticks(rotation=75)

plt.show()
sns.set(style='whitegrid')

sns.swarmplot(x=df1['happiness_score'])

plt.show()
sns.set(style="whitegrid")



sns.swarmplot(y=df1["happiness_score"],color='red')

sns.swarmplot(y=df1["dystopia_residual"],color='blue')

sns.swarmplot(y=df1["economy"],color='yellow')

sns.swarmplot(y=df1["trust"],color='purple')

sns.swarmplot(y=df1["health"],color='green')



plt.title('happiness_score & dystopia_residual & economy  & trust & health')

plt.show()
sns.swarmplot(x=df1['economy'],y=df1['health'])

plt.show()
sns.pairplot(df1)

plt.show()
sns.pairplot(df1,diag_kind='kde')

plt.show()
sns.pairplot(df1,kind='reg')

plt.show()
sns.countplot(df1['region'])

plt.xticks(rotation=75)

plt.show()
sns.stripplot(x=df1['happiness_score'])

plt.show()
ax = sns.distplot(df1['economy'], rug=True, hist=False)

plt.show()
ax = sns.distplot(df1['health'], vertical=True)

plt.show()
ax = sns.distplot(df1['trust'])

plt.show()
ax = sns.distplot(df1['family'], color="y")

plt.show()
plt.figure(figsize=(15,5))

sns.lineplot(x='happiness_score',y='economy',hue="region",estimator=None,data=df1)

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

plt.show()
df1.groupby('region')[['happiness_score','economy']].mean()
f, ax = plt.subplots(figsize=(6.5, 6.5))

sns.despine(f, left=True, bottom=True)

sns.scatterplot(x="happiness_score", y="economy",

                hue="region", size="region",data=df1)

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

plt.show()
sns.clustermap(df1.corr(), center=0, cmap="vlag",

               linewidths=.75, figsize=(13, 13))
sns.set(style="white")

# Plot miles per gallon against horsepower with other semantics

sns.relplot(x="economy",y="health",hue="region",

            sizes=(40, 400), alpha=.9, palette="muted",

            height=6, data=df1)

plt.show()
f, ax = plt.subplots(figsize=(6.5, 6.5))

sns.despine(f, left=True, bottom=True)

sns.scatterplot(x="economy", y="trust",

                hue="region", size="region",data=df1)

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

plt.show()
plt.figure(figsize=(16,5))

sns.barplot(x = df1.country[:30], y =df1.happiness_score[:30], hue = df1.region[:30], data = df1)

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

plt.xticks(rotation=45)

plt.show()
plt.figure(figsize=(16,5))

sns.barplot(x = df1.country[:20], y =df1.economy[:20], hue = df1.region[:20], data = df1)

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

plt.xticks(rotation=45)

plt.show()
plt.figure(figsize=(16,5))

sns.barplot(x = df1.country[:20], y =df1.family[:20], hue = df1.region[:20], data = df1)

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

plt.xticks(rotation=45)

plt.show()
plt.figure(figsize=(16,5))

sns.barplot(x = df1.country[:20], y =df1.health[:20], hue = df1.region[:20], data = df1)

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

plt.xticks(rotation=45)

plt.show()
plt.figure(figsize=(16,5))

sns.barplot(x = df1.country[:20], y =df1.freedom[:20], hue = df1.region[:20], data = df1)

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

plt.xticks(rotation=45)

plt.show()
plt.figure(figsize=(16,5))

sns.barplot(x = df1.country[:20], y =df1.trust[:20], hue = df1.region[:20], data = df1)

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

plt.xticks(rotation=45)

plt.show()
plt.figure(figsize=(16,5))

sns.barplot(x = df1.country[:20], y =df1.generosity[:20], hue = df1.region[:20], data = df1)

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

plt.xticks(rotation=45)

plt.show()
plt.figure(figsize=(16,5))

sns.barplot(x = df1.country[:20], y =df1.dystopia_residual[:20], hue = df1.region[:20], data = df1)

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

plt.xticks(rotation=45)

plt.show()
country_list1 = list(df1['country'])

economy_2015 = []

health_2015 =[]

freedom_2015=[]

family_2015=[]





for i in country_list1:

    x = df1[df1['country']==i]

    economy_2015.append(sum(x.economy)/len(x))

    health_2015.append(sum(x.health) / len(x))

    freedom_2015.append(sum(x.freedom) / len(x))

    family_2015.append(sum(x.family) / len(x))





# visualization



f,ax = plt.subplots(figsize = (15,45))

sns.barplot(x=economy_2015,y=country_list1,color='green',alpha = 0.5,label='economy_2015' )

sns.barplot(x=health_2015,y=country_list1,color='blue',alpha = 0.7,label='health_2015')

sns.barplot(x=freedom_2015,y=country_list1,color='cyan',alpha = 0.6,label='freedom_2015')

sns.barplot(x=family_2015,y=country_list1,color='yellow',alpha = 0.6,label='family_2015')





ax.legend(loc='lower right',frameon = True)

ax.set(xlabel='Percentage of Economy, Health, Freedom and Family 2015', ylabel='country',

       title = "Percentage of Country's According to Economy, Health, Freedom and Family 2015")

plt.show()
from wordcloud import WordCloud, ImageColorGenerator

text = " ".join(str(each) for each in df1.country[:20])

# Create and generate a word cloud image:

wordcloud = WordCloud(max_words=200, background_color="black").generate(text)

plt.figure(figsize=(15,10))

# Display the generated image:

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()
df1.head(10)
# import the library

import folium

 

# Make a data frame with dots to show on the map

data = pd.DataFrame({

   'lat':[46.20,64.12,55.67,59.91,43.76,60.19,52.37,58.29,-36.84,-35.47],

   'lon':[6.143,-21.82,12.56,10.75,79.41,24.94,4.89,12.96,174.7,149.0],

   'name':['Switzerland', 'Iceland', 'Denmark', 'Norway', 'Canada', 'Finland', 'Netherlands', 'Sweden',"New Zealand","Australia"],

   

})

data

 

 

# Make an empty map

m = folium.Map(location=[20, 0], tiles="Mapbox Bright", zoom_start=2)



# I can add marker one by one on the map

for i in range(0,len(data)):

    folium.Marker([data.iloc[i]['lon'], data.iloc[i]['lat']], popup=data.iloc[i]['name']).add_to(m)



# Save it as html

#m.save('312_markers_on_folium_map1.html')
