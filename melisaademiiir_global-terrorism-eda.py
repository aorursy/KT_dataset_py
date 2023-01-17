# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import pandas_profiling as pp

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from PIL import Image

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import scipy.stats as stats

import warnings

warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', 500)

df = pd.read_csv('../input/gtd/globalterrorismdb_0718dist.csv', encoding="ISO-8859-1", low_memory=False)

df.tail(10)
df.shape
df.rename(columns={'iyear':'Year','imonth':'Month','iday':'Day','country':'country1','country_txt':'Country','region':'Region1','region_txt':'Region','attacktype1':'attacktype1','attacktype1_txt':'AttackType','target1':'Target','nkill':'Killed','nwound':'Wounded','summary':'Summary','gname':'Group','targtype1_txt':'Target_type','weaptype1_txt':'Weapon_type','motive':'Motive'},inplace=True)
terror=df[['Year','Month','Day','Country','country1','Region','Region1', 'city','AttackType','attacktype1','Killed','Wounded','Target','Summary','Group','Target_type','Weapon_type','Motive']]

terror
terror.info() #type checking
terror['Day'].apply(np.floor)

terror['Month'].apply(np.floor)

terror['Killed'].apply(np.floor)

terror['Day'].apply(np.floor)

#I realized there are a couple of decimal numbers in the Day column. Then I fixed like this.
terror.describe(include=['O']) # a couple information abaout objects.
#replacing dtypes with correct ones.

terror['attacktype1'] = pd.Categorical(terror.attacktype1)

terror['Region1'] = pd.Categorical(terror.Region1)

terror['country1'] = pd.Categorical(terror.country1)

terror['Country'] = pd.Categorical(terror.Country)

terror['Region'] = pd.Categorical(terror.Region)

terror['city'] = pd.Categorical(terror.city)

terror['AttackType'] = pd.Categorical(terror.AttackType)

terror['Group'] = pd.Categorical(terror.Group)

terror['Target_type'] = pd.Categorical(terror.Target_type)

terror['Weapon_type'] = pd.Categorical(terror.Weapon_type)

terror.info() #looking good
terror.describe() #descriptive statistics for numeric variables
terror.groupby(by='Month').agg(['count']) #look what I found.0 as a month. We need to fix this. 
terror['Day'] = terror['Day'].apply(lambda x: np.random.randint(1,32) if x == 0 else x)

terror['Month'] = terror['Month'].apply(lambda x: np.random.randint(1,13) if x == 0 else x)



#There were too many 0 as Day value too. We changed 0 values with random integers between 1-31.  So we did not miss any value. 
a = terror.groupby(by='Day').agg(['count'])

b = a['Year']

b.columns = ['count']

b = b.reset_index()

b



#we checked is there any 0 value. looking good.
terror['date'] = pd.to_datetime(terror[['Year','Month', 'Day' ]], errors= 'coerce') #created a new column as merging year-month-day.

terror['date'] = pd.to_datetime(terror['date']) #double check

terror['day_of_week'] = terror['date'].dt.day_name() #a new columns as day of week. 

terror #let's see what's new.
terror.dropna(how="all",inplace=True) #If all values are NA, drop that row or column.
display(terror.isnull().sum().sort_values(ascending=False))
#missing values percent

def missing_values_(terror): 

    missing_value = terror.isnull().sum()

    missing_value_percent = 100 * terror.isnull().sum()/len(terror)

    missing_values_ = pd.concat([missing_value, missing_value_percent], axis=1)

    missing_values_last = missing_values_.rename(

    columns = {0 : 'Missing Values', 1 : '% '})

    return missing_values_last

missing_values_(terror)
terror.dropna(subset=["date"],inplace=True) #the point was the date in this data, to me. So I just delete that 23 rows.

terror['Motive'].fillna(value='nothing', inplace=True) #I did not want to miss any value. So I replaced them with a value that has the same type.

terror['Summary'].fillna(value='nothing', inplace=True)

terror['city'].fillna(value='Unknown', inplace=True)

terror['Target'].fillna(value='Unknown', inplace=True)

terror["Killed"].fillna(terror["Killed"].mean(), inplace=True)

terror["Wounded"].fillna(terror["Wounded"].mean(), inplace=True)



print('which country has the most terrorist attacks? :',terror['Country'].value_counts().index[0])

print('which region has the most terrorist attacks?:',terror['Region'].value_counts().index[0])

print('Maximum people killed in an attack are:',terror['Killed'].max(),'that took place in',terror.loc[terror['Killed'].idxmax()].Country)

print('How many people wounded in last 50 years due the Terrorist Attacks :',terror['Wounded'].sum())

print('How many people died in last 50 years due the Terrorist Attacks :',terror['Killed'].apply(np.floor).sum())



#pp.ProfileReport(terror) # I just learned why it dosen't work in kaggle. There is some version issues. But it works on Jypter Notebook. 
#boxplots



title_style = {'family': 'Century Gothic', 'color': 'green', 'size': 20 }

axis_style  = {'family': 'Century Gothic', 'color': 'darkblue', 'size': 25}

values = {'AttackType':1,'Target_type':2, 'day_of_week':3,   'Region':4,   'AttackType':5,   'Weapon_type':6 }

plt.figure(figsize=(15,45))



for value, i in values.items():

    plt.subplot(6,1,i)

    sns.boxplot(x=value, y="Year", data=terror,

            whis=[0, 100], palette="vlag")

    plt.xticks(rotation = 90)

    plt.title(value ,fontdict = title_style)



plt.show()
a = terror.corr()

mask = np.triu(np.ones_like(a, dtype=bool))

f, ax = plt.subplots(figsize=(11, 9))

cmap = sns.diverging_palette(230, 20, as_cmap=True)

sns.heatmap(a, mask=mask, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})

plt.title("Correlation Matrix (Data)")
plt.figure(figsize=(15,15))

sns.heatmap(a, annot=True, annot_kws={"size": 10}, linewidths=1, vmin=0, vmax=0.6, cmap='viridis')

plt.title("Correlation Matrix (Data)")

plt.show() #let's make it clear.
weapon_cross = pd.crosstab(terror["Weapon_type"], terror["Region"])

weapon_cross


print(stats.chisquare(weapon_cross, axis=None)) #p-value of 0 and test statistic of 1728110 indicate that differences are statistically significant.
plt.figure(figsize=(15,15))

sns.countplot(y="Weapon_type", hue="Region", data=terror)

plt.title("Weapon Types by Regions")

plt.ylabel("Weapon Types")

plt.xlabel("Count")

plt.show()
terror['date'] = pd.to_datetime(terror['date'])

k_by_year = terror.groupby('date')['Killed'].sum().reset_index()

terror['date'] = pd.to_datetime(terror['date'])

k_by_year = k_by_year.set_index('date')
k_by_year.plot(figsize=(15, 6))

plt.title("Number of Deaths due to Terrorism by Years")

plt.show()
rsample_k_by_year = k_by_year['Killed'].resample('MS').mean()

rsample_k_by_year.plot(figsize=(20, 9))

plt.title("The Average Number of Deaths due to Terrorism by Years")

plt.show()
k_by_year_1 = terror.groupby('Year')['Killed'].sum().reset_index()

k_by_year_1 = k_by_year_1.set_index('Year')

k_by_year_1.columns = ['Killed']

k_by_year_1 = k_by_year_1.reset_index()

Total = k_by_year_1['Killed'].sum()

k_by_year_1['Ratio'] = k_by_year_1['Killed']/ Total

k_by_year_1 #It's the long way I know but I did like this though. Not Pythonic that much.
plt.figure(figsize=(25, 10))

sns.barplot(x = 'Year', y = 'Killed', data = k_by_year_1)

plt.title("The Ratio of Yearly Deaths to Total Deaths")

plt.xticks(rotation=90)

plt.show() #but it works right. <3
coun_terror=terror['Country'].value_counts()[:15].to_frame()

coun_terror.columns=['Attacks']

coun_kill=terror.groupby('Country')['Killed'].sum().to_frame()

coun_terror.merge(coun_kill,left_index=True,right_index=True,how='left').plot.bar(width=0.9)

plt.title("The Ratio Between Attacks and Deaths")

fig=plt.gcf()

fig.set_size_inches(18,6)

plt.show() #actually I saw this from another notebook and I liked it. I will use this to the show you something different.
#terror_iq=terror[terror['Country']=='Iraq']

#terror_pk=terror[terror['Country']=='Pakistan']

#terror_sp=terror[terror['Country']=='Spain']





#f,ax=plt.subplots(1,2,figsize=(25,12))

#sns.countplot(y='AttackType',data=terror_pk,ax=ax[0])

#ax[0].set_title('Favorite Attack Types')

#ax[0].set_title('Favorite Attack Types Pakistan')



#sns.countplot(y='AttackType',data=terror_iq,ax=ax[1])

#ax[1].set_title('Favorite Attack Types Iraq')

#plt.subplots_adjust(hspace=0.3,wspace=0.6)

#ax[0].tick_params(labelsize=15)

#ax[1].tick_params(labelsize=15)

#plt.show() #doesn't work because of version
plt.subplots(figsize=(15,6))

sns.countplot(terror['Weapon_type'],palette='inferno',order=terror['Weapon_type'].value_counts().index)

plt.xticks(rotation=90)

plt.title('Favorite Weapons')

plt.show()
general_sum = terror.groupby(by='Region').sum()

a7 = general_sum['Killed']

a7.columns = ['sum']

a7 = a7.reset_index()

a7 



#how many peoople died and where.
plt.figure(figsize=(15,10))

sns.barplot(x = 'Region', y = 'Killed', data = terror)

plt.xticks(rotation = 90)

plt.title('Deaths by Regions')

plt.show()
plt.figure(figsize=(15, 15))

sns.lineplot(x = 'Year', y = 'Killed', data = terror, hue = 'Region')

plt.title('Deaths by Years&Regions')

plt.show()
terror.nlargest(10, ['Killed']) 
plt.figure(figsize=(20,10))

sns.countplot(x="day_of_week", hue="Region", data=terror)

plt.title("Day of Week by Regions")

plt.show()
a1 = terror.groupby(by='day_of_week').agg(['count'])

a2 = a1['Region']

a2.columns = ['count']

a2 = a2.reset_index()

a2 #how many attacks happened.
plt.figure(figsize=(15,10))

sns.barplot(y = 'day_of_week', x = 'count', data = a2)

plt.title("Attacks by Days")

plt.show()

#how many attacks happened.
#top_groups10=terror[terror['Group'].isin(terror['Group'].value_counts()[1:11].index)]

#pd.crosstab(top_groups10.Year,top_groups10.Group).plot(color=sns.color_palette('Paired',10))

#fig=plt.gcf()

#fig.set_size_inches(18,10)

#plt.show() #it doesn't work because of version
terror['Group'].value_counts()[:10].sort_values(ascending=False) #Terrorist groups by the attack numbers
a3 = terror.groupby(by='AttackType').agg(['count'])

a4 = a3['Year']

a4.columns = ['count']

a4 = a4.reset_index()

a4
plt.figure(figsize=(15,10))

sns.barplot(y = 'AttackType', x = 'count', data = a4)

plt.title("The Most Common Attack Types")

plt.show()
plt.subplots(figsize=(15,6))

sns.countplot(terror['Target_type'],palette='inferno',order=terror['Target_type'].value_counts().index)

plt.xticks(rotation=90)

plt.title('Target Groups')

plt.show()
text = terror.Motive.dropna()

text = " ".join(str(motive) for motive in terror.Motive )





# Create stopword list:

stopwords = set(STOPWORDS)

#add additional stopwords

stopwords.update(["say","NaN","specific" ,"carried","incident","responsibility","claimed","noted","minority", "nothing",

                  "party","Party","noted","attack","motive","source","sources","stated","part","new", "us","The", "specific", "motive", "for",

                  "attack", "is", "unknown", "which", "Unknown", "occured","Occured", "state", "reported", "member", "group", "area", "related", "intended",

                  "larger","trend","may","target","says","call","unknown","nan","NAN","majority","communities","victim", "killed" ,"people", "posited", "accused"])



mask = np.array(Image.open("../input/xxxxxx/xxx.png"))



#font = ImageFont.load_default() #font_path= font,

wordcloud_usa = WordCloud(stopwords=stopwords, background_color="white", mode="RGBA",  max_words=100000000, mask=mask).generate(text)



# create coloring from image

image_colors = ImageColorGenerator(mask)

plt.figure(figsize=[15,15])

plt.imshow( wordcloud_usa.recolor(color_func=image_colors),cmap=plt.cm.gist_heat, interpolation="bilinear")

plt.axis("off")

plt.title("What Motivate People to Do This?")

# store to file 

plt.savefig("motive_word.png", format="png")



plt.show()