import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt 

#plotly
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode()
import plotly.graph_objs as py

#for word cloud
from subprocess import check_output
from wordcloud import WordCloud, STOPWORDS
stopwords = set(STOPWORDS)

import re
import os
print(os.listdir("../input"))
from IPython.display import HTML
from IPython.display import display

from PIL import Image
from termcolor import colored

from nltk.corpus import stopwords
stop = set(stopwords.words('english'))
stop.update(['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}','']) # remove it if you need punctuation 

from nltk.stem import WordNetLemmatizer

import seaborn as sns

#to supress Warnings 
import warnings
warnings.filterwarnings("ignore")
df = pd.read_csv("../input/zomato-restaurants-data/zomato.csv",encoding='latin-1') #Main restaurants data
ri = pd.read_json("../input/recipe-ingredients-dataset/train.json")          #Cuisines recipe ingredients
fc = pd.read_csv("../input/food-choices/food_coded.csv")                     #Food Choice data of college students
sl = pd.read_csv("../input/store-locations/directory.csv")                   #starbucks data
#df.tail(10)
#word cloud - Cities with maximum Restaurants
maskCity = np.array(Image.open( "../input/beerimage/img_482334.png"))

wordcloud = (WordCloud( max_words=100,width=1440, height=1080,max_font_size=60, min_font_size=10, relative_scaling=0.5,mask=maskCity,background_color='white').generate_from_frequencies(df['City'].value_counts()))

fig = plt.figure(1,figsize=(15, 15))
plt.imshow(wordcloud,interpolation="gaussian")
plt.axis('off')
plt.show()
TopRatings=df.groupby(['Aggregate rating'],as_index = False)['Restaurant ID'].count()

TopRatings.columns = ['Rating','Counts']
Top = TopRatings.sort_values(by='Counts',ascending=False).head(11).reset_index(drop=True)
Top = Top.iloc[1:,:]
plt.figure(figsize=(15,9))
sns.barplot(Top['Rating'],Top['Counts'])
plt.title("Top 10 Ratings with count")
plt.show()
#Word cloud for cuisines -- lets see some popular cusines 
wave_mask = np.array(Image.open( "../input/beerimage/tulip-beer-glass.jpg"))
wordcloud = (WordCloud(width=1440, height=1080, mask = wave_mask, relative_scaling=0.5, stopwords=stopwords, background_color='grey').generate_from_frequencies(df['Cuisines'].value_counts()))


fig = plt.figure(1,figsize=(15, 15))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
cuisines_data = df.groupby(['Cuisines'],as_index = False)['Restaurant ID'].count()

#now time to get most popular cusinies on board, we know North Indian gonna hit the list
cuisines_data.columns = ['Popular Cusinies','Number of Restaurants']
cuisines_data.reindex(axis="index")
cuisines_data.sort_values(by='Number of Restaurants',ascending=False).head(20).reset_index(drop=True)

data1 = [dict(
    type='scattergeo',
    lon = df['Longitude'],
    lat = df['Latitude'],
    text = df['Restaurant Name'],
    mode = 'markers',
    marker = dict(
    cmin = 0,
    color = df['Aggregate rating'],
    cmax = df['Aggregate rating'].max(),
    colorbar=dict(
                title="Ratings"
            )
    )
    
)]
layout = dict(
    title = 'Where are the Resturants',
    hovermode='closest',
    geo = dict(showframe=False, countrywidth=1, showcountries=True,
               showcoastlines=True, showocean = True, projection=dict(type='Mercator'))
)
fig = py.Figure(data=data1, layout=layout)
iplot(fig)
#Data Prep for below visual
a = []
for i in ri["ingredients"]:
    for j in i:
        a.append(j)
commonIng = pd.DataFrame()
commonIng["common"] = pd.Series(a)
#commonIng
#using recipe-ingredients-dataset
maskCity = np.array(Image.open( "../input/beerimage/carlsberg-beer-650ml-mynepshopcom.jpg"))

wordcloud = (WordCloud(width=1440, height=1080, relative_scaling=0.5, stopwords=stopwords,mask=maskCity,max_words=100,max_font_size=60, min_font_size=10,background_color='black').generate_from_frequencies(commonIng["common"].value_counts()))

fig = plt.figure(1,figsize=(15, 15))
plt.imshow(wordcloud,interpolation="bilinear")
plt.axis('off')
plt.show()
#Data prep for below visual
a = []
for i in fc["comfort_food"]:
    i = str(i)
    for j in list(i.split(',')):
        a.append(j)
comfy = pd.DataFrame()
comfy["food"] = pd.Series(a)
#comfy["food"]
#what's my comfort food -> using Food_choices dataset
maskComfy = np.array(Image.open( "../input/beerimage/images.jpg"))

wordcloud = (WordCloud(width=1440, height=1080, relative_scaling=0.5, stopwords=stopwords,mask=maskComfy,max_words=1000,background_color='white').generate_from_frequencies(comfy["food"].value_counts()))

fig = plt.figure(1,figsize=(15, 15))
plt.imshow(wordcloud,interpolation="gaussian")
plt.axis('off')
plt.show()
#main functions to get the comfort food sorted from the day as per the reason
"""
1 – stress
2 – boredom
3 – depression/sadness
4 – hunger
5 – laziness
6 – cold weather
7 – happiness 
8- watching tv
"""

def searchComfy(mood):
    lemmatizer = WordNetLemmatizer()
    foodcount = {}
    for i in range(124):
        temp = [temps.strip().replace('.','').replace(',','').lower() for temps in str(fc["comfort_food_reasons"][i]).split(' ') if temps.strip() not in stop ]
        if mood in temp:
            foodtemp = [lemmatizer.lemmatize(temps.strip().replace('.','').replace(',','').lower()) for temps in str(fc["comfort_food"][i]).split(',') if temps.strip() not in stop ]
            for a in foodtemp:
                if a not in foodcount.keys():
                    foodcount[a] = 1 
                else:
                    foodcount[a] += 1
    sorted_food = []
    #print(sorted([(value,key) for (key,value) in foodcount.items()])) : for test
    sorted_food = sorted(foodcount, key=foodcount.get, reverse=True)
    return sorted_food


def findmycomfortfood(mood):
    topn = []
    topn = searchComfy(mood) #function create dictionary only for particular mood
    print(colored("3 Popular Comfort Foods in %s are:"%(mood),'blue'))
    print(colored(topn[0],'green'))
    print(colored(topn[1],'green'))
    print(colored(topn[2],'green'))
findmycomfortfood('stress')
#GPA Data Prep
dfc = fc
dfc = dfc.drop(dfc.index[15])
dfc = dfc.drop(dfc.index[60])
dfc = dfc.drop(dfc.index[100])
dfc = dfc.drop(dfc.index[101])
dfc.reset_index(inplace=True)
dfc["GPA"][71]= str(dfc["GPA"][71]).replace(' bitch','')
dfc["GPA"] = pd.to_numeric(dfc["GPA"])
#main function to plot definition
#coffee: 1-Creamy Frappuccino & 2-Espresso; breakfast: 1-cereal option & 2 – donut option; 
#soup: 1 – veggie soup and 2 – creamy soup; drink: 1 – orange juice and 2 – soda 
from itertools import cycle
cycol = cycle('kbgc')

def barGPA(category,label1, label2):
    barGPA = {} 
    barGPA[label1] = dfc[dfc[category]==1]["GPA"].mean() 
    barGPA[label2] = dfc[dfc[category]==2]["GPA"].mean()
    D = barGPA
    plt.bar(range(len(D)), list(D.values()), align='center',color=next(cycol))
    plt.xticks(range(len(D)), list(D.keys()))
    plt.ylim(ymin=2) 
    plt.yticks(np.arange(2, 3.6, 0.1))
    plt.ylabel("GPA")
    plt.title(category.upper()+" vs GPA")
    plt.grid()
#What can effect your GPA, comparing Average GPA against the choices Students made when asked->
fig = plt.figure(figsize=(12,9))

plt.subplot(2, 2, 1)
barGPA('coffee','Frappuccino','Espresso')

plt.subplot(2, 2, 2)
barGPA('drink','Orange Juice','Soda')


plt.subplot(2, 2, 3)
barGPA('breakfast','Cereal','Donut')

plt.subplot(2, 2, 4)
barGPA('soup','Veggie Soup','Creamy Soup')

plt.show()
#Data prep encoding price range!
dfcopy = df
dfcopy["Price range"] = dfcopy["Price range"].astype(str)
dfcopy["Price range"] = dfcopy["Price range"].str.replace('1', '₹').replace('2', '₹₹').replace('3', '₹₹₹').replace('4', '₹₹₹₹')
plt.figure(figsize=(15,9))
plt.title("Distribution of Price Ranges")
#plt.bar(x=[1,2,3,4],height=[2,4,8,16],hatch='x',edgecolor=['black']*6)
sns.countplot(dfcopy["Price range"],order=['₹','₹₹','₹₹₹','₹₹₹₹'],hatch='x')
plt.xlabel("Price Range (low to high)",color='green')
plt.ylabel("No. of Restaurants",color='green')
plt.show()
#plotting
g = sns.jointplot(x='Aggregate rating', y ='Votes' , data = df,kind='scatter',color='blue')
_ = g.ax_marg_x.hist(df['Aggregate rating'], color="g", alpha=.6)
_ = g.ax_marg_y.hist(df['Votes'], color="y",alpha=.6,
              orientation="horizontal")
#data prep
"""
1 -Single 
2 - In a relationship 
3 - Cohabiting 
4 - Married 
5 - Divorced 
6 - Widowed
"""
dfc = fc
dfc = dfc.dropna(subset=['marital_status','eating_out'])
dfc['marital_status']=dfc['marital_status'].astype(str)
dfc['marital_status'] = dfc['marital_status'].str.replace('1.0','Single')\
    .replace('2.0','In a relationship').replace('4.0','Married')  #since only 3 kind of Marital status available in data
AvgEatOut = dfc.groupby('marital_status')['eating_out'].mean()

#plotting
plt.figure(figsize=(15,9))
plt.bar(x=AvgEatOut.index, height =AvgEatOut.values , color='white',edgecolor=['red']*3,hatch='*')
plt.xlabel('Marital Status of Students (as per Data)',color='green')
plt.ylabel('Freq. of Eating Out (Average)',color='green')
plt.title("Eating Out Frequency as per Marital Status of Students",color='grey')
plt.show()
data1 = [dict(
    type='scattergeo',
    lon = sl['Longitude'],
    lat = sl['Latitude'],
    text = sl['Store Name'],
    mode = 'markers',
    marker = dict(
    color = '#6f4e37',
    )          
    
)]
layout = dict(
    title = 'Where is that Starbucks?',titlefont=dict(color='green'), 
    hovermode='closest',
    geo = dict(showframe=False, countrywidth=1, showcountries=True,showocean=True,showland=True,countrycolor='green', 
               showcoastlines=True, projection=dict(type='natural earth')),
    
)
fig = py.Figure(data=data1, layout=layout)
iplot(fig)
#Checking the Country Code: Indian Country Code is 1
CountryInfo = pd.read_excel('../input/zomato-restaurants-data/Country-Code.xlsx')
#CountryInfo.head()
#taking out Indian Data into new Data frame, which can be reused later
df_india = df[df['Country Code']==1]

TopCities = df_india.groupby(['City'],as_index = False)['Restaurant ID'].count()
TopCities.columns = ['Cities','Number of Restaurants']
Top10 = TopCities.sort_values(by='Number of Restaurants',ascending=False).head(10)


plt.figure(figsize=(20,10))
plt.bar(Top10['Cities'],Top10['Number of Restaurants'], color ='green')
plt.xlabel('Cities')
plt.ylabel('Number of Restaurants')
plt.title('Indian Cities with Maximum Restaurants', fontweight="bold")
plt.show()
#lets see which restaurant got the most shout-out 
#Grey color function
def grey_color_func(word, font_size, position, orientation, random_state=None,
                    **kwargs):
    return "hsl(0, 0%%, %d%%)" % np.random.randint(60, 100)

#Main WordCloud Code
wc = (WordCloud(width=1440, height=1080, relative_scaling=0.5, stopwords=stopwords).generate_from_frequencies(df_india['Restaurant Name'].value_counts()))

fig = plt.figure(1,figsize=(15, 15))
plt.imshow(wc.recolor(color_func=grey_color_func, random_state=3),
           interpolation="bilinear")
plt.axis('off')
plt.show()
df["Rating color"].value_counts()
Color_represents = df_india.groupby(['Rating color'],as_index = False)['Aggregate rating'].mean()
Color_represents.columns = ['Rating Color','Average Rating']
Color_represents =Color_represents.sort_values(by='Average Rating',ascending=False)
Color_represents = Color_represents[0:5]
Color_represents['Ratings']  = ['1.Excellent','2.Very Good','3.Good','4.Okay','5.Poor']
#Color_represents
#Plotting
plt.figure(figsize=(10,5))
a = ['#006400','green','yellow','Orange','Red']
e = ['blue']*5

plt.barh(Color_represents['Ratings'],Color_represents['Average Rating'], align='center', color =a, edgecolor = e, linewidth = 3)
plt.gca().invert_yaxis()
plt.xlabel('Average Rating')
plt.ylabel('Rating Types')
plt.title('Rating Color and the Average Rating they represent', fontweight="bold")
plt.show()
pd.set_option('display.max_columns', None)   #to display data fully without truncating some columns
df.head(1)
#df['Votes'].mean()
#df['Average Cost for two'].mean()
# Q. Okay I need some cheap Restaurants, say I have 500 rupees, but i need Restaurant with good Aggregate rating >=4.0 and Average votes (156), I would like to eat some North Indian food; I am in New Delhi, India
tempDF = df_india[(df_india['Votes']>=156) & (df_india['Aggregate rating'] >= 4.0) & (df_india['Average Cost for two'] <= 500 ) & (df_india['City'] == 'New Delhi') & (df_india['Cuisines']=="North Indian")]
rn = str(tempDF['Restaurant Name'].values[0])
ra = str(tempDF['Address'].values[0])
print(colored("Ans1. ",'green')+colored(rn,'blue')+", Address: "+ra)
#Ans. There you go my pal, Lets hit Jai Vaishno Rasoi
#Okay Dude, I live in Pune, India. I want to have Chinese Cuisine and I prefer locality to be Viman Nagar, I can spend approax 1000 for two, I wanted Restaurant rated >= 4.0
tempDF = df_india[((df_india['Locality']=='Koregaon Park') & (df_india['Aggregate rating'] >= 4.0) & (df_india['Average Cost for two'] <= 1000 ) & (df_india['City'] == 'Pune'))]
rn = str(tempDF['Restaurant Name'].values[0])
ra = str(tempDF['Address'].values[0])
print(colored("Ans2. ",'green')+colored(rn,'blue')+", Address: "+ra)
#Ans. Is that what you need, if not try-again!
