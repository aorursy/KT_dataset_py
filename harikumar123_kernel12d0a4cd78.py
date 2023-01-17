# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from wordcloud import WordCloud, STOPWORDS 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
## Reading the data in Dataframe



df = pd.read_csv('../input/chennai-zomato-restaurants-data/Chennai_Zomato_Data_updated.csv')

df.head()
## Columns information



df.info()
## Column name



df.columns
print("Number of Rows: {} and Number of Columns: {}".format(df.shape[0],df.shape[1]))
df.isnull().sum()
## Replacing "invalid" string value to NaN and assign to new Dataframe



df1 = df.replace('invalid',np.NaN)
df1.isnull().sum()
## Calculating % data has a missing value compare to total data



for feature in df1.columns:

    if df1[feature].isnull().sum() > 1:

        print(feature,"Has",round((df1[feature].isnull().sum()/df.shape[0])*100,1),"% of missing values")
for feature in df1.columns:

    if df1[feature].isnull().sum() > 1:

        print(feature,"Has",round(df1[feature].isnull().mean()*100,1),"% of missing values")
sns.heatmap(df1.isnull(),yticklabels=False,cbar=False)
null_counts = (df1.isnull().sum()/len(df1))*100

plt.figure(figsize=(16,8))

plt.xticks(np.arange(len(null_counts)),null_counts.index,rotation='vertical')

plt.ylabel('fraction of rows with missing data')

plt.bar(np.arange(len(null_counts)),null_counts)
Loc = df.Location

Exact_loc = []

for i in Loc:

    if i.find(','):

        data = i.split(", ")

        i = data[-1]

        Exact_loc.append(i)

    else:

        Exact_loc.append(i)



df1['Ex_loc'] = Exact_loc

df1
## Unique Location



df1.Ex_loc.unique()
plt.subplots(figsize=[10,40])

ax = sns.countplot(data=df1,y='Ex_loc',)

plt.xticks(rotation=70)

plt.xlabel('Count',size = 20)

plt.ylabel('Area',size = 20)

plt.title('Number of hotels on each area')



total = len(df1['Ex_loc'])

for p in ax.patches:

        percentage = '{:.2f}%'.format(100 * p.get_width()/total)

        x = p.get_x() + p.get_width() + 0.02

        y = p.get_y() + p.get_height()/2

        ax.annotate(percentage, (x, y))



plt.show()
df1 = df1.astype({'Ratings':np.float16})

df1.info()
##Finding average price for each area using Pivot table



table = pd.pivot_table(df1,index=['Ex_loc'],values=['Price for 2', 'Ratings'],

               aggfunc={'Price for 2':[np.mean],'Ratings':[np.mean]},fill_value=0)

table
## Creating a dataframe by using pivot table columns



Area = list(table.index)

Price = list(table['Price for 2','mean'])

Rating = list(table['Ratings','mean'])



Avg_price_area = pd.DataFrame(list(zip(Area,Price,Rating)),columns=['Area','Average_Price','Ratings'])

Avg_price_area
plt.subplots(figsize=[10,40])

ax = sns.barplot(data=Avg_price_area,y='Area',x='Average_Price')

plt.xticks(rotation=70)

plt.xlabel('Average Price',size = 20)

plt.ylabel('Area',size = 20)

plt.title('Average price per area', size=20)



for p,i in zip(ax.patches,Avg_price_area['Average_Price']):

        i = round(i,0)

        i = 'Rs.{:.2f}'.format(i)

        x = p.get_x() + p.get_width() + 0.02

        y = p.get_y() + p.get_height()/2

        ax.annotate(i, (x, y))



plt.show()
## Average rating on each area



plt.subplots(figsize=[10,40])

ax = sns.barplot(data=Avg_price_area,y='Area',x='Ratings')

plt.xticks(rotation=70)

plt.xlabel('Average rating',size = 20)

plt.ylabel('Area',size = 20)

plt.title('Average rating per area', size=20)



for p,i in zip(ax.patches,Avg_price_area['Ratings']):

        i = round(i,1)

        i = '{:.1f}'.format(i)

        x = p.get_x() + p.get_width() + 0.02

        y = p.get_y() + p.get_height()/2

        ax.annotate(i, (x, y))



plt.show()
## Finding Food category from all hotels

food = []



for i in df1.Cuisine:

    str1 = ""

    str1 = (str1.join(i))

    str1 = str1.split(', ')

    food.append(str1)

food
## The Food list contain nested list, So change the nested list to a single list



f_category = []

def reemovNestings(l): 

    for i in l: 

        if type(i) == list: 

            reemovNestings(i) 

        else: 

            f_category.append(i) 

            

for i in food:

    reemovNestings(i)



food_category_df = pd.DataFrame({"Food" : f_category})

f_category
plt.subplots(figsize=[10,40])

ax = sns.countplot(data=food_category_df,y='Food')

plt.xticks(rotation=70)

plt.xlabel('Count',size = 20)

plt.ylabel('Type of the food',size = 20)

plt.title('Food category in chennai',size = 20)



total = len(food_category_df['Food'])

for p in ax.patches:

        percentage = '{:.2f}%'.format(100 * p.get_width()/total)

        x = p.get_x() + p.get_width() + 0.02

        y = p.get_y() + p.get_height()/2

        ax.annotate(percentage, (x, y))



plt.show()
plt.subplots(figsize=[30,8])

sns.countplot(x='Ratings',data=df,order = df['Ratings'].value_counts().index)

plt.xticks(rotation=0,size = 20)

plt.yticks(rotation=0,size = 20)

plt.xlabel('Rating',size = 20)

plt.ylabel('Count',size = 25)

plt.title('Over all customer rating in chennai',size = 29)
##Wordcloud



comment_words = ' '

stopwords = set(STOPWORDS) 

  

# iterate through the csv file 

for val in food_category_df.Food: 

      

    # typecaste each val to string 

    val = str(val) 

  

    # split the value 

    tokens = val.split() 

      

    # Converts each token into lowercase 

    for i in range(len(tokens)): 

        tokens[i] = tokens[i].lower() 

          

    for words in tokens: 

        comment_words = comment_words + words + ' '

  

  

wordcloud = WordCloud(width = 1000, height = 1000, 

                background_color ='white', 

                stopwords = stopwords, 

                min_font_size = 10).generate(comment_words) 

  

# plot the WordCloud image                        

plt.figure(figsize = (50, 20), facecolor = 'black') 

plt.imshow(wordcloud) 

plt.axis("off") 

plt.tight_layout(pad = 0) 

  

plt.show() 