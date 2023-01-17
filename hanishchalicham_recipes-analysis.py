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
import matplotlib.pyplot as plt

from wordcloud import WordCloud, STOPWORDS

import seaborn as sns
fooddata=pd.read_csv('/kaggle/input/indian-food-101/indian_food.csv')

fooddata
fooddata.info()


fooddata.replace("-1", np.NaN, inplace = True)

fooddata.nunique()
fooddata.region.value_counts()

fooddata.state.value_counts()
diettype = fooddata.diet.value_counts().reset_index()

plt.figure(figsize=(7,7))

plt.pie(diettype.diet, labels = diettype['index'],autopct='%0.2f%%',shadow=True)

plt.title("Vegetarian vs Non-Vegetarian")

plt.show()
courseregion=fooddata.pivot_table(values='name',index=['region'],columns='course', aggfunc = 'count')

courseregion.plot(kind='bar')

plt.title("Course vs Region")
words = '' 

stopwords = set(STOPWORDS) 

  

for val in fooddata.state:       

    val = str(val)   

    tokens = val.split()      

    for i in range(len(tokens)): 

        tokens[i] = tokens[i].lower() 

      

    words += " ".join(tokens)+" "

  

wordcloud = WordCloud(width = 600, height = 600, 

                background_color ='white', 

                stopwords = stopwords, 

                min_font_size = 8).generate(words)                        

plt.figure(figsize = (5,5), facecolor = None) 

plt.imshow(wordcloud) 

plt.axis("off") 

plt.tight_layout(pad = 0) 

  

plt.show() 
flavorregion=fooddata.pivot_table(values='name',index=['region'],columns='flavor_profile', aggfunc = 'count')

flavorregion.plot(kind='bar')

plt.title("flavor vs region")
sns.countplot(x=fooddata['course'])

plt.title("Number of courses count")
sns.countplot(x=fooddata['flavor_profile'])

plt.title("Flavor profile count")
Time=fooddata.pivot_table(values=['prep_time','cook_time'],index='course',aggfunc=np.mean)

Time.plot(kind='bar',stacked=True,color=['black','purple'])

plt.title("Prep_time and cook_time")


fooddata['total_time'] = fooddata.prep_time + fooddata.cook_time

fooddata.sort_values('total_time',ascending = True).tail()[['name','course','total_time']]
word_list = ''

for word in fooddata.ingredients:

    splited = word.lower()

    word_list +=splited

    

wordcloud = WordCloud(width=800,height=800,background_color='white',min_font_size=4).generate(word_list)

plt.figure(figsize = (10, 5), facecolor = None) 

plt.imshow(wordcloud) 

plt.axis("off") 

plt.tight_layout(pad = -1) 

  

plt.show()
course_ing_count = (fooddata[fooddata['course'] == "dessert"][['name','ingredients']])

def ingredient_count(column):

    return len(column.split(","))

course_ing_count['ingredient_count'] = course_ing_count['ingredients'].apply(ingredient_count)

course_ing_count.sort_values('ingredient_count', ascending = True).tail()
statelist = fooddata[fooddata['state']=='Kerala'].reset_index(drop=True)

statelist
fooddata[(fooddata['diet'] == 'vegetarian') & (fooddata['course'] == 'dessert')].sort_values("total_time", ascending= False).head()
fooddata[(fooddata['diet'] == 'non vegetarian') & (fooddata['course'] == 'main course')].sort_values("total_time", ascending= True).tail()