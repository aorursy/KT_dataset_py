import numpy as np # linear algebra
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn.linear_model as skl_lm
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, LeaveOneOut, KFold, cross_val_score
from sklearn.preprocessing import PolynomialFeatures

from sklearn.preprocessing import scale
import statsmodels.api as sm
import statsmodels.formula.api as smf

%matplotlib inline
plt.style.use('seaborn-white')
cav = pd.read_csv("../input/CAvideos.csv")
dev = pd.read_csv("../input/DEvideos.csv")
frv = pd.read_csv("../input/FRvideos.csv")
gbv = pd.read_csv("../input/GBvideos.csv")
usv = pd.read_csv("../input/USvideos.csv")
cav.head()
frv.info()
usv.head()

usv.info()
usv.describe()
esVLD = smf.ols('views ~ likes + dislikes', usv).fit()
esVLD.summary()

sns.jointplot(x='views', y='likes', 
              data=usv, color ='red', kind ='reg', 
              size = 8.0)
plt.show()
import json


usv['category_id'] = usv['category_id'].astype(str)
# usv_cat_name['category_id'] = usv['category_id'].astype(str)

category_id = {}

with open('../input/US_category_id.json', 'r') as f:
    data = json.load(f)
    for category in data['items']:
        category_id[category['id']] = category['snippet']['title']

usv.insert(4, 'category', usv['category_id'].map(category_id))
# usv_cat_name.insert(4, 'category', usv_cat_name['category_id'].map(category_id))
category_list = usv['category'].unique()
category_list
# labels = usv.groupby(['category_id']).count().index
labels = category_list
trends  = usv.groupby(['category_id']).count()['title']
explode = (0, 0, 0.2, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0 ,0)  # only "explode" the 2nd slice (i.e. 'Hogs')

fig, ax = plt.subplots(figsize=(10,10))
ax.pie(trends, labels=labels, autopct='%1.1f%%',explode = explode,
        shadow=False, startangle=180)
ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
trends

plt.style.use('ggplot')
plt.figure(figsize=(20,10))

phour=usv.groupby("category").count()["comment_count"].plot.bar()
phour.set_xticklabels(phour.get_xticklabels(),rotation=45)
plt.title("Comment count vs category of Videos")
sns.set_context()
import glob
files = [file for file in glob.glob('../input/*.{}'.format('csv'))]
sorted(files)
ycd_initial = list()
for csv in files:
    ycd_partial = pd.read_csv(csv)
    ycd_partial['country'] = csv[9:11] #Adding the new column as "country"
    ycd_initial.append(ycd_partial)

ycd = pd.concat(ycd_initial)
ycd.info()
ycd.head()
ycd.apply(lambda x: sum(x.isnull()))
column_list=[] 
# Exclude Description column of description because many YouTubers don't include anything in description. This is important to not accidentally delete those values. This for loop will display existing columns in given dataset.
for column in ycd.columns:
    if column not in ["description"]:
        column_list.append(column)
print(column_list)
ycd.dropna(subset=column_list, inplace=True) 
# Drop NA values
ycd.head()
# Feature engineering

#Adjusting Date and Time format in right way
ycd["trending_date"]=pd.to_datetime(ycd["trending_date"],errors='coerce',format="%y.%d.%m")
ycd["publish_time"]=pd.to_datetime(ycd["publish_time"],errors='coerce')
#Create some New columns which will help us to dig more into this data.
ycd["T_Year"]=ycd["trending_date"].apply(lambda time:time.year).astype(int)
ycd["T_Month"]=ycd["trending_date"].apply(lambda time:time.month).astype(int)
ycd["T_Day"]=ycd["trending_date"].apply(lambda time:time.day).astype(int)
ycd["T_Day_in_week"]=ycd["trending_date"].apply(lambda time:time.dayofweek).astype(int)
ycd["P_Year"]=ycd["publish_time"].apply(lambda time:time.year).astype(int)
ycd["P_Month"]=ycd["publish_time"].apply(lambda time:time.month).astype(int)
ycd["P_Day"]=ycd["publish_time"].apply(lambda time:time.day).astype(int)
ycd["P_Day_in_Week"]=ycd["publish_time"].apply(lambda time:time.dayofweek).astype(int)
ycd["P_Hour"]=ycd["publish_time"].apply(lambda time:time.hour).astype(int)

plt.figure(figsize = (15,10))
ycd.describe()
sns.heatmap(ycd[["views", "likes","dislikes","comment_count"]].corr(), annot=True)
plt.show()
category_from_json={}
with open("../input/US_category_id.json","r") as file:
    data=json.load(file)
    for category in data["items"]:
        category_from_json[category["id"]]=category["snippet"]["title"]
        
        
list1=["views likes dislikes comment_count".split()] 
for column in list1:
    ycd[column]=ycd[column].astype(int)
#Similarly Convert The Category_id into String,because later we're going to map it with data extracted from json file    
list2=["category_id"] 
for column in list2:
    ycd[column]=ycd[column].astype(str)


from collections import OrderedDict

ycd["Category"]=ycd["category_id"].map(category_from_json)

ycd.groupby(["Category","country"]).count()["video_id"].unstack().plot.barh(figsize=(20,10), stacked=True, cmap = "inferno")
plt.yticks(rotation=0, fontsize=20) 
plt.xticks(rotation=0, fontsize=20) 
plt.title("Category analysis with respect to countries", fontsize=20)
plt.legend(handlelength=5, fontsize  = 10)
plt.show()
def trend_plot(country):
    ycd[ycd["country"] == country][["video_id", "trending_date"]].groupby('video_id').count().sort_values\
    (by="trending_date",ascending=False).plot.kde(figsize=(15,10), cmap = "rainbow")
    plt.yticks(fontsize=18) 
    plt.xticks(fontsize=15) 
    plt.title("\nYouTube trend in "+ country +"\n", fontsize=25)
    plt.legend(handlelength=2, fontsize  = 20)
    plt.show()
#country_list = df.groupby(['country']).count().index
country_list = ["FR", "CA", "GB","US","DE"]
for country in country_list:
    trend_plot(country)
from wordcloud import WordCloud
import nltk
#nltk.download("all")
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk import sent_tokenize, word_tokenize
from wordcloud import WordCloud, STOPWORDS
from collections import Counter
from nltk.tokenize import RegexpTokenizer
import re
def get_cleaned_data(tag_words):
    #Removes punctuation,numbers and returns list of words
    cleaned_data_set=[]
    cleaned_tag_words = re.sub('[^A-Za-z]+', ' ', tag_words)
    word_tokens = word_tokenize(cleaned_tag_words)
    filtered_sentence = [w for w in word_tokens if not w in en_stopwords]
    without_single_chr = [word for word in filtered_sentence if len(word) > 2]
    cleaned_data_set = [word for word in without_single_chr if not word.isdigit()]  
    return cleaned_data_set
MAX_N = 1000
#Collect all the related stopwords.
en_stopwords = nltk.corpus.stopwords.words('english')
de_stopwords = nltk.corpus.stopwords.words('german')
fr_stopwords = nltk.corpus.stopwords.words('french')   
en_stopwords.extend(de_stopwords)
en_stopwords.extend(fr_stopwords)

def word_cloud(category):
    tag_words = ycd[ycd['Category']== category]['tags'].str.lower().str.cat(sep=' ')
    temp_cleaned_data_set = get_cleaned_data(tag_words) #get_cleaned_data() defined above.
    
    #Lets plot the word cloud.
    plt.figure(figsize = (20,15))
    cloud = WordCloud(background_color = "white", max_words = 200,  max_font_size = 30)
    cloud.generate(' '.join(temp_cleaned_data_set))
    plt.imshow(cloud)
    plt.axis('off')
    plt.title("\nWord cloud for " + category + "\n", fontsize=40)
category_list = ["Music", "Entertainment","News & Politics"]
for category in category_list:
    word_cloud(category)
def best_publish_time(list, title):
    plt.style.use('ggplot')
    plt.figure(figsize=(16,8))
    #list3=df1.groupby("Publish_Hour").count()["Category"].plot.bar()
    list_temp = list.plot.bar()
    #list3.set_xticklabels(list3.get_xticklabels(),rotation=30, fontsize=15)
    list_temp.set_xticklabels(list_temp.get_xticklabels(),rotation=0, fontsize=15)
    plt.title(title, fontsize=25)
    plt.xlabel(s="Best Publishing hour", fontsize=20)
    sns.set_context(font_scale=1)
list = ycd[ycd['country'] == 'US'].groupby("P_Hour").count()["Category"]
title = "\nBest Publish Time for USA\n"
best_publish_time(list, title)

