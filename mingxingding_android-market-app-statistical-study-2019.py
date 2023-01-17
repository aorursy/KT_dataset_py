import numpy as np 
import pandas as pd

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

df_google=pd.read_csv('/kaggle/input/google-play-store-apps/googleplaystore.csv')
print('The shape of data set is: ',df_google.shape)
df_google.head(3)

# drop several columns which will not be useful for this analysis
# because in this study, I'm going to see principlely the category items, so the Genres column will be removed, too

df_google=df_google.drop(['Last Updated','Current Ver', 'Android Ver','Genres','Reviews'],axis=1)
df_google.head(2)
# check the types of attributes

df_google.dtypes
# check the missing value in dataset

df_google.isnull().sum(axis=0)
# because the missing Rating will not influence the statistics study of category, type..., I'll leave them like this at this moment
# check the types in content rating
df_google['Content Rating'].value_counts()
# in this study, I'll just study the APP available for all the public, I'll remove the other patterns of the Content Rating

df_google=df_google[df_google['Content Rating']!='Teen']
df_google=df_google[df_google['Content Rating']!='Mature 17+']
df_google=df_google[df_google['Content Rating']!='Everyone 10+']
df_google=df_google[df_google['Content Rating']!='Adults only 18+']
df_google=df_google[df_google['Content Rating']!='Unrated']
df_google[df_google['Content Rating']!='Everyone']
# from the above result, we can see the only one Null value in Concent Rating columns concerns a row incorrectly filled, remove this row

df_google.drop([10472],inplace=True)
df_google.isnull().sum(axis=0)
print('The shape of data set: ',df_google.shape)
print('The unique App number in dataset: ',df_google['App'].nunique())
# obviously, there are repetition of App names in this table, see what's that

df_countApp=df_google.groupby(by=['App']).count()
df_countApp=df_countApp[df_countApp['Category']!=1]
df_countApp.index.resetname={'App'}
df_countApp.head()
# check one App with repetition rows and see how is it

df_google[df_google['App']=='1800 Contacts - Lens Store']
# by checking several Apps who exists in differents rows, it seem they have no difference in
# the resting features, I'm going to keep just one row for these Apps repeated

list_i=[]
for APP in df_countApp.index:
    df_provi=df_google[df_google['App']==APP]
    for i in df_provi.index:
        if i>df_provi.index.min():
            list_i.append(i)     

for j in list_i:
    df_google.drop([j],inplace=True)

print('Now the shape of df_google :',df_google.shape)
print('And the unique App name in the table :',df_google['App'].nunique())
df_google.head(3)
print('The number of distinct category is: ',df_google['Category'].nunique())
# show all the category titles:

df_google['Category'].unique()
# get the top 15 category

df_categorycount=pd.DataFrame(df_google.groupby('Category').count()['App'])
df_category15=df_categorycount.sort_values(by=['App'],ascending=False).head(15)
df_category15
# import libraries and functions for plotting 

import matplotlib as mtl
import matplotlib.pyplot as plt
%matplotlib inline
import sklearn as sk
import seaborn as sns
from matplotlib import cm
mtl.cm.get_cmap
color=['darkorange','limegreen','royalblue','orange','springgreen','slateblue','maroon',
       'gold','seagreen','turquoise','blueviolet','tomato','olive','deepskyblue','orchid']

x=df_category15.index
width=0.75
ymax=df_category15['App'].max()+200

fig,ax=plt.subplots(figsize=(10,7))
bars=ax.bar(x,df_category15['App'],width,label=None,color=color)

ax.set_ylabel('App number',fontsize=14)
ax.set_title('Top 15 categories in GooglePlay',fontsize=17,weight='bold')
ax.set_xticklabels(x,rotation=90)
ax.set_ylim([0,ymax])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_color('grey')
ax.tick_params(bottom=False, left=False)

ax.set_axisbelow(True)
ax.yaxis.grid(True, color='lightgray')
ax.xaxis.grid(False)

def autolabel(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate('{}'.format(height),
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), 
                    textcoords="offset points",
                    ha='center', va='bottom')
        
autolabel(bars)

plt.show()
# It maybe also interesting to see which category has less App available, where is potiential to occupy the market earlier.
# we'll see the last 5

df_categorylast5=df_categorycount.sort_values(by=['App'],ascending=False).tail(5)
df_categorylast5


# because the value in Installs columns is in string type, I'm going to change them to int type by elimiting the + 
# but it should take into consideration that every value has a sense of more than this value

df_google['Installs'].value_counts().index
df_google['Installs'].replace(['1,000,000+', '100,000+', '10,000+', '1,000+', '10,000,000+', '100+',
       '5,000,000+', '5,000+', '50,000+', '500,000+', '10+', '500+', '50+',
       '50,000,000+', '100,000,000+', '5+', '1+', '500,000,000+', '0+',
       '1,000,000,000+'],[1000000, 100000, 10000, 1000, 10000000, 100,
       5000000, 5000, 50000, 500000, 10, 500, 50,
       50000000, 100000000, 5, 1, 500000000, 0,
       1000000000],inplace=True)

df_installrank=df_google.sort_values(by=['Installs'],ascending=False)
df_installrank.head(2)
# see how many App has a large amount of installs 

df_installrank['Installs'].value_counts()
# the value counts show there are 11 App having over 1,000,000,000,000 installs
# and there are 19 App having over 500,000,000 installs
# and I'm going to extrait the name of these Apps

df_installsb=df_google[df_google['Installs']==1000000000]
print('The Apps of whom the installs is over 1 billion:',df_installsb['App'].tolist())

print('')

df_installs500m=df_google[df_google['Installs']==500000000]
print('The Apps of whom the installs is over 500 million:',df_installs500m['App'].tolist())
# I'll see which category has that numerous installs

df_installsb_cat=pd.DataFrame(df_installsb['Category'].value_counts())
df_installs500m_cat=pd.DataFrame(df_installs500m['Category'].value_counts())
color=['maroon','seagreen','blueviolet','tomato','orchid']

x=df_installsb_cat.index
width=0.55


fig,ax1=plt.subplots(figsize=(5,2.5))
bars1=ax1.bar(x,df_installsb_cat['Category'],width,label=None,color=color)

ax1.set_title('App with over 1 billion installs',fontsize=13,weight='bold',pad=14)
ax1.set_xticklabels(x,rotation=90)

ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['left'].set_visible(False)
ax1.spines['bottom'].set_color('grey')
ax1.tick_params(bottom=False, left=False)

ax1.set_axisbelow(True)
ax1.yaxis.grid(True, color='lightgray')
ax1.xaxis.grid(False)

def autolabel(bars):
    for bar in bars:
        height = bar.get_height()
        ax1.annotate('{}'.format(height),
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0,-15), 
                    textcoords="offset points",
                    ha='center', va='bottom',color='white',weight='bold')
        
autolabel(bars1)

plt.show()
color=['turquoise','blueviolet','tomato','olive','deepskyblue','orchid']

x=df_installs500m_cat.index
width=0.7


fig,ax2=plt.subplots(figsize=(5,2.5))
bars2=ax2.bar(x,df_installs500m_cat['Category'],width,label=None,color=color)

ax2.set_title('App with over 500 million installs',fontsize=13,weight='bold',pad=14)
ax2.set_xticklabels(x,rotation=90)

ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax2.spines['bottom'].set_color('grey')
ax2.tick_params(bottom=False, left=False)

ax2.set_axisbelow(True)
ax2.yaxis.grid(True, color='lightgray')
ax2.xaxis.grid(False)

def autolabel(bars):
    for bar in bars:
        height = bar.get_height()
        ax2.annotate('{}'.format(height),
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0,-15), 
                    textcoords="offset points",
                    ha='center', va='bottom',color='white')
        
autolabel(bars2)

plt.show()
import numpy as np
# According to a test not showed here, the rating data focus mainly in the range of 3 - 5, 
# so the following study classe in detail the Apps in that range

df_google['Rating3_5'] = pd.Series()

for j in df_google.index:
    if (df_google.loc[j,'Rating']>4.5)&(df_google.loc[j,'Rating']<=5.0):
        df_google.loc[j,'Rating3_5']='4.5-5',
    elif (df_google.loc[j,'Rating']>4.0)&(df_google.loc[j,'Rating']<=4.5):
        df_google.loc[j,'Rating3_5']='4-4.5',
    elif (df_google.loc[j,'Rating']>3.5)&(df_google.loc[j,'Rating']<=4.0):
        df_google.loc[j,'Rating3_5']='3.5-4',
    elif (df_google.loc[j,'Rating']>3.0)&(df_google.loc[j,'Rating']<=3.5):
        df_google.loc[j,'Rating3_5']='3-3.5',
    elif (df_google.loc[j,'Rating']>0.0)&(df_google.loc[j,'Rating']<=3.0):
        df_google.loc[j,'Rating3_5']='<=3'

df_google['Rating3_5'] = df_google['Rating3_5'].replace(np.nan, 'NoRatingInfo')                
df_google.tail(3)
df_rating3_5=df_google.groupby(by=['Rating3_5']).count()[['App']]
df_rating3_5
# I'm going to remove the NoRatingInfo row and create a pie chart of the distribution of rating

df_rating3_5=df_rating3_5.drop(["NoRatingInfo"])
df_rating3_5=df_rating3_5.reindex(['4.5-5','4-4.5','3.5-4','3-3.5','<=3'])
# after removing the NoRatingInfo rows, the following pie chart is based on the rest 5195 Apps

explode_list = [0, 0, 0, 0, 0]

df_rating3_5['App'].plot(kind='pie',
                        figsize=(8,6),
                        startangle=90,
                        autopct='%1.1f%%',
                        labels=None,
                        explode=explode_list,
                        textprops=dict(color="w",weight="bold"))

plt.title(r'Rating distribution (based on 5195 Apps)',fontsize=16,weight='bold', pad=20)
plt.ylabel('')
plt.legend(labels=df_rating3_5.index, loc='upper left') 
plt.axis('equal')

plt.show()
# find the Apps which have larges quantities installs and high rating
# according to test, there are no element rated 4.5 - 5 and having over 1 billion installs; I'll try installs = 500000000

df_rat_insm=df_google[df_google['Rating3_5']=='4.5-5']
df_rat_insm=df_rat_insm[df_rat_insm['Installs']==500000000]
print('There are {} Apps who have over 500 million installs and a rating between 4.5 - 5'.format(df_rat_insm.shape[0]))
df_rat_insm
# And

df_rat_insb=df_google[df_google['Rating3_5']=='4-4.5']
df_rat_insb=df_rat_insb[df_rat_insb['Installs']==1000000000]
print('There are {} Apps who have over 1 billion installs and a rating between 4 - 4.5'.format(df_rat_insb.shape[0]))
df_rat_insb.sort_values(['Rating'],ascending=False)
# At first, how many App are paid for each category?

df_typeini=df_google[['Category','Type']]
df_typeini['Type'].replace(['Free','Paid'],[0,1],inplace=True)
df_typeini=df_typeini.groupby(['Category']).sum()
df_typeini.head()
# remove the dollar symble of the price and turn the column into float ty^pe

df_google['Price']=df_google['Price'].replace( '[\$,)]','', regex=True).astype(float)
df_typeini['Paid App perc']=pd.Series()
df_typeini['Average Price']=pd.Series()
# calculate the paid App percentage of each category and their average price (for those paid) respectively

for category in df_typeini.index:
    df_typeini.loc[category,'Paid App perc']=(df_typeini.loc[category,'Type'])/(df_google[df_google['Category']==category]['Type'].count())
    df_typeini.loc[category,'Average Price']=df_google[(df_google['Category']==category) & (df_google['Type']=='Paid')]['Price'].mean()
df_typeini.head()
# calculate the average paid app percentage and compare it to the percentage per categary in the plot below

average_paid_perc=(df_google[df_google['Type']=='Paid']['Price'].count())/(df_google['Price'].count())
average_paid_perc
# plot a scatter schema

x=df_typeini.index
y=df_typeini['Paid App perc']
plt.figure(figsize=(12,5))

plt.scatter(x,y,color='blue')
plt.axhline(y=0.083, color='r', linestyle='--')

plt.annotate('0.083',
            xy=(0,0.09),
            ha='center',
            va='bottom',
            fontsize=14,
            color='r')

plt.xticks(rotation=90, fontsize=12)
plt.ylabel('Paid App percentage', fontsize=14)
plt.yticks(fontsize=12)
plt.title('Paid App percentage per category VS average paid App percentage',pad=10, fontsize=15,weight='bold')

plt.grid(alpha=0.4)

plt.show()
# calculate the average price of all paid Apps 

average_price=(df_google['Price'].sum())/(df_google[df_google['Type']=='Paid']['Price'].count())
average_price
x=df_typeini.index
y=df_typeini['Average Price']
plt.figure(figsize=(12,5))

plt.bar(x,y,color='purple')
plt.axhline(y=14.84, color='r', linestyle='--')

plt.annotate('14.84',
            xy=(0,18),
            ha='center',
            va='bottom',
            fontsize=14,
            color='r')

plt.annotate('109.99',
            xy=(10,82),
            rotation=90,
            ha='center',
            va='bottom',
            fontsize=12,
            color='w')

plt.annotate('156.3',
            xy=(12,133),
            rotation=90,
            ha='center',
            va='bottom',
            fontsize=12,
            color='w')

plt.annotate('147.06',
            xy=(18,119),
            rotation=90,
            ha='center',
            va='bottom',
            fontsize=12,
            color='w')

plt.xticks(rotation=90,fontsize=12)
plt.ylabel('Average Price (for paid Apps and in dollar)',fontsize=13)
plt.yticks(fontsize=12)
plt.title('Average price of paid App per category VS average price of all paid App',pad=10, fontsize=15, weight='bold')

plt.grid(alpha=0.4)

plt.show()
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
df_review=pd.read_csv('/kaggle/input/google-play-store-apps/googleplaystore_user_reviews.csv')
df_review.head()

df_review.isnull().sum()
print('Shape of table: ',df_review.shape)
print('Number of distinct App: ',df_review['App'].nunique())
# It seems that there are rows without any information of all the columns besides 'App'
# I'm going to remove these rows

df_review.dropna(inplace=True)
print('Shape of table: ',df_review.shape)
print('Number of distinct App: ',df_review['App'].nunique())
print('')
print('The' 'Isnull' 'cells: \n',df_review.isnull().sum())
df_review.head(3)
# import libraries and functions

!conda install -c conda-forge wordcloud --yes
print('Wordcloud installed!')
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS
# from the study above, the top 3 categories are: Family, Tools, Game
# I'm going to previlege the rating at first and then the installs for choosing sample
# Family
df_family=df_google[df_google['Category']=='FAMILY']
df_family=df_family[['App','Rating','Installs','Rating3_5']]
df_family[df_family['Rating3_5']=='4.5-5'].sort_values(['Installs'],ascending=False).head(1)
# Because not all the App in df_google exist in df_review, after several try,I found 'Build a Bridge',which has rating of 4.6
# and more than 10,000,000 INSTALLS
df_f_bridge=df_review[df_review['App']=='Build a Bridge!']
print(df_f_bridge.shape)
df_f_bridge.head()
# get the number of positive, negative and neutral reviews

df_f_bridge_s=df_f_bridge.groupby(by=['Sentiment']).count()[['App']]
df_f_bridge_s.columns={'count'}
df_f_bridge_s
# calculate the average sentiment polarity and sentiment subjectivity

print('The average sentiment polarity is: %.2f'%df_f_bridge['Sentiment_Polarity'].mean())
print('The average sentiment subjectivity is: %.2f'%df_f_bridge['Sentiment_Subjectivity'].mean())
# add a pie chart to show this count

df_f_bridge_s['count'].plot(kind='pie',
                        figsize=(7.5,5),
                        autopct='%1.1f%%',
                        labels=None,
                        textprops=dict(color="w",weight="bold"))

plt.title('Build a Bridge!_Sentiment',fontsize=15)
plt.ylabel('')
plt.legend(labels=df_f_bridge_s.index, loc='upper left') 
plt.axis('equal')

plt.show()
text_f = " ".join(df_f_bridge['Translated_Review'][i] for i in df_f_bridge[df_f_bridge['Sentiment']=='Positive']['Translated_Review'].index)
text_f_n = " ".join(df_f_bridge['Translated_Review'][i] for i in df_f_bridge[df_f_bridge['Sentiment']=='Negative']['Translated_Review'].index)
    
print ("There are {} words in the combination of all positive review.".format(len(text_f)))
print ("There are {} words in the combination of all negative review.".format(len(text_f_n)))
# Build the word cloud according to the positive reveiews

stopwords = set(STOPWORDS)
stopwords.update(["game", "play", "bridge",'level'])

wordcloud_f = WordCloud(max_words=80, background_color="white",stopwords=stopwords).generate(text_f)

plt.imshow(wordcloud_f, interpolation='bilinear')
plt.axis("off")
plt.show()
# Build the word cloud according to the negative reveiews

stopwords = set(STOPWORDS)
stopwords.update(["game", "play", "bridge",'level','much','levels','watch'])

wordcloud_f_n = WordCloud(max_words=80, background_color="black",stopwords=stopwords).generate(text_f_n)

plt.imshow(wordcloud_f_n, interpolation='bilinear')
plt.axis("off")
plt.show()
# Tools
# with the same method, I found the 'CM Locker - Security Lockscreen' for TOOLS category
# with a rating of 4.6 and over 100,000,000 installs

df_t_locker=df_review[df_review['App']=='CM Locker - Security Lockscreen']
df_t_locker.shape
# get the number of positive, negative and neutral reviews

df_t_locker_s=df_t_locker.groupby(by=['Sentiment']).count()[['App']]
df_t_locker_s.columns={'count'}
df_t_locker_s
# calculate the average sentiment polarity and sentiment subjectivity

print('The average sentiment polarity is: %.2f'%df_t_locker['Sentiment_Polarity'].mean())
print('The average sentiment subjectivity is: %.2f'%df_t_locker['Sentiment_Subjectivity'].mean())
# add a pie chart to show this count

df_t_locker_s['count'].plot(kind='pie',
                        figsize=(7.5,5),
                        autopct='%1.1f%%',
                        labels=None,
                        textprops=dict(color="w",weight="bold"))

plt.title('CM Locker-Security Lockscreen_Sentiment',fontsize=15)
plt.ylabel('')
plt.legend(labels=df_t_locker_s.index, loc='upper left') 
plt.axis('equal')

plt.show()
text_t = " ".join(df_t_locker['Translated_Review'][i] for i in df_t_locker[df_t_locker['Sentiment']=='Positive']['Translated_Review'].index)
text_t_n = " ".join(df_t_locker['Translated_Review'][i] for i in df_t_locker[df_t_locker['Sentiment']=='Negative']['Translated_Review'].index)

print ("There are {} words in the combination of all positive review.".format(len(text_t)))
print ("There are {} words in the combination of all negative review.".format(len(text_t_n)))
# Build the word cloud according to the positive reveiews

stopwords = set(STOPWORDS)
stopwords.update(["phone","screen","even","really","locker",'lock','unlock','app','lockscreen'])

wordcloud_t = WordCloud(max_words=80, background_color="white",stopwords=stopwords).generate(text_t)

plt.imshow(wordcloud_t, interpolation='bilinear')
plt.axis("off")
plt.show()
# Build the word cloud according to the negative reveiews

stopwords = set(STOPWORDS)
stopwords.update(["lock",'screen','lockscreen','unlock','phone'])

wordcloud_t_n = WordCloud(max_words=80, background_color="black",stopwords=stopwords).generate(text_t_n)

plt.imshow(wordcloud_t_n, interpolation='bilinear')
plt.axis("off")
plt.show()
# Game
# Similaly, I found 'Hill Climb Racing 2' for Game category
# with a rating of 4.6 and over 100,000,000 installs

df_g_hill=df_review[df_review['App']=='Hill Climb Racing 2']
df_g_hill.shape
# get the number of positive, negative and neutral reviews

df_g_hill_s=df_g_hill.groupby(by=['Sentiment']).count()[['App']]
df_g_hill_s.columns={'count'}
df_g_hill_s
# add a pie chart to show this count

df_g_hill_s['count'].plot(kind='pie',
                        figsize=(7.5,5),
                        autopct='%1.1f%%',
                        labels=None,
                        textprops=dict(color="w",weight="bold"))

plt.title('Hill Climb Racing 2',fontsize=15)
plt.ylabel('')
plt.legend(labels=df_g_hill_s.index, loc='upper left') 
plt.axis('equal')

plt.show()
# calculate the average sentiment polarity and sentiment subjectivity

print('The average sentiment polarity is: %.2f'%df_g_hill['Sentiment_Polarity'].mean())
print('The average sentiment subjectivity is: %.2f'%df_g_hill['Sentiment_Subjectivity'].mean())
text_g = " ".join(df_g_hill['Translated_Review'][i] for i in df_g_hill[df_g_hill['Sentiment']=='Positive']['Translated_Review'].index)
text_g_n = " ".join(df_g_hill['Translated_Review'][i] for i in df_g_hill[df_g_hill['Sentiment']=='Negative']['Translated_Review'].index)
    
print ("There are {} words in the combination of all review.".format(len(text_g)))
print ("There are {} words in the combination of all review.".format(len(text_g_n)))
# Build the word cloud according to the positive reveiews

stopwords = set(STOPWORDS)
stopwords.update(['game','make','play','really','need','many','way'])

wordcloud_g = WordCloud(max_words=80, background_color="white",stopwords=stopwords).generate(text_g)

plt.imshow(wordcloud_g, interpolation='bilinear')
plt.axis("off")
plt.show()
# Build the word cloud according to the negative reveiews

stopwords = set(STOPWORDS)
stopwords.update(['game','fully'])

wordcloud_g_n = WordCloud(max_words=80, background_color="black",stopwords=stopwords).generate(text_g_n)

plt.imshow(wordcloud_g_n, interpolation='bilinear')
plt.axis("off")
plt.show()
