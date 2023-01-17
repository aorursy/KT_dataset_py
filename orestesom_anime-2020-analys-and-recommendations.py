# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

pd.options.mode.chained_assignment = None  # default='warn'



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("../input/anime-dataset/anime.csv", engine='python')

display(df.head())

print(df.describe())
C = df['rating'].mean()

m = df['votes'].quantile(0.85)

print('Mean rating {:.2}, quantite of votes needes to stay {:.0f}'.format(C,m))

df2 = df.loc[df['votes'] >= m]

print(df.shape)

print(df2.shape)
def weight_rating(x, m=m, C=C):

    v = x['votes']

    R = x['rating']

    

    return (v/(v+m) * R) + (m/(m+v) * C)
df2['score'] = df2.apply(weight_rating, axis=1)
import matplotlib.pyplot as plt

import seaborn as sns

from matplotlib.ticker import FuncFormatter





plt.figure(figsize=(12, 3), dpi=100)





color_map = ['#3CB7F1' for _ in range(10)]

color_map[0] = '#5DF13C'





best_score = df2.sort_values(by=['score'], ascending=False)[:10]







g = plt.bar(best_score["title"], best_score['score'], color=color_map)

plt.ylabel("Score", color='green')

plt.xticks(rotation=45, horizontalalignment='right')

plt.title('Really good animes', fontweight='bold', fontsize=15);

best_rating_not_filter = df.sort_values(by=['rating'], ascending=False)[:10]





plt.figure(figsize=(12, 3), dpi=100)

g = sns.barplot(best_rating_not_filter["title"], best_rating_not_filter['rating'], palette="Oranges_r")

plt.ylabel("Rating", color='orange', fontweight='bold')

plt.xlabel("")

g.set_xticklabels(g.get_xticklabels(), rotation=45,  horizontalalignment='right')

plt.title('Really good recent rated animes', fontweight='bold', fontsize=15);
best_scores = best_score[['score','title','watched', 'studios']].set_index('title')

display(best_scores)
dropped = df2.sort_values(by=['dropped', 'score'], ascending=[False, False])



plt.figure(figsize=(12, 3), dpi=100)



color_map = ['#f59dd0' for _ in range(5)]

color_map[2] = '#5DF13C'





plt.barh(dropped['title'].head(5), dropped['score'].head(5), align='center', color=color_map)

plt.ylabel('Animes')

plt.xlabel('Scores')

plt.title('Most dropped animes');



display(dropped[['score','title','dropped', 'studios']].set_index('title').head(10))



#These are very good animes, but they're also dropped very frequently.
#use all the data.



df['description'].head()
from sklearn.feature_extraction.text import TfidfVectorizer



#create the object vector 

tfidf = TfidfVectorizer(stop_words='english')



#fill nans

df['description'] = df['description'].fillna('')



#fit and transform the description in a Term Frequency-Inverse Document Frequency (TF-IDF) matrix

tfidf_matrix =  tfidf.fit_transform(df['description'])



tfidf_matrix.shape



from sklearn.metrics.pairwise import linear_kernel



#use the tfidf_matrix to pass into a linear kernel and get the cosine similarity matrix 

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

#create a index to pass the anime and get the idx

indice = pd.Series(df.index, index=df['title']).drop_duplicates()



display(indice.head())

print("A Silent Voice is in index: ", indice['A Silent Voice'])
#define a function to pass the anime and return the recommendations



def recommendation(title, cosine_sim=cosine_sim):

    #Get the index of the anime pass

    idx = indice[title]

    

    #make the pairwise similarity score

    sim_scores = list(enumerate(cosine_sim[idx]))

    

    

    #sort base on similarity

    sim_scores = sorted(sim_scores, key=lambda x:x[1], reverse =True)

    

    #get the 10 most similar

    sim_scores = sim_scores[1:11]

    

    #get the index in df

    anime_index = [i[0] for i in sim_scores]

    

    #return the animes

    return df['title'].iloc[anime_index]

    

    
recommendation('Attack on Titan 3rd Season: Part II')
recommendation('One-Punch Man')
#create a copy of studios, we'll use this in the future to explore the data

df['copy_studios'] = df['studios']



features = ['studios','contentWarn', 'tags']



print(df[features].isna().sum())
from ast import literal_eval



for feature in features:

    df[feature] = df[feature].apply(literal_eval)
#make a function to prepare the date

def clean_data(x):

    if isinstance(x, list):

        return [str.lower(i.replace(" ","")) for i in x]

    

    else:

        if isinstance(x, str):

            return str.lower(x.replace(" ",""))

        else:

            return ""
for feature in features:

    df[feature] = df[feature].apply(clean_data)
#create a function to put all the words in one 'soup'

def soup(x):

    return " ".join(x['studios']) + " " + " ".join(x['contentWarn']) + " " +" ".join(x['tags'])



df['soup'] = df.apply(soup, axis=1)

    
print(df['soup'][10])
from sklearn.feature_extraction.text import CountVectorizer



count = CountVectorizer(stop_words='english')



count_matrix = count.fit_transform(df['soup'])

from sklearn.metrics.pairwise import cosine_similarity



# Compute the Cosine Similarity matrix based on the count_matrix,

#the count matrix don't down-weight the number of times a tag appears,

#in this case this is better





cosine_2 = cosine_similarity(count_matrix, count_matrix)
#create a index Serie to pass the anime

df = df.reset_index()

indice_2 = pd.Series(df.index, index=df['title'])
#get the recommendation 

recommendation('One-Punch Man', cosine_2)
recommendation('Paprika', cosine_2)
display(df[['title', 'mediaType', 'eps', 'duration', 'studios', 'tags', 'contentWarn', 'rating']].loc[df['title'] == 'Paprika'])

display(df[['title', 'mediaType', 'eps', 'duration', 'studios', 'tags', 'contentWarn', 'rating']].iloc[[5453, 6113, 3394, 3877]])
#get rid of the animes that hasn't been released 

df = df.loc[df['startYr'] <= 2020]
#make bins and labels for decades

bins = [i for i in range(1910,2021,10)]

labels = [str(i)+str("-")+str(i+10) for i in range(1910,2020,10)]



df['decade_of_released'] = pd.cut(df['startYr'], bins=bins, labels=labels)

tv_data = df.loc[df['mediaType'] == 'TV']

tv_data = tv_data.groupby('decade_of_released').count()['title']



#plot



plt.figure(figsize=(12, 4), dpi=120)

sns.set_style("ticks")



splot = sns.countplot(x='decade_of_released', hue='mediaType', data=df)



for p in splot.patches:

    if p.get_height() in tv_data.values:

        splot.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()), 

                       ha = 'right', va = 'center', xytext = (0, 4), textcoords = 'offset points', color='blue')

        

plt.ylabel("Amount", color='orange', fontweight='bold')

plt.xlabel("Decade of Released", color='orange', fontweight='bold', labelpad=15)

plt.legend(loc='upper left', facecolor='#42d3ff', framealpha=1)

sns.despine()

plt.show();
plt.figure(figsize=(12, 4), dpi=120)

scatter = sns.violinplot(data=df, x='decade_of_released', y='rating')



plt.ylabel("Rating", color='orange', fontweight='bold')

plt.xlabel("Decade of Released", color='orange', fontweight='bold', labelpad=15)

plt.title('Decade and ratings', fontweight='bold');
plt.figure(figsize=(12, 4), dpi=120)

scatter = sns.violinplot(data=df, x='mediaType', y='rating')

plt.ylabel("Rating", color='orange', fontweight='bold')

plt.xlabel("Media Type", color='orange', fontweight='bold', labelpad=15)

plt.title('Media Type', fontweight='bold');
plt.figure(figsize=(18, 4), dpi=120)

sns.violinplot(data=df, x='decade_of_released', y='rating', color="white")

sns.stripplot(data=df, x='decade_of_released', y='rating', hue='mediaType', jitter=True,

                   dodge=True, 

                   marker='o', 

                   alpha=0.2)







plt.ylabel("Rating", color='orange', fontweight='bold')

plt.xlabel("Decade of Released", color='orange', fontweight='bold', labelpad=15)

plt.title('Decade, Ratings and Media Type', fontweight='bold');
#fill the NaN with zeros

df['rating'] = df['rating'].fillna(0)

#The Copy Studios is not a list, but a string, so we need to clean that.

df['copy_studios'] = df['copy_studios'].str.replace('[', '')

df['copy_studios'] = df['copy_studios'].str.replace(']', '')

df['copy_studios'] = df['copy_studios'].str.replace("'", "")

df['copy_studios'] = df['copy_studios'].str.split(",")

#create a dict with each studio

cnt = {}



for idx, row in df.iterrows():

    rating = row['rating']

    studios = row['copy_studios']

    for studio in studios:      

        if not studio in cnt:

            cnt[studio] = {}

            cnt[studio].setdefault('productions', 1)

            score = float(rating) 

            cnt[studio]['rating'] = []

            cnt[studio]['rating'].append(score)

        else:

            score = float(rating)

            cnt[studio]['productions'] += 1

            cnt[studio]['rating'].append(score)

            



import numpy as np

#get the mean rating of the studios

for studio in cnt:

    cnt[studio]['rating'] = round(np.mean(cnt[studio]['rating']),2)

    
#make the dict a data frame

studios = pd.DataFrame.from_dict(cnt, orient='index')

#let's see the most prolific studios



more_productive_st = studios.sort_values(by=['productions', 'rating'], ascending = [False, False])[:20]

more_productive_st
#plot the results



more_productive_st['studios'] = more_productive_st.index



sns.set_style(style="whitegrid")

plt.figure(figsize=(12, 6), dpi=100)





gx2 = sns.scatterplot(x='studios', y="productions",data= more_productive_st[1:], size='rating', sizes=(20, 200), color="skyblue")





plt.xticks(rotation=45, horizontalalignment='right')

plt.xlabel('Studios', fontweight='bold', labelpad=10, color='green')

plt.ylabel('Productions 1917-2020', fontweight='bold', labelpad=5, color='green')

plt.title('Studios, Productions and Ratings', fontweight='bold', color='darkblue', fontsize=13)





plt.show()