import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style("whitegrid")



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



%matplotlib inline 
data = pd.read_csv("../input/Netflix Shows.csv", encoding='cp437')

data.head(10)
data.isnull().sum()
data['user rating score'].dropna().plot(kind = "density")
print("The median: {}".format(np.median(data['user rating score'].dropna())))

print("The mean : {}".format(np.mean(data['user rating score'].dropna())))
user_rating_median = np.median(data['user rating score'].dropna())

data['user rating score'] = data['user rating score'].fillna(user_rating_median)



data['ratingLevel'].value_counts().head()
fillna_ratingLevel = "Parents strongly cautioned. May be unsuitable for children ages 14 and under."

data['ratingLevel'] = data['ratingLevel'].fillna(fillna_ratingLevel)
#Now after substituing lets check if everything has been done well. 

data.isnull().sum()
data['user rating size'].plot(kind = "density")
data = data.drop('user rating size', axis=1)
movies_in_year = data["release year"].value_counts().to_frame().reset_index()

movies_in_year.columns = ['release year', 'release number']



movies_in_year = movies_in_year.sort_values('release year', ascending = False)
movies_in_year.corr()
sns.barplot(x = "release year", y = 'release number', data = movies_in_year.head(15))
data['ratingLevel'].unique()



words = ['rude', 'sex', 'scary', 'violence', 'sex_related', 'adult', 'drug', 'sexual', 'nudity', 'parents', 'children']

#adding parents, children since it is indicative that this isn't for children.



def compute_score(data):

    score = 0

    for i in words:

        if(i in data):

            score = score + data.count(i)

    return(score)



data['adult_score'] = [compute_score(datum) for datum in data['ratingLevel']]
data['adult_score'].plot(kind = "density")
data['adult_score'].value_counts().plot(kind = "bar")
data[data['adult_score'] > 3]



data_adultscore = data.copy()

data_adultscore = data_adultscore[data_adultscore['ratingLevel'] != "This movie has not been rated."]

data_adultscore[data_adultscore['adult_score'] ==0].head(10)
del(data_adultscore)
from wordcloud import WordCloud, STOPWORDS



wordcloud = WordCloud(

                          stopwords=STOPWORDS,

                          background_color='white',

                          width=1500,

                          height=1500

                         ).generate(" ".join(data['title']))

plt.imshow(wordcloud)

plt.axis('off')

plt.show()