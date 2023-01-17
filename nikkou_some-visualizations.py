import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from wordcloud import WordCloud
filename = '../input/rateme.csv'
rateme = pd.read_csv(filename, index_col=0, encoding='utf-8', low_memory=False,
                     lineterminator='\n', parse_dates=['post_created', 'comment_created'])
rateme = rateme[~rateme.comment_is_submitter]  # remove comments on commenters' own posts for now
def plot_ratings(ratings_blue, ratings_purple, label_blue, label_purple):
    fig,ax = plt.subplots(1, figsize=(7.5, 6))
    sns.distplot(ratings_blue, bins=10, color='blue', label=label_blue)
    sns.distplot(ratings_purple, bins=10, color='purple', label=label_purple)
    plt.legend(fontsize=15)
    plt.xlabel('Rating', fontsize=15)
    ax.set_yticklabels([])
    ax.xaxis.set(ticks = np.arange(0.5, 11), ticklabels = np.arange(1, 11))
    plt.xlim(0, 10)
    ax.xaxis.set_ticks_position('none') 
    ax.yaxis.set_ticks_position('none') 
    plt.suptitle('Distribution of ratings', fontsize=20)
ratings_m = rateme[rateme.comment_author_gender == 'male'].groupby('comment_author').mean()['comment_rating'].values
ratings_f = rateme[rateme.comment_author_gender == 'female'].groupby('comment_author').mean()['comment_rating'].values
ratings_m = ratings_m[~np.isnan(ratings_m)]
ratings_f = ratings_f[~np.isnan(ratings_f)]
print(ratings_m.shape[0], ratings_f.shape[0])
# turns out males are 3 times more active commenters based on limited data about gender of commenters
plot_ratings(ratings_m, ratings_f, 'by men', 'by women')
fig,ax = plt.subplots(1, figsize=(7.5, 6))
sns.distplot(ratings_m, bins=10, color='blue', label='male commenters')
sns.distplot(ratings_f, bins=10, color='purple', label='female commenters')
plt.legend(fontsize=15)
plt.xlabel('Rating', fontsize=15)
ax.set_yticklabels([])
ax.xaxis.set(ticks = np.arange(0.5, 11), ticklabels = np.arange(1, 11))
plt.xlim(0, 10)
ax.xaxis.set_ticks_position('none') 
ax.yaxis.set_ticks_position('none') 
plt.suptitle('Distribution of ratings', fontsize=20)
ratings_mtof = rateme[(rateme.comment_author_gender == 'male') &
                    (rateme.post_gender == 'female')].groupby('comment_author').mean()['comment_rating'].values
ratings_ftof = rateme[(rateme.comment_author_gender == 'female') &
                    (rateme.post_gender == 'female')].groupby('comment_author').mean()['comment_rating'].values
ratings_mtom = rateme[(rateme.comment_author_gender == 'male') &
                    (rateme.post_gender == 'male')].groupby('comment_author').mean()['comment_rating'].values
ratings_ftom = rateme[(rateme.comment_author_gender == 'female') &
                    (rateme.post_gender == 'male')].groupby('comment_author').mean()['comment_rating'].values
ratings_mtof = ratings_mtof[~np.isnan(ratings_mtof)]
ratings_ftof = ratings_ftof[~np.isnan(ratings_ftof)]
ratings_mtom = ratings_mtom[~np.isnan(ratings_mtom)]
ratings_ftom = ratings_ftom[~np.isnan(ratings_ftom)]
plot_ratings(ratings_mtof, ratings_ftof, 'by men on women', 'by women on women')
plot_ratings(ratings_mtom, ratings_ftom, 'by men on men', 'by women on men')
genders = ['male', 'female']
for gender_pair in [genders, list(reversed(genders))]:
    age_ratings = []
    ages = range(18, 30)
    for age in ages:
        age_posts = rateme[(rateme.post_gender == gender_pair[0]) & (rateme.post_age == age) & (rateme.comment_author_gender == gender_pair[1])]
        ratings_mean = age_posts.comment_rating.mean()
        age_ratings.append(ratings_mean)
    plt.figure(figsize=(7, 7))
    plt.suptitle('Average rating vs age ({})'.format(gender_pair[0]))
    plt.xlabel('Age')
    plt.ylabel('Rating')
    plt.plot(ages, age_ratings)
for gender in ['male', 'female']:
    comments = rateme[(rateme.comment_rating == 10) & (rateme.post_gender == gender)].comment_body.values
    word_list = []
    for comment in comments:
        comment_words = comment.split()
        word_list.extend(comment_words)
    text = ' '.join(word_list)
    width = 2400
    height = 600
    wordcloud = WordCloud(width=width, height=height, max_words=500).generate(text)
    fig = plt.figure(figsize=(24, 6))
    plt.imshow(wordcloud)
