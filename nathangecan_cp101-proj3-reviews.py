import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import os

import json



from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
business_ids = ['DRiTXZ2laRiAgrB-Gkgv1g',

 '4s_kb5VXnx96fEJLZ_AkIQ',

 'mfQfNIb3TYeBSxsN1WD8eA',

 'iLlPXAA6NYTgWaG4DgCNEA',

 'd0N5HcdMCeXcX81hcRVUFw',

 '0imWty0eKpz8YpuZAKF83Q',

 'Bv5RS9NFbTsk9D7fvYx4Hw',

 'HKSyA7YZplj9i_vIdtwyzg',

 '3sXlZWM9B72_r-dYBIgmyg',

 'viry1tKdw736ZRBeX9UB5Q',

 'XydKWBaKYDO2BFXs8pXnbg',

 'O_YSJEMk2GzcmnwXWBj74g',

 '8Aswo1-FW3_8PKVyM_oVQg',

 '3ECiKRkd3KXO48GdDTp_8g',

 '9dz6ZYUDCqtD67LSPTLApQ',

 '5uvX6yqie9DpHR7UAiQZtw',

 'fcUZ0HlaO1bMW26tzQZVOQ',

 '9a9xBjlW2RxKxD9RMgmcuQ',

 'PJYjYb7oKtSwpE613W44bg',

 'ZYkAMs9UcYuwaXSiGczSkA',

 'QhMhM3Mhokv6V-H0MRuYTg',

 'YgL5oNrTj6rxr1KQjCM-fw',

 '5wTs_RnU7XE_S2uw20kU9A',

 '16NS8EICI94IGdn-S0yxkw',

 'XBIgiDL_sXYOroaX4ULaUw',

 '7kXrUSjG67NitjRfRFn9cw',

 'k0vkHxJTOex5GRpcznaayA',

 'W7Ylnm1JED6Ey0Kt9szqJA',

 'n0wVs3pvWsFRNY1KDuITyA',

 'kaFItSnFuTSpmohv2jDbrQ',

 'Wpt0sFHcPtV5MO9He7yMKQ',

 '_0ZIFTvfcA3UETO_S_JTNA',

 'yOvl22iVeqK3_EJbu5Qpaw',

 'l0-d7zbmkwvQ6YaUhFBE-w',

 'yLiaMaJFq03JxXPk4puloQ',

 'qorLOp7fZ6X7GSQG_0wPdg',

 'vGLl5xum2u2Qf8_AvzSenw',

 'qi-zm_G6qyxOybAx1VQ30A',

 'ajwGl90CuCZac3BKWY1ShQ',

 'sJ8B4Lq6AY-m5dzIbDNgyQ',

 'gsjxrwdHqKdTeNx3GeDKNQ',

 'QqrmTtv2OXWFRlqRCG4FRA',

 '4oC1z2SDrpraoZzRIQ-WsA',

 'KMxrk1YDlj_qFI1JXK4zRg',

 'CHW_DPKnTqudyRaDa4vwkg',

 'T6VKCBhHojQkCWp_IYO24w',

 '2fVnJlyKVkt1WjPOlNdP7g',

 'EEePiBwchl-TIBVcTdE0RA',

 'dTU6CKCfGNBZCT0AmBbyRA',

 'Cm-BH_7VPLP63FdI9ML6vA',

 'eJECvXFIUnfcDE6PzKS2Yw',

 'i0EtEzlEkOho6YcLG29bvw',

 'xI8O5h3Pa0MQlTWf93noTA',

 'me8j11RcqTmL0C5GEHP5Sw',

 'WM33kyvA8XfiL8lk2I2vcA',

 '4v9nhZ5h-KScIsG7uXjfzg',

 '4TyDUOjjB0Z1nffg-ZjC3A',

 'LBwFjTUlDeikknOoQTXxhg',

 '9R2UXiCt1DgV8Yhm3WX2KA',

 'Kb43diOoBV4l77bxpn2k3Q',

 'agySn0Jacnul-OiiNV8Rrg',

 '3WcVxBOl_gWxpRjkdLURSg',

 'rECgTdKelvefty5JIXHDbg',

 'kB1S2GJUjE2k3XN1WQS9tQ',

 'JHN09M-CRTGln2rRnS9hWw',

 'LsfKW67Qe_0bU4krXEkOAA',

 'ByXko9JQ6Jne7v8t4IaQjg',

 'P7Dqa0IbcFgRZ1aK7CPAOw',

 'SqVIolLCBmQYyKeai3pCTg',

 'tzLBPCVz6uB7vWgDTU6Ujw',

 'qE9yIXn2GQb2-4a_qOOKzg',

 'Wp1UkYg5LjeI3rDByzPxlg',

 'ncEJaX_79zZGSS1NQNzu4A',

 'Uw4z3M7H4buYtTGBrzFCSg']
count = 0

filename = '../input/review.json'

with open(filename, 'r') as f:

    data = {}

    for line in f:

        d = json.loads(line)

        if d['business_id'] in business_ids:

            data[count] = d

            count += 1
reviews = pd.DataFrame(data)

reviews = reviews.T
locations = reviews.groupby('business_id').agg(list)

locations['stars'] = locations['stars'].apply(np.mean)

locations.drop(columns=['date', 'user_id'], inplace=True)
def get_reviews(business_id, stars=np.arange(1, 6)):

    return ''.join(reviews[(reviews['business_id'] == business_id) & (reviews['stars'].isin(stars))]['text'].values)



def generate_wordcloud(business_id, stars=np.arange(1, 6)):

    """

    Function that will generate a wordcloud for the most common words used in Yelp reviews for that business_id that gave a rating within

    stars.

    

    stars : range of ratings to filter to generate the wordcloud

    """

    stopwords = set(STOPWORDS)

    stopwords.update(['McDonald\'s', 'McDonald', 'McDonalds'])

    

    text = get_reviews(business_id, stars)

    # Create and generate a word cloud image:

    wordcloud = WordCloud(stopwords=stopwords, width=800, height=400, background_color='white', max_words=100).generate(text)



    # Display the generated image:

    plt.figure(figsize=(10, 10))

    plt.imshow(wordcloud, interpolation='bilinear')

    plt.axis('off')

    plt.show()

# Strip, all

generate_wordcloud('DRiTXZ2laRiAgrB-Gkgv1g')
# University, all

generate_wordcloud('JHN09M-CRTGln2rRnS9hWw')
# University 2, all

generate_wordcloud('qE9yIXn2GQb2-4a_qOOKzg')
# Airport, all

generate_wordcloud('kB1S2GJUjE2k3XN1WQS9tQ')
# Strip, 3+

generate_wordcloud('DRiTXZ2laRiAgrB-Gkgv1g', np.arange(3, 6))
!pip install vaderSentiment

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
def get_score(business_id, stars=np.arange(0, 6)):

    """

    Return the mean sentiment compound score for all reviews of business_id within the stars range.

    """

    analyzer = SentimentIntensityAnalyzer()

    scores = []

    for review in reviews[(reviews['business_id'] == business_id) & (reviews['stars'].isin(stars))]['text'].values:

        scores = np.append(scores, analyzer.polarity_scores(review)['compound'])

    return np.mean(scores)
get_score('DRiTXZ2laRiAgrB-Gkgv1g')
get_score('JHN09M-CRTGln2rRnS9hWw', np.arange(4, 5))
get_score('me8j11RcqTmL0C5GEHP5Sw')
generate_wordcloud('JHN09M-CRTGln2rRnS9hWw', np.arange(1, 2))

print(get_score('JHN09M-CRTGln2rRnS9hWw', np.arange(1, 2)))

generate_wordcloud('JHN09M-CRTGln2rRnS9hWw', np.arange(2, 3))

print(get_score('JHN09M-CRTGln2rRnS9hWw', np.arange(2, 3)))

generate_wordcloud('JHN09M-CRTGln2rRnS9hWw', np.arange(3, 4))

print(get_score('JHN09M-CRTGln2rRnS9hWw', np.arange(3, 4)))

generate_wordcloud('JHN09M-CRTGln2rRnS9hWw', np.arange(4, 5))

print(get_score('JHN09M-CRTGln2rRnS9hWw', np.arange(4, 5)))

generate_wordcloud('JHN09M-CRTGln2rRnS9hWw', np.arange(5, 6))

print(get_score('JHN09M-CRTGln2rRnS9hWw', np.arange(5, 6)))
generate_wordcloud('DRiTXZ2laRiAgrB-Gkgv1g', np.arange(1, 2))

print(get_score('DRiTXZ2laRiAgrB-Gkgv1g', np.arange(1, 2)))

generate_wordcloud('DRiTXZ2laRiAgrB-Gkgv1g', np.arange(2, 3))

print(get_score('DRiTXZ2laRiAgrB-Gkgv1g', np.arange(2, 3)))

generate_wordcloud('DRiTXZ2laRiAgrB-Gkgv1g', np.arange(3, 4))

print(get_score('DRiTXZ2laRiAgrB-Gkgv1g', np.arange(3, 4)))

generate_wordcloud('DRiTXZ2laRiAgrB-Gkgv1g', np.arange(4, 5))

print(get_score('DRiTXZ2laRiAgrB-Gkgv1g', np.arange(4, 5)))

generate_wordcloud('DRiTXZ2laRiAgrB-Gkgv1g', np.arange(5, 6))

print(get_score('DRiTXZ2laRiAgrB-Gkgv1g', np.arange(5, 6)))
generate_wordcloud('qE9yIXn2GQb2-4a_qOOKzg', np.arange(1, 2))

print(get_score('qE9yIXn2GQb2-4a_qOOKzg', np.arange(1, 2)))

generate_wordcloud('qE9yIXn2GQb2-4a_qOOKzg', np.arange(2, 3))

print(get_score('qE9yIXn2GQb2-4a_qOOKzg', np.arange(2, 3)))

generate_wordcloud('qE9yIXn2GQb2-4a_qOOKzg', np.arange(3, 4))

print(get_score('qE9yIXn2GQb2-4a_qOOKzg', np.arange(3, 4)))

generate_wordcloud('qE9yIXn2GQb2-4a_qOOKzg', np.arange(4, 5))

print(get_score('qE9yIXn2GQb2-4a_qOOKzg', np.arange(4, 5)))

generate_wordcloud('qE9yIXn2GQb2-4a_qOOKzg', np.arange(5, 6))

print(get_score('qE9yIXn2GQb2-4a_qOOKzg', np.arange(5, 6)))
plt.figure(figsize=(10, 10))

sns.barplot(x=['1 to 2', '2 to 3', '3 to 4', '4 to 5', '5'], y=[-0.3752, 0.0948, 0.3696, 0.7376, 0.7262], color='gold')

plt.xlabel('Number of Stars')

plt.ylim(-1, 1)

plt.title('Distribution of Sentiment Scores for Reviews of The Strip McDonald\'s')

plt.ylabel('Compound Sentiment');
plt.figure(figsize=(10, 10))

sns.barplot(x=['1 to 2', '2 to 3', '3 to 4', '4 to 5', '5'], y=[-0.4016, -0.2925, 0.5567, 0.5589, 0.9398], color='green')

plt.xlabel('Number of Stars')

plt.ylim(-1, 1)

plt.title('Distribution of Sentiment Scores for Reviews of Paradise Road McDonald\'s')

plt.ylabel('Compound Sentiment');
plt.figure(figsize=(10, 10))

sns.barplot(x=['1 to 2', '2 to 3', '3 to 4', '4 to 5', '5'], y=[-0.5336, -0.3086, 0.2155, 0.8344, 0.5073], color='red')

plt.xlabel('Number of Stars')

plt.ylim(-1, 1)

plt.title('Distribution of Sentiment Scores for Reviews of Maryland Avenue McDonald\'s')

plt.ylabel('Compound Sentiment');