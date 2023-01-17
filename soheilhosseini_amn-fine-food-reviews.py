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
reviews = pd.read_csv('../input/amazon-fine-food-reviews/Reviews.csv')
print("There are {} observations and {} features in this dataset. "\

      .format(reviews.shape[0],reviews.shape[1]))
reviews.head()
import seaborn as sns

import matplotlib.pyplot as plt

sns.countplot(reviews['Score'])

plt.show()
sns.distplot(reviews['Score'], color='r')

plt.show()
reviews.info()
reviews.describe()
print("There are {} types of summary in this dataset such as\n {}... "\

      .format(len(reviews.Summary.unique()),", ".join(reviews.Summary.unique()[0:5])))
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
tex = reviews.Text
# Create and generate a word cloud image:





wordcloud = WordCloud().generate(tex[0])



# Display the generated image:

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()
# lower max_font_size, change the maximum number of word and lighten the background:

wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(tex[0])

plt.figure()

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.show()
text = " ".join(review for review in tex)

print ("There are {} words in the combination of all review.".format(len(text)))
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.naive_bayes import MultinomialNB



from warnings import filterwarnings

from sklearn.preprocessing import Normalizer

from sklearn.model_selection import cross_validate



from sklearn.pipeline import Pipeline
twitter_sentiment = Pipeline([('CVec', CountVectorizer(stop_words='english')),

                     ('Tfidf', TfidfTransformer()),

                     ('MNB', MultinomialNB())])
%%time

cv_pred = cross_validate(twitter_sentiment,

                             reviews['Text'], 

                             reviews['Score'], 

                             cv=5,

                             scoring=('roc_auc_ovr'), n_jobs=-1, verbose =10)
cv_pred['test_score']