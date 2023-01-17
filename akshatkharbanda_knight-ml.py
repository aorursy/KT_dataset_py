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
df=pd.read_csv('/kaggle/input/knight-ml-dataset/Knight ML Assignment/Data/train.csv')
test_df=pd.read_csv('/kaggle/input/knight-ml-dataset/Knight ML Assignment/Data/test.csv')
df.head(20)
test_df.head()
df.info()
#Dropping useless columns
df=df.drop('user_name', axis=1)
test_df=test_df.drop('user_name', axis=1)
#Dropping empty columns, except region 2 since that is allowed to be blank
arr=df.columns.values
arr=np.delete(arr,8)
df = df.dropna(subset=arr, axis=0)
df = df.reset_index()
df= df.drop(columns='index')
df.head(20)
#Drawing general insights about data
df.describe()
#Visualizing data for presentation
import matplotlib.pyplot as plt
plt.hist(df['points'])
plt.show()
#Points between 87.5 to 92.5. Good news.
df.groupby('country').points.agg([len,np.mean])
#USA has max wines, but highest rating is for Italian(and French). Increase production there.
df.groupby('country').price.agg([len,np.mean, max])
#Italian and French wines are costlier too. We can rake in higher profits.
(df['province'].value_counts().head(10) / len(df)).plot.bar(color=plt.cm.Paired(np.arange(len(df['province']))))
#40% of wine in California. Should diversify.
import seaborn as sns
sdf = np.log(df['price'])
sns.jointplot(y=sdf, x='points', data=df, kind='reg')
#The highest prices isn't of the wine with highest pontuation.
#The most expensive wine have ponctuation between 87 and 90. Not good for business, higher price should imply higher quality.
viodf = df[df.variety.isin(df.variety.value_counts().head(4).index)]
sns.violinplot(
    x='variety',
    y='points',
    data=viodf
)
#Pinot Noit amazing, Red Blend sucks
(df['winery'].value_counts().head(10) / len(df)).plot.bar(color=plt.cm.Paired(np.arange(len(df['winery']))))
#Max Testarossa, but overall diversified. Noice.
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import CountVectorizer
from nltk import word_tokenize  
from nltk.stem import WordNetLemmatizer 
from sklearn.model_selection import train_test_split
sw = stopwords.words('english')
labelEncoder = LabelEncoder()
df['variety'] = labelEncoder.fit_transform(df['variety'])
df['variety']
df['review_description'] = df['review_description'].astype(str)
df['review_title'] = df['review_title'].astype(str)
df['review_description'] = df.review_title.map(str) + " " + df.review_description
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
text = df.review_description.values
wordcloud = WordCloud(
    width = 3000,
    height = 2000,
    background_color = 'black',
    stopwords = STOPWORDS).generate(str(text))
fig = plt.figure(
    figsize = (40, 30),
    facecolor = 'k',
    edgecolor = 'k')
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()
class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, articles):
        return [self.wnl.lemmatize(t) for t in word_tokenize(articles)]
def get_vector_feature_matrix(description):
    vectorizer = CountVectorizer(lowercase=True, stop_words="english", tokenizer=LemmaTokenizer(), max_features=1000)
    vector = vectorizer.fit_transform(np.array(description))
    return vector, vectorizer
vectorizer = CountVectorizer(lowercase=True, stop_words="english", max_features=10)
vector, vectorizer = get_vector_feature_matrix(df['review_description'])
features = vector.todense()
X= features
y= df['variety']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(multi_class='ovr',solver='lbfgs', max_iter=5000)
model = lr.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)
print ("Accuracy is {}".format(accuracy))
test_df['review_description'] = test_df['review_description'].astype(str)
test_df['review_title'] = test_df['review_title'].astype(str)
test_df['review_description'] = test_df.review_title.map(str) + " " + test_df.review_description
vector, vectorizer = get_vector_feature_matrix(test_df['review_description'])
features = vector.todense()
predictions = model.predict(features)
predictions = labelEncoder.inverse_transform(predictions)
predictions = pd.Series(predictions)
test_df['variety'] = predictions.values
test_df.to_csv('results.csv')