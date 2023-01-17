#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcTKyJ3mcf7b2TnsgXfySc9un3iwG3iDR5EGWxc8JOujBF_Hy842&usqp=CAU',width=400,height=400)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import string

import nltk

nltk.download('stopwords')

from nltk.corpus import stopwords

from nltk.probability import FreqDist

from wordcloud import WordCloud, STOPWORDS

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import  TfidfTransformer

from sklearn.pipeline import Pipeline

from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import classification_report



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/aipowered-literature-review-csvs/kaggle/working/TIE/What do models for transmission predict_.csv")

df.head()
df = df.rename(columns={'Unnamed: 0':'unnamed', 'Mathematical Model': 'maths', 'Link to Source Code': 'link'})
df['maths_length']=df['maths'].apply(len)
df.describe()
import warnings

warnings.filterwarnings("ignore")



fig = plt.figure(figsize=(12,8))

axes1 = plt.subplot(2,2,1)

axes1 = sns.countplot(x='unnamed', data=df)

axes1.set_title('unnamed')

axes1.set_ylabel('Count')



axes2 = plt.subplot(2,2,2)

axes2 = sns.countplot(x='maths_length', data=df)

axes2.set_title('maths_length')

axes2.set_ylabel('Count')



plt.tight_layout()
# Maths Length

warnings.filterwarnings("ignore")

fig = plt.figure(figsize=(8,6))

sns.distplot(df['maths_length'], kde=True, bins=50)

plt.title('Maths Length Distribution')
sns.set(font_scale=1.4)

plt.figure(figsize = (8,4))

sns.heatmap(df.corr(),cmap='summer',annot=True,linewidths=.5)
sns.pairplot(df, hue='unnamed', palette='coolwarm')
warnings.filterwarnings("ignore")

sns.boxplot(x='unnamed', y='maths_length', data=df, palette='rainbow')
Maths = df[['unnamed', 'maths_length']]

Maths.head()
def remove_punc_stopword(text):

    """

    Takes in a string of text, then performs the following:

    1. Remove all punctuation

    2. Remove all stopwords

    3. Returns a list of the cleaned text

    """

    remove_punc = [word for word in text if word not in string.punctuation]

    remove_punc = ''.join(remove_punc)

    return [word.lower() for word in remove_punc.split() if word.lower() not in stopwords.words('english')]
df_models = df.copy()

df_models['maths'] = df_models['maths'].apply(remove_punc_stopword)

df_models.count()
df_models.head()
df_models_text = df_models['maths'].values



wordcloud = WordCloud(

    width = 3000,

    height = 2000,

    background_color = 'black',

    colormap = 'Set3',

    stopwords = STOPWORDS).generate(str(df_models_text))



fig = plt.figure(

    figsize = (10, 7),

    facecolor = 'k',

    edgecolor = 'k')

plt.imshow(wordcloud, interpolation = 'bilinear')

plt.axis('off')

plt.tight_layout(pad=0)
X = df['maths']

y = df['unnamed']

cv = CountVectorizer()

X = cv.fit_transform(X)



test_size = np.linspace(0.1, 1, num=9, endpoint=False)

random_state = np.arange(0, 43)

grid_results= []

for testsize in test_size:

    for randomstate in random_state:

        try:

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testsize, random_state=randomstate)

            mnb = MultinomialNB()

            mnb.fit(X_train, y_train)

            y_test_pred = mnb.predict(X_test)     

            grid_results.append([testsize, randomstate, mean_squared_error(y_test, y_test_pred)])

            grid_frame = pd.DataFrame(grid_results)

            grid_frame.rename(columns={0:'Test Size', 1:'Random State', 2:'MSE of Test'}, inplace=True)

        except Exception:

            print(Exception.with_traceback())

            print('error')

            continue



min_test_mse = grid_frame[grid_frame['MSE of Test'] == grid_frame['MSE of Test'].min()]

min_test_mse
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=21)

mnb = MultinomialNB()

mnb.fit(X_train, y_train)

y_test_pred = mnb.predict(X_test)

print(classification_report(y_test,y_test_pred))
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcRWiWNVE5aZ9--LI6Uizq9XeUqrotC3bMxLuwLjxAJimVU08Rq6&usqp=CAU',width=400,height=400)