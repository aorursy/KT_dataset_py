import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



import warnings



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
yelp_df = pd.read_csv('../input/yelp.csv')

yelp_df.head()
yelp_df = yelp_df.drop(['business_id', 'date', 'review_id', 'type', 'user_id'], axis=1)

yelp_df = yelp_df.rename(columns={'stars':'Stars', 'text':'Text', 'cool':'Cool'

                                  , 'useful':'Useful', 'funny':'Funny'})

yelp_df.head()
yelp_df['Text Length'] = yelp_df['Text'].apply(len)

yelp_df.head()
yelp_df.describe()
warnings.filterwarnings("ignore")



fig = plt.figure(figsize=(12,8))

axes1 = plt.subplot(2,2,1)

axes1 = sns.countplot(x='Stars', data=yelp_df)

axes1.set_title('Stars')

axes1.set_ylabel('Count')



axes2 = plt.subplot(2,2,2)

axes2 = sns.countplot(x='Cool', data=yelp_df)

axes2.set_title('Cool')

axes2.set_ylabel('Count')



axes3 = plt.subplot(2,2,3)

axes3 = sns.countplot(x='Useful', data=yelp_df)

axes3.set_title('Useful')

axes3.set_ylabel('Count')



axes4 = plt.subplot(2,2,4)

axes4 = sns.countplot(x='Funny', data=yelp_df)

axes4.set_title('Funny')

axes4.set_ylabel('Count')



plt.tight_layout()
# Text Length

warnings.filterwarnings("ignore")

fig = plt.figure(figsize=(12,8))

sns.distplot(yelp_df['Text Length'], kde=True, bins=50)

plt.title('Text Length Distribution')
yelp_cor = yelp_df[['Stars', 'Cool', 'Useful', 'Funny', 'Text Length']].corr()



fig = plt.figure(figsize=(12,8))

axes = sns.heatmap(yelp_cor, cmap='coolwarm', linewidth=1, linecolor='white', annot=True)

axes.set_title('Heatmap of Variables', fontsize=30)
sns.pairplot(yelp_df, hue='Stars', palette='coolwarm')
warnings.filterwarnings("ignore")

sns.boxplot(x='Stars', y='Text Length', data=yelp_df, palette='rainbow')
yelp_text = yelp_df[['Stars', 'Text']]

yelp_text.head()
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
yelp_text_allstars = yelp_text.copy()

yelp_text_allstars['Text'] = yelp_text_allstars['Text'].apply(remove_punc_stopword)

yelp_text_allstars.count()
yelp_text_allstars.head()
words_split = []

for i in range(0,9999): 

    for word in yelp_text_allstars['Text'][i]:

        words_split.append(word)

FreqDist(words_split).plot(30, cumulative=False)
yelp_allstars_text = yelp_text_allstars['Text'].values



wordcloud = WordCloud(

    width = 3000,

    height = 2000,

    background_color = 'black',

    stopwords = STOPWORDS).generate(str(yelp_allstars_text))



fig = plt.figure(

    figsize = (10, 7),

    facecolor = 'k',

    edgecolor = 'k')

plt.imshow(wordcloud, interpolation = 'bilinear')

plt.axis('off')

plt.tight_layout(pad=0)
warnings.filterwarnings("ignore")

yelp_text_stars1 = yelp_text[yelp_text['Stars']==1]

yelp_text_stars1['Text'] = yelp_text_stars1['Text'].apply(remove_punc_stopword)

yelp_text_stars1.count()
yelp_text_stars1 = yelp_text_stars1.reset_index(drop=True)

words_split = []

for i in range(0,749): 

    for word in yelp_text_stars1['Text'][i]:

        words_split.append(word)

FreqDist(words_split).plot(30, cumulative=False)
yelp_stars1_text = yelp_text_stars1['Text'].values



wordcloud = WordCloud(

    width = 3000,

    height = 2000,

    background_color = 'black',

    stopwords = STOPWORDS).generate(str(yelp_stars1_text))



fig = plt.figure(

    figsize = (10, 7),

    facecolor = 'k',

    edgecolor = 'k')

plt.imshow(wordcloud, interpolation = 'bilinear')

plt.axis('off')

plt.tight_layout(pad=0)
warnings.filterwarnings("ignore")

yelp_text_stars5 = yelp_text[yelp_text['Stars']==5]

yelp_text_stars5['Text'] = yelp_text_stars5['Text'].apply(remove_punc_stopword)

yelp_text_stars5.count()
yelp_text_stars5 = yelp_text_stars5.reset_index(drop=True)

words_split = []

for i in range(0,3337): 

    for word in yelp_text_stars5['Text'][i]:

        words_split.append(word)

FreqDist(words_split).plot(30, cumulative=False)
yelp_stars5_text = yelp_text_stars5['Text'].values



wordcloud = WordCloud(

    width = 3000,

    height = 2000,

    background_color = 'black',

    stopwords = STOPWORDS).generate(str(yelp_stars5_text))



fig = plt.figure(

    figsize = (10, 7),

    facecolor = 'k',

    edgecolor = 'k')

plt.imshow(wordcloud, interpolation = 'bilinear')

plt.axis('off')

plt.tight_layout(pad=0)
X = yelp_df['Text']

y = yelp_df['Stars']

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
tfidf_transformer = TfidfTransformer().fit(X)

X = tfidf_transformer.transform(X)



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
warnings.filterwarnings("ignore")

pipeline = Pipeline([('bow', CountVectorizer()), 

                     ('tfidf', TfidfTransformer()), 

                     ('classifier', MultinomialNB())])



X = yelp_df['Text']

y = yelp_df['Stars']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=31)

pipeline.fit(X_train, y_train)

y_test_pred = pipeline.predict(X_test)

print(classification_report(y_test,y_test_pred))
yelp_df12 = yelp_df[(yelp_df.Stars==1) | (yelp_df.Stars==5)]

X = yelp_df12['Text']

y = yelp_df12['Stars']

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
pipeline = Pipeline([('bow', CountVectorizer()), 

                     ('classifier', MultinomialNB())])



X = yelp_df12['Text']

y = yelp_df12['Stars']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=36)

pipeline.fit(X_train, y_train)

y_test_pred = pipeline.predict(X_test)

print(classification_report(y_test,y_test_pred))