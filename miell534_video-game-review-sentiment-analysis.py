#Essentials

%matplotlib inline

import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



#Web Scraping

import requests #request to scrape GameSpot's website

from bs4 import BeautifulSoup #turn its html into a manipulatable string

import seaborn as sns #graphs



#Text Classification

from textblob import TextBlob #text classification and sentiment

from textblob import classifiers

import nltk #formatting and cleaning english words

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize

nltk.download('punkt')

nltk.download('stopwords')



#Machine Learning and Predictive Modeling

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression #machine learning through logistic regression

from sklearn.linear_model import LinearRegression #machine learning through linear regression

from sklearn.preprocessing import StandardScaler  #more machine learning methods and tools

from sklearn.pipeline import Pipeline

from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import PolynomialFeatures

from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import PowerTransformer
#scrape GameSpot for top and bottom game reviews



gs_game_data = []



for i in range(1, 75):

    pos_url = "https://www.gamespot.com/reviews/?sort=gs_score_desc&page=" + str(i)

    neg_url = "https://www.gamespot.com/reviews/?sort=gs_score_desc&page=" + str(710-i)

    pos_request = requests.get(pos_url)

    neg_request = requests.get(neg_url)

    pos_soup = BeautifulSoup(pos_request.text, 'lxml')

    neg_soup = BeautifulSoup(neg_request.text, 'lxml')

    pos_game_data = pos_soup.find_all('article', {'class':'media-game'})

    neg_game_data = neg_soup.find_all('article', {'class':'media-game'})

    gs_game_data.append(pos_game_data)

    gs_game_data.append(neg_game_data)
#make sure to clean the data as much as possible by trying to find each element and if there is none, append 'N/A'



game_names = []

game_scores = []

game_taglines = []

game_boxarts = []

game_systems = []

game_links = []

game_classifiers = []

for i in range(len(gs_game_data)):

    for j in range(8):

        try:

            name = gs_game_data[i][j].find('h3', {'class':'media-title'}).text

            game_names.append(name.replace("Review", ""))

        except:

            game_names.append('N/A')

        try:

            game_scores.append(gs_game_data[i][j].find('span', {'class':'content'}).text)

        except:

            game_scores.append('N/A')

        try:

            game_taglines.append(gs_game_data[i][j].find('p', {'class':'media-deck'}).text)

        except:

            game_taglines.append('N/A')

        try:

            game_boxarts.append(gs_game_data[i][j].find('img').get('src'))

        except:

            game_boxarts.append('N/A')

        try:

            game_systems.append(gs_game_data[i][j].find('li', {'class':'system--pill'}).text)

        except:

            game_systems.append('N/A')

        try:

            game_links.append("https://www.gamespot.com" + str(gs_game_data[i][j].find('a').get('href')))

        except:

            game_links.append('N/A')

        if (float(gs_game_data[i][j].find('span', {'class':'content'}).text) > 5):

            game_classifiers.append('pos')

        else:

            game_classifiers.append('neg')
gamespot_game_data = ({'name':game_names, 'score':game_scores, 'tagline':game_taglines, 'boxart':game_boxarts, 'system':game_systems, 'link':game_links, 'classifier':game_classifiers})

gamespot_game_reviews = pd.DataFrame(data=gamespot_game_data)
#get full article text from each of the webpages

#WARNING: this takes a few minutes to run



full_articles = []

for i in range(gamespot_game_reviews.shape[0]):

    url = gamespot_game_reviews.link[i]

    gs_request = requests.get(url)

    gs_soup = BeautifulSoup(gs_request.text, 'lxml')

    article = gs_soup.find_all('section', {'class':'article-body'})

    full_articles.append(article)

    

article_text = []

for i in range(len(full_articles)):

    try:

        article_text.append(full_articles[i][0].find('div').text)

    except:

        article_text.append('N/A')

        

gamespot_game_reviews['article'] = article_text
#filtering most common words with nltk

#https://www.geeksforgeeks.org/removing-stop-words-nltk-python/



gamespot_game_reviews['filtered article'] = gamespot_game_reviews['article']



for i in range(gamespot_game_reviews.shape[0]):

    stop_words = set(stopwords.words('english')) 

    word_tokens = word_tokenize(gamespot_game_reviews['article'][i]) 

    filtered_sentence = [w for w in word_tokens if not w in stop_words] 

    filtered_sentence = [] 

    for w in word_tokens: 

        if w not in stop_words: 

            filtered_sentence.append(w)

    gamespot_game_reviews['filtered article'][i] = filtered_sentence
#training and testing in textblob

#https://www.analyticsvidhya.com/blog/2018/02/natural-language-processing-for-beginners-using-textblob/



training = []

testing = []

for i in range(100):

    training.append((gamespot_game_reviews['filtered article'][i], gamespot_game_reviews['classifier'][i]))

    training.append((gamespot_game_reviews['filtered article'][len(gamespot_game_reviews)-1-i], gamespot_game_reviews['classifier'][len(gamespot_game_reviews)-1-i]))
gs_classifier = classifiers.NaiveBayesClassifier(training)

print (gs_classifier.accuracy(testing))

gs_classifier.show_informative_features()
#add new columns to data for polarity and sujectivity of sentiment



gamespot_game_reviews['polarity'] = gamespot_game_reviews['score']

gamespot_game_reviews['subjectivity'] = gamespot_game_reviews['score']

for i in range(gamespot_game_reviews.shape[0]):

    gamespot_game_reviews['polarity'][i] = TextBlob(gamespot_game_reviews['article'][i]).sentiment.polarity

    gamespot_game_reviews['subjectivity'][i] = TextBlob(gamespot_game_reviews['article'][i]).sentiment.subjectivity
#method 1: LinearRegression

lr = LinearRegression()

lr.fit(gamespot_game_reviews[['polarity', 'subjectivity']], gamespot_game_reviews['score'])

print("Intercept: ", lr.intercept_, "\nCoefficients: ", lr.coef_)

preds = lr.predict(gamespot_game_reviews[['polarity', 'subjectivity']])

mse = mean_squared_error(gamespot_game_reviews['score'], preds)

print("MSE: ", mse)

rmse = np.sqrt(mse)

print("RMSE: ", rmse)
#method 2: train test split



X_train, X_test, y_train, y_test = train_test_split(gamespot_game_reviews[['polarity', 'subjectivity']], gamespot_game_reviews['score'])

lr2 = LinearRegression()

lr2.fit(X_train, y_train)

print("Intercept: ", lr2.intercept_, "\nCoefficients: ", lr2.coef_)

preds2 = lr2.predict(X_test)

mse2 = mean_squared_error(y_test, preds2)

print('MSE: ', mse2)

rmse2 = np.sqrt(mse2)

print('RMSE: ', rmse2)
#method 3: grid search cv



params = {'poly__degree':[1, 2, 3], 'lr__fit_intercept':[True, False]}

pipe = Pipeline([('poly', PolynomialFeatures()), ('lr', LinearRegression())])

grid = GridSearchCV(pipe, param_grid=params)

grid.fit(X_train, y_train)

preds3 = grid.predict(X_test)

mse3 = mean_squared_error(y_test, preds3)

print('MSE: ', mse3)

rmse3 = np.sqrt(mse3)

print('RMSE: ', rmse3)
#method 4: gradient boosting



from sklearn import ensemble

from sklearn.preprocessing import scale



X = scale(X_train)



#Model:



params = {'n_estimators': 100, 'max_depth': 3}

rf = ensemble.GradientBoostingRegressor(**params)

rfc = rf.fit(X, y_train)



# R:

R = rfc.score(X, y_train)

print('R^2 Score: {:0.4f}'.format(R))



# Predictions

y_pred = rf.predict(X)

RMSE = mean_squared_error(y_train, y_pred)**0.5

print('RMSE: {:0.3f}'.format(RMSE))

print('Minimum LE: {:0.1f}'.format(y_pred.min()))

print('Maximum LE: {:0.1f}'.format(y_pred.max()))

print('Average Predicted LE: {:0.1f}'.format(y_pred.mean()))

print('LE Standard Deviation: {:0.3f}'.format(y_pred.std()))

print('LE Variance: {:0.3f}'.format(y_pred.std()**2))