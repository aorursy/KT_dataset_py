import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns



import os

print(os.listdir('../input/storypointsfull'))
# reading the data

dataset_file = 'appceleratorstudio'

data = pd.read_csv("../input/storypointsfull/{}.csv".format(dataset_file))



# getting the shape

data.shape
data.head()
# deleting the unnamed columns 



data = data.drop(['issuekey'], axis = 1)



# getting the shape of new data

data.shape
data.describe()
data = data[(data.storypoint == 1) | (data.storypoint == 2) | (data.storypoint == 13)]



data.head()
# adding a column to represent the length of the tweet

data['titDescription'] = data[['title', 'description']].apply(lambda x: ' '.join(x.map(str)), axis=1)

data['lenTitDescription'] = data['titDescription'].str.len()



data = data.drop(['title'], axis = 1)

data = data.drop(['description'], axis = 1)



data.head(10)
# relation between length and storypoints



plt.rcParams['figure.figsize'] = (10, 7)

sns.boxenplot(x = data['storypoint'], y = data['lenTitDescription'])

plt.title('Relation between Story Points and Title Length', fontsize = 20)

plt.show()
# checking the most common words in the whole dataset



from wordcloud import WordCloud



wordcloud = WordCloud(background_color = 'gray', width = 1000, height = 1000, max_words = 50).generate(str(data['titDescription']))



plt.rcParams['figure.figsize'] = (10, 10)

plt.title('Most Common words in the dataset', fontsize = 20)

plt.axis('off')

plt.imshow(wordcloud)
sns.countplot(x='storypoint', data=data)



#Words frequency

from sklearn.feature_extraction.text import CountVectorizer





cv = CountVectorizer()

words = cv.fit_transform(data['titDescription'].values.astype('U'))



sum_words = words.sum(axis=0)



words_freq = [(word, sum_words[0, i]) for word, i in cv.vocabulary_.items()]

words_freq = sorted(words_freq, key = lambda x: x[1], reverse = True)



frequency = pd.DataFrame(words_freq, columns=['word', 'freq'])



frequency.head(30).plot(x='word', y='freq', kind='bar', figsize=(15, 7), color = 'orange')

plt.title("Most Frequently Occuring Words - Top 30")
data.head()
import re

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer

corpusTitDescription = []



for i in range(0, 416):  

  

  review = re.sub('[^a-zA-Z]', ' ', data['titDescription'].values.astype('U')[i])

  

  review = review.lower()

  review = review.split()  

  ps = PorterStemmer()

  # stemming

  review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]

  

  # joining them back with space

  review = ' '.join(review)

  

  corpusTitDescription.append(review)



print(corpusTitDescription)
data.head(10)
# creating bag of words



from sklearn.feature_extraction.text import CountVectorizer



cv = CountVectorizer()



x_title = cv.fit_transform(corpusTitDescription).toarray()

y_title = data.iloc[:, 0]

print(x_title)

print(y_title)

print(x_title.shape)

print(y_title.shape)
#Title

# splitting the training data into train and valid sets



from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(x_title, y_title, test_size = 0.25, random_state = 42)



print(x_train.shape)

print(x_test.shape)

print(y_train.shape)

print(y_test.shape)
# standardization



from sklearn.preprocessing import StandardScaler



sc = StandardScaler()



x_train = sc.fit_transform(x_train)

x_test = sc.transform(x_test)
# Random Forest



from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report



model = RandomForestClassifier()

model.fit(x_train, y_train)



y_pred = model.predict(x_test)



print("Training Accuracy :", model.score(x_train, y_train))

print("Testing Accuracy :", model.score(x_test, y_test))



# classification report

cr = classification_report(y_test, y_pred)

print(cr)



# confusion matrix

#cm = confusion_matrix(y_test, y_pred)

#sns.heatmap(cm, annot = True)