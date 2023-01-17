

import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns



import os

print(os.listdir('../input'))
# reading the data



data = pd.read_csv('../input/spam.csv', encoding = 'latin-1')



# getting the shape

data.shape
data.head()


# deleting the unnamed columns 



data = data.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis = 1)



# getting the shape of new data

data.shape
# renaming v1 and v2



data = data.rename(columns = {'v1': 'labels', 'v2': 'message'})



# getting the colums of the data

data.columns
data.head()
data.describe()
# adding a column to represent the length of the tweet



data['len'] = data['message'].str.len()

data['len'] = data['message'].str.len()



data.head(10)
# relation between spam messages and length



plt.rcParams['figure.figsize'] = (10, 7)

sns.boxenplot(x = data['labels'], y = data['len'])

plt.title('Relation between Messages and Length', fontsize = 20)

plt.show()
# distribution of length



sns.violinplot(data['len'], data['labels'])

plt.show()
# describing by labels



data.groupby('labels').describe()
# checking the most common words in the whole dataset



from wordcloud import WordCloud



wordcloud = WordCloud(background_color = 'gray', width = 1000, height = 1000, max_words = 50).generate(str(data['message']))



plt.rcParams['figure.figsize'] = (10, 10)

plt.title('Most Common words in the dataset', fontsize = 20)

plt.axis('off')

plt.imshow(wordcloud)
# let's encode the label attributes



data['labels'].replace('spam', 0, inplace = True)

data['labels'].replace('ham', 1, inplace = True)



# checking the values of the labels now

data['labels'].value_counts()
# visualize it in pie chart



size = [4825, 747]

labels = ['spam', 'ham']

colors = ['pink', 'lightblue']



plt.pie(size, colors = colors, labels = labels, shadow = True, autopct = '%.2f%%')

plt.axis('off')

plt.title('Pie Chart for Labels', fontsize = 20)

plt.legend()

plt.show()


# checking the most common words in spam messages



spam = ' '.join(text for text in data['message'][data['labels'] == 0])



wordcloud = WordCloud(background_color = 'pink', max_words = 50, height = 1000, width = 1000).generate(spam)



plt.rcParams['figure.figsize'] = (10, 10)

plt.axis('off')

plt.title('Most Common Words in Spam Messages', fontsize = 20)

plt.imshow(wordcloud)
# checking the most common words in ham messages



ham = ' '.join(text for text in data['message'][data['labels'] == 1])



wordcloud = WordCloud(background_color = 'purple', max_words = 50, height = 1000, width = 1000).generate(ham)



plt.rcParams['figure.figsize'] = (10, 10)

plt.axis('off')

plt.title('Most Common Words in Ham Messages', fontsize = 20)

plt.imshow(wordcloud)
from sklearn.feature_extraction.text import CountVectorizer





cv = CountVectorizer()

words = cv.fit_transform(data.message)



sum_words = words.sum(axis=0)



words_freq = [(word, sum_words[0, i]) for word, i in cv.vocabulary_.items()]

words_freq = sorted(words_freq, key = lambda x: x[1], reverse = True)



frequency = pd.DataFrame(words_freq, columns=['word', 'freq'])



frequency.head(30).plot(x='word', y='freq', kind='bar', figsize=(15, 7), color = 'orange')

plt.title("Most Frequently Occuring Words - Top 30")


# collecting the hashtags



def hashtag_extract(x):

    hashtags = []

    

    for i in x:

        ht = re.findall(r"#(\w+)", i)

        hashtags.append(ht)



    return hashtags



import re



# extracting hashtags from non racist/sexist tweets

HT_regular = hashtag_extract(data['message'][data['labels'] == 1])



# extracting hashtags from racist/sexist tweets

HT_negative = hashtag_extract(data['message'][data['labels'] == 0])



# unnesting list

HT_regular = sum(HT_regular,[])

HT_negative = sum(HT_negative,[])



# let's check no. of hastags

print("No. of Positive Hashtags :", HT_regular)

print("no. of negative Hastags :", HT_negative)
# removing unwanted patterns from the data



import re

import nltk



nltk.download('stopwords')

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer
corpus = []



for i in range(0, 5572):

  review = re.sub('[^a-zA-Z]', ' ', data['message'][i])

  review = review.lower()

  review = review.split()

  

  ps = PorterStemmer()

  

  # stemming

  review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]

  

  # joining them back with space

  review = ' '.join(review)

  corpus.append(review)
# creating bag of words



from sklearn.feature_extraction.text import CountVectorizer



cv = CountVectorizer()

x = cv.fit_transform(corpus).toarray()

y = data.iloc[:, 0]



print(x.shape)

print(y.shape)
# splitting the training data into train and valid sets



from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 42)



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

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot = True)