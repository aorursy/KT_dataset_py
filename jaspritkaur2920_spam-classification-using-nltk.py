from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords

import matplotlib.pyplot as plt

from wordcloud import WordCloud

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import string

import pandas as pd

%matplotlib inline
# loading data



mails = pd.read_csv("../input/sms-spam-collection-dataset/spam.csv",encoding = 'latin-1')

mails.head()
# drop the null columns namely Unnamed: 2, Unnamed: 3 and Unnamed: 4



mails.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1,inplace=True)

mails.head()
# renaming the columns v1 and v2 as labels and message



mails.rename(columns = {'v1':'labels', 'v2':'message'}, inplace=True)

mails.head()
# count of labels



mails['labels'].value_counts()
mails.shape
# now, we will see if our dataset contains duplicates, we will drop the duplicates



mails.drop_duplicates(inplace=True)
# after droping duplicates let's see the shape of dataset again



mails.shape
# check for any null values in the dataset



mails.isnull().sum()
# mapping the labels as 0 or 1

# 0 for ham and 1 for spam



mails['label'] = mails['labels'].map({'ham': 0, 'spam': 1})

mails.head()
# now, labels column is of no use so we will drop the labels columns



mails.drop(['labels'], axis=1, inplace=True)

mails.head()
def process_text(mess):

    """

    Takes in a string of text, then performs the following:

    1. Remove all punctuation

    2. Remove all stopwords

    3. Returns a list of the cleaned text

    """

    # Check characters to see if they are in punctuation

    nopunc = [char for char in mess if char not in string.punctuation]



    # Join the characters again to form the string.

    nopunc = ''.join(nopunc)

    

    # Now just remove any stopwords

    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
#show the tokenization 



mails['message'].head().apply(process_text)
spam_words = ' '.join(list(mails[mails['label'] == 1]['message']))

spam_wc = WordCloud(width = 512,height = 512).generate(spam_words)

plt.figure(figsize = (10, 8), facecolor = 'k')

plt.imshow(spam_wc)

plt.axis('off')

plt.tight_layout(pad = 0)

plt.show()
ham_words = ' '.join(list(mails[mails['label'] == 0]['message']))

ham_wc = WordCloud(width = 512,height = 512).generate(ham_words)

plt.figure(figsize = (10, 8), facecolor = 'k')

plt.imshow(ham_wc)

plt.axis('off')

plt.tight_layout(pad = 0)

plt.show()
# Convert a collection of text documents to a matrix of token counts

# message_bow stands for bag of words



from sklearn.feature_extraction.text import CountVectorizer

x = mails['message']

y = mails['label']

cv = CountVectorizer()

x= cv.fit_transform(x)
# split the data into 80% training and 20% testing



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)
#shape of data after Vectorization



x.shape
# Create and train the naive Bayes classifier

# The multinomial Naive Bayes classifier is suitable for classification with discrete features (e.g., word counts for text classification)



from sklearn.naive_bayes import MultinomialNB

classifier = MultinomialNB().fit(X_train, y_train)
# Evaluate the model on the test data set



from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

pred = classifier.predict(X_test)

print(classification_report(y_test, pred))

print()

print('Confusion Matrix:\n',confusion_matrix(y_test, pred))

print()

print('Accuracy : ',accuracy_score(y_test, pred))
# print the predictions

print(classifier.predict(X_test))



# print the actual values

print(y_test.values)
def sms(text):

    

    # creating a list of labels

    lab = ['not spam','spam'] 

    

    # perform tokenization

    x = cv.transform(text).toarray()

    

    # predict the text

    p = classifier.predict(x)

    

    # convert the words in string with the help of list

    s = [str(i) for i in p]

    a = int("".join(s))

    

    # show out the final result

    res = str("This message is looking: "+ lab[a])

    print(res)
sms(['Congratulations, your entry into our contest last month made you a WINNER! goto our website to claim your price! You have 24 hours to claim.'])
sms(['Your mobile number has won 1,615,000 million pounds in Apple iPhone UK. For claim email your name, country, occupation.'])
sms(['Our Summer Sale is live from 15th to 17th june! Get 40% off on select products. visit the store now.'])
sms(['Hey there! I am using kaggle'])
sms(['Did you here about the new tv show?'])
sms(['I am free after 11:00 in morning, meet you soon!'])