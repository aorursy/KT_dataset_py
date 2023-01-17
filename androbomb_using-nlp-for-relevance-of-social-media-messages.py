# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

print('\n ')

print('Getting traing dataset...')

training_data = pd.read_csv('../input/nlp-starter-test/social_media_clean_text.csv')

print('Traing data set obtained. \n')



print('Getting test dataset...')

test_data = pd.read_csv('../input/nlp-starter-test/test.csv')

print('Test data set obtained. \n')
training_data.head(3)
training_data['length'] = training_data['text'].apply(len)

training_data.head()
print(training_data['class_label'].unique())

print(training_data['choose_one'].unique())
plt.figure(figsize=(16,8))

c = sns.distplot(training_data['length'], kde=False)

c.axes.set_title("Length of All texts",fontsize=30)

c.set_xlabel("Length",fontsize=18)

c.set_ylabel("Frequency",fontsize=18)

plt.show()
plt.figure(figsize=(16,8))



b = sns.distplot(training_data[training_data['class_label']==1]['length'], kde=False, label = 'Relevant')

b.axes.set_title("Length of All texts",fontsize=30)

b.set_xlabel("Length",fontsize=18)

b.set_ylabel("Frequency",fontsize=18)



b = sns.distplot(training_data[training_data['class_label']==0]['length'], kde=False, label = 'Not Relevant')

b.axes.set_title("Length of All texts",fontsize=30)

b.set_xlabel("Length",fontsize=18)

b.set_ylabel("Frequency",fontsize=18)



plt.legend(fontsize=15)

plt.show()
def text_process(mess):

    """

    Takes in a string of text, then performs the following:

    1. Remove all punctuation

    2. Remove all stopwords

    3. Returns a list of the cleaned text

    """

    import string

    from nltk.corpus import stopwords



    # Check characters to see if they are in punctuation

    nopunc = [char for char in mess if char not in string.punctuation]



    # Join the characters again to form the string.

    nopunc = ''.join(nopunc)

    

    # Now just remove any stopwords

    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
text_process('True peace is not merely the absence of tension: it is the presence of justice.')
training_data['text'].head(5).apply(text_process)
from sklearn.feature_extraction.text import CountVectorizer



print('Bow transforming...')

bow_transformer = CountVectorizer(analyzer=text_process).fit(training_data['text'])

print('Done. \n')



# Print total number of vocabulary words

print('Total number of vocabulary words:')

print(len(bow_transformer.vocabulary_))
print('The original message:')

print(training_data['text'][4])

print('\n')



print('Count vectotized form:')

bow4 = bow_transformer.transform([training_data['text'][4]])

print(bow4)

print('Count vectorized shape:')

print(bow4.shape)
print('Bowing the message:')

messages_bow = bow_transformer.transform(training_data['text'])

print('Done. \n')



print('Shape of Sparse Matrix: ', messages_bow.shape)

print('Amount of Non-Zero occurences: ', messages_bow.nnz)
from sklearn.feature_extraction.text import TfidfTransformer



tfidf_transformer = TfidfTransformer().fit(messages_bow)

tfidf4 = tfidf_transformer.transform(bow4)

print(tfidf4)
messages_tfidf = tfidf_transformer.transform(messages_bow)

print(messages_tfidf.shape)
from sklearn.pipeline import Pipeline

from sklearn.naive_bayes import MultinomialNB

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.feature_extraction.text import CountVectorizer







pipeline = Pipeline([

    ('bow', CountVectorizer(analyzer=text_process)),  # strings to token integer counts

    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores

    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier

])
from sklearn.model_selection import train_test_split



print('Splitting train/test data...')

msg_train, msg_test, label_train, label_test = train_test_split(training_data['text'], training_data['class_label'], test_size=0.2)

print('Done. \n')



print('Length of train dataset:', len(msg_train))

print('Length of test dataset:', len(msg_test))

print('Length of whole dataset:', len(msg_train)+ len(msg_test))
print('Fitting the model...')

pipeline.fit(msg_train,label_train)

print('Fit done. \n')



print('Computing predictions...')

predictions = pipeline.predict(msg_test)

print('Predictions done. ')
from sklearn.metrics import classification_report



dictionary = {

    0 : 'Relevant',

    1 : 'Not Relevant', 

    2 : 'Can t Decide'

}







print('Classification report:')

print(classification_report(predictions,label_test))

print('\n')



from sklearn.metrics import confusion_matrix



print('Confusion matrix: ')

cm = confusion_matrix(label_test, predictions)

print(cm)





if (np.shape(np.array(cm))==(3,3)):

    vector = ['Relevant', 'Not Relevant', 'Can t Decide']

elif (np.shape(np.array(cm))==(2,2)):

    vector = ['Relevant', 'Not Relevant']

else: 

    pass



df_cm = pd.DataFrame(cm, index = vector, columns = vector)

plt.figure(figsize = (7,7))

sns.heatmap(df_cm, annot=True, cmap=plt.cm.Blues,  fmt='g')

plt.xlabel("Predicted Class", fontsize=18)

plt.ylabel("True Class", fontsize=18)



plt.show()