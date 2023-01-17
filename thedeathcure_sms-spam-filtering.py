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
import sys
import nltk
import sklearn
import pandas
import numpy
print('Python: {}'.format(sys.version))
print('NLTK: {}'.format(nltk.__version__))
print('Scikit-learn: {}'.format(sklearn.__version__))
print('Pandas: {}'.format(pandas.__version__))
print('Numpy: {}'.format(numpy.__version__))
## import pandas and numpy 
import pandas as pd
import numpy as np
# load the dataset of SMS messages
df = pd.read_table('../input/SMSSpamCollection', header=None, encoding='utf-8') # don't use latin-1
df.info()
df.describe() ## describe the data
## i am explaining the data, it has two columns and 5572 rows. Every msg is either Ham or Spam. So it has two unique values.
print(df.shape)
#show some data from the top of dataset 
print(df.head())
print(df.head()[0])
print(df.head()[1])
df.columns
df[1][2]
# check class distribution
classes = df[0]
print(classes.value_counts())
from sklearn.preprocessing import LabelEncoder
# so convert spam to 1 and ham tabso 0
encoder = LabelEncoder()
y = encoder.fit_transform(classes)

# see what changes are made by the label encoder
# list(y)
# for i in y:
#     print(i, end =" ")
print(y)
print(type(y))
text_messages = df[1]
print(text_messages[:10])
# Replace email addresses with 'email'
# you can use any regex expression they are basically taken from the wikipedia

processed = text_messages.str.replace(r'^.+@[^\.].*\.[a-z]{2,}$',
                                 'emailaddress')
# Replace URLs with 'webaddress'
# you can use any regex expression they are basically taken from the wikipedia

processed = processed.str.replace(r'^http\://[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,3}(/\S*)?$',
                                  'webaddress')
# Replace money symbols with 'moneysymb' (£ can by typed with ALT key + 156)
# you can use any regex expression they are basically taken from the wikipedia

processed = processed.str.replace(r'£|\$', 'moneysymb')
    
# Replace 10 digit phone numbers (formats include paranthesis, spaces, no spaces, dashes) with 'phonenumber'
# you can use any regex expression they are basically taken from the wikipedia

processed = processed.str.replace(r'^\(?[\d]{3}\)?[\s-]?[\d]{3}[\s-]?[\d]{4}$',
                                  'phonenumbr')
# Replace numbers with 'numbr'
# you can use any regex expression they are basically taken from the wikipedia

processed = processed.str.replace(r'\d+(\.\d+)?', 'numbr')
# Remove punctuation
# you can use any regex expression they are basically taken from the wikipedia

processed = processed.str.replace(r'[^\w\d\s]', ' ')

# Replace whitespace between terms with a single space
processed = processed.str.replace(r'\s+', ' ')

# Remove leading and trailing whitespace
processed = processed.str.replace(r'^\s+|\s+?$', '')

#as HORse horse Horse are same SO conver are letters to lower case
processed = processed.str.lower()
processed # here is the data after proessing....
# Now you have to remove stopwords, these are common words, use in every sentence and make nosense in prediction
nltk.download('stopwords')
from nltk.corpus import stopwords
s = stopwords.words('english')
#print(set(s))
processed = processed.apply(lambda x: ' '.join(
    term for term in x.split() if term not in s))
# Remove word stems using a Porter stemmer
ps = nltk.PorterStemmer() # it removes the synonyms and similar sounding words..

processed = processed.apply(lambda x: ' '.join(
    ps.stem(term) for term in x.split()))
# for i in processed:
#     print(i) # just checking everything at this point
    # here you can see effects of stemming
    # crazy -> crazi
    #early, earli, earlii -> earli
nltk.download('punkt')
from nltk.tokenize import word_tokenize

all_words = []

for message in processed:
    words = word_tokenize(message)
    for w in words:
        all_words.append(w)

all_words = nltk.FreqDist(all_words)
print(len(all_words))
print(all_words.most_common(100)) ## most common 100 words in bag of words
#visualizing the most common 20 words ...

import matplotlib.pyplot as plt
x = []
er = []
for i in all_words.most_common(20):
    x.append(i[0])
for i in all_words.most_common(20):
    er.append(i[1])

plt.figure(figsize=(60, 8))
plt.subplot(131)
plt.xlabel("FREQUENCY")
plt.ylabel("WORDS")
plt.title("20 most common words from bag of words")
plt.bar(x, er)
# now word featured
word_features = list(all_words.keys()) #using all most common words as features to increase accuracy
def find_features(msg):
    words = word_tokenize(msg)
    features = {}
    for word in word_features:
        features[word] = (word in words)
    return features
messages = zip(processed, y)
type(processed)
print(y[0:10])
print(processed[5])
# Now lets do it for all the messages 
# just merge the processed data and label Encoder o/p together
messages = zip(processed, y)

# define a seed for reproducibility
seed = 1
np.random.seed = seed
#np.random.shuffle(messages)

# call find_features function for each SMS message
featuresets = [(find_features(text), label) for (text, label) in messages]
from sklearn import model_selection
training, testing = model_selection.train_test_split(featuresets, test_size = 0.20, random_state = seed)
print(len(training)) # length of training data
print(len(testing)) #length oftesting data
from nltk.classify.scikitlearn import SklearnClassifier
#SVM classsifier (support vector machine)
from sklearn.svm import SVC
model1 = SklearnClassifier(SVC(kernel = 'linear'))
model1.train(training)
accuracy = nltk.classify.accuracy(model1, testing)
#import math
print("SVC Classifier accuracy {}%".format(round(accuracy * 100,4)))
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
txt_features, labels = zip(*testing)
prediction = model1.classify_many(txt_features)
print(classification_report(prediction,labels))
import pickle
Pkl_Filename = "Spam_Model.pkl"  
with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(model1, file)
with open(Pkl_Filename, 'rb') as file:  
    my_model = pickle.load(file)

my_model
#saved_model = pickle.dumps(model1) 
#model1 = pickle.loads(saved_model) 
#model1.classify_many(txt_features)
