# Import the necessary librairies for this notebook

import numpy as np

import pandas as pd



# Machine learning librairies

from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import SGDClassifier

from sklearn.feature_extraction.text import HashingVectorizer

from sklearn.cross_validation import train_test_split



# Text extraction & cleaning librairies

from nltk.corpus import stopwords



# General librairies

from subprocess import check_output

import datetime

import re



print(check_output(["ls", "../input"]).decode("utf8"))
# First load the data

df = pd.read_json('../input/train.json')
le_interest = LabelEncoder()

df['interest_level'] = le_interest.fit_transform(df['interest_level'])
# Define a function that we will use later to split our X_train dataframe

# when executing the partial_fit function

def calc_len_partial(X_train, limit=15):

    i=1

    partial_len = len(X_train)

    div_len=0

    while i:

        if partial_len%2:

            partial_len = len(X_train)/2

            div_len += 2

        elif partial_len%3:

            partial_len = len(X_train)/3

            div_len += 3

        elif partial_len%5:

            partial_len = len(X_train)/5

            div_len += 5

        elif partial_len%7:

            partial_len = len(X_train)/7

            div_len += 7

        elif partial_len%11:

            partial_len = len(X_train)/11

            div_len += 11

        else:

            break



        if div_len > limit:

            break

    

    return len(X_train)/div_len
# Define a preprocessor function that will help us clean the code

def preprocessor(text):

    text = str(text)

    text = re.sub('<[^>]*>', '', text)

    text = re.sub('[\W]+', ' ', text.lower())

    text = text.rstrip().lstrip()

    return text
stop = stopwords.words('english')

# We can also store the stop words in a pkl file

#stop = pickle.load(open('plk_objects/stopwords.pkl','rb'))



# We load Hashing Vectorize that will clean and preprocess the text

vect = HashingVectorizer(decode_error='ignore',

                         n_features=2**21,

                         preprocessor=preprocessor,

                         stop_words=stop,

                         ngram_range=(1, 3))



# For this prediction, we will use SGDC classifier

clf = SGDClassifier(loss='log',

                    random_state=1,

                    n_iter=1)
# Split the data with a majority for the training algorithm

X = df['description']

y = df['interest_level']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
print(datetime.datetime.now())



# Split the X_train dataframe into 20

len_partial = int(calc_len_partial(X_train=X_train, limit=20))



# Use partial_fit function of SGDC Classifier

# It would be too long normally and the gain in accuracy do not worth it

classes = np.array([0, 1, 2])

for i in range(round(len(X_train)/len_partial)):

    X_train_ml = X_train[i:(len_partial*(i+1))]

    y_train_ml = y_train[i:(len_partial*(i+1))]

    

    X_train_ml = vect.transform(X_train_ml)

    clf.partial_fit(X_train_ml, y_train_ml, classes=classes)

    

print(datetime.datetime.now())
print("Training accuracy: {:.3f}%".format(clf.score(vect.transform(X_train), y_train)*100))
X_test_ml = vect.transform(X_test)

print("Testing accuracy: {:.3f}%".format(clf.score(X_test_ml, y_test)*100))

clf = clf.partial_fit(X_test_ml, y_test)
# Integrate it into the general dataframe

df_desc_pred = clf.predict_proba(vect.transform(X))

df['desc_pred_0'] =  df_desc_pred[:,0]

df['desc_pred_1'] = df_desc_pred[:,1]

df['desc_pred_2'] = df_desc_pred[:,2]
df[1:3]
# Save it if you want

np.savetxt("df_desc_pred.csv", df_desc_pred, delimiter=";")