# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

#pd.options.display.max_columns = 9999 #Maximum columns



import warnings

warnings.filterwarnings("ignore")



import missingno as msno

import matplotlib.pyplot as plt



import re

import spacy



from collections import Counter



import plotly.express as px



from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, accuracy_score

from sklearn.model_selection import cross_val_score



from xgboost import XGBClassifier
df = pd.read_csv('../input/newyork-room-rentalads/room-rental-ads.csv')

df.sample(5)
df.info()
df.describe()
df.dtypes
df.isnull().sum().any()
df.isnull().sum()
msno.bar(df)

plt.show()
df.dropna(how='any', inplace=True)
df["Vague/Not"].value_counts()
df.rename(columns = {"Vague/Not":"Target"},inplace = True)

df.Target = df.Target.astype("int").astype("category")

df
#check for duplicates



len(df[df.duplicated()])
df = df.drop_duplicates(subset=['Description'])

print(df.head())

print(df.shape)
#Normalisation using spaCy



nlp = spacy.load('en')



def normalize(msg):

    

    msg = re.sub('[^A-Za-z]+', ' ', msg) #remove special character and intergers

    doc = nlp(msg)

    res=[]

    for token in doc:

        if(token.is_stop or token.is_punct or token.is_currency or token.is_space or len(token.text) <= 2): #Remove Stopwords, Punctuations, Currency and Spaces

            pass

        else:

            res.append(token.lemma_.lower())

    return res
df["Description"] = df["Description"].apply(normalize)

df.head()
words_collection = Counter([item for sublist in df['Description'] for item in sublist])

freq_word_df = pd.DataFrame(words_collection.most_common(20))

freq_word_df.columns = ['frequently_used_word','count']



freq_word_df.style.background_gradient(cmap='Blues', low=0, high=0, axis=0, subset=None)
fig = px.bar(freq_word_df, x='frequently_used_word', y='count', color='count', title='Most frequent words')

fig.show()
df["Description"] = df["Description"].apply(lambda m : " ".join(m))
c = TfidfVectorizer(ngram_range=(1,2)) # Convert our strings to numerical values

mat=pd.DataFrame(c.fit_transform(df["Description"]).toarray(),columns=c.get_feature_names(),index=None)

mat
X = mat

y = df["Target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
classifier = XGBClassifier()

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

print(cm)

accuracy_score(y_test, y_pred)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)

print("Accuracy: {:.2f} %".format(accuracies.mean()*100))

print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))