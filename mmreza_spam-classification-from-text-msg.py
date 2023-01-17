# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from matplotlib import pyplot as plt

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import nltk

import string

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer

from wordcloud import WordCloud



from sklearn.feature_extraction.text import CountVectorizer

from sklearn.preprocessing import LabelEncoder



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/SPAM text message 20170820 - Data.csv')

df.head(5)
# get all spam and ham massage as list

spam_data = df[df['Category']=='spam'].Message.tolist()

ham_data = df[df['Category']=='ham'].Message.tolist()

# join all lists

spam_data = ' '.join(spam_data)

ham_data = ' '.join(ham_data)
# remove punctuation and stop words from data

punctuation = string.punctuation

stopwords = stopwords.words('english')
# remove punctuation

filtered_ham_data = ''.join(i for i in ham_data if i not in punctuation)

filtered_spam_data = ''.join(i for i in spam_data if i not in punctuation)

filtered_spam_data[:200]
# remove stopwords

filtered_ham_data = ' '.join(i for i in filtered_ham_data.lower().split() if i not in stopwords)

filtered_spam_data = ' '.join(i for i in filtered_spam_data.lower().split() if i not in stopwords)

filtered_spam_data[:200]
# lemmatizer

lemmatizer = WordNetLemmatizer()

filtered_ham_data = ' '.join([lemmatizer.lemmatize(i) for i in filtered_ham_data.split()])

filtered_spam_data = ' '.join([lemmatizer.lemmatize(i) for i in filtered_spam_data.split()])

filtered_spam_data[:200]
# Generate WordCloud for Spam

x, y = np.ogrid[:300, :300]

mask = (x - 150) ** 2 + (y - 150) ** 2 > 130 ** 2

mask = 255 * mask.astype(int)



wc = WordCloud(max_font_size=40, max_words=200, background_color='white', random_state=1337, mask=mask).generate(filtered_spam_data)

plt.figure(figsize=(10,10))

plt.imshow(wc, interpolation='bilinear')

plt.axis("off")

plt.title("Spam Words", fontsize=20)

plt.show()
# Generate WordCloud for Ham

wc = WordCloud(max_font_size=40, max_words=200, background_color='white', random_state=1337).generate(filtered_ham_data)

plt.figure(figsize=(10,10))

plt.imshow(wc, interpolation='bilinear')

plt.axis("off")

plt.title("Ham Words", fontsize=20)

plt.show()
def data_clean(msg):

    w = ''.join(i for i in msg if i not in punctuation)

    w = ' '.join(i for i in w.lower().split() if i not in stopwords)

    lemmatizer = WordNetLemmatizer()

    lem = ' '.join(lemmatizer.lemmatize(i) for i in w.split())

    return lem
df['preprocess_message']=""

for i in range(df['Message'].count()):

    df['preprocess_message'][i]= data_clean(df['Message'][i])

    #df.iloc[i]['preprocess_message']= df.iloc[i]['Message']

df.head()
cv = CountVectorizer()

data = cv.fit_transform(df.preprocess_message)
le = LabelEncoder()

label = le.fit_transform(df.Category)
print(df.shape, data.shape, label.shape)
from sklearn.model_selection import train_test_split, KFold

from sklearn import tree

from sklearn.metrics import mean_absolute_error as mae

train_set_X, test_X, train_set_Y, test_Y = train_test_split(data, label, test_size=0.2)



clf = tree.DecisionTreeClassifier(random_state=1337)

clf.fit(train_set_X, train_set_Y)
from sklearn.metrics import accuracy_score



test_pred = clf.predict(test_X)

acc = accuracy_score(test_pred, test_Y)

print("Test accuracy {}".format(acc))