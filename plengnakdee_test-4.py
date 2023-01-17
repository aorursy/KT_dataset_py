import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from sklearn.model_selection import train_test_split
df = pd.read_csv('../input/swiftdata/swift-2.csv')

#df.head()



# drop columns with NaN or 0

data = df.drop(columns=['comment', 'comment_id', 'order_comment', 'local_code'])

data = data.loc[df['category'] == 'auto\n']

data.head()
# segment by demographic

gender = data.groupby('gender').count().reset_index()

gender
plt.figure(figsize=(10,6))

plt.title("Swift Market Segmentation, by Gender")

ax = sns.barplot(x=gender['gender'], y=gender['user_id'])

plt.xlabel("Gender")

plt.ylabel("Number of People")

ax.set_xticklabels(['Male', 'Female', 'Other'])

plt.show()
# segment by geographic

region = data.groupby('region').count().reset_index()

region
plt.figure(figsize=(10,6))

plt.title("Swift Market Segmentation, by Geographic")

sns.barplot(x=region['region'], y=region['user_id'])

plt.xlabel("Region")

plt.ylabel("Number of People")

plt.xticks(rotation=45)

plt.show()
from keras.preprocessing.text import Tokenizer
male_data = data.loc[(data['category'] == 'auto\n') & (data['gender'] == 1 )]

male_data.head()
female_data = data.loc[(data['category'] == 'auto\n') & (data['gender'] == 2 )]

female_data.head()
bkk_data = data.loc[(data['category'] == 'auto\n') & (data['region'] == 'Bangkok' )]

bkk_data.head()
positive_vocab = []

negative_vocab = []

swear_words = []



with open("../input/swiftdata/negative-sentiment-words.txt", 'r') as f:

    for line in f:

        negative_vocab.append(line.rstrip())



with open("../input/swiftdata/positive-sentiment-words.txt", 'r') as f:

    for line in f:

        positive_vocab.append(line.rstrip())

        

with open("../input/swiftdata/swear-words.txt", 'r') as f:

    for line in f:

        swear_words.append(line.rstrip())
#sentences = ['วันนี้ ดี','วันนี้ แย่' , 'วันนี้ ธรรมดา']

sentences = female_data['title'].apply(str)



#tokenizer = Tokenizer()

#tokenizer.fit_on_texts(data[:,0])

#tokenizer.fit_on_texts(sentences)

#sentences = tokenizer.word_index



for sentence in sentences:

    neg = 0

    pos = 0

    print(sentence)

    words = sentence.split(' ')

    for word in words:

        if word in positive_vocab:

            pos = pos + 1

        if word in negative_vocab or word in swear_words:

            neg = neg + 1



    if pos > neg:

        print('positive')

    elif neg > pos:

        print('negative')

    else:

        print('neutral')