#important libraries
import pandas as pd
import numpy as np
import nltk
import re
#importing stopwords is optional, in this case it decreased accuracy
#from nltk.corpus import stopwords
import itertools
import time
start_time = time.time()
data = pd.read_csv('../input/text-emotion/text_emotion.csv')
from nltk.stem.wordnet import WordNetLemmatizer 
lem = WordNetLemmatizer()
#comprehensive cleaning
def cleaning(text):
    txt = str(text)
    txt = re.sub(r"http\S+", "", txt)
    if len(txt) == 0:
        return 'no text'
    else:
        txt = txt.split()
        index = 0
        for j in tqdm(range(len(txt))):
            if txt[j][0] == '@':
                index = j
        txt = np.delete(txt, index)
        if len(txt) == 0:
            return 'no text'
        else:
            words = txt[0]
            for k in range(len(txt)-1):
                words+= " " + txt[k+1]
            txt = words
            txt = re.sub(r'[^\w]', ' ', txt)
            if len(txt) == 0:
                return 'no text'
            else:
                txt = ''.join(''.join(s)[:2] for _, s in itertools.groupby(txt))
                txt = txt.replace("'", "")
                txt = nltk.tokenize.word_tokenize(txt)
                #data.content[i] = [w for w in data.content[i] if not w in stopset]
                for j in range(len(txt)):
                    txt[j] = lem.lemmatize(txt[j], "v")
                if len(txt) == 0:
                    return 'no text'
                else:
                    return txt
data['content'] = data['content'].map(lambda x: cleaning(x))
from tqdm import tqdm
print(data['content'])
data = data.reset_index(drop=True)
for i in tqdm(range(len(data))):
    words = data.content[i][0]
    for j in range(len(data.content[i])-1):
        words+= ' ' + data.content[i][j+1]
    data.content[i] = words
print(data.content)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn import svm
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data.content, data.sentiment, test_size=0.25, random_state=0)

x_train = x_train.reset_index(drop = True)
x_test = x_test.reset_index(drop = True)

y_train = y_train.reset_index(drop = True)
y_test = y_test.reset_index(drop = True)
vectorizer = TfidfVectorizer(min_df=3, max_df=0.9)

train_vectors = vectorizer.fit_transform(x_train)
test_vectors = vectorizer.transform(x_test)

model = svm.SVC(kernel='linear') 
model.fit(train_vectors, y_train) 
predicted_sentiment = model.predict(test_vectors)
test_content = "shut up"
test_content = [test_content]
test_vector = vectorizer.transform(test_content)
predicted =  model.predict(test_vector)
print(predicted)
covid_data_1 = pd.read_csv('../input/coronavirus-covid19-tweets-late-april/2020-04-16 Coronavirus Tweets.CSV')
covid_data_part_1 = covid_data_1[:1000]
covid_data_part_1['text'] = covid_data_part_1['text'].map(lambda x: cleaning(x))
!pip install langdetect
!pip install guess_language-spirit
from langdetect import detect
from guess_language import guess_language
covid_data_part_1 = covid_data_part_1.reset_index(drop=True)
for i in tqdm(range(len(covid_data_part_1))):
    words = covid_data_part_1.text[i][0]
    for j in range(len(covid_data_part_1.text[i])-1):
        words+= ' ' + covid_data_part_1.text[i][j+1]
    covid_data_part_1.text[i] = words
print(len(covid_data_part_1))
new_words = []
for i in (range(len(covid_data_part_1))):
    if(guess_language(covid_data_part_1.text[i]) == "en"):
        new_words.append(covid_data_part_1.text[i])
print(new_words)
final_data =  vectorizer.transform(new_words)
final_predicted =  model.predict(final_data)
print(final_predicted)
prediction_df = pd.DataFrame({'Content':new_words, 'Emotion_predicted':final_predicted})
prediction_df.to_csv('emotion_recognizer_svm.csv', index = False)
elapsed_time = time.time() - start_time
print ("processing time:", elapsed_time, "seconds")