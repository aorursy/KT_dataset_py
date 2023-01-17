import pandas as pd

import numpy as np

from wordcloud import WordCloud

import matplotlib.pyplot as plt

import seaborn as sns

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer

from nltk.tokenize import word_tokenize
df = pd.read_csv("../input/1429_1.csv")
df.head()
df.info() #trying to understand all the column available in the dataset
data = df[["id","reviews.text","reviews.rating"]]

# the id has been chosen because under name there are more missing values. 

# from the description the ID represents a device sold by Amazon
data.head() 
data.describe(include=["O"])
data.info()
data = data.dropna()
rt = data['reviews.text']

wordcloud = WordCloud(background_color='white',

                      width=1000,

                      height=400

                     ).generate(" ".join(rt))

plt.figure(figsize=(10,5))

plt.imshow(wordcloud)

plt.title('All Words in the Reviews\n',size=20)

plt.axis('off')

plt.show()
words = ['awesome','great','fantastic','extraordinary','amazing','super',

                 'magnificent','stunning','impressive','wonderful','breathtaking',

                 'love','content','pleased','happy','glad','satisfied','lucky',

                 'shocking','cheerful','wow','sad','unhappy','horrible','regret',

                 'bad','terrible','annoyed','disappointed','upset','awful','hate']



rt = " ".join(data['reviews.text'])
diz = {}

for word in rt.split(" "):

    if word in words:

        diz[word] = diz.get(word,0)+1

        
wordcloud = WordCloud(background_color='white',

                      width=1000,

                      height=400

                     ).generate_from_frequencies(diz)

plt.figure(figsize=(10,5))

plt.imshow(wordcloud)

plt.title('Sentiment Words\n',size=20)

plt.axis('off')

plt.show()
plt.figure(figsize=(10,5))

sns.countplot(data['reviews.rating'])

plt.title('Count ratings')

plt.show()
data1 = data.groupby("id").mean().reset_index()
data1 = data1.sort_values(['reviews.rating']).reset_index()
plt.figure(figsize=(10,8))

sns.barplot(x=data1["reviews.rating"], y=data1["id"])

plt.title('Count ratings')

plt.show()
df2 = pd.read_csv("../input/Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv")

df3 = pd.read_csv("../input/Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv")
data2 = df2[["id","reviews.text","reviews.rating"]]

data3 = df3[["id","reviews.text","reviews.rating"]]
data2 = data2[data2["reviews.rating"]<=3]

data3 = data3[data3["reviews.rating"]<=3]
len(data2), len(data3)
plt.figure(figsize=(10,5))

sns.countplot(data2['reviews.rating'])

plt.title('Count ratings')

plt.show()
plt.figure(figsize=(10,5))

sns.countplot(data3['reviews.rating'])

plt.title('Count ratings')

plt.show()
frames = [data, data2, data3]

final = pd.concat(frames)
plt.figure(figsize=(10,5))

sns.countplot(final['reviews.rating'])

plt.title('Count ratings')

plt.show()
final.head()
#lower case all text

final["reviews.text"]=final["reviews.text"].str.lower() 



#tokenization of words

final['reviews.text'] = final.apply(lambda row: word_tokenize(row['reviews.text']), axis=1) 



#only alphanumerical values

final["reviews.text"] = final['reviews.text'].apply(lambda x: [item for item in x if item.isalpha()]) 



#lemmatazing words

final['reviews.text'] = final['reviews.text'].apply(lambda x : [WordNetLemmatizer().lemmatize(y) for y in x])



#removing useless words

stop = stopwords.words('english')

final['reviews.text'] = final['reviews.text'].apply(lambda x: [item for item in x if item not in stop])

final["reviews.text"] = final["reviews.text"].apply(lambda x: str(' '.join(x))) #joining all tokens
final.head()
sentiment = {1: 0,

            2: 0,

            3: 0,

            4: 1,

            5: 1}



final["sentiment"] = final["reviews.rating"].map(sentiment)
final.head()
len(final[final["sentiment"]==0]),len(final[final["sentiment"]==1])
# building tfidf matrix to train models 

from sklearn.feature_extraction.text import TfidfVectorizer



vectorizer =TfidfVectorizer(max_df=0.9)

text = vectorizer.fit_transform(final["reviews.text"])
from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(text, final["sentiment"], test_size=0.3, random_state=1)



# try logistic regression first

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state=1)

classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

y_pred_tr = classifier.predict(x_train)

print('Test accuracy', sum(y_test == y_pred)/len(y_test))

print('Train accuracy', sum(y_train == y_pred_tr)/len(y_train))
from sklearn.metrics import classification_report

print("Classification Report(Train)")

print(classification_report(y_train, y_pred_tr))

print("Classification Report(Test)")

print(classification_report(y_test, y_pred))
# Random Forests

from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier()

classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

y_pred_tr = classifier.predict(x_train)



print('Test accuracy', sum(y_test == y_pred)/len(y_test))

print('Train accuracy', sum(y_train == y_pred_tr)/len(y_train))
print("Classification Report(Train)")

print(classification_report(y_train, y_pred_tr))

print("Classification Report(Test)")

print(classification_report(y_test, y_pred))
from sklearn.model_selection import GridSearchCV

parameters = {"n_estimators": [10,50,100,200],

             "criterion":("gini","entropy")}

classifier = RandomForestClassifier()

clf = GridSearchCV(classifier, parameters, cv=5)

clf.fit(x_train, y_train)
#Viewing best parameters in Grid Search

best_parameter = clf.best_params_

best_accuracy = clf.best_score_ #best cros validated mean

print('Best parameter: ' + str(best_parameter))

print('Best accuracy: ' + str(best_accuracy))
classifier = RandomForestClassifier(criterion = best_parameter["criterion"], 

                                    n_estimators = best_parameter["n_estimators"])

classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

y_pred_tr = classifier.predict(x_train)

print('Test accuracy', sum(y_test == y_pred)/len(y_test))

print('Train accuracy', sum(y_train == y_pred_tr)/len(y_train))
print("Classification Report(Train)")

print(classification_report(y_train, y_pred_tr))

print("Classification Report(Test)")

print(classification_report(y_test, y_pred))
sentiment = {1: 0,

            2: 0,

            3: 1,

            4: 2,

            5: 2}



final["sentiment"] = final["reviews.rating"].map(sentiment)
final.head()
len(final[final["sentiment"]==0]),len(final[final["sentiment"]==1]),len(final[final["sentiment"]==2])
# building tfidf matrix to train models 

from sklearn.feature_extraction.text import TfidfVectorizer



vectorizer =TfidfVectorizer(max_df=0.9)

text = vectorizer.fit_transform(final["reviews.text"])
from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(text, final["sentiment"], test_size=0.3, random_state=1)



from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state=1)

classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

y_pred_tr = classifier.predict(x_train)

print('Test accuracy', sum(y_test == y_pred)/len(y_test))

print('Train accuracy', sum(y_train == y_pred_tr)/len(y_train))
from sklearn.metrics import classification_report

print("Classification Report(Train)")

print(classification_report(y_train, y_pred_tr))

print("Classification Report(Test)")

print(classification_report(y_test, y_pred))
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier()

classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

y_pred_tr = classifier.predict(x_train)



print('Test accuracy', sum(y_test == y_pred)/len(y_test))

print('Train accuracy', sum(y_train == y_pred_tr)/len(y_train))
print("Classification Report(Train)")

print(classification_report(y_train, y_pred_tr))

print("Classification Report(Test)")

print(classification_report(y_test, y_pred))
parameters = {"n_estimators": [10,50,100,200],

             "criterion":("gini","entropy")}

classifier = RandomForestClassifier()

clf = GridSearchCV(classifier, parameters, cv=5)

clf.fit(x_train, y_train)

#Viewing best parameters in Grid Search

best_parameter = clf.best_params_

best_accuracy = clf.best_score_ #best cros validated mean

print('Best parameter: ' + str(best_parameter))

print('Best accuracy: ' + str(best_accuracy))
classifier = RandomForestClassifier(criterion = best_parameter["criterion"] , 

                                    n_estimators = best_parameter["n_estimators"])

classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

y_pred_tr = classifier.predict(x_train)
print('Test accuracy', sum(y_test == y_pred)/len(y_test))

print('Train accuracy', sum(y_train == y_pred_tr)/len(y_train))
print("Classification Report(Train)")

print(classification_report(y_train, y_pred_tr))

print("Classification Report(Test)")

print(classification_report(y_test, y_pred))
from keras.preprocessing.sequence import pad_sequences

from keras.preprocessing.text import Tokenizer

from keras.models import Sequential

from keras.layers import Dense, Embedding, GRU, Dropout, LSTM

from keras.callbacks import EarlyStopping

from keras.optimizers import Adam
t = Tokenizer()
t.fit_on_texts(final["reviews.text"])
max_length = max([len(s.split()) for s in final["reviews.text"] ])

max_legth=max_length #the max length is aroun 1000 character. I would keep it shorter. 
vocab_size = len(t.word_index)+1
X_train, X_test, y_train, y_test = train_test_split(final["reviews.text"], final["sentiment"], test_size=0.25)

X_train = t.texts_to_sequences(X_train)

X_test = t.texts_to_sequences(X_test)
X_train = pad_sequences(X_train, maxlen=max_length, padding = "post",truncating = "post")

X_test = pad_sequences(X_test, maxlen=max_length, padding = "post", truncating = "post")
X_train = X_train[0:28290]

y_train = y_train[0:28290]

X_test = X_test[0:9430]

y_test = y_test[0:9430]

len(y_test),len(X_test),len(X_train),len(y_train)
from keras.utils import np_utils #converting to categorical

y_train = np_utils.to_categorical(y_train, num_classes=3)

y_test = np_utils.to_categorical(y_test, num_classes=3)
embedding_dim = 200

model = Sequential()

model.add(Embedding(vocab_size, embedding_dim, input_length=max_length))

model.add(LSTM(units = 32))

model.add(Dense(3,activation="softmax")) #since converted to categorical we will have three output nodes. softmax

                                         # assigns a probability distribution

    

model.compile(loss="binary_crossentropy", optimizer="adam",metrics=["accuracy"])

model.fit(X_train, y_train, batch_size=10, epochs=3)
model.evaluate(x=X_test, y=y_test, batch_size=10)
