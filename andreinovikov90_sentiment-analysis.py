import numpy as np

import pandas as pd

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#Читаем файлы.

NBChealth = pd.read_csv("/kaggle/input/task-1/NBChealth.txt", sep="|", header=None)

NBChealth["chanel"] = "NBChealth"

latimeshealth = pd.read_csv("/kaggle/input/task-1/latimeshealth.txt", sep="|", header=None)

latimeshealth["chanel"] = "latimeshealth"

msnhealthnews = pd.read_csv("/kaggle/input/task-1/msnhealthnews.txt", sep="|", header=None)

msnhealthnews["chanel"] = "msnhealthnews"

#собираем файлы в один файл

data = pd.concat([NBChealth, latimeshealth])

data = pd.concat([data, msnhealthnews])

del data[0]

del data[1]

data["text"] = data[2]

del data[2]

print(data)
sentiment_train = pd.read_csv("/kaggle/input/sentiment-amazon/amazon.csv", sep=";")

sentiment_train["chanel"] = "Amazon"

data = pd.concat([data, sentiment_train])

print(data)
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()



bow_data = vectorizer.fit_transform(data["text"].apply(lambda x: str(x).lower()))

bow_data = bow_data.todense()



new_df = pd.DataFrame(bow_data)

data = data.reset_index()

bow_data = pd.concat([data,new_df], axis=1)

del bow_data["text"]

del bow_data["index"]

print(bow_data)
columns=[]

for i in range(22856):

    columns.append(i)

sentimental_train_vectors = bow_data[bow_data["chanel"]=="Amazon"][columns]



from sklearn.model_selection import train_test_split



train_target = sentiment_train["mark"]

train_features = sentimental_train_vectors



from sklearn.naive_bayes import MultinomialNB



model = MultinomialNB()



for i in range(5):

    X_train, X_test, y_train, y_test = train_test_split(train_features, train_target, test_size=0.2)

    model.fit(X_train, y_train)

    print("model accuracy is ", model.score(X_test, y_test))

model.fit(train_features, train_target)
NBChealth_pred = model.predict(bow_data[bow_data["chanel"]=="NBChealth"][columns])

latimeshealth_pred = model.predict(bow_data[bow_data["chanel"]=="latimeshealth"][columns])

msnhealthnews_pred = model.predict(bow_data[bow_data["chanel"]=="msnhealthnews"][columns])
print("NBChealth mean positivity", NBChealth_pred.mean())

print("NBChealth polarization", NBChealth_pred.std())

print("latimeshealth mean positivity", latimeshealth_pred.mean())

print("latimeshealth polarization", latimeshealth_pred.std())

print("msnhealthnews mean positivity", msnhealthnews_pred.mean())

print("msnhealthnews polarization", msnhealthnews_pred.std())