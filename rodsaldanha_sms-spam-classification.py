from nltk.tokenize import word_tokenize

import en_core_web_sm

from nltk.corpus import stopwords

import matplotlib.pyplot as plt

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score, cross_val_predict

from sklearn import metrics

from keras.models import Sequential

from keras import layers

from sklearn.preprocessing import LabelEncoder 

from sklearn.model_selection import StratifiedKFold

from sklearn.pipeline import Pipeline
colnames=['label', 'sms']

filepath = '../input/sms-data-labelled-spam-and-non-spam/SMSSpamCollection'



data = pd.read_csv(filepath, sep="\t", header=None, names=colnames)

data.head()
le = LabelEncoder() 

  

data['target']= le.fit_transform(data['label']) 
sentences = data['sms'].values

y = data['target'].values
clf = Pipeline([('vect', CountVectorizer()), ('classifier', LogisticRegression())])



scores = cross_val_score(clf, sentences, y, cv=6, scoring='accuracy')

print('Cross-validation Accuracy:',scores, 'Average:',scores.mean())



predictions = cross_val_predict(clf, sentences, y, cv=6)

accuracy = metrics.accuracy_score(y, predictions)

print('Cross-Predicted Accuracy:', accuracy)
def create_model(X_train):

    input_dim = X_train.shape[1]  # Number of features



    model = Sequential()

    model.add(layers.Dense(10, input_dim=input_dim, activation='relu'))

    model.add(layers.Dense(1, activation='sigmoid'))

    

    model.compile(loss='binary_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])

    return model
X_train, X_test, y_train, y_test = train_test_split(sentences, y, test_size=0.2, random_state=1000)



vectorizer = CountVectorizer()

vectorizer.fit(X_train)



X_train = vectorizer.transform(X_train)

X_test  = vectorizer.transform(X_test)



model = create_model(X_train)



history = model.fit(X_train, y_train,

                    epochs=100,

                    verbose=True,

                    validation_data=(X_test, y_test),

                    batch_size=10)
loss, accuracy = model.evaluate(X_train, y_train, verbose=False)

print("Training Accuracy: {:.4f}".format(accuracy))

loss, accuracy = model.evaluate(X_test, y_test, verbose=False)

print("Testing Accuracy:  {:.4f}".format(accuracy))
plt.style.use('ggplot')



def plot_history(history):

    acc = history.history['accuracy']

    val_acc = history.history['val_accuracy']

    loss = history.history['loss']

    val_loss = history.history['val_loss']

    x = range(1, len(acc) + 1)



    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)

    plt.plot(x, acc, 'b', label='Training acc')

    plt.plot(x, val_acc, 'r', label='Validation acc')

    plt.title('Training and validation accuracy')

    plt.legend()

    plt.subplot(1, 2, 2)

    plt.plot(x, loss, 'b', label='Training loss')

    plt.plot(x, val_loss, 'r', label='Validation loss')

    plt.title('Training and validation loss')

    plt.legend()

    

plot_history(history)
predictions = model.predict(X_test)

predictions = [ int(x) for x in predictions ]

accuracy = metrics.accuracy_score(y_test, predictions)

print('Prediction Accuracy:', accuracy)