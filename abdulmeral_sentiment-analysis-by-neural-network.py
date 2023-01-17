#libraries
from keras.models import Sequential 
from keras.layers import Dense 
import matplotlib.pyplot as plt

#
from textblob import TextBlob
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble
#
from sklearn import metrics
#
import textblob, string
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers
#
#stopwords
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
#
from warnings import filterwarnings
filterwarnings('ignore')

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
seed = 7 
np.random.seed(seed)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train = pd.read_csv("/kaggle/input/imdb-dataset-sentiment-analysis-in-csv-format/Train.csv")
train.head()
train.info()
train.label.value_counts()
train.groupby("label").count()
def transformations(dataframe):
    # upper to lower character
    dataframe['text'] = dataframe['text'].apply(lambda x: " ".join(x.lower() for x in x.split()))
    #punctuations
    dataframe['text'] = dataframe['text'].str.replace('[^\w\s]','')
    #numbers
    dataframe['text'] = dataframe['text'].str.replace('\d','')
    # 
    sw = stopwords.words('english')
    dataframe['text'] = dataframe['text'].apply(lambda x: " ".join(x for x in x.split() if x not in sw))
    #rare characters deleting
    sil = pd.Series(' '.join(dataframe['text']).split()).value_counts()[-1000:]
    dataframe['text'] = dataframe['text'].apply(lambda x: " ".join(x for x in x.split() if x not in sil))
    #lemmi
    from textblob import Word
    #nltk.download('wordnet')
    dataframe['text'] = dataframe['text'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()])) 
    return dataframe
train = transformations(train)
train.head()
valid = pd.read_csv("/kaggle/input/imdb-dataset-sentiment-analysis-in-csv-format/Valid.csv")
valid = transformations(valid)
valid.head()
test = pd.read_csv("/kaggle/input/imdb-dataset-sentiment-analysis-in-csv-format/Test.csv")
test = transformations(test)
test.head()
train_x = train['text']
valid_x = valid["text"]
train_y = train["label"]
valid_y = valid["label"]
vectorizer = CountVectorizer()
vectorizer.fit(train_x)
x_train_count = vectorizer.transform(train_x)
x_valid_count = vectorizer.transform(valid_x)
x_test_count  = vectorizer.transform(test["text"])
model = Sequential() 
#layers
model.add(Dense(50,input_dim=x_train_count.shape[1], kernel_initializer="uniform", activation="relu")) 
#model.add(Dense(6, kernel_initializer="uniform", activation="relu")) 
model.add(Dense(1, kernel_initializer="uniform", activation="sigmoid")) 
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
# Fit the model
history = model.fit(x_train_count, train_y.values.reshape(-1,1), validation_data=(x_valid_count,valid_y), nb_epoch=2, batch_size=128)
# evaluate
loss, acc = model.evaluate(x_test_count, test["label"], verbose=0)
print('Test Accuracy: %f' % (acc*100))
comments = pd.Series(test["text"])
comments = vectorizer.transform(comments)
y_pred = model.predict_classes(comments)
nn_cm = metrics.confusion_matrix(test["label"],y_pred)
print(nn_cm)
import sklearn.metrics as metrics
# calculate the fpr and tpr for all thresholds of the classification
probs = model.predict_proba(x_valid_count)
preds = probs[:,:]
fpr, tpr, threshold = metrics.roc_curve(test["label"], y_pred)
roc_auc = metrics.auc(fpr, tpr)

# method I: plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.3f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
comment_1 = pd.Series("this film is very nice and good i like it")
comment_2 = pd.Series("no not good look at that shit very bad")
comment_1  = vectorizer.transform(comment_1)
comment_2 = vectorizer.transform(comment_2)
model.predict_classes(comment_1)
model.predict_classes(comment_2)