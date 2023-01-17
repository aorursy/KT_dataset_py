# This Python 2 environment comes with many helpful analytics libraries installed

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



from IPython.display import display

import matplotlib.pyplot as plt

import seaborn as sns

import re



from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer



from sklearn.naive_bayes import MultinomialNB

from sklearn.linear_model import LogisticRegression,SGDClassifier



from sklearn.multiclass import OneVsRestClassifier

from sklearn.preprocessing import LabelEncoder

from sklearn.svm import LinearSVC





from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import make_scorer,accuracy_score



from keras.models import Sequential

from keras.layers import Dense, Dropout

from keras.wrappers.scikit_learn import KerasClassifier

from keras.utils import np_utils



from sklearn.metrics import log_loss

# Any results you write to the current directory are saved as output.



%matplotlib inline
train = pd.read_json('../input/train.json', orient='columns')

test = pd.read_json('../input/test.json', orient='columns')

sample_submission = pd.read_csv("../input/sample_submission.csv")
display(train.head())

display(test.head())

display(sample_submission.head())
## convert multi-word ingredient into single word by substituting underscore on place of space

def sub_space(x):

    temp_value = list()

    for i in x:

        temp_value.append(re.sub(r'[^0-9a-zA-Z]+','_',i.lower()))

    return temp_value



train['ingredients_new'] = train['ingredients'].apply(sub_space)

test['ingredients_new'] = test['ingredients'].apply(sub_space)



## convert list of ingredients into a sentence

def convert_list_to_sent(x):

    return ' '.join(x)



train['ingredient_sent'] = train['ingredients_new'].apply(convert_list_to_sent)

test['ingredient_sent'] = test['ingredients_new'].apply(convert_list_to_sent)



display(train.head())

display(test.head())

X_train, X_val, y_train, y_val = train_test_split(train['ingredient_sent'], train['cuisine'], test_size=0.33, random_state=42)
## Getting Features using TfIDFVectorizer

tfidf_vect = TfidfVectorizer(lowercase=True,binary=True)



# binary value in feature set

X_train_tfidf = tfidf_vect.fit_transform(X_train)

X_val_tfidf = tfidf_vect.transform(X_val)

X_test_tfidf = tfidf_vect.transform(test['ingredient_sent'])

lb = LabelEncoder()

y_train_encode = lb.fit_transform(y_train)

y_val_encode = lb.transform(y_val)



y_train_dummy = np_utils.to_categorical(y_train_encode)

y_val_dummy = np_utils.to_categorical(y_val_encode)
input_shape = X_train_tfidf.shape[1]

def model_structure1():    

    mdl = Sequential()

    mdl.add(Dense(512, init='glorot_uniform', activation='relu',input_shape=(input_shape,)))

    mdl.add(Dropout(0.5))

    mdl.add(Dense(128, init='glorot_uniform', activation='relu'))

    mdl.add(Dropout(0.5))

    mdl.add(Dense(20, activation='softmax'))

    mdl.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

    mdl.summary()

    return mdl 
print("Compile model ...")

estimator = KerasClassifier(build_fn=model_structure1, epochs=10, batch_size=128)
# estimator.fit(X_train_tfidf.toarray(), y_train_dummy)

history = estimator.fit(X_train_tfidf.toarray(), y_train_dummy,\

                        validation_data=(X_val_tfidf.toarray(),y_val_dummy))

import matplotlib.pyplot as plt

plt.style.use('ggplot')



def plot_history(history):

    acc = history.history['acc']

    val_acc = history.history['val_acc']

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
# Predictions on train and validation data



mnb_train_prediction = estimator.predict_proba(X_train_tfidf.toarray())

mnb_val_prediction = estimator.predict_proba(X_val_tfidf.toarray())



mnb_tr_pred_value = estimator.predict(X_train_tfidf.toarray())

mnb_val_pred_value = estimator.predict(X_val_tfidf.toarray())

mnb_test_pred_value = estimator.predict(X_test_tfidf.toarray())

print("=====")

print("Log loss for TfIDFVectorizer features in MLP for Training set {}".format(log_loss(y_train_encode,mnb_train_prediction)))

print("Log loss for TfIDFVectorizer features in MLP for validation set {}".format(log_loss(y_val_encode,mnb_val_prediction)))

print("Accuracy for TfIDFVectorizer features in MLP for Training set {}".format(accuracy_score(y_train_encode,mnb_tr_pred_value)))

print("Accuracy for TfIDFVectorizer features in MLP for validation set {}".format(accuracy_score(y_val_encode,mnb_val_pred_value)))
mnb_val_pred_value
test_pred = list(lb.inverse_transform(mnb_test_pred_value))

print(test_pred)
result = pd.DataFrame({'id':test['id'],'cuisine':test_pred})

result.head()

result.to_csv('submission.csv',index=False)