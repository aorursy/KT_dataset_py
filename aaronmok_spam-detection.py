import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from nltk.tokenize import word_tokenize, sent_tokenize

from textblob import TextBlob

from nltk.corpus import stopwords

from string import punctuation

import seaborn as sns

from sklearn.preprocessing import Normalizer, StandardScaler

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import precision_recall_fscore_support

from sklearn.metrics import roc_curve, roc_auc_score

from sklearn.model_selection import train_test_split

from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

from imblearn.over_sampling import SMOTE

from keras.models import Sequential

from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding

from keras.optimizers import Adam

from keras.callbacks import EarlyStopping

from xgboost import XGBClassifier

#load the dataframe and rename the columns so that it's easier to understand



df = pd.read_csv('../input/spam.csv', encoding='latin-1')

df = df.iloc[:,:2]

df = pd.get_dummies(df, columns = ['v1'], drop_first = True)

df.rename(columns = {'v1_spam': 'Spam'}, inplace = True)

df.rename(columns = {'v2': 'Text'}, inplace = True)

df.head()
#check whats the balance between spam and ham 



sns.countplot(df['Spam'])

plt.title('# of Spam vs Ham')



df['Spam'].value_counts()
def AmountUpper(x):

    count = 0

    for letter in x:

        if letter.isupper():

            count = count + 1

            

    return count



df['Count_Upper'] = df['Text'].apply(AmountUpper)

df.head()
df.groupby(['Spam'])['Count_Upper'].mean().plot(kind = 'bar')

plt.ylabel('Average Upper Letters in Text')

plt.title('Upper Letters Spam vs Ham')
#preprocessing the data

#remove stopwords

removeWords = set(stopwords.words('english')+list(punctuation))





# Use TextBlob for stemming

def textblob_tokenizer(str_input):

    blob = TextBlob(str_input.lower())

    tokens = blob.words

    words = [token.stem() for token in tokens]

    return words



#tfidfvectorizer

vectorizer = TfidfVectorizer(lowercase=False, tokenizer = textblob_tokenizer, stop_words=removeWords)

X = vectorizer.fit_transform(df['Text'])

text_columns = vectorizer.get_feature_names()

#X_df = pd.SparseDataFrame(X,columns = text_columns, default_fill_value=0)



#X_df['Count_Upper'] = df['Count_Upper']



#Normalizer Scaler - Text

stdScaler = Normalizer()

X = stdScaler.fit_transform(X)



X_train, X_test, y_train, y_test = train_test_split(X, df['Spam'], test_size=0.25, random_state=101)
len(text_columns)
#logistic Regression

model = LogisticRegression()

model.fit(X_train,y_train)

score = model.score(X_test,y_test)

print('Logistic Regression')

print(score)
#support vector machine with a grid search

def SVM_Model_Reports(X_train,y_train,X_test,y_test):

    model = SVC()



    parameters = [

        {'kernel':['linear', 'poly', 'rbf'],

         'C':[1,10,100]}

                 ]



    Grid = GridSearchCV(model, parameters, cv = 4)

    Grid.fit(X_train,y_train)

    means = Grid.cv_results_['mean_test_score']

    stds = Grid.cv_results_['std_test_score']

    params = Grid.cv_results_['params']

    print('SVM with Grid Search')

    for mean, std, params in zip(means, stds, params):

        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

    

#create the confusion matrix

    

    y_pred = Grid.predict(X_test)

    _,recall,_,_ = precision_recall_fscore_support(y_test,y_pred)

    confusion_array= confusion_matrix(y_test,y_pred)

    confusion_df = pd.DataFrame(confusion_array, columns = ['Pred 0','Pred 1'], index = ['True 0', 'True 1'])

    sns.heatmap(confusion_df, annot = True, cmap="YlGnBu", fmt='g')

    plt.title(f'Recall {recall}')

    

    print('========================================================')

    

    print('Classification Report')

    print(classification_report(y_test, y_pred))
SVM_Model_Reports(X_train,y_train,X_test,y_test)

#recall of 87% can be improved
sm = SMOTE(random_state=101)

X_train_res, y_train_res = sm.fit_sample(X_train, y_train)



SVM_Model_Reports(X_train_res,y_train_res,X_test,y_test)
length = X_train.shape[1]



# simple early stopping

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience = 5)



model = Sequential()

model.add(Dense(64,input_shape=(length,) , activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(128,activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(32,activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(1, activation='sigmoid'))

model.summary()

model.compile(loss='binary_crossentropy',

              optimizer='adam',

              metrics=['acc'])
history = model.fit(X_train_res, y_train_res,batch_size=32,epochs=10, callbacks=[es], verbose=1,validation_split=0.2)
y_pred = model.predict_classes(X_test)



print('Classification Report from Kera Sequential Model')

print(classification_report(y_test, y_pred))
# plot training history

plt.plot(history.history['loss'], label='train')

plt.plot(history.history['val_loss'], label='test')

plt.legend()

plt.show()
model = XGBClassifier()

model.fit(X_train_res,y_train_res)

y_pred = model.predict(X_test)

predictions = [round(value) for value in y_pred]



print('Classification Report from XGBoost Model')

print(classification_report(y_test, predictions))