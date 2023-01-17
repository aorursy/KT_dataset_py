import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from keras.layers import Dense, Dropout, Activation  
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize,sent_tokenize
from bs4 import BeautifulSoup
import re,string,unicodedata
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.stem import PorterStemmer,WordNetLemmatizer
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split
from string import punctuation
from nltk import pos_tag
from nltk.corpus import wordnet
import keras
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
true = pd.read_csv("../input/noticias-falsas-en-espaol/onlytrue1000.csv")
false = pd.read_csv("../input/noticias-falsas-en-espaol/onlyfakes1000.csv")
true['category'] = 0
false['category'] = 1
df = pd.concat([true,false]) 
df.isna().sum()
df.head()
stop = set(stopwords.words('spanish'))
punctuation = list(string.punctuation)
stop.update(punctuation)
stop
stemmer = PorterStemmer()
def stem_text(text):
    final_text = []
    for i in text.split():
        if i.strip().lower() not in stop:
            word = stemmer.stem(i.strip())
            final_text.append(word)
    return " ".join(final_text)    

df.text = df.text.apply(stem_text)
x_train,x_test,y_train,y_test = train_test_split(df.text,df.category)
cv=CountVectorizer(min_df=0,max_df=1,ngram_range=(1,2))
cv_train_reviews=cv.fit_transform(x_train)
cv_test_reviews=cv.transform(x_test)

print('BOW_cv_train:',cv_train_reviews.shape)
print('BOW_cv_test:',cv_test_reviews.shape)
model = Sequential()
model.add(Dense(units = 100 , activation = 'relu' , input_dim = cv_train_reviews.shape[1]))
model.add(Dense(units = 50 , activation = 'relu'))
model.add(Dense(units = 25 , activation = 'relu'))
model.add(Dense(units = 10 , activation = 'relu'))
model.add(Dense(units = 1 , activation = 'sigmoid'))

model.add(Dropout(0.2))
model.compile(optimizer = 'adam' , loss = 'binary_crossentropy' , metrics = ['accuracy'])
model.fit(cv_train_reviews, y_train, epochs = 9)
pred = model.predict(cv_test_reviews)
for i in range(len(pred)):
    if(pred[i] > 0.5):
        pred[i] = 1
    else:
        pred[i] = 0
accuracy_score(pred,y_test)
cv_report = classification_report(y_test,pred,target_names = ['0','1'])
print(cv_report)
cm_cv = confusion_matrix(y_test,pred)
cm_cv
cm_cv = pd.DataFrame(cm_cv, index=[0,1], columns=[0,1])
cm_cv.index.name = 'Actual'
cm_cv.columns.name = 'Predicted'
plt.figure(figsize = (10,10))
sns.heatmap(cm_cv,cmap= "Blues",annot = True, fmt='')