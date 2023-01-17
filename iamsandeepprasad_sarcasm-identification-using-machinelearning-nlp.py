import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
data_path="../input/the-best-sarcasm-annotated-dataset-in-spanish/sarcasmo.tsv"
data=pd.read_csv(data_path, sep='\t')
data.head()
data.isnull().sum()
features=data.iloc[:,1:2]
labels_Sarcasmo=data.iloc[:,2].values
labels_Sarcasmo=pd.get_dummies(labels_Sarcasmo)
labels_Sarcasmo=labels_Sarcasmo.iloc[:,0].values
labels_Sarcasmo.shape
import nltk
nltk.download()
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
corpus=[]
lemmatizer=WordNetLemmatizer()
for i in range(len(features.Locución)):
  sent=features.Locución[i]
  sent=re.sub("[^a-zA-Z]"," ",sent)
  sent=sent.lower()
  sent=sent.split()
  
  word_wt_sw=[]
  for word in sent:
    if word not in stopwords.words("spanish"):
      word_wt_sw.append(lemmatizer.lemmatize(word))
  sentence=" ".join(word_wt_sw)
  corpus.append(sentence)

corpus[1]
features.Locución[1]
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=5000)
X=cv.fit_transform(corpus)
y=np.array(labels_Sarcasmo)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
from sklearn.naive_bayes import MultinomialNB
naivbas=MultinomialNB()
naivbas.fit(X_train,y_train)
y_pred=naivbas.predict(X_test)
from sklearn.metrics import confusion_matrix,accuracy_score
print("Accuarcy Score",accuracy_score(y_pred,y_test))
confusion_matrix(y_pred,y_test)
