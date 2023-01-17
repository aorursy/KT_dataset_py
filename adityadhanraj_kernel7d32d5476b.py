import numpy as np
import pandas as pd
df = pd.read_csv(r'../input/spam.csv',encoding='latin-1')

df.drop(labels=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace=True,axis=1)
import re
import nltk
from nltk.stem.porter import PorterStemmer
pc = PorterStemmer()
corpus = []
for i in range(0,len(df['v2'])):
    mesage = re.sub('[^a-zA-Z]',' ',df['v2'][i])
    mesage = mesage.lower().split()
    mesage = [pc.stem(word) for word in mesage]
    mesage = ' '.join(mesage)
    corpus.append(mesage)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(stop_words={'english'})
X =cv.fit_transform(corpus).toarray()
y = df.iloc[:,0].values
from sklearn.linear_model import LogisticRegression
reg = LogisticRegression()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
reg.fit(X_train,y_train)
predict = reg.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predict))
print(confusion_matrix(y_test,predict))
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = reg, X = X_train, y = y_train, cv = 10)
print("Average accuracy {}".format(accuracies.mean()))
print("Average accuracy {}".format(accuracies.mean()))