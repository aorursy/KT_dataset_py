import os
print(os.listdir("../input"))
import pandas as pd
import numpy as np
# Encoding the data using only the first columns: the other seems to be an issue of the data (empty)

df = pd.read_csv('../input/spam.csv', sep=',', encoding='latin-1', usecols=lambda col: col not in ["Unnamed: 2","Unnamed: 3","Unnamed: 4"])
df.head(1)
df = df.rename(columns={"v1":"label", "v2":"text"})
df.head(5)
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer()
# Splitting the data into training and test

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(df["text"],df["label"], test_size = 0.2, random_state = 10)
# Fitting the CountVectorizer using the training data

vect.fit(X_train)
# Transforming the dataframes

X_train_df = vect.transform(X_train)
X_test_df = vect.transform(X_test)
type(X_train_df)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train_df,y_train)
# Making predictions

prediction = dict()

prediction["Logistic"] = model.predict(X_test_df)
# Reviewing the metrics

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
accuracy_score(y_test,prediction["Logistic"])
print(classification_report(y_test,prediction["Logistic"]))