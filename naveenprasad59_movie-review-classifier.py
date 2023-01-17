import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.svm import LinearSVC

from sklearn.preprocessing import StandardScaler

from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

import matplotlib.pyplot as plt 
dataframe = pd.read_csv('../input/moviereviews.tsv',sep='\t')
dataframe.head(10)
dataframe.isnull().sum()
dataframe.dropna(inplace=True)
dataframe.isnull().sum()
blanks = []

for i,label,review in dataframe.itertuples():

    if review.isspace():

        blanks.append(i)
print("No of blank space in review:",len(blanks))
dataframe.drop(blanks,inplace=True)
dataframe['label'].value_counts()
data = dataframe['review']

label = dataframe['label']

data_train, data_test, label_train, label_test = train_test_split(data, label, test_size=0.33, random_state=42)
model = Pipeline([

                ('vectorizing',TfidfVectorizer()),

                ('SVC algorithm',LinearSVC(random_state=0, tol=1e-5,C=2)),

])

model.fit(data_train,label_train)
model_prediction = model.predict(data_test)
print(accuracy_score(label_test,model_prediction))
print(confusion_matrix(label_test,model_prediction))
print(classification_report(label_test,model_prediction))
pred = model.predict(['You will not enjoy this movie.such a piece of shit.'])

pred1 = model.predict(['A well written one.'])

print(f"The first review is predicted as : {pred}")

print(f"The second review is predicted as: {pred1}")