# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split #split data into train and test sets
from sklearn.feature_extraction.text import CountVectorizer #convert text comment into a numeric vector
from sklearn.feature_extraction.text import TfidfTransformer #use TF IDF transformer to change text vector created by count vectorizer
from sklearn.svm import SVC# Support Vector Machine
from sklearn.pipeline import Pipeline #pipeline to implement steps in series
from gensim import parsing # To stem data

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory


#Read csv into a dataframe
df = pd.read_csv("../input/Sharktankpitchesdeals.csv")
#print first 5 rows of dataset
print(df.head())

# Any results you write to the current directory are saved as output.
#for grouping similar words such as 'trying" and "try" are same words
def parse(s):
    parsing.stem_text(s)
    return s

#applying parsing to comments.
for i in range(0,len(df)):
    df.iloc[i,2]=parse(df.iloc[i,2])
    
#Seperate data into feature and results
X, y = df['Pitched_Business_Desc'].tolist(), df['Deal_Status'].tolist()

#Split data in train and test sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)



#Use pipeline to carry out steps in sequence with a single object
#SVM's rbf kernel gives highest accuracy in this classification problem.
text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', SVC(kernel='rbf'))])

#train model
text_clf.fit(X_train, y_train)


#predict class form test data 
predicted = text_clf.predict(X_test)

