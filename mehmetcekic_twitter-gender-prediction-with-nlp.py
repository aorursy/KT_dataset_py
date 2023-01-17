# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))


# Any results you write to the current directory are saved as output.
import pandas as pd
import numpy as np
data = pd.read_csv("../input/gender-classifier-DFE-791531.csv",encoding="latin1")
data = pd.concat([data.gender,data.description],axis=1)

#make some preperation for prediction, first change male->0 female ->1
data.gender = [1 if each =="female" else 0 for each in data.gender]
data.dropna(axis=0,inplace=True)
data.head(5)
#Now prepare text of description data for prediction. Like, making lowercase, omitting unnecessary words,stopping words etc. 
import re
import nltk
description_list = []
lemma = nltk.WordNetLemmatizer()
for description in data.description:
    description = re.sub("[^a-zA-z]"," ",description)
    description = description.lower()
    description = nltk.word_tokenize(description)
    description = [lemma.lemmatize(word) for word in description]
    description = " ".join(description)
    description_list.append(description)
print(description_list[:4])


from sklearn.feature_extraction.text import CountVectorizer
max_features = 3000
count_vect = CountVectorizer(max_features = max_features, stop_words="english")
matrix = count_vect.fit_transform(description_list).toarray()

from sklearn.model_selection import train_test_split
y = data.iloc[:,0].values
x = matrix
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=42)

from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(x_train,y_train)

y_pred = nb.predict(x_test)
print("accuracy: ",nb.score(x_test,y_test))


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators = 18, random_state = 42)
rfc.fit(x_train,y_train)

y_pred = rfc.predict(x_test)
print("accuracy: ",rfc.score(x_test,y_test))

#Model selection for best prediction
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
grid = {"n_estimators":np.arange(1,20)}
rfc = RandomForestClassifier()
rfc_cv = GridSearchCV(rfc,grid,cv=10)
rfc_cv.fit(x_train,y_train)

print("Best n_estimators value: ",rfc_cv.best_params_)
print("With best n_estimator values best score: ",rfc_cv.best_score_)