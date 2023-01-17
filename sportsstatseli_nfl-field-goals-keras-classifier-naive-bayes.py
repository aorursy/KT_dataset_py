# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.naive_bayes import *

import matplotlib.pyplot as plt



data = pd.read_csv("/kaggle/input/fieldgoals/4.csv")

data_filter = data[["Good?","Dist"]]

data_filter.Dist=data_filter.Dist.astype(str)



examples = data_filter.iloc[480:]



data_filter = data_filter.drop([x for x in range(479,500,1)])



plt.figure(0)

plt.subplot(211)

data_filter['Good?'].value_counts().plot.bar()

plt.subplot(212)

data_filter['Good?'].value_counts().plot.pie()

plt.show()
data = data.replace(to_replace = ['Y','N'],value = ['1','0'])

data
data2 = data[["Dist", "Good?", "Blk?"]]

#data2

data2["Good?"] = pd.to_numeric(data2["Good?"], downcast="float")

data2["Blk?"] = pd.to_numeric(data2["Blk?"], downcast="float")
data2.describe()
all_features = data2[['Dist', 'Blk?']].values

all_classes = data2['Good?'].values

feature_names = ['Dist', 'Blk?']



from sklearn import preprocessing



scaler = preprocessing.StandardScaler()

all_features_scaled = scaler.fit_transform(all_features)

#all_features_scaled
from tensorflow.keras.layers import Dense

from tensorflow.keras.models import Sequential



def create_model():

    model = Sequential() 

    model.add(Dense(2, input_dim=2, kernel_initializer='normal', activation='relu'))

    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
from sklearn.model_selection import cross_val_score

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier



estimator = KerasClassifier(build_fn=create_model, epochs=20, verbose=1)



cv_scores = cross_val_score(estimator, all_features_scaled, all_classes, cv=10)
cv_scores.mean()
vectorizer = CountVectorizer()

counts = vectorizer.fit_transform(data_filter['Dist'].values)



classifier = ComplementNB()

targets = data_filter['Good?'].values

classifier.fit(counts, targets)
test = examples.drop(columns= ["Good?"])

test1 = [x for x in test.values]

predict = []

for y in test1:

    example_counts = vectorizer.transform(y)

    predictions = classifier.predict(example_counts)

    predict.append(predictions)

    print(predictions)

examples.head()
test2 = examples.drop(columns= ["Dist"])

test3 = [x for x in test2.values]



correct = [i for i, j in zip(predict, test3) if i == j]



print("The accuracy is:")

print(len(correct)/len(test3))
classifier2 = MultinomialNB()

classifier2.fit(counts, targets)



predict2 = []

for y in test1:

    example_counts = vectorizer.transform(y)

    predictions2 = classifier2.predict(example_counts)

    predict2.append(predictions2)



correct2 = [i for i, j in zip(predict2, test3) if i == j]



print("The accuracy is:")

print(len(correct2)/len(test3))    

examples2 = ['55', '49', '33', '22', '13', '38', '57', '45', '24']

example_counts2 = vectorizer.transform(examples2)

predictions3 = classifier.predict(example_counts2) ###Complement 

predictions4 = classifier2.predict(example_counts2) ###Bernoulli

print(predictions3)

print(predictions4)
