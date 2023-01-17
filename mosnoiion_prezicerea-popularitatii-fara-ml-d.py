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

        

import matplotlib.pyplot as plt

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/girlsgoit-competition-2020/train.csv')

data.head()
test = pd.read_csv('/kaggle/input/girlsgoit-competition-2020/test.csv')

test.head()
submit = pd.read_csv('/kaggle/input/girlsgoit-competition-2020/sampleSubmission.csv')

submit.info()
data.describe()
data.info()
data.corr()
data.columns
data[["year", "popularity"]].groupby(["year",'popularity']).count().max(level=0)
#data[["year", "popularity"]].groupby('year').count()

max_per_year = data.groupby(['year'])['popularity'].apply(lambda x: x.value_counts().index[0]).reset_index()

max_per_year
plt.plot(max_per_year['year'],max_per_year['popularity'])
data[data['year']==1921]['popularity'].hist()
toPopularity = {i:j for [i,j] in max_per_year.to_numpy()}

toPopularity[2010]
def pred_for_year(x):

    """functia care prezice popularitatea dupa an"""

    return  x['year'].apply(lambda x: toPopularity[x])

pred_for_year(data)
columns = [ 'acousticness', 'danceability', 'duration_ms',

       'energy', 'explicit', 'instrumentalness', 'key', 'liveness', 'loudness',

       'mode', 'speechiness', 'tempo', 'valence',

       'year']

columns = ['year']#,'acousticness','danceability','energy','loudness']

X = data[columns]

X.head()
X.describe()
Y = data[['popularity']]

Y.head()
from sklearn.model_selection import train_test_split



dataX, dataY = X, Y

train_ratio = 0.70

validation_ratio = 0.15

test_ratio = 0.15



# train is now 75% of the entire data set

# the _junk suffix means that we drop that variable completely

x_train, x_test, y_train, y_test = train_test_split(dataX, dataY, test_size=1 - train_ratio)



# test is now 10% of the initial data set

# validation is now 15% of the initial data set

x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=test_ratio/(test_ratio + validation_ratio)) 



print(x_train.shape, x_val.shape, x_test.shape)
from sklearn.tree import DecisionTreeClassifier



# clf = DecisionTreeClassifier()#(max_depth=9)
# clf.fit(x_train, y_train)
from sklearn.tree import DecisionTreeClassifier, plot_tree



import matplotlib.pyplot as plt

plt.figure(figsize=(20, 20))

#plot_tree(clf)
Y_predict =  pred_for_year(x_val)#clf.predict(x_val) #
from sklearn.metrics import accuracy_score, f1_score, recall_score



print("Accuracy score:", accuracy_score(y_val, Y_predict))

print("F1 score:", f1_score(y_val, Y_predict, average='micro'))

print("Recall score:", recall_score(y_val, Y_predict, average='micro'))
from sklearn.metrics import confusion_matrix



cm = confusion_matrix(y_val, Y_predict)

cm
columns = ['year' , 'acousticness', 'danceability', 'energy', 'duration_ms',

        'explicit', 'instrumentalness', 'key', 'liveness', 'loudness',

       'mode', 'speechiness', 'tempo', 'valence'

       ]

info = []

for c in range(1,4):

    break

    for leaf in range(3, 30,3):

        for deapth in range(3, 30,3):

            cs = columns[:c]

            X = data[cs]

            x_train, x_test, y_train, y_test = train_test_split(X, dataY, test_size=1 - train_ratio)

            x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=test_ratio/(test_ratio + validation_ratio)) 

            

            clf = DecisionTreeClassifier(random_state=1, max_leaf_nodes=leaf, max_depth=deapth)

            clf.fit(x_train, y_train)

            Y_predict = clf.predict(x_val)

            info.append([','.join(cs), leaf, deapth , accuracy_score(y_val, Y_predict), clf])

            
plt.title("Cum influenteaza X  asupra acuratetei")

plt.plot([ i[-2] for i in info], 'x')

plt.xlabel('params & colums')

plt.ylabel('Accuracy score')
so = sorted(info, key=lambda x:x[-2])[-5:]
test_data = pd.read_csv('https://girlsgoitpublic.z6.web.core.windows.net/test.csv')



# todo: preprocesare datele text_data ca la train.csv

# columns = [...]



test_data['popularity'] = pred_for_year(test_data[columns])# so[-1][-1].predict(test_data[so[-1][0].split(',')])# clf.predict(test_data[columns])

test_data[['ID', 'popularity']].to_csv('submission.csv',index=False)