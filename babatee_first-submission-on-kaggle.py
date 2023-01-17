# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/train.csv', index_col= 'id')

print(df.head())



testset = pd.read_csv('../input/test.csv', index_col= 'id')

print(testset.head())
print(df.describe(), '\n\n ----------------')

print(pd.isnull(df).any())
y = df.type.copy()

X = df.drop('type', axis = 1)

print(X.dtypes)
#get the column of list of testset

columns = testset.columns.ravel().tolist()

#concatenate train and test sets together

concat_data = pd.concat([X, testset])



print(concat_data.tail(), '\n\n', testset.tail())

#Looks the same?
#define a function to encode a pandas.series

def encode_data(series):

    encoded = series.astype('category').cat.codes

    return encoded



y_encode = encode_data(y)

concat_data['encode_color'] = encode_data(concat_data.color)

concat_data.drop('color', axis = 1, inplace = True)
index_list = testset.index.ravel().tolist()



train_data, test_data = concat_data[~concat_data.index.isin(index_list)], concat_data[concat_data.index.isin(index_list)]

print(test_data.head(), '\n\n', train_data.head())
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train_data, y_encode, test_size = 0.3, random_state = 7)



##### Try Support Vector on Data

from sklearn.svm import SVC

svc = SVC(kernel='linear')

svc.fit(X_train, y_train)

score = svc.score(X_test, y_test)

print('Naively, SVC classifies data with {:.4}'.format(score*100) + ' accuracy')
#If not splitting training data, Accuracy is a little better 

from sklearn.model_selection import cross_val_score

crossValScore = cross_val_score(svc, train_data, y_encode, cv = 10).mean()

print('SVC model on all data classifies with {:.4}'.format(crossValScore*100) + ' accuracy')



#Fitting the model on all the data

#svc.fit(train_data, y_encode)
y_guess = svc.predict(test_data)



y_guess = pd.Series(y_guess).replace([0, 1, 2], ['Ghost', 'Ghoul', 'Goblin'])
submission = pd.DataFrame(pd.concat([pd.Series(index_list), y_guess], axis = 1))

submission.columns = ['id', 'type']

submission.to_csv('GhoulData.csv', index = False)