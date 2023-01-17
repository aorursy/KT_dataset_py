# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
fashion_mnist_df = pd.read_csv('/kaggle/input/fashionmnist/fashion-mnist_train.csv')
fashion_mnist_df.head(10)
fashion_mnist_df['label'].unique()
fashion_mnist_df = fashion_mnist_df.sample(frac=0.3).reset_index(drop=True)
import matplotlib.pyplot as plt
LOOKUP = {0: 'T-shirt',
         1: 'Trouser',
         2: 'Pullover',
         3: 'Dress',
         4: 'Coat',
         5: 'Sandal',
         6: 'Shirt',
         7: 'Sneaker',
         8: 'Bag',
         9: 'Ankle boot'}
def display_image(features, actual_label):
    print('Actual label: ', LOOKUP[actual_label])
    plt.figure(figsize=(12, 10))
    plt.imshow(features.reshape(28,28))
X = fashion_mnist_df[fashion_mnist_df.columns[1:]]
Y = fashion_mnist_df['label']
display_image(X.loc[5].values, Y.loc[5])
display_image(X.loc[15].values, Y.loc[15])
display_image(X.loc[500].values, Y.loc[500])
# Scale
X = X / 255
X.head()
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
x_train.shape, y_train.shape
import sklearn.metrics as m
def summarize_classification(y_test, y_pred, avg_method='weighted'):
	acc = m.accuracy_score(y_test, y_pred, normalize=True)
	num_acc = m.accuracy_score(y_test, y_pred, normalize=False)
	prec = m.precision_score(y_test, y_pred, average=avg_method)
	rec = m.recall_score(y_test, y_pred, average=avg_method)
	print('Test data count\taccuracy_count\taccuracy_score\tprecision_score\trecall_score')
	print(30 * '-+-')
	print('%.7f\t%.7f\t%.7f\t%.7f\t%.7f'%(len(y_test), 
	                                      num_acc,
	                                      acc,
	                                      prec,
	                                      rec))
	print()
	print()
from sklearn.linear_model import LogisticRegression
logistic_model = LogisticRegression(solver='sag', multi_class='auto', max_iter=10000).fit(x_train, y_train)
y_pred = logistic_model.predict(x_test)
summarize_classification(y_test, y_pred)
