# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



df_train = pd.read_csv("../input/train.csv")

df_test = pd.read_csv("../input/test.csv")

#We have 42,000 training images and we have 28,000 testing images.

#print('We have {} training images'.format(len(pd.read_csv('../input/train.csv'))))

#print('We have {} testing images'.format(len(pd.read_csv('../input/test.csv'))))

# Any results you write to the current directory are saved as output.
#Note: we only receive an output tab when we actually output a file, like below.

#np.savetxt('/kaggle/working/image.csv', df.iloc[0, 1:].values.reshape(28, 28), delimiter=',')

df_train.head()
#Stored in csv, label is the first column, features are the remaining columns.

from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df_train.loc[:, 'pixel0':].values, 

                                                    df_train['label'].values)
#So we have created our test and training sets. Now we can do some machine learning.
#Start with a basic linear regression

from sklearn.linear_model import LinearRegression

linear = LinearRegression()

linear.fit(X_train, y_train)

linear.score(X_test, y_test)
X_test[0]