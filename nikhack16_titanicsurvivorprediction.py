# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

from sklearn import svm



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



df = pd.read_csv('../input/train.csv')

df.head()



X = df[['Age','Pclass','Sex','Fare']]

y = df['Survived']







X.head()



#X['Sex'].isnull().sum()





X['Age'] = X['Age'].fillna(X['Age'].median())



X['Age'].isnull().sum()



d = {'male':0,'female':1}

X['Sex'] = X['Sex'].apply(lambda x:d[x])

X['Sex'].head()



X_train ,X_test,y_train, y_test = train_test_split(X,y,test_size = 0.33,random_state = 42)



clf = svm.LinearSVC()



clf.fit(X_train,y_train)



pred = clf.predict(X_test)

actual = np.array(y_test)

np.mean(pred == actual)


