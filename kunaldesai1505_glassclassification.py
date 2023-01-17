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
import pandas as pd

import numpy as np

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split

df = pd.read_csv("../input/glass.csv", sep = ',')

X = np.array(df[['Na','Al','Si','K','Ca']])

y = np.array(df['Type'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 300)

clf = KNeighborsClassifier(n_neighbors=7)

clf.fit(X_train,y_train)

print(clf.score(X_test,y_test))

#print(df.head())