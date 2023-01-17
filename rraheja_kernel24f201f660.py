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

print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

from sklearn.naive_bayes import MultinomialNB

from sklearn.model_selection import GridSearchCV,cross_val_score,train_test_split

from sklearn.svm import LinearSVC,SVC

from sklearn.metrics import accuracy_score, mean_absolute_error

from sklearn.decomposition import PCA

from sklearn.ensemble import RandomForestClassifier

import time
digit_file_path = '../input/digit-recognizer/train.csv'

train = pd.read_csv(digit_file_path)



digit_file_path = '../input/digit-recognizer/test.csv'

test = pd.read_csv(digit_file_path)
X = train.iloc[:,1:]

y = train.label

train_X, val_X, train_y, val_y = train_test_split(X, y,random_state=1)
rf_param={

    "n_estimators":[100,200],

    "max_depth":[15,25]

}
S_t = time.time()

RF = RandomForestClassifier()

grid_rf=GridSearchCV(RF,rf_param,cv=5)

grid_rf.fit(train_X,train_y)

ft_t = time.time()-S_t

print("Time consumed to fit model: ",time.strftime("%H:%M:%S", time.gmtime(ft_t)))
rf_val_prediction=grid_rf.predict(val_X)

val_ac = accuracy_score(val_y,rf_val_prediction)

print("Validation Accuracy Score for Random Forest: {:.2f}".format(val_ac))
result = grid_rf.predict(test)



np.savetxt('results.csv', 

           np.c_[range(1,len(test)+1),result], 

           delimiter=',', 

           header = 'ImageId,Label', 

           comments = '', 

           fmt='%d')