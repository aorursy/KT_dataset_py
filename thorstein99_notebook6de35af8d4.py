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


train.head(1)
import numpy as np

import pandas as pd

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



#train.fillna(value = '99999', inplace = True)



def handle_non_numeric_data(train):

    columns = df.columns.values

    

    for column in columns:

        text_digit_vals = {}

        def convert_to_int(val):

            return text_digit_vals[val]

        

        if df[column].dtype != np.int64 and df[column].dtype != np.float64:

            column_contents = df[column].values.tolist()

            unique_elements = set(column_contents)

            x = 0

            for unique in unique_elements:

                if unique not in text_digit_vals:

                    text_digit_vals[unique] = x

                    x+=1

            df[column] = list(map(convert_to_int, df[column]))

    return df



train = handle_non_numeric_data(train).copy()

#y_test2 = handle_non_numeric_data(y_test2).copy()
from sklearn.cluster import KMeans

from sklearn import preprocessing, cross_validation, svm

from sklearn.neighbors import KNeighborsClassifier
train.drop('Survived' , 1,inplace = True)

X = preprocessing.scale(train)
y = pd.read_csv('../input/train.csv')

y = y.Survived.copy()
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X ,y, test_size = 0.2)
clf = svm.SVC
clf.fit(X_train, y_train)
y_train