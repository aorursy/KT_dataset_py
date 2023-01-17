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
def handle_non_numerical_values(df):

    columns = df.columns.values

    for column in columns:

        text_numerical_value = {}

        def convert_to_int(val):

            return text_numerical_value[val]

        

        if df[column].dtype!=np.int64 and df[column].dtype!=np.float64:

            column_elements = df[column].values.tolist()

            unique_elements = set(column_elements)

            x=0

            for unique in unique_elements:

                if unique not in text_numerical_value:

                    text_numerical_value[unique] = x

                    x+=1

            df[column] = list(map(convert_to_int,df[column]))

    return df
train = pd.read_csv('titanic_train.csv')

test = pd.read_csv('titanic_test.csv')



train.fillna(train.mean(),inplace=True)

test.fillna(test.mean(),inplace=True)
X_train = handle_non_numerical_values(train).drop(['PassengerId','Survived','Name','Ticket'],axis=1)

y_train = handle_non_numerical_values(train)['Survived']

X_test = handle_non_numerical_values(test).drop(['PassengerId','Name','Ticket'],axis =1)
clf = KNeighborsClassifier()

clf.fit(X_train,y_train)



predicted = clf.predict(X_test)