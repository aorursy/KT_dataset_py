import numpy as np # linear algebra

import pandas as pd

from sklearn import preprocessing,cross_validation,neighbors

from sklearn.model_selection import cross_val_score



def handle_non_numeric(df):

    columns = df.columns.values

    for col in columns:

        text_digit_vals = {}

        def convert_to_int(val):

            return text_digit_vals[val]

        

        if df[col].dtype != np.int64 and df[col].dtype != np.float64:

            column_contents = df[col].values.tolist()

            unique_elements = set(column_contents)

            x = 0

            for unique in unique_elements:

                if unique not in text_digit_vals:

                    text_digit_vals[unique] = x

                    x += 1

            df[col] = list(map(convert_to_int,df[col]))

    return df



df = pd.read_csv("../input/data.csv")

df = df.drop('Unnamed: 32',1)

df = df.drop('id',1)

df = handle_non_numeric(df)

print(df.info())
X = np.array(df.drop('diagnosis',1))

X = preprocessing.scale(X)

y = np.array(df['diagnosis'])



clf = neighbors.KNeighborsClassifier()

acc = cross_val_score(clf, X, y, cv=10)

print(np.mean(acc))