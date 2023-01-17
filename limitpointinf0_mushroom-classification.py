import numpy as np # linear algebra

import pandas as pd

from sklearn.cluster import KMeans

from sklearn import preprocessing,cross_validation,neighbors



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



df_o = pd.read_csv("../input/mushrooms.csv")
df = handle_non_numeric(df_o)

X = np.array(df.drop(['class'],1).astype(float))

X = preprocessing.scale(X)

y = np.array(df['class'])



clf = KMeans(n_clusters = 2)

clf.fit(X)



correct = 0

for i in range(len(X)):

    predict_me = np.array(X[i].astype(float))

    predict_me = predict_me.reshape(-1,len(predict_me))

    prediction = clf.predict(predict_me)

    if prediction[0] == y[i]:

        correct += 1

print('accuracy',correct/len(X))
df = handle_non_numeric(df_o)

X = np.array(df.drop(['class'],1).astype(float))

X = preprocessing.scale(X)

y = np.array(df['class'])



X_train,X_test,y_train,y_test = cross_validation.train_test_split(X,y,test_size=0.2)



clf = neighbors.KNeighborsClassifier()

clf.fit(X_train,y_train)

accuracy = clf.score(X_test,y_test)

print('accuracy',accuracy)