# imports



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

digit_df = pd.read_csv("../input/train.csv")

test_df  = pd.read_csv("../input/test.csv")



# preview the data

digit_df.head()
digit_df.info()

print("----------------------------")

test_df.info()
X_train = digit_df.drop("label",axis=1)

Y_train = digit_df["label"]

X_test = test_df
# Random Forests



random_forest = RandomForestClassifier(n_estimators=100)



random_forest.fit(X_train, Y_train)



Y_pred = random_forest.predict(X_test)



random_forest.score(X_train, Y_train)