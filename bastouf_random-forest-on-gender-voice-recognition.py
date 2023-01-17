import numpy as np

import pandas as pd



from sklearn.ensemble import RandomForestClassifier



voice_df = pd.read_csv("../input/voice.csv")
# Splitting the dataset between training and testing set



msk = np.random.rand(len(voice_df)) < 0.8

train_df=voice_df.loc[msk]

test_df=voice_df.loc[~msk]
X_train = train_df.drop("label", axis=1)

Y_train = train_df["label"]

X_test = test_df.drop("label", axis=1)

Y_test = test_df["label"]
random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)

print("Random forrest : {}".format(random_forest.score(X_test, Y_test)))