import pandas as pd

import pickle



DATA_PATH = "/kaggle/input/defi-ia-insa-toulouse"

train_df = pd.read_json(DATA_PATH+"/train.json")

test_df = pd.read_json(DATA_PATH+"/test.json")

train_label = pd.read_csv(DATA_PATH+"/train_label.csv")
train_df["description_lower"] = [x.lower() for x in train_df.description]

test_df["description_lower"] = [x.lower() for x in test_df.description]
from sklearn.feature_extraction.text import TfidfVectorizer

transformer = TfidfVectorizer().fit(train_df["description_lower"].values)

print("NB features: %d" %(len(transformer.vocabulary_)))

X_train = transformer.transform(train_df["description_lower"].values)

X_test = transformer.transform(test_df["description_lower"].values)

X_train
from sklearn.linear_model import LogisticRegression

Y_train = train_label.Category.values

model = LogisticRegression()

model.fit(X_train, Y_train)
predictions = model.predict(X_test)

predictions
test_df["Category"] = predictions

baseline_file = test_df[["Id","Category"]]

baseline_file.to_csv("/kaggle/working/baseline.csv", index=False)