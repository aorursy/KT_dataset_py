import numpy as np
import pandas as pd
train_df = pd.read_csv("../input/explicit-content-detection/train.csv")

train_df.head()
test_df = pd.read_csv("../input/explicit-content-detection/test.csv")

test_df.head()
train_df["target"].value_counts()
from sklearn.metrics import f1_score
X_train = train_df["title"].values
X_test = test_df["title"].values
y_train = train_df["target"].astype(int).values
y_pred = [int("порно" in text) for text in X_train]
f1_score(y_train, y_pred)
test_df["target"] = [("порно" in text) for text in X_test]

test_df[["id", "target"]].to_csv("simple_baseline.csv", index=False)

!cat simple_baseline.csv | head
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier

vectorizer = CountVectorizer()

model = DecisionTreeClassifier(max_depth=30)
X_train_vectorized = vectorizer.fit_transform(X_train)
X_train_vectorized
feature_names = np.array(vectorizer.get_feature_names())
id_ = 42

print(X_train[id_])

x_vector = X_train_vectorized.getrow(id_).toarray()[0]

#[feature for feature in feature_names[x_vector > 0]]
x_vector.shape
%%time

model.fit(
    X_train_vectorized,
    y_train
)

y_pred = model.predict(
    X_train_vectorized
)
f1_score(y_train, y_pred)
X_test_vectorized = vectorizer.transform(X_test)

test_df["target"] = model.predict(X_test_vectorized).astype(bool)

test_df[["id", "target"]].to_csv("ml_baseline.csv", index=False)

!cat ml_baseline.csv | head
