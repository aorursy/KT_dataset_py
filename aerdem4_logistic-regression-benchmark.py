import pandas as pd

print("reading the data...")
train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")
print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)
target = "bankrupt"

print("preprocessing...")
train_df = pd.get_dummies(train_df.fillna(0), columns=["country"])
test_df = pd.get_dummies(test_df.fillna(0), columns=["country"])
print("Columns:", list(train_df.columns))
from sklearn.model_selection import train_test_split

train_df, val_df = train_test_split(train_df, test_size=0.2, 
                                    stratify=train_df[target], shuffle=True, random_state=0)
print("Train shape:", train_df.shape)
print("Validation shape:", val_df.shape)
from sklearn.linear_model import LogisticRegression

numeric_features = ["A_score", "B_score", "C_score", "D_score", "num_employees",
                    "country_CN", "country_EN", "country_NL", "country_TR", "country_US",
                    "revenue2014", "revenue2015", "revenue2016"]

lr = LogisticRegression(solver="liblinear")
print("training...")
lr.fit(train_df[numeric_features], train_df[target])

print("Feature weights:")
for feature, weight in zip(numeric_features, lr.coef_[0].tolist()):
    print(feature, weight)
from sklearn.metrics import roc_auc_score

val_df["pred"] = lr.predict_proba(val_df[numeric_features])[:, 1]

print("Validation score:", roc_auc_score(val_df[target], val_df["pred"]))
test_df[target] = lr.predict_proba(test_df[numeric_features])[:, 1]

print("creating submission...")
test_df[["id", target]].to_csv("lr_submission.csv", index=False)
