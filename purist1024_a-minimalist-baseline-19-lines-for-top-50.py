import pandas as pd

from sklearn.linear_model import LogisticRegression



Xy_labelled = pd.read_csv("../input/cat-in-the-dat/train.csv", index_col="id")

X_labelled = Xy_labelled.drop(columns=["target"])

y_labelled = Xy_labelled["target"]

X_test = pd.read_csv("../input/cat-in-the-dat/test.csv", index_col="id")



target_mean = y_labelled.mean()

for col in X_labelled.columns:

    stats = Xy_labelled.groupby(col)["target"].agg(["sum", "count"])

    likelihoods = ((stats["sum"] + (800 * target_mean)) / (stats["count"] + 800)).to_dict()

    X_labelled[col] = X_labelled[col].map(likelihoods)

    X_test[col] = X_test[col].map(lambda v: likelihoods.get(v, target_mean))



model = LogisticRegression(solver="saga", n_jobs=-1, random_state=0).fit(X_labelled, y_labelled)

proba_predictions = model.predict_proba(X_test)[:, 1]

output = pd.DataFrame({"id": X_test.index, "target": proba_predictions})

output.to_csv("submission.csv", index=False)
