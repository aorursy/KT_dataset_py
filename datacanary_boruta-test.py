import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from boruta import BorutaPy
df = pd.DataFrame.from_csv("../input/Iris.csv")

X = df.drop("Species", axis=1).values

y = df["Species"].values
print(X.__class__)

print(y.__class__)

print(X.shape)

print(y.shape)
rf = RandomForestClassifier(n_jobs=-1, class_weight='auto', max_depth=5)

feat_selector = BorutaPy(rf, n_estimators='auto', verbose=2)

feat_selector.fit(X, y)