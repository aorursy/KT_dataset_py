import pandas as pd

# The competition datafiles are in the directory ../input
# Read competition data files:
train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")

# Write to the log:
print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))
print("Test set has {0[0]} rows and {0[1]} columns".format(test.shape))
# Any files you write to the current directory get shown as outputs
import pandas as pd
import numpy as np

xcols = [col for col in train.columns if col != "label"]
X = train[xcols].values
y = train["label"].values
Xtest = test.values

from sklearn.ensemble import RandomForestClassifier
mod1 = RandomForestClassifier(n_estimators=500)
mod1.fit(X, y)
ypred = mod1.predict(Xtest)

result = pd.DataFrame(np.arange(28000)+1, columns=["ImageId"])
result["Label"] = ypred
result.to_csv("choushabi.csv", index=False)
