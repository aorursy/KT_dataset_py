import pandas as pd
import numpy as np

# The competition datafiles are in the directory ../input
# Read competition data files:
train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")


xcols = [col for col in train.columns if col != "label"]
X = train[xcols].values
y = train["label"].values
Xtest = test.values


from sklearn.neural_network import MLPClassifier
mod2 = MLPClassifier(algorithm='l-bfgs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=1)
mod2.fit(X, y)
ypred = mod2.predict(Xtest)
result = pd.DataFrame(np.arange(28000)+1, columns=["ImageId"])
result["Label"] = ypred
result.to_csv("dierdan.csv", index=False)
