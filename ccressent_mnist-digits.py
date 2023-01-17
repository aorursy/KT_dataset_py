import numpy as np

import pandas as pd



from sklearn.neural_network import MLPClassifier



train_dataset = pd.read_csv("../input/train.csv")

test_dataset  = pd.read_csv("../input/test.csv")



train_dataset.head()
seed = 42



X = train_dataset.drop("label", axis=1)

y = train_dataset["label"]



classifier = MLPClassifier(solver="lbfgs", alpha=1e-5, random_state=seed)

classifier.fit(X, y)
predictions = classifier.predict(test_dataset)

out = pd.DataFrame(predictions, columns=["Label"])

out.to_csv("result.csv")