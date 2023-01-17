import numpy as np

import pandas as pd

import sklearn
!ls ../input -l
test_data = pd.read_csv("../input/eval-lab-2-f464/test.csv")

train_data = pd.read_csv("../input/eval-lab-2-f464/train.csv")

# train_data["chem_6"] = (train_data["chem_6"]>0).astype(np.int)

# test_data["chem_6"] = (test_data["chem_6"]>0).astype(np.int)

# train_data["chem_5"] = (train_data["chem_5"]>0).astype(np.int)

# test_data["chem_5"] = (test_data["chem_5"]>0).astype(np.int)
X = train_data.drop(columns=["class"])[["chem_1", "chem_4", "chem_6", "attribute"]]

y = train_data["class"]



from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

classifiers = []

for i in list(range(100)):

    rfc = RandomForestClassifier(n_estimators=2000, random_state=i, n_jobs=-1)

    classifiers.append((str(i), rfc))
eclf1 = VotingClassifier(estimators=classifiers, voting='soft')

eclf1.fit(X, y)
df_sub = pd.DataFrame({

    "id": test_data["id"],

    "class": eclf1.predict(test_data[["chem_1", "chem_4", "chem_6", "attribute"]])

})
df_sub.to_csv("12.csv", index=False)