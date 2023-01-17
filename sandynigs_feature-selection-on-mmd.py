import pandas as pd

import numpy as np

from sklearn.feature_selection import SelectPercentile, chi2

import time
asm_final_features = pd.read_csv('../input/asm_final_features.csv', index_col = 0 )
asm_final_features = asm_final_features.set_index("Id")
asm_final_features.head()
asm_final_features = asm_final_features.dropna(axis=1)
asm_final_features.shape
asm_final_features = asm_final_features.reset_index()
asm_final_features.shape
labels = pd.read_csv("../input/trainLabels.csv")
asm_final_features = pd.merge(asm_final_features, labels, on = "Id")
y = asm_final_features["Class"]
X = asm_final_features.drop(["Id", "Class"], axis = 1)
model = SelectPercentile(chi2, percentile = 50)

X_new = model.fit_transform(X,y)
X_new.shape
reduced_df = X.iloc[:, model.get_support(indices=True)]
useful_features = list(reduced_df.columns)

useful_features.insert(0, "Id")
asm_reduced_final = asm_final_features[useful_features]
asm_reduced_final = asm_reduced_final.set_index("Id")
asm_reduced_final.to_csv("./asm_reduced_final.csv")
asm_reduced_final = pd.read_csv("./asm_reduced_final.csv")
asm_reduced_final.shape