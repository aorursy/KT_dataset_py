import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

Train_Data = pd.read_csv("../input/data_set_ALL_AML_train.csv")
Test_Data = pd.read_csv("../input/data_set_ALL_AML_independent.csv")
Actual = pd.read_csv("../input/actual.csv")
print(Train_Data.isna().sum().max())
print(Test_Data.isna().sum().max())
# drop call columns for now, dont know what they mean. numbers represent patients...?

# patients (72 in total, 38 used for training)
patients = [str(i) for i in range(1, 73, 1)]
Data = pd.concat([Train_Data, Test_Data], axis=1)[patients]
Data.head()
# Transpose so that each row matches a patient
Data = Data.T

# to join on
Data["patient"] = pd.to_numeric(patients)

# AML is 1, MML is 0
Actual["cancer"]= pd.get_dummies(Actual.cancer, drop_first=True)

# add the cancer column to train data
Data = pd.merge(Data, Actual, on="patient")

# split in train and test, firts 38 are train, the rest is test
Train_Data, Test_Data = Data.iloc[:39,:], Data.iloc[39:,:]
# perform pca on the Data to reduce the amount of features
X, y = Train_Data.drop(columns=["cancer"]), Train_Data["cancer"]
pca = PCA()
X_transformed = pca.fit_transform(X)

fig, axes = plt.subplots(1,2, figsize=(10,5))

f = np.vectorize(lambda x: 1 - x)
axes[0].plot(f(pca.explained_variance_ratio_))
axes[0].set_ylim(0,1)
axes[0].set_ylabel("Explained variance")
axes[0].set_xlabel("Component")

# for coloring the MML and AML class
colors = lambda x: "Red" if x==1 else "Green"
f = np.vectorize(colors)
axes[1].scatter(X_transformed[:,0], X_transformed[:,1], c=f(y))

# use fist ~20 components to make svc
components_to_use = 20

pca = PCA(n_components=components_to_use)
X_transformed = pca.fit_transform(X)

# do a grid search
grid = { "C": [i/10 for i in range(1, 50, 5)],
               "kernel": ["linear", "rbf", "poly"],
                "gamma":["auto"],
               "decision_function_shape" : ["ovo", "ovr"],
              }
            
seach = GridSearchCV(SVC(), grid, cv=3)
seach.fit(X_transformed, y)
# select best svc
best_svc = seach.best_estimator_

for parameter, value in best_svc.get_params().items():
    print(parameter, value, sep='\t')
    
print("\nscore on train: {0}".format(best_svc.score(X_transformed, y)))
# test best svc on test
X_test, y_test = Test_Data.drop(columns=["cancer"]), Test_Data["cancer"]


X_test_transformed = pca.transform(X_test)

print("accuracy on test set {0}".format(best_svc.score(X_test_transformed, y_test)))
