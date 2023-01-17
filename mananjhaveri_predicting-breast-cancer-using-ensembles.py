import pandas as pd 

import numpy as np

import matplotlib.pyplot as plt 

import seaborn as sns 

from sklearn.metrics import classification_report

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import VotingClassifier
# Importing data 

df = pd.read_csv("/kaggle/input/breast-cancer-prediction-dataset/Breast_cancer_data.csv")
# understanding data

print("Shape:", df.shape)

print("\n---------------------------------\n")

print(df.head())

print("\n---------------------------------\n")

print(df.info())

print("\n---------------------------------\n")

print(df.describe())

print("\n---------------------------------\n")
# print additonal info

print(df["diagnosis"].value_counts())
# plots

ax, fig = plt.subplots()

ax = sns.heatmap(df.corr(), annot=True)

plt.show()
class preprocess:



    def __init__(self):

        return 



    def fit_transform(self, df, drop_features=False):

        self.df = df.copy() 



        # splitting into X and y

        X, y = self.df.drop(["diagnosis"], axis=1), self.df["diagnosis"]



        # drop features 

        # dropping  "mean_radius", "mean_perimeter" bcause highly correlated with "mean_area"

        if drop_features:

            X.drop(["mean_radius", "mean_perimeter"], axis=1, inplace=True)



        # Scaling 

        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()

        X = scaler.fit_transform(X)



        # splitting the data into train and test

        from sklearn.model_selection import StratifiedShuffleSplit 

        split = StratifiedShuffleSplit(n_splits=1, test_size=0.1)



        for train_index, test_index in split.split(X, y):

            X_train, X_test = X[train_index], X[test_index]

            y_train, y_test = y.loc[train_index], y.loc[test_index] 

        

        return X_train, y_train, X_test, y_test 
def predict(model):

    model.fit(X_train, y_train)

    y_preds = model.predict(X_test)



    print(classification_report(y_test, y_preds))
# preprocessing

# from preprocessing import preprocess

pre = preprocess()

X_train, y_train, X_test, y_test = pre.fit_transform(df)
# predict 

models = [("lr", LogisticRegression()), ("svc", SVC()), ("rfc", RandomForestClassifier())]



for model in models:

    print(model[0])

    predict(model[1])

    print("\n---------------------------------------\n")
print("voting clf")

voting_clf = VotingClassifier(estimators=models, voting="hard")

predict(voting_clf)