import pandas as pd

df = pd.read_csv("../input/train.csv")

df1 = pd.read_csv("../input/gender_submission.csv")

X = df.loc[:,["Pclass","Sex","Age","SibSp","Parch"]]
y = df.iloc[:,1:2].values

X = X.fillna(X.mean())

X["Sex"] = X["Sex"].astype('category').cat.codes
X = X.values


df2 = pd.read_csv("../input/test.csv")
X_test = df2.loc[:,["Pclass","Sex","Age","SibSp","Parch"]]

X_test = X_test.fillna(X_test.mean())

X_test["Sex"] = X_test["Sex"].astype('category').cat.codes
X_test = X_test.values
y_test = df1.loc[:,["Survived"]].values

from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(13,13,13),max_iter=500,random_state=3)
mlp.fit(X,y)
y_pred = mlp.predict(X_test)

print (mlp.score(X_test,y_test))


from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

new_df = pd.DataFrame()
new_df["PassengerId"] = df2["PassengerId"]
new_df["Survived"] = y_pred

new_df.to_csv("submission.csv",index=False)
