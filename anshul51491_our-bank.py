import numpy as np

import pandas as pd

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))

df=pd.read_csv("../input/bankbankbank/bank.csv")



print(df)
df_bank=df.drop(['education','default','contact','duration','pdays','previous','poutcome','y','campaign','day','month','marital'], axis=1)

print(df_bank)
df_bank = pd.get_dummies(df_bank, columns=["job"])

df_bank.head(10)

df_bank = pd.get_dummies(df_bank, columns=["housing"])

df_bank.head(10)

X=df_bank.iloc[:,[0,1]]

Y=df_bank.iloc[:,[2]]

print(X)

print(Y)

df_bank.loan.replace(('yes','no'),(1,0),inplace=True)

print(df_bank['loan'])

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 100)

clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,max_depth=3, min_samples_leaf=5)

clf_gini.fit(X_train, y_train)
clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100,max_depth=3, min_samples_leaf=5)

clf_entropy.fit(X_train, y_train)
clf_gini.predict([[33,4789]])
y_pred = clf_gini.predict(X_test)

y_pred
y_pred_en = clf_entropy.predict(X_test)

y_pred_en
print ("Accuracy is ", accuracy_score(y_test,y_pred)*100)
print ("Accuracy is ", accuracy_score(y_test,y_pred_en)*100)