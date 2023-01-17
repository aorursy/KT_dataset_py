# !pip install kaggle
# !pip install xgboost 
# !pip install graphviz
import matplotlib.pyplot as plt
import xgboost
import kaggle
import pandas
from sklearn.metrics import classification_report
train = pandas.read_csv("../input/train.csv")
train["train_test"] = "train"
test = pandas.read_csv("../input/test.csv")
test["train_test"] = "test"
data = pandas.concat([train, test])
data[:10]
data = pandas.get_dummies(data)
data[:3]
data.loc[ data['Fare'] == 0, 'Fare'] = 0
data.loc[(data['Fare'] > 0) & (data['Fare'] <= 50), 'Fare'] = 1
data.loc[data['Fare'] > 50, 'Fare'] = 2
data[:10]
data.Fare.describe()
#Introducing age groups
data.loc[ data['Age'] <= 13, 'Age'] = 0
data.loc[(data['Age'] > 13) & (data['Age'] <= 48), 'Age'] = 1
data.loc[(data['Age'] > 48) & (data['Age'] <= 64), 'Age'] = 2
data.loc[ data['Age'] > 64, 'Age'] = 4
data[:5]

# data.loc[ data['Age'] <= 16, 'Age'] = 0
# data.loc[(data['Age'] > 16) & (data['Age'] <= 32), 'Age'] = 1
# data.loc[(data['Age'] > 32) & (data['Age'] <= 48), 'Age'] = 2
# data.loc[(data['Age'] > 48) & (data['Age'] <= 64), 'Age'] = 3
# data.loc[ data['Age'] > 64, 'Age']
X_train = data[(data["train_test_train"] == 1)].drop(["train_test_test", "train_test_train", "Survived"], axis=1)
y_train = data[(data["train_test_train"] == 1)][["Survived"]]
X_test = data[(data["train_test_test"] == 1)].drop(["train_test_test", "train_test_train", "Survived"], axis=1)
# y_test = data[(data["train_test_test"] == 1)][["SalePrice"]]
clf = xgboost.XGBClassifier(n_jobs=-1)#DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
# print(classification_report(y_test, y_pred))
xgboost.plot_tree(clf)
plt.rcParams["figure.figsize"] = (1, 1)
plt.show()
xgboost.plot_importance(clf)
plt.rcParams["figure.figsize"] = (20, 20)
plt.show()
# Generate Submission File 
submission = pandas.DataFrame({ 'PassengerId': X_test.PassengerId,'Survived': [int(a) for a in y_pred] })
submission.to_csv("Submissive.csv", index=False)
submission[:3]



