import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
data = pd.read_csv("../input/adult-para-tarefa-4/train_data.csv")
data.head()
data.shape
data.describe()
fig,ax = plt.subplots(figsize=(20,5))
data["native.country"].value_counts().plot(kind="bar")
fig,ax = plt.subplots(figsize=(20,5))
data["occupation"].value_counts().plot(kind="bar")
ndata = data.isnull()
i = 0
for row in ndata.values.tolist():
    for column in row:
        if column == True:
            i += 1
print(i)
ndata.shape
data["income"] = pd.get_dummies(pd.DataFrame(data["income"]))["income_>50K"]
data = data.drop("Id",axis=1)
data.head()
corrMatrix = np.array(data.corr())
features = ["age","fnlwgt","education.num","capital.gain","capital.loss","hours.per.week","income"]
corrMatrix.shape
fig, ax = plt.subplots(figsize=(8,8))
im = ax.imshow(corrMatrix)

ax.set_xticks(np.arange(len(features)))
ax.set_yticks(np.arange(len(features)))
ax.set_xticklabels(features)
ax.set_yticklabels(features)

plt.setp(ax.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")

for i in range(len(features)):
    for j in range(len(features)):
        text = ax.text(j,i,round(corrMatrix[i,j],2),ha="center",va="center",color="w",size=15)

ax.set_title("Matriz de correlação da base de dados")
fig.tight_layout()
plt.show()
data = pd.get_dummies(data,drop_first=True)
xTrain = data.drop(["income"],axis=1)
yTrain = data.income
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
RF = RandomForestClassifier()
RF.get_params()
#param_grid = {"criterion":["gini","entropy"],"n_estimators":[i for i in range(100,210,10)]}
#grid = GridSearchCV(RF,param_grid,cv=10)
#grid.fit(xTrain,yTrain)
#print(grid.best_estimator_)
#print(grid.best_score_)
#RF = grid.best_estimator_
RF = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=130, n_jobs=1,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)
test = pd.read_csv("../input/adult-para-tarefa-4/test_data.csv")
test.head()
test = pd.get_dummies(test,drop_first=True)
id_list = test.Id.tolist()
test = test.drop("Id",axis=1)
for feature in data.drop("income",axis=1).columns.values.tolist():
    if feature not in test.columns.values.tolist():
        test[feature] = len(test.age.tolist())*[0]
RF.fit(xTrain,yTrain)
pred_RF = RF.predict(test).tolist()
for i in range(0,len(pred_RF)):
    if pred_RF[i] == 0:
        pred_RF[i] = "<=50K"
    else:
        pred_RF[i] = ">50K"
pd.DataFrame({"Id":id_list,"income":pred_RF}).to_csv("pred_RF.csv",index=False)
from sklearn.svm import SVC
SVM = SVC()
SVM.get_params()
#param_grid = {"kernel":["rbf","poly","sigmoid"]}
#grid = GridSearchCV(SVM,param_grid,cv=10)
#grid.fit(xTrain,yTrain)
#print(grid.best_estimator_)
#print(grid.best_score_)
#SVM = grid.best_estimator_
SVM = SVC(kernel="sigmoid")
SVM.fit(xTrain,yTrain)
pred_SVM = SVM.predict(test).tolist()
for i in range(0,len(pred_SVM)):
    if pred_SVM[i] == 0:
        pred_SVM[i] = "<=50K"
    else:
        pred_SVM[i] = ">50K"
pd.DataFrame({"Id":id_list,"income":pred_SVM}).to_csv("pred_SVM.csv",index=False)