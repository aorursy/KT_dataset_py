# Importação das bibliotecas 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

from collections import Counter

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve, train_test_split, KFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

sns.set(style='white', context='notebook', palette='deep')
#Lendo a base de treino
traindata = pd.read_csv("../input/adult-database/train_data.csv",
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")
#Modificando a visalização da base de treino
traindata.iloc[0:20,:]
traindata = traindata.drop(columns = ['Id'])
# Reformat Column We Are Predicting
traindata['income']=traindata['income'].map({'<=50K': 0, '>50K': 1, '<=50K.': 0, '>50K.': 1})

# Identify Numeric features
numeric_features = ['age','fnlwgt','education.num','capital.gain','capital.loss','hours.per.week','income']

# Identify Categorical features
cat_features = ['workclass','education','marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country']
# Correlation matrix between numerical values
g = sns.heatmap(traindata[numeric_features].corr(),annot=True, fmt = ".2f", cmap = "coolwarm")

# Explore Native Nation vs Income
g = sns.barplot(x="native.country",y="income",data=traindata)
g = g.set_ylabel("Income >50K Probability")

# Explore Sex vs Income
g = sns.barplot(x="sex",y="income",data=traindata)
g = g.set_ylabel("Income >50K Probability")

# Explore Relationship vs Income
g = sns.factorplot(x="relationship",y="income",data=traindata,kind="bar", size = 6 ,
palette = "muted")
g.despine(left=True)
g = g.set_ylabels("Income >50K Probability")

# Explore Marital Status vs Income
g = sns.factorplot(x="marital.status",y="income",data=traindata,kind="bar", size = 6 ,
palette = "muted")
g.despine(left=True)
g = g.set_ylabels("Income >50K Probability")

# Explore Workclass vs Income
g = sns.factorplot(x="workclass",y="income",data=traindata,kind="bar", size = 6 ,
palette = "muted")
g.despine(left=True)
g = g.set_ylabels("Income >50K Probability")

# Convert Sex value to 0 and 1
traindata["sex"] = traindata["sex"].map({"Male": 0, "Female":1})

# Create Married Column - Binary Yes(1) or No(0)
traindata["marital.status"] = traindata["marital.status"].replace(['Never-married','Divorced','Separated','Widowed'], 'Single')
traindata["marital.status"] = traindata["marital.status"].replace(['Married-civ-spouse','Married-spouse-absent','Married-AF-spouse'], 'Married')
traindata["marital.status"] = traindata["marital.status"].map({"Married":1, "Single":0})
traindata["marital.status"] = traindata["marital.status"].astype(int)

# Drop the data you don't want to use
traindata.drop(labels=["workclass","education","occupation","relationship","race","native.country"], axis = 1, inplace = True)
print('Dataset with Dropped Labels')
print(traindata.head())
traindata.shape
features = traindata[["age","fnlwgt","education.num","marital.status","sex","capital.gain","capital.loss","hours.per.week"]]
target = traindata['income']
gnb = GaussianNB()

gnb.fit(features, target)
scores = cross_val_score(gnb, features, target, cv=10)

print(scores)
# Params for Random Forest
num_trees = 100
max_features = 3
array = traindata.values
X = array[:,0:8]
Y = array[:,8]
print('Split Data: X')
print(X)
print('Split Data: Y')
print(Y)
validation_size = 0.20
seed = 7
num_folds = 10
scoring = 'accuracy'
features, X_validation, target, Y_validation = train_test_split(X,Y,
    test_size=validation_size,random_state=seed)
random_forest = RandomForestClassifier(n_estimators=250,max_features=5)
random_forest.fit(features, target)
predictions = random_forest.predict(X_validation)
print("Accuracy: %s%%" % (100*accuracy_score(Y_validation, predictions)))
decision_tree = DecisionTreeClassifier()
decision_tree.fit(features, target)
predictions = decision_tree.predict(X_validation)
print("Accuracy: %s%%" % (100*accuracy_score(Y_validation, predictions)))
# Submissão do arquivo teste
testdata = pd.read_csv("../input/adult-database/test_data.csv",
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")
#Modificando a visalização da base de treino
testdata.iloc[0:20,:]

testdata.shape
# Convert Sex value to 0 and 1
testdata["sex"] = testdata["sex"].map({"Male": 0, "Female":1})

features_test = testdata[["age","fnlwgt","education.num","marital.status","sex","capital.gain","capital.loss","hours.per.week"]]
# Create Married Column - Binary Yes(1) or No(0)
testdata["marital.status"] = testdata["marital.status"].replace(['Never-married','Divorced','Separated','Widowed'], 'Single')
testdata["marital.status"] = testdata["marital.status"].replace(['Married-civ-spouse','Married-spouse-absent','Married-AF-spouse'], 'Married')
testdata["marital.status"] = testdata["marital.status"].map({"Married":1, "Single":0})
testdata["marital.status"] = testdata["marital.status"].astype(int)

# Drop the data you don't want to use
testdata.drop(labels=["workclass","education","occupation","relationship","race","native.country"], axis = 1, inplace = True)
features_test = testdata[["age","fnlwgt","education.num","marital.status","sex","capital.gain","capital.loss","hours.per.week"]]

testdata.shape
x_val_test = features_test
y_val_test = random_forest.predict(x_val_test)

dfSave = pd.DataFrame(data={"Id" : testdata["Id"], "income" : y_val_test})
dfSave['Id'] = dfSave['Id'].astype(int)
dfSave["income"] = dfSave["income"].map({0:'<=50K', 1:'>50K'})
pd.DataFrame(dfSave[["Id", "income"]], columns = ["Id", "income"]).to_csv("Output.csv", index=False)
dfSave