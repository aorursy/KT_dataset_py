import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
#from sklearn.metrics import confusion_matrix
%matplotlib inline
train_data=pd.read_csv("../input/train.csv")
test_data=pd.read_csv("../input/test.csv")
del(train_data["Cabin"])
del(test_data["Cabin"])
train_age_ave= int(train_data["Age"].mean())
test_age_ave = int(test_data["Age"].mean())
train_data["Age"][np.isnan(train_data["Age"])] = train_age_ave
test_data["Age"][np.isnan(test_data["Age"])] = test_age_ave
Gen_map = {"male":0, "female":1}
embarked_map = {"S":0, "C":1,"Q":2}
train_data["Sex"] = train_data["Sex"].map(Gen_map)
train_data["Embarked"] = train_data["Embarked"].map(embarked_map)
test_data["Sex"] = test_data["Sex"].map(Gen_map)
test_data["Embarked"] = test_data["Embarked"].map(embarked_map)
import matplotlib.pyplot as plt
def plot_corr(f,size=14):
    corr = f.corr()
    fig,ax = plt.subplots(figsize=(size,size))
    ax.matshow(corr)
    plt.xticks(range(len(corr.columns)),corr.columns)
    plt.yticks(range(len(corr.columns)),corr.columns)
plot_corr(train_data)
train_data.corr()
plot_corr(test_data)
test_data.corr()
train_data["Embarked"][np.isnan(train_data["Embarked"])] = 0
columns = ["PassengerId","Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]
target = ["Survived"]
X = train_data[columns].values
Y = train_data[target].values
X1 = test_data[columns].values

test_data["Fare"][np.isnan(test_data["Fare"])] = test_data["Fare"].mean()
x1 = test_data[columns].values
train_data
rf_model = RandomForestClassifier(max_depth=2, random_state=0)
rf_model.fit(X,Y)
prediction_rf = rf_model.predict(x1)
pred = pd.concat([test_data["PassengerId"], pd.Series(prediction_rf, name="Survived")], axis=1)
pred








