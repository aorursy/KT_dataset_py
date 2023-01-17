import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
train = pd.read_csv("../input/train_data.csv")
train.info()
train.head()
major = train.query('income == ">50K"')
minor = train.query('income == "<=50K"')
fig, axs = plt.subplots(6,3,sharey=True)
fig.subplots_adjust(right=3)
fig.subplots_adjust(top=5)

axs[0][0].hist(major['age'])
axs[0][0].set_title("major age")
axs[0][1].hist(minor['age'])
axs[0][1].set_title("minor age")
axs[0][2].hist(train['age'])
axs[0][2].set_title("total age")

axs[1][0].hist(major['fnlwgt'])
axs[1][0].set_title("fnlwgt")
axs[1][1].hist(minor['fnlwgt'])
axs[1][1].set_title("fnlwgt")
axs[1][2].hist(train['fnlwgt'])
axs[1][2].set_title("fnlwgt")

axs[2][0].hist(major['education.num'])
axs[2][0].set_title("edu.num")
axs[2][1].hist(minor['education.num'])
axs[2][1].set_title("edu.num")
axs[2][2].hist(train['education.num'])
axs[2][2].set_title("edu.num")

axs[3][0].hist(major['capital.gain'])
axs[3][0].set_title("cap.gain")
axs[3][1].hist(minor['capital.gain'])
axs[3][1].set_title("cap.gain")
axs[3][2].hist(train['capital.gain'])
axs[3][2].set_title("cap.gain")

axs[4][0].hist(major['capital.loss'])
axs[4][0].set_title("cap.loss")
axs[4][1].hist(minor['capital.loss'])
axs[4][1].set_title("cap.loss")
axs[4][2].hist(train['capital.loss'])
axs[4][2].set_title("cap.loss")

axs[5][0].hist(major['hours.per.week'])
axs[5][0].set_title("hpw")
axs[5][1].hist(minor['hours.per.week'])
axs[5][1].set_title("hpw")
axs[5][2].hist(train['hours.per.week'])
axs[5][2].set_title("hpw")

plt.show()
plt.close()
minor.describe()
major.describe()
sns.countplot(y='workclass', hue='income', data= train)
sns.countplot(y='workclass', hue='income', data= train)
sns.countplot(y='marital.status', hue='income', data= train)
sns.countplot(y='occupation', hue='income', data= train)
sns.countplot(y='relationship', hue='income', data= train)
sns.countplot(y='race', hue='income', data= train)
sns.countplot(y='sex', hue='income', data= train)
sns.countplot(y='native.country', hue='income', data= train)
dummieanswer = pd.get_dummies(train.income)
dummieanswer.head()
new = train.drop("income", axis=1)
new = new.join(dummieanswer)
new.head()
Xtrain = new.filter(["age", "fnlwgt", "education.num", "capital.gain", "capital.loss", "hours.per.week"])
Ytrain = new.filter([">50K"])
Xtrain = ((Xtrain-Xtrain.min())/(Xtrain.max()-Xtrain.min()))
Xtrain.head(5)
Ytrain = Ytrain[">50K"].ravel()
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from statistics import mean
nnclf = MLPClassifier(alpha=1e-5, hidden_layer_sizes=(1, 2), random_state=1)
ant = 0
besti = 0
bestj = 0
for i in range(1, 4):
    for j in range(1, 4):
        nnclf = MLPClassifier(alpha=1e-5, hidden_layer_sizes=(i, j), random_state=1)
        score = cross_val_score(nnclf, Xtrain, Ytrain, cv=10)
        media = mean(score)
        if media > ant:
            ant = media
            besti = i
            bestj = j
print("média:",media)
print("best i:",besti)
print("best j:",bestj)
nnclf = MLPClassifier(alpha=1e-5, hidden_layer_sizes=(besti, bestj), random_state=1)
from sklearn.linear_model import LogisticRegression
lrclf = LogisticRegression(random_state=0, solver='saga', multi_class='multinomial') 
score = cross_val_score(lrclf, Xtrain, Ytrain, cv=10)
print("média:",mean(score))
from sklearn.ensemble import RandomForestClassifier
rfclf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=2)
score = cross_val_score(lrclf, Xtrain, Ytrain, cv=10)
print("média:",mean(score))
test = pd.read_csv("../input/test_data.csv")
test.head()
Xtest = test.filter(["age", "fnlwgt", "education.num", "capital.gain", "capital.loss", "hours.per.week"])
Xtest = ((Xtest-Xtest.min())/(Xtest.max()-Xtest.min()))
nnclf.fit(Xtrain, Ytrain)
Ypred = nnclf.predict(Xtest)
answer = {"Id":test.Id,"ans":Ypred, "income":[0]*16280}
answer = pd.DataFrame(answer)
answer["income"] = np.where(answer["ans"]==0,"<=50K",">50K")
answer.sample(5)
answer = answer.filter(["Id","income"])
answer.to_csv(...)