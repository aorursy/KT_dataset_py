import pandas as pd

from sklearn import preprocessing

from sklearn.ensemble import GradientBoostingClassifier
tr = pd.read_csv("../input/train-titanic/train.csv")

tst = pd.read_csv("../input/testmodifiedtitanic/test.csv")

le = preprocessing.LabelEncoder()
tr.head()
tr["Age"] = tr["Age"].fillna(0)

tst["Age"] = tst["Age"].fillna(0)

tst["Fare"] = tst["Fare"].fillna(tst["Fare"].median())

#tr["Cabin"] = tr["Cabin"].fillna(0)

#tst["Cabin"] = tst["Cabin"].fillna(0)
sutr = tr["Survived"].values.tolist()

agtr = tr["Age"].values.tolist()

cltr = tr["Pclass"].values.tolist()

setr = tr["Sex"].values.tolist()

partr = tr["Parch"].values.tolist()

sibtr = tr["SibSp"].values.tolist()

ftr = tr["Fare"].values.tolist()

cabtr = tr["Cabin"].values.tolist()

emtr = tr["Embarked"].values.tolist()



emtr = list(le.fit_transform(emtr))

cabtr = list(le.fit_transform(cabtr))
tmp = []

for i in setr:

    if i == 'female':

        tmp.append(1)

    else :

        tmp.append(0)

setr = tmp

feat = []

for i in range(len(sutr)):

    alone = 1 if sibtr[i]+partr[i] == 0 else 0

    feat.append([agtr[i],cltr[i],setr[i],ftr[i],alone,cabtr[i],emtr[i]])

lbl = sutr
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(feat, lbl, test_size=0.33, random_state=50)
gbc = GradientBoostingClassifier()
gbc.fit(x_train,y_train)
from sklearn.metrics import accuracy_score as score



print(score(y_test,gbc.predict(x_test)))
agtst = tst["Age"].values.tolist()

cltst = tst["Pclass"].values.tolist()

setst = tst["Sex"].values.tolist()

partst = tst["Parch"].values.tolist()

ftst = tst["Fare"].values.tolist()

sibtst = tst["SibSp"].values.tolist()

cabtst = tst["Cabin"].values.tolist()

emtst = tst["Embarked"].values.tolist()

le = preprocessing.LabelEncoder()

emtst = list(le.fit_transform(emtst))

cabtst = list(le.fit_transform(cabtst))

tmp = []



for i in setst:

    if i == "female":

        tmp.append(1)

    else :

        tmp.append(0)

setst = tmp

featst = []

for i in range(len(agtst)):

    featst.append([agtst[i],cltst[i],setst[i],ftst[i],1 if sibtst[i]+partst[i] == 0 else 0,cabtst[i],emtst[i]])
gbc.fit(feat,lbl)
fin = gbc.predict(featst)
print(fin)
idp = tst["PassengerId"].values.tolist()
df = pd.DataFrame({"PassengerId":idp, "Survived":fin})
df.to_csv('Output.csv')