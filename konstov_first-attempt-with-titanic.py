import pandas as pd

import numpy as np



import matplotlib.pyplot as plt

%matplotlib inline
train = pd.read_csv("../input/train.csv", index_col="PassengerId")

test = pd.read_csv("../input/test.csv", index_col="PassengerId")
print(train.shape)

print(test.shape)
y = train["Survived"]

train = train.drop(["Survived"], axis=1)
all_p = pd.concat([train,test])
all_p["Sex"] = all_p["Sex"].apply(lambda x: 1 if x == "male" else 0)

all_p = all_p.fillna(all_p.mean())
all_p.head()
# добавлю признак - титул .. title

all_p["title"] = all_p["Name"].apply(lambda x: x[x.index(",")+2:x.index(".")])
# добавлю всего родственников ..

all_p["relative"] = all_p["SibSp"] + all_p["Parch"]
# превращу класс в строку, чтобы потом он стал категорией .. maybe these features better make categoriсal

all_p["Pclass"] = all_p["Pclass"].apply(str)

all_p["SibSp"] = all_p["SibSp"].apply(str)

all_p["Parch"] = all_p["Parch"].apply(str)
# create a age groups

# Добавлю признак: возрастная группа

def age_baskets(x):

    if x < 10:

        return "child"

    if 10 <= x < 17:

        return "teen"

    if 17 <= x < 25:

        return "youngMan"

    if 25 <= x < 35:

        return "middle1"

    if 35 <= x < 50:

        return "middle2"

    if 50 <= x:

        return "old"



all_p["age_baskets"] = all_p["Age"].apply(age_baskets)
all_p.head()
# Беру букву из названия каюты, если номера нет или только цифры, то пишу флоат .. take a letter from cabin

all_p["CabinLetter"] = all_p["Cabin"].apply(lambda x: x[0] if type(x) == str else "float")
# Предположу, что буквы в билете означают крутизну .. divide tickets on groups

all_p["Ticket_som"] = all_p["Ticket"].apply(lambda x: "coll" if type(x) == "str" else "soso")
# Эти признаки превращу в категориальные .. make a dummy features

all_p_dum = pd.get_dummies(all_p[["Pclass", "Age", "title", "relative", "age_baskets", "SibSp", "Parch", "Embarked", "Ticket_som", "CabinLetter"]])
# Удалю исходные .. drop 

all_p = all_p.drop(["Pclass", "Age", "title", "relative", "age_baskets", "SibSp", "Parch", "Embarked", "Ticket", "Ticket_som", "CabinLetter"], axis=1)

all_p1 = pd.concat([all_p, all_p_dum], axis=1)
# Удалю те, с которыми не знаю что делать .. drop other

all_p1 = all_p1.drop(["Name", "Cabin"], axis=1)
train1 = all_p1.ix[:train.shape[0],:]

test1 = all_p1.ix[train.shape[0]+1:,:]
train1.shape
test1.shape
from sklearn.ensemble import RandomForestClassifier as rfc
classifier = rfc(n_estimators=30, max_depth=7, random_state=42)

# classifier.fit(train1, y)
from sklearn.model_selection import cross_val_score



print(cross_val_score(classifier, train1, y, cv=5).mean())
classifier.fit(train1, y)
res = pd.DataFrame(classifier.predict(test1), index=test1.index, columns=["Survived"])
res[res == 1].count()
res.to_csv("res.csv")