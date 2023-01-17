import pandas as pd

import numpy as np

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold
def getX(data):

    pclass = data.loc[:,"Pclass"]

    sex = data.loc[:, "Sex"]

    sex_tmp = list()

    

    for i in sex:

        if i == "male":

            sex_tmp.append(1)

        else:

            sex_tmp.append(0)   

    sex = np.array(sex_tmp)

    x = np.array([[pclass[i], sex[i]] for i in range(sex.size)])

    return x
train_data =  pd.read_csv('../input/train.csv', delimiter=',', header = 0)

test_data =  pd.read_csv('../input/test.csv', delimiter=',', header = 0)
print("\nПроцент выживших в зависимости от пола и класса каюты:")

print("------------------------------------------------------")

print(train_data.groupby(["Pclass", "Sex"])["Survived"].value_counts(normalize=True))

print("------------------------------------------------------\n")
# тренировочные данные

train_x = getX(train_data)

train_y = train_data.loc[:, "Survived"]



# тестовые данные

test_x = getX(test_data)
cv = KFold(n_splits=3, shuffle=True, random_state=42)

cv.get_n_splits(train_x, train_y)
# классификатор RF

rf = RandomForestClassifier(random_state=42, n_estimators=500, min_samples_split=8, min_samples_leaf=2)

# оценка точности RandomForest

scores = cross_val_score(rf, train_x, train_y, cv=cv, n_jobs=-1)

print("Точность: {}%".format(round(scores.mean(),4)*100))

# обучение

rf.fit(train_x, train_y)

# предсказывание по тестовой выборке

predictions = rf.predict(test_x)

predictions = np.array([int(round(i)) for i in predictions])
submission = pd.DataFrame({

    "PassengerId": test_data["PassengerId"],

    "Survived": predictions

})

submission.to_csv("titanic-submission.csv", index=False)