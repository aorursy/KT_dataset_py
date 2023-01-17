import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sklearn

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import matplotlib.pyplot as plt



from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score as cross

from sklearn.metrics import accuracy_score as acs

adult = pd.read_csv("/kaggle/input/adult-pmr3508/train_data.csv",

        sep=r'\s*,\s*',

        engine='python',

        na_values="?")
adult.head()
def percent(colum):

    return colum*100//float(colum[-1])



incomexWorkclass = pd.crosstab(adult["income"], adult["workclass"],margins = True)

incomexWorkclass

incomexWorkclass.apply(percent,axis=0)
no_number = ["Federal-gov","Local-gov","Never-worked","Private","Self-emp-inc","Self-emp-not-inc","State-gov","Without-pay"]

number = ["38", "29", "0", "21","55","28","27","0"]
incomexEducation = pd.crosstab(adult["income"], adult["education"],margins = True)

incomexEducation

incomexEducation.apply(percent,axis=0)
Education = ["10th","11th","12th","1st-4th","5th-6th","7th-8th","9th","Assoc-acdm","Assoc-voc","Bachelors","Doctorate","HS-grad","Masters","Preschool","Prof-school","Some-college"]

Porcentagem = ["6", "5", "7", "3","4","6","5","24","26","41","74","15","55","0","73","19"]

no_number += Education

number += Porcentagem
incomexOccupation = pd.crosstab(adult["income"], adult["occupation"],margins = True)

incomexOccupation

incomexOccupation.apply(percent,axis=0)
Occupation = ["Adm-clerical","Armed-Forces","Craft-repair","Exec-managerial","Farming-fishing","Handlers-cleaners","Machine-op-inspct","Other-service","Priv-house-serv","Prof-specialty","Protective-serv","Sales","Tech-support","Transport-moving"]

Porcentagem = ["13", "11", "22", "48","11","6","12","4","0","44","32","26","30","20"]

no_number += Occupation

number += Porcentagem
incomexRelationship = pd.crosstab(adult["income"], adult["relationship"],margins = True)

incomexRelationship

incomexRelationship.apply(percent,axis=0)
Relationship = ["Husband","Not-in-family","Other-relative","Own-child","Unmarried","Wife"]

Porcentagem = ["44", "10", "3", "1","6","47"]

no_number += Relationship

number += Porcentagem
incomexSex = pd.crosstab(adult["income"], adult["sex"],margins = True)

incomexSex

incomexSex.apply(percent,axis=0)
Sex = ["Female","Male"]

Porcentagem = ["10","30"]

no_number += Sex

number += Porcentagem
income = ["<=50K",">50K"]

Porcentagem = ["0","1"]

no_number += income

number += Porcentagem
"""Esta função substitui um dado não numérico por um número relacionado com a porcentagem de pessoas que apresentam aquele dado

   e a característica ">50k"

"""

def num_func(label):

    for i in range(len(number)):

        if label == no_number[i]:

            return number[i]

    return label
adult["workclass"] = adult["workclass"].apply(num_func)

adult["education"] = adult["education"].apply(num_func)

adult["occupation"] = adult["occupation"].apply(num_func)

adult["relationship"] = adult["relationship"].apply(num_func)

adult["sex"] = adult["sex"].apply(num_func)
nadult = adult.dropna()
nadult
Xnadult = nadult[["workclass","education","education.num","occupation","relationship","sex","capital.gain","capital.loss","hours.per.week"]]

Xnadult
Ynadult = nadult.income

Ynadult
knn = KNeighborsClassifier(n_neighbors=23)
scores = cross(knn, Xnadult, Ynadult)

scores
arquivo2 = '/kaggle/input/adult-pmr3508/test_data.csv'

tester = pd.read_csv(arquivo2,na_values="?")
tester["sex"] = tester["sex"].apply(num_func)

tester["workclass"] = tester["workclass"].apply(num_func)

tester["marital.status"] = tester["marital.status"].apply(num_func)

tester["relationship"] = tester["relationship"].apply(num_func)

tester["occupation"] = tester["occupation"].apply(num_func)

tester["race"] = tester["race"].apply(num_func)
#ntester = tester.dropna()

ntester = tester.interpolate(method='pad')

ntester
Xntester = ntester[["workclass","education","education.num","occupation","relationship","sex","capital.gain","capital.loss","hours.per.week"]]
Xntester["workclass"] = Xntester["workclass"].apply(num_func)

Xntester["education"] = Xntester["education"].apply(num_func)

Xntester["occupation"] = Xntester["occupation"].apply(num_func)

Xntester["relationship"] = Xntester["relationship"].apply(num_func)

Xntester["sex"] = Xntester["sex"].apply(num_func)
knn.fit(Xnadult,Ynadult)
Ytestpred = knn.predict(Xntester)

Ytestpred
acs(Ynadult,knn.predict(Xnadult))
savepath = "predictions.csv"

prev = pd.DataFrame(Ytestpred, columns = ["income"])

prev.to_csv(savepath, index_label="Id")

prev