import pandas as pd

import sklearn

import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score
nadult=pd.read_csv("/kaggle/input/adult-pmr3508/train_data.csv", index_col=['Id'], na_values = "?")

ntest=pd.read_csv("/kaggle/input/adult-pmr3508/test_data.csv", index_col=['Id'], na_values = "?")
nadult.head()
nadult.isnull().sum(axis = 0)
nadult=nadult.replace(to_replace =['Private','Self-emp-not-inc','Local-gov','State-gov','Self-emp-inc','Federal-gov','Without-pay','Never-worked'], value =[0, 1, 2, 3, 4, 5, 6, 7])



nadult=nadult.replace(to_replace =['HS-grad','Some-college','Bachelors','Masters','Assoc-voc','11th','Assoc-acdm','10th','7th-8th','Prof-school','9th','12th','Doctorate',

                                   '5th-6th','1st-4th','Preschool'], value =[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])



nadult=nadult.replace(to_replace =['Married-civ-spouse','Never-married','Divorced','Separated','Widowed','Married-spouse-absent','Married-AF-spouse'], value =[0, 1, 2, 3, 4, 5, 6])



nadult=nadult.replace(to_replace =['Prof-specialty','Craft-repair','Exec-managerial','Adm-clerical','Sales','Other-service','Machine-op-inspct','Transport-moving',

                                   'Handlers-cleaners','Farming-fishing','Tech-support','Protective-serv','Priv-house-serv','Armed-Forces'], value =[0, 1, 2, 3, 4, 5, 6, 7, 8, 9,

                                                                                                                                                    10, 11, 12, 13])



nadult=nadult.replace(to_replace =['Husband','Not-in-family','Own-child','Unmarried','Wife','Other-relative'], value =[0, 1, 2, 3, 4, 5])



nadult=nadult.replace(to_replace =['White','Black','Asian-Pac-Islander','Amer-Indian-Eskimo','Other'], value =[0, 1, 2, 3, 4])



nadult=nadult.replace(to_replace =['Male','Female'], value =[0, 1])



nadult=nadult.replace(to_replace =['United-States','Mexico','Philippines','Germany','Puerto-Rico','Canada','India','El-Salvador','Cuba','England','Jamaica','South','China','Italy',

                                   'Dominican-Republic','Vietnam','Guatemala','Japan','Poland','Columbia','Taiwan','Haiti','Iran','Portugal','Nicaragua','Peru','Greece','France',

                                   'Ecuador','Ireland','Hong','Cambodia','Trinadad&Tobago','Thailand','Laos','Yugoslavia','Outlying-US(Guam-USVI-etc)','Hungary','Honduras',

                                   'Scotland','Holand-Netherlands'], value =[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,

                                                                            28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40])



nadult=nadult.replace(to_replace =['<=50K','>50K'], value =[0, 1])



nadult.mean(axis = 0, skipna = True)

nadult['workclass'].fillna(0.651445, inplace = True)

nadult['occupation'].fillna(3.777517, inplace = True)

nadult['native.country'].fillna(0.910811, inplace = True)

ntest=ntest.replace(to_replace =['Private','Self-emp-not-inc','Local-gov','State-gov','Self-emp-inc','Federal-gov','Without-pay','Never-worked'], value =[0, 1, 2, 3, 4, 5, 6, 7])



ntest=ntest.replace(to_replace =['HS-grad','Some-college','Bachelors','Masters','Assoc-voc','11th','Assoc-acdm','10th','7th-8th','Prof-school','9th','12th','Doctorate',

                                   '5th-6th','1st-4th','Preschool'], value =[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])



ntest=ntest.replace(to_replace =['Married-civ-spouse','Never-married','Divorced','Separated','Widowed','Married-spouse-absent','Married-AF-spouse'], value =[0, 1, 2, 3, 4, 5, 6])



ntest=ntest.replace(to_replace =['Prof-specialty','Craft-repair','Exec-managerial','Adm-clerical','Sales','Other-service','Machine-op-inspct','Transport-moving',

                                   'Handlers-cleaners','Farming-fishing','Tech-support','Protective-serv','Priv-house-serv','Armed-Forces'], value =[0, 1, 2, 3, 4, 5, 6, 7, 8, 9,

                                                                                                                                                    10, 11, 12, 13])



ntest=ntest.replace(to_replace =['Husband','Not-in-family','Own-child','Unmarried','Wife','Other-relative'], value =[0, 1, 2, 3, 4, 5])



ntest=ntest.replace(to_replace =['White','Black','Asian-Pac-Islander','Amer-Indian-Eskimo','Other'], value =[0, 1, 2, 3, 4])



ntest=ntest.replace(to_replace =['Male','Female'], value =[0, 1])



ntest=ntest.replace(to_replace =['United-States','Mexico','Philippines','Germany','Puerto-Rico','Canada','India','El-Salvador','Cuba','England','Jamaica','South','China','Italy',

                                   'Dominican-Republic','Vietnam','Guatemala','Japan','Poland','Columbia','Taiwan','Haiti','Iran','Portugal','Nicaragua','Peru','Greece','France',

                                   'Ecuador','Ireland','Hong','Cambodia','Trinadad&Tobago','Thailand','Laos','Yugoslavia','Outlying-US(Guam-USVI-etc)','Hungary','Honduras',

                                   'Scotland','Holand-Netherlands'], value =[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,

                                                                            28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40])



ntest.mean(axis = 0, skipna = True)

ntest['workclass'].fillna(0.6656, inplace = True)

ntest['occupation'].fillna(3.831788, inplace = True)

ntest['native.country'].fillna(0.873985, inplace = True)
Xtreino=nadult[['age','workclass','education','education.num','marital.status','occupation','relationship','race','sex','capital.gain','capital.loss','hours.per.week']]

Ytreino=nadult.income





Xtest=ntest[['age','workclass','education','education.num','marital.status','occupation','relationship','race','sex','capital.gain','capital.loss','hours.per.week']]

knn = KNeighborsClassifier(n_neighbors=30)



scores = cross_val_score(knn, Xtreino, Ytreino, cv=10)

scores

scores.mean()
knn.fit(Xtreino,Ytreino)

YtestPred = knn.predict(Xtest)

YtestPred
entrega=pd.DataFrame()

entrega[0] = ntest.index

entrega[1] = YtestPred

entrega.columns = ['Id','income']



entrega=entrega.replace([0, 1], ['<=50K', '>50K'])

entrega['Id'][0]=0

entrega['Id'][1]=1



entrega
entrega.to_csv('submission.csv',index = False)