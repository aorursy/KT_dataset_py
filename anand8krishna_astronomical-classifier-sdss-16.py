import pandas as pd



#load data

#data = pd.read_csv("../input/Skyserver_SQL2_27_2018 6_51_39 PM.csv")

data16 = pd.read_csv("../input/sloan-digital-sky-survey-dr16/Skyserver_12_30_2019 4_49_58 PM.csv")
#data preview 

data16.head()
#data summary

data16.info()
data16.describe()
#converting class column into scalar

data16["class"] = data16["class"].map(dict(GALAXY=1, STAR=2, QSO=3))
#frequency of different classes

data16["class"].value_counts()
#spliting data into 80--20 sets of training and testing resp.

#keeping the ratio of each of the 3 class-galaxy,star,qso-same

#in test and train set.





from sklearn.model_selection import StratifiedShuffleSplit



split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(data, data["class"]):

    strat_train_set = data16.loc[train_index]

    strat_test_set = data16.loc[test_index]
strat_test_set["class"].value_counts()
strat_train_set["class"].value_counts()
#unexpected ratio value of training and test sets obtained

#should be 0.2      got 0.25 approx



strat_test_set["class"].value_counts()/strat_train_set["class"].value_counts()
#saperating class column --class is Y



train_set_label = strat_train_set["class"]

test_set_label = strat_test_set["class"]
#data analysis was done manually for observing which dimensions are not determining the

#class of object. "class" is dropped because it will be predicted and

#our ML model will learn from strat_train_set and train_set_label

#we will test our model on strat_test_set and it will predict some classes for each objects

#then we will compare our results from the actual results, ie. test_set_label

#getting the rounded difference between these two will give us the number of correct prediction

#dividing it by total size of test set will give us the accuracy of our model



strat_test_set = strat_test_set.drop(columns=["objid","class","run","camcol","field","ra","dec",

                                              "rerun","specobjid","plate",

                                              "fiberid","mjd"])

strat_train_set = strat_train_set.drop(columns=["objid","class","run","camcol","field","ra","dec",

                                                "rerun","specobjid","plate",

                                                "fiberid", "mjd"])
#summary of reduced dimension data

strat_test_set.head()
#selecting and training the model





import sklearn.neighbors



model = sklearn.neighbors.KNeighborsRegressor(n_neighbors=2)

model.fit(strat_train_set, train_set_label)
#prediction using model

#on test set



x1 = model.predict(strat_test_set)
#calculating accuracy on test set



SET1 = x1 - test_set_label

SET1 =(round(SET1))

cnt1=SET1.value_counts()

cnt1[0]/SET1.count()

#prediction using model

#on train set



x2 = model.predict(strat_train_set)
#calculating accuracy on train set



SET2 = x2 - train_set_label

SET2 =(round(SET2))

cnt2=SET2.value_counts()

cnt2[0]/SET2.count()

#Accuracy on test set = 0.961

#Accuracy on train set= 0.991625
data = pd.read_csv("../input/sdss-14/Skyserver_SQL2_27_2018 6_51_39 PM.csv")

data["class"] = data["class"].map(dict(GALAXY=1, STAR=2, QSO=3))

test_label = data["class"]

test = data.drop(columns=["objid","class","run","camcol","field","ra","dec",

                                              "rerun","specobjid","plate",

                                              "fiberid","mjd"])

x3 = model.predict(test)

SET3 = x3 - test_label

SET3 = round(SET3)

cnt3=SET3.value_counts()

cnt3[0]/SET3.count()