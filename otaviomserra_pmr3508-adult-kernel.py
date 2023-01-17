import pandas as pd
trainAdult = pd.read_csv("../input/adult-kernel/train_data.csv",na_values="?")
trainAdult.head()
trainAdult.describe()
trainAdult["workclass"].value_counts()
trainAdult["education"].value_counts()
trainAdult["native.country"].value_counts()
trainAdult["race"].value_counts()
bTrainAdult = pd.get_dummies(trainAdult,dummy_na=True)[["age","education.num","capital.gain","capital.loss",
                                                        "hours.per.week","workclass_Private",
                                                        "native.country_United-States",
                                                        "race_White","sex_Male","income_>50K"]]
bTrainAdult = bTrainAdult.rename(index=str,columns={"workclass_Private":"works.private",
                                                      "native.country_United-States":"native.us",
                                                      "race_White":"is.white","sex_Male":"sex",
                                                      "income_>50K":"target"})
bTrainAdult.head()
testAdult = pd.read_csv("../input/adult-kernel/test_data.csv",na_values="?")
bTestAdult = pd.get_dummies(testAdult,dummy_na=True)[["age","education.num","capital.gain","capital.loss",
                                                      "hours.per.week","workclass_Private",
                                                      "native.country_United-States",
                                                      "race_White","sex_Male"]]
bTestAdult = bTestAdult.rename(index=str,columns={"workclass_Private":"works.private",
                                                      "native.country_United-States":"native.us",
                                                      "race_White":"is.white","sex_Male":"sex"})
bTestAdult.head()
XTrainAdult = bTrainAdult[["age","education.num","capital.gain","capital.loss","hours.per.week",
                            "works.private","native.us","is.white","sex"]]
YTrainAdult = bTrainAdult.target
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
values = [1,5,10,15,20,25,30,35,40]
scores = []

for k in values:
    knn = KNeighborsClassifier(n_neighbors=k)
    score = cross_val_score(knn,XTrainAdult,YTrainAdult,cv=10)
    avg = sum(score)/10
    scores.append(avg)
scores
new_values = [26,27,28,29,31,32,33,34]
new_scores = []

for k in new_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    score = cross_val_score(knn,XTrainAdult,YTrainAdult,cv=10)
    avg = sum(score)/10
    new_scores.append(avg)
new_scores
knn = KNeighborsClassifier(n_neighbors=26)
knn.fit(XTrainAdult,YTrainAdult)
target_pred = knn.predict(bTestAdult).tolist()
for i in range(len(target_pred)):
    if target_pred[i]==0:
        target_pred[i]="<=50K"
    else:
        target_pred[i]=">50K"
        
target_pred[:10]
pd.DataFrame({"Id":range(16280),"income":target_pred}).to_csv("prediction.csv",index=False)
