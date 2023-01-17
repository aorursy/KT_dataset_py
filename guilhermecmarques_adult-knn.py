%matplotlib inline

import matplotlib
import pandas as pd
import numpy as np

# remove error in pd manipulation
pd.options.mode.chained_assignment = None

# import dataset as a dataframe
adult = pd.read_csv("../input/adultdataset/train_data.csv",
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")
adult = adult.replace(np.nan,' ', regex=True)
# view dataframe
adult
[adult['age'].sum(), # Total sum of ages
 adult['age'].mean(), # Mean age
 adult['age'].median(), # Median of ages
 adult['age'].nunique(), # Number of unique ages
 adult['age'].max(), # Maximum age
 adult['age'].min()] # Minimum age
# view country labels
country_inc = adult['native.country'].value_counts()
print(country_inc)
# countries in log scale
country_inc.plot.bar(log='True')
import matplotlib.pyplot as plt
adult["age"].value_counts().plot(kind="bar")
adult["sex"].value_counts().plot(kind="bar")
adult["education"].value_counts().plot(kind="bar")
adult["occupation"].value_counts().plot(kind="bar")
# import test data
testAdult = pd.read_csv("../input/adultdataset/test_data.csv",
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")
testAdult = testAdult.replace(np.nan,' ', regex=True)
# dataframe manipulation

# convert sex into binary
testAdult["sex"] = testAdult["sex"].str.replace("Male", "1", regex = False)
testAdult["sex"] = testAdult["sex"].str.replace("Female", "0", regex = False)
adult["sex"] = adult["sex"].str.replace("Male", "1", regex = False)
adult["sex"] = adult["sex"].str.replace("Female", "0", regex = False)

# us X all
for nation in testAdult["native.country"].unique().tolist():
    if nation == "United-States":
        testAdult["native.country"] = testAdult["native.country"].str.replace("United-States", "1", regex = False)
    else:
        testAdult["native.country"] = testAdult["native.country"].str.replace(nation, "0", regex = False)
for nation in adult["native.country"].unique().tolist():
    if nation == "United-States":
        adult["native.country"] = adult["native.country"].str.replace("United-States", "1", regex = True)
    else:
        adult["native.country"] = adult["native.country"].str.replace(nation, "0", regex = False)

# binary 'race'
for race in adult["race"].unique().tolist():
    if race == "White":
        adult["race"] = adult["race"].str.replace("White", "0", regex = False)
        testAdult["race"] = testAdult["race"].str.replace("White", "0", regex = False)
    else:
        adult["race"] = adult["race"].str.replace(race, "1", regex = False)
        testAdult["race"] = testAdult["race"].str.replace(race, "1", regex = False)

# job category
for job in adult["occupation"].unique().tolist():
    if job == "Exec-managerial" or job == "Prof-specialty" or job == "Adm-clerical" or job == "Transport-moving":
        adult["occupation"] = adult["occupation"].str.replace(job, "1", regex = False)
        testAdult["occupation"] = testAdult["occupation"].str.replace(job, "1", regex = False)
    else:
        adult["occupation"] = adult["occupation"].str.replace(job, "0", regex = False)
        testAdult["occupation"] = testAdult["occupation"].str.replace(job, "0", regex = False)
    
    
# select used categories
Xadult = adult[["age","education.num","occupation","race","sex","capital.gain","capital.loss","hours.per.week","native.country"]]
XtestAdult = testAdult[["age","education.num","race","occupation","sex","capital.gain","capital.loss","hours.per.week", "native.country"]]
# interest variable
Yadult = adult.income
# import classifier and X-validation
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

# test k's between 1 and 30 (including)
h = 0
for k in range(1,31):
    # the classifier   
    knn = KNeighborsClassifier(n_neighbors=k)
    # results in 10 fold validation
    scores = cross_val_score(knn, Xadult, Yadult, cv=10)
    # mean of x-validations
    mean = np.mean(scores)
    if mean > h:
        print(k, mean)
        h = mean
# estimates train data against the interest variable
knn.fit(Xadult,Yadult)
# predicts test data income
YtestPred = knn.predict(XtestAdult)
# put the predictions in a dataframe
preds = pd.DataFrame(testAdult.Id)
preds["income"] = YtestPred
preds
# save predictions
preds.to_csv("prediction.csv", index=False)
