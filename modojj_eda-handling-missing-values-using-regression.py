import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

print(os.listdir("../input"))



import csv

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression, LinearRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from scipy import stats

from sklearn.model_selection import GridSearchCV

import statistics

from xgboost import XGBClassifier
# defining directory paths

train_dir = "../input/train.csv"

test_dir = "../input/test.csv"
df = pd.read_csv(train_dir)

test_df = pd.read_csv(test_dir)
df.drop(["Ticket"], axis = 1, inplace = True)

test_df.drop(["Ticket"], axis = 1, inplace = True)

df.info()

df.head()
# checking class distribution

print(df["Survived"].value_counts())

df["Survived"].value_counts().plot(kind = "pie")
df.drop("PassengerId", axis = 1, inplace = True)

test_df.drop("PassengerId", axis = 1, inplace = True)
f,ax = plt.subplots(figsize=(15, 13))

sns.heatmap(df.corr(), annot=True, cmap = "Blues", linewidths=.5, fmt= '.2f',ax = ax)

plt.show()
df_survived = df[df["Survived"]==1]

df_notsurvived = df[df["Survived"]==0]

gb_pclass_surv = df_survived.groupby("Pclass")["Survived"].sum()

#a = gb_pclass_surv.plot(kind= "bar")

gb_pclass_notsurv = df_notsurvived.groupby("Pclass")["Survived"].count()

#b = gb_pclass_notsurv.plot(kind= "bar")



fig = plt.figure(figsize = (10,4))

f1 = fig.add_subplot(1, 2, 1)

f1.set_ylim([0,400])

f2 = fig.add_subplot(1,2,2)

f2.set_ylim([0,400])

gb_pclass_surv.plot(kind= "bar", title = "Survived", ax = f1)

gb_pclass_notsurv.plot(kind= "bar", title = "Not Survived", ax = f2);
sns.catplot(x = 'Pclass', y = "Survived", data = df, kind = "bar");
pclass_dum = pd.get_dummies(df["Pclass"])

test_pclass_dum = pd.get_dummies(test_df["Pclass"])



df = pd.concat([df, pclass_dum], axis = 1)

test_df = pd.concat([test_df, test_pclass_dum], axis = 1)



df.rename({1:"pclass1", 2:"pclass2", 3:"pclass3"}, axis = 1, inplace = True)

test_df.rename({1:"pclass1", 2:"pclass2", 3:"pclass3"}, axis = 1, inplace = True)



df.drop(["Pclass"], axis = 1, inplace = True)

test_df.drop(["Pclass"], axis = 1, inplace = True)
print("SibSp unqiue value counts :\n" + str(df["SibSp"].value_counts()))



fig = plt.figure(figsize = (15,5))

f1 = fig.add_subplot(1, 3, 1)

f1.set_ylim([0,700])

f2 = fig.add_subplot(1,3,2)

f2.set_ylim([0,700])

f3 = fig.add_subplot(1,3, 3)

f3.set_ylim([0,700])

df["SibSp"].value_counts().plot(kind= "bar", title = "(SibSp) Total", ax = f1)

df_survived["SibSp"].value_counts().plot(kind= "bar", title = "(SibSp) Survived", ax = f2)

df_notsurvived["SibSp"].value_counts().plot(kind= "bar", title =  "(SibSp) Not Survived", ax = f3)

plt.show()
print("Parch unique value counts : \n" + str(df["Parch"].value_counts()))



fig = plt.figure(figsize = (15,5))

f1 = fig.add_subplot(1, 3, 1)

f1.set_ylim([0,700])

f2 = fig.add_subplot(1,3,2)

f2.set_ylim([0,700])

f3 = fig.add_subplot(1,3, 3)

f3.set_ylim([0,700])

df["Parch"].value_counts().plot(kind= "bar", title = "(Parch) Total", ax = f1)

df_survived["Parch"].value_counts().plot(kind= "bar", title = "(Parch) Survived", ax = f2)

df_notsurvived["Parch"].value_counts().plot(kind= "bar", title =  "(Parch) Not Survived", ax = f3)

plt.show()
df["Sex"].replace("male", 0, inplace = True)

test_df["Sex"].replace("male", 0, inplace = True)

df["Sex"].replace("female", 1, inplace = True)

test_df["Sex"].replace("female", 1, inplace = True)



df["Embarked"].fillna("S", inplace = True)

test_df["Embarked"].fillna("S", inplace = True)



pclass_dum = pd.get_dummies(df["Embarked"])

test_pclass_dum = pd.get_dummies(test_df["Embarked"])



df = pd.concat([df, pclass_dum], axis = 1)

test_df = pd.concat([test_df, test_pclass_dum], axis = 1)



df.rename({"S":"embarked_s", "C":"embarked_c", "Q":"embarked_q"}, axis = 1, inplace = True)

test_df.rename({"S":"embarked_s", "C":"embarked_c", "Q":"embarked_q"}, axis = 1, inplace = True)



df.drop(["Embarked"], axis = 1, inplace = True)

test_df.drop(["Embarked"], axis = 1, inplace = True)
df["n_fam_mem"] = df["SibSp"] + df["Parch"] + 1

test_df["n_fam_mem"] = df["SibSp"] + df["Parch"] + 1

df_survived["n_fam_mem"] = df_survived["SibSp"] + df_survived["Parch"]

df_notsurvived["n_fam_mem"] = df_notsurvived["SibSp"] + df_notsurvived["Parch"]



fig = plt.figure(figsize = (15,5))

f1 = fig.add_subplot(1, 3, 1)

f1.set_ylim([0,600])

f2 = fig.add_subplot(1,3,2)

f2.set_ylim([0,600])

f3 = fig.add_subplot(1,3, 3)

f3.set_ylim([0,600])



df["n_fam_mem"].value_counts().plot(kind = "bar", title = "all", ax = f1)

df_survived["n_fam_mem"].value_counts().plot(kind = "bar", title = "Survived", ax = f2)

df_notsurvived["n_fam_mem"].value_counts().plot(kind = "bar", title = "Not Survived", ax = f3);
def create_family_ranges(df):

    familysize = []

    for members in df["n_fam_mem"]:

        if members == 1:

            familysize.append(1)

        elif members == 2:

            familysize.append(2)

        elif members>2 and members<=4:

            familysize.append(3)

        elif members > 4:

            familysize.append(4)

    return familysize



famsize = create_family_ranges(df)

df["familysize"] = famsize



test_famsize = create_family_ranges(test_df)

test_df["familysize"] = test_famsize
fsizedummies = pd.get_dummies(df["familysize"])

test_fsizedummies = pd.get_dummies(test_df["familysize"])



df = pd.concat([df, fsizedummies], axis = 1)

test_df = pd.concat([test_df, test_fsizedummies], axis = 1)



df.rename({1:"fam_single",2:"fam_small",3:"fam_medium", 4:"fam_big"}, axis = 1, inplace = True)

test_df.rename({1:"fam_single",2:"fam_small",3:"fam_medium", 4:"fam_big"}, axis = 1, inplace = True)
df.head()
reg_df = df.drop(["Survived", "Name", "Cabin"], axis = 1)

reg_df_test = test_df.drop(["Name", "Cabin"], axis = 1)

    

age_reg_df = reg_df[reg_df["Age"].isna() == False]

age_reg_df_test = reg_df_test[reg_df_test["Age"].isna() == False]



new_age_df = age_reg_df.append(age_reg_df_test)

    

new_age_X = new_age_df.drop(["Age"], axis = 1)

new_age_y = new_age_df["Age"]



new_age_X["Fare"].fillna(df["Fare"].median(), inplace = True)



linear_reg_model = LinearRegression().fit(new_age_X, new_age_y)
# get indexes of rows that have NaN value

def get_age_indexes_to_replace(df):

    age_temp_list = df["Age"].values.tolist()

    indexes_age_replace = []

    age_temp_list = [str(x) for x in age_temp_list]

    for i, item in enumerate(age_temp_list):

        if item == "nan":

            indexes_age_replace.append(i)

    return indexes_age_replace



indexes_to_replace_main = get_age_indexes_to_replace(df)

indexes_to_replace_test = get_age_indexes_to_replace(test_df)



# make predictions on the missing values

def linear_age_predictions(reg_df, indexes_age_replace):

    reg_df_temp = reg_df.drop(["Age"], axis = 1)

    age_predictions = []

    for i in indexes_age_replace:

        x = reg_df_temp.iloc[i]

        x = np.array(x).reshape(1,-1)

        pred = linear_reg_model.predict(x)

        age_predictions.append(pred)

    return age_predictions



age_predictions_main = linear_age_predictions(reg_df, indexes_to_replace_main)

age_predictions_test = linear_age_predictions(reg_df_test, indexes_to_replace_test)



# fill the missing values with predictions

def fill_age_nan(df, indexes_age_replace, age_predictions):

    for i, item in enumerate(indexes_age_replace):

        df["Age"][item] =  age_predictions[i]

    return df



df = fill_age_nan(df, indexes_to_replace_main, age_predictions_main)

df_test = fill_age_nan(test_df, indexes_to_replace_test, age_predictions_test)
def age_to_int(df):

    agelist = df["Age"].values.tolist()

    for i in range(len(agelist)):

        if agelist[i] < 14: #children

            agelist[i] = 0

        elif agelist[i] >= 14 and agelist[i] < 25: #youth

            agelist[i] = 1

        elif agelist[i]>=25 and agelist[i]<60:# adult

            agelist[i] = 2

        elif agelist[i]>=60:# senior

            agelist[i] = 3

    ageint = pd.DataFrame(agelist)

    return ageint
ageint = age_to_int(df)

df["Ageint"] = ageint

df.drop("Age", axis = 1, inplace = True)



test_ageint = age_to_int(test_df)

test_df["Ageint"] = test_ageint

test_df.drop("Age", axis = 1, inplace = True)
fig = plt.figure(figsize = (15,5))

f1 = fig.add_subplot(1, 3, 1)

f1.set_ylim([0,400])

f2 = fig.add_subplot(1,3,2)

f2.set_ylim([0,400])

df["Ageint"][df["Survived"] == 1].value_counts().plot(kind = "pie", title = "Survived", ax = f1)

df["Ageint"][df["Survived"] == 0].value_counts().plot(kind = "pie", title = "Not Survived", ax = f2);
test_df["Fare"].fillna(test_df["Fare"].median(), inplace = True)



df["actual_fare"] = df["Fare"]/df["n_fam_mem"]



test_df["actual_fare"] = test_df["Fare"]/test_df["n_fam_mem"]



df["actual_fare"].plot()

df["actual_fare"].describe()
def conv_fare_ranges(df): 

    fare_ranges = []

    for fare in df["actual_fare"]:

        if fare < 7:

            fare_ranges.append(0)

        elif fare >=7 and fare < 14:

            fare_ranges.append(1)

        elif fare >=14 and fare < 30:

            fare_ranges.append(2)

        elif fare >=30 and fare < 50:

            fare_ranges.append(3)

        elif fare >=50:

            fare_ranges.append(4)

    return fare_ranges

        

fare_ranges = conv_fare_ranges(df)

df["fare_ranges"] = fare_ranges



test_fare_ranges = conv_fare_ranges(test_df)

test_df["fare_ranges"] = test_fare_ranges
df_nonsurv_fare = df[df["Survived"]==0]

df_surv_fare = df[df["Survived"]==1]



fig = plt.figure(figsize = (15,5))

f1 = fig.add_subplot(1, 3, 1)

f1.set_ylim([0,500])

f2 = fig.add_subplot(1,3,2)

f2.set_ylim([0,500])

f3 = fig.add_subplot(1,3, 3)

f3.set_ylim([0,500])



df["fare_ranges"].value_counts().plot(kind="bar", title = "Fare Ranges all", ax = f1)

df_surv_fare["fare_ranges"].value_counts().plot(kind="bar", title =  "Survived", ax = f2)

df_nonsurv_fare["fare_ranges"].value_counts().plot(kind="bar", title = "Not Survived", ax = f3);
df["Cabin"].fillna("unknown", inplace = True)

test_df["Cabin"].fillna("unknown", inplace = True)
cabins = [i[0]  if i!= 'unknown' else 'unknown' for i in df['Cabin']]

test_cabins = [i[0]  if i!= 'unknown' else 'unknown' for i in test_df['Cabin']]



df.drop(["Cabin"], axis = 1, inplace = True)

test_df.drop(["Cabin"], axis = 1, inplace = True)



df["cabintype"] = cabins

test_df["cabintype"] = test_cabins
fig = plt.figure(figsize = (15,5))

f1 = fig.add_subplot(1, 3, 1)

f1.title.set_text('Upper class')

f2 = fig.add_subplot(1,3,2)

f2.title.set_text('Middle class')

f3 = fig.add_subplot(1,3, 3)

f3.title.set_text('Lower class')

sns.catplot(y="pclass1",x="cabintype",data = df, kind = "bar",order = ['A','B','C','D','E','F','G','unknown'], ax = f1)

sns.catplot(y="pclass2",x="cabintype",data = df, kind = "bar",order = ['A','B','C','D','E','F','G','unknown'], ax = f2)

sns.catplot(y="pclass3",x="cabintype",data = df, kind = "bar",order = ['A','B','C','D','E','F','G','unknown'], ax = f3)

plt.close(2)

plt.close(3)

plt.close(4)
sns.catplot(y="Survived",x="cabintype",data = df, kind = "bar",order = ['A','B','C','D','E','F','G','unknown']);
df.drop(["cabintype"], axis = 1, inplace = True)

test_df.drop(["cabintype"], axis = 1, inplace = True)
def name_to_int(df):

    name = df["Name"].values.tolist()

    namelist = []

    for i in name:

        index = 1

        inew = i.split()

        if inew[0].endswith(","):

            index = 1

        elif inew[1].endswith(","):

            index = 2

        elif inew[2].endswith(","):

            index = 3

        namelist.append(inew[index])

        

    titlelist = []

    

    for i in range(len(namelist)): 

        titlelist.append(namelist[i])

    return titlelist
titlelist = name_to_int(df)

df["titles"] = titlelist

df["titles"].value_counts()

testtitlelist = name_to_int(test_df)

test_df["titles"] = testtitlelist

df["titles"].value_counts()
df["titles"].replace(["Jonkheer.","the","Don.","Capt.","Sir.","Col.","Major.","Dr.","Rev."], "sometitle", inplace = True)

test_df["titles"].replace(["Jonkheer.","the","Don.","Capt.","Sir.","Col.","Major.","Dr.","Rev.","Dona."],"sometitle", inplace = True)



df["titles"].replace(["Mlle.","Lady.","Mme.","Ms."],"Miss.", inplace = True)

test_df["titles"].replace(["Mlle.","Lady.","Mme.","Ms."],"Miss.", inplace = True)



plot = sns.catplot(y="Survived",x="titles",data = df, kind = "bar",order = ["Mr.","Miss.","Mrs.","Master.","sometitle"])

plot.set_ylabels("Survival Probability")
df["titles"].replace(["Mr.", "Miss.", "Mrs.", "Master.","sometitle"],[0,1,2,3,4], inplace = True)

df["titles"].astype("int64")



test_df["titles"].replace(["Mr.", "Miss.", "Mrs.", "Master.", "sometitle"],[0,1,2,3,4], inplace = True)

test_df["titles"].astype("int64")



df.drop(["Name"], axis = 1, inplace = True)

test_df.drop(["Name"], axis = 1, inplace = True)
df.drop(["Fare","n_fam_mem","actual_fare"], axis = 1, inplace = True)

test_df.drop(["Fare","n_fam_mem","actual_fare"], axis = 1, inplace = True)
df.info()
labels = df["Survived"]

data = df.drop("Survived", axis = 1)
final_clf = None

clf_names = ["Logistic Regression", "KNN(3)", "XGBoost Classifier", "Random forest classifier", "Decision Tree Classifier",

            "Gradient Boosting Classifier", "Support Vector Machine"]
classifiers = []

scores = []

for i in range(10):

    

    X_train, X_test, Y_train, Y_test = train_test_split(data, labels, test_size = 0.1)

    tempscores = []

    

    # logistic Regression

    lr_clf = LogisticRegression()

    lr_clf.fit(X_train, Y_train)

    tempscores.append((lr_clf.score(X_test, Y_test))*100)

    

    # KNN n_neighbors = 3

    knn3_clf = KNeighborsClassifier(n_neighbors = 3)

    knn3_clf.fit(X_train, Y_train)

    tempscores.append((knn3_clf.score(X_test, Y_test))*100)



    # XGBoost

    xgbc = XGBClassifier(n_estimators=15, seed=41)

    xgbc.fit(X_train, Y_train)

    tempscores.append((xgbc.score(X_test, Y_test))*100)



    # Random Forest

    rf_clf = RandomForestClassifier(n_estimators = 100)

    rf_clf.fit(X_train, Y_train)

    tempscores.append((rf_clf.score(X_test, Y_test))*100)



    # Decision Tree

    dt_clf = DecisionTreeClassifier()

    dt_clf.fit(X_train, Y_train)

    tempscores.append((dt_clf.score(X_test, Y_test))*100)



    # Gradient Boosting 

    gb_clf = GradientBoostingClassifier()

    gb_clf.fit(X_train, Y_train)

    tempscores.append((gb_clf.score(X_test, Y_test))*100)



    #SVM

    svm_clf = SVC(gamma = "scale")

    svm_clf.fit(X_train, Y_train)

    tempscores.append((svm_clf.score(X_test, Y_test))*100)

    

    scores.append(tempscores)
scores = np.array(scores)

clfs = pd.DataFrame({"Classifier":clf_names})

for i in range(len(scores)):

    clfs['iteration' + str(i)] = scores[i].T



means = clfs.mean(axis = 1)

means = means.values.tolist()



clfs["Average"] = means
clfs.set_index("Classifier", inplace = True)

print("Accuracies : ")

clfs["Average"].head(10)
# defining multiple SVM classifiers.

def create_multiple():    

    ensembles = []

    ensemble_scores = []

    for i in range(5):

        X_train, X_test, Y_train, Y_test = train_test_split(data, labels, test_size = 0.07)

        svm_clf = SVC(gamma = "scale")

        svm_clf = svm_clf.fit(X_train, Y_train)

        ensemble_scores.append((svm_clf.score(X_test, Y_test))*100)

        ensembles.append(svm_clf)

    return ensembles, ensemble_scores

SVM_ensembles, SVM_ensemble_scores = create_multiple()
def print_ensemble_score(ensemble_scores, model_name):    

    e_score = 0

    for i in range(len(ensemble_scores)):

        e_score = e_score + ensemble_scores[i]

    print("SCORE (ENSEMBLE MODELS) " +str(model_name)+ " : " + str(e_score/len(ensemble_scores)))

    return



print_ensemble_score(SVM_ensemble_scores, "SVM")
def per_model_prediction(ensembles):    

    test_data = test_df

    predictions_ensembles = []

    for clf in ensembles:

        temppredictions = clf.predict(test_data)

        predictions_ensembles.append(temppredictions)

    return predictions_ensembles
def get_predictions_modes(predictions_ensembles):    

    final_predictions_list = []

    for i in range(len(predictions_ensembles[0])):

        temp = [predictions_ensembles[0][i], predictions_ensembles[1][i], predictions_ensembles[2][i], predictions_ensembles[3][i], predictions_ensembles[4][i]]

        final_predictions_list.append(temp)



    final_predictions_list = np.array(final_predictions_list)

    pred_modes = stats.mode(final_predictions_list, axis = 1)



    final_predictions = []

    for i in pred_modes[0]:

        final_predictions.append(i[0])

    

    return final_predictions
SVM_predictions_ensembles = per_model_prediction(SVM_ensembles)

SVM_final_predictions = get_predictions_modes(SVM_predictions_ensembles)
passengerid = [892 + i for i in range(len(SVM_final_predictions))]

sub = pd.DataFrame({'PassengerId': passengerid, 'Survived':SVM_final_predictions})

sub.to_csv('submission.csv', index = False)