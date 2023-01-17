import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
import matplotlib.pyplot as plt

import seaborn as sns



df = pd.read_csv("../input/KaggleV2-May-2016.csv")
#Convert No-show: "Yes" = 1, "No" = 0

df['No-show'].replace(to_replace="Yes", value=1, inplace = True)

df['No-show'].replace(to_replace="No", value=0, inplace = True)

df[['No-show']].head(10)
#Convert to datetime

df["ScheduledDay"] = pd.to_datetime(df["ScheduledDay"])

df["AppointmentDay"] = pd.to_datetime(df["AppointmentDay"])

df[["ScheduledDay","AppointmentDay"]].info()

df[["ScheduledDay","AppointmentDay"]].head()
#Extract Month, Day of week, and Hour from datetime(unit Hour only for "ScheduleDay') 

df["Month_Scheduled"]=df["ScheduledDay"].dt.month

df["Month_Appointment"]=df["AppointmentDay"].dt.month



df["DOW_Scheduled"]=df["ScheduledDay"].dt.dayofweek

df["DOW_Appointment"]=df["AppointmentDay"].dt.dayofweek



df["Hour_Scheduled"]=df["ScheduledDay"].dt.hour  # unit Hour only for "ScheduleDay'



#Create elapsed time between ScheduleDay and AppointmentDay

#temporarily change to object type to edit format for "ScheduleDay" column

df["ScheduledDay"] = df["ScheduledDay"].astype(str)

df["ScheduledDay"] = df["ScheduledDay"].map(lambda x: x[:-8])

df["ScheduledDay"] = pd.to_datetime(df["ScheduledDay"])

df["AppointmentDay"] = pd.to_datetime(df["AppointmentDay"])

df["Appoint gap"] = df["AppointmentDay"] - df["ScheduledDay"]

df["Appoint gap"] = df["Appoint gap"].astype(str)

df["Appoint gap"] = df["Appoint gap"].map(lambda x: x[:-23])

df.drop(df[df["Appoint gap"] == "-1 d"].index, inplace=True)

df.drop(df[df["Appoint gap"] == "-6 d"].index, inplace=True)

df["Appoint gap"] = df["Appoint gap"].astype(int)



df[["Month_Scheduled","Month_Appointment","DOW_Scheduled","DOW_Appointment","Hour_Scheduled","Appoint gap"]].head()
#Eval Frequency of Past Missed Appointments

df['Num_Apt_Missed'] = df.groupby('PatientId')['No-show'].apply(lambda x: x.cumsum())



df[['Num_Apt_Missed']].head()
#Deleting rows with Age < 0

df = df[df.Age>0]
df_data =  df.drop(["PatientId", "AppointmentID", "ScheduledDay", "AppointmentDay"], 1)



#Iterate ober removing / adding features 

df_data.drop(["Alcoholism","Diabetes","Scholarship","Hipertension","DOW_Scheduled","DOW_Appointment","Month_Appointment","Gender","Hour_Scheduled","Handcap"], 1, inplace = True)

df_data.info()

# Apply LabelEncoder to "Neighborhood" coloumn



from sklearn.preprocessing import LabelEncoder

number = LabelEncoder()

df_data["Neighbourhood"] = number.fit_transform(df_data["Neighbourhood"].astype("str"))

#df_data["Gender"] = number.fit_transform(df_data["Gender"].astype("str"))
from sklearn import preprocessing

from sklearn import metrics 

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report



from sklearn.metrics import accuracy_score

from sklearn.metrics import recall_score

from sklearn.metrics import precision_score

from sklearn.metrics import f1_score





X = df_data.copy()

X = X.drop("No-show", axis = 1).values

y = df_data.loc[:,"No-show"].values



print("Proportion of response")

for i in np.unique(y) :

    print("The number of {} is {} accouting for {}%.".format(i, np.bincount(y)[i], np.round(np.bincount(y)[i]/len(y), 3)*100 ))
from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .25, random_state=25)



print("Proportion of response in train set")

for i in np.unique(y_train) :

    print("The number of {} is {} accouting for {}%.".format(i, np.bincount(y_train)[i], np.round(np.bincount(y_train)[i]/len(y_train), 3)*100 ))

print("\nProportion of response in test set")

for i in np.unique(y_test) :

    print("The number of {} is {} accouting for {}%.".format(i, np.bincount(y_test)[i], np.round(np.bincount(y_test)[i]/len(y_test), 3)*100 ))
# Over-sampling

from imblearn.over_sampling import SMOTE

X_train, y_train = SMOTE().fit_sample(X_train, y_train)



print("Proportion of response in train set using SMOTE")

for i in np.unique(y_train) :

    print("The number of {} is {} accouting for {}%.".format(i, np.bincount(y_train)[i], np.round(np.bincount(y_train)[i]/len(y_train), 3)*100 ))
# RandomForestClassifier

from sklearn.ensemble import RandomForestClassifier



# Build a forest

model = RandomForestClassifier(n_estimators=100,

                              random_state=1)

# Train the model using the training sets

model.fit(X_train, y_train)



# Check the importance for each predictor

importances = model.feature_importances_

std = np.std([tree.feature_importances_ for tree in model.estimators_],

             axis=0)

indices = np.argsort(importances)[::-1]

X_label = []

for ix in indices :

    X_label.append(df_data.drop("No-show", axis = 1).columns[ix])



for f in range(X_train.shape[1]):

    print("{}. {}  {}".format(f + 1, X_label[f], np.round(importances[indices[f]], decimals=3)))

    

# Plot the feature importances of the forest

plt.figure()

plt.title("Feature importances")

plt.bar(range(X_train.shape[1]), importances[indices],

       color="r", yerr=std[indices], align="center")

plt.xticks(range(X_train.shape[1]), X_label, rotation='vertical')

plt.xlim([-1, X_train.shape[1]])

plt.show()



#Predict Output

y_pred= model.predict(X_test)



#Classification report

from sklearn.metrics import classification_report

RF_result = classification_report(y_test, y_pred)

print(RF_result)



#Store performance metrics

RF_accuracy = accuracy_score(y_test, y_pred)

RF_precision = precision_score(y_test, y_pred)

RF_recall = recall_score(y_test, y_pred)

RF_f1 = f1_score(y_test, y_pred)
# Adaboost

from sklearn.ensemble import AdaBoostClassifier



ada = AdaBoostClassifier(n_estimators=100,random_state=1)

ada.fit(X_train, y_train)



#Predict Output

y_pred= ada.predict(X_test)



#Classification report

ADA_result = classification_report(y_test, y_pred)

print(ADA_result)



#Store performance metrics

ADA_accuracy = accuracy_score(y_test, y_pred)

ADA_precision = precision_score(y_test, y_pred)

ADA_recall = recall_score(y_test, y_pred)

ADA_f1 = f1_score(y_test, y_pred)
# Gradient Boosting

from sklearn.ensemble import GradientBoostingClassifier



gb = GradientBoostingClassifier(n_estimators=100,random_state=1)

gb.fit(X_train, y_train)



#Predict Output

y_pred= gb.predict(X_test)



#Classification report

GB_result = classification_report(y_test, y_pred)

print(GB_result)



#Store performance metrics

GB_accuracy = accuracy_score(y_test, y_pred)

GB_precision = precision_score(y_test, y_pred)

GB_recall = recall_score(y_test, y_pred)

GB_f1 = f1_score(y_test, y_pred)
#Nearest Neighborhood Clissifier



from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

from sklearn import neighbors

n_neighbors = 5

nnc = neighbors.KNeighborsClassifier(n_neighbors)

nnc.fit(X_train, y_train)



#Predict Output

y_pred= nnc.predict(X_test)



#Classification report

NNC_result = classification_report(y_test, y_pred)

print(NNC_result)



#Store performance metrics

NNC_accuracy = accuracy_score(y_test, y_pred)

NNC_precision = precision_score(y_test, y_pred)

NNC_recall = recall_score(y_test, y_pred)

NNC_f1 = f1_score(y_test, y_pred)
# How to choose N

from sklearn.metrics import recall_score

from sklearn.metrics import precision_score

from sklearn.metrics import f1_score



range_neighbors = range(1,100)

recall = []

precision = []

f1 = []





for n in range_neighbors :

    nnc = neighbors.KNeighborsClassifier(n)

    nnc.fit(X_train, y_train)

    y_pred= nnc.predict(X_test)

    recall.append(recall_score(y_test, y_pred))

    precision.append(precision_score(y_test, y_pred))

    f1.append(f1_score(y_test, y_pred))



plt.plot(range_neighbors, recall, label="Avg. recall")

plt.plot(range_neighbors, precision, label="Avg. precision")

plt.plot(range_neighbors, f1, label="Avg. f1-score")

plt.xlabel('Number of Neighbors N')

plt.ylabel('%')

plt.legend(loc="best")

plt.show()
#Bernoulli Naive Bayes & Analyzation

#does not use regularized dataset

#http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html#sklearn.naive_bayes.BernoulliNB



from sklearn.naive_bayes import BernoulliNB



bernb = BernoulliNB()

bernb.fit(X_train, y_train)



#Predict Output

y_pred= bernb.predict(X_test)



#Classification report

BERNB_result = classification_report(y_test, y_pred)

print(BERNB_result)



#Store performance metrics

BERNB_accuracy = accuracy_score(y_test, y_pred)

BERNB_precision = precision_score(y_test, y_pred)

BERNB_recall = recall_score(y_test, y_pred)

BERNB_f1 = f1_score(y_test, y_pred)
#Gaussian Naive Bayes & Analyzation

#does not use regularized dataset

#http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html

#75% Training and 25% Test Dataset split



from sklearn.naive_bayes import GaussianNB



gausb = GaussianNB()

gausb.fit(X_train, y_train)



#Predict Output

y_pred= gausb.predict(X_test)



#Classification report

GAUSB_result = classification_report(y_test, y_pred)

print(GAUSB_result)



#Store performance metrics

GAUSB_accuracy = accuracy_score(y_test, y_pred)

GAUSB_precision = precision_score(y_test, y_pred)

GAUSB_recall = recall_score(y_test, y_pred)

GAUSB_f1 = f1_score(y_test, y_pred)
#Multinomial Naive Bayes & Analyzation

#Does not use regularized dataset; all values had to be positive

#http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html#sklearn.naive_bayes.MultinomialNB



from sklearn.naive_bayes import MultinomialNB



multnb = MultinomialNB()

multnb.fit(X_train, y_train)



#Predict Output

y_pred= multnb.predict(X_test)



#Classification report

MULTNB_result = classification_report(y_test, y_pred)

print(MULTNB_result)



#Store performance metrics

MULTNB_accuracy = accuracy_score(y_test, y_pred)

MULTNB_precision = precision_score(y_test, y_pred)

MULTNB_recall = recall_score(y_test, y_pred)

MULTNB_f1 = f1_score(y_test, y_pred)
#Logistic regression

from sklearn.linear_model import LogisticRegression



lreg = LogisticRegression()

lreg.fit(X_train, y_train)





#Predict Output

y_pred= lreg.predict(X_test)





#Classification report

LOGIS_result = classification_report(y_test, y_pred)

print(LOGIS_result)



#Store performance metrics

LOGIS_accuracy = accuracy_score(y_test, y_pred)

LOGIS_precision = precision_score(y_test, y_pred)

LOGIS_recall = recall_score(y_test, y_pred)

LOGIS_f1 = f1_score(y_test, y_pred)
# Accuracy : the proportion of the total number of predictions that were correct.



# Positive Predictive Value(PPV) or Precision : the proportion of positive cases that were correctly identified.



# Negative Predictive Value(NPV) : the proportion of negative cases that were correctly identified.



# Sensitivity(TPR) or Recall : the proportion of actual positive cases which are correctly identified.



# Specificity(TNR) : the proportion of actual negative cases which are correctly identified.





# Which metric do we care most? Accuracy, Sensitivity(TPR), F1-score

# - Accuracy: How many Shows and No-shows are correctly predicted? 

# - Sessitivity(TPR): How many No-shows are correctly predicted among unseen observations a model predicted as No-show? 

#   * The proportion of actual No-shows(TP) among predictions lebled as No-show(TP + FN)

#   * The larger the TPR, the lesser false positives a model predicts.

# - F1-score: Weighted accuracy between recall and precision.

#   * We can get a sense of a model's performance in general.
# data to plot

n_groups = 8



 

# create plot

fig, ax = plt.subplots()

index = np.arange(n_groups)

bar_width = 0.15

opacity = 0.8

 

rects1 = plt.bar(index, [ADA_accuracy, GB_accuracy ,RF_accuracy, NNC_accuracy, GAUSB_accuracy, BERNB_accuracy, MULTNB_accuracy, LOGIS_accuracy] , bar_width,

                 alpha=opacity,

                 color='b',

                 label='ACC')

 

rects2 = plt.bar(index + bar_width, [ADA_recall, GB_recall ,RF_recall, NNC_recall, GAUSB_recall, BERNB_recall, MULTNB_recall, LOGIS_recall], bar_width,

                 alpha=opacity,

                 color='g',

                 label='TPR')



rects3 = plt.bar(index + 2*bar_width, [ADA_f1, GB_f1 ,RF_f1, NNC_f1, GAUSB_f1, BERNB_f1, MULTNB_f1, LOGIS_f1], bar_width,

                 alpha=opacity,

                 color='r',

                 label='F1_score')





plt.xlabel('Classifiers')

plt.ylabel('Scores')

plt.title('Classification scores')

plt.xticks(index + bar_width, ("ADA","GB",'RF', 'NNC', "G_NB", "B_NB","M_NB","Logis"))

plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.tight_layout()

plt.show()





# ACC: Bernoulli_NB > RF > Logistic

# TPR: Logistic > Bernoulli_NB > RF

# F1_score: Bernoulli_NB > RF > Logistic



# Conclusion: Choose Logistic!