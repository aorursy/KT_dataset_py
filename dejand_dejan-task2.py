# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python





import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.linear_model import LogisticRegression

from sklearn import metrics

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

#read student information

studentInfo = pd.read_csv("/kaggle/input/open-university-learning-analytics-dataset/anonymiseddata/studentInfo.csv")

vle = pd.read_csv("/kaggle/input/open-university-learning-analytics-dataset/anonymiseddata/vle.csv")

#print out the 10 first rows of the data

studentInfo.head(10)

studentInfo["disability"].unique()
studentInfo["imd_band"].unique()
# lets see all potential final results

studentInfo["final_result"].unique()
#create a new column to classify final results. classify studets with a pass or distinction as "1", the rest as "0"

studentInfo["result.class"] = 1



#studentInfo["result.class"] = studentInfo["final_result"].apply(lambda x: 0 if (x == 'Fail') | x == "Withdrawn") else 1)

studentInfo["result.class"].loc[(studentInfo["final_result"] == "Withdrawn") | (studentInfo["final_result"] == "Fail")] = 0

studentInfo["result.class"].loc[(studentInfo["imd_band"] == "90-100%") | (studentInfo["imd_band"] == "80-90%") | (studentInfo["imd_band"] == "70-80%") | (studentInfo["imd_band"] == "nan")] = 0

studentInfo["result.class"].loc[(studentInfo["disability"] == "N")] = 0

#and look at the dataset again

studentInfo.head()
Xfactors = studentInfo[["gender", "region", "highest_education", "imd_band", "age_band", "num_of_prev_attempts", "studied_credits", "disability"]]

X_noncat = pd.get_dummies(Xfactors)



X_noncat.head(5)
Youtcome = studentInfo["result.class"].values

Youtcome
#fit the model

X_train, X_test, y_train, y_test = train_test_split(X_noncat, Youtcome, test_size=0.3, random_state=0)

OurModel = LogisticRegression()

OurModel.fit(X_train, y_train)
#predict on a testset

y_pred = OurModel.predict(X_test)

print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(OurModel.score(X_test, y_test)))
#read additional data

studentAssessments = pd.read_csv("/kaggle/input/open-university-learning-analytics-dataset/anonymiseddata/studentAssessment.csv")

assessments = pd.read_csv("/kaggle/input/open-university-learning-analytics-dataset/anonymiseddata/assessments.csv")



#retrieve the ids only of the teacher assessments (TMA)

TAM = assessments.loc[assessments['assessment_type'] == "TMA"]





#then keep the students assessments (grades) that were only given by the teacher (TAM) and remove unknown entries ("?")

TAM_student_grades = studentAssessments.loc[studentAssessments.id_assessment.isin(TAM["id_assessment"])]

TAM_student_grades = TAM_student_grades.loc[TAM_student_grades['score'] != '?']
#create an empty list where we will save the average grade for each and every student

avg_grades = [] 

#for each student find all TMA scores for the course we are interested, and get the mean value



for i in range (0, len(studentInfo['id_student'])):

    

    this_student = studentAssessments.loc[(studentAssessments['id_student'] == studentInfo['id_student'][i]) &

                                          (studentAssessments['score'] != '?')]

    

    assmt = list(this_student['id_assessment'])

    score = list(this_student['score'])

                 

    #score = list(this_student['score'].astype(float))

    

    final_score = 0

    for j in range(0, len(assmt)):

        idx = assessments.loc[assessments.id_assessment == assmt[j]].index[0]

        if((assessments.code_module[idx] == studentInfo['code_module'][i]) & (assessments.assessment_type[idx] == "TMA")):

            final_score = final_score + (float(assessments.weight[idx])*score[j])/100

            

    avg_grades.append(final_score)

    

#add the new information about average TAM grades to the student information dataframe



studentInfo['avg_TMA_assessment'] = avg_grades
#add the new information about average TAM grades to our model



Xfactors_updated = studentInfo[["gender", "region", "highest_education", "imd_band", "age_band", "num_of_prev_attempts", "studied_credits", "disability", "avg_TMA_assessment"]]

X_noncat_updated = pd.get_dummies(Xfactors_updated)

X_noncat_updated = X_noncat_updated.fillna(0)

X_noncat_updated.head(5)
#fit the model again

X_train, X_test, y_train, y_test = train_test_split(X_noncat_updated, Youtcome, test_size=0.3, random_state=0)

OurModelUpdated = LogisticRegression()

OurModelUpdated.fit(X_train, y_train)
#predict on a testset

y_predUpdated = OurModelUpdated.predict(X_test)

print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(OurModelUpdated.score(X_test, y_test)))
#in our example...

#lets calculate the confusion matrix for the first model: OurModel

from sklearn.metrics import confusion_matrix



OurModelCM = confusion_matrix(y_test, y_pred)

print(OurModelCM)
#now lets calculate the confusion matrix for the second model: OurModelUpdated



OurModelUpdatedCM = confusion_matrix(y_test, y_predUpdated)

print(OurModelUpdatedCM)
from sklearn.metrics import classification_report

#compute the above metrics for the first model (OurModel)

print(classification_report(y_test, y_pred))
#compute the above metrics for the second model (OurModelUpdated)

print(classification_report(y_test, y_predUpdated))
#try out the ROC curve of the first model (OurModel)



from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve

import matplotlib.pyplot as plt 



X_train, X_test, y_train, y_test = train_test_split(X_noncat, Youtcome, test_size=0.3, random_state=0)

OurModel = LogisticRegression()

OurModel.fit(X_train, y_train)



logit_roc_auc = roc_auc_score(y_test, OurModel.predict(X_test))

fpr, tpr, thresholds = roc_curve(y_test, OurModel.predict_proba(X_test)[:,1])

plt.figure()

plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic for Original Model')

plt.legend(loc="lower right")

plt.savefig('Log_ROC')

plt.show()
#try out the ROC curve of the second model (OurModelUpdated)



X_train, X_test, y_train, y_test = train_test_split(X_noncat_updated, Youtcome, test_size=0.3, random_state=0)

OurModelUpdated = LogisticRegression()

OurModelUpdated.fit(X_train, y_train)



logit_roc_auc = roc_auc_score(y_test, OurModelUpdated.predict(X_test))

fpr, tpr, thresholds = roc_curve(y_test, OurModelUpdated.predict_proba(X_test)[:,1])

plt.figure()

plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic for Updated Model')

plt.legend(loc="lower right")

plt.savefig('Log_ROC')

plt.show()