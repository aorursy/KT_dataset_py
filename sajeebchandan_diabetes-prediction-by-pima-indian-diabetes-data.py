import pandas as pandas

import numpy as numpy

import matplotlib as matplotlib

from matplotlib import pyplot as pyplot

from sklearn.model_selection import train_test_split

from sklearn import impute

from sklearn.naive_bayes import GaussianNB

from sklearn import metrics

from sklearn.ensemble import RandomForestClassifier

pandas.set_option("max_column", None)

from sklearn.linear_model import LogisticRegression

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

from sklearn.model_selection import cross_val_score

from sklearn.neighbors import KNeighborsClassifier

import seaborn
matplotlib.style.use('classic')

data_frame = pandas.read_csv('../input/pima-data.csv')
data_frame.head()
def corr_heatmap(data_frame, size=11):

    # Getting correlation using pandas

    correlation = data_frame.corr()

    fig, heatmap = pyplot.subplots(figsize=(size, size))



    # Plotting the correlation heatmap

    seaborn.heatmap(correlation,annot=True,cmap='coolwarm')



    # adding x-tics and y-tics

#     pyplot.xticks(range(len(correlation.columns)), correlation.columns)

#     pyplot.yticks(range(len(correlation.columns)), correlation.columns)

    # pyplot.scatter(range(len(correlation.columns)), range(len(correlation.columns)), s=25, label="Cat", color='r')

    pyplot.show()
corr_heatmap(data_frame, 11)
del data_frame['skin']
corr_heatmap(data_frame, 11)
map_diabetes = {True: 1, False: 0}

data_frame['diabetes'] = data_frame['diabetes'].map(map_diabetes)
feature_column_names = [

    'num_preg', 'glucose_conc', 'diastolic_bp', 'thickness', 'insulin', 'bmi',

    'diab_pred', 'age'

]



predicted_class_name = ['diabetes']
feature_set = data_frame[feature_column_names].values

label_class_set = data_frame[predicted_class_name].values
split_test_size = 0.30

Feature_train, Feature_test, Label_class_train, Label_class_test = train_test_split(

    feature_set, label_class_set, test_size=split_test_size, random_state=42)
print("{} Parcent data is being used for train".format(

    (len(Feature_train) / len(data_frame)) * 100))

print("{} Pacrcent of data is being used for test".format(

    (len(Feature_test) / len(data_frame)) * 100))

# train_test_split(feature_data_set(array of array), predicted_data_set(array or array), test_data_size, drom wchich column splitting will start)
print("Total rows in data frame:: {}".format(len(data_frame)))

print("Total val==0 rows in num_preg:: {}".format(

    len(data_frame.loc[data_frame["num_preg"] == 0])))

print("Total val==0 rows in glucose_conc:: {}".format(

    len(data_frame.loc[data_frame["glucose_conc"] == 0])))

print("Total val==0 rows in diastolic_bp:: {}".format(

    len(data_frame.loc[data_frame["diastolic_bp"] == 0])))

print("Total val==0 rows in thickness:: {}".format(

    len(data_frame.loc[data_frame["thickness"] == 0])))

print("Total val==0 rows in insulin:: {}".format(

    len(data_frame.loc[data_frame["insulin"] == 0])))

print("Total val==0 rows in bmi:: {}".format(

    len(data_frame.loc[data_frame["bmi"] == 0])))

print("Total val==0 rows in diab_pred:: {}".format(

    len(data_frame.loc[data_frame["diab_pred"] == 0])))

print("Total val==0 rows in age:: {}".format(

    len(data_frame.loc[data_frame["age"] == 0])))
fill_0 = impute.SimpleImputer(missing_values=0, strategy="mean", verbose=0)
Feature_train = fill_0.fit_transform(Feature_train)

Feature_test = fill_0.fit_transform(Feature_test)
naive_bayes_model = GaussianNB()

naive_bayes_model.fit(Feature_train, Label_class_train.ravel())
prediction_from_trained_data = naive_bayes_model.predict(

    Feature_train)

prediction_from_test_data = naive_bayes_model.predict(

    Feature_test)
accuracy_from_trained_data=metrics.accuracy_score(Label_class_train, prediction_from_trained_data)
accuracy_from_test_data=metrics.accuracy_score(Label_class_test, prediction_from_test_data)
print(f"Accuracy From Trained Data {accuracy_from_trained_data*100}%\n")

print(f"Accuracy From Test Data {accuracy_from_test_data*100}%")
print("confusion Matrix\n")

print(metrics.confusion_matrix(Label_class_test, prediction_from_test_data,labels=[1,0]))
print("classification Report\n")

print(metrics.classification_report(Label_class_test, prediction_from_test_data, labels=[1,0]))
random_forest_model=RandomForestClassifier(random_state=42)
random_forest_model.fit(Feature_train, Label_class_train.ravel())
rf_prediction_from_train_data=random_forest_model.predict(Feature_train)
rf_prediction_from_test_data=random_forest_model.predict(Feature_test)
rf_train_accuracy=metrics.accuracy_score(Label_class_train, rf_prediction_from_train_data)
rf_test_accuracy=metrics.accuracy_score(Label_class_test, rf_prediction_from_test_data)
print("Random Forest Train Acuracy")

print(rf_train_accuracy*100, "%")
print("Random Forest Test Accuracy")

print(rf_test_accuracy*100, "%")
print("Random Forest Confusion Matrix\n")

print(metrics.confusion_matrix(Label_class_test, rf_prediction_from_test_data))
print("Random Forest Classification Metrics\n")

print(metrics.classification_report(Label_class_test, rf_prediction_from_test_data, labels=[1,0]))
logistic_regression_model=LogisticRegression(penalty='l1',dual=False,max_iter=110, solver='liblinear')
logistic_regression_model.fit(Feature_train, Label_class_train.ravel())
print("Logistic Resgession Score On Train Data", logistic_regression_model.score(Feature_train, Label_class_train)*100,"%")
print("Logistic Resgession Score On Test Data", logistic_regression_model.score(Feature_test, Label_class_test)*100,"%")
print("Logistic Regression Confusion Matrix\n")

print(metrics.confusion_matrix(Label_class_test, logistic_regression_model.predict(Feature_test), labels=[1, 0]))
print("Logistic Regression Classification Report\n")

print(metrics.classification_report(Label_class_test, logistic_regression_model.predict(Feature_test), labels=[1,0]))
lr_model_sp=LogisticRegression(C=0.7, random_state=42, solver='liblinear')
lr_model_sp.fit(Feature_train, Label_class_train.ravel())
print("Logistic Resgession With Static Penalty (C) Score On Train Data", lr_model_sp.score(Feature_train, Label_class_train)*100,"%")
print("Logistic Resgession With Static Penalty (C) Score On Test Data", lr_model_sp.score(Feature_test, Label_class_test)*100,"%")
print("Logistic Regression With Static Penalty (C) Confusion Matrix\n")

print(metrics.confusion_matrix(Label_class_test, lr_model_sp.predict(Feature_test), labels=[1, 0]))
print("Logistic Regression With Static Penalty (C) Classification Report\n")

print(metrics.classification_report(Label_class_test, lr_model_sp.predict(Feature_test), labels=[1,0]))
def Logistic_Regression_With_Dynamic_Penalty(c_start, c_end, c_inc, feature_train, label_class_train, feature_test, label_class_test, weight, randomState=42):

    c_value_ist, recall_score_list=[],[]

    c_val=c_start

    best_recall_score=0

    while c_val<c_end:

        c_value_ist.append(c_val)

        if weight==None:

            lr_hyper_model=LogisticRegression(C=c_val, random_state=randomState, solver='liblinear')

        elif weight is not None:

            lr_hyper_model=LogisticRegression(C=c_val, class_weight=weight, random_state=randomState, solver='liblinear')

        else:

            lr_hyper_model=LogisticRegression(C=c_val, random_state=randomState, solver='liblinear')

        

        lr_hyper_model.fit(feature_train, label_class_train.ravel())

        recall_score=metrics.recall_score(label_class_test, lr_hyper_model.predict(feature_test))

        recall_score_list.append(recall_score)

        if recall_score>best_recall_score:

            best_recall_score=recall_score

        

        c_val=c_val+c_inc

        

    best_C_value=c_value_ist[recall_score_list.index(best_recall_score)]

    

    print(f'Max Recall {best_recall_score} at C={best_C_value}')

    

    pyplot.plot(c_value_ist, recall_score_list, '-')

    pyplot.xlabel("C Value")

    pyplot.ylabel("Recall Score")

    return best_C_value
Logistic_Regression_With_Dynamic_Penalty(0.1,5,0.1,Feature_train, Label_class_train, Feature_test, Label_class_test,None, randomState=42)
Logistic_Regression_With_Dynamic_Penalty(0.1,5,0.1,Feature_train, Label_class_train, Feature_test, Label_class_test,'balanced', randomState=42)
best_c_score_unweighted=Logistic_Regression_With_Dynamic_Penalty(0.1,5,0.1,Feature_train, Label_class_train, Feature_test, Label_class_test,None, randomState=42)

best_c_score_weighted=Logistic_Regression_With_Dynamic_Penalty(0.1,5,0.1,Feature_train, Label_class_train, Feature_test, Label_class_test,'balanced', randomState=42)
best_c_score_unweighted
best_c_score_weighted
lr_hyperparameterized_model=LogisticRegression(C=best_c_score_unweighted, random_state=42, solver="liblinear")
lr_hyperparameterized_model.fit(Feature_train, Label_class_train.ravel())
print(f"Accuracy Score= {metrics.accuracy_score(Label_class_test, lr_hyperparameterized_model.predict(Feature_test))*100}%")



print(f'\nConfussion Matrix\n{metrics.confusion_matrix(Label_class_test, lr_hyperparameterized_model.predict(Feature_test), labels=[1,0])}\n')

print(f'\nClassification Report\n{metrics.classification_report(Label_class_test, lr_hyperparameterized_model.predict(Feature_test), labels=[1,0])}')

print(f"\nRecall: {metrics.recall_score(Label_class_test, lr_hyperparameterized_model.predict(Feature_test))}")
lr_hyperparameterized_weighted_model=LogisticRegression(C=best_c_score_weighted, class_weight='balanced', random_state=42, solver='liblinear')
lr_hyperparameterized_weighted_model.fit(Feature_test, Label_class_test.ravel())
print(f"Accuracy Score= {metrics.accuracy_score(Label_class_test, lr_hyperparameterized_weighted_model.predict(Feature_test))*100}%")



print(f'\nConfussion Matrix\n{metrics.confusion_matrix(Label_class_test, lr_hyperparameterized_weighted_model.predict(Feature_test), labels=[1,0])}\n')

print(f'\nClassification Report\n{metrics.classification_report(Label_class_test, lr_hyperparameterized_weighted_model.predict(Feature_test), labels=[1,0])}')

print(f"\nRecall: {metrics.recall_score(Label_class_test, lr_hyperparameterized_weighted_model.predict(Feature_test))}")
naive_bayes_model_cv_score_accuracy_mean=cross_val_score(naive_bayes_model, feature_set, label_class_set.ravel(), cv=10, scoring='accuracy').mean()*100

print(naive_bayes_model_cv_score_accuracy_mean,"%")
random_forest_model_cv_score_accuracy_mean=cross_val_score(random_forest_model, feature_set, label_class_set.ravel(), cv=10,scoring='accuracy').mean()*100

print(random_forest_model_cv_score_accuracy_mean,"%")
logistic_regression_model_cv_score_accuracy_mean=cross_val_score(logistic_regression_model, feature_set, label_class_set.ravel(), cv=10,scoring='accuracy').mean()*100

print(logistic_regression_model_cv_score_accuracy_mean,"%")
lr_model_sp_unweighted_cv_score_accuracy_mean=cross_val_score(LogisticRegression(C=6.9, random_state=42, solver='liblinear'), feature_set, label_class_set.ravel(), cv=10,scoring='accuracy').mean()*100

print(lr_model_sp_unweighted_cv_score_accuracy_mean,"%")
lr_model_sp_weighted_cv_score_accuracy_mean=cross_val_score(LogisticRegression(C=6.9, random_state=42, solver='liblinear', class_weight='balanced'), feature_set, label_class_set.ravel(), cv=10,scoring='accuracy').mean()*100

print(lr_model_sp_weighted_cv_score_accuracy_mean,"%")
knn_model=KNeighborsClassifier(n_neighbors=5)
knn_model.fit(Feature_train, Label_class_train.ravel())
print(f"Accuracy Score: {knn_model.score(Feature_test, Label_class_test)*100}%")
print(f"Confusion Matrix\n {metrics.confusion_matrix(Label_class_test, knn_model.predict(Feature_test), labels=[1,0])}")
print(f"Classification Report\n {metrics.classification_report(Label_class_test, knn_model.predict(Feature_test), labels=[1,0])}")
knn_model_sp_weighted_cv_score_accuracy_mean=cross_val_score(knn_model, feature_set, label_class_set.ravel(), cv=10,scoring='accuracy').mean()*100

print(lr_model_sp_weighted_cv_score_accuracy_mean,"%")
k_range=numpy.arange(1,31,1)

k_score=[]

best_accuracy_score_knn=0

best_k=0

for k in k_range:

    score=cross_val_score(KNeighborsClassifier(n_neighbors=k), feature_set, label_class_set.ravel(), cv=10, scoring='accuracy').mean()

    k_score.append(score)

    if best_accuracy_score_knn<score:

        best_accuracy_score_knn=score

        best_k=k

knn_plot=pyplot

knn_plot.plot(k_range, k_score)

knn_plot.show()

print(f"KNN with N-Fold Cross Validation is giving best score as {best_accuracy_score_knn*100}% for K={best_k}")