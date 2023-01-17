import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from imblearn.over_sampling import SMOTE

from mlxtend.plotting import plot_confusion_matrix

from sklearn.linear_model import LogisticRegression

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from sklearn.metrics import precision_score, accuracy_score, recall_score, roc_curve, precision_recall_curve, auc

from sklearn.tree import DecisionTreeClassifier

import warnings
np.random.seed(1)



warnings.filterwarnings("ignore")



data = pd.read_csv("../input/auto-insurance-claims-data/insurance_claims.csv")



drop_columns = ["_c39", "auto_model", "policy_bind_date", "policy_state", "incident_date",

               "incident_state", "incident_city", "incident_location", "policy_csl"]



data = data.drop(drop_columns, axis=1)



new_response = []

response = data.iloc[:, -1]

for i in range(len(response)):

    new_response.append(1 if response[i]=='Y' else 0)



data["fraud_reported"] = pd.Series(new_response)
plt.hist(data.age)

plt.title("Histogram of age of the customers")

plt.xlabel("Age of the customers")

plt.ylabel("Number of customers")
plt.hist(data.insured_sex)

plt.title("Histogram of gender count of the customers")

plt.xlabel("Gender")

plt.ylabel("Number of customers")
plt.hist(data.fraud_reported)

plt.title("Histogram of fraud reported")

plt.xlabel("response")

plt.ylabel("Number of responses")
sns.heatmap(data.corr())

plt.show()
predictors = data.iloc[:,:-1]

response = data.iloc[:, -1]



# new_response = []



# for i in range(len(response)):

#     new_response.append(1 if response[i]=='Y' else 0)

    

# response = pd.Series(new_response)



categorical_data = predictors.select_dtypes(exclude="number")

categorical_predictors = categorical_data.columns



predictors = predictors.drop(categorical_predictors, axis=1)
one_hot_data = pd.get_dummies(categorical_data)

predictors = predictors.join(one_hot_data)



predictor_columns = predictors.columns

response_columns = response



predictors_train, predictors_test, response_train, response_test = train_test_split(predictors,

                                                                                    response,

                                                                                    test_size=0.3)
sm = SMOTE(random_state=24)

predictors, response = sm.fit_resample(predictors_train, response_train)



predictors_train = pd.DataFrame(predictors, columns=predictor_columns)

response_train = pd.Series(response)



model_preds = {}
model = LogisticRegression()

model.fit(predictors_train, response_train)

predictions_test = model.predict(predictors_test)

predictions_train = model.predict(predictors_train)



conf_matrix = confusion_matrix(predictions_test, response_test)

plot_confusion_matrix(conf_matrix)



precision = precision_score(predictions_test, response_test)

recall = recall_score(predictions_test, response_test)



print("*****************************************")

print("Results on testing data:")

print("*****************************************")

print("Accuracy = "+str(accuracy_score(predictions_test, response_test)))

print("Precision = "+str(precision))

print("Recall = "+str(recall))



tpr, fpr, threshold = roc_curve(predictions_test, response_test, pos_label=1)

model_preds["Logistic Regression"] = [tpr, fpr]

print()

print("AUC value = "+str(auc(tpr, fpr)))
knn = KNeighborsClassifier()

knn.fit(predictors_train, response_train)



predictions_train = knn.predict(predictors_train)

predictions_test = knn.predict(predictors_test)



conf_matrix = confusion_matrix(predictions_test, response_test)

plot_confusion_matrix(conf_matrix)



precision = precision_score(predictions_test, response_test)

recall = recall_score(predictions_test, response_test)



print("*****************************************")

print("Results on testing data:")

print("*****************************************")

print("Accuracy = "+str(accuracy_score(predictions_test, response_test)))

print("Precision = "+str(precision))

print("Recall = "+str(recall))



tpr, fpr, threshold = roc_curve(predictions_test, response_test, pos_label=1)

model_preds["K Nearest Neighbor"] = [tpr, fpr]

print()

print("AUC value = "+str(auc(tpr, fpr)))
## Since it has a lot of categorical variables and the dataset is also not huge we 

## will use decision trees to get more accuracy.



predictors_train, predictors_test, response_train, response_test = train_test_split(predictors,response,test_size=0.3)



tree = DecisionTreeClassifier()

tree.fit(predictors_train, response_train)

predictions_test = tree.predict(predictors_test)

predictions_train = tree.predict(predictors_train)



conf_matrix = confusion_matrix(predictions_test, response_test)

plot_confusion_matrix(conf_matrix)



precision = precision_score(predictions_test, response_test)

recall = recall_score(predictions_test, response_test)



print("*****************************************")

print("Results on testing data:")

print("*****************************************")

print("Accuracy = "+str(accuracy_score(predictions_test, response_test)))

print("Precision = "+str(precision))

print("Recall = "+str(recall))



tpr, fpr, threshold = roc_curve(predictions_test, response_test, pos_label=1)

model_preds["Decision Tree"] = [tpr, fpr]

print()

print("AUC value = "+str(auc(tpr, fpr)))
from sklearn.ensemble import RandomForestClassifier



random_forest = RandomForestClassifier()

random_forest.fit(predictors_train, response_train)

predictions_test = random_forest.predict(predictors_test)

predictions_train = random_forest.predict(predictors_train)



conf_matrix = confusion_matrix(predictions_test, response_test)

plot_confusion_matrix(conf_matrix)



precision = precision_score(predictions_test, response_test)

recall = recall_score(predictions_test, response_test)



print("*****************************************")

print("Results on testing data:")

print("*****************************************")

print("Accuracy = "+str(accuracy_score(predictions_test, response_test)))

print("Precision = "+str(precision))

print("Recall = "+str(recall))



tpr, fpr, threshold = roc_curve(predictions_test, response_test, pos_label=1)

model_preds["Random Forest"] = [tpr, fpr]

print()

print("AUC value = "+str(auc(tpr, fpr)))
lda = LinearDiscriminantAnalysis()

lda.fit(predictors_train, response_train)

predictions_test = lda.predict(predictors_test)

predictions_train = lda.predict(predictors_train)



conf_matrix = confusion_matrix(predictions_test, response_test)

plot_confusion_matrix(conf_matrix)



precision = precision_score(predictions_test, response_test)

recall = recall_score(predictions_test, response_test)



print("*****************************************")

print("Results on testing data:")

print("*****************************************")

print("Accuracy = "+str(accuracy_score(predictions_test, response_test)))

print("Precision = "+str(precision_score(predictions_test, response_test)))

print("Recall = "+str(recall_score(predictions_test, response_test)))



tpr, fpr, threshold = roc_curve(predictions_test, response_test, pos_label=1)

model_preds["Linear Discriminant Analysis"] = [tpr, fpr]

print()

print("AUC value = "+str(auc(tpr, fpr)))
plt.title("ROC curve for various classifiers:")

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")



for key, value in model_preds.items():

    model_list = model_preds[key]

    plt.plot(model_list[0], model_list[1], label=key)

    plt.legend()

plt.show()