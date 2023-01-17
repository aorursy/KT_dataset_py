import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
sns.set(style="darkgrid")
dataset = pd.read_csv("../input/placement-data-full-class/Placement_data_full_class.csv")
dataset.head()
dataset.isnull().sum()
sns.distplot(dataset.salary)
dataset = dataset.fillna(0)
sns.barplot(x = dataset['gender'],y = dataset['salary'])
sns.barplot(x=dataset["gender"],y=dataset["ssc_p"])
sns.barplot(x=dataset["gender"],y=dataset["hsc_p"])
sns.barplot(x=dataset["gender"],y=dataset["degree_p"])
sns.barplot(x=dataset["gender"],y=dataset["mba_p"])
sns.boxplot(x=dataset["status"],y=dataset["mba_p"])
sns.boxplot(x=dataset["status"],y=dataset["degree_p"])
sns.boxplot(x=dataset["status"],y=dataset["hsc_p"])
sns.boxplot(x=dataset["status"],y=dataset["ssc_p"])
dataset=dataset[dataset.salary<600000]
sns.jointplot(x=dataset["ssc_p"],y=dataset["salary"],kind="reg",color="g")
sns.jointplot(x=dataset["hsc_p"],y=dataset["salary"],kind="reg",color="g")
sns.jointplot(x=dataset["degree_p"],y=dataset["salary"],kind="reg",color="g")
sns.jointplot(x=dataset["mba_p"],y=dataset["salary"],kind="reg",color="g")
sns.jointplot(x=dataset["etest_p"],y=dataset["salary"],kind="reg",color="g")
plt.rc("axes",labelsize=13)
plt.rc("xtick",labelsize=13)
plt.rc("ytick",labelsize=13)
sns.countplot(x=dataset["specialisation"],hue=dataset["status"],palette="muted").set_title("Barplot showing placement among specalisation")
plt.rc("axes",labelsize=13)
plt.rc("xtick",labelsize=13)
plt.rc("ytick",labelsize=13)
sns.countplot(x=dataset["workex"],hue=dataset["status"],palette="muted").set_title("Barplot showing placement acocording work experiences")
dataset.dtypes
numeric_data= dataset.select_dtypes(include=[np.number])
categorical_data = dataset.select_dtypes(exclude=[np.number])
print ("There are {} numeric and {} categorical columns in dataset"
.format(numeric_data.shape[1],categorical_data.shape[1]))
# using label Encoder to change categorical data to numerical
from sklearn.preprocessing import LabelEncoder
le =  LabelEncoder()
# implementing le on gender
le.fit(dataset.gender.drop_duplicates())
dataset.gender = le.transform(dataset.gender)
# implementing le on ssc_b
le.fit(dataset.ssc_b.drop_duplicates())
dataset.ssc_b = le.transform(dataset.ssc_b)
# implementing le on hsc_b
le.fit(dataset.hsc_b.drop_duplicates())
dataset.hsc_b = le.transform(dataset.hsc_b)
# implementing le on hsc_b
le.fit(dataset.hsc_b.drop_duplicates())
dataset.hsc_b = le.transform(dataset.hsc_b)
# implementing le on hsc_s
le.fit(dataset.hsc_s.drop_duplicates())
dataset.hsc_s = le.transform(dataset.hsc_s)
# implementing le on degree_t
le.fit(dataset.degree_t.drop_duplicates())
dataset.degree_t = le.transform(dataset.degree_t)
# implementing le on workex
le.fit(dataset.workex.drop_duplicates())
dataset.workex = le.transform(dataset.workex)
# implementing le on specialisation	
le.fit(dataset.specialisation.drop_duplicates())
dataset.specialisation = le.transform(dataset.specialisation)
# implementing le on status
le.fit(dataset.status.drop_duplicates())
dataset.status = le.transform(dataset.status)
# now dataset is converted into numerical value
dataset.head()
plt.figure(figsize=(12,10))
corrMatrix=dataset.corr()
sns.heatmap(corrMatrix,annot=True)
plt.show()
x = dataset.iloc[:, 1:13].values
y = dataset.iloc[:, -2].values
x
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.25,random_state=0)
print(x_train.shape)
print(x_test.shape)
x_train
x_test
from imblearn.combine import SMOTETomek
smk = SMOTETomek(random_state = 42)
x,y = smk.fit_sample(x,y)
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression (solver='liblinear', random_state=0)
classifier.fit(x_train,y_train)
y_pred = classifier.predict(x_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
accuracy = accuracy_score(y_test,y_pred)
precision =  precision_score(y_test,y_pred,average="weighted")
recall = recall_score(y_test,y_pred,average="weighted")
f1 = f1_score(y_test,y_pred,average="weighted")
print("Accuracy - {}".format(accuracy))
print("Precision - {}".format(precision))
print("Recall- {}".format(recall))
print("f1 - {}".format(f1))
from sklearn.metrics import average_precision_score
average_precision = average_precision_score(y_test,y_pred)
print(average_precision)
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
disp = plot_precision_recall_curve(classifier,x_test,y_test)
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
y_score2 = classifier.predict_proba(x_test)[:,1]
false_positive_rate2, true_positive_rate2, threshold2 = roc_curve(y_test, y_score2)
print('roc_auc_score for Logistic Regression: ', roc_auc_score(y_test, y_score2))
plt.subplots(1, figsize=(8,6))
plt.title('Receiver Operating Characteristic - Logistic regression')
plt.plot(false_positive_rate2, true_positive_rate2)
plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
from sklearn.neighbors import KNeighborsClassifier
classifierr = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                     metric_params=None, n_jobs=None, n_neighbors=5, p=2,
                     weights='uniform')
classifierr.fit(x_train,y_train)
y_pred = classifierr.predict(x_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)
accuracy = accuracy_score(y_test,y_pred)
precision = precision_score(y_test,y_pred, average="weighted")
recall = recall_score(y_test,y_pred, average="weighted")
f1 = f1_score(y_test,y_pred, average="weighted")
print("Accuracy - {}".format(accuracy))
print("Precision - {}".format(precision))
print("Recall - {}".format(recall))
print("f1 - {}".format(f1))
average_precision = average_precision_score(y_test,y_pred)
print(average_precision)
disp = plot_precision_recall_curve(classifierr,x_test,y_test)
y_score1 = classifierr.predict_proba(x_test)[:,1]
false_positive_rate1, true_positive_rate1, threshold2 = roc_curve(y_test, y_score1)
print('roc_auc_score for k_nearest_neibour: ', roc_auc_score(y_test, y_score1))
plt.subplots(1, figsize=(8,6))
plt.title('Receiver Operating Characteristic - k_nearest_neibour')
plt.plot(false_positive_rate1, true_positive_rate1)


plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend()
plt.show()
from sklearn.svm import SVC
classifier1 = SVC(kernel="linear",random_state=0,C=1.0, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='scale',
    max_iter=-1, probability=True,shrinking=True, tol=0.001,
    verbose=False)
classifier1.fit(x_train,y_train)
y_pred = classifier1.predict(x_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)
accuracy = accuracy_score(y_test,y_pred)
precision = precision_score(y_test,y_pred, average="weighted")
recall = recall_score(y_test,y_pred, average="weighted")
f1 = f1_score(y_test,y_pred, average="weighted")
print("Accuracy - {}".format(accuracy))
print("Precision - {}".format(precision))
print("Recall - {}".format(recall))
print("f1 - {}".format(f1))
average_precision = average_precision_score(y_test,y_pred)
print(average_precision)
disp = plot_precision_recall_curve(classifier1,x_test,y_test)
y_score3 = classifier1.predict_proba(x_test)[:,1]
false_positive_rate3, true_positive_rate3, threshold3 = roc_curve(y_test, y_score3)
print('roc_auc_score for support vector machine ', roc_auc_score(y_test, y_score3))
plt.subplots(1, figsize=(8,6))
plt.title('Receiver Operating Characteristic - Support vector machine')
plt.plot(false_positive_rate3, true_positive_rate3)

plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend()
plt.show()
from sklearn.svm import SVC
classifier2 = SVC(kernel="rbf",random_state=0,probability=True)
classifier2.fit(x_train,y_train)
y_pred = classifier2.predict(x_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
cm = confusion_matrix(y_test,y_pred)
print(cm)
accuracy_score(y_test,y_pred)
accuracy = accuracy_score(y_test,y_pred)
precision = precision_score(y_test,y_pred, average="weighted")
recall = recall_score(y_test,y_pred, average="weighted")
f1 = f1_score(y_test,y_pred, average="weighted")
print("Accuracy - {}".format(accuracy))
print("Precision - {}".format(precision))
print("Recall - {}".format(recall))
print("f1 - {}".format(f1))
average_precision = average_precision_score(y_test,y_pred)
print(average_precision)
disp = plot_precision_recall_curve(classifier2,x_test,y_test)
y_score4 = classifier2.predict_proba(x_test)[:,1]
false_positive_rate4, true_positive_rate4, threshold4 = roc_curve(y_test, y_score4)
print('roc_auc_score for support vector machine ', roc_auc_score(y_test, y_score4))
plt.subplots(1, figsize=(8,6))
plt.title('Receiver Operating Characteristic - Kernel SVM')
plt.plot(false_positive_rate4, true_positive_rate4)


plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend()
plt.show()
from sklearn.naive_bayes import GaussianNB
classifier3 = GaussianNB()
classifier3.fit(x_train, y_train)
y_pred = classifier3.predict(x_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

cm = confusion_matrix(y_test,y_pred)
print(cm)
accuracy_score(y_test,y_pred)
accuracy = accuracy_score(y_test,y_pred)
precision = precision_score(y_test,y_pred, average="weighted")
recall = recall_score(y_test,y_pred, average="weighted")
f1 = f1_score(y_test,y_pred, average="weighted")
print("Accuracy - {}".format(accuracy))
print("Precision - {}".format(precision))
print("Recall - {}".format(recall))
print("f1 - {}".format(f1))
average_precision = average_precision_score(y_test,y_pred)
print(average_precision)
disp = plot_precision_recall_curve(classifier3,x_test,y_test)
y_score5 = classifier3.predict_proba(x_test)[:,1]
false_positive_rate5, true_positive_rate5, threshold5 = roc_curve(y_test, y_score5)
print('roc_auc_score for support vector machine ', roc_auc_score(y_test, y_score5))
plt.subplots(1, figsize=(8,6))
plt.title('Receiver Operating Characteristic - Naive Bayes')
plt.plot(false_positive_rate5, true_positive_rate5)


plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend()
plt.show()
from sklearn.tree import DecisionTreeClassifier
classifier4 = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier4.fit(x_train, y_train)
y_pred = classifier4.predict(x_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
cm = confusion_matrix(y_test,y_pred)
print(cm)
accuracy_score(y_test,y_pred)
accuracy = accuracy_score(y_test,y_pred)
precision = precision_score(y_test,y_pred, average="weighted")
recall = recall_score(y_test,y_pred, average="weighted")
f1 = f1_score(y_test,y_pred, average="weighted")
print("Accuracy - {}".format(accuracy))
print("Precision - {}".format(precision))
print("Recall - {}".format(recall))
print("f1 - {}".format(f1))
average_precision = average_precision_score(y_test,y_pred)
print(average_precision)
disp = plot_precision_recall_curve(classifier4,x_test,y_test)
y_score6 = classifier4.predict_proba(x_test)[:,1]
false_positive_rate6, true_positive_rate6, threshold6 = roc_curve(y_test, y_score6)
print('roc_auc_score for support vector machine ', roc_auc_score(y_test, y_score6))
plt.subplots(1, figsize=(8,6))
plt.title('Receiver Operating Characteristic - decision tree')
plt.plot(false_positive_rate6, true_positive_rate6)

plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend()
plt.show()
from sklearn.ensemble import RandomForestClassifier
classifier5 = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier5.fit(x_train, y_train)
y_pred = classifier5.predict(x_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
cm = confusion_matrix(y_test,y_pred)
print(cm)
accuracy_score(y_test,y_pred)
accuracy = accuracy_score(y_test,y_pred)
precision = precision_score(y_test,y_pred, average="weighted")
recall = recall_score(y_test,y_pred, average="weighted")
f1 = f1_score(y_test,y_pred, average="weighted")
print("Accuracy - {}".format(accuracy))
print("Precision - {}".format(precision))
print("Recall - {}".format(recall))
print("f1 - {}".format(f1))
average_precision = average_precision_score(y_test,y_pred)
print(average_precision)
disp = plot_precision_recall_curve(classifier5,x_test,y_test)
y_score7 = classifier5.predict_proba(x_test)[:,1]
false_positive_rate7, true_positive_rate7, threshold6 = roc_curve(y_test, y_score7)
print('roc_auc_score for support vector machine ', roc_auc_score(y_test, y_score7))
plt.subplots(1, figsize=(8,8))
plt.title('Receiver Operating Characteristic - Random forest')
plt.plot(false_positive_rate7, true_positive_rate7)

plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend()
plt.show()
classifiers = [LogisticRegression(random_state=0),
               KNeighborsClassifier(),
               SVC(random_state=0,probability=True), 
               GaussianNB(), 
               DecisionTreeClassifier(random_state=0),
               RandomForestClassifier(random_state=0)]
result_table = pd.DataFrame(columns=['classifiers', 'fpr','tpr','auc'])
for cls in classifiers:
    model = cls.fit(x_train, y_train)
    yproba = model.predict_proba(x_test)[:,1]
    
    fpr, tpr, _ = roc_curve(y_test,  yproba)
    auc = roc_auc_score(y_test, yproba)
    
    result_table = result_table.append({'classifiers':cls.__class__.__name__,
                                        'fpr':fpr, 
                                        'tpr':tpr, 
                                        'auc':auc}, ignore_index=True)
result_table.set_index('classifiers', inplace=True)

fig = plt.figure(figsize=(8,6))

for i in result_table.index:
    plt.plot(result_table.loc[i]['fpr'], 
             result_table.loc[i]['tpr'], 
             label="{}, AUC={:.3f}".format(i, result_table.loc[i]['auc']))
    
plt.plot([0,1], [0,1], color='orange', linestyle='--')

plt.xticks(np.arange(0.0, 1.1, step=0.1))
plt.xlabel("Flase Positive Rate", fontsize=15)

plt.yticks(np.arange(0.0, 1.1, step=0.1))
plt.ylabel("True Positive Rate", fontsize=15)

plt.title('ROC Curve Analysis', fontweight='bold', fontsize=15)
plt.legend(prop={'size':13}, loc='lower right')

plt.show()