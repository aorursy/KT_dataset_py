import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
df_train = pd.read_csv("../input/train.csv")
df_train.head(5)
df_train.shape
y = df_train['label']
df = df_train.drop(['label'], axis=1)
df.head()
# To plot pretty figures
%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12


digit_to_predict_raw = np.array(df.iloc[5000,:])
digit_to_predict  = np.array(digit_to_predict_raw).reshape(28,28)
digit_to_predict.shape
plt.imshow(digit_to_predict,cmap = matplotlib.cm.binary)
plt.show()
y[5000]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.30, random_state=42)
y_train_8 = np.array(y_train == 8)
y_test_8 = np.array(y_test == 8)
from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(max_iter=5, random_state=42)
sgd_clf.fit(X_train, y_train_8)
pred = sgd_clf.predict(X_test)
pred
y_test[:4]
from sklearn.model_selection import cross_val_score
cross_val_score(sgd_clf, X_train, y_train_8, cv=5, scoring="accuracy")
import collections
collections.Counter(y_train_8)
num_of_8_not_occur = collections.Counter(y_train_8)[0]
print("The accuracy of model if we predict there are NO 8 present in the dataset :",
      num_of_8_not_occur/len(y_train_8))
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
y_train_8_pred = cross_val_predict(sgd_clf, X_train, y_train_8, cv=3)
confusion_matrix(y_train_8, y_train_8_pred)

from sklearn import metrics
def get_metrics(true_labels, predicted_labels):
    
    print('Accuracy:', np.round(
                        metrics.accuracy_score(true_labels, 
                                               predicted_labels),
                        4))
    print('Precision:', np.round(
                        metrics.precision_score(true_labels, 
                                               predicted_labels,
                                               average='weighted'),
                        4))
    print('Recall:', np.round(
                        metrics.recall_score(true_labels, 
                                               predicted_labels,
                                               average='weighted'),
                        4))
    print('F1 Score:', np.round(
                        metrics.f1_score(true_labels, 
                                               predicted_labels,
                                               average='weighted'),
                        4))
get_metrics(y_train_8, y_train_8_pred)
def display_confusion_matrix(true_labels, predicted_labels, classes=[1,0]):
    
    total_classes = len(classes)
    level_labels = [total_classes*[0], list(range(total_classes))]

    cm = metrics.confusion_matrix(y_true=true_labels, y_pred=predicted_labels, 
                                  labels=classes)
    cm_frame = pd.DataFrame(data=cm, 
                            columns=pd.MultiIndex(levels=[['Predicted:'], classes], 
                                                  labels=level_labels), 
                            index=pd.MultiIndex(levels=[['Actual:'], classes], 
                                                labels=level_labels)) 
    print(cm_frame) 
display_confusion_matrix(y_train_8, y_train_8_pred)
y_scores = cross_val_predict(sgd_clf, X_train, y_train_8, cv=3,
                             method="decision_function")
from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(y_train_8, y_scores)
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
    plt.xlabel("Threshold", fontsize=16)
    plt.legend(loc="upper left", fontsize=16)
    plt.ylim([0, 1])

plt.figure(figsize=(8, 4))
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.xlim([-1700000, 1700000])
plt.show()
y_train_prec_90 = (y_scores > 700000)
from sklearn.metrics import precision_score, recall_score
precision_score(y_train_8, y_train_prec_90)
recall_score(y_train_8, y_train_prec_90)
from sklearn.metrics import roc_auc_score, roc_curve
fpr, tpr, thresholds = roc_curve(y_train_8, y_scores)
def plot_roc_curve(fpr,tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label = label)
    plt.plot([0,1], [0,1],'k--')
    plt.axis([0,1,0,1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    
plot_roc_curve(fpr,tpr)

print("The AUC score is :", roc_auc_score(y_train_8, y_scores))
from sklearn.ensemble import RandomForestClassifier
forest_clf = RandomForestClassifier(n_jobs=-1)
y_forest_pred = cross_val_predict(forest_clf, X_train, y_train_8, cv=3, method='predict_proba')
y_forest_pred_f = cross_val_predict(forest_clf, X_train, y_train_8, cv=3)
y_scores_forest= y_forest_pred[:,1]
fpr_f, tpr_f, thresholds_f = roc_curve(y_train_8, y_scores_forest)
plt.plot(fpr, tpr, "b:", label="SGD")
plot_roc_curve(fpr_f, tpr_f, "Random Forest")
plt.legend(loc="lower right")
plt.show()

print("Classification metrics for SGD :")
print("The AUC score for SGD is :", roc_auc_score(y_train_8, y_scores))
get_metrics(y_train_8, y_train_8_pred)
print("\nClassification metrics for RandomForest :")
print("The AUC score is RandomForest is :", roc_auc_score(y_train_8, y_scores_forest))
get_metrics(y_train_8, y_forest_pred_f)
sgd_multi_clf = SGDClassifier(max_iter=5, random_state=42)
sgd_multi_clf.fit(X_train, y_train)  
## we'll be using y_train here as scikit learn trains the model against actual 
## target labels 0 to 9 (10 classifiers each for 1 class); it'll compare 10 
## classification scores and choose the class having the highest score 
## for a new data point.
sgd_multi_clf.predict([digit_to_predict_raw])
## So, here we can see the scores against all the 10 classifiers
digit_to_predict_scores = sgd_multi_clf.decision_function([digit_to_predict_raw])
max_score_index = np.argmax(digit_to_predict_scores)  ## index of the maximum score
print("Scores of 10 classifiers :", digit_to_predict_scores)
## We'll find out the class corresponding to the index value of the mximum score
print("Maximum score among all these Scores is for Class : {} having Score of : {}" 
      .format(sgd_multi_clf.classes_[max_score_index], digit_to_predict_scores[0][8])) 
cross_val_score(sgd_multi_clf, X_train, y_train, cv=3, scoring="accuracy")
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
cross_val_score(sgd_multi_clf, X_train_scaled, y_train, cv=3, scoring="accuracy")
from sklearn.multiclass import OneVsOneClassifier
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
ovo_clf = OneVsOneClassifier(sgd_multi_clf)
ovo_clf.fit(X_train_scaled, y_train)
ovo_clf.predict([digit_to_predict_raw])

len(ovo_clf.estimators_)
from sklearn.ensemble import RandomForestClassifier
forest_clf = RandomForestClassifier(n_jobs=-1)
forest_clf.fit(X_train, y_train)
forest_clf.predict([digit_to_predict_raw])
cross_val_score(forest_clf, X_train, y_train, cv=3, scoring="accuracy")
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
y_train_pred_svm = cross_val_predict(sgd_multi_clf, X_train_scaled, y_train, cv=3)
conf_mx = confusion_matrix(y_train, y_train_pred_svm)
plt.matshow(conf_mx, cmap=plt.cm.gray)
plt.show()

## most images are on the main diagonal, which means that
## they were classified correctly
from sklearn.neighbors import KNeighborsClassifier
knn_clf = KNeighborsClassifier(n_jobs=-1, weights = 'distance', n_neighbors=4)
knn_clf.fit(X_train, y_train)
knn_clf.predict([digit_to_predict_raw])
cross_val_score(knn_clf, X_train, y_train, cv=3, scoring="accuracy")
