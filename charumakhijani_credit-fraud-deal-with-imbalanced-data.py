import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, precision_recall_curve, f1_score, accuracy_score, recall_score, precision_score
from sklearn.manifold import TSNE
dataset = pd.read_csv("../input/creditcardfraud/creditcard.csv")
dataset.head()
dataset.Class.value_counts()
dataset.shape
dataset.info()
dataset.describe()
dataset.isna().sum()
# Plot Fraud vs Non-fraud cases
plt.figure(figsize=(10,5))
ax = dataset.Class.value_counts().plot(kind = 'bar')
plt.xlabel("Fraud vs Non-fraud cases")
plt.ylabel("Count")
plt.title("Fraud vs Non-fraud cases Count")
fig, ax = plt.subplots(1, 2, figsize=(20,5))

sns.distplot(dataset['Amount'].values, ax=ax[0], color='g')
ax[0].set_title('Amount Distribution')
ax[0].set_xlim([min(dataset['Amount'].values), max(dataset['Amount'].values)])

sns.distplot(dataset['Time'].values, ax=ax[1], color='b')
ax[1].set_title('Time Distribution')
ax[1].set_xlim([min(dataset['Time'].values), max(dataset['Time'].values)])

plt.show()
# Heatmap to find any high correlations
plt.figure(figsize=(20,10))
sns.heatmap(data=dataset.corr(), cmap="seismic")
plt.show()
X = dataset.drop(["Class"], axis = 1)
y = dataset["Class"]
tsne = TSNE(n_components=2, random_state=0)
X_tsne = tsne.fit_transform(X.values)
import matplotlib.patches as mpatches
blue_patch = mpatches.Patch(color='#0A0AFF', label='No Fraud')
red_patch = mpatches.Patch(color='#AF0000', label='Fraud')
plt.figure(figsize=(20,10))
plt.scatter(X_tsne[:,0], X_tsne[:,1], c=(y == 0), cmap='coolwarm', label='No Fraud', linewidths=2)
plt.scatter(X_tsne[:,0], X_tsne[:,1], c=(y == 1), cmap='coolwarm', label='Fraud', linewidths=2)
plt.title('t-SNE')
plt.legend(handles=[blue_patch, red_patch])
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
X_trainA = X_train.drop(["Time"], axis = 1)
X_testA = X_test.drop(["Time"], axis = 1)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_trainA)
X_test = sc.transform(X_testA)
print(y_train[y_train.values == 0].shape[0])
print(y_train[y_train.values == 1].shape[0])
print(y_test[y_test.values == 0].shape[0])
print(y_test[y_test.values == 1].shape[0])
def fit_and_predict(classifier, X_train, X_test, y_train, y_test):
    classifier.fit(X_train, y_train)
    ypred = classifier.predict(X_test)
    
    print("Accuracy Score:", accuracy_score(y_test, ypred))
    print("Recall Score:", recall_score(y_test, ypred))
    print("Precision Score:", precision_score(y_test, ypred))
    
    print("\n*********Confusion Matrix*********")
    cm = confusion_matrix(y_test, ypred)
    print(cm)
    fig= plt.figure(figsize=(10,5))
    sns.heatmap(cm,cmap="coolwarm_r",annot=True,linewidths=0.5)
    plt.title("Confusion_matrix")
    plt.xlabel("Predicted_class")
    plt.ylabel("Real class")
    plt.show()
    print("\n*********Classification Report*********")
    print(classification_report(y_test, ypred))
    
    test_df = pd.DataFrame(X_test, columns = X.columns[1:])
    test_df['Actual'] = y_test.values
    test_df['Predicted'] = ypred
    test_df.head()
    tp = test_df[(test_df['Actual'] == 1) & (test_df['Predicted'] == 1)].shape[0]
    actual_positive = test_df[(test_df['Actual'] == 1)].shape[0]
    print("True Positives: ", tp)
    print("Accuracy for fraud cases: ", (tp / actual_positive))
    print("ROC AUC Score: ", roc_auc_score(y_test, ypred))
    return (y_test, ypred)
#     return roc_curve(y_test, ypred)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
y_test, ypred = fit_and_predict(lr, X_train, X_test, y_train, y_test)
lr_fp, lr_tp, lr_threshold = roc_curve(y_test, ypred)
lr_precision, lr_recall, _ = precision_recall_curve(y_test, ypred)
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
y_test, ypred = fit_and_predict(dtree, X_train, X_test, y_train, y_test)
dtree_fp, dtree_tp, dtree_threshold = roc_curve(y_test, ypred)
dtree_precision, dtree_recall, _ = precision_recall_curve(y_test, ypred)
# print("Threshold:", dtree_threshold)
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100)
y_test, ypred = fit_and_predict(rf, X_train, X_test, y_train, y_test)
rf_fp, rf_tp, rf_threshold = roc_curve(y_test, ypred)
rf_precision, rf_recall, _ = precision_recall_curve(y_test, ypred)
# print("Threshold:", rf_threshold)
from sklearn.ensemble import AdaBoostClassifier
adb = AdaBoostClassifier(n_estimators = 100, random_state = 0)
y_test, ypred = fit_and_predict(adb, X_train, X_test, y_train, y_test)
adb_fp, adb_tp, adb_threshold = roc_curve(y_test, ypred)
adb_precision, adb_recall, _ = precision_recall_curve(y_test, ypred)
from xgboost import XGBClassifier
xgb = XGBClassifier(random_state = 0)
y_test, ypred = fit_and_predict(xgb, X_train, X_test, y_train, y_test)
xgb_fp, xgb_tp, xgb_threshold = roc_curve(y_test, ypred)
xgb_precision, xgb_recall, _ = precision_recall_curve(y_test, ypred)
plt.figure(figsize=(20,10))
plt.plot([0, 1], [0, 1], linestyle = "--")
plt.plot(lr_fp, lr_tp, color="red", label ="Logistic Regression")
plt.plot(dtree_fp, dtree_tp, color="green", label = "Decision Tree")
plt.plot(rf_fp, rf_tp, color="blue", label = "Random Forest")
plt.plot(adb_fp, adb_tp, color="orange", label = "AdaBoost")
plt.plot(xgb_fp, xgb_tp, color="cyan", label = "XGBoost")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
fig = plt.figure(figsize=(20,10))
plt.step(lr_recall, lr_precision, marker='.', color='red', label ="Logistic Regression")
plt.step(dtree_recall, dtree_precision, marker='.', color='green', label = "Decision Tree")
plt.step(rf_recall, rf_precision, marker='.', color='blue', label = "Random Forest")
plt.step(adb_recall, adb_precision, marker='.', color='orange', label = "AdaBoost")
plt.step(xgb_recall, xgb_precision, marker='.', color='cyan', label = "XGBoost")

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall curve')
plt.legend()
print(X_trainA.shape, y_train.shape)
print(X_testA.shape, y_test.shape)
X_train1 = X_trainA
X_train1['Class'] = y_train
X_train1.head()
X_train_0_class, X_train_1_class = X_train1.Class.value_counts()
print(X_train_0_class, X_train_1_class)
X_train1_0_df = X_train1[X_train1['Class']==0]
X_train1_1_df = X_train1[X_train1['Class']==1]
print(X_train1_0_df.shape, X_train1_1_df.shape)
X_train1_1_df = X_train1_1_df.sample(X_train_0_class, replace=True, random_state=0)
print(X_train1_0_df.shape, X_train1_1_df.shape)
X_train1 = pd.concat([X_train1_0_df, X_train1_1_df])
X_train1.shape
print(X_train1[X_train1['Class']==0].shape)
print(X_train1[X_train1['Class']==1].shape)
X_trainB = X_train1.drop("Class", axis =1)
y_trainB = X_train1["Class"]
print(X_trainB.shape, y_trainB.shape)
print(X_testA.shape, y_test.shape)
X_trainB = sc.fit_transform(X_trainB)
X_testA = sc.transform(X_testA)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
y_test, ypred = fit_and_predict(lr, X_trainB, X_testA, y_trainB, y_test)
lr_fp, lr_tp, lr_threshold = roc_curve(y_test, ypred)
lr_precision, lr_recall, _ = precision_recall_curve(y_test, ypred)
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
y_test, ypred = fit_and_predict(dtree, X_trainB, X_testA, y_trainB, y_test)
dtree_fp, dtree_tp, dtree_threshold = roc_curve(y_test, ypred)
dtree_precision, dtree_recall, _ = precision_recall_curve(y_test, ypred)
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100)
y_test, ypred = fit_and_predict(rf, X_trainB, X_testA, y_trainB, y_test)
rf_fp, rf_tp, rf_threshold = roc_curve(y_test, ypred)
rf_precision, rf_recall, _ = precision_recall_curve(y_test, ypred)
from sklearn.ensemble import AdaBoostClassifier
adb = AdaBoostClassifier(n_estimators = 100, random_state = 0)
y_test, ypred = fit_and_predict(adb, X_trainB, X_testA, y_trainB, y_test)
adb_fp, adb_tp, adb_threshold = roc_curve(y_test, ypred)
adb_precision, adb_recall, _ = precision_recall_curve(y_test, ypred)
from xgboost import XGBClassifier
xgb = XGBClassifier(random_state = 0)
y_test, ypred = fit_and_predict(xgb, X_trainB, X_testA, y_trainB, y_test)
xgb_fp, xgb_tp, xgb_threshold = roc_curve(y_test, ypred)
xgb_precision, xgb_recall, _ = precision_recall_curve(y_test, ypred)
import keras
from keras.layers import Dense
from keras.models import Sequential
classifier = Sequential()
classifier.add(Dense(units=16, activation="relu", input_dim=29))
classifier.add(Dense(units=2, activation="softmax"))
classifier.summary()
classifier.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
classifier.fit(X_trainB, y_trainB, batch_size=10, epochs=20)
ypred_nn = classifier.predict_classes(X_testA)
print(confusion_matrix(y_test, ypred_nn))
print(classification_report(y_test, ypred_nn))
print("Accuracy Score:", accuracy_score(y_test, ypred_nn))

test_df = pd.DataFrame(X_test, columns = X.columns[1:])
test_df['Actual'] = y_test.values
test_df['Predicted'] = ypred_nn
test_df.head()
tp = test_df[(test_df['Actual'] == 1) & (test_df['Predicted'] == 1)].shape[0]
actual_positive = test_df[(test_df['Actual'] == 1)].shape[0]
print("True Positives: ", tp)
print("Accuracy for fraud cases: ", (tp / actual_positive))
print("ROC AUC Score: ", roc_auc_score(y_test, ypred_nn))
nn_fp, nn_tp, nn_threshold = roc_curve(y_test, ypred_nn)
nn_precision, nn_recall, _ = precision_recall_curve(y_test, ypred_nn)
classifier.evaluate(X_testA, y_test)
plt.figure(figsize=(20,10))
plt.plot([0, 1], [0, 1], linestyle = "--")
plt.plot(lr_fp, lr_tp, color="red", label ="Logistic Regression")
plt.plot(dtree_fp, dtree_tp, color="green", label = "Decision Tree")
plt.plot(rf_fp, rf_tp, color="blue", label = "Random Forest")
plt.plot(adb_fp, adb_tp, color="orange", label = "AdaBoost")
plt.plot(xgb_fp, xgb_tp, color="cyan", label = "XGBoost")
plt.plot(nn_fp, nn_tp, color="purple", label = "Neural Networks")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
fig = plt.figure(figsize=(20,10))
plt.step(lr_recall, lr_precision, marker='.', color='red', label ="Logistic Regression")
plt.step(dtree_recall, dtree_precision, marker='.', color='green', label = "Decision Tree")
plt.step(rf_recall, rf_precision, marker='.', color='blue', label = "Random Forest")
plt.step(adb_recall, adb_precision, marker='.', color='orange', label = "AdaBoost")
plt.step(xgb_recall, xgb_precision, marker='.', color='cyan', label = "XGBoost")
plt.step(nn_recall, nn_precision, marker='.', color="purple", label = "Neural Networks")

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall curve')
plt.legend()
X_trainA.columns[:-1]
feature_importance_df = pd.DataFrame(X_trainA.columns[:-1], columns=["Feature"])
feature_importance_df["Importance"] = rf.feature_importances_
feature_importance_df.sort_values('Importance', ascending=False, inplace=True)
feature_importance_df = feature_importance_df.head(20)
feature_importance_df
plt.figure(figsize=(15,5))
ax = feature_importance_df['Feature']
plt.bar(range(feature_importance_df.shape[0]), feature_importance_df['Importance']*100)
plt.xticks(range(feature_importance_df.shape[0]), feature_importance_df['Feature'], rotation = 20)
plt.xlabel("Features")
plt.ylabel("Importance")
plt.title("Plot Feature Importances")
