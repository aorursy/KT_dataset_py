#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# importing dataset
df = pd.read_csv("../input/Pokemon.csv", index_col = "#")
#print(df.shape)
#print(df.head(2))
df.columns.values
columns = ['Name', 'Type_1', 'Type_2', 'Total', 'HP', 'Attack', 'Defense','Sp_Atk', 'Sp_Def', 'Speed', 'Generation', 'Legendary']
df.columns = columns
df.head(2)
df.info()
df.describe()
df.columns.values
# converting legendary column vales into labels {true: 1, false: 0}
df.Legendary.replace({True:1,False:0}, inplace = True)
df.describe()
df.head(3)
# converting categorical values into dummies
type_1_dummies = pd.get_dummies(df.Type_1)
type_2_dummies = pd.get_dummies(df.Type_2)
print(type_1_dummies.shape)
type_1_dummies.head(2)
print(type_2_dummies.shape)
type_2_dummies.head(2)
types_df = pd.DataFrame(index = df.index)
types_df.shape
types = list(type_2_dummies.columns.values)
for t in types:
    types_df[t] = type_1_dummies[t] + type_2_dummies[t]
    
types_df.head(3)
temporary_df = pd.concat([df, types_df], sort = False, axis = 1)
print(temporary_df.shape)
temporary_df.head(3)
labels_df  = temporary_df.Legendary
temporary_df.drop(["Name","Type_1","Type_2","Legendary"], axis = "columns", inplace = True)
temporary_df.head(3)
labels_df.shape
num_df = temporary_df.iloc[:,0:7]
num_df.head(3)
cat_df = temporary_df.iloc[:,7:]
cat_df.head(3)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(num_df)
scaled_df = pd.DataFrame(np.array(scaler.transform(num_df)), columns = ["Total","HP","Attack","Defense","Sp_Atk","Sp_Def","Speed"], index = df.index)
print(scaled_df.shape)
scaled_df.head(3)
cat_df.shape
final_df = pd.concat([scaled_df,cat_df], axis = "columns")
final_df.head(3)
plt.figure(figsize = (20,20))
sns.heatmap(final_df.corr(), annot=True)
# train / test splitting of data
from sklearn.model_selection import train_test_split
feature_train, feature_test, label_train, label_test = train_test_split(final_df, labels_df, test_size = 0.2, random_state = 2)
print(feature_train.shape, feature_test.shape)
print(label_train.shape, label_test.shape)
feature_train.head(3)
label_train.head(3)
# Bernoulli naive bayes classifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
bnb = BernoulliNB()
bnb.fit(feature_train, label_train)
predicted_labels = bnb.predict(feature_test)
print("Train Score : ", bnb.score(feature_train, label_train))
print("Test Score : ", bnb.score(feature_test, label_test))
print("Accuracy : ",accuracy_score(label_test, predicted_labels))
# Confusion Matrix
from sklearn.metrics import confusion_matrix
cnf_matrix = confusion_matrix(label_test, predicted_labels)

p = sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, fmt = 'd')
plt.xlabel("Predicted Values")
plt.ylabel("Actual Values")
plt.title("Confusion Matrix for Bernoulli Naive Bayes Classifier")
# roc auc curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
pred_score = bnb.predict_proba(feature_test)[:,1]
fpr, tpr, thresholds = roc_curve(label_test, pred_score)
print("ROC-AUC Score : ", roc_auc_score(label_test, pred_score))
plt.plot([0,1],[0,1])
plt.plot(fpr, tpr)
plt.xlabel("fpr")
plt.ylabel("tpr")
plt.title("ROC-AUC Curve for Bernoulli Naive Bayes Classifier")
plt.show()
# Gaussian Naive Bayes Classifier
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(feature_train, label_train)
predicted_labels = gnb.predict(feature_test)
print("Train Score : ", gnb.score(feature_train, label_train))
print("Test Score : ", gnb.score(feature_test, label_test))
print("Accuracy : ",accuracy_score(label_test, predicted_labels))
# Confusion Matrix for Gaussian Naive Bayes Classifier
from sklearn.metrics import confusion_matrix
cnf_matrix = confusion_matrix(label_test, predicted_labels)

sns.heatmap(pd.DataFrame(cnf_matrix), fmt = "d", annot=True)
plt.ylabel("Actual Values")
plt.xlabel("Predicted Values")
plt.title("Confusion Matrix for Gaussian Naive Bayes Classifier")
# roc-auc curve for gaussian naive bayes classifier
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
pred_prob = gnb.predict_proba(feature_test)[:,1]
fpr, tpr, thresholds = roc_curve(label_test, pred_prob)
print("ROC-AUC Score : ", roc_auc_score(label_test, pred_prob))
plt.plot(fpr, tpr)
plt.plot([0,1],[0,1])
plt.xlabel("fpr")
plt.ylabel("tpr")
# CART Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
cart_model = DecisionTreeClassifier(criterion = "gini", random_state = 2)
cart_model.fit(feature_train, label_train)
predicted_labels = cart_model.predict(feature_test)
print("Train Score : ", cart_model.score(feature_train, label_train))
print("Test Score : ", cart_model.score(feature_test, label_test))
print("Accuracy : ", accuracy_score(label_test, predicted_labels))
# confusion matrix for decision tree cart classifier
from sklearn.metrics import confusion_matrix
cnf_matrix = confusion_matrix(label_test, predicted_labels)
sns.heatmap(pd.DataFrame(cnf_matrix), fmt = "d", annot = True)
plt.xlabel("Predicated values")
plt.ylabel("Actual values")
# roc-auc curve for decision tree cart classifier
from sklearn.metrics import roc_auc_score 
from sklearn.metrics import roc_curve
pred_prob = cart_model.predict_proba(feature_test)[:,1]
fpr,tpr, thresholds = roc_curve(label_test, pred_prob)
print("ROC-AUC Score : ", roc_auc_score(label_test, pred_prob))
plt.plot(fpr,tpr)
plt.plot([0,1],[0,1])
plt.xlabel('fpr')
plt.ylabel("tpr")
# C50 Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
c5 = DecisionTreeClassifier(criterion="entropy", random_state = 2)
c5.fit(feature_train, label_train)
predicted_labels = c5.predict(feature_test)
print("Train Score : ", c5.score(feature_train, label_train))
print("Test Score : ", c5.score(feature_test, label_test))
print("Accuracy : ", accuracy_score(label_test, predicted_labels))
# confusion matrix for decision tree c50 classifier
from sklearn.metrics import confusion_matrix
cnf_matrix = confusion_matrix(label_test, predicted_labels)
sns.heatmap(pd.DataFrame(cnf_matrix), fmt= "d", annot = True)
plt.xlabel("Predicted Values")
plt.ylabel("Actual Values")
# roc auc curve for decision tree c50 classifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
pred_prob = c5.predict_proba(feature_test)[:,1]
fpr,tpr,thresholds = roc_curve(label_test, pred_prob)
print("ROC-AUC Score : ", roc_auc_score(label_test, pred_prob))
plt.plot(fpr,tpr)
plt.plot([0,1],[0,1])
plt.xlabel("fpr")
plt.ylabel("tpr")
# K Nearest Neighbour Classsifier
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5, algorithm="auto",n_jobs=-1)
knn.fit(feature_train, label_train)
predicted_labels = knn.predict(feature_test)
print("Train Score : ", knn.score(feature_train, label_train))
print("Test Score : ", knn.score(feature_test, label_test))
print("Accuracy : ", accuracy_score(label_test, predicted_labels))
# confusion matrix for knn classifier
from sklearn.metrics import confusion_matrix
cnf_matrix = confusion_matrix(label_test, predicted_labels)
sns.heatmap(pd.DataFrame(cnf_matrix), fmt= "d", annot = True)
plt.xlabel("Predicted Values")
plt.ylabel("Actual Values")
# roc auc curve for knn classifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
pred_prob = knn.predict_proba(feature_test)[:,1]
fpr,tpr,thresholds = roc_curve(label_test, pred_prob)
print("ROC-AUC Score : ", roc_auc_score(label_test, pred_prob))
plt.plot(fpr,tpr)
plt.plot([0,1],[0,1])
plt.xlabel("fpr")
plt.ylabel("tpr")
# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_jobs=-1, n_estimators=300, random_state=2, max_depth=3)
rf.fit(feature_train, label_train)
predicted_labels = rf.predict(feature_test)
print("Train Score : ", rf.score(feature_train, label_train))
print("Test Score : ", rf.score(feature_test, label_test))
print("Accuracy : ", accuracy_score(label_test, predicted_labels))
# confusion matrix for random forest classifier
from sklearn.metrics import confusion_matrix
cnf_matrix = confusion_matrix(label_test, predicted_labels)
sns.heatmap(pd.DataFrame(cnf_matrix), fmt= "d", annot = True)
plt.xlabel("Predicted Values")
plt.ylabel("Actual Values")
# roc auc curve for random forest classifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
pred_prob = rf.predict_proba(feature_test)[:,1]
fpr,tpr,thresholds = roc_curve(label_test, pred_prob)
print("ROC-AUC Score : ", roc_auc_score(label_test, pred_prob))
plt.plot(fpr,tpr)
plt.plot([0,1],[0,1])
plt.xlabel("fpr")
plt.ylabel("tpr")
# Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier
gbm = GradientBoostingClassifier(n_estimators=400, learning_rate=0.01, subsample=0.4, max_depth=3, random_state=1)
gbm.fit(feature_train, label_train)
predicted_labels = gbm.predict(feature_test)
print("Train Score : ", gbm.score(feature_train, label_train))
print("Test Score : ", gbm.score(feature_test, label_test))
print("Accuracy : ", accuracy_score(label_test, predicted_labels))
# confusion matrix for Gradient Boost classifier
from sklearn.metrics import confusion_matrix
cnf_matrix = confusion_matrix(label_test, predicted_labels)
sns.heatmap(pd.DataFrame(cnf_matrix), fmt= "d", annot = True)
plt.xlabel("Predicted Values")
plt.ylabel("Actual Values")
# roc auc curve for gradient boost classifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
pred_prob = gbm.predict_proba(feature_test)[:,1]
fpr,tpr,thresholds = roc_curve(label_test, pred_prob)
print("ROC-AUC Score : ", roc_auc_score(label_test, pred_prob))
plt.plot(fpr,tpr)
plt.plot([0,1],[0,1])
plt.xlabel("fpr")
plt.ylabel("tpr")