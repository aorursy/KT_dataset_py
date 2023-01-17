# Import libraries necessary for this project
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

%matplotlib inline


# Load the provided data set for Credit Card dataset
data = pd.read_csv("/kaggle/input/creditcardfraud/creditcard.csv")

# Lets see the top 5 data set ,if its been loaded correctyly and or not
data.head(5)
data.describe()
data.shape
data_count_class=data
class_count = pd.value_counts(data['Class'], sort = True).sort_index()
sns.countplot(x="Class", data=data_count_class)
plt.title("Class Count")
plt.xlabel("Class")
plt.ylabel("Frequency")
fraud_data = data[data_count_class.Class == 1]
normal_data = data[data_count_class.Class == 0]
fraud_data.shape
normal_data.shape
sns.distplot(normal_data.Time, color='b')
plt.title("Time feature distribution of Non-Fraud Transaction")
plt.xlabel("Time")
plt.ylabel("Frequency")
sns.distplot(fraud_data.Time, color='g')
plt.title("Time feature distribution over all the Fraud Transaction")
plt.xlabel("Time")
plt.ylabel("Frequency")
sns.pairplot(fraud_data)
sns.distplot(normal_data.Amount, color='g')
plt.title("Distribution of Amount for Reguler Transaction")
plt.xlabel("Amount")
plt.ylabel("Frequency")
sns.distplot(fraud_data.Amount, color='r')
plt.title("Amount feature distribution of Fraud Data")
plt.xlabel("Amount")
plt.ylabel("Dist Frequency")
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
f.suptitle('Amount of transaction')
bins = 50

ax1.hist(fraud_data.Amount, bins = bins)
ax1.set_title('Fraud Data')

ax2.hist(normal_data.Amount, bins = bins)
ax2.set_title('Non Fraud Data')

plt.xlabel('Amount')
plt.ylabel('Number of Transactions')
plt.xlim((0, 21000))
plt.yscale('log')
plt.show()
f, (ax, ax_1) = plt.subplots(2, 1, sharex=True)
f.suptitle('Time for transaction vs Amount')

ax.scatter(fraud_data.Time, fraud_data.Amount)
ax.set_title('Fraud Data')
ax.grid(color='r', linestyle='-', linewidth=0.1)


ax_1.scatter(normal_data.Time, normal_data.Amount)
ax_1.set_title('Non Fraud Data')
ax_1.grid(color='r', linestyle='-', linewidth=0.1)
plt.xlabel('Time in Seconds')
plt.ylabel('Amount')
plt.show()
from sklearn.preprocessing import StandardScaler
model_data = data.drop(['Time'], axis=1)
model_data['Amount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))
model_data.head()
model_train = model_data.drop("Class", 1).values
model_train.shape
model_test = model_data["Class"].values
model_test.shape
from imblearn.over_sampling import SMOTE
sampling_train=model_train
sampling_test=model_test
sampler = SMOTE(random_state = 0, n_jobs = -1)
model_train_lr , model_test_lr = sampler.fit_sample(sampling_train, sampling_test)
model_train_lr.shape
model_test_lr.shape
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(model_train_lr, model_test_lr, test_size = 0.25, random_state = 0)


X_train.shape
X_test.shape
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(max_iter=300, random_state=0, n_jobs = -1)
lr.fit(X_train, Y_train)
lr_prediction = lr.predict(X_test)
print(lr_prediction)
lr_prediction.shape
#import sns and matplotlib for the graph & data visualization 
from matplotlib import pyplot
import seaborn as sns

# import all the accuracy parameters : accuracy_score, recall_score, confusion_matrix, roc_auc_score
from sklearn.metrics import accuracy_score,confusion_matrix,roc_auc_score,recall_score



sc_lr_accuracy = accuracy_score(Y_test, lr_prediction)
sc_lr_recall = recall_score(Y_test, lr_prediction)
sc_lr_cm = confusion_matrix(Y_test, lr_prediction)
sc_lr_auc = roc_auc_score(Y_test, lr_prediction)

print("Model has a Score_Accuracy: {:.3%}".format(sc_lr_accuracy))
print("Model has a Score_Recall: {:.3%}".format(sc_lr_recall))
print("Model has a Score ROC AUC: {:.3%}".format(sc_lr_auc))

sc_lr_cm = pd.DataFrame(sc_lr_cm, ['True Regular','True Fraud'],['Prediction Regular','Prediction Fraud'])
pyplot.figure(figsize = (8,4))
sns.set(font_scale=1.4)
sns.heatmap(sc_lr_cm, annot=True,annot_kws={"size": 16},fmt='g')
print(sc_lr_cm)
#importing sampling SMOTETomek
from imblearn.combine import SMOTETomek


# Sample the data againg with SMOTETomek
sampling_train=model_train
sampling_test=model_test
SMOTEtomek_sampler = SMOTETomek(random_state = 0, n_jobs = -1)
model_input_rf , model_output_rf = SMOTEtomek_sampler.fit_sample(sampling_train, sampling_test)
model_input_rf.shape
from sklearn.model_selection import StratifiedShuffleSplit

shuffle_splits = StratifiedShuffleSplit(n_splits=20, test_size=0.25, random_state=0)
for train, test in shuffle_splits.split(model_input_rf, model_output_rf):
    X_train, X_test = model_input_rf[train], model_input_rf[test]
    Y_train, Y_test = model_output_rf[train], model_output_rf[test]
X_train.shape
Y_train.shape
from sklearn.ensemble import RandomForestClassifier


RandomForest_model = RandomForestClassifier(n_estimators= 200, criterion = 'entropy', random_state = 0, n_jobs = -1)
RandomForest_model.fit(X_train, Y_train)
RandomForest_predict = RandomForest_model.predict(X_test)
print(RandomForest_predict)
sc_rf_accuracy = accuracy_score(Y_test, RandomForest_predict)
sc_rf_recall = recall_score(Y_test, RandomForest_predict)
sc_rf_cm = confusion_matrix(Y_test, RandomForest_predict)
sc_rf_auc = roc_auc_score(Y_test, RandomForest_predict)

print("Model has a Score Accuracy: {:.3%}".format(sc_rf_accuracy))
print("Model has a Score Recall: {:.3%}".format(sc_rf_recall))
print("Model has a Score ROC AUC: {:.3%}".format(sc_rf_auc))
sc_rf_cm = pd.DataFrame(sc_rf_cm, ['True Regular','True Fraud'],['Prediction Regular','Prediction Fraud'])
pyplot.figure(figsize = (8,4))
sns.set(font_scale=1.4)
sns.heatmap(sc_rf_cm, annot=True,annot_kws={"size": 16},fmt='g')
from imblearn.under_sampling import TomekLinks

TomekLinks_sampler = TomekLinks()
model_input_km , model_output_km = SMOTEtomek_sampler.fit_sample(model_train, model_test)
from sklearn.cluster import KMeans
inertia = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, max_iter = 200, random_state = 0, n_jobs = -1)
    kmeans.fit(model_input_km)
    inertia.append(kmeans.inertia_)
plt.plot(range(1, 11), inertia)
plt.title('Elbow score vs No. of clusters')
plt.xlabel('No. of clusters')
plt.ylabel('Score')
plt.show()
kmeans = KMeans(n_clusters = 2, max_iter = 200, random_state = 0, n_jobs = -1).fit(model_input_km)
k_centers = kmeans.cluster_centers_
from scipy import spatial
kmean_distance = pd.DataFrame(spatial.distance.cdist(model_input_km, k_centers, 'euclidean'))
kmean_distance['distance_mean'] = kmean_distance.apply(np.mean, axis=1)
kmean_distance.head()
cut_off = np.percentile(kmean_distance['distance_mean'], 95)
model_predict_km = np.where(kmean_distance['distance_mean'] >= cut_off, 1, 0)
sc_km_accuracy = accuracy_score(model_output_km, model_predict_km)
sc_km_recall = recall_score(model_output_km, model_predict_km)
sc_km_cm = confusion_matrix(model_output_km, model_predict_km)
sc_km_auc = roc_auc_score(model_output_km, model_predict_km)

print("Model has a score Accuracy: {:.3%}".format(sc_km_accuracy))
print("Model has a score Recall: {:.3%}".format(sc_km_recall))
print("Model has a score ROC AUC: {:.3%}".format(sc_km_auc))
sc_km_cm = pd.DataFrame(sc_km_cm, ['True Regular','True Fraud'],['Prediction Regular','Prediction Fraud'])
pyplot.figure(figsize = (8,4))
sns.set(font_scale=1.4)
sns.heatmap(sc_km_cm, annot=True,annot_kws={"size": 20},fmt='g')
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

X_train_knn, X_test_knn, Y_train_knn, Y_test_knn = train_test_split(model_train_lr[0:100000], model_test_lr[0:100000], test_size = 0.35, random_state = 0)
# Lets take few no between 0 to 6 for K values
no = list(range(0,6))

KNeighbors = list(filter(lambda x: x%2!=0, no))

CV_Sc = []

for k in KNeighbors:
    KNN = KNeighborsClassifier(n_neighbors = k, algorithm = 'kd_tree')
    recall_scores = cross_val_score(KNN, X_train_knn, Y_train_knn, cv = 5, scoring='recall')
    CV_Sc.append(recall_scores.mean())
import matplotlib.pyplot as plt
plt.figure(figsize = (14, 12))
plt.plot(KNeighbors, CV_Sc)
plt.title("Neighbors Vs Recall Score for best KNeighbor Value", fontsize=25)
plt.xlabel("Number of Neighbors", fontsize=25)
plt.ylabel("Recall Score", fontsize=25)
plt.grid(linestyle='-', linewidth=0.5)
best_k = KNeighbors[CV_Sc.index(max(CV_Sc))]
#we choose k on high CV_Score
print("Best value of K= "+str(best_k)+" ")


from sklearn.metrics import recall_score

KNN_best_model = KNeighborsClassifier(n_neighbors = best_k, algorithm = 'kd_tree')
KNN_best_model.fit(X_train_knn, Y_train_knn)
KNN_prediction = KNN_best_model.predict(X_test_knn)

recallTest = recall_score(Y_test_knn, KNN_prediction)

print("Recall Score of the knn classifier for best k values of "+str(best_k)+" is: "+str(recallTest))

cm = confusion_matrix(Y_test_knn, KNN_prediction)

print(cm)
tn, fp, fn, tp = cm.ravel()
(tn, fp, fn, tp)
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
%matplotlib inline
X_train_xgb,X_test_xgb,y_train_xgb,y_test_xgb =train_test_split(model_train_lr, model_test_lr, stratify=model_test_lr, random_state=42)
#Lets take diffetrnt no of Tree ranging 2 to 30 in interval of 5
tree_range = range(2, 100, 10)

score1=[]
score2=[]
for tree in tree_range:
    xgb=XGBClassifier(n_estimators=tree)
    xgb.fit(X_train_xgb,y_train_xgb)
    score1.append(xgb.score(X_train_xgb,y_train_xgb))
    score2.append(xgb.score(X_test_xgb,y_test_xgb))
    
%matplotlib inline
plt.plot(tree_range,score1,label= 'Accuracy : Training set')
plt.plot(tree_range,score2,label= 'Accuracy : Testing set')
plt.xlabel('Value of number of trees in XGboost')
plt.ylabel('Accuracy Score')
plt.legend()
xgb=XGBClassifier(n_estimators=40)
xgb.fit(X_train_xgb,y_train_xgb)
print('Accuracy of XGB n=6 on the testing dataset is :{:.3f}'.format(xgb.score(X_test_xgb,y_test_xgb)))
xgb_predict = xgb.predict(X_test_xgb)
# import all the accuracy parameters : accuracy_score, recall_score, confusion_matrix, roc_auc_score
from sklearn.metrics import accuracy_score,confusion_matrix,roc_auc_score,recall_score



sc_xgb_accuracy = accuracy_score(y_test_xgb, xgb_predict)
sc_xgb_recall = recall_score(y_test_xgb, xgb_predict)
sc_xgb_cm = confusion_matrix(y_test_xgb, xgb_predict)
sc_xgb_auc = roc_auc_score(y_test_xgb, xgb_predict)

print("Model has a Score Accuracy: {:.3%}".format(sc_xgb_accuracy))
print("Model has a Score Recall: {:.3%}".format(sc_xgb_recall))
print("Model has a Score ROC AUC: {:.3%}".format(sc_xgb_auc))
Prediction_Accuracy={
    'Logistic Regression': sc_lr_accuracy,
    'Random Forest': sc_rf_accuracy,
    'K-Means': sc_km_accuracy,
    'XGBoost': sc_xgb_accuracy
}

Prediction_Recall={
    'Logistic Regression': sc_lr_recall,
    'Random Forest': sc_rf_recall,
    'K-Means': sc_km_recall,
    'KNN':recallTest,
    'XGBoost': sc_xgb_recall
}

Prediction_AUC={
    'Logistic Regression': sc_lr_auc,
    'Random Forest': sc_rf_auc,
    'K-Means': sc_km_auc,
    'XGBoost': sc_xgb_auc
}
#set colors
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'g']


pyplot.title('Accuracy of different models')
pyplot.barh(range(len(Prediction_Accuracy)), list(Prediction_Accuracy.values()), align='center',color=colors)
pyplot.yticks(range(len(Prediction_Accuracy)), list(Prediction_Accuracy.keys()))
pyplot.xlabel('Accuracy_Score')
pyplot.title('Recall Score of different models')
pyplot.barh(range(len(Prediction_Recall)), list(Prediction_Recall.values()), align='center',color=colors)
pyplot.yticks(range(len(Prediction_Recall)), list(Prediction_Recall.keys()))
pyplot.xlabel('Recall Score of different models')
pyplot.title('AUC Score of different models')
pyplot.barh(range(len(Prediction_AUC)), list(Prediction_AUC.values()), align='center',color=colors)
pyplot.yticks(range(len(Prediction_AUC)), list(Prediction_AUC.keys()))
pyplot.xlabel('AUC Score of different models')
RandomForest_predict_check = RandomForest_model.predict(model_train_lr)
sc_rf_accuracy_ck = accuracy_score(model_test_lr, RandomForest_predict_check)
sc_rf_recall_ck = recall_score(model_test_lr, RandomForest_predict_check)
sc_rf_cm_ck = confusion_matrix(model_test_lr, RandomForest_predict_check)
sc_rf_auc_ck = roc_auc_score(model_test_lr, RandomForest_predict_check)

print("Model has a Score Accuracy: {:.3%}".format(sc_rf_accuracy_ck))
print("Model has a Score Recall: {:.3%}".format(sc_rf_recall_ck))
print("Model has a Score ROC AUC: {:.3%}".format(sc_rf_auc_ck))
print(sc_rf_cm_ck)