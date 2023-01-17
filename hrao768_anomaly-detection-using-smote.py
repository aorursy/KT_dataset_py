import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split,cross_val_predict,cross_val_score, GridSearchCV,RandomizedSearchCV
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.metrics import confusion_matrix,classification_report,f1_score,recall_score,precision_score,accuracy_score,precision_recall_curve,roc_curve,roc_auc_score

from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.cluster import KMeans
from imblearn.over_sampling import SMOTE

from collections import Counter

plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

import warnings
warnings.filterwarnings('ignore')

import random
random.seed(0)
def data_preparation(data):
    features = data.iloc[:,0:-1]
    label = data.iloc[:,-1]
    x_train,x_test,y_train,y_test = train_test_split(features,label,test_size=0.2,random_state=0)

    #Standarad scaler is not applied since all the features are outcomes of PCA and are already standardized.
    #sc = StandardScaler()
    #x_train = sc.fit_transform(x_train)
    #x_test = sc.transform(x_test)
    
    print("Length of training data",len(x_train))
    print("Length of test data",len(x_test))
    return x_train,x_test,y_train,y_test
    
def build_model_train_test(model,x_train,x_test,y_train,y_test):
    model.fit(x_train,y_train)

    y_pred = model.predict(x_train)
    
    print("\n----------Accuracy Scores on Train data------------------------------------")
    print("F1 Score: ", f1_score(y_train,y_pred))
    print("Precision Score: ", precision_score(y_train,y_pred))
    print("Recall Score: ", recall_score(y_train,y_pred))


    print("\n----------Accuracy Scores on Test data------------------------------------")
    y_pred_test = model.predict(x_test)
    
    print("F1 Score: ", f1_score(y_test,y_pred_test))
    print("Precision Score: ", precision_score(y_test,y_pred_test))
    print("Recall Score: ", recall_score(y_test,y_pred_test))

    #Confusion Matrix
    plt.figure(figsize=(18,6))
    gs = gridspec.GridSpec(1,2)

    ax1 = plt.subplot(gs[0])
    cnf_matrix = confusion_matrix(y_train,y_pred)
    row_sum = cnf_matrix.sum(axis=1,keepdims=True)
    cnf_matrix_norm =cnf_matrix / row_sum
    sns.heatmap(cnf_matrix_norm,cmap='YlGnBu',annot=True)
    plt.title("Normalized Confusion Matrix - Train Data")

    ax2 = plt.subplot(gs[1])
    cnf_matrix = confusion_matrix(y_test,y_pred_test)
    row_sum = cnf_matrix.sum(axis=1,keepdims=True)
    cnf_matrix_norm =cnf_matrix / row_sum
    sns.heatmap(cnf_matrix_norm,cmap='YlGnBu',annot=True)
    plt.title("Normalized Confusion Matrix - Test Data")

#Loading Dataset
cc_dataset = pd.read_csv("../input/creditcard.csv")
cc_dataset.shape
cc_dataset.head()
cc_dataset.describe()
#Code for checking if any feature has null values. Here the output confirms that there are no null values in this data set.
cc_dataset.isnull().any()
#Counts for each class in the dataset. As you can see, we have only 492 (0.17%) fraud cases out of 284807 records. Remaining 284315 (99.8%) of the records belong to genuine cases.
#So the dataset is clearly imbalanced!
cc_dataset['Class'].value_counts()
#Data Visualization for checking the distribution for Genuine cases & Fraud cases for each feature
v_features = cc_dataset.columns
plt.figure(figsize=(12,31*4))
gs = gridspec.GridSpec(31,1)

for i, col in enumerate(v_features):
    ax = plt.subplot(gs[i])
    sns.distplot(cc_dataset[col][cc_dataset['Class']==0],color='g',label='Genuine Class')
    sns.distplot(cc_dataset[col][cc_dataset['Class']==1],color='r',label='Fraud Class')
    ax.legend()
plt.show()
cc_dataset.drop(labels = ['Time'], axis = 1, inplace=True)
cc_dataset['Amount'] = StandardScaler().fit_transform(cc_dataset['Amount'].reshape(-1,1))
#Data Preparation
x_train,x_test,y_train,y_test = data_preparation(cc_dataset)
os = SMOTE(random_state=0)
#Generate the oversample data
os_res_x,os_res_y=os.fit_sample(x_train,y_train)
#Counts of each class in oversampled data
print(sorted(Counter(os_res_y).items()))
#RandomForest for training over-sampled data set. 
rnd_clf = RandomForestClassifier(n_estimators=100,criterion='gini',n_jobs=-1, random_state=0)
#Train the model on oversampled data and check the performance on original test data
build_model_train_test(rnd_clf,os_res_x,x_test,os_res_y,y_test)
# #Elbow Curve for identifying the best number of clusters
# wcss = [] # Within Cluster Sum of Squares
# for k in range(1, 21):
#     kmeans = KMeans(n_clusters = k, init = 'k-means++', random_state = 0)
#     kmeans.fit(x_train)
#     wcss.append(kmeans.inertia_)
# plt.plot(range(1, 21), wcss)
# plt.title('The Elbow Method')
# plt.xlabel('Number of clusters - k')
# plt.ylabel('WCSS')
# plt.show()
#Clustering with 11 clusters. I used the elbow method to derive on number of clusters. I commented the above code to save the run time
kmeans_best = KMeans(n_clusters = 11, init = 'k-means++', random_state = 0)
train_clusters = kmeans_best.fit_predict(x_train)
#Merge clusters with other input features on Train Data
x_train2 = np.c_[(x_train,train_clusters )]
x_train2.shape
#Predict the cluster for test data & merge it with other features
test_clusters = kmeans_best.predict(x_test)
x_test2 = np.c_[(x_test,test_clusters )]
x_test2.shape
#Generate the oversample data for training purpose
os_res_x2,os_res_y2=os.fit_sample(x_train2,y_train)
#Counts of each class in oversampled data
print(sorted(Counter(os_res_y2).items()))

#RandomForest for training over-sampled data set. 
rnd_clf2 = RandomForestClassifier(n_estimators=100,criterion='gini',n_jobs=-1, random_state=0)
#Train the model on oversampled data and check the performance on actual test data
build_model_train_test(rnd_clf2,os_res_x2,x_test2,os_res_y2,y_test)

#Let us check cross validation scores on the orginal train data
cv_score = cross_val_score(rnd_clf2,x_train2,y_train,cv=5,scoring='f1')
print("Average F1 score CV",cv_score.mean())

cv_score = cross_val_score(rnd_clf2,x_train2,y_train,cv=5,scoring='recall')
print("Average Recall score CV",cv_score.mean())

