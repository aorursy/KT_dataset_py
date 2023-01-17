import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder # for creating dummy variables
from sklearn.preprocessing import MinMaxScaler # for normalising data
# Reading csv file into dataframe
df = pd.read_csv("../input/uci-wisconsin-breast-cancer/BreastCancer.csv")
df.head()
# Dropping id column
df1 = df.drop(columns="id")
df1.head()
# Total missing values for each feature
df1.isnull().sum()
# summary of the DataFrame
df1.info()
#  some basic statistical details for all features
df1.describe()
# Check the number of malignant(M) and benign(B) cases
sns.countplot(x="diagnosis", data=df1)
sns.set(rc={'figure.figsize':(5,5)})
plt.subplot(2, 2, 1)
sns.boxplot(x='diagnosis', y='radius_mean', data=df1)
plt.ylabel('Radius Mean')
plt.xlabel('Diagnosis')
plt.title('Dianosis vs Radius Mean')
plt.subplot(2, 2, 2)
sns.boxplot(x='diagnosis', y='perimeter_mean', data=df1)
plt.ylabel('Perimeter Mean')
plt.xlabel('Diagnosis')
plt.title('Dianosis vs Perimeter Mean')
plt.subplot(2, 2, 3)
sns.boxplot(x='diagnosis', y='area_mean', data=df1)
plt.ylabel('Area Mean')
plt.xlabel('Diagnosis')
plt.title('Dianosis vs Area Mean')
plt.subplot(2, 2, 4)
sns.boxplot(x='diagnosis', y='smoothness_mean', data=df1)
plt.ylabel('Smoothness Mean')
plt.xlabel('Diagnosis')
plt.title('Dianosis vs Smoothness Mean')
plt.tight_layout()
plt.show()
labels = ['radius_mean', 'perimeter_mean','smoothness_mean', 'compactness_mean', 'concavity_mean',
       'texture_mean', 'symmetry_mean','diagnosis']

# let's examine how features determine prognostics
sns.pairplot(df1[labels], hue='diagnosis')
plt.show()
corr_matrix = round(df1.corr(), 2)
sns.set(rc={'figure.figsize':(15,15)})
sns.heatmap(corr_matrix, cmap='BuPu', annot_kws={'size': 8}, cbar = True, annot=True)
plt.title('Variable Correlation Plot')
plt.show()
# dividing the data into X and Y
X=df1.iloc[:,1:31]
X.head(2)
Y=df1.iloc[:,0:1]
Y.head(2)
le = LabelEncoder()
# converting diagnosis to dummy variables
Y['diagnosis_new'] = le.fit_transform(Y.diagnosis)
Y.head()
Y_new=Y.iloc[:,1:2]
Y_new.tail()
scaler = MinMaxScaler()
scaler.fit(X)
X1 = scaler.transform(X)
X_new=pd.DataFrame(X1, columns=X.columns)
X_new.head(2)
X_new.describe()
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict, cross_val_score # This is for cross-validation
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, balanced_accuracy_score
# search for an optimal value of K for KNN

# range of k we want to try
k_range = range(1, 31)
# empty list to store scores
k_scores = []

# 1. we will loop through reasonable values of k
for k in k_range:
    # 2. run KNeighborsClassifier with k neighbours
    knn = KNeighborsClassifier(n_neighbors=k)
    # 3. obtain cross_val_score for KNeighborsClassifier with k neighbours
    scores = cross_val_score(knn, X_new, Y_new, cv=10,  n_jobs=10)
    # 4. append mean of scores for k neighbors to k_scores list
    k_scores.append(scores.mean())

# plot the value of K for KNN (x-axis) versus the cross-validated accuracy (y-axis)
sns.set(rc={'figure.figsize':(5,5)})
plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-validated accuracy')
#finding the best k
k_df = pd.DataFrame(k_scores, index=k_range)
best_kest = int(k_df.idxmax())
best_kest
knn = KNeighborsClassifier(n_neighbors=best_kest)
svm = SVC(random_state=100, C=1.0,
    kernel='linear',
    probability=True,
    ) 
logit = LogisticRegression(penalty='l2',
    tol=0.0001,
    random_state=10)
etc = ExtraTreesClassifier(criterion='entropy',
    min_samples_split=3,
    min_samples_leaf=1,
    n_jobs=10,
    random_state=100,
    verbose=2
    )
bagging = BaggingClassifier(n_estimators=1000,
    n_jobs=10,
    random_state=100,
    verbose=0)
nb = GaussianNB()
rf = RandomForestClassifier(n_estimators=10, random_state=None)
# Fitting models that does not require scaling
models_1 = [["DecisionTreeClassifier",etc],
         ["BaggingClassifier",bagging],
         ["GaussianNB",nb],
         ["RandomForestClassifier",rf]]
m_accuracy = []
for i in models_1:
    y_predict = cross_val_predict(i[1], X, Y_new, cv=10, n_jobs=10)
    ACC = round(accuracy_score(Y_new, y_predict), 2) 
    recall = round(recall_score(Y_new, y_predict, average='weighted'), 2) 
    B_ACC = round(balanced_accuracy_score(Y_new, y_predict), 2)
    Specificiti = round(2 * B_ACC - recall, 2)
    m_accuracy.append([i[0],ACC,recall,B_ACC,Specificiti]) 
# Fitting models that require scaling
models_2 = [["LogisticRegression",logit],
         ["SupportVector Machine",svm],
         ["KNeighborsClassifier",knn]]
for i in models_2:
    y_predict = cross_val_predict(i[1], X_new, Y_new, cv=10, n_jobs=10)
    ACC = round(accuracy_score(Y_new, y_predict), 2) 
    recall = round(recall_score(Y_new, y_predict, average='weighted'), 2) 
    B_ACC = round(balanced_accuracy_score(Y_new, y_predict), 2)
    Specificiti = round(2 * B_ACC - recall, 2)
    m_accuracy.append([i[0],ACC,recall,B_ACC,Specificiti]) 
    
performace_table = pd.DataFrame(m_accuracy)
performace_table.columns = ['Model','Accuracy', 'Recall','Bal. Accuracy','Specificity']
performace_table.style.bar(subset=["Accuracy",], color='#0d8ca6')\
                 .bar(subset=["Recall"], color='#50cce6')\
                 .bar(subset=["Bal. Accuracy"], color='#17990e')\
.bar(subset=["Specificity"], color='#6ed667')
plt.figure(figsize=(10,5))
plt.barh(performace_table.Model, performace_table.Accuracy, color='#f5ec42', edgecolor='black')
plt.tight_layout()
plt.show()
# list of feature importance in desecending order
rf.fit(X, Y_new)
importance = pd.DataFrame(rf.feature_importances_, index=X_new.columns, columns=['FeatureImportance'])
importance.sort_values(by='FeatureImportance', ascending=False)
# Now, try to train again with the full data
svm.fit(X_new,Y_new)
# Python pickle module is used for serializing and de-serializing a Python object structure
import pickle
# Save the model
f1=open('breat_cancer_svm_model','wb') # wb => write binary
pickle.dump(svm, f1)
# better close (or flush) a file when done.
f1.close()
