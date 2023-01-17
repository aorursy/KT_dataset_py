#import Useful Library
import pandas as pd
import numpy as np

#for making graph
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

#for warnings
import warnings
warnings.filterwarnings("ignore")
df = pd.read_csv('../input/social-network-ads/Social_Network_Ads.csv')
df.head()
df.info()
df.Purchased.value_counts()
df.Gender.value_counts()
sns.boxplot(y='Age', x='Purchased', data=df)
sns.boxplot(y='EstimatedSalary', x='Purchased', data=df)
plt.hist(x="Purchased", data=df);
plt.title('Distribution of Purchase');
plt.ylabel('Count');
plt.xlabel('Purchase');
plt.figure(figsize=(20,5))
bins_size = np.arange(15000,150000+10000,1000)
plt.hist(x="EstimatedSalary", data=df, bins= bins_size,rwidth=0.9);
plt.title('Distribution of Salary');
plt.ylabel('Count');
plt.xlabel('Salary');
df.EstimatedSalary.mean()
plt.figure(figsize=(15,5))
bins_size = np.arange(18,65,2)
plt.hist(x="Age", data=df, bins= bins_size,rwidth=0.9);
plt.title('Distribution of Age');
plt.ylabel('Count');
plt.xlabel('Age');
sns.distplot(df.Purchased);
df.describe()
df.Age.unique()
df.Age.nunique()
df.EstimatedSalary.unique()
df.EstimatedSalary.nunique()
df.Purchased.unique()
df.duplicated().sum()
plt.figure(figsize =(8,8))
ax= sns.heatmap(df.corr(),square = True, annot = True,cmap= 'Spectral' )
ax.set_ylim(4.0, 0)
ax = sns.regplot(x="EstimatedSalary", y="Purchased", data=df)
ax = sns.regplot(x="Age", y="Purchased", data=df)
col = sns.color_palette()[0]
sns.barplot(x="Purchased", y="EstimatedSalary", data=df, color=col)
sns.barplot(x="Purchased", y="EstimatedSalary",hue='Gender', data=df)
sns.pairplot(df, vars=["Age", "EstimatedSalary","Purchased"])
sns.pairplot(df, vars=["Age", "EstimatedSalary","Purchased"], hue = "Gender")
df.drop_duplicates(inplace=True)
df.drop(columns=['User ID'], inplace = True)
df.loc[((df.Age >58) & (df.Purchased==0)), 'Age'] = np.nan
df.fillna(53,inplace=True)
df.loc[(df.EstimatedSalary>120000) & (df.Purchased==0), 'EstimatedSalary'] = np.nan
df.fillna(120000,inplace=True)
from scipy import stats
z = np.abs(stats.zscore(df['EstimatedSalary']))
threshold = 3
print(np.where(z > 3))
z = np.abs(stats.zscore(df['Age']))
print(np.where(z > 3))
df.Age = df.Age.astype("int64")
df.EstimatedSalary = df.EstimatedSalary.astype("int64")
df.info()
a = df.groupby(['Gender', 'Age'])
a.first()
a = df.groupby(['Purchased','EstimatedSalary'])
a.first()
sns.scatterplot(y="EstimatedSalary", x="Purchased", data=df)
plt.figure(figsize = (15,8))
sns.scatterplot(y="EstimatedSalary", x="Age", data=df, hue = 'Purchased')
sns.catplot(y="EstimatedSalary", x="Purchased", data=df, hue = 'Gender')
sns.catplot(y="Age", x="Purchased", data=df, hue = 'Gender')
df.Gender.replace({'Male':1,
                   'Female':0}, inplace=True)
df.head()
X = df.iloc[:, [1, 2]]
y = df.iloc[:, 3]
X
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)
knn_pred = knn.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,knn_pred))
print(classification_report(y_test,knn_pred))
error_rate = []

for i in range(1,40):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))
plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
knn = KNeighborsClassifier(n_neighbors=4)

knn.fit(X_train,y_train)
knn_pred = knn.predict(X_test)

print('WITH K=4')
print('\n')
print(confusion_matrix(y_test,knn_pred))
print('\n')
print(classification_report(y_test,knn_pred))
from sklearn.metrics import accuracy_score
print ('accuracy_score : ', accuracy_score(y_test,knn_pred))
from sklearn.model_selection import cross_val_score
knn_accuracy = cross_val_score(knn,X,y, cv = 5)
knn_accuracy
knn_accuracy.mean()
from sklearn.metrics import roc_curve, auc

knn_fpr, knn_tpr, threshold = roc_curve(y_test, knn_pred)
auc_knn = auc(knn_fpr, knn_tpr)

plt.figure(figsize=(5, 5), dpi=100)
plt.plot(knn_fpr, knn_tpr, marker='.', label='Knn (auc = %0.3f)' % auc_knn)

plt.xlabel('False Positive Rate -->')
plt.ylabel('True Positive Rate -->')

plt.legend()

plt.show()
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X_train,y_train)
log_pred = log_reg.predict(X_test)
print(confusion_matrix(y_test,log_pred))
print('\n')
print(classification_report(y_test,log_pred))
log_accuracy = cross_val_score(log_reg,X,y, cv = 5)
print(log_accuracy)
print("mean value of accuracy",log_accuracy.mean())
from sklearn.svm import SVC
svc_classifier = SVC(kernel = 'rbf', random_state = 0)
svc_classifier.fit(X_train, y_train)
svc_pred = svc_classifier.predict(X_test)
print(confusion_matrix(y_test,svc_pred))
print('\n')
print(classification_report(y_test,svc_pred))
svc_accuracy = cross_val_score(svc_classifier,X,y, cv = 5)
print(svc_accuracy)
print("mean value of accuracy",svc_accuracy.mean())
from sklearn.tree import DecisionTreeClassifier
dt_classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
dt_classifier.fit(X_train, y_train)
dt_pred = dt_classifier.predict(X_test)
print(confusion_matrix(y_test,dt_pred))
print('\n')
print(classification_report(y_test,dt_pred))
dt_accuracy = cross_val_score(dt_classifier,X,y, cv = 5)
print(dt_accuracy)
print("mean value of accuracy",dt_accuracy.mean())
from sklearn.ensemble import RandomForestClassifier
rf_classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
rf_classifier.fit(X_train, y_train)
rf_pred = rf_classifier.predict(X_test)
print(confusion_matrix(y_test,rf_pred))
print('\n')
print(classification_report(y_test,rf_pred))
rf_accuracy = cross_val_score(rf_classifier,X,y, cv = 5)
print(rf_accuracy)
print("mean value of accuracy",rf_accuracy.mean())
print("For Random Forest Classifier::")
print(confusion_matrix(y_test,rf_pred))
print('\n')
print(classification_report(y_test,rf_pred))
#Applying grid search
from sklearn.model_selection import GridSearchCV
parameters = [{"C": [1, 10, 100, 1000], "kernel": ['linear']}, 
              {"C": [1, 10, 100, 1000], "kernel": ['rbf'], 'gamma': [0.5, 0.1, 0.01, 0.001]}]

#Use this list to train
grid_search = GridSearchCV(estimator = svc_classifier, param_grid = parameters, scoring = 'accuracy', cv = 10, n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)

#Use attributes of grid_search to get the results
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_

print("Best accuracy: ",best_accuracy)
print(best_parameters)
from sklearn.svm import SVC
svc_classifier = SVC(kernel = 'rbf', random_state = 0, C =10, gamma=0.1)
svc_classifier.fit(X_train, y_train)
svc_pred = svc_classifier.predict(X_test)
print("For SVM Classifier::")
print(confusion_matrix(y_test,svc_pred))
print('\n')
print(classification_report(y_test,svc_pred))
n_estimators = [100, 300, 500]
max_depth = [5, 8, 15]
min_samples_leaf = [1, 2] 

hyperF = dict(n_estimators = n_estimators, max_depth = max_depth,min_samples_leaf = min_samples_leaf)

gridF = GridSearchCV(rf_classifier, hyperF, cv = 5, verbose = 1, 
                      n_jobs = -1)
bestF = gridF.fit(X_train, y_train)
bestF.best_estimator_
bestF.best_params_
from sklearn.ensemble import RandomForestClassifier
rf_classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0, max_depth=5,min_samples_leaf=1)
rf_classifier.fit(X_train, y_train)
rf_pred = rf_classifier.predict(X_test)
print(confusion_matrix(y_test,rf_pred))
print('\n')
print(classification_report(y_test,rf_pred))
criterion = ['gini', 'entropy']
max_depth = [4,6,8,12]

parameters = dict(criterion=criterion,max_depth=max_depth)

  
clf = GridSearchCV(rf_classifier, hyperF, cv = 5, verbose = 1, n_jobs = -1)

# Fit the grid search
clf.fit(X_train, y_train)
from sklearn.tree import DecisionTreeClassifier
dt_classifier = DecisionTreeClassifier(criterion='entropy', random_state=0, max_depth=5)
dt_classifier.fit(X_train, y_train)
dt_pred = dt_classifier.predict(X_test)
print(confusion_matrix(y_test,dt_pred))
print('\n')
print(classification_report(y_test,dt_pred))
param_grid = [    
    {'solver' : ['lbfgs','newton-cg','liblinear','sag','saga'],
    'max_iter' : [100, 1000,2500, 5000]
    }
]
clf = GridSearchCV(log_reg, param_grid = param_grid, cv = 3, verbose=True, n_jobs=-1)
best_clf = clf.fit(X_train, y_train)
a = best_clf.best_estimator_
a
best_clf.best_params_
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(C=1.0, solver='lbfgs',max_iter=100 )
log_reg.fit(X_train,y_train)
log_pred = log_reg.predict(X_test)
print(confusion_matrix(y_test,log_pred))
print('\n')
print(classification_report(y_test,log_pred))
#List Hyperparameters that we want to tune.
leaf_size = list(range(1,10))
n_neighbors = list(range(1,10))
p=[1,2]

#Convert to dictionary
hyperparameters = dict(leaf_size=leaf_size, n_neighbors=n_neighbors, p=p)

#Use GridSearch
clf = GridSearchCV(knn, hyperparameters, cv=10, verbose=True, n_jobs=-1)
#Fit the model
best_model = clf.fit(X_train,y_train)
best_model.best_estimator_
best_model.best_params_
knn = KNeighborsClassifier(n_neighbors=9, p = 1, leaf_size=1)

knn.fit(X_train,y_train)
knn_pred = knn.predict(X_test)

print(confusion_matrix(y_test,knn_pred))
print('\n')
print(classification_report(y_test,knn_pred))
print(confusion_matrix(y_test,dt_pred))
print('\n')
print(classification_report(y_test,dt_pred))
df.head()
X1 = df.iloc[:,0: 3]
y1 = df.iloc[:, 3]
X1
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size = 0.30, random_state = 0)
dt_classifier = DecisionTreeClassifier(criterion='entropy', random_state=0, max_depth=5)
dt_classifier.fit(X_train, y_train)
dt_pred = dt_classifier.predict(X_test)
print(confusion_matrix(y_test,dt_pred))
print('\n')
print(classification_report(y_test,dt_pred))