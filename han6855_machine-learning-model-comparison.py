import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
df = pd.read_csv('../input/column_3C_weka.csv')
df.head(5)
df.info()
df.describe()
df.columns.values
df['class'].unique()
df['class'] = df['class'].map({'Hernia':1,
                              'Spondylolisthesis':2, 
                              'Normal':3})
sns.pairplot(df, hue='class')
var_names = df.columns.values[:-1]
plt.figure(figsize=(20,10))

for i in range(0,len(var_names)):    
    plt.subplot(2,3,i+1)
    sns.boxplot(x='class',y=var_names[i],data=df)
from sklearn.model_selection import train_test_split, KFold
from sklearn.cross_validation import cross_val_score, cross_val_predict, StratifiedKFold
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
x = df.drop(['class'], axis=1)
y = df['class']
x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                   test_size=0.30,random_state=101)
model_svc = SVC()
model_svc.fit(x_train, y_train)
svc_predictions = model_svc.predict(x_test)
print(confusion_matrix(y_test, svc_predictions))
print(classification_report(y_test, svc_predictions))
print(accuracy_score(y_test, svc_predictions))
param_grid = {'C':[0.1,1,10,100,1000], 'gamma':[1,0.1,0.01,0.001,0.0001], 
             'kernel':['rbf']}
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=3)
grid.fit(x_train, y_train)
grid.best_params_
grid.best_estimator_
grid_predictions = grid.predict(x_test)
print(confusion_matrix(y_test, grid_predictions))
print(classification_report(y_test, grid_predictions))
print(accuracy_score(y_test, grid_predictions))
stratif = StratifiedKFold(y,n_folds=10)
scores_svc = cross_val_score(SVC(C=10, gamma=0.0001), x, y, cv=stratif)
scores_svc
print(scores_svc.mean())
print(scores_svc.std())
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df.drop('class', axis=1))
df_feat = pd.DataFrame(scaled_features, columns=df.columns[:-1])
df_feat.head(3)
x_knn_train, x_knn_test, y_knn_train, y_knn_test = train_test_split(scaled_features, y, 
                                                                   test_size=0.30, random_state=101)
model_knn = KNeighborsClassifier(n_neighbors=1)
model_knn.fit(x_knn_train, y_knn_train)
knn_predictions = model_knn.predict(x_knn_test)
print(confusion_matrix(y_knn_test,knn_predictions))
print(classification_report(y_knn_test,knn_predictions))
error_rate = []

for i in range(1,75):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_knn_train,y_knn_train)
    pred_i = knn.predict(x_knn_test)
    error_rate.append(np.mean(pred_i != y_knn_test))
min(error_rate)
for i in range(0, len(error_rate)):
    if error_rate[i] == min(error_rate):
        print('The number of neighbors that gives the lowest error rate is:',i+1)
    else:
        i += 1
plt.figure(figsize=(10,6))
plt.plot(range(1,75),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
model_knn_refit = KNeighborsClassifier(n_neighbors=30)
model_knn_refit.fit(x_knn_train, y_knn_train)
knn_predictions_refit = model_knn_refit.predict(x_knn_test)
print(confusion_matrix(y_knn_test,knn_predictions_refit))
print(classification_report(y_knn_test,knn_predictions_refit))
print(accuracy_score(y_knn_test, knn_predictions_refit))
scores_knn = cross_val_score(KNeighborsClassifier(n_neighbors=30), scaled_features, y, cv=stratif)
print(scores_knn.mean())
print(scores_knn.std())
x_dt_train, x_dt_test, y_dt_train, y_dt_test = train_test_split(x, y, 
                                                               test_size=0.30)
model_dt = DecisionTreeClassifier()
model_dt.fit(x_dt_train, y_dt_train)
dt_predictions = model_dt.predict(x_dt_test)
print(confusion_matrix(y_dt_test, dt_predictions))
print(classification_report(y_dt_test, dt_predictions))
print(accuracy_score(y_dt_test, dt_predictions))
scores_dt = cross_val_score(DecisionTreeClassifier(), x, y, cv=stratif)
print(scores_dt.mean())
print(scores_dt.std())
x_rf_train, x_rf_test, y_rf_train, y_rf_test = train_test_split(x, y, 
                                                               test_size=0.30)
model_rf = RandomForestClassifier(n_estimators=100)
model_rf.fit(x_rf_train, y_rf_train)
rf_predictions = model_rf.predict(x_rf_test)
print(confusion_matrix(y_rf_test, rf_predictions))
print(classification_report(y_rf_test, rf_predictions))
print(accuracy_score(y_rf_test, rf_predictions))
scores_rf = cross_val_score(RandomForestClassifier(n_estimators=100), x, y, cv=stratif)
print(scores_rf.mean())
print(scores_rf.std())
list_name = ['SVM', 'KNN', 'DT', 'RF']
list_mean = [scores_svc.mean(), scores_knn.mean(), 
            scores_dt.mean(), scores_rf.mean()]
list_std = [scores_svc.std(), scores_knn.std(), 
            scores_dt.std(), scores_rf.std()]
cross_validation_results = pd.DataFrame({
    'Model':list_name,
    'Mean Scores':list_mean,
    'Standard Deviation':list_std
})
column_names = ['Model', 'Mean Scores', 'Standard Deviation']
cross_validation_results.reindex(columns=column_names)