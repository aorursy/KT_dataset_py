# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import lightgbm as lgbm

# preprocessing
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression,RidgeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier  
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import recall_score,roc_auc_score,accuracy_score,roc_curve
from sklearn import metrics
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
#Data Ingestion 
data_train=pd.read_csv('../input/aps-failure-at-scania-trucks-data-set/aps_failure_training_set.csv',error_bad_lines=False) 
data_test=pd.read_csv('../input/aps-failure-at-scania-trucks-data-set/aps_failure_test_set.csv',error_bad_lines=False)
data_train.head()
data_test.head()
print("Number of positive classes = ", sum(data_train['class'] == 'pos'))
print("Number of negative classes = ", sum(data_train['class'] == 'neg'))
data_train = data_train.rename(columns = {'class' : 'Flag'})
data_train['Flag'] = data_train.Flag.map({'neg':0, 'pos':1})
data_train = data_train.replace(['na'],np.nan)
data_train.head()
data_test = data_test.rename(columns = {'class' : 'Flag'})
data_test['Flag'] = data_test.Flag.map({'neg':0, 'pos':1})
data_test = data_test.replace(['na'],np.nan)
data_test.head()
missing_percent_threshold = 0.50
total_num_data = len(data_train.index)
missing_data_count = pd.DataFrame(data_train.isnull().sum().sort_values(ascending=False), columns=['Number'])
missing_data_percent = pd.DataFrame(data_train.isnull().sum().sort_values(ascending=False)/total_num_data, columns=['Percent'])
missing_data = pd.concat([missing_data_count, missing_data_percent], axis=1)
missing_data
missing_data_percent.plot.bar(figsize=(50,10))
plt.show()
missing_column_headers = missing_data[missing_data['Percent'] > missing_percent_threshold].index
print(missing_column_headers)  #are the missing data header with more than 50%
data_train = data_train.drop(columns=missing_column_headers)
print("Training data-set shape after dropping features is ", data_train.shape)
data_test = data_test.drop(columns=missing_column_headers)
print("Test data-set shape after dropping features is ", data_test.shape)
print(data_train.describe())
y_train = data_train.loc[:, 'Flag']
x_train = data_train.drop('Flag', axis=1)
y_test = data_test.loc[:, 'Flag']
x_test = data_test.drop('Flag', axis=1)
impute_median = SimpleImputer(strategy='median')
impute_median.fit(x_train.values)
x_train = impute_median.transform(x_train.values)
x_test = impute_median.transform(x_test.values)
scaler = StandardScaler()
scaler.fit(x_train)
x_train_scaled = scaler.transform(x_train)
scaler.fit(x_test)
x_test_scaled = scaler.transform(x_test)
x_train_scaled_head=x_train_scaled[0:1000,3]
x_test_scaled_head=x_test_scaled[0:1000,3]
x_train_head=x_train[0:1000,3]
x_test_head=x_test[0:1000,3]

fig = plt.figure(figsize = (8, 8))
fig.add_subplot(1,2,1)
plt.plot(x_train_head,label='train')
plt.plot(x_test_head,label='test')
plt.ylabel('Original unit')
fig.add_subplot(1,2,2)
plt.plot(x_train_scaled_head,label='scaled_train')
plt.plot(x_test_scaled_head,label='scaled_test')
plt.ylabel('Scaled unit')
plt.show()
Count = pd.value_counts(y_train, sort = True).sort_index()
Count.plot(kind = 'bar')
plt.title("Class count")
plt.xlabel("Flag")
plt.ylabel("Frequency")
sm = SMOTE()
x_train_new, y_train_new = sm.fit_sample(x_train, y_train)
x_train_scaled_new, y_train_scaled_new = sm.fit_sample(x_train_scaled, y_train)
pca = PCA().fit(x_train_scaled_new)
plt.rcParams["figure.figsize"] = (12,6)
fig, ax = plt.subplots()
xi = np.arange(1, 163, step=1)
y = np.cumsum(pca.explained_variance_ratio_)
plt.ylim(0.0,1.1)
plt.plot(xi, y, marker='o', linestyle='--', color='b')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative variance (%)')
plt.title('The number of components needed to explain variance')
plt.axhline(y=0.70, color='r', linestyle='-')
plt.text(0.5, 0.73, '70% cut-off threshold', color = 'red', fontsize=16)
plt.axhline(y=0.90, color='r', linestyle='-')
plt.text(0.5, 0.85, '90% cut-off threshold', color = 'red', fontsize=16)

ax.grid(axis='x')
plt.show()
n_comp=[0.70,0.75,0.80,0.90]
pca = PCA(n_components=n_comp[0])
pca.fit(x_train_scaled_new)
x_train_new_0 = pca.transform(x_train_scaled_new)
x_test_0 = pca.transform(x_test_scaled)
print("Number of features after PCA = ", x_test_0.shape[1])
corrmat_pca = pd.DataFrame(x_train_new_0).corr()
sn.heatmap(corrmat_pca, vmax=.8, square=True);
plt.show()
x_train_final_70 = x_train_new_0
y_train_final_70 = y_train_scaled_new
x_test_final_70= x_test_0
y_test_final_70 = y_test
pca = PCA(n_components=n_comp[1])
pca.fit(x_train_scaled_new)
x_train_new_1 = pca.transform(x_train_scaled_new)
x_test_1 = pca.transform(x_test_scaled)
print("Number of features after PCA = ", x_test_1.shape[1])
corrmat_pca = pd.DataFrame(x_train_new_1).corr()
sn.heatmap(corrmat_pca, vmax=.8, square=True);
plt.show()
x_train_final_75 = x_train_new_1
y_train_final_75 = y_train_scaled_new
x_test_final_75 = x_test_1
y_test_final_75 = y_test
pca = PCA(n_components=n_comp[2])
pca.fit(x_train_scaled_new)
x_train_new_2 = pca.transform(x_train_scaled_new)
x_test_2 = pca.transform(x_test_scaled)
print("Number of features after PCA = ", x_test_2.shape[1])
corrmat_pca = pd.DataFrame(x_train_new_1).corr()
sn.heatmap(corrmat_pca, vmax=.8, square=True);
plt.show()
x_train_final_80 = x_train_new_2
y_train_final_80 = y_train_scaled_new
x_test_final_80 = x_test_2
y_test_final_80 = y_test
pca = PCA(n_components=n_comp[3])
pca.fit(x_train_scaled_new)
x_train_new_3 = pca.transform(x_train_scaled_new)
x_test_3 = pca.transform(x_test_scaled)
print("Number of features after PCA = ", x_test_3.shape[1])
corrmat_pca = pd.DataFrame(x_train_new_1).corr()
sn.heatmap(corrmat_pca, vmax=.8, square=True);
plt.show()
x_train_final_90 = x_train_new_3
y_train_final_90 = y_train_scaled_new
x_test_final_90 = x_test_3
y_test_final_90 = y_test
def confusionmatrix(y_test,y_predict,x='name of model'):
    cm=metrics.confusion_matrix(y_test,y_predict)
    # recall=print(round(recall_score(y_test, y_predict, average='macro')*100,2))
    plt.figure(figsize=(10,7))
    sn.heatmap(cm,annot=True,cbar=False, fmt='g')
    cm1 = pd.DataFrame(cm.reshape((1,4)), columns=['TN', 'FP', 'FN', 'TP'])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(x)
    TC= 10*cm1.FP + 500*cm1.FN   
    return [ plt.show(),print(cm1),print(TC)]
from IPython.display import Image
Image("../input/image-req/Capture.JPG")
c_parameter_range = [0.0001,0.001,0.01,0.1,1,10,100]

logistic_acc_table_70 = pd.DataFrame(columns = ['C_parameter','Recall'])
logistic_acc_table_70['C_parameter'] = c_parameter_range

    
p=0

for c_param in c_parameter_range:
    lr = LogisticRegression(C = c_param)
    lr.fit(x_train_final_70, y_train_final_70.values.ravel())
    y_pred = lr.predict(x_test_final_70)
    logistic_acc_table_70.iloc[p,1] = recall_score(y_test_final_70,y_pred)
    p+=1
    

x=logistic_acc_table_70.isin([max(logistic_acc_table_70.Recall)])
seriesObj = x.any()
columnNames = list(seriesObj[seriesObj == True].index)
for col in columnNames:
        rows = list(x[col][x[col] == True].index)
index = np.array(rows)
max_c_param=logistic_acc_table_70.C_parameter[index[0]]
print(max(logistic_acc_table_70.Recall))  

logreg = LogisticRegression(C=max_c_param)
logreg.fit(x_train_final_70, y_train_final_70)
# visualization using confusion matrix
confusionmatrix(y_test_final_70,logreg.predict(x_test_final_70),x='Logistic regression_70')
logistic_acc_table_75 = pd.DataFrame(columns = ['C_parameter','Recall'])
logistic_acc_table_75['C_parameter'] = c_parameter_range


    
j=0

for c_param in c_parameter_range:
    lr = LogisticRegression(C = c_param)
    lr.fit(x_train_final_75, y_train_final_75.values.ravel())
    y_pred = lr.predict(x_test_final_75)
    logistic_acc_table_75.iloc[j,1] = recall_score(y_test_final_75,y_pred)
    j+=1


x=logistic_acc_table_75.isin([max(logistic_acc_table_75.Recall)])
seriesObj = x.any()
columnNames = list(seriesObj[seriesObj == True].index)
for col in columnNames:
        rows = list(x[col][x[col] == True].index)
index = np.array(rows)
max_c_param=logistic_acc_table_75.C_parameter[index[0]]


print(max(logistic_acc_table_75.Recall)) 
logreg = LogisticRegression(C=max_c_param)
logreg.fit(x_train_final_75, y_train_final_75)
# visualization using confusion matrix
confusionmatrix(y_test_final_75,logreg.predict(x_test_final_75),x='Logistic regression_75')
logistic_acc_table_80 = pd.DataFrame(columns = ['C_parameter','Recall'])
logistic_acc_table_80['C_parameter'] = c_parameter_range

    
k=0

for c_param in c_parameter_range:
    lr = LogisticRegression(C = c_param)
    lr.fit(x_train_final_80, y_train_final_80.values.ravel())
    y_pred = lr.predict(x_test_final_80)
    logistic_acc_table_80.iloc[k,1] = recall_score(y_test_final_80,y_pred)
    k+=1
x=logistic_acc_table_80.isin([max(logistic_acc_table_80.Recall)])
seriesObj = x.any()
columnNames = list(seriesObj[seriesObj == True].index)
for col in columnNames:
        rows = list(x[col][x[col] == True].index)
index = np.array(rows)
max_c_param=logistic_acc_table_80.C_parameter[index[0]]



print(max(logistic_acc_table_80.Recall)) 
logreg = LogisticRegression(C=max_c_param)
logreg.fit(x_train_final_80, y_train_final_80)
# visualization using confusion matrix
confusionmatrix(y_test_final_80,logreg.predict(x_test_final_80),x='Logistic regression_80')
logistic_acc_table_90 = pd.DataFrame(columns = ['C_parameter','Recall'])
logistic_acc_table_90['C_parameter'] = c_parameter_range

    
m=0

for c_param in c_parameter_range:
    lr = LogisticRegression(C = c_param)
    lr.fit(x_train_final_90, y_train_final_90.values.ravel())
    y_pred = lr.predict(x_test_final_90)
    logistic_acc_table_90.iloc[m,1] = recall_score(y_test_final_90,y_pred)
    m+=1
x=logistic_acc_table_90.isin([max(logistic_acc_table_90.Recall)])
seriesObj = x.any()
columnNames = list(seriesObj[seriesObj == True].index)
for col in columnNames:
        rows = list(x[col][x[col] == True].index)
index = np.array(rows)
max_c_param=logistic_acc_table_90.C_parameter[index[0]]

print(max(logistic_acc_table_90.Recall))  
logreg = LogisticRegression(C=max_c_param)
logreg.fit(x_train_final_90, y_train_final_90)
# visualization using confusion matrix
confusionmatrix(y_test_final_90,logreg.predict(x_test_final_90),x='Logistic regression_90')
clf = SVC(C = 0.03593813663804628,kernel = 'rbf',gamma = 0.1778279410038923)
clf.fit(x_train_final_75, y_train_final_75)
y_pred = clf.predict(x_test_final_75)
Recall = recall_score(y_test_final_75,y_pred)
confusionmatrix(y_test_final_75,clf.predict(x_test_final_75),x='SUPPORT VECTOR MACHINE')
def confusionmatrix_knn(y_test,y_predict):
    cm=metrics.confusion_matrix(y_test,y_predict)
    cm1 = pd.DataFrame(cm.reshape((1,4)), columns=['TN', 'FP', 'FN', 'TP'])
    TC= 10*cm1.FP + 500*cm1.FN   
    return [TC]
total_cost_knn= np.empty((10, 1))

for i in range(0,10):
      knn =KNeighborsClassifier(n_neighbors=i+1)
      knn.fit(x_train_final_75, y_train_final_75)
      total_cost_knn[i,:]=confusionmatrix_knn(y_test_final_75,knn.predict(x_test_final_75))
      
      
plt.figure()        
l = range(1,11)
for j in range(len(l)):     
    plt.plot( l, total_cost_knn)
    plt.xlabel('Values of n_neighbors')
    plt.ylabel('Total cost')
    plt.title('Variation of total cost with different n values in knn method')
minElement = np.amin(total_cost_knn)    
result = np.where(total_cost_knn == np.amin(total_cost_knn))
min_cost_index=result[0]+1
#visualization using confusion matrix for maximum accuracy as it comes at n_neighbors=9
knn =KNeighborsClassifier(n_neighbors=min_cost_index[0])
knn.fit(x_train_final_75, y_train_final_75)
confusionmatrix(y_test_final_75,knn.predict(x_test_final_75),x='KNN CLASSIFIER')
def confusionmatrix_RANDOM(y_test,y_predict):
    cm=metrics.confusion_matrix(y_test,y_predict)
    cm1 = pd.DataFrame(cm.reshape((1,4)), columns=['TN', 'FP', 'FN', 'TP'])
    TC= 10*cm1.FP + 500*cm1.FN   
    return [TC,print(cm1),print(TC)]
total_cost_RANDOM= []



for m in range(100,200):
    random_forest=RandomForestClassifier(n_estimators= m,random_state=1)
    random_forest.fit(x_train_final_75, y_train_final_75)
    total_cost=confusionmatrix_RANDOM(y_test_final_75,random_forest.predict(x_test_final_75))
    total_cost_RANDOM.append(total_cost)


plt.figure()        
l = range(100,200)
for j in range(len(l)):     
    plt.plot( l, total_cost_RANDOM)
    plt.xlabel('Values of n_estimators')
    plt.ylabel('Total cost')
    plt.title('Variation of total cost with different n values in random forest')
def Extract(lst): 
    return [item[0] for item in lst]

total_cost_min=Extract(total_cost_RANDOM)

new_list = []
for item in total_cost_min:
    new_list.append(float(item))
total_cost_RANDOM=np.asarray(new_list)
total_cost_RANDOM = pd.DataFrame(total_cost_RANDOM)
total_cost_RANDOM.insert(0, "index", range(100,200), True) 

x=total_cost_RANDOM.isin([min(total_cost_RANDOM[0])])
seriesObj = x.any()
columnNames = list(seriesObj[seriesObj == True].index)
for col in columnNames:
        rows = list(x[col][x[col] == True].index)
index = np.array(rows)
index_value=total_cost_RANDOM.index[index[0]]
index_value_random=int(total_cost_RANDOM.loc[index_value]['index'])
#visualization using confusion matrix for maximum accuracy as it comes at n=80
random_forest=RandomForestClassifier(n_estimators= index_value_random,random_state=1)
random_forest.fit(x_train_final_75, y_train_final_75)
confusionmatrix(y_test_final_75,random_forest.predict(x_test_final_75),x='Random Forest')
