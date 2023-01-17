!pip install -U imbalanced-learn
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from collections import Counter
from scipy.special import softmax
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv("/kaggle/input/website-phishing-data-set/Website Phishing.csv")
feats = data.columns.tolist()[:-1]
num_feats = len(feats)
y = data["Result"].tolist()
X = data.drop(columns=["Result"]).to_numpy()
classes = set(y)
print("Features: ", feats)
print("Classes: ", classes)
feats = data.columns.tolist()
num_exem = len(data[feats[0]])
num_missing_values = 0
for feat in feats:
    num_missing_values += abs(len(data[feat]) - num_exem)
    
print("Number of missing values: ",num_missing_values)  
for label in classes:
    print("Number of exemples of class {}: {}".format(label, len(data[data["Result"]==label])))
from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=0)
X_balanced, y_balanced = ros.fit_resample(X, y)
print("Nova quantidade de exemplos: ", Counter(y_balanced))
y_balanced = np.array(y_balanced)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scl_X_balanced = scaler.fit_transform(X_balanced)
scl_X = scaler.fit_transform(X_balanced)
def get_percentil(X):
    percent_array=np.zeros((num_feats, 100))

    # Calculates the percentiles for each feature
    for f in range(num_feats):
        for i in range(1,101):
            percent_array[f][i-1] = np.percentile(X[:,f], i)
 
    return percent_array

def plot_info(feats_mean, feats_std, percent_array):
    fig, axis = plt.subplots(1,2,figsize=(16,5))

    plt.xlabel("Features")

    axis[0].hist(feats_mean) 
    axis[1].hist(feats_std)
    
    axis[0].legend(["Mean"])
    axis[1].legend(["Stantard Desviation"])
   
    plt.xlabel("Percentiles")
    fold1 = percent_array[:3]
    fold2 = percent_array[3:6]
    fold3 = percent_array[6:9]
    for fold in [fold1,fold2,fold2]:
        fig1, axis1 = plt.subplots(1,3,figsize=(25,10))
        for i, percent in enumerate(fold):
            axis1[i].plot(range(100), percent)
    fig1.subplots_adjust(wspace=0.5)
feats_mean, feats_std = feats_mean = np.mean(X, axis=0), np.std(X, axis=0)
percent_array = get_percentil(X)
plot_info(feats_mean, feats_std, percent_array)
feats_mean, feats_std = feats_mean = np.mean(scl_X_balanced, axis=0), np.std(scl_X_balanced, axis=0)
percent_array = get_percentil(scl_X_balanced)
plot_info(feats_mean, feats_std, percent_array)
from sklearn.cluster import KMeans
fig = plt.figure(1, figsize=(4, 3))

kmeans = KMeans(n_clusters=3)
kmeans.fit(scl_X_balanced[:100])
y_pred = kmeans.predict(scl_X_balanced[100:200])

plt.scatter(range(len(y_pred)), y_pred)
classes = ['class '+str(c) for c in classes] # tornando as classes strings
from sklearn import svm
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, shuffle=True)
svm_clf = svm.SVC()

svm_clf.fit(X_train,y_train)
preds = svm_clf.predict(X_test)

print("Training Count: ", Counter(y_train))
print("Testing Count: ", Counter(y_test))
print(classification_report(y_test, preds, target_names=classes))
X_train, X_test, y_train, y_test = train_test_split(scl_X_balanced, y_balanced,test_size=0.2, shuffle=True)
print("Training Count: ", Counter(y_train))
print("Testing Count: ", Counter(y_test))
#Classification using SVM -> Support Vector Machine 
# spliting training and validation

svm_clf = svm.SVC()

svm_clf.fit(X_train,y_train)
preds = svm_clf.predict(X_test)

svm_result = classification_report(y_test, preds, target_names=classes, output_dict=True)
print(classification_report(y_test, preds, target_names=classes))
from sklearn.ensemble import RandomForestClassifier

clf_rf = RandomForestClassifier()
clf_rf.fit(X_train, y_train)
preds = clf_rf.predict(X_test)
rf_result = classification_report(y_test, preds, target_names=classes, output_dict=True)
print(classification_report(y_test, preds, target_names=classes))
from sklearn.ensemble import AdaBoostClassifier

clf_ada = AdaBoostClassifier(n_estimators=100)
y_pred = clf_ada.fit(X_train, y_train).predict(X_test)
ada_result = classification_report(y_test, y_pred, target_names=classes, output_dict=True)
print(classification_report(y_test, y_pred, target_names=classes))
print(rf_result.keys())
metrics = ['accuracy']

results = []


for metric in metrics:
    score = []
    for i, r in enumerate([svm_result, rf_result, ada_result]):
        print(r[metric])
        score.append(r[metric]) 
    
    results.append(score)
    
for j,result in enumerate(results): 
    fig, axis = plt.subplots(1,1,figsize=(16,5))
    fig.suptitle('Resultado de Classificação Metric: {}'.format(metrics[j]), fontsize=16)
    for i, res in enumerate(results):
        print(res)
        axis.bar([1,2,3],res,
                    tick_label=['svm_result', 'rf_result', 'ada_result'], color=['green', 'blue', 'yellow'])
# É preciso rolar o output para visualizar todas as métricas
metrics = ['f1-score', 'recall', 'precision']
metric_avg = ['macro avg', 'weighted avg']
for metric in metrics:
    for label in metric_avg:
        score = []
        for i, result in enumerate([svm_result, rf_result, ada_result]):
            result_f1 = result[label][metric]
            score.append(result_f1) 
        results.append(score)
    for j,label in enumerate(metric_avg): 
        fig, axis = plt.subplots(1,1,figsize=(5,3))
        fig.suptitle('Resultado de Classificação {} Metric: {}'.format(metric_avg[j], metric), fontsize=16)
        for i, result in enumerate(results):
            axis.bar([1,2,3],result,
                        tick_label=['svm_result', 'rf_result', 'ada_result'], color=['green', 'blue', 'yellow'])        

