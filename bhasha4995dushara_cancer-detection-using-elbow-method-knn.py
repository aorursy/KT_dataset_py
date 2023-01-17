import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

import os
print(os.listdir("../input"))
labels = ['id', 'Clump Thickness', 'Uniformity of Cell Size', 
       'Uniformity of Cell Shape', 'Marginal Adhesion', 
       'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin',
       'Normal Nucleoli', 'Mitoses', 'Class']
df = pd.read_csv('../input/breast-cancer-wisconsin.data.txt')
df.columns = labels
#data.set_index('id', inplace=True)
df.head()
df.columns[1:10]
print(np.where(df.isnull()))
print(df.describe())
df.info()
df['Bare Nuclei'].describe()
df['Bare Nuclei'].value_counts()
bare_index = df[df['Bare Nuclei'] == '?'].index
b = np.array(bare_index)
df.loc[b,'Bare Nuclei'] = 0
df['Class'].value_counts()
df['Class'] = df['Class'] / 2 - 1
df['Class'].value_counts()
features = df[['Clump Thickness', 'Uniformity of Cell Size', 
       'Uniformity of Cell Shape', 'Marginal Adhesion', 
       'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin',
       'Normal Nucleoli', 'Mitoses']]
target = df['Class']
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler_feature = scaler.fit_transform(features)
df_feature = pd.DataFrame(scaler_feature,columns=df.columns[1:10])
df_feature.iloc[1:3]
X = df_feature
y = target
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)
error_rate = []

for i in range(1,40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    y_pred = knn.predict(X_test)
    error_rate.append(np.mean(y_pred != y_test))
plt.figure(figsize=(10,5))
plt.xlabel('K')
plt.ylabel('Error Rate')
plt.title('Error Rate vs. K')
plt.plot(range(1,40),error_rate,color='blue',marker='o',markerfacecolor='pink',markersize=5,ls='--')
knn = KNeighborsClassifier(n_neighbors=24)
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,y_pred))
tn,fp,fn,tp = confusion_matrix(y_test,y_pred).ravel()
print(tn,fp,fn,tp)
from sklearn.metrics import accuracy_score
# inbuilt function
print('Accuracy Score :',accuracy_score(y_test,y_pred))
#own calculation
total = sum(sum(confusion_matrix(y_test,y_pred)))
accuracy = (tn + tp)/total
print("Accuracy : ",accuracy)
# recall(ACTUAL) and sensitivity (TPR)
sensitivity = tp / (tp + fn)
print('Sensitivity : ', sensitivity )
#precision(PREDICT) and specificity (TNR)
specificity = tn /(tn + fp )
print('Specificity : ', specificity)
