import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/glass-identification-dataset/glass_data.csv')
df.head() 
df.info()
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df.drop('column_k',axis=1))
Scaled = scaler.transform(df.drop('column_k',axis=1))
scaledview = pd.DataFrame(Scaled,columns=df.columns[:-1])
scaledview.head()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(Scaled,df['column_k'],
                                                    test_size=0.30)

import math
print(math .sqrt(len(y_train)))
print(len(y_train))
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=11,p=2,metric='euclidean')
knn.fit(X_train,y_train)
y_pre = knn.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix,f1_score,accuracy_score
print(classification_report(y_test,y_pre))
confusion_matrix(y_test,y_pre)
accuracy_score(y_test, y_pre)
residuals = y_test - y_pre
residuals.mean()
plt.scatter(y_pre, residuals)
error_rate = []

# Will take some time
for i in range(1,20):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))
plt.figure(figsize=(10,6))
plt.plot(range(1,20),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
#better keep the K value as square root of data points: thats is 11
#K =11 is the best




