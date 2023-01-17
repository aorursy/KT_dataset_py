import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,confusion_matrix
df = pd.read_csv("../input/Iris.csv")
df.head()
sns.pairplot(df, hue='Species')
#Scaling the features
scaler = StandardScaler()
scaler.fit(df.drop('Species',axis=1))
scaled_features = scaler.transform(df.drop('Species',axis=1))
#Creating training and test datasets
data_train, data_test, label_train, label_test = train_test_split(scaled_features,df['Species'],
                                                    test_size=0.30)
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(data_train,label_train)
pred = knn.predict(data_test)
print(classification_report(label_test,pred))
#Choosing the best value for k based on the error rate
error_rate = []
kmax=75
for i in range(1,kmax):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(data_train,label_train)
    pred = knn.predict(data_test)
    error_rate.append(np.mean(pred != label_test))
plt.figure(figsize=(10,6))
plt.plot(range(1,kmax),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(data_train,label_train)
predict = knn.predict(data_test)
print(classification_report(label_test,predict))