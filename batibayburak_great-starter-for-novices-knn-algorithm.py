import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np

%matplotlib inline
df = pd.read_csv('../input/pima-indians-diabetes-database/diabetes.csv')
df.info()
df.head()
# Use seaborn on the dataframe to create a pairplot with the hue indicated by the Outcome column.

# It is very large plot

sns.pairplot(df,hue='Outcome',palette='coolwarm');
sns.pairplot(df)
# import the main KNN ilibrary

from sklearn.preprocessing import StandardScaler
# StandardScaler() object called scaler

scaler = StandardScaler()
# Fit the scaler to the features

scaler.fit(df.drop('Outcome',axis=1))
# Transform the features to a scaled version 

scaled_features = scaler.transform(df.drop('Outcome',axis=1))
# Convert the scaled features to a dataframe 

df_feat = pd.DataFrame(scaled_features,columns=df.columns[:-1])

df_feat.head()
# Train and test split

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(scaled_features,df['Outcome'],

                                                    test_size=0.30)
# Create a KNN model instance with n_neighbors=1# 



from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train,y_train)
# Fit this KNN model to the training data



pred = knn.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))
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
#After that we choose some K Value for available algorihmas value

# Retrain with new K Value

knn = KNeighborsClassifier(n_neighbors=1)



knn.fit(X_train,y_train)

pred = knn.predict(X_test)



print('WITH K=1')

print('\n')

print(confusion_matrix(y_test,pred))

print('\n')

print(classification_report(y_test,pred))
knn = KNeighborsClassifier(n_neighbors=23)



knn.fit(X_train,y_train)

pred = knn.predict(X_test)



print('WITH K=23')

print('\n')

print(confusion_matrix(y_test,pred))

print('\n')

print(classification_report(y_test,pred))