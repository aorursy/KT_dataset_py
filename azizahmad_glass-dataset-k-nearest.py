import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

sns.set_style('whitegrid')
glass = pd.read_csv('../input/glass/glass.csv')
glass
sns.pairplot(glass,hue='Type')
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(glass.drop('Type',axis=1))
scaled_features = scaler.transform(glass.drop('Type',axis=1))
scaled_df = pd.DataFrame(data=scaled_features,columns=glass.columns[:-1])

scaled_df.head()
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(scaled_df,glass['Type'],test_size=0.33)
from sklearn.neighbors import KNeighborsClassifier
error_rate = []

for i in range(1,50):

    kn = KNeighborsClassifier(n_neighbors=i)

    kn.fit(X_train,y_train)

    pred_i = kn.predict(X_test)

    error_rate.append(np.mean(y_test != pred_i))
plt.figure(figsize=(11,5))

plt.plot(range(1,50),error_rate,marker='o',markerfacecolor='red')
# It seems the best n_neighbors to use is either 1 or 2. As it goes higher, the number of errors raise too.
kn = KNeighborsClassifier(n_neighbors=2)
kn.fit(X_train,y_train)
pred = kn.predict(X_test)
from sklearn.metrics import confusion_matrix,classification_report

print(confusion_matrix(y_test,pred))

print('\n')

print(classification_report(y_test,pred))