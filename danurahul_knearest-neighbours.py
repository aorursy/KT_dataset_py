import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
df=pd.read_csv('https://raw.githubusercontent.com/krishnaik06/K-Nearest-Neighour/master/Classified%20Data',index_col=0)
df
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x=scaler.fit(df.drop('TARGET CLASS',axis=1))
scaled_features=scaler.transform(df.drop('TARGET CLASS',axis=1))
df_feat=pd.DataFrame(scaled_features,columns=df.columns[:-1])
df_feat.head()
import seaborn as sns
sns.pairplot(df,hue='TARGET CLASS')
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(scaled_features, df['TARGET CLASS'],test_size=0.30)
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train,y_train)
pred=knn.predict(x_test)
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import cross_val_score
print(confusion_matrix(pred,y_test))
print(classification_report(pred,y_test))
accuracy_rate = []
for i in range(1,40):
    knn=KNeighborsClassifier(n_neighbors=i)
    scores=cross_val_score(knn,df_feat,df['TARGET CLASS'],cv=10)
    accuracy_rate.append(scores.mean())
    
error_rate = []
for i in range(1,40):
    knn=KNeighborsClassifier(n_neighbors=i)
    scores=cross_val_scores(knn,df_feat,df['TARGET CLASS'],cv=10)
    error_rate.append(1-scores.mean())
plt.figure(figsize=(10,6))
plt.plot(range(1,40),accuracy_rate,color='blue',linestyle='dashed',marker='o')

knn=KNeighborsClassifier(n_neighbors=32)
knn.fit(x_train,y_train)
pred=knn.predict(x_test)
print(confusion_matrix(pred,y_test))
print(classification_report(pred,y_test))
