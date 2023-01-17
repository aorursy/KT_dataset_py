import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
df = pd.DataFrame(np.random.randint(100,20000,size=(1000,9)),columns=['A','B','C','D','E','F','G','K','L'])
df.head()
# Create Y label
def Y(x):
    y = []
    for i in range(df.shape[0]):
        A = x['A'][i]
        B = x['B'][i]
        C = x['C'][i]
        D = x['D'][i]
        E = x['E'][i]
        F = x['F'][i]
        G = x['G'][i]
        SUM = A + B + C + D + E + F + G
        
        if SUM <= 69000:
            y.append(1)
        else :y.append(0)
    return y
    


x = df[['A','B','C','D','E','F','G']]

df['Y'] = Y(x)
sns.pairplot(df ,hue='Y')
from sklearn.preprocessing import StandardScaler
Scaler = StandardScaler()
Scaler.fit(df.drop('Y',axis=1))
scaled_features = Scaler.transform(df.drop('Y',axis=1))
df_feet = pd.DataFrame(scaled_features ,columns=df.columns[:-1])
df_feet.head()
from sklearn.model_selection import train_test_split
X = df_feet
y = df['Y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)
pred = knn.predict(X_test)
from sklearn.metrics import classification_report ,confusion_matrix
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))
error_rate = []

for i in range(1,40):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    
    knn.fit(X_train,y_train)
    
    pred_i = knn.predict(X_test)
    
    error_rate.append(np.mean(pred_i !=y_test))
    
plt.figure(figsize=(10,6))

plt.plot(range(1,40) ,error_rate ,color='red',linestyle='--',marker='o' ,
         markerfacecolor='black' ,markersize=10)
plt.title('Error_reate vs K')
plt.xlabel('K')
plt.ylabel('Error Rate')

knn = KNeighborsClassifier(n_neighbors=35)
    
knn.fit(X_train,y_train)
    
pred_n = knn.predict(X_test)
    

print(confusion_matrix(y_test,pred))
print('---------------------------------------------------')
print(classification_report(y_test,pred))
