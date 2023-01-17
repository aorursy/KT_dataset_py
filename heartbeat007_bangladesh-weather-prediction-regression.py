## importing basic modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#try:
#    !pip install tensorflow-gpu
#except:
#!pip install tensorflow
import tensorflow as tf

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (20,20)


dataset = pd.read_csv('../input/Temp_and_rain.csv')
dataset.head()
dataset.isnull().sum()
dataset[['rain']].plot()
dataset.rain.hist()
dataset.tem.hist()
plt.bar(dataset['Year'],dataset['rain'])
plt.xlabel("Year")
plt.ylabel("Rain")
plt.legend()
plt.bar(dataset['tem'],dataset['rain'])
plt.xlabel("TEMP")
plt.ylabel("Rain")
plt.legend()
import seaborn as sns
correlation = dataset.corr()
correlation
sns.heatmap(correlation,cmap='coolwarm',annot=True)
## setting the style first
sns.set(style="whitegrid",color_codes=True) ## change style
sns.distplot(dataset['rain'], kde=False, bins=100);
sns.distplot(dataset['tem'],kde=False, bins=100);

sns.relplot(x="Year", y="rain", data=dataset);
sns.relplot(x="Year", y="tem", data=dataset);
sns.relplot(x="Year", y="tem", hue="rain", data=dataset);
sns.boxplot(data=dataset,orient='h')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
dataset.head()
X = dataset.drop('rain',axis=1)
X = X.drop('tem',axis=1)
y = dataset[['rain','tem']]
X.head()
y.head()
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=.2)

from keras import Sequential
from keras.layers import Dense,Dropout,LSTM,Flatten
print (x_train.shape)
print (x_test.shape)
x_train = np.array(x_train)
x_test = np.array(x_test)


x_train
print (x_train.shape)
print (x_test.shape)
print (y_train.shape)
print (y_test.shape)
n_col = x_train.shape[1]
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
visible = Input(shape=(n_col,))
hidden1 = Dense(100, activation='relu')(visible)
hidden2 = Dense(200, activation='relu')(hidden1)
hidden3 = Dense(100, activation='relu')(hidden2)
hidden4 = Dense(100, activation='relu')(hidden3)
hidden5 = Dense(100, activation='relu')(hidden4)
hidden6 = Dense(100, activation='relu')(hidden5)
hidden7 = Dense(100, activation='relu')(hidden6)
output = Dense(2)(hidden7)
model = Model(inputs=visible, outputs=output)
model.compile(optimizer='adam',loss='mean_absolute_error')
model.fit(x_train,y_train,epochs = 100)
y_pred = model.predict(x_test)
y_pred
model.evaluate(x_test,y_test)
dataset = pd.read_csv('../input/Temp_and_rain.csv')
X = dataset.drop('rain',axis=1)
X = X.drop('tem',axis=1)
y = dataset[['rain','tem']]
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=.2)
from sklearn.neighbors import KNeighborsRegressor
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(x_train,y_train)
predicted=knn.predict(x_test)
predicted
model.evaluate(x_test,y_test)
accuracy=[]
for k in range(1,50):
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(x_train,y_train)
    accuracy.append(knn.score(x_test,y_test))


    
plt.plot(range(1,50),accuracy)
training_accuracy=[]
testing_accuracy=[]

neighbors = list(range(1,50))


for k in neighbors:
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(x_train,y_train)
    training_accuracy.append(knn.score(x_train,y_train))
    testing_accuracy.append(knn.score(x_test,y_test))    

plt.plot(neighbors,training_accuracy,label='training accuracy')
plt.plot(neighbors,testing_accuracy,label='testing accuracy')
plt.ylabel("Accuracy")
plt.xlabel("K value")
plt.legend()


from sklearn.model_selection import cross_val_score

knn = KNeighborsRegressor(n_neighbors=5)

scores = cross_val_score(knn,X,y,cv=10)

print (scores)
print (scores.mean())

print ("Mean Accuracy "+str(scores.mean()))






k_range = range(1,50)
k_scores = []

for k in k_range:
    knn = KNeighborsRegressor(n_neighbors=k)
    scores = cross_val_score(knn,X,y,cv=10)
    k_scores.append(scores.mean())
    
print (k_scores)


plt.plot(k_range,k_scores)
plt.xlabel("k range")
plt.ylabel("scores")
from sklearn.tree import DecisionTreeRegressor
tree_clf = DecisionTreeRegressor(max_depth=2,random_state=42)
tree_clf.fit(X,y)
tree_clf.score(X,y)
accuracy=[]
for depth in range(1,50):
    dt = DecisionTreeRegressor(max_depth=depth,random_state=42)
    dt.fit(x_train,y_train)
    accuracy.append(dt.score(x_test,y_test))
plt.plot(range(1,50),accuracy)
from sklearn.ensemble import RandomForestRegressor
rnd = RandomForestRegressor(max_depth=10)
rnd.fit(x_train,y_train)
rnd.score(x_test,y_test)
accuracy=[]
for depth in range(1,50):
    dt = RandomForestRegressor(max_depth=depth,random_state=42)
    dt.fit(x_train,y_train)
    accuracy.append(dt.score(x_test,y_test))
plt.plot(range(1,50),accuracy)
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR,LinearSVR
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor

from mlxtend.regressor import StackingRegressor
dtc=DecisionTreeRegressor()
knnc=KNeighborsRegressor()
gbc=GradientBoostingRegressor()
rfc=RandomForestRegressor()


stregr = StackingRegressor(regressors=[dtc,knnc,gbc,rfc], 
                           meta_regressor=knnc)
y_train
stregr.fit(x_train, y_train['tem'])
prediction = stregr.predict(x_test)
stregr.score(x_test,y_test['tem'])
stregr.fit(x_train, y_train['rain'])
prediction = stregr.predict(x_test)
stregr.score(x_test,y_test['rain'])
