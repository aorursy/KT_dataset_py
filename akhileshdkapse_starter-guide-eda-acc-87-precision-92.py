import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
df=pd.read_csv('/kaggle/input/pima-indians-diabetes-database/diabetes.csv')
df.head()
df.info()
# Distribution of our data
df.hist(bins=25,figsize=(20,8))
# Correlation 
corr=df.corr()
f,ax=plt.subplots(1,1,figsize=(12,8))
sns.heatmap(corr, annot=True, cmap="YlGnBu", ax=ax)
ax=sns.countplot('Outcome', data=df)
print('1:- Diabetes....|||||....0:- healthy')
df=df.loc[(df.BMI>10) & (df.BloodPressure>20) & (df.Glucose>25)]
# Distribution of our data
df.hist(bins=25,figsize=(20,8))
df.info()
df.describe()
df.SkinThickness.hist(bins=20)
df.loc[(df.SkinThickness<5)& (df.Outcome==0), 'SkinThickness']=int(df[(df.Outcome==0)]['SkinThickness'].median())
df.loc[(df.SkinThickness<5)& (df.Outcome==1), 'SkinThickness']=int(df[(df.Outcome==1)]['SkinThickness'].median())
df.loc[(df.Insulin==0)& (df.Outcome==0), 'Insulin']=int(df[(df.Outcome==0)]['Insulin'].median())
df.loc[(df.Insulin==0)& (df.Outcome==1), 'Insulin']=int(df[(df.Outcome==1)]['Insulin'].median())
df.Insulin.hist(bins=20)
df.sample(6)
scaler = StandardScaler()
data_x=scaler.fit_transform(df.drop(['Outcome'], axis=1))
#data_x=df.drop(['Outcome'], axis=1)
data_y=df.Outcome.values
#data_y=data_y.reshape((-1,1))
data_x.shape,data_y.shape
from sklearn.decomposition import PCA
xtrain,xtest,ytrain,ytest=train_test_split(data_x,data_y,random_state=998)
xtrain.shape, xtest.shape
pca=PCA(n_components=2)
pca.fit(xtrain)
pca_xtrain=pca.transform(xtrain)
pca_xtest=pca.transform(xtest)
pca_xtrain.shape, pca_xtest.shape
def plot_2d(x_train,y_train,x_test,y_test):
    plt.figure(figsize=(16,8))
    sns.scatterplot(x=x_train[:,0], y=x_train[:,1], hue=y_train, marker = 'v', alpha=0.9,)
    sns.scatterplot(x=x_test[:,0], y=x_test[:,1], hue=y_test, alpha=0.8,  marker = 'o')
    
plot_2d(pca_xtrain,ytrain, pca_xtest, ytest)
# imports we need............
from sklearn.model_selection import cross_val_score, ShuffleSplit, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
def fit(model, cv):
    return cross_val_score(model,data_x, data_y, cv=cv).mean()
cv = ShuffleSplit(n_splits=10, test_size=0.18)
acc=[]
for i in range(1,21):
    log_clf=LogisticRegression(C=i)
    acc.append(fit(log_clf,cv))
plt.grid(True)
plt.plot(acc ,marker='o')
cv = ShuffleSplit(n_splits=10, test_size=0.18)
acc=[]
for i in tqdm(range(1,76)):
    log_clf=KNeighborsClassifier(n_neighbors=i)
    acc.append(fit(log_clf,cv))
plt.figure(figsize=(12,5))
plt.grid(True)
plt.plot(acc ,marker='o')
cv = ShuffleSplit(n_splits=10, test_size=0.18)
acc=[]
for i in tqdm(range(1,60)):
    log_clf=SVC(C=i)
    acc.append(fit(log_clf,cv))

plt.figure(figsize=(12,5))
plt.grid(True)
plt.plot(acc ,marker='o')
cv = ShuffleSplit(n_splits=10, test_size=0.18)
acc=[]
dict_={}
for i in tqdm(range(1,152)):
    log_clf=RandomForestClassifier(n_estimators=i)
    Accuracy=fit(log_clf,cv)
    acc.append(Accuracy)
    dict_[i]=Accuracy

plt.figure(figsize=(12,5))
plt.grid(True)
plt.plot(acc ,marker='o')
sorted(dict_.items(), key=lambda x: x[1], reverse=True)[:6]
from sklearn.metrics import confusion_matrix,accuracy_score,roc_curve, classification_report
xtrain,xtest,ytrain,ytest=train_test_split(data_x,data_y,random_state=998)
rand_clf=RandomForestClassifier(n_estimators=101)
rand_clf.fit(xtrain,ytrain)
print(confusion_matrix(ytest, rand_clf.predict(xtest)))
print('Accuracy of our model is: ', accuracy_score(ytest, rand_clf.predict(xtest)))
print(classification_report(ytest, rand_clf.predict(xtest)))
rand_clf=RandomForestClassifier(n_estimators=91)
cross_val_score(rand_clf,data_x, data_y, cv=cv).mean()
%cd /kaggle/working
import pickle
Pkl_Filename = "Pima_final.pkl"  
with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(rand_clf, file)
xtrain,xtest,ytrain,ytest=train_test_split(data_x,data_y,random_state=998)
ytrain.sum(),len(ytrain),ytest.sum(),len(ytest)
ytest=ytest.reshape(-1,1)
ytrain=ytrain.reshape(-1,1)
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.regularizers import l1_l2
check_point=tf.keras.callbacks.ModelCheckpoint(
    filepath='diabetes.h5', monitor='val_loss', verbose=1, save_best_only=True,
    save_weights_only=False, mode='min')
model=Sequential([
    Dense(80,activation='relu',input_shape=(None,8)),
    Dropout(0.5),
    Dense(120,activation='relu', kernel_regularizer=l1_l2()),
    Dropout(0.5),
    Dense(128,activation='relu'),
    Dropout(0.5),
    Dense(60,activation='relu'),
    Dropout(0.5),
    Dense(30,activation='relu'),
    Dense(1,activation='sigmoid')
])

model.compile(loss='BinaryCrossentropy', optimizer=tf.keras.optimizers.Adam(0.001), metrics=['accuracy'])
model.summary()
history=model.fit(xtrain,ytrain,epochs=300,validation_data=(xtest,ytest), callbacks=[check_point])
plt.figure(1, figsize = (25, 12))
plt.subplot(1,2,1)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.plot( history.history["loss"], label = "Training Loss")
plt.plot( history.history["val_loss"], label = "Validation Loss")
plt.grid(True)
plt.legend()

plt.subplot(1,2,2)
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.plot( history.history["accuracy"], label = "Training Accuracy")
plt.plot( history.history["val_accuracy"], label = "Validation Accuracy")
plt.grid(True)
plt.legend()
model_new=keras.models.load_model('diabetes.h5')
model_new.evaluate(xtest,ytest)
print(confusion_matrix(ytest, model_new.predict_classes(xtest)))
print('Accuracy of our model is: ', accuracy_score(ytest, model_new.predict_classes(xtest)))
print(classification_report(ytest, model_new.predict_classes(xtest)))
