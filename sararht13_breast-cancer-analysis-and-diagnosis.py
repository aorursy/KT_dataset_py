from flask import *
import numpy as np # linear algebra
from sklearn.decomposition import PCA
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # data visualization library  
import matplotlib.pyplot as plt

np.random.seed(0)
data = pd.read_csv('../input/data.csv') 
# Data size
print (data.shape)

# Look at the 5 first rows
data.head() 
data.isnull().sum()
# Data names
col = data.columns      

# Diagnosis includes our labels and x includes our features
y = data.diagnosis    # M or B 

# Drop the last column, ID and diagnosis
df=data.drop(['Unnamed: 32','id'],axis=1)
x = df.drop('diagnosis',axis = 1 )
x.head()
ax = sns.countplot(y,label="Count")
B, M = y.value_counts()
print('Number of Benign: ',B)
print('Number of Malignant : ',M)
y_df= pd.get_dummies(y,drop_first=True) # dropping the column called diagnosis and having a columns of 0 and 1
y_df.head()
y_df=y_df['M']
prueba=pd.get_dummies(df,'diagnosis')
prueba.drop('diagnosis_B',axis=1)
x.describe()
# Correlation map
f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(x.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax);
list_mean=['radius_mean','texture_mean','perimeter_mean','area_mean',
      'smoothness_mean','compactness_mean','concavity_mean',
      'concave points_mean','symmetry_mean','fractal_dimension_mean']
x_mean=x[list_mean]
x_mean.head()
list_SE=['radius_se','texture_se','perimeter_se','area_se',
      'smoothness_se','compactness_se','concavity_se',
      'concave points_se','symmetry_se','fractal_dimension_se']
x_SE=x[list_SE]
x_SE.head()
list_worst=['radius_worst','texture_worst','perimeter_worst','area_worst',
      'smoothness_worst','compactness_worst','concavity_worst',
      'concave points_worst','symmetry_worst','fractal_dimension_worst']
x_worst=x[list_worst]
x_worst.head()
#correlation map
f,ax = plt.subplots(figsize=(9, 8))
sns.heatmap(x_mean.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax);
#correlation map
f,ax = plt.subplots(figsize=(9, 8))
sns.heatmap(x_SE.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax);
#correlation map
f,ax = plt.subplots(figsize=(9, 8))
sns.heatmap(x_worst.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax);
x.head()
x_radio=x[['radius_mean','radius_se','radius_worst']]
x_texture=x[['texture_mean','texture_se','texture_worst']]
x_perimeter=x[['perimeter_mean','perimeter_se','perimeter_worst']]
x_area=x[['area_mean','area_se','area_worst']]
x_smoothness=x[['smoothness_mean','smoothness_se','smoothness_worst']]
f,ax = plt.subplots(figsize=(5, 4))
#Change the x_area for x_radio, x_texture... to see another different map
sns.heatmap(x_area.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax);
# Convert data into an array
X = x.values

# Call PCA method
# let's call n_components = 2
pca = PCA(n_components=2)
pca_2d = pca.fit_transform(X)

# Plot the PCA
plt.figure(figsize = (16,11))
plt.scatter(pca_2d[:,0],pca_2d[:,1], c = y_df,
            cmap = "coolwarm", edgecolor = "None", alpha=0.5,);
plt.title('PCA Scatter Plot');

import matplotlib.patches as mpatches
rects=[]
rects.append(mpatches.Patch(color='blue', label='Benign'));
rects.append(mpatches.Patch(color='red', label='Malignant'));
plt.legend(handles=rects);

from sklearn.manifold import TSNE
# Se invoca el método TSNE.
# let's call n_components = 2
tsne= TSNE(n_components=2)
tsne_2d=tsne.fit_transform(X)
# Plot the T-SNE
plt.figure(figsize = (16,11))
plt.scatter(tsne_2d[:,0],tsne_2d[:,1], c = y_df,
            cmap = "coolwarm", edgecolor = "None", alpha=0.5,);
plt.title('TSNE Scatter Plot');

import matplotlib.patches as mpatches
rects=[]
rects.append(mpatches.Patch(color='blue', label='Benign'));
rects.append(mpatches.Patch(color='red', label='Malignant'));
plt.legend(handles=rects);

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
df_train, df_test = train_test_split(df, test_size = 0.3)
x_train=df_train.drop('diagnosis',axis=1)
x_test=df_test.drop('diagnosis',axis=1)
y_train=df_train['diagnosis']
y_test=df_test['diagnosis']
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
#Create the model
modelo_rl= LogisticRegression()
#Fit the model
modelo_rl.fit(X=x_train,y=y_train)
#Prediction
predicion_rl = modelo_rl.predict(x_test)
#Results:

#Clasification report
results_rl=metrics.classification_report(y_true=y_test, y_pred=predicion_rl)
print(results_rl)

#Confusion matrix
cm_rl=metrics.confusion_matrix(y_true=y_test, y_pred=predicion_rl)
f,ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm_rl, annot=True, linewidths=.5, fmt= '.1f',ax=ax);
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
# Create the model
modelo_ad= DecisionTreeClassifier()
# Fit the model
modelo_ad.fit(X=x_train,y=y_train)
# Prediction
predicion_ad = modelo_ad.predict(x_test)
#Results:

#Clasification report
results_ad=metrics.classification_report(y_true=y_test, y_pred=predicion_ad)
print(results_ad)

#Confusion Matrix
cm_ad=metrics.confusion_matrix(y_true=y_test, y_pred=predicion_ad)
f,ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm_ad, annot=True, linewidths=.5, fmt= '.1f',ax=ax);
from sklearn.ensemble import RandomForestClassifier
#Create the model
modelo_rf= RandomForestClassifier()
#Fit the model
modelo_rf.fit(X=x_train,y=y_train)
#Prediction
predicion_rf = modelo_rf.predict(x_test)
#Results:

#Clasification report
results_rf=metrics.classification_report(y_true=y_test, y_pred=predicion_rf)
print(results_rf)

#Confusion matrix
cm_rf=metrics.confusion_matrix(y_true=y_test, y_pred=predicion_rf)
f,ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm_rf, annot=True, linewidths=.5, fmt= '.1f',ax=ax);
from sklearn.svm import SVC
# Create the model
modelo_svm= SVC(kernel='linear', C = 1.0)
# Fit the model
modelo_svm.fit(X=x_train,y=y_train)
# Prediction
predicion_svm = modelo_svm.predict(x_test)
#Results:

#Clasification report
results_svm=metrics.classification_report(y_true=y_test, y_pred=predicion_svm)
print(results_svm)

#Confusion matrix
cm_svm=metrics.confusion_matrix(y_true=y_test, y_pred=predicion_svm)
f,ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm_svm, annot=True, linewidths=.5, fmt= '.1f',ax=ax);
from sklearn.neighbors import KNeighborsClassifier
# Create the model
modelo_knn= KNeighborsClassifier(n_neighbors=10)
# Fit the model
modelo_knn.fit(X=x_train,y=y_train)
# Prediction
predicion_knn = modelo_knn.predict(x_test)
# Results:

# Clasification report
results_knn=metrics.classification_report(y_true=y_test, y_pred=predicion_knn)
print(results_knn)

# Confusion matrix
cm_knn= metrics.confusion_matrix(y_true=y_test, y_pred=predicion_knn)
f,ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm_knn, annot=True, linewidths=.5, fmt= '.1f',ax=ax);
from sklearn.naive_bayes import GaussianNB
# Create the model
modelo_NB= GaussianNB()
# Fit the model
modelo_NB.fit(X=x_train,y=y_train)
#Prediction
predicion_NB = modelo_NB.predict(x_test)
#Results

#Clasification report
results_NB=metrics.classification_report(y_true=y_test, y_pred=predicion_NB)
print(results_NB)

#Confusion Matrix
cm_NB=metrics.confusion_matrix(y_true=y_test, y_pred=predicion_NB)
f,ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm_NB, annot=True, linewidths=.5, fmt= '.1f',ax=ax);
np.random.seed(0)

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
from keras.utils.np_utils import to_categorical
from keras import callbacks

# Create the model: many layers
model = Sequential()
model.add(Dense(units=160, kernel_initializer='uniform', activation='relu', input_dim=30))

# Adding the second hidden layer
model.add(Dense(units=160, kernel_initializer='uniform', activation='relu'))
model.add(Dense(units=160, kernel_initializer='uniform', activation='relu'))
model.add(Dense(units=160, kernel_initializer='uniform', activation='relu'))
model.add(Dense(units=160, kernel_initializer='uniform', activation='relu'))
model.add(Dense(units=160, kernel_initializer='uniform', activation='relu'))
model.add(Dense(units=160, kernel_initializer='uniform', activation='relu'))
model.add(Dense(units=160, kernel_initializer='uniform', activation='relu'))
model.add(Dense(units=160, kernel_initializer='uniform', activation='relu'))
model.add(Dense(units=160, kernel_initializer='uniform', activation='relu'))
model.add(Dense(units=160, kernel_initializer='uniform', activation='relu'))

# Adding the output layer
model.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
# Using "Binary_crossentropy"
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Adjusted the model using the previous cost optimizer and function
earlystop=callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5, mode='auto',verbose=1)
history=model.fit(x_train, pd.get_dummies(y_train,drop_first=True)['M'].values, validation_split=0.2, epochs=500, batch_size=5000, verbose=0, callbacks=[earlystop])
# Class prediction
predicion_NN = model.predict_classes(x_test, batch_size=32)
# Clasification report
results_NN =metrics.classification_report(y_true=pd.get_dummies(y_test,drop_first=True), y_pred=predicion_NN)
print (results_NN)
# Summarize history for loss
plt.plot(history.history['loss']);
plt.plot(history.history['val_loss']);
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','validation'], loc='upper right');
plt.show()
# Confusion Matrix
cm_NN=metrics.confusion_matrix(y_true=pd.get_dummies(y_test,drop_first=True), y_pred=predicion_NN)
f,ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm_NN, annot=True, linewidths=.5, fmt= '.1f',ax=ax)
# We define a function that returns the total (the last row) of the classification report in a float array
# In the same order: precision, recall, F1 score, support

def resultados_classification_report(cr):
    total=[]
    lines = cr.split('\n')
    total_aux=lines[5].split()
    for i in range(3,6):
        total.append(float(total_aux[i]))
    return total            

#We collect all the classification reports and obtain the total
names=['Decision Trees','KNN','Logistic Regression','NN','Naive-Bayes','Random Forest','SVM']
modelResults=[]
modelResults.append(results_ad)
modelResults.append(results_knn)
modelResults.append(results_rl)
modelResults.append(results_NN)
modelResults.append(results_NB)
modelResults.append(results_rf)
modelResults.append(results_svm)


totalResults=[]
totalPrecision=[]
totalRecall=[]
totalF=[]
for i in range(len(modelResults)):
    totalResults.append(resultados_classification_report(modelResults[i]))
   
for i in range(len(totalResults)):
     totalPrecision.append(totalResults[i][0])
     totalRecall.append(totalResults[i][1])
     totalF.append(totalResults[i][2])


index = np.arange(7)
# Set position of bar on X axis
barWidth=0.3
r1 = np.arange(len(totalPrecision))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]

plt.subplots(figsize=(16, 7))
plt.bar(r1,totalPrecision,width=barWidth, label='Precision', color="purple")
plt.bar(r2,totalRecall,width=barWidth, label='Recall', color="mediumpurple")
plt.bar(r3,totalF,width=barWidth,  label='F1-score', color="pink")
plt.axis([-0.5,7,0, 1])
plt.legend(loc='lower left')
plt.title('Results: 30 features')

# Add xticks on the middle of the group bars
plt.xticks([r + barWidth for r in range(len(totalPrecision))], names)
# Leyenda personalizada
rects=[]
rects.append(mpatches.Patch(color='blue', label='Precision'))
rects.append(mpatches.Patch(color='orange', label='Recall'))
rects.append(mpatches.Patch(color='green', label='F1 Score'))

plt.subplots(figsize=(10, 5))
plt.plot(names,totalResults)
plt.legend(handles=rects);
plt.title('Results: 30 features');

#Plot matrix
f,ax = plt.subplots(figsize=(20, 2))
plt.subplot(1,7,1)
plt.title('Decision Trees')
sns.heatmap(cm_ad, annot=True, linewidths=.5, fmt= '.1f')

plt.subplot(1,7,2)
plt.title('KNN')
sns.heatmap(cm_knn, annot=True, linewidths=.5, fmt= '.1f')

plt.subplot(1,7,3)
plt.title('Logistic Regression')
sns.heatmap(cm_rl, annot=True, linewidths=.5, fmt= '.1f')

plt.subplot(1,7,4)
plt.title('NN')
sns.heatmap(cm_NN, annot=True, linewidths=.5, fmt= '.1f')

plt.subplot(1,7,5)
plt.title('Naive-Bayes')
sns.heatmap(cm_NB, annot=True, linewidths=.5, fmt= '.1f')

plt.subplot(1,7,6)
plt.title('Random Forest')
sns.heatmap(cm_rf, annot=True, linewidths=.5, fmt= '.1f')

plt.subplot(1,7,7)
plt.title('SVM')
sns.heatmap(cm_svm, annot=True, linewidths=.5, fmt= '.1f')

np.random.seed(0)

#We eliminate 15 characteristics according to the criteria explained before
lista_borrar_aux=['radius_mean','radius_se','radius_worst','perimeter_mean','perimeter_se','perimeter_worst','compactness_mean','compactness_se','compactness_worst','symmetry_mean','symmetry_se','symmetry_worst','smoothness_mean','smoothness_se','smoothness_worst']

df_train2=df_train.drop(lista_borrar_aux,axis=1)
df_test2=df_test.drop(lista_borrar_aux,axis=1)
x_train2=df_train2.drop('diagnosis',axis=1)
x_test2=df_test2.drop('diagnosis',axis=1)
y_train2=df_train2['diagnosis']
y_test2=df_test2['diagnosis']

#Create the models
modelo_rl2= LogisticRegression()
modelo_ad2= DecisionTreeClassifier()
modelo_rf2= RandomForestClassifier()
modelo_svm2= SVC(kernel='linear', C = 1.0)
modelo_knn2= KNeighborsClassifier(n_neighbors=10)
modelo_NB2= GaussianNB()
#Fit the models
modelo_rl2.fit(X=x_train2,y=y_train2, )
modelo_ad2.fit(X=x_train2,y=y_train2)
modelo_rf2.fit(X=x_train2,y=y_train2)
modelo_svm2.fit(X=x_train2,y=y_train2)
modelo_knn2.fit(X=x_train2,y=y_train2)
modelo_NB2.fit(X=x_train2,y=y_train2)
#Predictions
predicion_rl2 = modelo_rl2.predict(x_test2)
predicion_ad2 = modelo_ad2.predict(x_test2)
predicion_rf2 = modelo_rf2.predict(x_test2)
predicion_svm2 = modelo_svm2.predict(x_test2)
predicion_knn2= modelo_knn2.predict(x_test2)
predicion_NB2= modelo_NB2.predict(x_test2)
#Results
results_rl2=metrics.classification_report(y_true=y_test2, y_pred=predicion_rl2)
results_ad2=metrics.classification_report(y_true=y_test2, y_pred=predicion_ad2)
results_rf2=metrics.classification_report(y_true=y_test2, y_pred=predicion_rf2)
results_svm2=metrics.classification_report(y_true=y_test2, y_pred=predicion_svm2)
results_knn2=metrics.classification_report(y_true=y_test2, y_pred=predicion_knn2)
results_NB2=metrics.classification_report(y_true=y_test2, y_pred=predicion_NB2)

print('\n \033[1m Logistic Regression \033[0m \n'+ results_rl2)
print('\n \033[1m Decision Trees \033[0m \n'+ results_ad2)
print('\n \033[1m Random Forest \033[0m \n'+ results_rf2)
print('\n \033[1m SVM \033[0m \n'+ results_svm2)
print('\n \033[1m KNN \033[0m \n'+ results_knn2)
print('\n \033[1m Naive-Bayes \033[0m \n'+ results_NB2)
np.random.seed(0)

# Create the model: many layers
model2 = Sequential()
model2.add(Dense(units=160, kernel_initializer='uniform', activation='relu', input_dim=15))

# Adding the second hidden layer
model2.add(Dense(units=160, kernel_initializer='uniform', activation='relu'))
model2.add(Dense(units=160, kernel_initializer='uniform', activation='relu'))
model2.add(Dense(units=160, kernel_initializer='uniform', activation='relu'))
model2.add(Dense(units=160, kernel_initializer='uniform', activation='relu'))
model2.add(Dense(units=160, kernel_initializer='uniform', activation='relu'))
model2.add(Dense(units=160, kernel_initializer='uniform', activation='relu'))
model2.add(Dense(units=160, kernel_initializer='uniform', activation='relu'))
model2.add(Dense(units=160, kernel_initializer='uniform', activation='relu'))
model2.add(Dense(units=160, kernel_initializer='uniform', activation='relu'))
model2.add(Dense(units=160, kernel_initializer='uniform', activation='relu'))


# Adding the output layer
model2.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
model2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# we adjusted the model using the previous cost optimizer and function
earlystop=callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5, mode='auto',verbose=1)
history2=model2.fit(x_train2, pd.get_dummies(y_train2,drop_first=True)['M'].values, validation_split=0.2, epochs=500, batch_size=5000, verbose=0, callbacks=[earlystop])

# Prediction
predicion_NN2 = model2.predict_classes(x_test2, batch_size=32)

results_NN2=metrics.classification_report(y_true=pd.get_dummies(y_test2,drop_first=True), y_pred=predicion_NN2)
print(results_NN2)

# Summarize history for loss
plt.plot(history2.history['loss'])
plt.plot(history2.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','validation'], loc='upper right')
plt.show()
#Clasification report
modelResults2=[]
modelResults2.append(results_ad2)
modelResults2.append(results_knn2)
modelResults2.append(results_rl2)
modelResults2.append(results_NN2)
modelResults2.append(results_NB2)
modelResults2.append(results_rf2)
modelResults2.append(results_svm2)

totalResults2=[]
totalPrecision2=[]
totalRecall2=[]
totalF2=[]

for i in range(len(modelResults2)):
    totalResults2.append(resultados_classification_report(modelResults2[i]))

for i in range(len(totalResults)):
     totalPrecision2.append(totalResults2[i][0])
     totalRecall2.append(totalResults2[i][1])
     totalF2.append(totalResults2[i][2])


index = np.arange(7)

# Set position of bar on X axis
barWidth=0.3
r1 = np.arange(len(totalPrecision))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]

plt.subplots(figsize=(16, 7))
plt.bar(r1,totalPrecision2,width=barWidth, label='Precision', color="purple")
plt.bar(r2,totalRecall2,width=barWidth, label='Recall', color="mediumpurple")
plt.bar(r3,totalF2,width=barWidth,  label='F1-score', color="pink")
plt.axis([-0.5,7,0, 1])
plt.legend(loc='lower left')
plt.title('Results: 15 features')

# Add xticks on the middle of the group bars
plt.xticks([r + barWidth for r in range(len(totalPrecision))], names)
rects=[]
rects.append(mpatches.Patch(color='blue', label='Precision'))
rects.append(mpatches.Patch(color='orange', label='Recall'))
rects.append(mpatches.Patch(color='green', label='F1 Score'))

plt.subplots(figsize=(10, 5))
plt.plot(names,totalResults2)
plt.legend(handles=rects, loc='lower left')
plt.title('Results removing features ')
#Confusion matrix
cm_ad2=metrics.confusion_matrix(y_true=y_test2, y_pred=predicion_ad2)
cm_knn2=metrics.confusion_matrix(y_true=y_test2, y_pred=predicion_knn2)
cm_rl2=metrics.confusion_matrix(y_true=y_test2, y_pred=predicion_rl2)
cm_rf2=metrics.confusion_matrix(y_true=y_test2, y_pred=predicion_rf2)
cm_svm2=metrics.confusion_matrix(y_true=y_test2, y_pred=predicion_svm2)
cm_NB2=metrics.confusion_matrix(y_true=y_test, y_pred=predicion_NB2)
cm_NN2=metrics.confusion_matrix(y_true=pd.get_dummies(y_test,drop_first=True), y_pred=predicion_NN2)

#Plot matrix
f,ax = plt.subplots(figsize=(20, 2))
plt.subplot(1,7,1)
plt.title('Decision Trees')
sns.heatmap(cm_ad2, annot=True, linewidths=.5, fmt= '.1f')

plt.subplot(1,7,2)
plt.title('KNN')
sns.heatmap(cm_knn2, annot=True, linewidths=.5, fmt= '.1f')

plt.subplot(1,7,3)
plt.title('Logistic Regression')
sns.heatmap(cm_rl2, annot=True, linewidths=.5, fmt= '.1f')

plt.subplot(1,7,4)
plt.title('NN')
sns.heatmap(cm_NN2, annot=True, linewidths=.5, fmt= '.1f')

plt.subplot(1,7,5)
plt.title('Naive-Bayes')
sns.heatmap(cm_NB2, annot=True, linewidths=.5, fmt= '.1f')

plt.subplot(1,7,6)
plt.title('Random Forest')
sns.heatmap(cm_rf2, annot=True, linewidths=.5, fmt= '.1f')

plt.subplot(1,7,7)
plt.title('SVM')
sns.heatmap(cm_svm2, annot=True, linewidths=.5, fmt= '.1f')

index = np.arange(7)
# Set position of bar on X axis
barWidth=0.3
r1 = np.arange(len(totalPrecision))
r2 = [x + barWidth for x in r1]

plt.subplots(figsize=(15, 7))
plt.bar(r1,totalPrecision,width=barWidth, label='30 features', color="purple")
plt.bar(r2,totalPrecision2,width=barWidth, label='15 features', color="pink")
plt.axis([-0.5,7,0, 1])
plt.legend(loc='lower left')
plt.title('Comparative: Precision')

# Add xticks on the middle of the group bars
plt.xticks([r + 0.15 for r in range(len(totalPrecision))], names)
print(totalPrecision)
print (totalPrecision2)

plt.subplots(figsize=(10, 5))
plt.plot(names,totalPrecision, label='30 features')
plt.plot(names,totalPrecision2, label='15 features')
plt.legend()
plt.title('Comparative: Precision')
f,ax = plt.subplots(figsize=(18, 5))

#Plot matrix: 30 features
plt.subplot(2,7,1)
plt.title('Decision Trees')
sns.heatmap(cm_ad, annot=True, linewidths=.5, fmt= '.1f')

plt.subplot(2,7,2)
plt.title('KNN')
sns.heatmap(cm_knn, annot=True, linewidths=.5, fmt= '.1f')

plt.subplot(2,7,3)
plt.title('Logistic Regression')
sns.heatmap(cm_rl, annot=True, linewidths=.5, fmt= '.1f')

plt.subplot(2,7,4)
plt.title('NN')
sns.heatmap(cm_NN, annot=True, linewidths=.5, fmt= '.1f')

plt.subplot(2,7,5)
plt.title('Naive Bayes')
sns.heatmap(cm_NB, annot=True, linewidths=.5, fmt= '.1f')

plt.subplot(2,7,6)
plt.title('Random Forest')
sns.heatmap(cm_rf, annot=True, linewidths=.5, fmt= '.1f')

plt.subplot(2,7,7)
plt.title('SVM')
sns.heatmap(cm_svm, annot=True, linewidths=.5, fmt= '.1f')


#Plot matrix: 15 features
plt.subplot(2,7,8)
plt.title('Decision Trees')
sns.heatmap(cm_ad2, annot=True, linewidths=.5, fmt= '.1f')

plt.subplot(2,7,9)
plt.title('KNN')
sns.heatmap(cm_knn2, annot=True, linewidths=.5, fmt= '.1f')

plt.subplot(2,7,10)
plt.title('Logistic Regression')
sns.heatmap(cm_rl2, annot=True, linewidths=.5, fmt= '.1f')

plt.subplot(2,7,11)
plt.title('NN')
sns.heatmap(cm_NN2, annot=True, linewidths=.5, fmt= '.1f')

plt.subplot(2,7,12)
plt.title('Naive Bayes')
sns.heatmap(cm_NB2, annot=True, linewidths=.5, fmt= '.1f')

plt.subplot(2,7,13)
plt.title('Random Forest')
sns.heatmap(cm_rf2, annot=True, linewidths=.5, fmt= '.1f')

plt.subplot(2,7,14)
plt.title('SVM')
sns.heatmap(cm_svm2, annot=True, linewidths=.5, fmt= '.1f')


plt.tight_layout()

