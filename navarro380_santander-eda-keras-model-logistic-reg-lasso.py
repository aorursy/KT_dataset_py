import pandas as pd

import numpy as np

import os

print(os.listdir("../input"))

import matplotlib.pyplot as plt
data=pd.read_csv('../input/train.csv')

data_pred=pd.read_csv('../input/test.csv')
data.describe()
c=data['target'].value_counts()

print('% of 0 ---> ', c[0]/(c[0]+c[1]))

print('% of 1 ---> ', c[1]/(c[0]+c[1]))
data['target'].hist()
data[data.columns[2:102]].plot(kind='box', figsize=[15,4], title='Non standarized values')

data[data.columns[103:]].plot(kind='box', figsize=[15,4], title='Non standarized values')

import seaborn as sns
values=data.columns.drop(['ID_code', 'target'])

plt.figure(figsize=(20,10))

for val in values:

    sns.distplot(data[val], hist=False)



plt.title('Density non Stadarized Data')

plt.xlabel('features')

plt.ylabel('density')

val_max=pd.DataFrame(data=data.max(), columns=['max'])

val_max=val_max[2:]

val_max['var']=val_max.index

val_max[['var', 'max']].head()
val_max['max'].plot(kind='hist', title='Max values distribution')
val_max['max'].plot(kind='box', title='Max values distribution')
data.kurt().head(10)

Kur_max=pd.DataFrame(data=(data.kurtosis()) , columns=['Kurtosis'])

Kur_max['var']=Kur_max.index

Kur_max.sort_values('Kurtosis', ascending=False).head()

features=data.drop(columns=['ID_code', 'target'])
correlations = data.corr().unstack().sort_values(ascending=True)

cor_abs=correlations.abs().reset_index()
cor_abs=cor_abs[cor_abs['level_0']!=cor_abs['level_1']]

cor_abs=cor_abs.set_axis(['level_0', 'level_1', 'cor'],axis=1, inplace=False)

cor_abs.tail(10)
corr=data.corr()

plt.figure(figsize=(17,12))



sns.heatmap(corr, cmap='coolwarm')
from sklearn.model_selection import train_test_split
train, test=train_test_split(data, test_size=0.25)
train.head()
x=train[train.columns[2:202]]

y_train=train[train.columns[1:2]]
xt=test[test.columns[2:202]]

y_test=test[test.columns[1:2]]
from sklearn import preprocessing

std=preprocessing.StandardScaler()
x_names=x.columns
x_tr=std.fit_transform(x)

x_train=pd.DataFrame(x_tr, columns=x_names)
xts=std.fit_transform(xt)

x_test=pd.DataFrame(xts, columns=x_names)
data[data.columns[2:102]].plot(kind='box', figsize=[15,4], title='Non standarized values')

x_train[x_train.columns[:100]].plot(kind='box', figsize=[15,4], title='Standarized values')

data[data.columns[103:]].plot(kind='box', figsize=[15,4], title='Non standarized values')

x_train[x_train.columns[101:]].plot(kind='box', figsize=[15,4], title='Standarized values')
values=data.columns.drop(['ID_code', 'target'])

plt.figure(figsize=(20,10))

for val in values:

    sns.distplot(data[val], hist=False)

plt.title('Density non Stadarized Data')

plt.xlabel('features')

plt.ylabel('density')



plt.figure(figsize=(20,10))

for val in values:

    sns.distplot(x_train[val], hist=False)

plt.title('Density Stadarized Data')

plt.xlabel('features')

plt.ylabel('density')

from sklearn.decomposition import PCA

#import mglearn
array=x_train.values

pca=PCA(n_components=3)

pca.fit(array)

threeD=pca.transform(array)

threeD

three_Df = pd.DataFrame(data = threeD, columns = ['PCA1', 'PCA2', 'PCA3']) 
df_pca = pd.concat([three_Df, y_train], axis = 1)
df_pca.head()
from mpl_toolkits.mplot3d import Axes3D

from matplotlib import animation


fig = plt.figure(figsize = (8,8))

ax = fig.add_subplot(111, projection='3d') 

ax.set_xlabel('PCA1', fontsize = 15)

ax.set_ylabel('PCA2', fontsize = 15)

ax.set_zlabel('PCA3', fontsize = 15)

ax.set_title('3 component PCA', fontsize = 20)

targets = [0,1]

colors = ['r','g']

for target, color in zip(targets,colors):

    indicesToKeep = df_pca['target'] == target

    ax.scatter(df_pca.loc[indicesToKeep, 'PCA1']

               , df_pca.loc[indicesToKeep, 'PCA2']

               , df_pca.loc[indicesToKeep, 'PCA3']

               , c = color

               , s = 50)



ax.legend(targets)

ax.grid()



ax.view_init(azim=50)



    

pca.explained_variance_ratio_
from sklearn.linear_model import Lasso
from scipy.stats import uniform
alphas=uniform.rvs(loc=0, scale=0.2, size=30)

alphas
alphas=[0.12301225, 0.14288355, 0.18551073, 0.05006723, 0.0333933 ,

       0.03646111, 0.04268822, 0.10610886, 0.19878154, 0.01463984,

       0.09548202, 0.13826288, 0.12977404, 0.06173418, 0.09480236,

       0.15044969, 0.05521685, 0.00238981, 0.13915425, 0.15324187,

       0.18726584, 0.0666834 , 0.01948747, 0.02757435, 0.13793408,

       0.09817728, 0.02072232, 0.1429758 , 0.11844789, 0.04484972]
from sklearn.model_selection import GridSearchCV

model=Lasso()

grid=GridSearchCV(estimator=model,param_grid=dict(alpha=alphas), cv=10)
grid.fit(x_train, y_train)
print('Best alpha--->', grid.best_params_)

print('Best score--->', grid.best_score_)
model_lasso=Lasso(alpha=0.00238981)

model_lasso.fit(x_train, y_train)
lasso_cf=list(model_lasso.coef_)

feature_names=x_train.columns.values.tolist()

coef_lasso=pd.DataFrame({'feature': feature_names, 'Coef':lasso_cf})

features_filter=coef_lasso[coef_lasso['Coef']!=0]

features_sel=features_filter['feature'].tolist()

print(features_sel)

len(features_sel)

x=train[train.columns[2:102]]

grid.fit(x, y_train)
print('Best alpha--->', grid.best_params_)

print('Best score--->', grid.best_score_)
model_lasso=Lasso(alpha=0.00238981)

model_lasso.fit(x, y_train)
lasso_cf=list(model_lasso.coef_)

feature_names=x.columns.values.tolist()

coef_lasso=pd.DataFrame({'feature': feature_names, 'Coef':lasso_cf})

features_filter=coef_lasso[coef_lasso['Coef']!=0]

features_sel=features_filter['feature'].tolist()

print(features_sel)

len(features_sel)
x_lasso_non=x_train[features_sel]

x_lasso_non.head()
xtest_lasso_non=x_test[features_sel]

x_lasso=x_train[features_sel]

x_lasso.columns
xtest_lasso=x_test[features_sel]

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix
log_reg=LogisticRegression()

log_reg1=LogisticRegression()

log_reg2=LogisticRegression()

log_reg3=LogisticRegression()

xtot_non=train[train.columns[2:202]]

xtot_non.head()
log_reg.fit(xtot_non,y_train)
log_reg1.fit(x_train,y_train)
log_reg2.fit(x_lasso , y_train)
log_reg3.fit(x_lasso_non , y_train)
y_pred=log_reg.predict(xt)
y_pred1=log_reg1.predict(x_test)
y_pred2=log_reg2.predict(xtest_lasso)
y_pred3=log_reg3.predict(xtest_lasso_non)
print('score of 200 features normalized----->', log_reg1.score(x_test,y_test))

print('score of 200 features NO normalized----->', log_reg.score(xt,y_test))



print('score of 48 features normalizer------>', log_reg2.score(xtest_lasso,y_test))

print('score of 48 features NO normalizer--->', log_reg3.score(xtest_lasso_non,y_test))
conf_matrix1=confusion_matrix(y_test, y_pred1)

conf_matrix2=confusion_matrix(y_test, y_pred2)

conf_matrix3=confusion_matrix(y_test, y_pred3)
tp1=conf_matrix1[0,0]+conf_matrix1[1,1]

tp2=conf_matrix2[0,0]+conf_matrix2[1,1]

tp3=conf_matrix3[0,0]+conf_matrix3[1,1]

fp1=conf_matrix1[0,1]+conf_matrix1[1,0]

fp2=conf_matrix2[0,1]+conf_matrix2[1,0]

fp3=conf_matrix3[0,1]+conf_matrix3[1,0]
print('True predictions 200 features normalized---->',tp1)

print('True predictions 48 features normalized----->',tp2)

print('True predictions 48 features NO normalized-->',tp3)



print('False predictions 200 features normalized--->',fp1)

print('False predictions 48 features normalized---->',fp2)

print('False predictions 48 features NO normalized->',fp3)







import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras import optimizers

from tensorflow.keras.utils import to_categorical

from tensorflow.keras import backend as k

from tensorflow.keras.models import model_from_json







features=xtot_non.shape[1]
data=xtot_non.as_matrix()

lab=y_train.as_matrix()

label=to_categorical(lab)

data_test=xt.as_matrix()

lab_test=y_test.as_matrix()

label_test=to_categorical(lab_test)



model=tf.keras.Sequential()

k.clear_session()



model.add(layers.Dense(400, activation='relu', input_shape=(features,)))

model.add(layers.Dense(400, activation='relu'))

model.add(layers.Dense(200, activation='relu'))

model.add(layers.Dense(100, activation='relu'))

model.add(layers.Dense(2, activation='softmax'))



model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])



model.fit(data, label, epochs=25, batch_size=512)
model.evaluate(data_test, label_test, batch_size=512)
features1=x_train.shape[1]
data1=x_train.as_matrix()

lab=y_train.as_matrix()

label=to_categorical(lab)
data1_test=x_test.as_matrix()

lab_test=y_test.as_matrix()

label_test=to_categorical(lab_test)
model1=tf.keras.Sequential()
k.clear_session()
model1.add(layers.Dense(800, activation='relu', input_shape=(features1,)))

model1.add(layers.Dense(800, activation='relu'))

model1.add(layers.Dense(400, activation='relu'))

model1.add(layers.Dense(200, activation='relu'))

model1.add(layers.Dense(2, activation='softmax'))
model1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model1.fit(data1,label, epochs=300, batch_size=512)
model1.evaluate(data1_test, label_test, batch_size=512)
model_js=model1.to_json()

with open("model.json1", "w") as json_file:

    json_file.write(model_js)

# serialize weights to HDF5

model1.save_weights("model1.h5")

print("Saved model to disk")
features2=x_lasso.shape[1]



data2=x_lasso.as_matrix()

data2_test=xtest_lasso.as_matrix()



model2=tf.keras.Sequential()

k.clear_session()



model2.add(layers.Dense(400, activation='relu', input_shape=(features2,)))

model2.add(layers.Dense(400, activation='relu'))

model2.add(layers.Dense(200, activation='relu'))

model2.add(layers.Dense(100, activation='relu'))

model2.add(layers.Dense(2, activation='softmax'))



model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])



model2.fit(data2, label, epochs=25, batch_size=512)
model2.evaluate(data2_test, label_test, batch_size=512)
model_js=model2.to_json()

with open("model.json2", "w") as json_file:

    json_file.write(model_js)

# serialize weights to HDF5

model2.save_weights("model2.h5")

print("Saved model to disk")
features3=x_lasso_non.shape[1]



data3=x_lasso_non.as_matrix()

data3_test=xtest_lasso_non.as_matrix()



model3=tf.keras.Sequential()

k.clear_session()



model3.add(layers.Dense(400, activation='relu', input_shape=(features3,)))

model3.add(layers.Dense(400, activation='relu'))

model3.add(layers.Dense(200, activation='relu'))

model3.add(layers.Dense(100, activation='relu'))

model3.add(layers.Dense(2, activation='softmax'))



model3.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])



model3.fit(data3, label, epochs=25, batch_size=512)
model3.evaluate(data3_test, label_test, batch_size=512)
model_js=model3.to_json()

with open("model.json3", "w") as json_file:

    json_file.write(model_js)

# serialize weights to HDF5

model3.save_weights("model3.h5")

print("Saved model to disk")
from sklearn.ensemble import RandomForestClassifier as rfc

model=rfc(n_jobs=2, random_state=0)

model.fit(x_train, y_train)
x_train.shape
from sklearn.metrics import accuracy_score

accuracy_score(y_train, model.predict(x_train))
y_pred=model.predict(xt)

accuracy_score(y_test, y_pred)


x_pred=data_pred[data_pred.columns[1:201]]

x_var=x_pred.columns


x_norm=std.fit_transform(x_pred)

x_norm=pd.DataFrame(x_norm, columns=x_var)
prediction=log_reg1.predict(x_norm)
prediction=pd.DataFrame(data=prediction , columns=['target'])

prediction.head()
ID_code=[]



for i in range(len(prediction)):

    s=str(i)

#    t=str(prediction[i])

    line='test_'+s

    ID_code.append(line)

    
ID=pd.DataFrame(data=ID_code, columns=['ID_code'])
ID.head()
ID['target']=prediction.target

ID['target'].hist()
p=ID['target'].value_counts()

print('% of 0 ---> ', p[0]/(p[0]+p[1]))

print('% of 1 ---> ', p[1]/(p[0]+p[1]))
ID.to_csv('submission.csv', index=False)