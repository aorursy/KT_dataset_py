import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import warnings  

warnings.filterwarnings('ignore')
iris_data = pd.read_csv('../input/Iris.csv', index_col='Id')

iris_data.head()
iris_data.isnull().sum()
iris_data.dtypes
from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import confusion_matrix,accuracy_score
X = iris_data.iloc[:,:-1].values

Y = iris_data.iloc[:,-1:].values

X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size=0.33,random_state=0)
lr = LogisticRegression()

lr.fit(X_train,Y_train)

lr_predict = lr.predict(X_test)

lr_confusion = confusion_matrix(Y_test,lr_predict)

print(lr_confusion)

lr_acc = accuracy_score(Y_test,lr_predict)

print(lr_acc)
sc = StandardScaler()

X_train_scaled = sc.fit_transform(X_train)

X_test_scaled = sc.transform(X_test)



knn = KNeighborsClassifier(n_neighbors=2)

knn.fit(X_train_scaled,Y_train)

knn_predict = knn.predict(X_test_scaled)

knn_confusion = confusion_matrix(Y_test,knn_predict)

print(knn_confusion)

knn_acc = accuracy_score(Y_test,knn_predict)

print(knn_acc)
svc = SVC(kernel='rbf')

svc.fit(X_train,Y_train)

svc_rbf_predict = svc.predict(X_test)

svc_rbf_confusion = confusion_matrix(Y_test,svc_rbf_predict)

print(svc_rbf_confusion)

svc_rbf_acc = accuracy_score(Y_test,svc_rbf_predict)

print(svc_rbf_acc)
svc_linear = SVC(kernel='linear')

svc_linear.fit(X_train,Y_train)

svc_linear_predict = svc_linear.predict(X_test)

svc_linear_confusion = confusion_matrix(Y_test,svc_linear_predict)

print(svc_linear_confusion)

svc_linear_acc = accuracy_score(Y_test,svc_linear_predict)

print(svc_linear_acc)
svc_poly = SVC(kernel='poly')

svc_poly.fit(X_train,Y_train)

svc_poly_predict = svc_poly.predict(X_test)

svc_poly_confusion = confusion_matrix(Y_test,svc_poly_predict)

print(svc_poly_confusion)

svc_poly_acc = accuracy_score(Y_test,svc_poly_predict)

print(svc_poly_acc)
gaussian = GaussianNB()

gaussian.fit(X_train,Y_train)

gaussian_predict = gaussian.predict(X_test)

gaussian_confusion = confusion_matrix(Y_test,gaussian_predict)

print(gaussian_confusion)

gaussian_acc = accuracy_score(Y_test,gaussian_predict)

print(gaussian_acc)
dtc = DecisionTreeClassifier(criterion='entropy')

dtc.fit(X_train,Y_train)

dtc_predict = dtc.predict(X_test)

dtc_confusion = confusion_matrix(Y_test,dtc_predict)

print(dtc_confusion)

dtc_acc = accuracy_score(Y_test,dtc_predict)

print(dtc_acc)
dtc_gini = DecisionTreeClassifier(criterion='gini')

dtc_gini.fit(X_train,Y_train)

dtc_gini_predict = dtc_gini.predict(X_test)

dtc_gini_confusion = confusion_matrix(Y_test,dtc_gini_predict)

print(dtc_gini_confusion)

dtc_gini_acc = accuracy_score(Y_test,dtc_gini_predict)

print(dtc_gini_acc)
rf = RandomForestClassifier(criterion='entropy',n_estimators=10)

rf.fit(X_train,Y_train)

rf_predict = rf.predict(X_test)

rf_confusion = confusion_matrix(Y_test,rf_predict)

print(rf_confusion)

rf_acc = accuracy_score(Y_test,rf_predict)

print(rf_acc)
list_acc = [lr_acc,knn_acc,svc_linear_acc,svc_poly_acc,svc_rbf_acc,gaussian_acc,dtc_acc,dtc_gini_acc,rf_acc]

column_names = ['linear','knn','svc_linear','svc_poly','svc_rbf','gaussian','decission tree','decision tree gini','random forest']

data_acc = pd.DataFrame({'models' : column_names,'accuracy' : list_acc })

data_acc
sns.catplot(x='models',y='accuracy',data= data_acc,kind='bar')
sns.catplot(x='models',y='accuracy',data=data_acc,kind='point')

plt.title('accuracy for models')
kfold_models = [lr.fit(X_train,Y_train),knn.fit(X_train_scaled,Y_train),svc.fit(X_train,Y_train),svc_linear.fit(X_train,Y_train),svc_poly.fit(X_train,Y_train),gaussian.fit(X_train,Y_train),dtc.fit(X_train,Y_train),dtc_gini.fit(X_train,Y_train),rf.fit(X_train,Y_train)]

cross_score_mean = []

cross_score_std = []

for model in kfold_models:

    cvr = cross_val_score(estimator=model,X=X,y=Y.ravel(),cv=5)

    cross_score_mean.append(cvr.mean())

    cross_score_std.append(cvr.std())

cross_score_for_models = pd.DataFrame({'models' : column_names, 'cross_val_score_mean' : cross_score_mean,'cross_val_score_std' : cross_score_std})

cross_score_for_models
sns.relplot(x='cross_val_score_std',y='cross_val_score_mean', data=cross_score_for_models,kind='scatter')

for i in range(len(cross_score_for_models)):

  plt.text(cross_score_for_models.iloc[i,2],cross_score_for_models.iloc[i,1],cross_score_for_models.iloc[i,0])
params = [{'C' : [10,12,13,14,15],'kernel' : ['rbf'], 'gamma':[0.1,0.2,0.3,0.4]}]

gscv = GridSearchCV(estimator=svc,param_grid=params,scoring='accuracy',cv=5)

success = gscv.fit(X_train,Y_train.ravel())

print(success.best_score_)

print(success.best_params_)
last_model = SVC(C=10,gamma=0.1,kernel='rbf')

last_model.fit(X_train,Y_train)

last_predict = last_model.predict(X_test)

predict_data = pd.DataFrame(X_test, columns=["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"])

predict_data['Species'] = last_predict

predict_data.head(3)
test_data = pd.DataFrame(X_test, columns=["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"])

test_data['Species'] = Y_test

test_data.head(3)
sns.relplot(x='SepalLengthCm',y='SepalWidthCm',hue='Species',data=predict_data)

sns.relplot(x='SepalLengthCm',y='SepalWidthCm',hue='Species',data=test_data)