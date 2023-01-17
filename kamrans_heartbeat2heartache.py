# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns

sns.set()

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import MinMaxScaler

from sklearn.svm import SVC

from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.neural_network import MLPClassifier

from tensorflow.keras import Sequential

from tensorflow.keras.layers import Dense, Dropout

from tensorflow.keras.callbacks import EarlyStopping
df = pd.read_csv('/kaggle/input/heart-disease-uci/heart.csv')
df.info()
df.describe().T
sns.countplot(df['target'])
df['age'].plot(kind='hist')
age_tgt_0 = df[df['target'] == 0]['age']

age_tgt_1 = df[df['target'] == 1]['age']



#f, axes = plt.subplots(1, 2, figsize=(10, 4))



sns.distplot(age_tgt_0, label='age category 0')

sns.distplot(age_tgt_1, label='age category 1')

plt.legend()

plt.show()
# sex - (1 = male; 0 = female)



sns.countplot(df['sex'])
# cp - chest pain type



sns.countplot(df['cp'])
# trestbps - resting blood pressure (in mm Hg on admission to the hospital)



df['trestbps'].plot(kind='hist')
# chol - serum cholestoral in mg/dl



df['chol'].plot(kind='hist')
# fbs - (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)



sns.countplot(df['fbs'])
# restecg - resting electrocardiographic results



sns.countplot(df['restecg'])
# thalach - maximum heart rate achieved



df['thalach'].plot(kind='hist')
# exang - exercise induced angina (1 = yes; 0 = no)



sns.countplot(df['exang'])
# oldpeak - ST depression induced by exercise relative to rest



df['oldpeak'].plot(kind='hist')
# slop - ethe slope of the peak exercise ST segment



sns.countplot(df['slope'])
# ca - number of major vessels (0-3) colored by flourosopy



sns.countplot(df['ca'])
# thal - 3 = normal; 6 = fixed defect; 7 = reversable defect



sns.countplot(df['thal'])
# Exploring gender differences



male_df = df[df['sex'] == 1]

female_df = df[df['sex'] == 0]
bins = [25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]

male_df['age'].plot(kind='hist', label='male age', bins=bins)

female_df['age'].plot(kind='hist', label='female age', bins=bins)

plt.legend()
# Males get more heartache



sns.countplot(x='cp', data=df, hue='sex')

#sns.countplot(female_df['cp'], label='chest pain femaile')

plt.legend(labels=['Female', 'Male'])

plt.show()
figure, axs = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(10, 6))

sns.distplot(male_df['trestbps'], ax=axs[0])

axs[0].legend(['Male BP'])

sns.distplot(female_df['trestbps'], ax=axs[1], )

axs[1].legend(['Female BP'])

axs[0].set_title('BP By Sex', size=16)

plt.show()
bins = np.linspace(0, 600, 25, dtype='int')



fig, axs = plt.subplots(2, figsize=(10, 6), sharex=True)



sns.distplot(male_df['chol'], ax=axs[0], bins=bins)

axs[0].legend(['Male Cholestrol'])

axs[0].set_xlabel('')



sns.distplot(female_df['chol'], ax=axs[1], bins=bins)

axs[0].legend(['Female Cholestrol'])

axs[1].set_xlabel('serum cholestoral in mg/dl')
sns.countplot(x='fbs', hue='sex', data=df)

plt.legend(['Female', 'Male'])

plt.xlabel('fasting blood sugar > 120 mg/dl \n 1 = true; 0 = false')
sns.countplot(x='restecg', hue='sex', data=df)

plt.legend(['Female', 'Male'])

plt.xlabel('resting electrocardiographic results')
bins = np.linspace(50, 220, 18)



fig, axs = plt.subplots(2, sharex=True, sharey=True, figsize=(10, 6))



sns.distplot(male_df['thalach'], ax=axs[0], bins=bins)

axs[0].legend(['Male Max HR'])

axs[0].set_xlabel('')



sns.distplot(female_df['thalach'], ax=axs[1], bins=bins)

axs[1].legend(['Female Max HR'])

axs[1].set_xlabel('maximum heart rate achieved')
sns.countplot(x='exang', hue='sex', data=df)

plt.legend(['Females', 'Males'])

plt.xlabel('exercise induced angina \n (1 = yes; 0 = no)')
fig, axs = plt.subplots(2, sharex=True, sharey=True)



sns.distplot(female_df['oldpeak'], ax=axs[0])

sns.distplot(male_df['oldpeak'], ax=axs[1])

axs[0].legend(['Females'])

axs[1].legend(['Males'])

axs[0].set_xlabel('')

axs[1].set_xlabel('ST depression induced by exercise relative to rest')

plt.show()
sns.countplot(x='slope', hue='sex', data=df)

plt.legend(['Females', 'Males'])

plt.xlabel('the slope of the peak exercise ST segment')
ax = sns.countplot(x='ca', hue='sex', data=df)

ax.legend(['Female', 'Male'])

ax.set_xlabel('number of major vessels (0-3) colored by flourosopy')
ax = sns.countplot(x='thal', hue='sex', data=df)

ax.legend(['Female', 'Male'])

ax.set_xlabel('3 = normal; 6 = fixed defect; 7 = reversable defect')
# Converting categorical variables to data type string, so onehotencoding 

# can be applied or dummy variables can be created



df[['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']] = df[['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']].astype('str')



df_dumms = pd.get_dummies(df, drop_first=True)
X = df_dumms.drop(columns='target').values

y = df_dumms['target'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
logreg = LogisticRegression()



grid = {'penalty' : ['l1', 'l2', 'elasticnet', 'none'], 

       'C' : [1, 10, 100, 1000], 'solver' : ['liblinear', 'lbfgs']}



gsearch = GridSearchCV(estimator=logreg, param_grid=grid, n_jobs=-1, 

                       cv=5, verbose=True)
gsearch.fit(X_train, y_train)
gsearch.best_params_
logestic_regression = LogisticRegression(**gsearch.best_params_)



logestic_regression.fit(X_train, y_train)



log_reg_preds = logestic_regression.predict(X_test)
log_reg_cm = confusion_matrix(y_test, log_reg_preds)

sns.heatmap(log_reg_cm, annot=True, annot_kws={'size':24})
log_reg_accu = accuracy_score(y_test, log_reg_preds)

print('The Logestic Regression model achieved an accuracy score of {}'.format(log_reg_accu))
gnb = GaussianNB()

gnb.fit(X_train, y_train)

gnb_preds = gnb.predict(X_test)
gnb_cm = confusion_matrix(y_test, gnb_preds)



sns.heatmap(gnb_cm, annot=True, annot_kws={'size':24})
gnb_accu = accuracy_score(y_test, gnb_preds)

print('The Gaussian NB Classifier accuracy score is {}'.format(gnb_accu))
dtree = DecisionTreeClassifier()



dtree_grid = {'criterion' : ['gini', 'entropy'], 'splitter' : ['best', 'random'], 

             'min_samples_split' : [20, 25, 30, 35, 40, 50]}



gsearch = GridSearchCV(dtree, param_grid=dtree_grid, n_jobs=-1, cv=5, verbose=True)



gsearch.fit(X_train, y_train)
dec_tree = DecisionTreeClassifier(**gsearch.best_params_)



dec_tree.fit(X_train, y_train)



dtree_preds = dec_tree.predict(X_test)
dtree_cm = confusion_matrix(y_test, dtree_preds)



sns.heatmap(dtree_cm, annot=True, annot_kws={'size':24})
dtree_acc = accuracy_score(y_test, dtree_preds)

print('The Decesion Tree model yielded an accuracy of {}'.format(dtree_acc))
rf = RandomForestClassifier()



grid = {'n_estimators' : [18, 20, 22], 

       'criterion' : ['gini'], 'min_samples_split' : [8, 10, 12, 15], 

       'min_samples_leaf' : [7, 8, 9], 'max_depth' : [12, 15, 18], 

       'max_features' : [1, 2]}



gridsearch = GridSearchCV(estimator=rf, param_grid=grid, n_jobs=-1, cv=5, verbose=3)



gridsearch.fit(X_train, y_train)
rf_clf = RandomForestClassifier(**gridsearch.best_params_)



rf_clf.fit(X_train, y_train)



rf_preds = rf_clf.predict(X_test)
rf_cm = confusion_matrix(y_test, rf_preds)



sns.heatmap(rf_cm, annot=True, annot_kws={'size':24})
rf_accu = accuracy_score(y_test, rf_preds)



print('The Random Forest Classifier model yielded an accuracy of {}'.format(rf_accu))
mm_scaler = MinMaxScaler()

mm_scaler.fit(X_train)

X_train_scaled = mm_scaler.transform(X_train)

X_test_scaled = mm_scaler.transform(X_test)
knn = KNeighborsClassifier()



knn_grid = {'n_neighbors' : [13, 15, 17, 19],

            'weights' : ['distance'],

            'algorithm' : ['auto'],

            'leaf_size' : [2, 3, 5, 7, 9],

            'metric' : ['manhattan']}



gs = GridSearchCV(estimator=knn, param_grid=knn_grid, n_jobs=-1, verbose=3)



gs.fit(X_train_scaled, y_train)
knn_clf = KNeighborsClassifier(**gs.best_params_)

knn_clf.fit(X_train_scaled, y_train)

knn_preds = knn_clf.predict(X_test_scaled)
knn_cm = confusion_matrix(y_test, knn_preds)



sns.heatmap(knn_cm, annot=True, annot_kws={'size':24})
knn_acc = accuracy_score(y_test, knn_preds)



print('KNN accuracy score is {}'.format(knn_acc))
svc = SVC()



svc_grid = {'C' : [0.1, 1.0, 10.0],

            'kernel' : ['rbf', 'linear'],

            'gamma' : [0.00001, 0.0001, 0.001, 0.01]}



svc_gs = GridSearchCV(estimator=svc, param_grid=svc_grid, n_jobs=-1, verbose=3)



svc_gs.fit(X_train, y_train)
svm_clf = SVC(**svc_gs.best_params_)



svm_clf.fit(X_train, y_train)



svm_preds = svm_clf.predict(X_test)
svm_cm = confusion_matrix(y_test, svm_preds)



sns.heatmap(svm_cm, annot=True, annot_kws={'size':24})
svm_accu = accuracy_score(y_test, svm_preds)



print('The SVM Classifier achieved an accuracy score of {}'.format(svm_accu))
gpc = GaussianProcessClassifier()

gpc.fit(X_train, y_train)



gpc_preds = gpc.predict(X_test)
gpc_cm = confusion_matrix(y_test, gpc_preds)



sns.heatmap(gpc_cm, annot=True, annot_kws={'size' : 24})
gpc_accu = accuracy_score(y_test, gpc_preds)



print('The GP Classifier had an accuracy score of {}'.format(gpc_accu))
mlp = MLPClassifier()



mlp_grid = {'hidden_layer_sizes' : [(7, 3), (8, 4, 2), (14, 7, 3), (10,), (7,), (4,)],

            'activation' : ['identity', 'logistic', 'tanh', 'relu'],

            'solver' : ['lbfgs', 'sgd', 'adam'],

            'alpha' : [0.00001, 0.0001, 0.001],

            'learning_rate' : ['constant', 'invscaling', 'adaptive']}



mlp_gs = GridSearchCV(estimator=mlp, param_grid=mlp_grid, n_jobs=-1, verbose=4)



mlp_gs.fit(X_train_scaled, y_train)
mlp_cls = MLPClassifier(**mlp_gs.best_params_)



mlp_cls.fit(X_test_scaled, y_test)



mlp_preds = mlp_cls.predict(X_test_scaled)
mlp_cm = confusion_matrix(y_test, mlp_preds)



sns.heatmap(mlp_cm, annot=True, annot_kws={'size' : 24})
mlp_accu = accuracy_score(y_test, mlp_preds)



print('MLPClassifier achieved an accuracy score of {}'.format(mlp_accu))
model = Sequential()



model.add(Dense(units=12, activation='relu'))

model.add(Dropout(0.25))



model.add(Dense(units=4, activation='relu'))

#model.add(Dropout(0.25))



model.add(Dense(units=1, activation='sigmoid'))



model.compile(optimizer='adam', loss='binary_crossentropy')



early_stopping = EarlyStopping(monitor='val_loss', patience=10)



model.fit(x=X_train_scaled, y=y_train, epochs=1000, callbacks=[early_stopping], 

          validation_data=(X_test_scaled, y_test))
loss = pd.DataFrame(model.history.history)

loss.plot()
ann_preds = model.predict_classes(X_test_scaled)
ann_cm = confusion_matrix(y_test, ann_preds)



sns.heatmap(ann_cm, annot=True, annot_kws={'size' : 24})
ann_accu = accuracy_score(y_test, ann_preds)



print('The TF Keras ANN Model achieved an accuracy score of {}'.format(ann_accu))
# Classifiers compared



cls_scores = pd.DataFrame(data= [log_reg_accu, gnb_accu, dtree_acc, rf_accu, knn_acc,

                    svm_accu, gpc_accu, mlp_accu, ann_accu],

             index= ['Logistic Cls', 'GaussianNP Cls', 'DecisionTree Cls',

                     'RandomForest Cls', 'KNNeighbours Cls', 'SVM Cls',

                     'GaussianProcess Cls', 'MLP Cls', 'ANNs'], 

             columns= ['Accuracy Scores'])
cls_scores = cls_scores.sort_values(by='Accuracy Scores', ascending=False)

cls_scores
cls_scores.plot(kind='bar', figsize=(10, 4))