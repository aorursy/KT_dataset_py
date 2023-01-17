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
df_heartdisease = pd.read_csv("/kaggle/input/heart-disease-uci/heart.csv")
df_heartdisease.shape
df_heartdisease.head()
df_heartdisease.describe()
df_heartdisease.dtypes
df_heartdisease.isna().sum()
df_heartdisease['target'].value_counts()
num_features = ['age','trestbps','chol','thalach','oldpeak']

cat_features = ['sex','cp','fbs','restecg','exang','slope','ca','thal','target']
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



sns.heatmap(df_heartdisease[num_features].corr(method='pearson'),annot=True)

plt.figure(figsize=(25,20))
def cat_univariate_eda(df, cat_col_names, n_cols = 3, figsize = None, save_fig = False):

    

    # univariate eda for categorical features

    n_rows = len(cat_col_names)/n_cols

    

    plt.figure(figsize=(5*n_cols,5*n_rows))

    

    if figsize:

        plt.figure(figsize=figsize)

    

    for i in range(0,len(cat_col_names)):

        plt.subplot(n_rows+1, n_cols, (i+1))

        sns.countplot(df[cat_col_names[i]],hue=df['target'])

    

    if save_fig:

        plt.savefig('./cat_col_eda1.png')

    

    plt.show()
cat_univariate_eda(df_heartdisease,cat_features, save_fig = False)
sns.boxplot(df_heartdisease['target'],df_heartdisease['age'])
sns.boxplot(df_heartdisease['target'],df_heartdisease['trestbps'])
sns.boxplot(df_heartdisease['target'],df_heartdisease['chol'])
from sklearn.model_selection import train_test_split

X=pd.DataFrame(df_heartdisease[['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']])

y=pd.DataFrame(df_heartdisease['target'])

print(X.shape)

print(y.shape)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)
from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2



#apply SelectKBest class to extract top 10 best features

bestfeatures = SelectKBest(score_func=chi2, k=10)

fit = bestfeatures.fit(X,y)

dfscores = pd.DataFrame(fit.scores_)

dfcolumns = pd.DataFrame(X.columns)

#concat two dataframes for better visualization 

featureScores = pd.concat([dfcolumns,dfscores],axis=1)

featureScores.columns = ['Specs','Score']  #naming the dataframe columns

print(featureScores.sort_values(by='Score')) 
from sklearn.linear_model import LogisticRegression

lmmodel = LogisticRegression()

lmmodel.fit(X_train,y_train)
print(lmmodel.score(X_train,y_train))

print(lmmodel.score(X_test,y_test))
lm_predictions = lmmodel.predict(X_test)

print(lm_predictions)
from sklearn.metrics import confusion_matrix,r2_score,roc_auc_score,roc_curve

confusion_matrix(lm_predictions,y_test)
roc_auc = roc_auc_score(y_test,lm_predictions)

print(roc_auc)
def plot_curve(model, X_test, y_test,score, model_label):

    

    # function to plot roc curve for the given model

    y_score = pd.DataFrame(model.predict_proba(X_test))[1]

    fpr,tpr, threshold = roc_curve(y_test, y_score)

    plt.figure()

    lw = 2

    plt.plot(fpr, tpr, color='darkorange', label='{} {}'.format(model_label,np.round(score,2)))

    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')

    plt.xlim([0.0, 1.0])

    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.title('ROC Curve')

    plt.legend(loc="lower right")
plot_curve(lmmodel, X_test, y_test,roc_auc, 'Logistic')
lmmodel_featureselect = LogisticRegression()

X_train_featureselect = X_train[['thalach','oldpeak','ca','cp','exang','chol','age','trestbps','slope','sex']]

X_test_featureselect = X_test[['thalach','oldpeak','ca','cp','exang','chol','age','trestbps','slope','sex']]

lmmodel_featureselect.fit(X_train_featureselect,y_train)

print(lmmodel_featureselect.score(X_train_featureselect,y_train))

print(lmmodel_featureselect.score(X_test_featureselect,y_test))
lmfeatures_predictions = lmmodel_featureselect.predict(X_test_featureselect)

roc_auc_features = roc_auc_score(y_test,lmfeatures_predictions)

print(roc_auc_features)
from sklearn.tree import DecisionTreeClassifier

clf_params = DecisionTreeClassifier(random_state=3)

clf_params.fit(X_train,y_train)

print(clf_params.score(X_train,y_train))

print(clf_params.score(X_test,y_test))
# Optimizing the Decision tree to reduce overfitting problem



clf = DecisionTreeClassifier(max_depth=3,min_samples_split=10,random_state=3)

clf.fit(X_train,y_train)

print(clf.score(X_train,y_train))

print(clf.score(X_test,y_test))
#Finding the best hyper parameters with GridSearchCV 

from sklearn.model_selection import GridSearchCV

parameters={'min_samples_split' : np.arange(10,20),'max_depth': np.arange(3,8),'max_features': np.arange(6,12)}

clf_tree=DecisionTreeClassifier(random_state=2)

clf=GridSearchCV(clf_tree,parameters,cv=10, scoring='accuracy')

clf.fit(X,y)

print(clf.best_score_)

print(clf.best_params_)

print(clf.best_estimator_)
print(clf.score(X_train,y_train))

print(clf.score(X_test,y_test))
from xgboost import XGBClassifier

xgb = XGBClassifier(random_state=3)

xgb.fit(X_train,y_train)

print(xgb.score(X_train,y_train))

print(xgb.score(X_test,y_test))
# As above model is overfitting the data so using GridSearchCV

parameters_xgb={'eta' : [0.001, 0.01, 0.1],'min_child_weight': np.arange(3,8,2),'subsample':[0.5,0.6,0.7,0.8] ,'max_depth': np.arange(2, 8),'colsample_bytree': [0.5,0.6,0.7,0.8,0.9]}

xgb_tree=XGBClassifier(seed=5,n_jobs=-1,n_estimators=500)

xgb_params=GridSearchCV(xgb_tree,parameters_xgb,cv=10, scoring='accuracy')

xgb_params.fit(X,y)

print(xgb_params.best_score_)

print(xgb_params.best_params_)

print(xgb_params.best_estimator_)
print(xgb_params.score(X_train,y_train))

print(xgb_params.score(X_test,y_test))
predict_XGB = xgb_params.predict(X_test)

confusion_matrix(predict_XGB,y_test)
roc_auc_XGB = roc_auc_score(y_test,predict_XGB)

print(roc_auc_XGB)
from sklearn.neighbors import KNeighborsClassifier



accuracies = []



K=list(range(1,100,2))

for i in K:

    KNNclass = KNeighborsClassifier(n_neighbors=i)

    KNNclass.fit(X_train,y_train)

    trainacc = np.mean(KNNclass.predict(X_train)==y_train['target'])

    testacc = np.mean(KNNclass.predict(X_test)==y_test['target'])

    accuracies.append([trainacc,testacc])

    



plt.plot(K,[i[0] for i in accuracies],"bo-")



plt.plot(K,[i[1] for i in accuracies],"ro-")
import keras

from keras.models import Sequential

from keras.layers import Dense
tensorflow_trainX = X_train.values

tensorflow_testX = X_test.values

tensorflow_trainy = y_train.values

tensorflow_testy = y_test.values
model = Sequential()

model.add(Dense(20, input_dim=13, activation='relu'))

model.add(Dense(10, activation='relu'))

model.add(Dense(1, activation='sigmoid'))
model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error','accuracy'])

model.summary()
model.fit(tensorflow_trainX, tensorflow_trainy, epochs=150, batch_size=100, validation_split = 0.2)
model_1 = Sequential()

model_1.add(Dense(20, input_dim=13, activation='relu'))

model_1.add(Dense(20, activation='relu'))

model_1.add(Dense(20, activation='relu'))

model_1.add(Dense(20, activation='relu'))

model_1.add(Dense(20, activation='relu'))

model_1.add(Dense(20, activation='relu'))

model_1.add(Dense(1, activation='sigmoid'))
model_1.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error','accuracy'])

model_1.summary()
model_1.fit(tensorflow_trainX, tensorflow_trainy, epochs=800, batch_size=150)
predict_ANN = model_1.predict_classes(tensorflow_testX)

confusion_matrix(predict_ANN,tensorflow_testy)
roc_auc_ANN = roc_auc_score(tensorflow_testy,predict_ANN)

print(roc_auc_ANN)
# As we can observe XGBoost is having higher accuracy, saving and deploying the model

import pickle



# Save the trained model as a pickle string. 

saved_model = pickle.dumps(xgb_params) 

  

# Load the pickled model 

from_pickle = pickle.loads(saved_model) 

  

# Use the loaded pickled model to make predictions 

from_pickle.predict(X_test) 