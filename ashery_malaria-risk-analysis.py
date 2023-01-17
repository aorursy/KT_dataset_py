from __future__ import print_function



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from matplotlib.pylab import rcParams

%matplotlib inline

import seaborn as sns

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import AdaBoostClassifier

import xgboost as xgb

from xgboost.sklearn import XGBClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.metrics import make_scorer,accuracy_score

from sklearn.model_selection import GridSearchCV

from sklearn.base import TransformerMixin

from sklearn.preprocessing import StandardScaler,LabelEncoder

from sklearn.model_selection import train_test_split

import lime

import lime.lime_tabular









import os

print(os.listdir("../input"))



#loading and showing a sample from the dataset

data = pd.read_csv('../input/malaria_risk_dataset.csv')

data = data.drop(columns="location_extId")

data.sample(3)
#rcParams['figure.figsize'] = 12,4

sns.set(style="whitegrid",rc={'figure.figsize':(20,10),'axes.labelsize':16})

sns.barplot(x="wallstructure", y="nbofrooms", hue="RDT_test_result", data=data)
data.RDT_test_result.value_counts()/data.RDT_test_result.count()
class DataFrameImputer(TransformerMixin):

    def __init__(self):

        """Imputer missing values.

        -Columns of dtype are imputed with the most frequent value(mode) in column.

        -Columns of other type are imputed with the mean of the column"""

    

    def fit(self, X,y=None):

        self.fill = pd.Series([X[c].value_counts().index[0] 

                           if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],

                          index = X.columns)

        return self

    

    def transform(self,X,y=None):

        return X.fillna(self.fill)

data = DataFrameImputer().fit_transform(data)
feature_names = list(data.columns)

feature_names = feature_names[:-1]
#data preproccessing



#scaler = StandardScaler()

encoder = LabelEncoder()

    

#Standardizing the continous variables (nbofrooms & nbofroomsforsleeping)

#data[data.columns[:2].tolist()] = scaler.fit_transform(data[data.columns[:2].tolist()])



values = data.values

data = values[:,:-1]

labels = values[:,-1]

    

#encode the categorical values

categorical_features = [2,3,4,5,6,7,8,9,10]

categorical_names = {}



for feature in categorical_features:

    le = encoder.fit(data[:,feature])

    data[:,feature] = le.transform(data[:,feature])

    categorical_names[feature] = le.classes_



#One-Hot Encoding

#data = pd.get_dummies(data, columns = features_to_encode)

#X = data[data.columns[:-1]]

#y = data[data.columns[-1]]



data = data.astype(float)

labels = labels.astype(int)

    

X_train,X_test,y_train,y_test = train_test_split(data, labels, test_size =0.20,random_state = 2)
acc_score = make_scorer(accuracy_score)



#Model 1 - Decision Tree

model_dt= DecisionTreeClassifier(class_weight = 'balanced')

model_dt.fit(X_train,y_train)

model_dt_accuracy = accuracy_score(y_test,model_dt.predict(X_test))





#Model 2 - Random Forest Classifier

model_rf = RandomForestClassifier()

    

parameters_rf = {'n_estimators':[10,20,30,40,50,100,200],

                 'max_features':['log2','sqrt','auto'],

                 'criterion':['entropy','gini'],

                 'max_depth':[2,3,4],

                 'min_samples_split':[2,3,4],

                 'min_samples_leaf':[2,3,4]

                 }

    

grid_obj_rf = GridSearchCV(model_rf, parameters_rf, scoring = acc_score,cv = 5)

grid_obj_rf = grid_obj_rf.fit(X_train, y_train)

model_rf = grid_obj_rf.best_estimator_

#print("the best estimator values are: ",model_rf)

model_rf.fit(X_train,y_train)

model_rf_accuracy = accuracy_score(y_test, model_rf.predict(X_test))

feature_importances_rf = model_rf.feature_importances_





#Model 3 - AdaBoost Classifier

model_adaboost = AdaBoostClassifier()

    

parameters_adaboost = {'base_estimator':[model_dt],

                       'learning_rate':[k for k in np.arange(0.001,1,0.01)],

                       'n_estimators':[k for k in range(10,100,10)]}

    

grid_obj_adaboost = GridSearchCV(model_adaboost, parameters_adaboost, scoring = acc_score,cv=5)

grid_obj_adaboost = grid_obj_adaboost.fit(X_train, y_train)

model_adaboost = grid_obj_adaboost.best_estimator_

#print("the best estimator values are: ",model_adaboost)

model_adaboost.fit(X_train,y_train)

model_adaboost_accuracy = accuracy_score(y_test, model_adaboost.predict(X_test))

feature_importances_adaboost = model_adaboost.feature_importances_

  

#Model 4 - XGBoost Classifier

#def xgboost(X_train,X_test,y_train,y_test,useTrainCV=True,cv_folds=5,early_stopping_rounds = 50):

useTrainCV = True



model_xgboost = XGBClassifier(learning_rate = 0.1, n_estimators=1000, max_depth = 5, 

                              min_child_weight = 1,gamma = 0, subsample = 0.8,

                              colsample_bytree = 0.8, objective = 'binary:logistic',

                              scale_pos_weight=1, seed =27)

    

if useTrainCV:

    xgb_param = model_xgboost.get_xgb_params()

    xgtrain = xgb.DMatrix(X_train, label=y_train)

    cvresult = xgb.cv(xgb_param,xgtrain,num_boost_round=model_xgboost.get_params()['n_estimators'],

                    nfold = 5, metrics='auc',early_stopping_rounds = 50)

    model_xgboost.set_params(n_estimators=cvresult.shape[0])

    model_xgboost.fit(X_train,y_train)

    model_xgboost_accuracy = accuracy_score(y_test, model_xgboost.predict(X_test))

    feature_importances_xgboost = model_xgboost.feature_importances_

  

        

        

#Model 5 - Neural Net

model_nn = MLPClassifier()

  

parameters_nn = {'hidden_layer_sizes' : [(8,),(20,),(100,)],

                  'activation' : ['tanh','relu'],

                  'solver' : ['lbfgs','adam'],

                  'max_iter' : [1000]}

grid_obj_nn = GridSearchCV(model_nn, parameters_nn, scoring = acc_score, cv = 5,n_jobs =-1,verbose=1)

grid_obj_nn = grid_obj_nn.fit(X_train, y_train)

model_nn = grid_obj_nn.best_estimator_

model_nn.fit(X_train,y_train)

model_nn_accuracy = accuracy_score(y_test, model_nn.predict(X_test))


model_accuracies = [model_dt_accuracy, model_rf_accuracy, model_adaboost_accuracy, model_xgboost_accuracy, model_nn_accuracy]

model_results = pd.DataFrame(

    {"Algorithm":["DT",

                  "RF",

                  "AdaBoost",

                  "XGBoost",

                  "MLP"

                 ],

    "Accuracy": model_accuracies})



model_results


graph = sns.barplot("Accuracy","Algorithm",data = model_results, palette="Set3",orient = "h")

#graph = graph.set_title("Algorithms Scores",size = 20)
feature_names = feature_names

sizes = feature_importances_rf 

explode = (0,0.1,0,0,0,0,0,0,0.2,0,0)



fig, ax = plt.subplots(figsize = (20,18))

patches, texts ,autotexts = ax.pie(sizes,labels= feature_names,explode = explode, autopct='%1.1f%%',

                                   shadow=True, startangle=90)

ax.axis('equal') 

plt.setp(texts, size = 12, weight = "bold")

plt.show()
#explaining model predictions with lime



predict_fn  = lambda x : model_rf.predict_proba(x).astype(float)

#predict_xgboost  = lambda x : model_rf.predict_proba(x).astype(float)

explainer = lime.lime_tabular.LimeTabularExplainer(X_train, feature_names = feature_names , class_names =['0','1'], 

                                                   categorical_features = categorical_features , categorical_names = categorical_names ,

                                                   kernel_width =3)
# Pick the observation in the test set for which explanation is required

observation_1 = 6

exp = explainer.explain_instance(X_test[observation_1], predict_fn , num_features = 5)

exp.show_in_notebook(show_all = False)
observation_1 = 22

exp = explainer.explain_instance(X_test[observation_1], predict_fn , num_features = 5)

exp.show_in_notebook(show_all = False)