import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix, auc
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import RandomOverSampler


from collections import Counter
train_data = pd.read_csv("../input/kepler-labelled-time-series-data/exoTrain.csv") 
test_data=pd.read_csv('../input/kepler-labelled-time-series-data/exoTest.csv')
train_data.head()
train_data['LABEL'].value_counts()
train_data.info()
nulls = train_data.isnull().sum()
nulls[nulls > 0]
x_train=train_data.drop('LABEL',axis=1)
y_train=train_data[['LABEL']]
os=RandomOverSampler(0.8)
x_train_ns,y_train_ns=os.fit_sample(x_train,y_train)
y_train_ns['LABEL'].value_counts()
print("The number of classes before fit {}".format(Counter(y_train)))
print("The number of classes after fit {}".format(Counter(y_train_ns)))
x_test=test_data.drop('LABEL',axis=1)
y_test=test_data[['LABEL']]
def model(algo):
    algo_model = algo.fit(x_train_ns, y_train_ns)
    global y_prob, y_pred
    y_prob = algo.predict_proba(x_test)
    y_pred = algo_model.predict(x_test)

    print('Accuracy Score: {}\n\nConfusion Matrix:\n {}'
      .format(accuracy_score(y_test,y_pred), confusion_matrix(y_test,y_pred),roc_auc_score(y_test,y_pred)))
print('Decision Tree\n')
model(DecisionTreeClassifier(max_depth = 12))
print('Decision Tree\n')
model(DecisionTreeClassifier(max_depth = 6))
print('Random Forest\n')
model(RandomForestClassifier())
from xgboost import XGBClassifier

xgb_classifier = XGBClassifier()
xgb_classifier.fit(x_train_ns, y_train_ns)
y_pred_xgb = xgb_classifier.predict(x_test)
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
print(accuracy_xgb)
xgb_params={
 "learning_rate" : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
 "max_depth" : [ 3, 4, 5, 6, 8, 10, 12, 15],
 "min_child_weight" : [ 1, 3, 5, 7 ],
 "gamma" : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
 "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ] 
}
from xgboost import XGBClassifier

xgb_classifier = XGBClassifier()
from sklearn.model_selection import RandomizedSearchCV

xgb_random_search = RandomizedSearchCV(xgb_classifier, param_distributions = xgb_params,
                                       scoring= 'roc_auc',
                                       n_jobs= -1, verbose= 3)

xgb_random_search.fit(x_train_ns, y_train_ns)
xgb_random_search.best_params_
tuned_xgb_classifier = XGBClassifier(min_child_weight = 5,
                                     max_depth = 3,
                                     learning_rate = 0.05,
                                     gamma = 0.3,
                                     colsample_bytree = 0.3)
tuned_xgb_classifier.fit(x_train_ns, y_train_ns)
y_pred_tuned_xgb = tuned_xgb_classifier.predict(x_test)
accuracy_tuned_xgb = accuracy_score(y_test, y_pred_tuned_xgb)
print(accuracy_tuned_xgb)
print("Done")