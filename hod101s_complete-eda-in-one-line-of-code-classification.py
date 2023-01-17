!pip install autoviz
import numpy as np

import pandas as pd

from autoviz.AutoViz_Class import AutoViz_Class
data = pd.read_csv('../input/health-care-data-set-on-heart-attack-possibility/heart.csv')

data.head()
data.shape
dft = AutoViz_Class().AutoViz("../input/health-care-data-set-on-heart-attack-possibility/heart.csv", sep=",", depVar="target", dfte="pandasDF",verbose=1, lowess=False, chart_format="svg", max_rows_analyzed=350, max_cols_analyzed=15)
from sklearn.model_selection import GridSearchCV, train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import plot_confusion_matrix, classification_report
X_train, X_test , y_train, y_test = train_test_split(data.drop('target',axis=1),data.target,random_state = 32,test_size = 0.2)
rfc = RandomForestClassifier(n_jobs=-1,max_features= 'sqrt' ,n_estimators=50, oob_score = True, bootstrap = True,) 



param_grid = { 

    'n_estimators': [70, 100,150],

    'max_features': ['auto', 'sqrt', 'log2'],

    'random_state' : [0,32,48],

}



CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 6)
%time CV_rfc.fit(X_train,y_train)
%time preds = CV_rfc.predict(X_test)
plot_confusion_matrix(CV_rfc, X_test, y_test)
print(classification_report(y_test,preds))
rfc2 = RandomForestClassifier(n_jobs=-1,max_features= 'sqrt' ,n_estimators=50, oob_score = True,) 



param_grid = { 

    'n_estimators': [70, 100,150],

    'max_features': ['auto', 'sqrt', 'log2'],

    'random_state' : [0,32,48],

}



CV_rfc2 = GridSearchCV(estimator=rfc2, param_grid=param_grid, cv= 6)
%time CV_rfc2.fit(X_train,y_train)
%time preds2 = CV_rfc2.predict(X_test)
plot_confusion_matrix(CV_rfc2, X_test, y_test)

print(classification_report(y_test,preds2))