import numpy as np

import pandas as pd

from sklearn.preprocessing import RobustScaler

from sklearn.model_selection import train_test_split



data = pd.read_csv('../input/creditcard.csv')



# Separata data into X/y

y = data['Class'].values

X = data.drop(['Class', 'Time'], axis=1).values



num_neg = (y==0).sum()

num_pos = (y==1).sum()



# Scaling..

scaler = RobustScaler()

X = scaler.fit_transform(X)



# Split into train/test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
import seaborn as sns



print(data.groupby('Class').size())



sns.countplot(x="Class", data=data)
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix, classification_report

from mlxtend.plotting import plot_decision_regions, plot_confusion_matrix

from matplotlib import pyplot as plt



lr = LogisticRegression()



# Fit..

lr.fit(X_train, y_train)



# Predict..

y_pred = lr.predict(X_test)



# Evaluate the model

print(classification_report(y_test, y_pred))

plot_confusion_matrix(confusion_matrix(y_test, y_pred))
lr = LogisticRegression(class_weight='balanced')



# Fit..

lr.fit(X_train, y_train)



# Predict..

y_pred = lr.predict(X_test)



# Evaluate the model

print(classification_report(y_test, y_pred))

plot_confusion_matrix(confusion_matrix(y_test, y_pred))
from sklearn.model_selection import GridSearchCV



weights = np.linspace(0.05, 0.95, 20)



gsc = GridSearchCV(

    estimator=LogisticRegression(),

    param_grid={

        'class_weight': [{0: x, 1: 1.0-x} for x in weights]

    },

    scoring='f1',

    cv=3

)

grid_result = gsc.fit(X, y)



print("Best parameters : %s" % grid_result.best_params_)



# Plot the weights vs f1 score

dataz = pd.DataFrame({ 'score': grid_result.cv_results_['mean_test_score'],

                       'weight': weights })

dataz.plot(x='weight')
lr = LogisticRegression(**grid_result.best_params_)



# Fit..

lr.fit(X_train, y_train)



# Predict..

y_pred = lr.predict(X_test)



# Evaluate the model

print(classification_report(y_test, y_pred))

plot_confusion_matrix(confusion_matrix(y_test, y_pred))
from imblearn.over_sampling import SMOTE

from imblearn.pipeline import make_pipeline



pipe = make_pipeline(

    SMOTE(),

    LogisticRegression()

)



# Fit..

pipe.fit(X_train, y_train)



# Predict..

y_pred = pipe.predict(X_test)



# Evaluate the model

print(classification_report(y_test, y_pred))

plot_confusion_matrix(confusion_matrix(y_test, y_pred))
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning) 



pipe = make_pipeline(

    SMOTE(),

    LogisticRegression()

)



weights = np.linspace(0.005, 0.05, 10)



gsc = GridSearchCV(

    estimator=pipe,

    param_grid={

        #'smote__ratio': [{0: int(num_neg), 1: int(num_neg * w) } for w in weights]

        'smote__ratio': weights

    },

    scoring='f1',

    cv=3

)

grid_result = gsc.fit(X, y)



print("Best parameters : %s" % grid_result.best_params_)



# Plot the weights vs f1 score

dataz = pd.DataFrame({ 'score': grid_result.cv_results_['mean_test_score'],

                       'weight': weights })

dataz.plot(x='weight')
pipe = make_pipeline(

    SMOTE(ratio=0.015),

    LogisticRegression()

)



# Fit..

pipe.fit(X_train, y_train)



# Predict..

y_pred = pipe.predict(X_test)



# Evaluate the model

print(classification_report(y_test, y_pred))

plot_confusion_matrix(confusion_matrix(y_test, y_pred))