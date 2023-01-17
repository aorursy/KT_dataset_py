import numpy as np

import pandas as pd



from sklearn.experimental import enable_hist_gradient_boosting  # noqa

from sklearn.preprocessing import RobustScaler, StandardScaler

from sklearn.compose import ColumnTransformer

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, HistGradientBoostingClassifier

from sklearn.metrics import roc_auc_score, classification_report, roc_curve, auc, plot_confusion_matrix



from imblearn.over_sampling import SMOTE

from imblearn.pipeline import Pipeline

from sklearn import set_config

set_config(display='diagram')
data = pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')
data.head()
data.info()
X = data.drop(['Time', 'Class'], axis=1)

X.head()
y = data['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
column_trans = Pipeline([('scaler', StandardScaler())])



preprocessing = ColumnTransformer(transformers=[

    ('column_trans', column_trans, ['Amount'])

], remainder='passthrough')
pipe = Pipeline([

    ('preprocessing', preprocessing),

    ('smote', SMOTE()), # The data is highly imbalanced, hence oversample minority class with SMOTE 

    ('classifier', RandomForestClassifier(n_estimators=10))

])
grid_search_models = {

    'classifier': [RandomForestClassifier(n_estimators=20), AdaBoostClassifier(n_estimators=20), HistGradientBoostingClassifier(max_iter=20)]

}
pipe = GridSearchCV(pipe, grid_search_models, verbose=50, cv=2, scoring='roc_auc')
pipe.fit(X_train, y_train)
print('Best estimator with score of {}' % pipe.best_score_)

pipe.best_estimator_
y_hat = pipe.predict_proba(X_test)

preds = y_hat[:,1]
fpr, tpr, threshold = roc_curve(y_test, preds)

roc_auc = auc(fpr, tpr)

roc_auc
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

plt.title('Receiver Operating Characteristic')

plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)

plt.legend(loc = 'lower right')

plt.plot([0, 1], [0, 1],'r--')

plt.axhline()

plt.axvline(linewidth=5)

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0, 1])

plt.ylim([0, 1])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()
# Plot the confusion matrix

plot_confusion_matrix(pipe, X_test, y_test)

plt.grid(b=None)