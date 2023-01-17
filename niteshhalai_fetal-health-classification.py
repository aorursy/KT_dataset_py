import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline



import warnings

warnings.filterwarnings("ignore")



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv('/kaggle/input/fetal-health-classification/fetal_health.csv')

df
def data_cleaner(data):

    data.fillna(0, inplace=True)

    return data
df = data_cleaner(df)

df
df.columns
from sklearn.model_selection import KFold, cross_val_score

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from xgboost.sklearn import XGBClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split



target = ['fetal_health']

features = ['baseline value', 'accelerations', 'fetal_movement',

       'uterine_contractions', 'light_decelerations', 'severe_decelerations',

       'prolongued_decelerations', 'abnormal_short_term_variability',

       'mean_value_of_short_term_variability',

       'percentage_of_time_with_abnormal_long_term_variability',

       'mean_value_of_long_term_variability', 'histogram_width',

       'histogram_min', 'histogram_max', 'histogram_number_of_peaks',

       'histogram_number_of_zeroes', 'histogram_mode', 'histogram_mean',

       'histogram_median', 'histogram_variance', 'histogram_tendency']



X = df[features]

y = df[target]





X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, shuffle=True)



models = []

models.append(('LogisticRegression', LogisticRegression()))

models.append(('DecisionTreeClassifier', DecisionTreeClassifier()))

models.append(('XGBClassifier', XGBClassifier()))

models.append(('GradientBoostingClassifier', GradientBoostingClassifier()))

models.append(('KNeighborsClassifier', KNeighborsClassifier()))

models.append(('RandomForestClassifier', RandomForestClassifier()))



results = []

names = []

for name, model in models:

    kfold = KFold(n_splits=10, random_state=0)

    cv_results = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')

    results.append(cv_results)

    names.append(name)

    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

    print(msg)
model=XGBClassifier(random_state=0)

model.fit(X_train,y_train)

y_pred=model.predict(X_test)



from sklearn import metrics



cm = metrics.confusion_matrix(y_test, y_pred)

plt.figure(figsize=(9,9))

sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');

plt.ylabel('Actual label');

plt.xlabel('Predicted label');

all_sample_title = 'Confusion Matrix - score:'+str(metrics.accuracy_score(y_test,y_pred))

plt.title(all_sample_title, size = 15);

plt.show()

print(metrics.classification_report(y_test,y_pred))