# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import numpy as np



from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split, GridSearchCV



from sklearn.metrics import (

    confusion_matrix,

    ConfusionMatrixDisplay,

    roc_curve,

    RocCurveDisplay,

    precision_recall_curve,

    PrecisionRecallDisplay

)
data = pd.read_csv('/kaggle/input/adult-census-income/adult.csv')
# Add cross-features



data['f1'] = data[['occupation','race']].apply(lambda x: x[0].strip() + '-' + x[1].strip(), axis=1)

data['f2'] = data[['occupation','sex']].apply(lambda x: x[0].strip() + '-' + x[1].strip(), axis=1)

data['f3'] = data[['race','sex']].apply(lambda x: x[0].strip() + '-' + x[1].strip(), axis=1)
y = data['income'].map(lambda item: 1.0 if item.strip() == '>50K' else 0.0)
y.value_counts()
X = data.drop(['income'], axis=1)
X
NUM_COLS = [

    'age',

    'fnlwgt',

    'capital.gain',

    'capital.loss',

    'hours.per.week'

]
CAT_COLS = [

    'workclass',

    'education',

    'marital.status',

    'occupation',

    'relationship',

    'race',

    'sex',

    'native.country',

    

    'f1',

    'f2',

    'f3'

]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
numeric_transformer = Pipeline(steps=[

    ('scaler', StandardScaler()),

])



categorical_transformer = Pipeline(steps=[

    ('onehot', OneHotEncoder(handle_unknown='ignore'))])



preprocessor = ColumnTransformer(

    transformers=[

        ('num', numeric_transformer, NUM_COLS),

        ('cat', categorical_transformer, CAT_COLS)])
clf = Pipeline(steps=[('preprocessor', preprocessor),

                      ('classifier', LogisticRegression(C=100., max_iter=1000, class_weight='balanced', random_state=0))])



clf.fit(X_train, y_train)

print("model score: %.3f" % clf.score(X_test, y_test))
y_pred = clf.predict(X_test)

cm = confusion_matrix(y_test, y_pred)



cm_display = ConfusionMatrixDisplay(cm).plot()
y_score = clf.predict_proba(X_test)[:,1]



fpr, tpr, _ = roc_curve(y_test, y_score, pos_label=clf.classes_[1])

roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
prec, recall, _ = precision_recall_curve(y_test, y_score, pos_label=clf.classes_[1])

pr_display = PrecisionRecallDisplay(precision=prec, recall=recall).plot()