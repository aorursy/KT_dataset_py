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
import pandas as pd
!pip install pycaret
df=pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')
df.head()
df['Class'].value_counts()
df['Class'].value_counts().plot(kind='bar')
from pycaret.classification import *
clf = setup(data=df, target='Class')
compare_models()
catboost = create_model('catboost')
tuned_catboost = tune_model('catboost')
tuned_catboost.get_all_params()
interpret_model(catboost)
from sklearn.metrics import confusion_matrix, classification_report
y_pred = catboost.predict(df.drop(columns=['Class'], axis=1))
confusion_matrix(df['Class'], y_pred)
print(classification_report(df['Class'], y_pred))
