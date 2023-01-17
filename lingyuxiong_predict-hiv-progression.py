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

train_data = pd.read_csv('../input/hivprogression/training_data.csv')

test_data = pd.read_csv('../input/hivprogression/test_data.csv')

train_data.head()
import seaborn as sns

sns.countplot(train_data['Resp'])
corr = train_data.corr()

sns.heatmap(corr)
from sklearn.model_selection import train_test_split

X = train_data[['VL-t0','CD4-t0']]

Y = train_data['Resp'].values

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=2)
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score

knn_model = KNeighborsClassifier()

knn_model.fit(x_train, y_train)

predicted = knn_model.predict(x_test)

print('KNN', accuracy_score(predicted, y_test))
from sklearn.ensemble import RandomForestClassifier

rfc_model = RandomForestClassifier()

rfc_model.fit(x_train, y_train)

predicted = rfc_model.predict(x_test)

print('Random Forest', accuracy_score(y_test, predicted))
from sklearn.svm import SVC

svc_model = SVC(gamma='auto')

svc_model.fit(x_train, y_train)

predicted = svc_model.predict(x_test)

print('SVM', accuracy_score(y_test, predicted))
from xgboost import XGBClassifier

xgb_model = XGBClassifier()

xgb_model.fit(x_train, y_train)

predicted = xgb_model.predict(x_test)

print('XGBoost', accuracy_score(y_test, predicted))
test = test_data[['VL-t0','CD4-t0']]

predict = svc_model.predict(test)
submissions = pd.DataFrame({

    'PatientID': np.arange(1,predict.shape[0]+1),

    'ResponderStatus': predict

})

submissions.to_csv('submission.csv', index=False)