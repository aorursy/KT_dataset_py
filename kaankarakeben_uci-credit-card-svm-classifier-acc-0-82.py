# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/UCI_Credit_Card.csv')
X = df.iloc[:,1:24]
Y = df['default.payment.next.month']

df.head()
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

corr=df.corr()
corr = (corr)
plt.figure(figsize=(20,20))
sns.heatmap(corr, cbar = True,  square = True, annot=True, fmt= '.2f',annot_kws={'size': 13},
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
plt.title('Heatmap of Correlation Matrix')
from sklearn.preprocessing import OneHotEncoder
OHE = OneHotEncoder(sparse=False)
sex = OHE.fit_transform(X[['SEX']])
marriage = OHE.fit_transform(X[['MARRIAGE']])
education = OHE.fit_transform(X[['EDUCATION']])

cat_variables = np.hstack((sex, marriage, education))

cat_var_names = ['SEX','MARRIAGE', 'EDUCATION']

num_variables = X.drop(cat_var_names, axis=1)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(num_variables)
stdz_num_variables = scaler.transform(num_variables)

final_X = np.hstack((cat_variables,stdz_num_variables))
from sklearn.feature_selection import SelectKBest

test = SelectKBest(k=10)
fit = test.fit(final_X,Y)
np.set_printoptions(precision=1)
print(fit.scores_)
features = fit.transform(final_X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features,Y, test_size=0.33, random_state=42)
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

estimator = SVC(kernel='rbf')
selector = estimator.fit(X_train,y_train)
#print(X_train.columns[selector.support_])
y_predict_test = selector.predict(X_test)
y_predict_train = selector.predict(X_train)
print("Train Accuracy Score:", accuracy_score(y_train, y_predict_train))
print("Test Accuracy Score:", accuracy_score(y_test, y_predict_test))