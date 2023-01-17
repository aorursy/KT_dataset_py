import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sb



%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Loading the data set into a pandas dataframe

df=pd.read_csv('../input/pima-indians-diabetes-database/diabetes.csv')
df.info()
df.describe()
df.head(3)
df.columns
df1=df.copy()
df1.head(3)
df1.describe()
plt.hist('Pregnancies',data=df1);
plt.hist('Glucose',data=df1);
plt.hist('BloodPressure',data=df1);
plt.hist('SkinThickness',data=df1);
plt.hist('Insulin',data=df1);
plt.hist('BMI',data=df1);
plt.hist('Age',data=df1);
plt.hist('DiabetesPedigreeFunction',data=df1);
plt.scatter('Outcome','SkinThickness',data=df1);
f, ax = plt.subplots(figsize=(20, 8))

corr = df1.corr()

sb.heatmap(corr, mask=np.zeros_like(corr,dtype=np.bool),cmap=sb.diverging_palette(220, 10, as_cmap=True),annot=True,ax=ax,);
pd.plotting.scatter_matrix(df1, alpha=0.2, figsize=(10, 10))

plt.xlabel(rotation=90)

plt.ylabel(rotation=90)

plt.show();
X=df1

y=df1['Outcome']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
from sklearn.linear_model import LogisticRegression

clf_LR = LogisticRegression(random_state = 0)

clf_LR.fit(X_train, y_train)
y_pred_train = clf_LR.predict(X_train)

y_pred_test = clf_LR.predict(X_test)
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred_test)

cm