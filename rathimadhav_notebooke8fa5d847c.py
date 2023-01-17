# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
path = "/kaggle/input/pima-indians-diabetes-database/diabetes.csv"

df = pd.read_csv(path)
df.shape

df.isnull().any()
sns.pairplot(df)
corr = df.corr()

plt.figure(figsize=(10, 8))



ax = sns.heatmap(corr, vmin = -1, vmax = 1, annot = True)

bottom, top = ax.get_ylim()

ax.set_ylim(bottom + 0.5, top - 0.5)

plt.show()

corr
#as we can see here,almost all the inputs plays a part in deciding the outcome,

#especially, Glucose, BMI, Age and going on further
#X= df.loc[:, df.columns != 'Outcome']

X = df[['Pregnancies', 'Glucose', 'BloodPressure',

      'BMI', 'DiabetesPedigreeFunction', 'Age']]

y = df ['Outcome']

from sklearn.ensemble import ExtraTreesClassifier



model = ExtraTreesClassifier()

model.fit(X,y)

print(model.feature_importances_)

#use inbuilt class feature_importances of tree based classifiers



#plot graph of feature importances for better visualization

feat_importances = pd.Series(model.feature_importances_, index=X.columns)

feat_importances.nlargest(10).plot(kind='barh')

plt.show()
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

X =  pd.DataFrame(sc_X.fit_transform(df.drop(["Outcome", 'SkinThickness', 'Insulin'],axis = 1),),

        columns=['Pregnancies', 'Glucose', 'BloodPressure',

       'BMI', 'DiabetesPedigreeFunction', 'Age'])
from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier(6)

knn.fit(train_X,train_y)

print("Accuracy : ",knn.score(val_X,val_y))
from sklearn.model_selection import train_test_split

train_X, val_X, train_y, val_y = train_test_split(X, y,random_state = 0)
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error



forest_model = RandomForestRegressor(n_estimators = 750,random_state=5)

forest_model.fit(train_X, train_y)

predictions = forest_model.predict(val_X)



preds = pd.Series(np.array(predictions))

preds = preds.round().astype(int)

print("Accuracy:",metrics.accuracy_score(val_y, preds))



from sklearn import metrics

from sklearn.linear_model import LogisticRegression

reg = LogisticRegression()

reg.fit(train_X, train_y)

y_pred = reg.predict(val_X)

print("Accuracy:",metrics.accuracy_score(val_y, y_pred))