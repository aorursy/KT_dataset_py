# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split

import keras

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from xgboost import XGBClassifier

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Dropout

import plotly.express as px

from sklearn.metrics import accuracy_score

from sklearn.metrics import mean_squared_log_error,mean_squared_error, r2_score,mean_absolute_error

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
df = pd.read_csv('../input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv')

df.head()
df.describe()
df.info()
df.isna().sum()
co = df.corr()

co.sort_values(by=["DEATH_EVENT"],ascending=False).iloc[0].sort_values(ascending=False)
corr = df.corr()

ax, fig = plt.subplots(figsize=(15,15))

sns.heatmap(corr, vmin=-1, cmap='coolwarm', annot=True)

plt.show()
sns.catplot(x="sex",y="age", hue="DEATH_EVENT", kind="bar", data=df)
sns.catplot(x="sex",y="age", hue="smoking", kind="bar", data=df)
sns.catplot(x="sex",y="age", hue="high_blood_pressure", kind="bar", data=df)
sns.catplot(x="sex",y="age", hue="diabetes", kind="bar", data=df)
ds = df['anaemia'].value_counts().reset_index()

ds.columns = ['anaemia', 'count']



WIDTH = 500

HEIGHT = 500



fig = px.pie(

    ds, 

    values='count', 

    names="anaemia", 

    title='Anaemia bar chart', 

    width=WIDTH, 

    height=HEIGHT

)



fig.show()
plt.rcParams['figure.figsize']=20,10 

sns.set_style("darkgrid")



x = df.iloc[:, :-1]

y = df.iloc[:,-1]



from sklearn.ensemble import ExtraTreesClassifier

import matplotlib.pyplot as plt

model = ExtraTreesClassifier()

model.fit(x,y)

print(model.feature_importances_) 

feat_importances = pd.Series(model.feature_importances_, index=x.columns)

feat_importances.nlargest(12).plot(kind='barh')

plt.show()
X = df[['age','ejection_fraction', 'serum_creatinine', 'serum_sodium', 'time']]

Y = df['DEATH_EVENT']
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,random_state=2,test_size=.25)
scale=StandardScaler()

X_train_scale=scale.fit_transform(X_train)

X_test_scale=scale.transform(X_test)


model = LogisticRegression()



# fit the model with the training data

model.fit(X_train_scale,Y_train)



# coefficeints of the trained model

print('Coefficient of model :', model.coef_)

print('                             ')



# intercept of the model

print('Intercept of model',model.intercept_)

print('                             ')

# predict the target on the test dataset

predict_test = model.predict(X_test_scale)

print('Target on test data',predict_test) 

print('                             ')

# Accuracy Score on test dataset

accuracy_test = accuracy_score(Y_test,predict_test)

print('accuracy_score on test dataset : ', accuracy_test)
model = DecisionTreeClassifier(max_depth=3)



# fit the model with the training data

model.fit(X_train_scale,Y_train)



# depth of the decision tree

print('Depth of the Decision Tree :', model.get_depth())

print('                             ')

# predict the target on the test dataset

predict_test = model.predict(X_test_scale)

print('Target on test data',predict_test) 

print('                             ')

# Accuracy Score on test dataset

accuracy_test = accuracy_score(Y_test,predict_test)

print('accuracy_score on test dataset : ', accuracy_test)
model = RandomForestClassifier(max_depth=3)



# fit the model with the training data

model.fit(X_train_scale,Y_train)



# predict the target on the test dataset

predict_test = model.predict(X_test_scale)

print('Target on test data',predict_test) 

print('                             ')



# Accuracy Score on test dataset

accuracy_test = accuracy_score(Y_test,predict_test)

print('accuracy_score on test dataset : ', accuracy_test)
model = GradientBoostingClassifier(n_estimators=100,max_depth=3)



# fit the model with the training data

model.fit(X_train_scale,Y_train)



# predict the target on the test dataset

predict_test = model.predict(X_test_scale)

print('\nTarget on test data',predict_test) 



# Accuracy Score on test dataset

print('                                  ')

accuracy_test = accuracy_score(Y_test,predict_test)

print('\naccuracy_score on test dataset : ', accuracy_test)
model = XGBClassifier(max_depth=5)



# fit the model with the training data

model.fit(X_train_scale,Y_train)



# predict the target on the test dataset

predict_test = model.predict(X_test_scale)

print('\nTarget on test data',predict_test) 



# Accuracy Score on test dataset

accuracy_test = accuracy_score(Y_test,predict_test)

print('                                          ')

print('\naccuracy_score on test dataset : ', accuracy_test)
model = SVC(kernel='linear')



# fit the model with the training data

model.fit(X_train_scale,Y_train)



# predict the target on the test dataset

predict_test = model.predict(X_test_scale)

print('Target on test data',predict_test) 



# Accuracy Score on test dataset

accuracy_test = accuracy_score(Y_test,predict_test)

print('                            ')

print('accuracy_score on test dataset : ', accuracy_test)
model = GaussianNB()



# fit the model with the training data

model.fit(X_train_scale,Y_train)



# predict the target on the test dataset

predict_test = model.predict(X_test_scale)

print('Target on test data',predict_test) 



# Accuracy Score on test dataset

print('                       ')

accuracy_test = accuracy_score(Y_test,predict_test)

print('accuracy_score on test dataset : ', accuracy_test)