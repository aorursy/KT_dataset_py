# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

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

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('../input/students-performance-in-exams/StudentsPerformance.csv')

df.head()
df['average'] = (df['math score']+df['reading score']+df['writing score'])/3

df
df.describe()
df.info()
df.isna().sum()
plt.figure(dpi=100)

plt.title('Correlation Analysis')

sns.heatmap(df.corr(),annot=True,lw=1,linecolor='white',cmap='viridis')

plt.xticks(rotation=60)

plt.yticks(rotation = 60)

plt.show()
sns.distplot(df['average'],hist_kws=dict(edgecolor="k", linewidth=1))
sns.distplot(df['math score'],hist_kws=dict(edgecolor="k", linewidth=1))
sns.distplot(df['reading score'],hist_kws=dict(edgecolor="k", linewidth=1))
sns.distplot(df['writing score'],hist_kws=dict(edgecolor="k", linewidth=1))
df['parental level of education'].value_counts().head(30).plot(kind='barh', figsize=(10,10))
df['race/ethnicity'].value_counts().head(30).plot(kind='barh', figsize=(10,10))
df.groupby(['race/ethnicity','gender']).size().unstack().plot(kind='bar',stacked=True)

plt.show()
new = df[['math score','reading score','writing score']].copy()



math_df = new['math score'].sum()

reading_df = new['reading score'].sum()

writing_df = new['writing score'].sum()



total = [math_df,reading_df,writing_df]

columns = ['Math','Reading','Writing']



fig1, ax1 = plt.subplots()

ax1.pie(total, labels=columns, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.



plt.show() 
df.groupby(['reading score','gender']).size().unstack().plot(kind='bar',stacked=True,figsize=(15,15))

plt.show()
df.groupby(['writing score','gender']).size().unstack().plot(kind='bar',stacked=True,figsize=(15,15))

plt.show()
df.groupby(['math score','gender']).size().unstack().plot(kind='bar',stacked=True,figsize=(15,15))

plt.show()
df.groupby(["test preparation course"]).mean().plot.bar()

plt.show()
df.groupby(["parental level of education"]).mean().plot.bar()

plt.show()
df.groupby(["race/ethnicity"]).mean().plot.bar()

plt.show()
bplot = sns.boxplot( y = 'average' ,x ='parental level of education'  ,data = df)

_ = plt.setp(bplot.get_xticklabels(), rotation=90)
sns.boxplot(x="parental level of education", y="average", hue="gender", data=df, palette="Set1")

#sns.plt.show()
X = df.drop(columns=['gender'],axis=1)

Y = df['gender']
from sklearn import preprocessing 

   

label_encoder = preprocessing.LabelEncoder() 

Y = label_encoder.fit_transform(Y)
X = pd.get_dummies(X)
# FEATURE SELECTION



plt.rcParams['figure.figsize']=15,6 

sns.set_style("darkgrid")



from sklearn.ensemble import ExtraTreesClassifier

import matplotlib.pyplot as plt

model = ExtraTreesClassifier()

model.fit(X,Y)

print(model.feature_importances_) 

feat_importances = pd.Series(model.feature_importances_, index=X.columns)

feat_importances.nlargest(12).plot(kind='barh')

plt.show()
from sklearn.model_selection import train_test_split



X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)


model = LogisticRegression()



# fit the model with the training data

model.fit(X_train,Y_train)



# coefficeints of the trained model

print('Coefficient of model :', model.coef_)

print('                             ')



# intercept of the model

print('Intercept of model',model.intercept_)

print('                             ')

# predict the target on the test dataset

predict_test = model.predict(X_test)

print('Target on test data',predict_test) 

print('                             ')

# Accuracy Score on test dataset

accuracy_test = accuracy_score(Y_test,predict_test)

print('accuracy_score on test dataset : ', accuracy_test)
model = DecisionTreeClassifier(max_depth=6)



# fit the model with the training data

model.fit(X_train,Y_train)



# depth of the decision tree

print('Depth of the Decision Tree :', model.get_depth())

print('                             ')

# predict the target on the test dataset

predict_test = model.predict(X_test)

print('Target on test data',predict_test) 

print('                             ')

# Accuracy Score on test dataset

accuracy_test = accuracy_score(Y_test,predict_test)

print('accuracy_score on test dataset : ', accuracy_test)
model = RandomForestClassifier(max_depth=8)



# fit the model with the training data

model.fit(X_train,Y_train)



# predict the target on the test dataset

predict_test = model.predict(X_test)

print('Target on test data',predict_test) 

print('                             ')



# Accuracy Score on test dataset

accuracy_test = accuracy_score(Y_test,predict_test)

print('accuracy_score on test dataset : ', accuracy_test)
model = GradientBoostingClassifier(n_estimators=100,max_depth=3)



# fit the model with the training data

model.fit(X_train,Y_train)



# predict the target on the test dataset

predict_test = model.predict(X_test)

print('\nTarget on test data',predict_test) 



# Accuracy Score on test dataset

print('                                  ')

accuracy_test = accuracy_score(Y_test,predict_test)

print('\naccuracy_score on test dataset : ', accuracy_test)
model = XGBClassifier(max_depth=5)



# fit the model with the training data

model.fit(X_train,Y_train)



# predict the target on the test dataset

predict_test = model.predict(X_test)

print('\nTarget on test data',predict_test) 



# Accuracy Score on test dataset

accuracy_test = accuracy_score(Y_test,predict_test)

print('                                          ')

print('\naccuracy_score on test dataset : ', accuracy_test)
model = SVC(kernel='linear')



# fit the model with the training data

model.fit(X_train,Y_train)



# predict the target on the test dataset

predict_test = model.predict(X_test)

print('Target on test data',predict_test) 



# Accuracy Score on test dataset

accuracy_test = accuracy_score(Y_test,predict_test)

print('                            ')

print('accuracy_score on test dataset : ', accuracy_test)