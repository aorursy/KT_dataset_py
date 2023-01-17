#It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

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

df = pd.read_csv('../input/abalone-dataset/abalone.csv')
df.head()
df.info()
df.describe()
df.shape
df["age"] = df["Rings"] + 1.5

#df.drop("Rings",axis = 1, inplace = True)
df.head()
df.isnull().sum()
df.info()
df.var()
import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls

import warnings
plt.figure(figsize=(8,4))

sns.countplot(x='Sex', data=df , palette= "YlOrRd")
plt.figure(figsize=(12,8))

sn = sns.countplot(x='age',data=df, hue='Sex', palette=['pink','crimson',"Yellow"])
atributes_sex = df[['Sex','Length','Diameter',	'Height',	'Whole weight',	'Shucked weight',	'Viscera weight',	'Shell weight'	]].groupby('Sex').mean()

cols = ['Length','Diameter',	'Height',	'Whole weight',	'Shucked weight',	'Viscera weight',	'Shell weight']

atributes_sex.columns = cols
list(atributes_sex.iloc[0])
atributes_sex.columns.values
atributes_sex
from plotly.offline import iplot

trace1 = go.Bar(

    y=list(atributes_sex.iloc[2]),

    x=atributes_sex.columns.values,

    name='Men',

    marker=dict(

        color='navy'

    )

)

trace2 = go.Bar(

    y=list(atributes_sex.iloc[0]),

    x=atributes_sex.columns.values,

    name='Women',

    marker=dict(

        color='mediumslateblue'

    )

)

trace3 = go.Bar(

    y=list(atributes_sex.iloc[1]),

    x=atributes_sex.columns.values,

    name='Infant',

    marker=dict(

        color='cornflowerblue'

    )

)



data = [trace1, trace2,trace3]

layout = go.Layout(

    title='Features',

    font=dict(

        size=18

    ),

    legend=dict(

        font=dict(

            size=18

        )

    )

)

fig = go.Figure(data=data, layout=layout)

iplot(fig)

plt.show()
X = df[['Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight',

       'Viscera weight', 'Shell weight']]

Y = df["Rings"]

sns.jointplot(x = "Length", y = "age", data=df, kind="reg")

plt.show()
sns.lmplot(x='Diameter',y='age',data=df,col='Sex',palette = "red")
sns.lmplot(x='Whole weight',y='age',data=df ,hue = "Sex",palette = "YlOrRd")
sns.lmplot(x = 'Height', y = 'age', data = df, hue = 'Sex', palette = 'magma', scatter_kws={'edgecolor':'white', 'alpha':0.4, 'linewidth':0.5})
sns.pairplot(df,hue = "Sex",palette= "coolwarm")
plt.figure(figsize = (8,6))

corr = df.corr()

sns.heatmap(corr, annot = True)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size=0.33, random_state=42)
X_train.head()
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit_transform(X_train)
from sklearn.linear_model import LinearRegression

reg = LinearRegression().fit(X_train, y_train)

print("Test score "+str(reg.score(X_test, y_test)))

print("Train score "+str(reg.score(X_train, y_train)))
from sklearn.svm import SVR

svr_rbf = SVR(kernel='rbf', C=100, gamma=1, epsilon=.01).fit(X_train, y_train)

svr_lin = SVR(kernel='linear', C=100, gamma='auto').fit(X_train, y_train)

svr_poly = SVR(kernel='poly', C=100, gamma='auto', degree=3, epsilon=.1, coef0=1).fit(X_train, y_train)



print("Test score for rbf :",str(svr_rbf.score(X_test, y_test)))

print("Test score for lin :",str(svr_rbf.score(X_test, y_test)))

print("Test score for poly :",str(svr_rbf.score(X_test, y_test)))
from sklearn.tree import DecisionTreeRegressor

# create an estimator, optionally specifying parameters

model = DecisionTreeRegressor()

# fit the estimator to the data

model.fit(X_train,y_train)

# apply the model to the test and training data

predicted_test_y = model.predict(X_test)

predicted_train_y = model.predict(X_train)

def scatter_y(y_test, predicted_y):

    """Scatter-plot the predicted vs true number of rings

    

    Plots:

       * predicted vs true number of rings

       * perfect agreement line

       * +2/-2 number dotted lines



    Returns the root mean square of the error

    """

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.plot(y_test, predicted_y, '.k')

    

    ax.plot([0, 30], [0, 30], '--k')

    ax.plot([0, 30], [2, 32], ':k')

    ax.plot([2, 32], [0, 30], ':k')

    

    rms = (y_test - predicted_y).std()

    

    ax.text(25, 3,

            "Root Mean Square Error = %.2g" % rms,

            ha='right', va='bottom')



    ax.set_xlim(0, 30)

    ax.set_ylim(0, 30)

    

    ax.set_xlabel('True age')

    ax.set_ylabel('Predicted age')

    

    return rms

scatter_y(y_train, predicted_train_y)

plt.title("Training data")

scatter_y(y_test, predicted_test_y)

plt.title("Test data");
model = DecisionTreeRegressor(max_depth=10)

# fit the estimator to the data

model.fit(X_train,y_train)

# apply the model to the test and train data

predicted_test_y = model.predict(X_test)

predicted_train_y = model.predict(X_train)



scatter_y(y_train, predicted_train_y)

plt.title("Training data")

rms_decision_tree = scatter_y(y_test, predicted_test_y)

plt.title("Test data");
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=10)

model.fit(X_train, y_train)

predicted_test_y = model.predict(X_test)

rms_random_forest = scatter_y(y_test, predicted_test_y)
df["age"].value_counts()
df_1 = df.copy()

Age = []

for i in df_1["age"]:

    if i < 9.33:

        Age.append("1")

    if i > 9.33 and i< 18.66 :

        Age.append("2")

    if i > 18.66:

        Age.append("3")

df_1["Age"] = Age

df_1.drop("age" , axis =1,inplace=True)

df_1.head()
X = df_1[['Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight',

       'Viscera weight', 'Shell weight']]

Y = df_1["Age"]

from sklearn.model_selection import train_test_split

X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split( X, Y, test_size=0.33, random_state=42)
scaler = StandardScaler()

scaler.fit_transform(X_train_1)
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(random_state=0).fit(X_train_1, y_train_1)

y_pred = clf.predict(X_test)

print("Test score "+str(clf.score(X_test_1, y_test_1)))

print("Train score "+str(clf.score(X_train_1, y_train_1)))

from sklearn.metrics import classification_report

print("Classification report :")

print(classification_report(y_test_1, y_pred))
from sklearn.neighbors import KNeighborsClassifier

neigh = KNeighborsClassifier(n_neighbors=3)

neigh.fit(X_train_1, y_train_1)

y_pred = neigh.predict(X_test)

print("Test score "+str(neigh.score(X_test_1, y_test_1)))

print("Train score "+str(neigh.score(X_train_1, y_train_1)))

from sklearn.metrics import classification_report

print("Classification report :")

print(classification_report(y_test_1, y_pred))



error_rate = []

for i in range(1,40):

    knn = KNeighborsClassifier(n_neighbors= i)

    knn.fit(X_train, y_train)

    y_predi = knn.predict(X_test)

    error_rate.append(np.mean(y_test != y_predi))

    

plt.figure(figsize = (10,8))

plt.plot(range(1,40), error_rate, color='blue', linestyle='dashed', marker='o', markerfacecolor='red', markersize=10)

plt.show()
from sklearn.svm import SVC

from sklearn.pipeline import make_pipeline



from sklearn.preprocessing import StandardScaler



regr = make_pipeline(StandardScaler(), SVC(gamma = 'auto'))



regr.fit(X_train_1, y_train_1)



y_pred = regr.predict(X_test_1)

print("Test score "+str(regr.score(X_test_1, y_test_1)))

print("Train score "+str(regr.score(X_train_1, y_train_1)))

from sklearn.metrics import classification_report

print("Classification report :")

print(classification_report(y_test_1, y_pred))
from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier(random_state=0)

dtc.fit(X_train_1, y_train_1)

y_pred = dtc.predict(X_test)

print("Test score "+str(dtc.score(X_test_1, y_test_1)))

print("Train score "+str(dtc.score(X_train_1, y_train_1)))

from sklearn.metrics import classification_report

print("Classification report :")

print(classification_report(y_test_1, y_pred))

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(max_depth=2, random_state=0)

clf.fit(X_train_1, y_train_1)

y_pred = clf.predict(X_test)

print("Test score "+str(clf.score(X_test_1, y_test_1)))

print("Train score "+str(clf.score(X_train_1, y_train_1)))

from sklearn.metrics import classification_report

print("Classification report :")

print(classification_report(y_test_1, y_pred))
from sklearn.ensemble import AdaBoostClassifier

from sklearn.datasets import make_classification

clf = AdaBoostClassifier(n_estimators=100, random_state=0)

clf.fit(X_train_1, y_train_1)

y_pred = clf.predict(X_test)

print("Test score "+str(clf.score(X_test_1, y_test_1)))

print("Train score "+str(clf.score(X_train_1, y_train_1)))

from sklearn.metrics import classification_report

print("Classification report :")

print(classification_report(y_test_1, y_pred))
from sklearn.ensemble import GradientBoostingClassifier

gbc = GradientBoostingClassifier(random_state=0)

gbc.fit(X_train_1, y_train_1)

y_pred = gbc.predict(X_test_1)

print("Test score "+str(gbc.score(X_test_1, y_test_1)))

print("Train score "+str(gbc.score(X_train_1, y_train_1)))

from sklearn.metrics import classification_report

print("Classification report :")

print(classification_report(y_test_1, y_pred))
from sklearn.utils import resample

X_ = pd.concat([X_train_1, y_train_1], axis=1)



# separate minority and majority classes

label1 = X_[X_.Age== "1"]

label2 = X_[X_.Age== "2"]

label3 = X_[X_.Age== "3"]

# upsample minority

label3_upsampled = resample(label3,

                          replace=True, # sample with replacement

                          n_samples=len(label2), # match number in majority class

                          random_state=27) # reproducible results

label1_upsampled = resample(label1,

                                replace = True, # sample without replacement

                                n_samples = len(label2), # match minority n

                                random_state = 27) # reproducible results



# combine majority and upsampled minority

sampled = pd.concat([label2,label1_upsampled,label3_upsampled])



# check new class counts

print(sampled.Age.value_counts())



y_train_s= sampled.Age

X_train_s = sampled.drop('Age', axis=1)
scaler = StandardScaler()

scaler.fit_transform(X_train_s)
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(random_state=0).fit(X_train_s, y_train_s)

y_pred = clf.predict(X_test_1)

print("Test score "+str(clf.score(X_test_1, y_test_1)))

print("Train score "+str(clf.score(X_train_s, y_train_s)))

from sklearn.metrics import classification_report

print("Classification report :")

print(classification_report(y_test_1, y_pred))
from sklearn.neighbors import KNeighborsClassifier

neigh = KNeighborsClassifier(n_neighbors=3)

neigh.fit(X_train_s, y_train_s)

y_pred = neigh.predict(X_test)

print("Test score "+str(neigh.score(X_test_1, y_test_1)))

print("Train score "+str(neigh.score(X_train_s, y_train_s)))

from sklearn.metrics import classification_report

print("Classification report :")

print(classification_report(y_test_1, y_pred))



error_rate = []

for i in range(1,40):

    knn = KNeighborsClassifier(n_neighbors= i)

    knn.fit(X_train, y_train)

    y_predi = knn.predict(X_test)

    error_rate.append(np.mean(y_test != y_predi))

    

plt.figure(figsize = (10,8))

plt.plot(range(1,40), error_rate, color='blue', linestyle='dashed', marker='o', markerfacecolor='red', markersize=10)

plt.show()
from sklearn.svm import SVC

from sklearn.pipeline import make_pipeline



from sklearn.preprocessing import StandardScaler



regr = make_pipeline(StandardScaler(), SVC(gamma = 'auto'))



regr.fit(X_train_s, y_train_s)



y_pred = regr.predict(X_test_1)

print("Test score "+str(regr.score(X_test_1, y_test_1)))

print("Train score "+str(regr.score(X_train_s, y_train_s)))

from sklearn.metrics import classification_report

print("Classification report :")

print(classification_report(y_test_1, y_pred))