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

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline 
df = pd.read_csv("../input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv", engine = "python")

df.head()
df.info()
df.isnull().sum()
#Replacing all the null values with zero

df['salary'].fillna(0, inplace = True)
data = df



status = {'Placed': 1,'Not Placed': 0} 

data['status'] = [status[item] for item in data['status']] 
df.describe()
def myplot(data,x,y):

    plt.Figure(figsize =(12,12))

    sns.boxplot(x = data[x],y= data[y])

    g = sns.FacetGrid(data, row = y)

    g = g.map(plt.hist,x)

    plt.show()
sns.set_style("ticks")

myplot(data,"salary","ssc_b")
sns.set_style("ticks")

myplot(data, "salary", "gender")

sns.countplot(data['status'],hue=data['gender'])
fig, axes = plt.subplots(2,3, figsize=(12,12))

sns.barplot(x="hsc_s", y="status", data=data, ax = axes[(0,0)] )

sns.barplot(x="hsc_s", y="hsc_p", data=data, ax = axes[(0,1)])

sns.barplot(x="degree_t", y="status", data=data, ax = axes[(0,2)])

sns.barplot(x="status", y="degree_p", data=data, ax = axes[(1,0)])

sns.barplot(x="degree_t", y="degree_p", data=data, ax = axes[(1,1)])

plt.tight_layout(pad = 3)
fig, axes = plt.subplots(2,2, figsize=(12,12))

sns.barplot(x="workex", y="status", data=data, ax = axes[(0,0)])

sns.barplot(x="status", y="etest_p", data=data, ax = axes[(0,1)])

sns.barplot(x="specialisation", y="status", data=data, ax = axes[(1,0)])

sns.barplot(x="status", y="mba_p", data=data, ax = axes[(1,1)])

plt.tight_layout(pad = 3)
# encoding for the features



import category_encoders as ce

encoder = ce.BackwardDifferenceEncoder(cols=['ssc_b', "hsc_b", "hsc_s", "degree_t", "workex", "specialisation", "gender"])

data_new = encoder.fit_transform(data)



data_new.head()

data_new.drop(['intercept','sl_no'], axis=1, inplace=True)

data_new.head()
x = data_new["salary"]

labels = data_new['status']

features = data_new.iloc[:, :-2 ]

features = pd.concat([features, x], axis=1, join='inner')

features.head()

from sklearn.feature_selection import RFE

from sklearn.linear_model import LogisticRegression



# feature extraction

model = LogisticRegression(solver='lbfgs')

rfe = RFE(model, 10)

fit = rfe.fit(features, labels)

print("Num Features: %d" % fit.n_features_)

print("Selected Features: %s" % fit.support_)

print("Feature Ranking: %s" % fit.ranking_)
#dropping the features

features.drop(["gender_0", "ssc_b_0", "hsc_b_0", "degree_t_0", "specialisation_0"], axis=1, inplace=True)

features.head()
df_final = features

df_final.to_numpy()

distinct_labels = list(set(labels))
from sklearn.manifold import TSNE

import matplotlib.patheffects as PathEffects



y = labels

X_raw = df_final

y_raw = np.array(y, dtype = 'int')

tsne = TSNE(n_components=2, random_state=0, perplexity = 50, n_iter = 5000)

X_2d = tsne.fit_transform(X_raw)

X1 = X_2d[:,0:1]

Y1 = X_2d[:,1:2]



sns.set_style('ticks')

sns.set_palette('muted')

sns.set_context("notebook", font_scale=1.5,

                rc={"lines.linewidth": 2.5})





category_to_color = {0: 'red', 1: 'blue'}

category_to_label = {0: 'Unplaced', 1:"Placed"}



fig, ax = plt.subplots(1,1)

for category, color in category_to_color.items():

    mask = y == category

    ax.plot(X_2d[mask, 0], X_2d[mask, 1], 'o',

            color=color, label=category_to_label[category], ms = 6)



ax.legend(loc='best')

ax.axis('on')

ax.axis('tight')

plt.xlabel('Dimension1')

plt.ylabel('Dimension2')

plt.title(' t-SNE plot')
from sklearn import preprocessing

scaler = preprocessing.StandardScaler()

x = scaler.fit_transform(X_raw)

y = y_raw
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.metrics import classification_report,confusion_matrix



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.18)
model1 = RandomForestClassifier()

model1.fit(x_train,y_train)

model1.score(x_test,y_test)

predictions1 = model1.predict(x_test)

print(confusion_matrix(y_test,predictions1))

print(classification_report(y_test,predictions1))
model2 = DecisionTreeClassifier()

model2.fit(x_train,y_train)



predictions2 = model2.predict(x_test)

print(confusion_matrix(y_test,predictions2))

print(classification_report(y_test,predictions2))
model3 = KNeighborsClassifier()

model3.fit(x_train,y_train)



predictions3 = model3.predict(x_test)

print(confusion_matrix(y_test,predictions3))

print(classification_report(y_test,predictions3))
model4 = XGBClassifier()

model4.fit(x_train,y_train)

predictions4 = model4.predict(x_test)

print(confusion_matrix(y_test,predictions4))

print(classification_report(y_test,predictions4))
model5 = GaussianNB()

model5.fit(x_train,y_train)

predictions5 = model5.predict(x_test)

print(confusion_matrix(y_test,predictions5))

print(classification_report(y_test,predictions5))
#neural network classifier

model6 = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)

model6.fit(x_train,y_train)

predictions6 = model6.predict(x_test)

print(confusion_matrix(y_test,predictions6))

print(classification_report(y_test,predictions6))