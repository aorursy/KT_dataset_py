import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import preprocessing

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.metrics import f1_score,confusion_matrix

from sklearn.metrics import accuracy_score

df=pd.read_csv("../input/cancer-wisconsin.csv")

# df = df.replace('?', np.NaN) 

df.head()
df.info()
df['Bare Nuclei'].value_counts()
df = df.replace('?', np.NaN)
df.describe()
classes = df.Class.value_counts().keys()                            # Get classes of data set

le = preprocessing.LabelEncoder()

df['Class'] = le.fit_transform(df['Class']) 

# List features for visualization

features = ['Clump Thickness', 'Uniformity of Cell Size',

      'Uniformity of Cell Shape', 'Marginal Adhesion',

      'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin',

      'Normal Nucleoli', 'Mitoses']

# X = df[features].fillna(0)

# X.head()
import seaborn as sns

%matplotlib inline

f, ax = plt.subplots(figsize=(15, 6))

sns.heatmap(data=df.isnull(), yticklabels=False, cbar =False, cmap = 'viridis')
df["Bare Nuclei"].isnull().sum()
df=df.fillna(0)
df.info()
df["Bare Nuclei"] = df['Bare Nuclei'].astype(int)
df.info()
features=df.drop(columns=['Class'])

# sns.set(style="ticks", color_codes=True)

# iris = sns.load_dataset("features")

g = sns.pairplot(df,vars=['Clump Thickness', 'Uniformity of Cell Size',

      'Uniformity of Cell Shape', 'Marginal Adhesion',

      'Single Epithelial Cell Size'])
import seaborn as sns

sns.pairplot(df, hue='Class', vars=['Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion'])
df['Class'].value_counts()
from yellowbrick.target import ClassBalance

f, ax = plt.subplots(figsize=(20, 9))

visualizer = ClassBalance(labels=classes)

visualizer.fit(y_train=df['Class'])

visualizer.poof()
df_train=df.drop(columns=['Class',"Unnamed: 0"])

corrmat = df_train.corr()

f, ax = plt.subplots(figsize=(20, 9))

sns.heatmap(corrmat, vmax=.8, annot=True)
from sklearn.model_selection import train_test_split

features=df.drop(columns=['Class'])

lable=df['Class']

x_train, x_test, y_train, y_test = train_test_split(features, lable, test_size=0.3, random_state=28)
x_train.info()
clf_rf = RandomForestClassifier(random_state=23)      

clr_rf = clf_rf.fit(x_train,y_train)
y_predict = clf_rf.predict(x_test)

accuracy = accuracy_score(y_test, y_predict )

print('Accuracy: ', accuracy)
from yellowbrick.classifier import ConfusionMatrix

from yellowbrick.classifier import ClassPredictionError

f, ax = plt.subplots(figsize=(20, 9))

cm = ConfusionMatrix(model=RandomForestClassifier())

cm.fit(X=x_train, y=y_train)

cm.score(X=x_test, y=y_test)

cm.poof()
from sklearn.ensemble import RandomForestClassifier

from yellowbrick.classifier import ClassPredictionError

f, ax = plt.subplots(figsize=(20, 9))

visualizer = ClassPredictionError(model=RandomForestClassifier())

visualizer.fit(X=x_train, y=y_train)

visualizer.score(X=x_test, y=y_test)

visualizer.poof()
from sklearn.ensemble import RandomForestClassifier

from yellowbrick.classifier import ClassificationReport

f, ax = plt.subplots(figsize=(20, 9))

visualizer = ClassificationReport(model=RandomForestClassifier(), support=True)

visualizer.fit(x_train, y_train)  

visualizer.score(x_test, y_test)  

visualizer.poof()