#load packages



import pandas as pd 

import matplotlib 

import numpy as np 

import sklearn #collection of machine learning algorithms



#Algs

from sklearn import  tree, linear_model



#Helpers

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from sklearn import feature_selection

from sklearn import model_selection

from sklearn import metrics



#Viz

import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.pylab as pylab

import seaborn as sns

from pandas.tools.plotting import scatter_matrix



#Configure Visualization Defaults

#%matplotlib inline = show plots in Jupyter Notebook browser

%matplotlib inline

mpl.style.use('ggplot')

sns.set_style('white')

pylab.rcParams['figure.figsize'] = 12,8
data_raw = pd.read_csv('../input/Train_AdquisicionAhorro.csv')

data_val = pd.read_csv('../input/Test_AdquisicionAhorro.csv')
data1 = data_raw.copy(deep = True)

data2 = data_val.copy(deep = True)

data_cleaner = [data1, data_val]

print (data_raw.info()) 

print('Train null values:\n', data1.isnull().sum())

print("-"*10)



print('Test/Validation null values:\n', data_val.isnull().sum())

print("-"*10)
data_raw.describe(include = 'all')
drop_column = ['coddoc']

data1.drop(drop_column, axis=1, inplace = True)

Target = ['Adq_Ahorro']



#creando dummys para las variables categoricas

data1_dummy = pd.get_dummies(data1)

data1_x_dummy =[ 'edad', 'balance', 'dia', 'duracion', 'campana', 'pdias', 'previo',  'estciv_divorced', 'estciv_married', 'estciv_single', 'educacion_desconocido', 'educacion_primaria', 'educacion_secundaria', 'educacion_terciario', 'mora_no', 'mora_si', 'vivienda_no', 'vivienda_si', 'prestamo_no', 'prestamo_si']

data1_xy_dummy = Target + data1_x_dummy

print('Dummy X Y: ', data1_xy_dummy, '\n')



data1_dummy.head()
# Splitting the dataset into the Training set and Test set

train1_x_dummy, test1_x_dummy, train1_y_dummy, test1_y_dummy = model_selection.train_test_split(data1_dummy[data1_x_dummy], data1[Target], test_size = 0.2, random_state = 0)

print("Data1 Shape: {}".format(data1.shape))

print("Train1 Shape: {}".format(train1_x_dummy.shape))

print("Test1 Shape: {}".format(test1_x_dummy.shape))


from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(train1_x_dummy)

X_test = sc.transform(test1_x_dummy)
from sklearn.decomposition import PCA

pca = PCA(n_components = 11)

X_train = pca.fit_transform(X_train)

X_test = pca.transform(X_test)

explained_variance = pca.explained_variance_ratio_

print(explained_variance)

var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)

print(var1)

plt.plot(var1)
y_train = train1_y_dummy

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state = 0)

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

y_pred_prob = classifier.predict_proba(X_test)[:,1]



from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report

from sklearn import metrics





y_test = test1_y_dummy

cm = confusion_matrix(y_test, y_pred)

print(cm)

print(accuracy_score(y_test, y_pred))

print(classification_report(y_test, y_pred))

fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_prob)

print(metrics.auc(fpr, tpr))