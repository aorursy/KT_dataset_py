import pandas as pd

import numpy as np

import os

import glob

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.model_selection import train_test_split

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import PolynomialFeatures

from sklearn.impute import SimpleImputer
train = '/kaggle/input/sepsisb/training_setB/'

print('training setA have {} files'.format(len(os.listdir(train))))
os.chdir('/kaggle/input/sepsisb/training_setB/')

extension ='psv'

filenames = [i for i in glob.glob('*{}'.format(extension))]
trainn = pd.concat([pd.read_csv(f , sep='|') for f in filenames])

trainn.to_csv(r'trainn.csv')
from IPython.display import FileLink

FileLink('/kaggle/input/trainn.csv/')
X = trainn.drop(columns =['SepsisLabel','PaCO2','SBP','MAP','pH','DBP','PTT','FiO2','Potassium','Lactate','Glucose','AST','TroponinI','Hct', 'Hgb','O2Sat','Unit1','Unit2'])
X.head()
y = trainn['SepsisLabel']
y.head()
imputer = SimpleImputer(strategy="median")

imputer.fit(X)

X = imputer.transform(X)
clf = ExtraTreesClassifier( n_estimators=100, criterion="entropy", max_features="auto", min_samples_leaf=1, min_samples_split=5)



clf.fit(X,y)
import pickle
filename = '/kaggle/working/model_v1.pkl'

with open( filename, 'wb') as file:

    pickle.dump(clf, file) 
with open(filename ,'rb') as f:

    loaded_model = pickle.load(f)

loaded_model
X_test = np.array([120,21,16,43,34,25,45,22,45,6,104,0.9,20,8.6,8.9,52,2.2,22,32,28,0,-0.98,258])

X_test = X_test.reshape(1,-1)

y_predict = clf.predict(X_test)
y_predict
loaded_model.predict(X_test)