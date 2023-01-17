# Imported Libraries

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")
!pip install pycaret
# from google.colab import drive
# drive.mount('/content/drive')
# data = pd.read_csv("/content/drive/My Drive/creditcard.csv")
# data.head(10)
data = pd.read_csv("/kaggle/input/creditcardfraud/creditcard.csv")
data.head(10)
data.shape
data.info()
sns.countplot(data['Class'])
noFraud = len(data[data.Class == 0.000000])
Fraud = len(data[data.Class == 1.000000])
print("Fair trasactions: {:.2f}%".format((noFraud / (len(data.Class))*100)))
print("Fraud trasactions: {:.2f}%".format((Fraud / (len(data.Class))*100)))
# Importing module and initializing setup
from pycaret.classification import *
clf1 = setup(data = data, target = 'Class')
compare_models(fold=5)
# creating logistic regression model
model = create_model('xgboost')
model
model=tune_model('xgboost')
 