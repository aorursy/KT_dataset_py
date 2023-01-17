import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

data = pd.read_csv("../input/creditcard.csv")
data.head()
data.drop('Time',1,inplace=True)
X = data.iloc[:,:-1]
X.head()
y = data.iloc[:,-1]
y.head()
# Normalizing the amount attribute
from sklearn.preprocessing import StandardScaler

data['nAmount'] = StandardScaler().fit_transform(data['Amount'].reshape(-1, 1))
data = data.drop(['Amount'],axis=1)
data.head()