# Data Loading and Processing
import pandas as pd
pd.set_option('max_rows',5)
# Visualization
import matplotlib.pyplot as plt
# Linear Algebra
import numpy as np
# ML 
from sklearn import svm
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics

# We start by loading the data
data = pd.read_csv('../input/voice.csv')
df = pd.DataFrame(data)
df.head(5)
# No missing data
df.describe()
# Our features are well correlated 
df.corr()
df.plot()
# Let's get some useful insights
info = pd.Series([df.shape[0],df.shape[1]],index=['Number of Instances','Number of Features'])
info
# Our labels are balanced
df.label.value_counts()
# Separating our features and labels
features = df.iloc[:,:-1]
labels = df.iloc[:,-1]
# Transform the data 
scaler = StandardScaler()
scaler.fit(features)
features = scaler.transform(features)
# Map the label values
df['Slabel'] = df.label.map({"male":1,"female":0})
labels = df.Slabel.as_matrix()
# Split our data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(features,labels,test_size=0.2)
# Create classifier, feed data and evaluate it.
clf = svm.SVC()
clf.fit(x_train,y_train)
pred = clf.predict(x_test)
acc = metrics.accuracy_score(pred,y_test)
print(acc)

