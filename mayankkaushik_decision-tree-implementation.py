import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt 

import seaborn as sns 

%matplotlib inline


from sklearn.model_selection import  train_test_split



from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier 





df = pd.read_csv("../input/dataa.csv")
df.head()




df.describe()



df.shape
df.dtypes
sns.countplot(df['diagnosis'])
data_M = df[df['diagnosis']=='M']

data_B = df[df['diagnosis']=='B']

df.diagnosis.value_counts().plot(kind ='bar')

plt.title("Histogram of Breast Cancer Diagnosis results")

plt.ylabel("Frequency")
fig, ax = plt.subplots(figsize=(12,8))

ax.scatter(df['smoothness_mean'], df['compactness_mean'], s=50, c='b', marker='o', label='Malignant')

ax.scatter(df['smoothness_mean'], df['compactness_mean'], s=50, c='r', marker='x', label='Benign')

ax.legend()

ax.set_xlabel('smoothness mean')

ax.set_ylabel('compactness mean')
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = model_selection.train_test_split(df.loc[:,'radius_mean':], df['diagnosis'], test_size=0.3, random_state=0)
x_train.shape, y_train.shape, x_test.shape, y_test.shape


 
x_train,x_test,y_train,y_test=train_test_split(x,y)


clf = DecisionTreeClassifier()  #defining DecisionTree Classifier

clf.fit(x_train,y_train) 

y_pred=clf.predict(x_test)


model = RandomForestClassifier()

classification_model(model,train,predictor_var,outcome_var)


