# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Libraries for data visualizations
import matplotlib.pyplot as plt #matplotlib
import matplotlib as mpl 
import seaborn as sns; sns.set() #seaborn
#import chart_studio.plotly as py # plotly library to make visualizations
import plotly.graph_objs as go
from sklearn import linear_model
import warnings
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
warnings.filterwarnings('ignore') #ignoring warnings
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import graphviz
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
def create_label_encoder_dict(df):
    from sklearn.preprocessing import LabelEncoder
    
    label_encoder_dict = {}
    for column in df.columns:
        # Only create encoder for categorical data types
        if not np.issubdtype(df[column].dtype, np.number) and column != 'Age':
            label_encoder_dict[column]= LabelEncoder().fit(df[column])
    return label_encoder_dict
data_frame = pd.read_csv("../input/diseases/dataset.csv")
backup=data_frame
data_frame

df= data_frame[['Disease_CUI', 'Symptom_CUI', 'BMI_Level', 'Age','Gender']]
#Transformation of data in dataset

# label encoding for gender
from sklearn.preprocessing import LabelEncoder #importing label encoder from sklearn

# creating the encoder
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])

df['Gender'].value_counts()
#binning bmi 
# 0 - underweight,1- normal, 2- overwifgt, 3- obese

bins = [0,18.5,24.9,29.9, 60]
labels=[0,1,2, 3]
df['BMI_Level'] = pd.cut(df['BMI_Level'], bins=bins, labels=labels, include_lowest=True)

df
#binning age



bins = [0, 14, 24, 34, 54, 74 ]
labels=[0,1,2, 3, 4]
df['Age'] = pd.cut(df['Age'], bins=bins, labels=labels, include_lowest=True)

df
disease_meta= backup[['Disease', 'Disease_CUI']]
#Transformation of data in dataset

# label encoding for gender
from sklearn.preprocessing import LabelEncoder #importing label encoder from sklearn

# creating the encoder
le = LabelEncoder()
disease_meta['disease_label'] = le.fit_transform(df['Disease_CUI'])
disease_meta['disease_label'].value_counts()
disease_meta['Symptom']= backup[['Symptoms']]
disease_meta['Symptom_CUI']= backup[['Symptom_CUI']]
disease_meta
#Transformation of data in dataset

# label encoding for gender
from sklearn.preprocessing import LabelEncoder #importing label encoder from sklearn

# creating the encoder
le = LabelEncoder()
disease_meta['symptom_label'] = le.fit_transform(backup['Symptoms'])
disease_meta['symptom_label'].value_counts()
disease_meta.to_csv('mycsvfile.csv',index=False)
#Transformation of data in dataset

# label encoding for gender
from sklearn.preprocessing import LabelEncoder #importing label encoder from sklearn

# creating the encoder
le = LabelEncoder()
df['Disease_CUI'] = le.fit_transform(df['Disease_CUI'])
df['Disease_CUI'].value_counts()
#Transformation of data in dataset

# label encoding for gender
from sklearn.preprocessing import LabelEncoder #importing label encoder from sklearn

# creating the encoder
le = LabelEncoder()
df['Disease_CUI'] = le.fit_transform(df['Disease_CUI'])
df['Disease_CUI'].value_counts()
#Transformation of data in dataset

# label encoding for gender
from sklearn.preprocessing import LabelEncoder #importing label encoder from sklearn

# creating the encoder
le = LabelEncoder()
df['Symptom_CUI'] = le.fit_transform(df['Symptom_CUI'])
df['Symptom_CUI'].value_counts()

#splitting the dataset into dependent and independent variables
x = df.drop(['Disease_CUI'], axis = 1) #Independent Variables
y = df['Disease_CUI'] #Dependent variable
# splitting the dataset into training and testing sets
#Most data is used for training

from sklearn.model_selection import train_test_split #importing train_test_split from sklearn

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 45) #setting the test size and randomly generate the data

reg = linear_model.LinearRegression()
reg.fit(X_train,y_train)
print("Regression Coefficients")
pd.DataFrame(reg.coef_,index=X_train.columns,columns=["Coefficient"])
# Make predictions using the testing set
test_predicted = reg.predict(X_test)
df3 = X_test.copy()
df3['predicted Y']=test_predicted
df3['Actual Y']=y_test
df3.head(20)
sns.residplot(test_predicted, y_test, lowess=True, color="g")
#trying to see if this plot gives better results
# it does, whats the diff

df3.plot.scatter(
    x='predicted Y',
    y='Actual Y',
    figsize=(12,8)
)

plt.suptitle("Predicted Y vs Actual Y",size=12)
plt.ylabel("Actual Y")
plt.xlabel("Predicted Y")
#what does this r2 value imply?
print('R squared score is %.2f' % r2_score(y_test, test_predicted))

e2 = pd.read_csv("../input/mycsvf/mycsvfile.csv")
e2['age']=df['Age']

#splitting the dataset into dependent and independent variables
x = df.drop(['Disease_CUI'], axis = 1) #Independent Variables
y = df['Disease_CUI'] #Dependent variable

X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=0.30)
clf = DecisionTreeClassifier(criterion='entropy',min_samples_split=2)
clf.fit(X_train, y_train)
pd.DataFrame([ "%.2f%%" % perc for perc in (clf.feature_importances_ * 100) ], index = X_data.columns, columns = ['Matrix of how factors affect Disease '])
df
clf = DecisionTreeClassifier(criterion='entropy',min_samples_split=2) 
clf.fit(X_train, y_train)
class_names = np.unique([str(i) for i in y_train])
class_names
dot_data = tree.export_graphviz(clf,out_file=None, 
                                feature_names=X_data.columns, 
                                class_names=class_names,
                                max_depth=7,
                         filled=True, rounded=True,  proportion=True,
                                node_ids=True, #impurity=False,
                         special_characters=True)
graph = graphviz.Source(dot_data) 
graph