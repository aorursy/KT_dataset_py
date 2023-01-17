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
Heart_data =  pd.read_csv("/kaggle/input/heart-disease-uci/heart.csv")
Heart_data.head(10)
Heart_data.describe()
Heart_data.info()
Heart_data.shape
# Does the dataset contain null values?
Heart_data.isnull().sum()
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

plt.style.available
plt.figure(figsize=(25,35))
for i, col in enumerate(['trestbps', 'chol','thalach','oldpeak', 'trestbps', 'ca','thal', 'exang']):
    plt.subplot(4,2,i+1)
    sns.kdeplot(Heart_data[col],shade=True)
plt.show()
plt.style.use('ggplot')

plt.figure(figsize=(14, 10))

sns.heatmap(data=Heart_data.corr(), annot = True)
plt.title('Heatmap for the Dataset', fontsize = 20)
plt.show()
new_values={"sex":{1:"Male",0:"Female"},
         "cp":{0:"typical angina",1: "non-anginal pain" ,2: "atypical angina" ,3: "asymptomatic"},
         "fbs":{0:"<=120",1:">120"},
         "exang":{0:"no",1:"yes"},
         "restecg" :{0:"normal" ,1:"ST-T wave abnormality",2:"probable or definite left ventricular hypertrophy"},
         "target" :{ 0:"No Heart Disease",1 : "heart-disease"},
         "thal" :{ 1 : "fixed defect",0 : "normal",2 : "reversable defect",3:"NA"}
         
}

Heart_data_copy = Heart_data.copy()
Heart_data_copy.replace(new_values,inplace=True)

sns.countplot(Heart_data_copy['target'])
Heart_data_copy.head()
# Plot the count of each gender
sns.countplot(Heart_data_copy['sex'])
sns.catplot('target',col='sex',data=Heart_data_copy,kind='count')
# to make sure exactly then use  pd.crosstab() --> Compute a simple cross tabulation of two (or more) factors.
pd.crosstab(Heart_data_copy.target, Heart_data_copy.sex)
sns.catplot(x="target", y="age", hue="sex",data=Heart_data_copy, palette='rainbow', kind='box') 
plt.figure(figsize=(20,7))
sns.countplot("age", hue="target",data=Heart_data_copy, palette='rainbow') 
plt.figure(figsize=(10,10))

sns.countplot("target", hue='cp', data=Heart_data_copy, palette='rainbow')
sns.lmplot(x='age',y='thalach', hue='target', data=Heart_data_copy)
plt.title("Heart Disease in function of Age and Max Heart Rate")

plt.figure(figsize=(15,7))

sns.lineplot(x='age',y='thalach', hue='target', data=Heart_data_copy)
sns.boxplot(x='target',y='thalach', data=Heart_data_copy, palette='rainbow')
plt.figure(figsize=(12,10))

sns.countplot("target", hue='restecg' , data=Heart_data_copy, palette='rainbow')
sns.countplot("target", hue='thal' , data=Heart_data_copy, palette='rainbow')
sns.catplot(x="target",y="trestbps",data=Heart_data_copy, palette='rainbow' ,kind='violin')
plt.show()
sns.catplot(x="target",y="chol",data=Heart_data_copy, palette='rainbow' ,kind='box')
sns.countplot("target", hue='fbs' , data=Heart_data_copy, palette='rainbow')
sns.countplot("target", hue='exang' , data=Heart_data_copy, palette='rainbow')
sns.catplot(x="target",y="oldpeak",data=Heart_data_copy, palette='rainbow', kind="box")
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler


# define min max scaler
scaler = MinMaxScaler()
# transform data
MinMaxScaled = scaler.fit_transform(Heart_data)

# plot both data together to compare

fig, ax = plt.subplots(1,2)
sns.distplot(Heart_data, ax=ax[0])
ax[0].set_title("Original Data")

sns.distplot(MinMaxScaled, ax=ax[1])
ax[1].set_title("MinMaxScaled data")
MinMaxScaled
# define RobustScaler scaler
scaler = RobustScaler()
# transform data
RobustScaled = scaler.fit_transform(Heart_data)

# plot both data together to compare

fig, ax = plt.subplots(1,2)
sns.distplot(Heart_data, ax=ax[0])
ax[0].set_title("Original Data")

sns.distplot(RobustScaled, ax=ax[1])
ax[1].set_title("RobustScaled data")
RobustScaled
# define StandardScaler scaler
scaler = StandardScaler()
# transform data
StandardScaled = scaler.fit_transform(Heart_data)

# plot both data together to compare

fig, ax = plt.subplots(1,2)
sns.distplot(Heart_data, ax=ax[0])
ax[0].set_title("Original Data")

sns.distplot(StandardScaled, ax=ax[1])
ax[1].set_title("StandardScaled data")
StandardScaled
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split

X = Heart_data.drop('target',axis=1)
y = Heart_data['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

# Keep 5 features
selector = SelectKBest(f_classif, k=5)

X_new = selector.fit_transform(X_train, y_train)
X_new
# Get back the features we've kept, zero out all other features
selected_features = pd.DataFrame(selector.inverse_transform(X_new), 
                                 columns=Heart_data.columns.drop('target'))

# Dropped columns have values of all 0s, so var is 0, drop them
selected_columns = selected_features.columns[selected_features.var() != 0.0]

# Get the valid dataset with the selected features.
X_test[selected_columns].head()
# Keep 7 features
selector = SelectKBest(f_classif, k=9)

X_new = selector.fit_transform(X_train, y_train)
X_new
# Get back the features we've kept, zero out all other features
selected_features = pd.DataFrame(selector.inverse_transform(X_new), 
                                 columns=Heart_data.columns.drop('target'))
# Dropped columns have values of all 0s, so var is 0, drop them
selected_columns = selected_features.columns[selected_features.var() != 0.0]

# Get the valid dataset with the selected features.
X_test[selected_columns].head()
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel

X = Heart_data.drop('target',axis=1)
y = Heart_data['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

# Set the regularization parameter C=0.1
logistic = LogisticRegression(C=0.1, penalty="l1", solver='liblinear', random_state=7).fit(X, y)
model = SelectFromModel(logistic, prefit=True)

X_new = model.transform(X)
X_new
# Get back the features we've kept, zero out all other features
selected_features = pd.DataFrame(model.inverse_transform(X_new), 
                                 columns=Heart_data.columns.drop('target'))

# Dropped columns have values of all 0s, keep other columns 
selected_columns = selected_features.columns[selected_features.var() != 0]
X_test[selected_columns].head()
# Set the regularization parameter C= 0.5
logistic = LogisticRegression(C=0.5, penalty="l1", solver='liblinear', random_state=7).fit(X, y)
model = SelectFromModel(logistic, prefit=True)

X_new = model.transform(X)

selected_features = pd.DataFrame(model.inverse_transform(X_new), 
                                 columns=Heart_data.columns.drop('target'))

selected_columns = selected_features.columns[selected_features.var() != 0]
X_test[selected_columns].head()
# Set the regularization parameter C=0.1 and L2
logistic = LogisticRegression(C=0.1, penalty="l2", solver='liblinear', random_state=7).fit(X, y)
model = SelectFromModel(logistic, prefit=True)

X_new = model.transform(X)

selected_features = pd.DataFrame(model.inverse_transform(X_new), 
                                 columns=Heart_data.columns.drop('target'))

selected_columns = selected_features.columns[selected_features.var() != 0]
X_test[selected_columns].head()
# Set the regularization parameter C=1 and L2
logistic = LogisticRegression(C=1, penalty="l2", solver='liblinear', random_state=7).fit(X, y)
model = SelectFromModel(logistic, prefit=True)

X_new = model.transform(X)

selected_features = pd.DataFrame(model.inverse_transform(X_new), 
                                 columns=Heart_data.columns.drop('target'))

selected_columns = selected_features.columns[selected_features.var() != 0]
X_test[selected_columns].head()