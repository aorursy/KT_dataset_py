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
df=pd.read_csv('/kaggle/input/pima-indians-diabetes-database/diabetes.csv')
df.head()
df.describe()
#better to replace zeros with nan since after that counting them 
#would be easier and zeros need to be replaced with suitable values
df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = df[['Glucose',
  'BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)

#Lets check the missing value of the dataset
df.isnull().sum() 
#To fill these Nan values the data distribution needs to be understood
f, axes = plt.subplots(4,2, figsize=(20,20))
sns.distplot(df.Pregnancies, ax=axes[0,0])
sns.distplot(df.Glucose, ax=axes[0,1])
sns.distplot(df.BloodPressure, ax=axes[1,0])
sns.distplot(df.SkinThickness, ax=axes[1,1])
sns.distplot(df.Insulin, ax=axes[2,0])
sns.distplot(df.BMI, ax=axes[2,1])
sns.distplot(df.DiabetesPedigreeFunction, ax=axes[3,0])
sns.distplot(df.Age, ax=axes[3,1])
#Now we replace the missing value
df['Glucose'] = df['Glucose'].fillna(df['Glucose'].mean())
df['BloodPressure'] = df['BloodPressure'].fillna(df['BloodPressure'].mean())
df['SkinThickness'] = df['SkinThickness'].fillna(df['SkinThickness'].median())
df['Insulin'] = df['Insulin'].fillna(df['Insulin'].mean())
df['BMI'] = df['BMI'].fillna(df['BMI'].mean())
#Lets Check our target variable
#Here zeros means not diabetic and one means diabetic
import seaborn as sns
sns.countplot(df['Outcome'], label = 'Count')
corr=df.corr()
corr


#Now lets see the correlation by plotting heatmap
import matplotlib.pyplot as plt
colormap = sns.diverging_palette(100, 15, as_cmap = True)
plt.figure(figsize = (8,6))
sns.heatmap(corr,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values,
            annot=True,fmt='.2f',linewidths=0.30,
            cmap = colormap, linecolor='white')
plt.title('Correlation of df Features', y = 1.05, size=10)
#Lets look the correlation score
print (corr['Outcome'].sort_values(ascending=False), '\n')
#we can see that % of Glucose is the bigest one that have correlation with the outcome that's why diabetes prediction depend essentially in the rate of this feature#Lets take our matrix of features and target variable
x = df.iloc[:, 0:8].values
y = df.iloc[:, -1].values
#Lets take our matrix of features and target variable
x = df.iloc[:, 0:8].values
y = df.iloc[:, -1].values
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 52)
 #Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


from sklearn.linear_model import LogisticRegression
import sklearn.metrics as metrics

diab_lr_model = LogisticRegression(C=0.7, random_state=52)
diab_lr_model.fit(x_train, y_train.ravel())
lr_test_predict = diab_lr_model.predict(x_test)

print("Model Accuracy: {0:.2f}".format(metrics.accuracy_score(y_test, lr_test_predict)))
print("")
print("Confusion Matrix")
print(metrics.confusion_matrix(y_test, lr_test_predict, labels=[1, 0]))
print("")
print("Classification Report")
print(metrics.classification_report(y_test, lr_test_predict, labels=[1, 0]))
from sklearn.naive_bayes import GaussianNB # I am using Gaussian algorithm from Naive Bayes

# Lets creat the model
GNB = GaussianNB()
GNB.fit(x_train, y_train.ravel())
GNB_train_predict= GNB.predict(x_train)
GNB_test_predict= GNB.predict(x_test)

from sklearn import metrics

print("Model Accuracy: {0:.4f}".format(metrics.accuracy_score(y_train, GNB_train_predict)))
print()

print("Confusion Matrix")
print(metrics.confusion_matrix(y_test, GNB_test_predict, labels=[1, 0]))
print("")
print("Classification Report")
print(metrics.classification_report(y_test, GNB_test_predict, labels=[1, 0]))
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(random_state=52)
rf_model.fit(x_train, y_train.ravel())
rf_train_predict = rf_model.predict(x_train)
rf_test_predict = rf_model.predict(x_test)

print("Model Accuracy: {0:.2f}".format(metrics.accuracy_score(y_train, rf_train_predict)))
print("Confusion Matrix")
print(metrics.confusion_matrix(y_test, rf_test_predict, labels=[1, 0]))
print("")
print("Classification Report")
print(metrics.classification_report(y_test, rf_test_predict, labels=[1, 0]))
!pip install pydotplus
from sklearn.tree import DecisionTreeClassifier
import sklearn.metrics as metrics
dtree=DecisionTreeClassifier()
dtree.fit(x_train,y_train.ravel())
dtree_train_predict = dtree.predict(x_train)
print("Model Accuracy: {0:.2f}".format(metrics.accuracy_score(y_train, dtree_train_predict)))

from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
dot_data = StringIO()
export_graphviz(dtree, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())
