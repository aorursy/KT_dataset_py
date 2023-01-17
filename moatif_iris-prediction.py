# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # Data Visualization

import seaborn as sns # Data Visualization

plt.rcParams['figure.figsize'] = [12,7] # setting figsize permanantly

from sklearn.preprocessing import LabelEncoder,MinMaxScaler # preprocessing and normalizing techniques



from sklearn.model_selection import train_test_split # Splitting data

from sklearn.neighbors import KNeighborsClassifier # K nearest neighbours classifier

from sklearn.svm import SVC # Support Vector Classifier

from sklearn.tree import DecisionTreeClassifier # Decision Tree Classifier



from sklearn.metrics import confusion_matrix,classification_report # metrics



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# Reading data into dataframe

df=pd.read_csv('/kaggle/input/iris/Iris.csv') 
# Printing first five records in dataframe with dataset columns/features.

df.head()
df.columns
df.shape
# Dropping Id column

df.drop(columns=['Id'],inplace=True)
# Printing unique Species

df['Species'].unique()
# Frequency plot for each species

sns.countplot(df['Species'])
# Five number Summary

df.describe().transpose()
# Columns info

df.info()
# Checking Missing values

df.isna().sum()
for col in ['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']:

    sns.boxplot(df[col])

    plt.show()
# Scatterplot of features with hue as Species.

sns.pairplot(df,hue='Species')
df.columns
sns.violinplot(x='Species',y='SepalLengthCm',data=df)
sns.violinplot(x='Species',y='SepalWidthCm',data=df)
sns.violinplot(x='Species',y='PetalLengthCm',data=df)
sns.violinplot(x='Species',y='PetalWidthCm',data=df)
labelencoder=LabelEncoder()
df['Species']=labelencoder.fit_transform(df['Species'])
df.head()
minmax=MinMaxScaler()
col=df.drop(columns=['Species']).columns
df_encoded=minmax.fit_transform(df[col])
df_encoded=pd.DataFrame(df_encoded,columns=df.drop(columns=['Species']).columns)

df_encoded.head()
df_encoded['Species']=df['Species']
df_encoded.head()
sns.heatmap(df_encoded.corr(),annot=True,cmap='Blues')
df_encoded[['SepalLengthCm','PetalWidthCm']].var()
X=df_encoded.drop(columns=['Species','SepalWidthCm','PetalLengthCm'])

y=df_encoded['Species']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)
model_knn=KNeighborsClassifier(n_neighbors=5)

model_knn.fit(X_train,y_train)

print("KNN R2 score:",model_knn.score(X_test,y_test))

y_pred_knn=model_knn.predict(X_test)

print("\n\nConfusion Matrix:")

print(pd.DataFrame(confusion_matrix(y_pred_knn,y_test)))

print("\n\nClassification Report:")

print(classification_report(y_pred_knn,y_test))
model_dt=DecisionTreeClassifier(max_depth=3,min_samples_split=5,min_impurity_decrease=0.05)

model_dt.fit(X_train,y_train)

print("Decision Tree R2 score:",model_dt.score(X_test,y_test))

y_pred_dt=model_knn.predict(X_test)

print("\n\nConfusion Matrix:")

print(pd.DataFrame(confusion_matrix(y_pred_dt,y_test)))

print("\n\nClassification Report:")

print(classification_report(y_pred_dt,y_test))
model_svc=SVC(kernel='rbf')

model_svc.fit(X_train,y_train)

print("SVC model R2 score:",model_svc.score(X_test,y_test))

y_pred_svc=model_svc.predict(X_test)

print("\n\nConfusion Matrix:")

print(pd.DataFrame(confusion_matrix(y_pred_svc,y_test)))

print("\n\nClassification Report:")

print(classification_report(y_pred_svc,y_test))