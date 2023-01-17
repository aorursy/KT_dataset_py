import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
print(os.listdir("../input"))
df = pd.read_csv('../input/diabetes.csv')
df.head()
df.info()
df.describe()
sns.heatmap(df.isnull(),yticklabels=False,cmap='viridis',cbar=False)
sns.set_style('whitegrid')
sns.countplot(x='Outcome',data=df)
sns.pairplot(df,hue='Outcome')
plt.figure(figsize=(14,6))
sns.heatmap(df.corr(),annot=True,cmap='viridis')
from sklearn.preprocessing import StandardScaler
X=df.iloc[:,0:-1].values
Y=df.iloc[:,-1].values
sc=StandardScaler()
X_scaled=sc.fit_transform(X)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X_scaled, Y, test_size=0.3)
logr = LogisticRegression()
logr.fit(X_train,y_train)
logr_predictions = logr.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,logr_predictions))




