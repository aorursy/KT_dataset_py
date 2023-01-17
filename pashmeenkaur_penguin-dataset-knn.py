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
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

path=r"/kaggle/input/palmer-archipelago-antarctica-penguin-data/penguins_size.csv"

df=pd.read_csv(path)
df.head()
df.info()
df.describe()
sns.countplot(x=df['species'])
sns.countplot(x=df['species'],hue=df['sex'])
sns.jointplot(x='culmen_length_mm', y='culmen_depth_mm',data=df)
sns.scatterplot(x='culmen_length_mm', y='culmen_depth_mm',data=df,hue='species')
sns.countplot(x=df['species'],hue=df['island'])
df=pd.get_dummies(df,columns=['sex','island'],drop_first=True)
df.head()
df.info()
sns.heatmap(df.isnull())
df=df.fillna(0)
sns.heatmap(df.isnull())
from sklearn.preprocessing import StandardScaler
scale=StandardScaler()
scale.fit(df.drop(['species'],axis=1))
transformed=scale.transform(df.drop(['species'],axis=1))
df_scaled=pd.DataFrame(transformed,columns=df.columns[1:])

df_scaled.info()
X=df_scaled
y=df['species']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=101)
from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=1)

knn.fit(x_train,y_train)
out1=knn.predict(x_test)
from sklearn.metrics import classification_report,confusion_matrix

print(classification_report(y_test,out1))
print(confusion_matrix(y_test,out1))
error_rate=[]

for i in range(1,20):
    knn_i=KNeighborsClassifier(n_neighbors=i)
    knn_i.fit(x_train,y_train)
    out_i=knn_i.predict(x_test)
    error_rate.append(np.mean(out_i!=y_test))
plt.plot(range(1,20),error_rate,marker='x',markerfacecolor='red')
plt.xlabel('# KNeighbors')
plt.ylabel('Error_rate')
plt.title('Best KNeighbors')