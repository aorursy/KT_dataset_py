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
df = pd.read_csv(r'/kaggle/input/forest-cover-type-dataset/covtype.csv')
df.shape
df.head()
df.info()
df['Cover_Type'].value_counts()
y = df['Cover_Type']
X = df.drop('Cover_Type',axis=1)
X1 = X[['Elevation','Aspect','Slope','Horizontal_Distance_To_Hydrology','Vertical_Distance_To_Hydrology','Horizontal_Distance_To_Roadways',
      'Hillshade_9am','Hillshade_Noon','Hillshade_3pm','Horizontal_Distance_To_Fire_Points']]
X1.head()
import matplotlib.pyplot as plt
import seaborn as sns
corr=X1.corr()
top_features=corr.index
plt.figure(figsize=(20,20))
sns.heatmap(X1[top_features].corr(),annot=True)
X.drop('Hillshade_9am',axis=1,inplace=True)
from sklearn.feature_selection import mutual_info_classif
mutual_info=mutual_info_classif(X1,y)
mutual_data=pd.Series(mutual_info,index=X1.columns)
mutual_data.sort_values(ascending=False)
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
model=ExtraTreesClassifier()
model.fit(X1,y)
print(model.feature_importances_)
ranked_features=pd.Series(model.feature_importances_,index=X1.columns)
ranked_features.nlargest(10).plot(kind='barh')
plt.show()
X.drop(['Hillshade_3pm','Slope'],axis=1,inplace=True)
X.columns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split,GridSearchCV
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size = 0.30, random_state=1220)
rand_clf = RandomForestClassifier()
rand_clf.fit(x_train,y_train)
rand_clf.score(x_train,y_train)
print('The accuracy of the model is:',rand_clf.score(x_test,y_test)*100,'percent')
from sklearn.metrics import classification_report
y_pred = rand_clf.predict(x_test)
print(classification_report(y_test,y_pred))
