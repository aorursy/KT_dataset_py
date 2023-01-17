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
df=pd.read_csv('/kaggle/input/zoo-animal-classification/zoo.csv')

df.set_index('animal_name',inplace=True)

df.head()

df['class_type'].replace({1:'Mammal',2:'Bird',3:'Reptile',4:'Fish',5:'Amphibian',6:'Bug',7:'Invertebrate'},inplace=True)
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(df.drop('class_type',axis=1),df['class_type'],random_state=0,test_size=0.3)
from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.metrics import accuracy_score
for x in range(2,8):

    for y in [100,50,10]:

        forest=RandomForestClassifier(max_depth=x,n_estimators=y)

        forest.fit(X_train,y_train)

        y_pred=forest.predict(X_test)

        print("depth= ",x,' estimators= ',y,' accuracy= ',accuracy_score(y_test,y_pred))
for x in range(1,10):

    neg=KNeighborsClassifier(n_neighbors=x)

    neg.fit(X_train,y_train)

    y_pred=neg.predict(X_test)

    print('neighbors= ',x,' accuracy= ',accuracy_score(y_test,y_pred))
for x in np.logspace(3,6):

    for y in np.logspace(4,6):

        clf=SVC(C=x,gamma=y,kernel='rbf')

        clf.fit(X_train,y_train)

        y_pred=clf.predict(X_test)

        print("C= ",x,' gamma= ',y,' accuracy= ',accuracy_score(y_test,y_pred))

        
forest=RandomForestClassifier(max_depth=4)

forest.fit(X_train,y_train)

y_pred=forest.predict(X_test)

final_predict_df=pd.DataFrame({'Name of animal':X_test.index,'Predicted Type':y_pred,'Actual Type':y_test})
final_predict_df
df.loc['seasnake']