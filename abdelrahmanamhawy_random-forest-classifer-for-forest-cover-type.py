# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import Pipeline

from sklearn.metrics import precision_score,recall_score

from sklearn.model_selection import cross_val_predict

from sklearn.metrics import accuracy_score

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split



from sklearn.model_selection import GridSearchCV





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_file_name_class= pd.read_csv("../input/forest-cover-type-prediction/train.csv")

labels=train_file_name_class["Cover_Type"]

feautres=train_file_name_class.drop(columns='Id')

print(feautres.shape)

#train_file_name_class.head

#print(labels)

colors = (0,0,0)

area = np.pi*3



plt.scatter(feautres.iloc[:, 0] ,labels ,s=area, c=colors, alpha=0.5)

plt.title('Scatter plot')

plt.xlabel('X')

plt.ylabel('y')

plt.show()

X_train, X_test, y_train, y_test = train_test_split(feautres, labels, test_size=0.2, 

                                                    random_state=123)
clf=Pipeline([

   # ("kpca",KernelPCA(n_components=4,gamma=0.03,kernel="rbf")),

    ("scaler", StandardScaler()),

    ("forest",RandomForestClassifier(n_estimators=100,max_depth=10,min_samples_split=6,))

])

clf.fit(X_train,y_train)
y_pred=cross_val_predict(clf,X_train,y_train,cv=3)
accuracy_score(y_train, y_pred, normalize=True)
test_file_names= pd.read_csv("../input/forest-cover-type-prediction/test.csv", header=0)



Id=test_file_names['Id']

output=pd.DataFrame(Id)

output["Cover_Type"]=clf.predict(test_file_names)

output.to_csv("output.csv",index=False)


