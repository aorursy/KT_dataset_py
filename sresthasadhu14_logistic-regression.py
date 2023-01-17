# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

diabetes = pd.read_csv("../input/pima-indians-diabetes-database/diabetes.csv")

diabetes

x=diabetes.iloc[:,0:8]

y=diabetes.iloc[:,8]

#print(x)

#print(y)

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=123)



from sklearn.linear_model import LogisticRegression



logreg=LogisticRegression()



model=logreg.fit(x_train,y_train)



#print(model)



prediction=model.predict(x_test)



#print(prediction)



from sklearn.metrics import confusion_matrix



confusion_met=metrics.confusion_matrix(y_test,prediction)

#print(confusion_met)



from sklearn.metrics import accuracy_score



print("accuracy",metrics.accuracy_score(y_test,prediction))



from sklearn.metrics import precision_score



print("precision",metrics.precision_score(y_test,prediction))



from sklearn.metrics import recall_score



print("recall",metrics.recall_score(y_test,prediction))