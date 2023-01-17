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
data_master=pd.read_csv("/kaggle/input/personal-loan/Bank_Personal_Loan_Modelling-1.xlsx")
data_master.info()
X=data_master.drop(["Personal Loan"], axis=1)



y=data_master["Personal Loan"]



from sklearn.model_selection import train_test_split

X_train,X_test, Y_train, Y_test=train_test_split(X,y,test_size=0.30,random_state=40)
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()

clf.fit(X_train,Y_train)

Y_pred=clf.predict(X_test)

print(Y_pred[:50])
from sklearn.metrics import accuracy_score

score = accuracy_score(Y_test, Y_pred, normalize=True)

score