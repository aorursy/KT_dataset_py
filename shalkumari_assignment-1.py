# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
loan1 = pd.read_csv("../input/train.csv")
loan2 = pd.read_csv("../input/test.csv")
loan=pd.merge(loan1 ,loan2 ,on='Loan_ID' ,how='outer')
loan.head()
loan=loan.dropna(how='any', subset=['Self_Employed_x', 'LoanAmount_x', 'Loan_Amount_Term_x', 'Credit_History_x', 'Property_Area_x'])
loan.shape

x=loan.iloc[:,6:10]
y=loan.iloc[:,12]
x=pd.get_dummies(x)
from sklearn.neighbors import KNeighborsClassifier
clsfr=KNeighborsClassifier()
clsfr.fit(x,y)
y_pred=clsfr.predict(x)
print(y_pred)
