# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
%matplotlib inline



import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import linear_model
data = pd.read_csv("../input/HR_comma_sep.csv")

data.info()
data.head()
#data.sort(["satisfaction_level","left"],ascending=[1,1])

department_groups = {'sales': 1, 

                     'marketing': 2, 

                     'product_mng': 3, 

                     'technical': 4, 

                     'IT': 5, 

                     'RandD': 6, 

                     'accounting': 7, 

                     'hr': 8, 

                     'support': 9, 

                     'management': 10 

                    }

data['sales_index'] = data.sales.map(department_groups)

salary_groups = {'low': 0, 'medium': 1, 'high': 2}

data['salary_index']=data.salary.map(salary_groups)

data['salary_index']
left_emp = data[data["left"] == 1]

left_to_total = left_emp.groupby(["sales"]).count() / data.groupby(["sales"]).count()

left_to_total["satisfaction_level"]
plt.plot(left_emp["satisfaction_level"], left_emp["number_project"],  "o")

plt.show()
data[data.columns[:]].corr()['left'][:]
data.groupby(["sales"]).count()

cols = data.columns.tolist()

#cols

cols = cols[0:6] + cols[7:10] + cols[6:7]

#cols

data = data[cols]

cols=data.shape[1]
X = data.iloc[:,0:cols-1]

Y = data.iloc[:,cols-1:cols]

X,Y
mask = np.random.rand(len(data)) < 0.8

train = data[mask]

test = data[~mask]

len(train), len(test)

X1 = train.iloc[:,0:cols-1]

Y1 = train.iloc[:,cols-1:cols]

X2 = test.iloc[:,0:cols-1]

Y2 = test.iloc[:,cols-1:cols]

X1, Y1
X1 = X1.drop(["sales", "salary"], axis=1)

X2 = X2.drop(["sales", "salary"], axis=1)
model = linear_model.LogisticRegression(penalty='l2', C=1.0)

model.fit(X1, Y1)
model.score(X2, Y2)
model.score(X1, Y1)