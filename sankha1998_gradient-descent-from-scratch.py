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
df=pd.read_csv('/kaggle/input/years-of-experience-and-salary-dataset/Salary_Data.csv')
df.head()
X=df.iloc[:,0].values

y=df.iloc[:,1].values
def fit(X,y):

    m=0

    b=0

    lr=0.0001 ## learning rate

    n_iteration=1000

    for i in range(n_iteration):

        ## ss= (py-ay)**2 

       

        py=m*X+b ##  predict y eqution

        

        

        ss_b = 2  * sum((py-y) * 1)   ## derivative w.r.t b

        

        ss_m = 2 * sum(X * (py-y))    ## derivative w.r.t m

        

        ## step size of m,b

        stepm = ss_m*lr 

        stepb = ss_b*lr

        

        

        m= m - stepm

        b= b - stepb

        

    return m,b
def predict(x):

    m,b=fit(X,y)

    return (m*x+b)
predict(7.8)