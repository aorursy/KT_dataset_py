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
import pandas as pd
heading = ['preg','plas','pres','skin','test','mass','pedi','age','clas']
df=pd.read_csv("../input/pima-indians-diabetes.csv",names=heading)
df
import statsmodels.formula.api as smf
df.columns
linear=smf.ols('clas ~ preg+plas+pres+skin+test+mass+pedi+age',data=df).fit()
df.shape
df.columns
linear.summary()
linear_v=smf.ols('clas~preg',data=df).fit()

linear_v.summary()

linear_v=smf.ols('clas~plas',data=df).fit()

linear_v.summary()
linear_v=smf.ols('clas~pres',data=df).fit()

linear_v.summary()
linear_v=smf.ols('clas~skin',data=df).fit()

linear_v.summary()
linear_v=smf.ols('clas~test',data=df).fit()

linear_v.summary()
linear_v=smf.ols('clas~mass',data=df).fit()

linear_v.summary()
linear_v=smf.ols('clas~pedi',data=df).fit()

linear_v.summary()

linear_v=smf.ols('clas~age',data=df).fit()

linear_v.summary()
df.head()
from sklearn import linear_model
x=df.iloc[:,:8]
x_train=x[:750]
x_test=x[750:]
Y=df.iloc[:,8:9]
y_train=Y[:750]

y_test=Y[750:]
regr=linear_model.LinearRegression()
regr.fit(x_train,y_train)
y_pred=regr.predict(x_test)
y_pred