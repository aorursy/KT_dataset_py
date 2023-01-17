import pandas as pd

import seaborn as sns

import sklearn
a=pd.read_csv("../input/real-estate/realestate.csv")

a
a.columns

a1=a.iloc[:,2:7]

a1

y=a.iloc[:,7:8]

y

a.info()

a.describe()

sns.pairplot(a)

#normalizing the data:

from sklearn import preprocessing

a_nor=preprocessing.normalize(a)

a_nor=pd.DataFrame(a_nor)

a_nor

a_nor.columns=a.columns

a_nor
#splitting x and y:

x=a_nor.iloc[:,2:7]

x
y=a_nor.iloc[:,7:8]

y
#paiplot for a_nor:

sns.pairplot(a_nor)
sns.pairplot(x)
x.corr()

a_nor.corr()
#splitting training and test data set:

from sklearn import model_selection

from sklearn import linear_model

x_train,x_test,y_train,y_test=model_selection.train_test_split(x,y,test_size=0.2)

x_train.shape

x_test.shape

y_train.shape

y_test.shape

x_train
#applying linear regression:

lm=linear_model.LinearRegression()

model=lm.fit(x_train,y_train)

model.coef_

model.intercept_

pred=lm.predict(x_train)

pred
#checking accuracy using r2:

from sklearn.metrics import r2_score

r2_score(pred,y_train)

predd=lm.predict(x_test)

r2_score(predd,y_test)