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
import pandas as pd

import numpy as np



train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



train.head()

train.SalePrice.describe()
import matplotlib.pyplot as plt

plt.style.use(style='ggplot')

plt.rcParams['figure.figsize'] = (10, 6)

print ("Skew is:", train.SalePrice.skew())

plt.hist(train.SalePrice, color='blue')

plt.show()
target = np.log(train.SalePrice)

print ("Skew is:", target.skew())

plt.hist(target, color='blue')

plt.show()
plt.scatter(x=train['GrLivArea'], y=target)

plt.ylabel('Sale Price')

plt.xlabel('Above grade (ground) living area square feet')

plt.show()
data = train.select_dtypes(include=[np.number]).interpolate().dropna() 

y = np.log(train.SalePrice)

X = data.drop(['SalePrice', 'Id'], axis=1)



X.describe()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=.33)



X_train.head()
from sklearn import linear_model

lr = linear_model.LinearRegression()

lr_model = lr.fit(X_train, y_train)



lr_model.score(X_test, y_test)



br =  linear_model.BayesianRidge()

br_model = br.fit(X_train, y_train)



br_model.score(X_test, y_test)
predictions_lm = lr_model.predict(X_test)
rm = linear_model.Ridge(alpha=0.1)

rm_model = rm.fit(X_train, y_train)
predictions_rm = rm_model.predict(X_test)
submission = pd.DataFrame()

submission['Id'] = test.Id

feats = test.select_dtypes(

        include=[np.number]).drop(['Id'], axis=1).interpolate()



submission.plot()
predictions = br_model.predict(feats)

final_predictions = np.exp(predictions)

submission['SalePrice'] = final_predictions

submission.head()
submission.to_csv('submission2.csv', index=False)