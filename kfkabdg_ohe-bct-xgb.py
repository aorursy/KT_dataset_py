

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from scipy import stats

import matplotlib.pyplot as plt

import seaborn as sns

import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np

from sklearn.preprocessing import PowerTransformer

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

import pandas as pd

import lightgbm

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelBinarizer

from xgboost import XGBRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import Imputer

import matplotlib.pyplot as plt

import math 

import numpy as np

from sklearn.metrics import mean_absolute_error
#bring in the six packs

df = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
df.head()
SP=(df['SalePrice'])

SP
df= df.drop('SalePrice',axis=1)
pd.set_option('display.max_rows', 100)

df.dtypes
# plotting modules



pt = PowerTransformer()



# generate non-normal data

#original_data = df

original_data= df.select_dtypes(include=float)



#original_data = SP



# split into testing & training data

train,test = train_test_split(original_data, shuffle=False)



# transform training data & save lambda value

train_data = pt.fit_transform(train)



# use lambda value to transform test data

test_data = pt.fit_transform(test)



# (optional) plot train & test

fig, ax=plt.subplots(1,2)

sns.distplot(train_data, ax=ax[0])

sns.distplot(test_data, ax=ax[1])
import pandas as pd





sns.pairplot(test_data, size=2.5)

plt.show()
import pandas_profiling as pdp

pdp.ProfileReport(df)











my_pipeline = make_pipeline(Imputer(), RandomForestRegressor())



df_ohe= pd.get_dummies(df)

df_ohe
# データをトレーニング用、評価用に分割



from sklearn.model_selection import train_test_split

x, x_test, y, y_test  = train_test_split(

    df_ohe, SP, test_size=0.3)




my_model = XGBRegressor()

# Add silent=True to avoid printing out updates with each cycle

my_model.fit(x, y, verbose=False)
# make predictions

predictions = my_model.predict(x_test)





print("Mean Absolute Error : " + str(mean_absolute_error(predictions, y_test)))




# pyplot.plot(x,y)

plt.scatter(predictions,y_test)

plt.xlabel("SP_predictions")

plt.ylabel("SalesPrice")

plt.show()
comparison=pd.DataFrame(predictions,round(y_test))

comparison.head()
pd.options.display.float_format = '{:.0f}'.format

comparison.head()
#submi



test_data = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")

test_data.head()
test_data= pd.get_dummies(test_data)

test_data

#列データの数が学習時と合わないな。OHEで抜け落ちたんだろうな。。。。
predictions = my_model.predict(test_data)