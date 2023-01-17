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
import pandas as pd					# data manipulation using dataframes

import numpy as np					# data statistical analysis



import seaborn as sns				# Statistical data visualization

import matplotlib.pyplot as plt		# data visualisation

%matplotlib inline
import tensorflow as tf



from tensorflow import keras

from tensorflow.keras import Sequential

from tensorflow.keras.layers import Dense, Input

from tensorflow.keras.optimizers import Adam
raw_df = pd.read_csv("../input/ice-cream-revenue/IceCreamData.csv")
df = raw_df.copy()
df.shape
df.head()
df.tail()
# concise summary



df.info()
# statistical analysis



df.describe().transpose()
# Univariate Analysis



df.hist( bins = 10, figsize = (10,4), color = 'r')

plt.show()
# Multivariate Analysis



sns.scatterplot(x='Temperature',y='Revenue',data=df)

plt.show()
### Define X&Y ###



X = df.iloc[:, :-1].values

y = df.iloc[:, -1].values
### Splitting Dataset ###



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split( X, y , test_size = 0.2, random_state = 0)
print(X_train.shape, y_train.shape)

print(X_test.shape, y_test.shape)
model = Sequential()

model.add(Input (shape = (1)))

model.add(Dense (16))

model.add(Dense (1))
model.summary()
model.compile(optimizer=Adam(0.01), loss='mse', metrics = ['mae', 'mse'])
%%time

history = model.fit(X, y,

                    epochs = 1000,

                    verbose = 2,

                    validation_split = 0.2 )
plt.plot(history.history['loss'], label='train_loss')

plt.plot(history.history['val_loss'], label='val_loss')

plt.xlabel('Epoch')

plt.ylabel('Loss')

plt.legend()

plt.grid(True)
model.get_weights()
history.history.keys()
y_pred = model.predict(X_test)
test_df = pd.DataFrame(y_pred, columns = ['pred'])

test_df['actual'] = y_test

test_df.head(10)
# Visualization training results



plt.scatter(X_train, y_train, color = 'blue')

plt.plot(X_train, model.predict(X_train), color = 'red')

plt.xlabel('Revenue')

plt.ylabel('Temperature')

plt.title('Revenue Vs Temperature')

plt.grid(True)
# Visualization test results



plt.scatter(X_test, y_test, color = 'blue')

plt.plot(X_train, model.predict(X_train), color = 'red')

plt.xlabel('Revenue')

plt.ylabel('Temperature')

plt.title('Revenue Vs Temperature')

plt.grid(True)
### RMSE ###



from sklearn import metrics

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
### R-squared ###



from sklearn.metrics import r2_score

r2_score(y_test, y_pred)