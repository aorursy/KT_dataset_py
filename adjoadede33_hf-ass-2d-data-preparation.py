import pandas



data = pandas.read_csv('../input/Road Accidents-Regression.csv')



print(data.shape)
data.head(10)

data.info()
data.sample()
data = data.rename(columns={"Layout of Township": "Layout"})
data['Place'] = data['Place'].astype('category')

data['Nature of roads'] = data['Nature of roads'].astype('category')

data['Layout'] = data['Layout'].astype('category')
data.dtypes
data.head()
pandas.get_dummies(data, columns=['Place', 'Nature of roads', 'Layout']).head()
data = data.values





X = data[:,0:5]

Y = data[:,5]
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1

                                                    ,random_state = 0)



X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size = 0.11

                                                   ,random_state = 0)
X_train.shape
X_val.shape
X_test.shape
Y_train.shape
Y_val.shape
Y_test.shape