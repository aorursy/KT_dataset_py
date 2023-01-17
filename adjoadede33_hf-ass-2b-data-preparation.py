import pandas



data = pandas.read_csv('../input/Internship Application  - Classification.csv')

print(data.shape)
data.head(10)
data.sample(6)
data.info()
data = data.rename(columns={"Experience Level": "Experience"})
data.dtypes
data['Gender'] = data['Gender'].astype('category')

data['Experience'] = data['Experience'].astype('category')
data.dtypes

data['Gender'] = data['Gender'].cat.codes

data['Experience'] = data['Experience'].cat.codes
data.head()
pandas.get_dummies(data, columns=["Gender", "Experience"]).head()
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