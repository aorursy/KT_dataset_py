from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd 

import numpy as np 

import os



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
submission = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv")

train = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")

test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")
train.head()
train.describe()
train = train.drop(["Id", "BsmtFinSF2"], axis=1)

test = test.drop(["Id", "BsmtFinSF2"], axis=1)
def plot_labels(data):

    fig, axes = plt.subplots(ncols=2, figsize=(10, 5))

    fig.subplots_adjust(hspace=0.2)



    sns.distplot(data["SalePrice"], ax=axes[0])

    sns.distplot(np.log1p(data["SalePrice"]), ax=axes[1])

    

    labels = ["labels", "log1p labels"]



    for i, label in enumerate(labels):

        axes[i].set_title(label)

        

    plt.plot()

    

    

plot_labels(train)
data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],

                  test.loc[:,'MSSubClass':'SaleCondition']))
def fillna_obiect(data):

    

    col = data.select_dtypes(include=['object']).columns

    for i in col:

        data[i] = data[i].fillna(data[i].mode()[0])

        

    return data





data = fillna_obiect(data)
def fillna(data):

    obiect_col = data.select_dtypes(include=['object']).columns

    

    for i in data.columns:

        if i not in obiect_col:

            data[i] = data[i].fillna(data[i].median())

        

    return data





data = fillna(data)
data.isnull().sum().sum()
def label_encoder(data, columns):

    

    for i in columns:

        le = LabelEncoder() 

        le.fit(list(data[i].values)) 

        data[i] = le.transform(list(data[i].values))

        

    return data





columns = data.select_dtypes(include=['object']).columns



data = label_encoder(data, columns)
data.head()
train_data = data[:train.shape[0]].values

test_data = data[train.shape[0]:].values



labels = train['SalePrice'].values
labels = np.log1p(labels)
train_size = int(len(train_data)*0.2)
x_train = train_data[train_size:]

x_test = train_data[:train_size]



y_train = labels[train_size:]

y_test = labels[:train_size]
print("X train shape: ", x_train.shape)

print("X test shape: ", x_test.shape)



print("Y train shape: ", y_train.shape)

print("Y test shape: ", y_test.shape)
class KNNRegressor:

    def __init__(self, k, d_metric, p=1):

        self.k = k

        self.d_metric = d_metric

        self.d_metric_to_fn = {

            'euclidean': self.euclidean,

            'manhattan': self.manhattan,

            'minkowski': self.minkowski

        }

        self.p = p



    def fit(self, X, y):

        self.X = np.copy(X)

        self.y = np.copy(y)



    def manhattan(self, x_test):

        return np.sum(np.abs(self.X - x_test), axis=-1)



    def euclidean(self, x_test):

        sq_diff = (self.X - x_test) ** 2

        return np.sqrt(np.sum(sq_diff, axis=-1))



    def minkowski(self, x_test):

        abs_diff = np.abs(self.X - x_test)

        sum_p_diff = np.sum(abs_diff ** self.p, axis=-1)

        pth_root = sum_p_diff ** (1 / self.p)

        return pth_root



    def distance(self, x_test):

        return self.d_metric_to_fn[self.d_metric](x_test)



    

    def predict(self, x_test):

        preds = []

        for index in range(x_test.shape[0]):

            distances = self.distance(x_test[index])

            sorted_labels = self.y[np.argsort(distances)]

            k_sorted_labels = sorted_labels[:self.k]

            

            pred = k_sorted_labels.sum()/len( k_sorted_labels)

                

            preds.append(pred)

        return np.array(preds)  
def mean_squared_error(Y, Y_pred):

     return np.square(Y - Y_pred).mean()
def result_knn(metric):

    mse = {}



    for k in range(1, 21):

        knn = KNNRegressor(k=k, d_metric=metric)

        knn.fit(x_train, y_train)

    

        mse[k] = mean_squared_error(y_test, knn.predict(x_test))

        

    print("min mse:", min(mse.values()))

    print("best k", np.argmin([*mse.values()])+1)

        

    plt.figure(figsize=(12, 4))

    plt.plot(mse.keys(), mse.values())

    plt.xticks(range(1, 21))

    plt.xlabel("k")

    plt.ylabel("MSE")

    plt.title("metrics: " + metric, fontsize=15)

    plt.show()
result_knn('euclidean')
result_knn('manhattan')
result_knn('minkowski')
knn = KNNRegressor(d_metric='minkowski', k=6)

    

knn.fit(train_data, labels)
predict = np.expm1(knn.predict(test_data))
submission['SalePrice'] = predict

submission.to_csv('submission.csv',index=False)
submission.head()
!ls -ls