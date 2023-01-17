import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing


train_set = pd.read_csv("/kaggle/input/mercedesbenz-greener-manufacturing/train.csv")
print("Shape of training set: ", train_set.shape)
test_set = pd.read_csv("/kaggle/input/mercedesbenz-greener-manufacturing/test.csv")
print("Shape of testing set: ", test_set.shape)

print(train_set.head())


#check if there are any null rows
print("Is null data present in training set: ", train_set.isnull().any().any())
print("Is null data present in testing set: ", test_set.isnull().any().any())
print(train_set.dtypes)
train_set = train_set.drop("ID", axis=1)
#check for string data types and encoding them to integer
for columns in train_set.columns:
    if (train_set[columns].dtype == "object"):
        train_set[columns] = LabelEncoder().fit_transform(train_set[columns])

print(train_set.dtypes)
#Dividing into dependent and independent variables
X = train_set.iloc[:, train_set.columns != 'y']
Y = train_set.iloc[:, train_set.columns == 'y']
print(X.shape)
print(Y.shape)
#normalizing the data
min_max_scaler = preprocessing.MinMaxScaler()
X_normalized = min_max_scaler.fit_transform(X)
X = pd.DataFrame(X_normalized)
#since there are 377 independent variables, applying PCA to remove any highly correlated variables
from sklearn.decomposition import PCA

pca = PCA(n_components=0.95)
principalComponents = pca.fit_transform(X)
X = pd.DataFrame(principalComponents)
print(X.head)
from sklearn.model_selection import train_test_split
(X_train, X_Test, Y_train, Y_Test) = train_test_split(X, Y, test_size = 0.33, random_state = 1)
from xgboost import XGBClassifier

model = XGBClassifier()
model.fit(X_train, Y_train)
from sklearn.metrics import mean_squared_error
Y_pred = model.predict(X_Test)
mean_squared_error(Y_Test, Y_pred)
print(Y_pred)
print(np.sqrt(mean_squared_error(Y_Test, Y_pred)))
#Preprocessing test data as well
test_set = pd.read_csv("/kaggle/input/mercedesbenz-greener-manufacturing/test.csv")
test_set = test_set.drop("ID",axis=1)

#label encoding
for columns in test_set.columns:
    if (test_set[columns].dtype == "object"):
        test_set[columns] = LabelEncoder().fit_transform(test_set[columns])

#normalize the data
min_max_scaler = preprocessing.MinMaxScaler()
X_normalized = min_max_scaler.fit_transform(test_set)
test_set = pd.DataFrame(X_normalized)

#applying pca
test_set_pca = pca.transform(test_set)
test_set_df = pd.DataFrame(test_set_pca)

test_pred = model.predict(test_set_df)