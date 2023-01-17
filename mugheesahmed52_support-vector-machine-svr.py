import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn.preprocessing import Imputer



from sklearn import svm



import warnings

warnings.filterwarnings('ignore')



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
train = pd.read_csv('../input/train.csv')

labels=train["SalePrice"]

test = pd.read_csv('../input/test.csv')

data = pd.concat([train,test],ignore_index=True)

data = data.drop("SalePrice", 1)

ids = test["Id"]
train.head()
# Count the number of rows in train

train.shape[0]

# Count the number of rows in total

data.shape[0]
# Count the number of NaNs each column has.

nans=pd.isnull(data).sum()

nans[nans>0]


data=data.drop("Id", 1)

data=data.drop("Alley", 1)

data=data.drop("Fence", 1)

data=data.drop("MiscFeature", 1)

data=data.drop("PoolQC", 1)

data=data.drop("FireplaceQu", 1)
# Count the column types

data.dtypes.value_counts()
data.head()
all_columns = data.columns.values

non_categorical = ["LotFrontage", "LotArea", "MasVnrArea", "BsmtFinSF1", 

                   "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "1stFlrSF", 

                   "2ndFlrSF", "LowQualFinSF", "GrLivArea", "GarageArea", 

                   "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch", 

                   "ScreenPorch","PoolArea", "MiscVal"]



categorical = [value for value in all_columns if value not in non_categorical]
# One Hot Encoding and nan transformation

data = pd.get_dummies(data)



imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)

data = imp.fit_transform(data)



# Log transformation

data = np.log(data)

labels = np.log(labels)



# Change -inf to 0 again

data[data==-np.inf]=0
# Split traing and test

train = data[:1460]

test = data[1460:]
train.shape
labels.shape
clf = svm.SVR(kernel="rbf")
clf.fit(train,labels)
test_labels = clf.predict(test)
test_labels.shape
print(test_labels)
price_label = np.exp(test_labels)

price_labels = price_label.reshape(-1,)
print(price_labels)
sub = pd.DataFrame({

        "SalePrice": price_labels

    })
sub.head()
test_data = pd.read_csv('../input/test.csv')
result =  test_data.append(sub)
result['SalePrice'] = sub['SalePrice']
result['SalePrice'].head()
result.to_csv("sample_submission.csv", index=False)