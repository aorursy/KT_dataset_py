import pandas as pd

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


'''print(train.shape)
print(test.shape)

print(train.head(n=5))
print(test.head(n=5))'''

x_train = train.drop("QuoteConversion_Flag", axis=1)
y_train = train["QuoteConversion_Flag"]
x_test = test.copy()

# Data (datetime) preparing

for data in [x_train, x_test]:
    data['Year']  = data['Original_Quote_Date'].apply(lambda x: int(str(x)[:4]))
    data['Month'] = data['Original_Quote_Date'].apply(lambda x: int(str(x)[5:7]))
    data['Week']  = data['Original_Quote_Date'].apply(lambda x: int(str(x)[8:10]))
    data.drop(['Original_Quote_Date'], axis=1,inplace=True)
import numpy as np

from sklearn import preprocessing

for f in x_train.columns:
    if x_train[f].dtype=='object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(np.unique(list(x_train[f].values) + list(x_test[f].values)))
        x_train[f] = lbl.transform(list(x_train[f].values))
        x_test[f] = lbl.transform(list(x_test[f].values))
from sklearn.preprocessing import Imputer
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
x_train_nonan = imp.fit_transform(x_train)
x_test_nonan = imp.fit_transform(x_test)

'''
x_train_nonan = x_train.fillna(-1)
x_test_nonan = x_test.fillna(-1)'''
'''
x_train_nonan = x_train.interpolate()
x_test_nonan = x_test.interpolate()
'''
# Data preparing

for data in [x_train, x_test]:
    data.drop(['QuoteNumber'], axis=1,inplace=True)

from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV

'''
model = LogisticRegression()
model.fit(x_train_nonan, y_train)'''
#model = RidgeCV(alphas=(0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0))
model = LassoCV(alphas=(0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0))

model.fit(x_train_nonan, y_train)
y_test_bool = model.predict(x_train_nonan)
print(accuracy_score(y_train, y_test_bool))
y_test_bool = model.predict(x_train_nonan)
'''
from sklearn.metrics import accuracy_score

for i in range(1, 20):
    y_test = [(1 if x>=i/20.0 else 0) for x in y_test_bool]
    print(i/20.0, accuracy_score(y_test, y_train))'''
#y_test = [(1 if x>=0.4 else 0) for x in model.predict(x_test_nonan)]
y_test = model.predict(x_test_nonan)
submission = pd.DataFrame()
submission["QuoteNumber"]          = test["QuoteNumber"]
submission["QuoteConversion_Flag"] = y_test

submission.to_csv('homesite.csv', index=False)
