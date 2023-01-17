import pandas as pd

import numpy as np

import matplotlib.pylab as plt



from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import accuracy_score,mean_squared_error, r2_score

train = pd.read_csv('../input/train.csv')
train = train.fillna(0)
train = pd.get_dummies(train)



print("Shape of our dataset is {}".format(train.shape))
y = pd.DataFrame(train['SalePrice'])

X = pd.DataFrame(train.drop(columns=['SalePrice']).values)

print("X {}".format(X.shape))
X = pd.get_dummies(X)
test_df = pd.read_csv('../input/test.csv')

test_df = test_df.fillna(0)

test_df = pd.get_dummies(test_df)
col_dif2 = X.columns.difference(test_df.columns)

print(col_dif2)





X = pd.DataFrame(X.drop(columns=['Condition2_RRAe', 'Condition2_RRAn', 'Condition2_RRNn', 'Electrical_0','Electrical_Mix', 'Exterior1st_ImStucc', 'Exterior1st_Stone',

       'Exterior2nd_Other', 'GarageQual_Ex', 'Heating_Floor', 'Heating_OthW',

       'HouseStyle_2.5Fin', 'MiscFeature_TenC', 'PoolQC_Fa',

       'RoofMatl_ClyTile', 'RoofMatl_Membran', 'RoofMatl_Metal',

       'RoofMatl_Roll', 'Utilities_NoSeWa'],axis=1))



col_dif2 = test_df.columns.difference(X.columns)

print(col_dif2)



test_df = pd.DataFrame(test_df.drop(columns=['Exterior1st_0', 'Exterior2nd_0', 'Functional_0', 'KitchenQual_0',

       'MSZoning_0', 'SaleType_0', 'Utilities_0'],axis=1))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=24) 
model = LinearRegression()

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(model.score(X_test,y_test))
plt.plot(y_test, y_pred, '.')

x = np.linspace(0, 700000, 100)

y = x

plt.plot(x, y)

plt.show()
print("Ortalama Karesel Hata: %.2f" % mean_squared_error(y_test, y_pred))

print('Varyans skoru: %.2f' % r2_score(y_test, y_pred))
test_pred = model.predict(test_df)

Id = test_df["Id"].tolist()

pred = test_pred.tolist()



sonuc= pd.DataFrame({'Id': Id, 'SalePrice': pred})

sonuc.to_csv('submission1.csv', index=False)



print("{}".format(test_df.shape))