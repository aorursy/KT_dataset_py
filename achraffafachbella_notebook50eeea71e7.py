#Import necessary packages 
import pandas as pd
import numpy as np
import seaborn as sns
import warnings
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn import metrics

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam

warnings.filterwarnings("ignore")
data = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
data.head()
print("data shape ", data.shape)
for elm in data.columns:
    if data[elm].dtype == 'object':
        if data[elm].isnull().sum() != 0:
            print(elm, data[elm].isnull().sum())
            pass
        pass
    pass
pass

data = data.drop(["Id", "Alley", "MiscFeature", "Fence", "PoolQC", "FireplaceQu"], axis = 1)
for elm in data.columns:
    if data[elm].dtype == 'object':
        if data[elm].isnull().sum() != 0:
            data[elm] = data[elm].fillna(data[elm].value_counts().idxmax())
    else:
        if data[elm].isnull().sum() != 0:
            data[elm] = data[elm].fillna(int(np.mean(data[elm])))
            pass
        pass
    pass
pass
numerical_features   = list()
categorical_features = list()
for elm in data.columns:
    if data[elm].dtype == 'object':
        print(elm," ",data[elm].unique())
        categorical_features.append(elm)
    else:
        numerical_features.append(elm)
        pass
    pass
pass
data.isnull().sum().any()
for elm in numerical_features:
    fig = plt.figure(figsize=(16,5))
    fig.add_subplot(2,2,1)
    sns.scatterplot(data[elm], data['SalePrice'])
    plt.tight_layout()
    pass
pass
for elm in categorical_features:
    fig = plt.figure(figsize=(16,5))
    fig.add_subplot(2,2,1)
    sns.countplot(data[elm])
    plt.tight_layout()
    pass
pass
features = ['LotArea', 'YearBuilt', 'YearRemodAdd'
           , 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea']
from scipy import stats
outliers_values = list()
for elm in features:
    z = np.abs(stats.zscore(data[elm]))
    threshold = 3
    l = list(np.where(z > 3))
    print(l)
    pass
pass
values = [53,  249,  313,  335,  384,  451,  457,  661,  706,  769,  848,
       1298, 1396, 304,  630,  747, 1132, 1137, 1349, 137,  224,  278,  477,  496,  581,  678,  774,  798,  932, 1267
         , 224,  332,  440,  496,  523,  691, 1044, 1182, 1298, 1373, 224,  440,  496,  523,  529,  691,  898, 1024, 1044, 1182, 1298,
       1373, 304,  691, 1169, 1182,1,   88,  125,  170,  185,  197,  198,  263,  267,  406,  589,
        635,  729,  873,  883, 1009, 1031, 1173, 1349, 1440, 118,  185,  197,  304,  496,  523,  608,  635,  691,  769,  798,
       1169, 1182, 1268, 1298, 1353 ]
outlier_values = list(set(values))
data1 = data.drop(index = outlier_values)
print("the new shape of data", data1.shape)
X = data1[features]
y = data1['SalePrice']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)
from sklearn.preprocessing import StandardScaler
import numpy as np
s_scaler = StandardScaler()
X_train = s_scaler.fit_transform(X_train.astype(np.float))
X_test = s_scaler.transform(X_test.astype(np.float))
Linear_Regression = LinearRegression()
Linear_Regression.fit(X_train, y_train)
print(Linear_Regression.intercept_)
print(Linear_Regression.coef_)
y_pred = Linear_Regression.predict(X_test)

coeff_df = pd.DataFrame({"features" : features, "Coefficient" : Linear_Regression.coef_ }) 
coeff_df
fig = plt.figure(figsize=(10,5))
residuals = (y_test- y_pred)
sns.distplot(residuals)
print('MAE:', metrics.mean_absolute_error(y_test, y_pred))  
print('MSE:', metrics.mean_squared_error(y_test, y_pred))  
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('VarScore:',metrics.explained_variance_score(y_test,y_pred))
datasets = pd.DataFrame({ 'SalePrice': y_pred })
datasets.to_csv('Submission.csv', index=False)
def build_model(dense_dimension = 200):
    model = Sequential()
    model.add(Dense(dense_dimension,  activation='relu'))
    model.add(Dense(dense_dimension,  activation='relu'))
    model.add(Dense(dense_dimension, activation='relu'))
    model.add(Dense(dense_dimension, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='Adam',loss='mean_squared_error')
    return model
model = build_model(200)
history = model.fit(x=X_train,y=y_train,
          validation_data=(X_test,y_test),
          batch_size=200,epochs=2000)
model.summary()
loss_df = pd.DataFrame(model.history.history)
loss_df.plot(figsize=(12,8))
y_pred = model.predict(X_test)
print('MAE:', metrics.mean_absolute_error(y_test, y_pred))  
print('MSE:', metrics.mean_squared_error(y_test, y_pred))  
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
from keras.wrappers.scikit_learn import KerasClassifier

keras_classifier = KerasClassifier(build_model, epochs = 1000)

keras_classifier.fit(X_train, y_train)

y_pred = keras_classifier.predict(X_test)
print('MAE:', metrics.mean_absolute_error(y_test, y_pred))  
print('MSE:', metrics.mean_squared_error(y_test, y_pred))  
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
f = ['MSZoning','SaleCondition']

MSZoning = preprocessing.LabelEncoder()
MSZoning.fit(['RL', 'RM', 'C (all)', 'FV', 'RH'])
data1['MSZoning'] = MSZoning.transform(data1['MSZoning'])


SaleCondition = preprocessing.LabelEncoder()
SaleCondition.fit(['Normal' ,'Abnorml' ,'Partial' ,'AdjLand' ,'Alloca' ,'Family'])
data1['SaleCondition'] = SaleCondition.transform(data1['SaleCondition'])

X1 = data1[f]
y = data1['SalePrice']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X1, y, test_size=0.33, random_state=101)
Linear_Regression1 = LinearRegression()
Linear_Regression1.fit(X_train, y_train)
y_pred = Linear_Regression1.predict(X_test)

print('MAE:', metrics.mean_absolute_error(y_test, y_pred))  
print('MSE:', metrics.mean_squared_error(y_test, y_pred))  
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('VarScore:',metrics.explained_variance_score(y_test,y_pred))


