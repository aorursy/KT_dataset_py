# pandas to open data files & processing it.

import pandas as pd



# numpy for numeric data processing

import numpy as np



# sklearn to do preprocessing & ML models

from sklearn import preprocessing

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

from sklearn.metrics import r2_score

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeRegressor

from xgboost import XGBRegressor as xgbr # for modelling

from sklearn.pipeline import Pipeline # for making pipleine 

from sklearn.impute import SimpleImputer # for handling missing variables either categorical or numerical

from sklearn.preprocessing import OneHotEncoder # for one hot encoding categorical variables

from sklearn.metrics import mean_absolute_error # for Mean absolute error

from sklearn.model_selection import train_test_split,GridSearchCV

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import StandardScaler

from scipy.stats import skew



# Matplotlob & seaborn to plot graphs & visulisation

import matplotlib.pyplot as plt 

import seaborn as sns

import missingno as msno



# to fix random seeds

import random, os



# ignore warnings

import warnings

warnings.simplefilter(action='ignore')

pd.set_option('display.max_columns', None)
model_df = pd.read_csv('../input/home-data-for-ml-course/train.csv',index_col='Id')

pred_df = pd.read_csv('../input/home-data-for-ml-course/test.csv',index_col='Id')

dframes = [model_df,pred_df]

df=pd.concat([model_df,pred_df], ignore_index=False, sort =False)
numeric_data = df.select_dtypes(include=[np.number])

categorical_data = df.select_dtypes(exclude=[np.number])

numeric_data.head(5)
categorical_data.head(5)
model_df['YrSold'] = model_df['YrSold'].astype(str)

model_df['GarageYrBlt'] = model_df['GarageYrBlt'].astype(str)

model_df['YearRemodAdd'] = model_df['YearRemodAdd'].astype(str)

model_df['YearBuilt'] = model_df['YearBuilt'].astype(str)

model_df['MoSold'] = model_df['MoSold'].astype(str)

model_df['MSSubClass'] = model_df['MSSubClass'].astype(str)

model_df['OverallCond'] = model_df['OverallCond'].astype(str)



pred_df['YrSold'] = pred_df['YrSold'].astype(str)

pred_df['GarageYrBlt'] = pred_df['GarageYrBlt'].astype(str)

pred_df['YearRemodAdd'] = pred_df['YearRemodAdd'].astype(str)

pred_df['YearBuilt'] = pred_df['YearBuilt'].astype(str)

pred_df['MoSold'] = pred_df['MoSold'].astype(str)

pred_df['MSSubClass'] = pred_df['MSSubClass'].astype(str)

pred_df['OverallCond'] = pred_df['OverallCond'].astype(str)
#NaN values with %, we have some NaN, later we will deal with that...

total = df.isnull().sum().sort_values(ascending=False)

percent_1 = df.isnull().sum()/df.isnull().count()*100

percent_2 = (round(percent_1, 1)).sort_values(ascending=False)

missing_data = pd.concat([total, percent_2], axis=1, keys=['Total', '%'])

print(missing_data.head(30))
for dataframe in dframes:

    dataframe.drop(['Utilities','BsmtFinSF1', 'BsmtFinSF2'], axis=1, inplace=True)
model_df['Alley'].value_counts()
for col in ('PoolQC','MiscFeature','Alley','Fence','FireplaceQu','GarageType', 'GarageFinish', 'GarageQual'):

    model_df[col] = model_df[col].fillna('None')

    pred_df[col] = pred_df[col].fillna('None')

for col in ('GarageYrBlt', 'GarageArea', 'GarageCars',):

    model_df[col] = model_df[col].fillna(0)

    pred_df[col] = pred_df[col].fillna(0)

for col in ('BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath','MasVnrArea'):

    model_df[col] = model_df[col].fillna(0)

    pred_df[col] = pred_df[col].fillna(0)

for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2','MasVnrType','Electrical'):

    model_df[col] = model_df[col].fillna('None')

    pred_df[col] = pred_df[col].fillna('None')

    

#model_df['GarageCond'] = model_df['GarageCond'].fillna('TA')

#pred_df['GarageCond'] = pred_df['GarageCond'].fillna('TA')

    

model_df["LotFrontage"] = model_df.groupby("Neighborhood")["LotFrontage"].transform(

    lambda x: x.fillna(x.median()))

pred_df["LotFrontage"] = pred_df.groupby("Neighborhood")["LotFrontage"].transform(

    lambda x: x.fillna(x.median()))
model_df = model_df.fillna(model_df.mode().iloc[0])

pred_df = pred_df.fillna(pred_df.mode().iloc[0])
#NaN values with %, we have some NaN, later we will deal with that...

total = model_df.isnull().sum().sort_values(ascending=False)

percent_1 = model_df.isnull().sum()/model_df.isnull().count()*100

percent_2 = (round(percent_1, 1)).sort_values(ascending=False)

missing_data = pd.concat([total, percent_2], axis=1, keys=['Total', '%'])

print(missing_data.head(3))
cat_col = df.select_dtypes(include='object').columns



columns = len(cat_col)/4+1



fg, ax = plt.subplots(figsize=(20, 25))



for i, col in enumerate(cat_col):

    fg.add_subplot(columns, 4, i+1)

    sns.countplot(df[col])

    plt.xlabel(col)

    plt.xticks(rotation=90)



plt.tight_layout()

plt.show()
num_col = df.select_dtypes(exclude='object').columns



columns = len(num_col)/4+1



fg, ax = plt.subplots(figsize=(20, 25))



for i, col in enumerate(num_col):

    fg.add_subplot(columns, 4, i+1)

    sns.distplot(df.select_dtypes(exclude='object')[col],rug=True, hist=False, label='UW', kde_kws={'bw':0.1})

    plt.xlabel(col)

    plt.xticks(rotation=90)



plt.tight_layout()

plt.show()
num_col = df.select_dtypes(exclude='object').columns

columns = len(num_col)/4+1



fg, ax = plt.subplots(figsize=(25, 35))



for i, col in enumerate(num_col):

    fg.add_subplot(columns, 4, i+1)

    sns.scatterplot(df.select_dtypes(exclude='object').iloc[:, i],df['SalePrice'])

    plt.xlabel(col)

    plt.xticks(rotation=90)



plt.tight_layout()

plt.show()
model_df.shape
model_df = model_df.drop(model_df[model_df['LotFrontage']>200].index)

model_df = model_df.drop(model_df[model_df['LotArea']>100000].index)

model_df = model_df.drop(model_df[model_df['MasVnrArea']>1200].index)

model_df = model_df.drop(model_df[model_df['TotalBsmtSF']>4000].index)

model_df = model_df.drop(model_df[(model_df['GrLivArea']>4000) & (model_df['SalePrice']<300000)].index)

model_df = model_df.drop(model_df[model_df['1stFlrSF']>4000].index)

model_df = model_df.drop(model_df[model_df['EnclosedPorch']>400].index)

model_df = model_df.drop(model_df[model_df['MiscVal']>5000].index)

model_df = model_df.drop(model_df[(model_df['LowQualFinSF']>600) & (model_df['SalePrice']>400000)].index)
model_df.shape
model_df.head()
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()



failed_features = []

for columns in model_df.select_dtypes(include='object'):

    try:

        model_df[columns] = le.fit_transform(model_df[columns])

        pred_df[columns] = le.fit_transform(pred_df[columns])

    except:

        failed_features.append(columns)

        

failed_features
model_df.head()
model_df.loc[:, (model_df.columns != 'SalePrice') & (model_df.columns != 'LotArea')].boxplot()

plt.xticks(rotation=70)

plt.gcf().set_size_inches(20, 6)
model_df2 = model_df.copy()

pred_df2 = pred_df.copy()



cols_to_norm = model_df2.loc[:, model_df2.columns != 'SalePrice']._get_numeric_data().columns

model_df2[cols_to_norm] = model_df2[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))



cols_to_norm = pred_df2._get_numeric_data().columns

pred_df2[cols_to_norm] = pred_df2[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
model_df2.loc[:, model_df.columns != 'SalePrice'].boxplot()

plt.xticks(rotation=70)

plt.gcf().set_size_inches(20, 6)
scaler = StandardScaler()



model_df3 = pd.DataFrame(scaler.fit_transform(model_df))

pred_df3 = pd.DataFrame(scaler.fit_transform(pred_df))



model_df3.loc[:, model_df3.columns != 'SalePrice'].boxplot()

plt.xticks(rotation=0)

plt.gcf().set_size_inches(20, 6)
a = model_df.corr()

a = a['SalePrice'].sort_values()

plt.gcf().set_size_inches(20, 6)

plt.xticks(rotation=70)

a = sns.barplot(a.index[0:-1],a[0:-1])
model_df.head(10)
#select columns for the mode

model_Y = model_df['SalePrice']

model_X = model_df.loc[:, model_df.columns != 'SalePrice']
X_train, X_test, y_train, y_test = train_test_split(model_X, model_Y, train_size=0.75, test_size=0.25, random_state=42)
from xgboost import XGBRegressor

from sklearn.metrics import mean_squared_error

model_xgb = XGBRegressor(learning_Rate= 0.005, max_depth = 4, min_child_weight = 2, n_estimators = 1000, nthread = 4)

#model_xgb = XGBRegressor(n_estimators=100, learning_Rate=1)

model_xgb.fit(X_train, y_train, verbose=False)

y_pred = model_xgb.predict(X_test)



# evaluate predictions

rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("RMSE: %f" % (rmse))

print("MAE " + str(mean_absolute_error(y_test, y_pred)))

print(r2_score(y_test, y_pred))
y_pred = model_xgb.predict(pred_df)



output = pd.DataFrame({'Id': pred_df.index,

                       'SalePrice': y_pred})



output.to_csv('submission.csv', index=False)
def algorithm_pipeline(X_train_data, X_test_data, y_train_data, y_test_data, 

                       model, param_grid, cv=10, scoring_fit='neg_mean_squared_error',

                       do_probabilities = False):

    gs = GridSearchCV(

        estimator=model,

        param_grid=param_grid, 

        cv=cv, 

        n_jobs=-1, 

        scoring=scoring_fit,

        verbose=2

    )

    fitted_model = gs.fit(X_train_data, y_train_data)

    

    if do_probabilities:

      pred = fitted_model.predict_proba(X_test_data)

    else:

      pred = fitted_model.predict(X_test_data)

    

    return fitted_model, pred
model = XGBRegressor()



# Various hyper-parameters to tune

parameters = {'nthread':[4], #when use hyperthread, xgboost may become slower

              'learning_Rate': [.005, .01, 0.5, 1], #so called `eta` value

              'max_depth': [None, 4,6],

              'min_child_weight': [None, 2,4],

              'n_estimators': [100]}



model, pred = algorithm_pipeline(X_train, X_test, y_train, y_test, model, 

                                 parameters, cv=5)



# Root Mean Squared Error

print(model.best_params_)

# evaluate predictions

rmse = np.sqrt(-model.best_score_)

print("RMSE: %f" % (rmse))



#from tpot import TPOTClassifier



#tpot = TPOTClassifier(generations=10, population_size=50, verbosity=2, random_state=42,max_time_mins=2, max_eval_time_mins=0.2)

#tpot.fit(X_train, y_train)

#print(tpot.score(X_test, y_test))

#tpot.export('tpot_pipeline.py')
import keras 

from keras.models import Sequential # intitialize the ANN

from keras.layers import Dense      # create layers

from keras import backend as K

from keras.wrappers.scikit_learn import KerasRegressor

from keras import metrics



X_train, X_test, y_train, y_test = train_test_split(model_X, model_Y, train_size=0.6, test_size=0.4, random_state=42)



def create_model():

    # create model

    model = Sequential()

    model.add(Dense(30, input_dim=X_train.shape[1], activation='relu'))

    model.add(Dense(100, activation='relu'))

    model.add(Dense(100, activation='relu'))

    model.add(Dense(1))

    # Compile model

    keras.optimizers.Adam(lr=300)

    model.compile(optimizer ='adam', loss = 'mean_squared_error', 

              metrics =[metrics.mae])

    return model



model = create_model()

model.summary()
# Train the ANN

history = model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=20, batch_size=128)
# summarize history for accuracy

plt.plot(history.history['mean_absolute_error'])

plt.plot(history.history['val_mean_absolute_error'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()

# summarize history for loss

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("RMSE: %f" % (rmse))

print("MAE " + str(mean_absolute_error(y_pred, y_test)))
val_predictions = model.predict(pred_df)

val_predictions = np.array(val_predictions).ravel()

val_predictions
output = pd.DataFrame({'Id': pred_df.index,

                       'SalePrice': val_predictions})



output.to_csv('submission_keras.csv', index=False)