# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

from sklearn.metrics import mean_absolute_error

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder

from sklearn.tree import DecisionTreeRegressor

ncaa = pd.read_csv("../input/ncaaluis/ncaap.csv")
#ncaa.describe()

ncaa.isnull().sum()

ncaa.head()
ncaa['ScoreRate'] = ncaa['WScore']/ncaa['LScore']
ncaa.head()
ncaa.columns
# Declaro variables seleccionadas y = ScoreRate

# Se eliminará Wscore y LScrore

# Extraigo todo lo mayor a 2015 y se lo doy a X_test



X_test = ncaa.drop(ncaa[ncaa.Season < 2015].index)

ncaa = ncaa.drop(ncaa[ncaa.Season >= 2015].index)

y = ncaa.ScoreRate

ncaa_features = ['Season', 'DayNum', 'WTeamID', 'wcoach', 'WConfCode',

       'LTeamID', 'LCoach', 'LConfCode', 'WLoc']

X = ncaa[ncaa_features]
X.describe()
#Se fracciona la muestras entre entrenamiento y validación

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, random_state = 0)
def score_dataset(X_train, X_val, y_train, y_val):

    model = RandomForestRegressor(n_estimators=100, random_state=0)

    model.fit(X_train, y_train)

    preds = model.predict(X_val)

    return mean_absolute_error(y_val, preds)
#Solo para saber cuantas cacillas vacias habia

#X.isnull().sum()

#X_val.loc[:, X_val.isnull().any()]

#nan_col = [i for i in X_val.columns if X_val[i].isnull().any()]



#opcion con Label Encoder



from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import LabelEncoder

s = (X_train.dtypes == 'object')

object_cols = list(s[s].index)

label_X_train = X_train.copy()

label_X_valid = X_val.copy()

# Apply label encoder to each column with categorical data

label_encoder = LabelEncoder()

for col in object_cols:

    label_X_train[col] = label_encoder.fit_transform(X_train[col])

    label_X_valid[col] = label_encoder.transform(X_val[col])



print("MAE from Approach 2 (Label Encoding):") 

#Abajo entre comillas por tener aun valores con NA

#print(score_dataset(label_X_train, label_X_valid, y_train, y_val))
label_X_train.describe()
# Usand table compacta, no requiere usar imputer dado que todo los datos están acá

from sklearn.impute import SimpleImputer



# Imputation

my_imputer = SimpleImputer()

imputed_X_train = pd.DataFrame(my_imputer.fit_transform(label_X_train))

imputed_X_valid = pd.DataFrame(my_imputer.transform(label_X_valid))



# Imputation removed column names; put them back

imputed_X_train.columns = label_X_train.columns

imputed_X_valid.columns = label_X_valid.columns



print("MAE from Approach 2 (Imputation):")

print(score_dataset(imputed_X_train, imputed_X_valid, y_train, y_val))
#from sklearn.model_selection import train_test_split



#ncaa_model = RandomForestRegressor(random_state=1)

#ncaa_model.fit(train_X, train_y)

#ncaa_preds = ncaa_model.predict(val_X)

#print(mean_absolute_error(val_y, ncaa_preds))

# Prueba con Imputer Extended, Agrega columna donde hubo cambio. El resutado fue tirando a mejor, pero sólo 0.00002





# Get names of columns with missing values

cols_with_missing = [col for col in X_train.columns

                     if X_train[col].isnull().any()]



# Prueba con Imputer Extended, Agrega columna donde hubo cambio



X_train_plus = label_X_train.copy()

X_valid_plus = label_X_valid.copy()



# Make new columns indicating what will be imputed

for col in cols_with_missing:

    X_train_plus[col + '_was_missing'] = X_train_plus[col].isnull()

    X_valid_plus[col + '_was_missing'] = X_valid_plus[col].isnull()



# Imputation

my_imputer = SimpleImputer()

imputed_X_train_plus = pd.DataFrame(my_imputer.fit_transform(X_train_plus))

imputed_X_valid_plus = pd.DataFrame(my_imputer.transform(X_valid_plus))



# Imputation removed column names; put them back

imputed_X_train_plus.columns = X_train_plus.columns

imputed_X_valid_plus.columns = X_valid_plus.columns

print("MAE con Imputer extended usando Label:")

print(score_dataset(imputed_X_train_plus, imputed_X_valid_plus, y_train, y_val))
#Funcionó pero sólo con OH, Es mejor usar Label por eso es mejor no correr esta fase.



from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder

from sklearn.metrics import mean_absolute_error



numerical_transformer = SimpleImputer(strategy='mean')



categorical_cols = [cname for cname in X_train.columns if

                    X_train[cname].nunique() < 10 and 

                    X_train[cname].dtype == "object"]



# Select numerical columns

numerical_cols = [cname for cname in X_train.columns if 

                X_train[cname].dtype in ['int64', 'float64']]



# Preprocessing for categorical data

categorical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('onehot', OneHotEncoder(handle_unknown='ignore'))

])



# Bundle preprocessing for numerical and categorical data

preprocessor = ColumnTransformer(

    transformers=[

        ('num', numerical_transformer, numerical_cols),

        ('cat', categorical_transformer, categorical_cols)

    ])





model = RandomForestRegressor(n_estimators=300, random_state=0)



my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),

                              ('model', model)

                             ])



# Preprocessing of training data, fit model 

my_pipeline.fit(X_train, y_train)



# Preprocessing of validation data, get predictions

preds = my_pipeline.predict(X_val)



# Evaluate the model

score = mean_absolute_error(y_val, preds)

print('MAE:', score)



y_train.describe()
from xgboost import XGBRegressor

X_reg_train = label_X_train

X_reg_valid = label_X_valid



my_model = XGBRegressor(n_estimators=1700, learning_rate=0.13, n_jobs=4)

my_model.fit(X_reg_train, y_train, 

             early_stopping_rounds=6, 

             eval_set=[(X_reg_valid, y_val)], 

             verbose=False)



predictions = my_model.predict(X_reg_valid)

print("Mean Absolute Error: " + str(mean_absolute_error(predictions, y_val)))
X_test.describe()

chop_X_test = X_test.drop(['WScore','LScore','ScoreRate'], axis=1)
chop_X_test.head()
#Label Encoder para los valores de test



s = (X_test.dtypes == 'object')

object_cols = list(s[s].index)

label_X_test = chop_X_test.copy()



# Apply label encoder to each column with categorical data

label_encoder_test = LabelEncoder()

for col in object_cols:

    label_X_test[col] = label_encoder_test.fit_transform(X_test[col])

    
label_X_test.isnull().sum()
preds_test = my_model.predict(label_X_test)

output = pd.DataFrame({'Season': label_X_test.Season, 'WID': label_X_test.WTeamID, 'LID': label_X_test.LTeamID,

                       'Pred': preds_test})

#output.to_csv('submission.csv', index=False)
output.iloc[:50, :50]
output.head()
output2 = output
output2 = pd.DataFrame (columns = ['ID','Pred'])
output.isnull().sum()
mayor = output.where(output['WID'] > output['LID'])

menor = output.where(output['LID'] > output['WID'])

mayor = mayor.dropna()

menor = menor.dropna()

mayor['Season'] = mayor['Season'].astype(np.int64)

mayor['WID'] = mayor['WID'].astype(np.int64)

mayor['LID'] = mayor['LID'].astype(np.int64)

menor['Season'] = menor['Season'].astype(np.int64)

menor['WID'] = menor['WID'].astype(np.int64)

menor['LID'] = menor['LID'].astype(np.int64)
output2 = mayor.assign(ID = mayor.Season.astype(str) + "_" + mayor.LID.astype(str) + "_" + mayor.WID.astype(str))

output3 = menor.assign(ID = menor.Season.astype(str) + "_" + menor.WID.astype(str) + "_" + menor.LID.astype(str))

frames = [output2, output3]

output4 = pd.concat(frames)
'''  if output4['WID'] > output4['LID']:

    output4 = output4.assign(Pred2 = output4.Pred * 50)

else:

    output4 = output4.assign(Pred2 = (1/output4.Pred) * 50)

    ''' 

    

output4.loc[output4['WID'] > output4['LID'], 'Pred2'] = output4['Pred'] * 50

output4.loc[output4['WID'] < output4['LID'], 'Pred2'] = 1/output4['Pred'] * 50

outputf = output4.drop(['Season', 'WID', 'LID', 'Pred'], axis=1)

outputf
outputf = outputf.rename(columns={"Pred2": "Pred"})

outputf
outputf.to_csv('submission.csv', index=False)