import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor

from sklearn.metrics import r2_score, mean_absolute_error

from sklearn.impute import SimpleImputer



import warnings

warnings.filterwarnings('ignore')



pd.options.display.max_columns = 500



train_identity= pd.read_csv("../input/train_identity.csv")

train_transaction = pd.read_csv("../input/train_transaction.csv")

#print('Identity columns:',train_identity.shape,train_identity.columns)

#print()

#print('transaction columns: ',train_transaction.shape, train_transaction.columns)



combined = pd.merge(train_identity, train_transaction,on ='TransactionID')

object_set = combined.select_dtypes(include=['object'])
def convert_object(dataset):

    dataset['id_12'] = dataset['id_12'].map({'Found':1, 'NotFound':2})



    dataset['id_15'] = dataset['id_15'].map({'Found':1, 'NotFound':2, 'Unknown':3})

    dataset['id_15'].fillna(0)



    dataset['id_16'] = dataset['id_16'].map({'Found':1,'NotFound':2})

    dataset['id_16'].fillna(0)



    dataset['id_23'] = dataset['id_23'].map({'IP_PROXY:TRANSPARENT':1, 'IP_PROXY:ANONYMOUS':2, 

                                             'IP_PROXY:HIDDEN':3})

    dataset['id_23'].fillna(0)

    

    dataset['id_27'] = dataset['id_27'].map({'Found':1,'NotFound':2})

    dataset['id_27'].fillna(0)

    

    dataset['id_28'] = dataset['id_28'].map({'Found':1, 'New':2})

    dataset['id_28'].fillna(0)

    

    dataset['id_29'] = dataset['id_29'].map({'Found':1, 'NotFoundd':2})

    dataset['id_29'].fillna(0)

    

    dataset['id_34'] = dataset['id_34'].map({'match_status:2':1, 'match_status:1':2, 'match_status:0':3,

                                             'match_status:-1':4})

    dataset['id_34'].fillna(0)

    

    dataset['id_35'] = dataset['id_35'].map({'T':1, 'F':2})

    dataset['id_35'].fillna(0)

    

    dataset['id_36'] = dataset['id_36'].map({'T':1, 'F':2})

    dataset['id_36'].fillna(0)

    

    dataset['id_37'] = dataset['id_37'].map({'T':1, 'F':2})

    dataset['id_37'].fillna(0)

    

    dataset['id_38'] = dataset['id_38'].map({'T':1, 'F':2})

    dataset['id_38'].fillna(0)

    

    dataset['DeviceType'] = dataset['DeviceType'].map({'mobile':1, 'desktop':2})

    dataset['DeviceType'].fillna(0)

    

    dataset['ProductCD'] = dataset['ProductCD'].map({'H':1, 'C':2, 'S':3, 'R':4})

    dataset['ProductCD'].fillna(0)

    

    dataset['card4'] = dataset['card4'].map({'mastercard':1,'visa':2,'american express':3,

                                             'discover':4})

    dataset['card4'].fillna(0)

    

    dataset['card6'] = dataset['card6'].map({'credit':1, 'debit':2, 'charge card':3})

    dataset['card6'].fillna(0)

    

    return dataset

    

ready_set = convert_object(combined)

ready_set.describe()
feature_drop = ['TransactionID','dist1','D11','V1','V2','V3','V4','V5','V6','V7','V8','V9','V10','V11',

               'id_30','id_31','id_33','DeviceInfo','P_emaildomain','R_emaildomain','M1','M2','M3','M4',

                'M5','M6','M7','M8','M9']



ready_set.drop(feature_drop, axis = 1, inplace = True)

y = ready_set['isFraud']

X = ready_set

X.drop(['isFraud'], axis = 1, inplace = True)
train_X, val_X, train_y, val_y = train_test_split(X, y, test_size = 0.2, random_state = 0)



my_imputer = SimpleImputer()

imputed_train_X = pd.DataFrame(my_imputer.fit_transform(train_X))

imputed_val_X = pd.DataFrame(my_imputer.fit_transform(val_X))



imputed_train_X.columns = train_X.columns

imputed_val_X.columns = val_X.columns



model = RandomForestRegressor(n_estimators = 50)

model.fit(imputed_train_X, train_y)

pred = model.predict(imputed_val_X)



mean_absolute_error(val_y, pred)
r2_score(val_y, pred)