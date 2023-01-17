import pandas as pd
import io
import requests
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import gc
import random
from sklearn.preprocessing import StandardScaler 
from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import lightgbm
dataset = pd.read_csv("/kaggle/input/brasilian-houses-to-rent/houses_to_rent_v2.csv")
dataset.shape
dataset.head()
dataset.columns
dataset.describe()
dataset['city'].unique()
dataset['animal'].unique()
dataset['furniture'].unique()
dataset['floor'].unique()
dataset['city'] = dataset['city'].replace(['São Paulo', 'Porto Alegre', 'Rio de Janeiro', 'Campinas','Belo Horizonte'],['Sao_Paulo','Porto_Alegre','Rio_Janeiro','Campinas','Belo_Horizonte'])
dataset['city'].unique()
dataset['animal'] = dataset['animal'].replace(['acept', 'not acept'],['acept', 'not_acept'])
dataset['animal'].unique()
dataset['animal'] = dataset['animal'].replace(['furnished', 'not furnished'],['furnished', 'not_furnished'])
dataset['animal'].unique()
dataset['floor'] = dataset['floor'].replace(['7', '20', '6', '2', '1', '-', '4', '3', '10', '11', '24', '9',
                                             '8', '17', '18', '5', '13', '15', '16', '14', '26', '12', '21',
                                             '19', '22', '27', '23', '35', '25', '46', '28', '29', '301', '51','32'],
                                            ['7', '20', '6', '2', '1', '0', '4', '3', '10', '11', '24', '9',
                                             '8', '17', '18', '5', '13', '15', '16', '14', '26', '12', '21',
                                             '19', '22', '27', '23', '35', '25', '46', '28', '29', '301', '51','32'])
dataset['floor'] = dataset['floor'].astype(np.float64)
dataset['floor'].unique()
dataset['id'] = dataset.index*1
dataset['id']
dataset['target'] = dataset['total (R$)'].astype(np.float64)
dataset = dataset.drop(columns=["total (R$)"])
dataset.columns
dataset.columns = ['city', 'area', 'rooms', 'bathroom', 'parking_spaces', 'floor',
                   'animal', 'furniture', 'hoa', 'rent_amount',
                   'property_tax', 'fire_insurance', 'id', 'target']
dataset.head()
#### Generate Metadata Function

def GenerateMetadata(train,var_id,targetname): 
    print('Running metadata...')
    
    for ids in var_id:
        print('Renaming ---> ', ids,'to ---> ', 'ID_'+ids)
        train = train.rename(columns={ids: 'ID_'+ids})
   
    train = train.rename(columns={targetname: 'target'})
    # Verifying type of columns
    t = []
    for i in train.columns:
            t.append(train[i].dtype)

    n = []
    for i in train.columns:
            n.append(i)

    aux_t = pd.DataFrame(data=t,columns=["Type"])
    aux_n = pd.DataFrame(data=n,columns=["Features"])
    df_tipovars = pd.merge(aux_n, aux_t, left_index=True, right_index=True) 

    data = []
    for f in train.columns:
        # Defining variable roles:
        if f == 'target':
            role = 'target'
        elif f[0:3] == 'ID_':
            role = 'id'
        else:
            role = 'input'

        # Defining variable types: nominal, ordinal, binary ou interval
        if f == 'target':
            level = 'binary'
        if train[f].dtype == 'object' or f == 'id': 
            level = 'nominal'
        elif train[f].dtype in ['float','float64'] :
            level = 'interval'
        elif train[f].dtype in ['int','int64','int32'] :
            level = 'ordinal'
        else:
            level = 'NA'

        # Remove IDs
        keep = True
        if f[0:3] == 'ID_':
            keep = False

        #  Defining the type of input table variables
        dtype = train[f].dtype

        # Metadata list
        f_dict = {
            'Features': f,
            'Role': role,
            'Level': level,
            'Keep': keep,
            'Type': dtype
        }
        data.append(f_dict)

    meta = pd.DataFrame(data, columns=['Features', 'Role', 'Level', 'Keep', 'Type'])

    # Cardinality of columns
    card = []

    v = train.columns
    for f in v:
        dist_values = train[f].value_counts().shape[0]
        f_dict = {
                'Features': f,
                'Cardinality': dist_values
            }
        card.append(f_dict)

    card = pd.DataFrame(card, columns=['Features', 'Cardinality'])

    metadata = pd.merge(meta, card, on='Features')
    print('Metadada successfully completed')
    return metadata, train 
id_list = ['id']
targetname = 'target'
metadata, dataset_01 = GenerateMetadata(dataset,id_list,targetname)
metadata
### Convert numbers to "float64" and categorical to "str"

numeric_list = metadata[((metadata.Level  == 'ordinal')|(metadata.Level == 'interval')) & (metadata.Role == 'input')]
category_list = metadata[(metadata.Level  == 'nominal') & (metadata.Role == 'input')]

numeric_list = list(numeric_list['Features'].values)
category_list = list(category_list['Features'].values)
dataset_02 = dataset_01[numeric_list].astype(np.float64)
dataset_03 = pd.merge(dataset_02, dataset_01[category_list].astype(np.str), left_index=True, right_index=True)
dataset_03.shape
dataset_03['ID_id'] = dataset_01['ID_id'].values
dataset_03['target'] = dataset_01['target'].values
dataset_03.shape
def DataPrep(metadados,input_df,var_id,targetname):
    
    print('Starting data preparation ...')
    
    #-------------- Handling missing of numeric columns -----------------
    input_df.rename(columns={var_id: 'id', targetname: 'target'}, inplace=True)
    df_00 = input_df
    targetname = 'target'
    print('Executing')
    
    #--------- Numeric Features --------------------
    vars_numericas_df = metadados[((metadados.Level  == 'ordinal')|(metadados.Level == 'interval')) & (metadados.Role == 'input')]
    lista_vars_numericas = list(vars_numericas_df['Features'])
    df01 = df_00[lista_vars_numericas]
    df01 = df01.fillna(df01[lista_vars_numericas].mean())
    df01 = df01.round(4)
    
    print('Missings done')
    
    #-------------- Numeric Features - Standart Scaler -----------------
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    df01 = df01.astype(float)
    df01[lista_vars_numericas] = scaler.fit_transform(df01[lista_vars_numericas])
    
    print('Normalization done')

    #--------- Nominal Features - Low Cardinality --------------------
    vars_char_baix_cardin_df = metadados[(metadados.Level  == 'nominal') & (metadados.Role == 'input') & (metadados.Cardinality <= 50)]
    lista_char_baix_cardin_df = list(vars_char_baix_cardin_df['Features'])
    
    df_00[lista_char_baix_cardin_df].apply(lambda x: x.fillna(x.mode, inplace=True))
    df02 = df_00[lista_char_baix_cardin_df]
    
    df03 = pd.get_dummies(df02,columns=lista_char_baix_cardin_df,drop_first=True,
                          prefix=lista_char_baix_cardin_df,prefix_sep='_')
    print('Dummifications done')    
    
    #--------- Nominal Features - High Cardinality --------------------
    vars_char_alta_cardin_df = metadados[(metadados.Level  == 'nominal') & (metadados.Role == 'input') & (metadados.Cardinality > 50)]
    lista_char_alta_cardin_df = list(vars_char_alta_cardin_df['Features'])
    
    df_00[lista_char_alta_cardin_df].apply(lambda x: x.fillna(x.mode, inplace=True)) 
    df04 = df_00[lista_char_alta_cardin_df]

    def MultiLabelEncoder(columnlist,dataframe):
        for i in columnlist:
            labelencoder_X=LabelEncoder()
            dataframe[i]=labelencoder_X.fit_transform(dataframe[i])

    MultiLabelEncoder(lista_char_alta_cardin_df,df04)
    print('Label Encodings done')
    
    #---------- Checking IDs -----------------------
    vars_ids_df = metadados[(metadados.Role  == 'id')]
    lista_ids = list(vars_ids_df['Features'])

    df1_3 = pd.merge(df01, df03, left_index=True, right_index=True)
    df1_3_4 = pd.merge(df1_3, df04, left_index=True, right_index=True)
    
    lista_vars_keep = lista_ids + [targetname]
    
    df_out = pd.merge(input_df[lista_vars_keep], df1_3_4, left_index=True, right_index=True)    
    
    print('Data Preparation Sucess')
    
    return df_out
dataset_04 = DataPrep(metadata, dataset_03,'id','target')
dataset_04.shape
dataset_04.head()
## Train/Test split

X = dataset_04.drop(['target','ID_id'], axis=1)
y = dataset_04["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=123)

X_train.shape, X_test.shape
model_lgbm = lightgbm.LGBMRegressor(n_estimators = 300,
                                    learning_rate = 0.05,
                                    max_depth = 6,
                                    num_leaves = 40,
                                    random_state = 42)

model_lgbm.fit(X_train, y_train)

y_pred_train = model_lgbm.predict(X_train)
y_pred_test = model_lgbm.predict(X_test)

residual_train = (y_train - y_pred_train).astype("float")
residual_test = (y_test - y_pred_test).astype("float")
print('Train Set Performance \n')
print('R-Squared: ', np.round(metrics.r2_score(y_train, y_pred_train, multioutput='variance_weighted'),2))
print('Mean Absolute Error: ', np.round(metrics.mean_absolute_error(y_train, y_pred_train),2))  
print('Mean Squared Error: ', np.round(metrics.mean_squared_error(y_train, y_pred_train),2))  
print('Root Mean Squared Error: ', np.round(np.sqrt(metrics.mean_squared_error(y_train, y_pred_train)),2))
print('Test Set Performance \n')
print('R-Squared: ', np.round(metrics.r2_score(y_test, y_pred_test, multioutput='variance_weighted'),2))
print('Mean Absolute Error: ', np.round(metrics.mean_absolute_error(y_test, y_pred_test),2))  
print('Mean Squared Error: ', np.round(metrics.mean_squared_error(y_test, y_pred_test),2))  
print('Root Mean Squared Error: ', np.round(np.sqrt(metrics.mean_squared_error(y_test, y_pred_test)),2))
ax = lightgbm.plot_importance(model_lgbm, max_num_features=15)
plt.show()
