# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
import csv
import seaborn as sns
import math
import matplotlib
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV, Lasso
from sklearn.model_selection import cross_val_score
import xgboost as xgb
from sklearn import preprocessing, ensemble
import matplotlib.pyplot as plt

df_test = pd.DataFrame()
for row in csv.DictReader(open('../input/test_ver2.csv')):
    if row['fecha_dato'] not in ['2015-05-28', '2015-06-28', '2016-05-28', '2016-06-28']:
	    continue       
    df_test.append(row, ignore_index = True)
import seaborn as sns
import math
import matplotlib
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV, Lasso
from sklearn.model_selection import cross_val_score
import xgboost as xgb
from sklearn import preprocessing, ensemble
import matplotlib.pyplot as plt

df_train = pd.read_csv("../input/train_ver2.csv", nrows = 2700000)
df_test = pd.read_csv("../input/test_ver2.csv", nrows = 100000)

df_temp = pd.concat([df_train,df_test])
fields = df_temp.columns
target_y = []
target_x = []
for field in fields:
    if field[-5:-1] == '_ult':
        target_y.append(field)
    else:
        target_x.append(field)
    
previous_data = ['renta','ncodpers']+target_y
df_1505 = df_temp.loc[df_temp.fecha_dato == '2015-05-28',previous_data]
df_1506 = df_temp.loc[df_temp.fecha_dato == '2015-06-28',previous_data]
df_1604 = df_temp.loc[df_temp.fecha_dato == '2016-04-28',previous_data]
df_1605 = df_temp.loc[df_temp.fecha_dato == '2016-05-28']
df_1606 = df_temp.loc[df_temp.fecha_dato == '2016-06-28']

df_1606 = df_1606.join(df_1506, on = 'ncodpers', how = 'left', rsuffix = '_prev_year')
df_1606 = df_1606.join(df_1605[previous_data], on = 'ncodpers', how = 'left', rsuffix = '_prev_month')
df_1606 = df_1606.drop(['ncodpers_prev_year', 'ncodpers_prev_month'],axis = 1)

###### used outer join for test purposes. When reloaded fully with all data should use left join! 
df_1605 = df_1605.join(df_1505, on = 'ncodpers', how = 'outer', rsuffix = '_prev_year')
df_1605 = df_1605.join(df_1604, on = 'ncodpers', how = 'outer', rsuffix = '_prev_month')
df_1605 = df_1605.drop(['ncodpers_prev_year', 'ncodpers_prev_month'],axis = 1)

df_all = df_1605
df_test = df_1606
def process_data(data):
    data.loc[data['ind_nuevo'].isnull(),'ind_nuevo'] = 1
    
    data["age"]   = pd.to_numeric(data["age"], errors="coerce")
    data.loc[data.age < 18,"age"] = data.loc[(data.age >= 18) & (data.age <= 30),"age"].mean(skipna=True)
    data.loc[data.age > 100,'age'] = data.loc[(data.age>= 31) & (data.age <= 100),'age'].mean(skipna=True)
    data.loc[data.age.isnull(),'age'] = data.loc[(data.age>= 31) & (data.age <= 100),'age'].mean(skipna=True)
    
    data['fecha_dato'] = pd.to_datetime(data['fecha_dato'],format ="%Y-%m")
    data['fecha_dato_year'] = data.fecha_dato.apply(lambda x: x.year)
    data['fecha_dato_month'] = data.fecha_dato.apply(lambda x: x.month)
    data = data.drop('fecha_dato', axis = 1)
    
    data['fecha_alta'] = pd.to_datetime(data['fecha_alta'],format ="%Y-%m-%d")
    data['fecha_alta_year'] = data.fecha_alta.apply(lambda x: x.year)
    data['fecha_alta_month'] = data.fecha_alta.apply(lambda x: x.month)
    data = data.drop('fecha_alta', axis = 1)
    
    data.loc[data.ind_empleado.isnull(),'ind_empleado'] = 'N'
    
    data.loc[data.pais_residencia.isnull(),'pais_residencia'] = 'ES'
    
    data.loc[data.sexo.isnull(),'sexo'] = 'V'
    
    data['antiguedad'] = pd.to_numeric(data['antiguedad'], errors = 'coerce')
    data.loc[data.antiguedad == -999999,'antiguedad'] = 0
    data.loc[data.antiguedad.isnull(),'antiguedad'] = 0
    
    data.loc[data.indrel == 99,'indrel'] = 0
    
    data.loc[data.ult_fec_cli_1t.notnull(),'ult_fec_cli_1t_value'] = 1
    data.loc[data.ult_fec_cli_1t_value.isnull(),'ult_fec_cli_1t_value'] = 0
    data = data.drop('ult_fec_cli_1t', axis = 1)
    
    data.loc[data.canal_entrada.isnull(),'canal_entrada'] = 'OTH'
    
    data = data.drop('tipodom', axis = 1)
    
    data.loc[data.nomprov.isnull(),'nomprov'] = 'NonSpain' 
    data = data.drop('cod_prov', axis = 1)
    
    data.loc[data.segmento.isnull(),'segmento'] = '02 - PARTICULARES'
    data.loc[data.conyuemp.isnull(),'conyuemp'] = 'N'
    
    data.loc[data.ind_actividad_cliente.isnull(),'ind_actividad_cliente'] = -99
    
    data.loc[data.indext.isnull(),'indext'] = 'N'
    data.loc[data.indfall.isnull(),'indfall'] = 'N'
    data.loc[data.indrel.isnull(),'indrel'] = 1
    data.loc[data.indresi.isnull(),'indresi'] = 'S'
    data.loc[data.indrel_1mes.isnull(),'indrel_1mes'] = 1
    data.loc[data.tiprel_1mes.isnull(),'tiprel_1mes'] = -99

    data.loc[data.fecha_alta_year.isnull(),'fecha_alta_year'] = data.loc[data.fecha_alta_year.isnull(),'fecha_dato_year']
    data.loc[data.fecha_alta_month.isnull(),'fecha_alta_month']= data.loc[data.fecha_alta_month.isnull(),'fecha_dato_month']
    
    
    data["renta"] = pd.to_numeric(data["renta"], errors="coerce")
    data['renta_prev_year'] = pd.to_numeric(data['renta_prev_year'], errors = 'coerce')
    data['renta_prev_month'] = pd.to_numeric(data['renta_prev_month'], errors = 'coerce')
    
    data.loc[data.renta.isnull(),'renta'] = data.loc[data.renta.isnull(),'renta_prev_month']
    data.loc[data.renta_prev_month.isnull(),'renta_prev_month'] = data.loc[data.renta_prev_month.isnull(),'renta']
    data.loc[data.renta_prev_year.isnull(),'renta_prev_year'] = data.loc[data.renta_prev_year.isnull(),'renta']
    
    data.renta = data.renta.apply(lambda x: math.log(x))
    data.renta_prev_month = data.renta_prev_month.apply(lambda x: math.log(x))
    data.renta_prev_year = data.renta_prev_year.apply(lambda x: math.log(x))
    
    if str(data) == str(df_all):
        data.loc[(data.fecha_dato_month != 6) & (data.fecha_dato_year != 2016) & (data.ind_nomina_ult1.isnull()),'ind_nomina_ult1'] = 0
        data.loc[(data.fecha_dato_month != 6) & (data.fecha_dato_year != 2016) & (data.ind_nom_pens_ult1.isnull()),'ind_nom_pens_ult1'] = 0
    return data

process_data(df_all)
process_data(df_test)

def convert_dummies(data):
    to_dummies = ['ind_empleado','ind_actividad_cliente','pais_residencia','indrel_1mes','tiprel_1mes','canal_entrada','nomprov','segmento']
    for dummy in to_dummies:
        print('Converting ' +dummy+ ' to dummies')
        data = pd.get_dummies(data,columns = [dummy])
        print('Completed')
        print('____________________________________________________________')
    return data

convert_dummies(df_all)
convert_dummies(df_test)

def convert_categories(data):
    to_category_codes = ['sexo','indresi','indext','indfall', 'conyuemp']
    for category in to_category_codes:
        print('Converting ' +category+ ' to category codes')
        data[category] = data[category].astype('category').cat.codes
        print('Completed')
        print('____________________________________________________________')
    return data

convert_categories(df_all)
convert_categories(df_test)
df_all
def data_sets(data):
    x_data = data.loc[(data.renta.notnull())].drop(['renta','ncodpers']+target_y, axis = 1)
    x_data_test = data.loc[(data.renta.isnull())].drop(['renta','ncodpers']+target_y, axis = 1)
    y_data = data.loc[data.renta.notnull(),'renta']
    print(x_data)
    return(x_data, x_data_test, y_data)


#def rmse_cv(model):
#    rmse= np.sqrt(-cross_val_score(model, x_data, y_data, scoring="neg_mean_squared_error", cv = 3))
#    return(rmse)
    
def process_renta(x_data, y_data):
    alphas = [ 5, 10, 20,30]
    model_ridge = RidgeCV(alphas = alphas, cv = 3, scoring="neg_mean_squared_error").fit(x_data, y_data)
    return(model_ridge)

def do_renta(data):
    ridge_preds = model_ridge.predict(x_data_test)
    data.loc[data.renta.isnull(),'renta'] = ridge_preds
    data.loc[data.renta_prev_month.isnull(),'renta_prev_month'] = data.loc[data.renta_prev_month.isnull(),'renta']
    data.loc[data.renta_prev_year.isnull(),'renta_prev_year'] = data.loc[data.renta_prev_year.isnull(),'renta']
    
    data['renta_change_year'] = data['renta'] - data['renta_prev_year']
    data['renta_change_month'] = data['renta'] - data['renta_prev_month']
    data.drop(['renta_prev_year','renta_prev_month'], axis = 1)
    return(data)

def normalize(data):
    for field in target_x:
        if field != 'conyuemp':
            print(field)
            f_min = data[field].min()
            f_max = data[field].max()
            data[field] = (data[field]-f_min)/(f_max-f_min)
    return(data)

data_sets(df_all)
process_renta(x_data, y_data)
do_renta(df_all)
normalize(df_all)

data_sets(df_test)
process_renta(x_data, y_data)
do_renta(df_test)
normalize(df_test)
train_x = df_all.loc[df_all['ind_cco_fin_ult1'].notnull(),target_x].dropna()
train_y = df_all[target_y].dropna()
#train_x = np.array(df_all[target_x])
#train_y = np.array(df_all[target_y].dropna())

def runXGB(train_x, train_y, seed_val=123):
    param = {}
    param['objective'] = 'multi:softprob'
    param['eta'] = 0.05
    param['max_depth'] = 6
    param['silent'] = 1
    param['num_class'] = 22
    param['eval_metric'] = "mlogloss"
    param['min_child_weight'] = 2
    param['subsample'] = 0.9
    param['colsample_bytree'] = 0.9
    param['seed'] = seed_val
    num_rounds = 20
    
    plst = list(param.items())
    xgtrain = xgb.DMatrix(train_x, label=np.array(train_y))
    model = xgb.train(plst, xgtrain, num_rounds)	
    return model
print('function created')
model = runXGB(train_x, train_y, seed_val=0)

test_x = df_all.loc[df_all['ind_cco_fin_ult1'],target_x].isnull()
xgtest = xgb.DMatrix(test_x)
preds = model.predict(xgtest)
train_y = df_all[target_y].dropna()
print(np.array(train_y))
#train_x = df_all.loc[df_all['ind_cco_fin_ult1'].notnull(),target_x].dropna()
#train_y = df_all['ind_cco_fin_ult1'].dropna()
train_x = df_all.loc[df_all['ind_cco_fin_ult1'].notnull(),target_x].dropna()
train_y = df_all[targets_y].dropna()
#train_x = np.array(df_all[target_x])
#train_y = np.array(df_all[target_y].dropna())

def runXGB(train_x, train_y, seed_val=123):
    param = {}
    param['objective'] = 'multi:softprob'
    param['eta'] = 0.05
    param['max_depth'] = 6
    param['silent'] = 1
    param['num_class'] = 22
    param['eval_metric'] = "mlogloss"
    param['min_child_weight'] = 2
    param['subsample'] = 0.9
    param['colsample_bytree'] = 0.9
    param['seed'] = seed_val
    num_rounds = 20
    
    plst = list(param.items())
    xgtrain = xgb.DMatrix(train_x, label=train_y)
    model = xgb.train(plst, xgtrain, num_rounds)	
    return model
print('function created')
model = runXGB(train_x, train_y, seed_val=0)

test_x = df_all.loc[df_all['ind_cco_fin_ult1'],target_x].isnull()
xgtest = xgb.DMatrix(test_x)
preds = model.predict(xgtest)
def rmse_cv(model):
    rmse= np.sqrt(-cross_val_score(model, x_df_train, y_df_train, scoring="neg_mean_squared_error", cv = 3))
    return(rmse)

x_df_train = df_train.loc[df_train.renta.notnull()].drop(['renta','ncodpers','sexo'], axis = 1)
x_df_test = df_train.loc[df_train.renta.isnull()].drop(['renta','ncodpers','sexo'], axis = 1)
y_df_train = df_train.loc[df_train.renta.notnull(),'renta']
#df_train.loc[df_train.renta.notnull(),['age','sexo','antiguedad']]
#df_train.loc[df_train.renta.isnull()].drop(['renta','ncodpers'], axis = 1)
alphas = [ 5, 10, 20,30]
model_ridge = RidgeCV(alphas = alphas, cv = 3, scoring="neg_mean_squared_error").fit(x_df_train, y_df_train)
#cv_ridge = [rmse_cv(Ridge(alpha = alpha)).mean() 
#            for alpha in alphas]
cv_ridge = []
for alpha in alphas:
    cv_ridge.append(rmse_cv(Ridge(alpha = alpha)).mean())

print(cv_ridge)
print(rmse_cv(model_ridge).mean())

model_lasso = LassoCV(alphas = [5,10,20,30,50]).fit(x_df_train, y_df_train)
cv_lasso = []
for alpha in alphas:
    cv_lasso.append(rmse_cv(Lasso(alpha = alpha)).mean())
    
print(cv_lasso)
print(rmse_cv(model_lasso).mean())
coef = pd.Series(model_ridge.coef_, index = x_df_train.columns)
imp_coef = pd.concat([coef.sort_values().head(10),
                     coef.sort_values().tail(10)])

print(coef)

matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
imp_coef.plot(kind = "barh")
plt.title("Coefficients in the Ridge Model")
ridge_preds = model_ridge.predict(x_df_test)
sns.distplot(ridge_preds)
df_train.loc[df_train.renta.isnull(),'renta'] = ridge_preds
sns.distplot(df_train.renta)
train_y_cols = []
train_x_cols = []
fields = df_train.columns
for field in fields:
    if field[-4:-1] =='ult':
        train_y_cols.append(field)
    else:
        train_x_cols.append(field)
print(train_y_cols)
print('__________________')
print(train_x_cols)
train_x = df_train[train_x_cols].drop('ncodpers', axis = 1)
train_y = df_train[[train_y_cols]]
def runXGB(train_x, train_y, seed_val=123):
    param = {}
    param['objective'] = 'multi:softprob'
    param['eta'] = 0.05
    param['max_depth'] = 6
    param['silent'] = 1
    param['num_class'] = 22
    param['eval_metric'] = "mlogloss"
    param['min_child_weight'] = 2
    param['subsample'] = 0.9
    param['colsample_bytree'] = 0.9
    param['seed'] = seed_val
    num_rounds = 20
    
    plst = list(param.items())
    xgtrain = xgb.DMatrix(train_x, label=train_y)
    model = xgb.train(plst, xgtrain, num_rounds)	
    return model
model = runXGB(train_x, train_y, seed_val=123)
xgtest = xgb.DMatrix(train_x)
preds = model.predict(xgtest)
print(preds)