#Set-up (Please run this code)

import numpy as np 

import pandas as pd 

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error

from sklearn.impute import SimpleImputer
# Running this code may take a minute or so, please be patient

# Loading the data (please modify file_path if necessary)

file_path_CMgR = "../input/spikedata-2/costo_marginal_real.csv"

df_CMgR = pd.read_csv(file_path_CMgR)

file_path_CMgP = "../input/spikedata/costo_marginal_programado.csv"

df_CMgP = pd.read_csv(file_path_CMgP)





# Adjusting the names to merge the two files

df_CMgR = df_CMgR.set_index(['barra_mnemotecnico','fecha','hora'])

df_CMgP = df_CMgP.drop(['nombre_barra'],axis=1)

df_CMgP = df_CMgP.rename(columns={"mnemotecnico_barra": 'barra_mnemotecnico',"costo":"CMg_prog"}).set_index(['barra_mnemotecnico','fecha','hora'])



# Finally we merge the files, drop indexes and read <fecha> as a Datetime object

costo_marginal = df_CMgR.join(df_CMgP).reset_index()

costo_marginal['fecha'] = pd.to_datetime(costo_marginal['fecha'])



print("All set, thanks for being patient!")
# Drop and Rename variables

costo_marginal = costo_marginal.drop(['costo_en_pesos'],axis=1)

costo_marginal = costo_marginal.rename(columns={"costo_en_dolares": 'CMg_real'})



# Define new variables

costo_marginal['desviacion'] = costo_marginal.CMg_real-costo_marginal.CMg_prog

costo_marginal['desviacion_pct'] = costo_marginal.desviacion/costo_marginal.CMg_prog

costo_marginal['desviacion_cat'] = costo_marginal.desviacion_pct.abs() > 15                          
def time_plot_costo_barra(codigo_barra, fecha_inicial, fecha_final):

    aux = costo_marginal.loc[(costo_marginal.barra_mnemotecnico == codigo_barra) & (costo_marginal.fecha >=fecha_inicial) & (costo_marginal.fecha<=fecha_final)][["CMg_real","CMg_prog","fecha"]]

    aux = aux.set_index('fecha')

    plt.figure(figsize=(14,6))

    plt.title("Marginal cost in station "+codigo_barra)

    ax = sns.lineplot(data=aux)
fecha_inicial = '2019-01-04'

fecha_final = '2019-06-30'

codigo_barra = costo_marginal.barra_mnemotecnico[3]

time_plot_costo_barra(codigo_barra, fecha_inicial, fecha_final)
# Loading the data (please modify file_path if necessary)

file_path = "../input/spikedata/base_para_prediccion.csv"

df = pd.read_csv(file_path)
# Converting to Datetime

df['fecha'] = pd.to_datetime(df['fecha'])



# Adding new variables

df['year'] = df.fecha.dt.year

df['month'] = df.fecha.dt.month

df['day'] = df.fecha.dt.day

df['day_of_week'] = df.fecha.dt.dayofweek

df['weekend'] = (df.day_of_week > 4)
def plot_var_dates(codigo_estacion,variable,fechas):

    plt.figure(figsize=(14,6))

    plt.title("Evolution of "+variable+" in station "+codigo_estacion)

    plt.xlabel("Time of the day")

    aux = pd.DataFrame(columns = fechas) 

    for date in fechas:

        aux[date] = df.loc[(df.nemotecnico_se == codigo_estacion) & (df.fecha == date) , [variable,'hora']].set_index('hora')[variable]

    sns.lineplot(data=aux)
# First we need to define the variables

fechas = ['2019-01-10','2019-01-11','2019-01-12','2019-01-13','2019-01-14']

variable = 'gen_solar_total_mwh'

codigo_estacion_1 = 'SE005T002'

codigo_estacion_2 = 'SE127T005'



# Now we can generate the two graphs using the function we just defined

plot_var_dates(codigo_estacion_1,variable,fechas)

plot_var_dates(codigo_estacion_2,variable,fechas)
# Once again, we need to define the variables

fechas = ['2019-05-14','2019-05-15','2019-05-16','2019-05-17','2019-05-18','2019-05-19']

variable = 'gen_termica_total_mwh'

codigo_estacion_1 = 'SE020G213'

codigo_estacion_2 = 'SE106G216'



# Now we can generate the two graphs using the function we just defined

plot_var_dates(codigo_estacion_1,variable,fechas)

plot_var_dates(codigo_estacion_2,variable,fechas)
# Drop station with real marginal cost = 0

df=df.drop(df.loc[df.nemotecnico_se == df.nemotecnico_se[16431]].index)



# Target Variables

df['target'] = (df.cmg_desv_pct.abs() > 15) * 1



# Total Supply Variable

variables_supply = ['gen_eolica_total_mwh','gen_geotermica_total_mwh','gen_hidraulica_total_mwh','gen_solar_total_mwh','gen_termica_total_mwh']

df['en_total_mwh'] = df[variables_supply].sum(axis=1,skipna=True)



# Average Variables

df['cumavg_demand'] = df.sort_values(by=['fecha']).groupby(['hora','weekend','nemotecnico_se'])['demanda_mwh'].apply(lambda x: x.shift().expanding().mean())

df['cumavg_supply'] = df.sort_values(by=['fecha']).groupby(['hora','weekend','nemotecnico_se'])['en_total_mwh'].apply(lambda x: x.shift().expanding().mean())





# Absolute Deviations

df['abs_dev_demand'] = (df.cumavg_demand-df.demanda_mwh).abs()

df['abs_dev_supply'] = (df.cumavg_supply-df.en_total_mwh).abs()



# Lagged Variables

df['dev_demand_lagH1'] = df.sort_values(by=['fecha','hora']).groupby(['nemotecnico_se'])['abs_dev_demand'].shift(1)

df['dev_supply_lagH1'] = df.sort_values(by=['fecha','hora']).groupby(['nemotecnico_se'])['abs_dev_supply'].shift(1)

df['dev_demand_lagD1'] = df.sort_values(by=['fecha']).groupby(['nemotecnico_se','hora'])['abs_dev_demand'].shift(1)

df['dev_supply_lagD1'] = df.sort_values(by=['fecha']).groupby(['nemotecnico_se','hora'])['abs_dev_supply'].shift(1)



# Target and Explanatory Variables

y = df.target

features = ['dev_demand_lagH1','dev_demand_lagD1','dev_supply_lagH1','dev_supply_lagD1','month','day','hora']

X = df[features]
# Split data

train_X,val_X,train_y,val_y = train_test_split(X,y,random_state = 1)



# Imputation for NaN values

imputer = SimpleImputer()

imp_train_X = pd.DataFrame(imputer.fit_transform(train_X))

imp_val_X = pd.DataFrame(imputer.transform(val_X))

imp_train_X.columns = train_X.columns

imp_val_X.columns = val_X.columns



# Model

model = DecisionTreeRegressor(random_state=1)



# Fit model

model.fit(imp_train_X,train_y)



# Predictions and absolute error

val_pred = model.predict(imp_val_X)

print("Mean Absolute Error:  %f" %(mean_absolute_error(val_y, val_pred)))
# Moving Averages Variables

df['ma_demand'] = df.sort_values(by=['fecha','hora']).groupby(['nemotecnico_se'])['abs_dev_demand'].shift(11).rolling(24).mean()

df['ma_supply'] = df.sort_values(by=['fecha','hora']).groupby(['nemotecnico_se'])['abs_dev_supply'].shift(11).rolling(24).mean()



# Explanatory Variables

features_2 = ['ma_demand','dev_demand_lagD1','ma_supply','dev_supply_lagD1','month','day','hora']

X_2 = df[features_2]





# Split data

train_X_2,val_X_2,train_y_2,val_y_2 = train_test_split(X_2,y,random_state = 1)



# Imputation for NaN values

imputer_2 = SimpleImputer()

imp_train_X_2 = pd.DataFrame(imputer_2.fit_transform(train_X_2))

imp_val_X_2 = pd.DataFrame(imputer_2.transform(val_X_2))

imp_train_X_2.columns = train_X_2.columns

imp_val_X_2.columns = val_X_2.columns



# Model

model_2 = DecisionTreeRegressor(random_state=1)



# Fit model

model_2.fit(imp_train_X_2,train_y_2)



# Predictions and absolute error

val_pred_2 = model_2.predict(imp_val_X_2)

print("Mean Absolute Error:  %f" %(mean_absolute_error(val_y_2, val_pred_2)))
# Loading weather data (please modify file_path if necessary)

file_path = "../input/spikedata/datos_clima.csv"

df_C = pd.read_csv(file_path)



# Adjusting the names to merge the two files

df_2 = df.set_index(['nemotecnico_se','fecha'])

#df_CMgP = df_CMgP.drop(['nombre_barra'],axis=1)

df_C['fecha'] = pd.to_datetime(df_C['fecha'])

df_C = df_C.rename(columns={"subestacion": 'nemotecnico_se'}).set_index(['nemotecnico_se','fecha'])



# Finally we merge the files and drop indexes

df_2= df_2.join(df_C).reset_index()
# Cumulative Average Variables

df_2['cumavg_PRECTOT'] = df_2.sort_values(by=['fecha']).groupby(['month','nemotecnico_se'])['PRECTOT'].apply(lambda x: x.shift().expanding().mean())

df_2['cumavg_T2M'] = df_2.sort_values(by=['fecha']).groupby(['month','nemotecnico_se'])['T2M'].apply(lambda x: x.shift().expanding().mean())



# Absolute Deviations

df_2['abs_dev_PRECTOT'] = (df_2.cumavg_PRECTOT-df_2.PRECTOT).abs()

df_2['abs_dev_T2M'] = (df_2.cumavg_T2M-df_2.T2M).abs()



# Explanatory Variables

features_3 = ['ma_demand','dev_demand_lagD1','ma_supply','dev_supply_lagD1','month','day','hora','abs_dev_PRECTOT','abs_dev_T2M']

X_3 = df_2[features_3]





# Split data

train_X_3,val_X_3,train_y_3,val_y_3 = train_test_split(X_3,y,random_state = 1)



# Imputation for NaN values

imputer_3 = SimpleImputer()

imp_train_X_3 = pd.DataFrame(imputer_3.fit_transform(train_X_3))

imp_val_X_3 = pd.DataFrame(imputer_3.transform(val_X_3))

imp_train_X_3.columns = train_X_3.columns

imp_val_X_3.columns = val_X_3.columns



# Model

model_3 = DecisionTreeRegressor(random_state=1)



# Fit model

model_3.fit(imp_train_X_3,train_y_3)



# Predictions and absolute error

val_pred_3 = model_3.predict(imp_val_X_3)

print("Mean Absolute Error:  %f" %(mean_absolute_error(val_y_3,val_pred_3)))