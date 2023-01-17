###### Library de base

import pandas as pd

from datetime import datetime

import numpy as np

import random

import matplotlib.pyplot as plt

import seaborn as sns

import gc



###### Librabary Sklearn



from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_log_error

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import mean_squared_error

from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import RandomizedSearchCV





###### Library LightGBM



import lightgbm as lgb



###### Kaggle Specifique



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
##############

#### Importation de toute la base

##############





#train=pd.read_csv("/kaggle/input/ashrae-energy-prediction/train.csv",sep=",")

weather_train=pd.read_csv("/kaggle/input/ashrae-energy-prediction/weather_train.csv",sep=",")

building_meta=pd.read_csv("/kaggle/input/ashrae-energy-prediction/building_metadata.csv",sep=',')
##############

####  Importation d'un Pourcentage de la base

##############



random.seed(30)

# On selectionne 20% des lignes de la bases pour tester les modèles

p = 0.1  # 20% of the lines

# keep the header, then take only 1% of lines

# if random from [0,1] interval is greater than 0.01 the row will be skipped

train = pd.read_csv(

         "/kaggle/input/ashrae-energy-prediction/train.csv",sep=",",

         header=0, 

         skiprows=lambda i: i>0 and random.random() > p

)
##############

#### Fonction pour réduire la taille de la base

##############





def reduce_mem_usage(df, verbose=True):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024**2    

    for col in df.columns:

        col_type = df[col].dtypes

        if col_type in numerics:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)    

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df



# Source : https://www.kaggle.com/caesarlupum/ashrae-start-here-a-gentle-introduction
##############

#### Fonction pour calculer les données manquantes

##############







def missing_statistics(df):    

    statitics = pd.DataFrame(df.isnull().sum()).reset_index()

    statitics.columns=['COLUMN NAME',"MISSING VALUES"]

    statitics['TOTAL ROWS'] = df.shape[0]

    statitics['% MISSING'] = round((statitics['MISSING VALUES']/statitics['TOTAL ROWS'])*100,2)

    return statitics



# Source : https://www.kaggle.com/aitude/ashrae-missing-weather-data-handling
##############

#### Visualisation des valeurs manquantes

##############



def val_manq(df):

    percent = (df.isnull().sum()).sort_values(ascending=False)

    percent.plot(kind='bar', figsize = (20,10), fontsize=20)

    plt.xlabel('Columns')

    plt.ylabel('Number of rows')

    plt.title('Tot missing values')

val_manq(train)
##############

#### Pourcentage de données manquantes

##############





print(missing_statistics(train))

print(missing_statistics(weather_train))

print(missing_statistics(building_meta))
##############

#### Création de nouvelles colonnes

##############



weather_train["datetime"] = pd.to_datetime(weather_train["timestamp"])

weather_train["annee"] = weather_train["datetime"].dt.year

weather_train["mois"] = weather_train["datetime"].dt.month

weather_train["semaine"] = weather_train["datetime"].dt.week

weather_train["jour"] = weather_train["datetime"].dt.day

weather_train["heure"] = weather_train["datetime"].dt.hour

weather_train.drop(columns=["datetime"],inplace=True)
##############

#### Remplisssage des données manquantes de la base weather par des moyennes

##############





col_weather=["air_temperature","cloud_coverage","dew_temperature","precip_depth_1_hr","sea_level_pressure","wind_direction","wind_speed"]

mean_weather=weather_train.groupby(["site_id","mois"]).transform(lambda x: x.fillna(x.mean()))

for col in col_weather:

    weather_train[col] = mean_weather[col]

weather_train=weather_train.fillna(method='ffill')

building_meta.drop(columns=["year_built","floor_count"],inplace=True)
##############

#### Encoding d'une variables qualitative

##############

#A faire après les graphiques



labelencoder=LabelEncoder()

building_meta["primary_use"] = labelencoder.fit_transform(building_meta["primary_use"])
##############

#### Merge des 3 bases

##############







df=pd.merge(train,building_meta, left_on='building_id', right_on='building_id')

df=pd.merge(df,weather_train, on=['timestamp','site_id'])

df = reduce_mem_usage(df)

##############

#### Conversion d'unité d'un certain type de meter à un certain endroit

##############



df.loc[(df["site_id"] == 0) & (df["meter"] == 0), "meter_reading"]=df.loc[(df["site_id"] == 0) & (df["meter"] == 0), "meter_reading"]*0.2931
df.head()
print(min(df["timestamp"]))

print(max(df["timestamp"]))
##############

#### Description de la base

##############





df.describe()
corr= df.corr()

mask = np.zeros_like(corr)

mask[np.triu_indices_from(mask)] = True

plt.figure(figsize = (10,10))

ax = sns.heatmap(corr.round(2), mask=mask,

            vmin=-1,     

            cmap='coolwarm',

            annot=True)
##############

#### Calcul des consommations moyennes de chaques meter type par semaine

##############



moyenne_conso_semaine = df.groupby(['semaine']).mean()

moyenne_conso_semaine_0 = df[df["meter"]==0].groupby(['semaine']).mean()

moyenne_conso_semaine_1 = df[df["meter"]==1].groupby(['semaine']).mean()

moyenne_conso_semaine_2 = df[df["meter"]==2].groupby(['semaine']).mean()

moyenne_conso_semaine_3 = df[df["meter"]==3].groupby(['semaine']).mean()
##############

#### Plot Standard

##############





plt.figure(figsize=(10,10))

plt.plot(range(0,len(moyenne_conso_semaine)),moyenne_conso_semaine["meter_reading"],label='moyenne')

plt.plot(range(0,len(moyenne_conso_semaine)),moyenne_conso_semaine_0["meter_reading"],label='electricity')

plt.plot(range(0,len(moyenne_conso_semaine)),moyenne_conso_semaine_1["meter_reading"],label='chilledwater')

plt.plot(range(0,len(moyenne_conso_semaine)),moyenne_conso_semaine_2["meter_reading"],label='steam')

plt.plot(range(0,len(moyenne_conso_semaine)),moyenne_conso_semaine_3["meter_reading"],label='hotwater')

plt.ylabel('Consomation en Kwh')

plt.xlabel('Semaine')

plt.legend()
##############

#### Plot Log de la valeur

##############







plt.figure(figsize=(10,10))

plt.plot(range(0,len(moyenne_conso_semaine)),np.log1p(moyenne_conso_semaine["meter_reading"]),label='moyenne')

plt.plot(range(0,len(moyenne_conso_semaine)),np.log1p(moyenne_conso_semaine_0["meter_reading"]),label='electricity')

plt.plot(range(0,len(moyenne_conso_semaine)),np.log1p(moyenne_conso_semaine_1["meter_reading"]),label='chilledwater')

plt.plot(range(0,len(moyenne_conso_semaine)),np.log1p(moyenne_conso_semaine_2["meter_reading"]),label='steam')

plt.plot(range(0,len(moyenne_conso_semaine)),np.log1p(moyenne_conso_semaine_3["meter_reading"]),label='hotwater')

plt.ylabel('log1p de la Consomation en Kwh')

plt.xlabel('Semaine')

plt.legend()





del moyenne_conso_semaine

del moyenne_conso_semaine_0

del moyenne_conso_semaine_1

del moyenne_conso_semaine_2

del moyenne_conso_semaine_3
##############

#### Calcul des consommations moyennes chaque site par semaine

##############



moyenne_conso_semaine = df.groupby(['semaine']).mean()



plt.figure(figsize=(10,10))

plt.plot(range(0,len(moyenne_conso_semaine)),moyenne_conso_semaine["meter_reading"],label='moyenne')



for i,k in enumerate (df["site_id"].unique()):

    moyenne_conso_site= df[df["site_id"]==i].groupby(['semaine']).mean()

    plt.plot(range(0,len(moyenne_conso_site)),moyenne_conso_site["meter_reading"],label=i)

    

plt.legend()





###### Un site semble se détaché sur reste, nous passons à plot des valeurs en log pour voir plus en détails



##############

#### Calcul des consommations moyennes de chaques site par semaine version log

##############



moyenne_conso_semaine = df.groupby(['semaine']).mean()



plt.figure(figsize=(10,10))

plt.plot(range(0,len(moyenne_conso_semaine)),np.log(moyenne_conso_semaine["meter_reading"]),label='moyenne')



for i,k in enumerate (df["site_id"].unique()):

    moyenne_conso_site= df[df["site_id"]==i].groupby(['semaine']).mean()

    plt.plot(range(0,len(moyenne_conso_site)),np.log1p(moyenne_conso_site["meter_reading"]),label=i)



    

plt.ylabel('log1p de la Consomation en Kwh')

plt.xlabel('Semaine')

plt.legend()





###### Le plot n'est pas beaucoup plus clair





del moyenne_conso_semaine

del moyenne_conso_site
##############

#### Calcul des consommations moyennes de chaques site par semaine version log

##############





plt.figure(figsize=(10,10))

plt.hist(building_meta["square_feet"],bins=30,color='coral')

plt.ylabel('Nombre de batiment')

plt.xlabel('Taille des batiments en pied carré')
##############

#### Pourcentage de chaque type de consomation

##############





meter_type=train["meter"].value_counts()

meter_type_perc=meter_type*100/sum(meter_type)

nom=['electricity','chilledwater','steam', 'hotwater']

y_position=[0,1,2,3]

plt.figure(figsize=(10,10))

A=plt.barh(y_position,meter_type_perc,

       color='#66b3ff', align="center")

plt.yticks(y_position,nom)



for k in range(len(meter_type)):

    plt.text(meter_type_perc[k]+0.5,

             y_position[k]-0.05, 

             str(round(meter_type_perc[k],2))+'%', 

             fontsize=15,

             color='dimgrey')

    

plt.xlim([0, max(meter_type_perc)+10])

# Enléve l'échelle de l'axe x

plt.tick_params(

    axis='x',          # changes apply to the x-axis

    which='both',      # both major and minor ticks are affected

    bottom=False,      # ticks along the bottom edge are off

    top=False,         # ticks along the top edge are off

    labelbottom=False) # labels along the bottom edge are off

plt.xlabel("Pourcentage des batiments")

plt.ylabel('Type de chauffage')

plt.show()





del meter_type

del meter_type_perc
##############

#### Pourcentage d'est utilisation des batiments

##############





primary_use=building_meta["primary_use"].value_counts().reset_index()

primary_use_perc=primary_use["primary_use"]*100/sum(primary_use["primary_use"])

nom=['Education','Office','Entertainment/public assembly' ,'Public services','Lodging/residential' ,'Other' ,'Healthcare','Parking','Warehouse/storage','Manufacturing/industrial','Retail','Services','Technology/science','Food sales and service','Utility','Religious worship']

y_position=range(0,len(primary_use))





plt.figure(figsize=(10,10))

A=plt.barh(y_position,primary_use_perc,

       color='#66b3ff', align="center")

plt.yticks(y_position,nom)



for k in range(len(primary_use_perc)):

    plt.text(primary_use_perc[k]+0.5,

             y_position[k]-0.15, 

             str(round(primary_use_perc[k],2))+'%', 

             fontsize=15,

             color='dimgrey')

    



plt.xlim([0, max(primary_use_perc)+10])

# Enléve l'échelle de l'axe x

plt.tick_params(

    axis='x',          # changes apply to the x-axis

    which='both',      # both major and minor ticks are affected

    bottom=False,      # ticks along the bottom edge are off

    top=False,         # ticks along the top edge are off

    labelbottom=False) # labels along the bottom edge are off

plt.xlabel("Pourcentage des batiments")

plt.ylabel('Type de chauffage')

plt.show()

##############

#### Représentation de "indicator of the primary category of activities of the 

#### building"

##############



fig, ax = plt.subplots(figsize=(10,8))

sns.countplot(y='primary_use', data=df)
##############

#### Représentation de "indicator of the primary category of activities of the 

#### building" par mois

##############



fig, ax = plt.subplots(figsize=(10,8))

sns.countplot(y='primary_use', data=df, hue='mois')
##############

#### Graphiques qui ne fonctionnent pas

##############



'''for bldg_id in [df['building_id']]:

    plt.figure(figsize=(16,5))

    tmp_df = df[df['building_id'] == bldg_id].copy()

    tmp_df.set_index(tmp_df["timestamp"], inplace=True)

    tmp_df.resample("D").meter_reading.sum().plot()

    plt.title(f"Meter readings for building #{bldg_id} ")

    plt.xlabel("Sum of readings")

    plt.tight_layout()

    plt.show()

    

    plt.figure(figsize=(16,5))

    tmp_df = df.set_index("timestamp", inplace=True)

    tmp_df.meter_reading.sum().plot()

    plt.title(f"Meter readings for building")

    plt.xlabel("Sum of readings")

    plt.tight_layout()

    plt.show()

    

    from pandas import Grouper



groups = df['meter_reading'].groupby(Grouper(freq='M'))

months = concat([(x[1].values) for x in groups], axis=1)

months = df(mois)

months.columns = range(1,13)

months.boxplot()

pyplot.show()



df_tmpY = df.groupby('annee').count()

plt.plot(df_tmpY['meter_reading'])



df_tmp = df.groupby('mois').count()

plt.plot(df_tmp['meter_reading'])



df.plot(df['air_temperature'], figsize=(12,4))



sns.set()

df.resample('meter_reading').mean().plot(style=':')



df.set_index("timestamp", inplace=True)

df.resample('D').meter_reading.sum().plot()

'''
##############

#### Représentation de toutes les variables selon le temps

##############



# !!!! Attention long car toutes les valeurs de timestamp s'affichent !!!



df.set_index('timestamp', inplace=True) #le nouvel index correspond à timestamp ce qui permet de réaliser des graphiques temporels

df.plot(figsize=(12,4))
df.hist(figsize = (13,13))

plt.show()
df_categorical = df.loc[:,['wind_speed']]

sns.countplot(x="variable", hue="value",data= pd.melt(df_categorical));
df_categorical = df.loc[:,['dew_temperature']]

g = sns.countplot(x="variable", hue="value",data= pd.melt(df_categorical));

g._legend.remove()
df_categorical = df.loc[:,['air_temperature']]

g = sns.countplot(x="variable", hue="value",data= pd.melt(df_categorical));

g._legend.remove()
#Représentation de l'évolution des température par mois; on préfère le second graphique

#sous forme de ligne que le premier qui est un histogramme

'''

df.groupby('mois')['air_temperature'].mean().plot(kind='bar')

plt.ylabel('air_temperature')

plt.show()

'''

time_grp = df.groupby('mois')['air_temperature'].mean().reset_index()

sns.lineplot(x='mois', y='air_temperature', data=time_grp)

#Représentation de la vitesse du vent sur un an par mois



'''df.groupby('mois')['wind_speed'].mean().plot(kind='bar')

plt.ylabel('wind_speed')

plt.show()'''



time_grp = df.groupby('mois')['wind_speed'].mean().reset_index()

sns.lineplot(x='mois', y='wind_speed', data=time_grp)
#représentation de la température de dew par mois sur un an



'''df.groupby('mois')['dew_temperature'].mean().plot(kind='bar')

plt.ylabel('dew_temperature')

plt.show()'''



time_grp = df.groupby('mois')['dew_temperature'].mean().reset_index()

sns.lineplot(x='mois', y='dew_temperature', data=time_grp)
#représentation des nuages par mois pendant une annnée



'''df.groupby('mois')['cloud_coverage'].mean().plot(kind='bar')

plt.ylabel('cloud_coverage')

plt.show()'''



time_grp = df.groupby('mois')['cloud_coverage'].mean().reset_index()

sns.lineplot(x='mois', y='cloud_coverage', data=time_grp)
#Représentation des précipitations moyennes par heure par mois pendant une an



'''df.groupby('mois')['precip_depth_1_hr'].mean().plot(kind='bar')

plt.ylabel('air_temperature')

plt.show()'''



time_grp = df.groupby('mois')['precip_depth_1_hr'].mean().reset_index()

sns.lineplot(x='mois', y='precip_depth_1_hr', data=time_grp)
'''df.groupby('mois')['meter'].mean().plot(kind='bar')

plt.ylabel('meter')

plt.show()'''



time_grp = df.groupby('mois')['meter'].mean().reset_index()

sns.lineplot(x='mois', y='meter', data=time_grp)
df.groupby('mois')['meter_reading'].mean().plot(kind='bar')

plt.ylabel('meter_reading')

plt.show()
time_grp = df.groupby('mois')['meter_reading'].mean().reset_index()

sns.lineplot(x='mois', y='meter_reading', data=time_grp)
##############

#### Building par site

##############



df_build_site = df.site_id.value_counts()

fig, axes = plt.subplots(1,1,figsize=(10, 5), dpi=100)

sns.barplot(x=df_build_site.index,

           y=df_build_site.values,

           palette = 'Blues_d',

           saturation=.5)

axes.set_title('Nombre de buildings par site')

del df_build_site
##############

#### Suppression des bases initiales

##############



del train

del weather_train

del building_meta
##############

#### Création de la variable cible et des variables explicatioves

#### Split en App/Test

##############





X=df.drop(columns=["meter_reading"])

y=np.log1p(df["meter_reading"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)





del X

del y

del df



gc.collect()
##############

#### Création de forets dans le but de réduire la demande mémoire

##############





from sklearn.ensemble import forest

def set_rf_samples(n):

    forest._generate_sample_indices = (lambda rs, n_samples:

        forest.check_random_state(rs).randint(0, n_samples, n))

set_rf_samples(130000)





def reset_rf_samples():

    """ Undoes the changes produced by set_rf_samples.

    """

    forest._generate_sample_indices = (lambda rs, n_samples:

        forest.check_random_state(rs).randint(0, n_samples, n_samples))
model = RandomForestRegressor(n_estimators=60,

                              random_state=0,

                              n_jobs=-1)

model.fit(X_train,y_train)
y_pred=model.predict(X_test)

print('The rmse of prediction is:', mean_squared_log_error(y_test, y_pred) ** 0.5)
##############

#### Importance des variables dans la foret d'arbre

##############



feature_names = X_train.columns

importances = model.feature_importances_

std = np.std([tree.feature_importances_ for tree in model.estimators_],

             axis=0)

indices = np.argsort(importances)[::-1]



plt.figure(figsize=(10,10))

plt.title("Feature importances")

plt.bar(range(X_train.shape[1]), importances[indices],

       color="r", align="center")

plt.xticks(range(X_train.shape[1]), feature_names, rotation='vertical')

plt.xlim([-1, X_train.shape[1]])

plt.show()
##############

#### Kfold pour la réglage des parametres de la random forest

#### Attention beaucoup de calcul

##############



from sklearn.model_selection import KFold



kf = KFold(n_splits= 2,shuffle=True,random_state=42)



n_estimators = [100, 150]



moyenne_erreur_K_fold = []

ecart_type_erreur_K_fold =[]



for i in range(0,len(n_estimators)):

    erreur_fold = []

    for train_index, test_index in kf.split(X_train):

        X_train_K_fold, X_test_K_fold = X_train.iloc[train_index], X_train.iloc[test_index]

        y_train_K_fold, y_test_K_fold = y_train.iloc[train_index], y_train.iloc[test_index]

        model_K_fold = RandomForestRegressor(n_estimators=n_estimators[i],

                                             random_state=0,

                                             n_jobs=-1,

                                             )

        model_K_fold.fit(X_train_K_fold,y_train_K_fold)

        y_pred_K_fold = model_K_fold.predict(X_test_K_fold)

        erreur = mean_squared_error(y_test_K_fold, y_pred_K_fold) ** 0.5

        erreur_fold.append(erreur)

    moyenne_erreur_K_fold.append(np.mean(erreur))

    ecart_type_erreur_K_fold.append(np.std(erreur_fold))



del X_train_K_fold

del X_test_K_fold

del y_train_K_fold

del y_test_K_fold



gc.collect()

print(moyenne_erreur_K_fold)

print(ecart_type_erreur_K_fold)
##############

#### Reset des forêts

##############



reset_rf_samples()

from sklearn.neighbors import KNeighborsRegressor
X_train_knn, X_test_knn, y_train_knn, y_test_knn = train_test_split(X_train, y_train, test_size=0.33, random_state=42)

k_range = [5, 10, 15, 20]

erreur_knn = []

for i in range(0,len(k_range)):

    knn = KNeighborsRegressor(n_neighbors=k_range[i],n_jobs=-1, weights='distance')

    knn.fit(X_train_knn,y_train_knn)

    y_pred = knn.predict(X_test_knn)

    erreur = mean_squared_error(y_test_knn, y_pred)**0.5

    erreur_knn.append(erreur)



del X_train_knn

del X_test_knn

del y_test_knn

del y_train_knn

del y_pred



gc.collect()
plt.plot( k_range,erreur_knn , marker='o', markerfacecolor='blue', markersize=12, color='skyblue', linewidth=4)
##############

#### Création des DataSets de LightGBM & des paramètres de ses datasets

##############



categorical_features = ["building_id", "site_id", "meter", "primary_use"]



lgb_train = lgb.Dataset(X_train, 

                        label=y_train,

                        categorical_feature=categorical_features,

                        free_raw_data=False)



lgb_eval = lgb.Dataset(X_test, 

                       label=y_test,

                       categorical_feature=categorical_features,

                       free_raw_data=False)

##############

#### Paramètre du gradient boosting

##############



params_lgb = {

    "objective": "regression",

    "boosting": "gbdt",

    "num_leaves": 1000,

    "learning_rate": 0.05,

    "feature_fraction": 0.85,

    "reg_lambda": 2,

    "metric": "rmse",

}
##############

#### Entrainement du modele

##############



model_lgb = lgb.train(params_lgb, 

                train_set=lgb_train, 

                num_boost_round=100, 

                valid_sets=[lgb_train,lgb_eval], 

                verbose_eval=25, 

                early_stopping_rounds=50)