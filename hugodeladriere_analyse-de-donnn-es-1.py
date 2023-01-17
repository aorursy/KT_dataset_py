# Main packages
import numpy as np
import pandas as pd

#package sckit learn
#from sklearn.model_selection import train_test_split, RandomizedSearchCV
#from sklearn.linear_model import LogisticRegression

#from sklearn.metrics import mean_squared_error

#algo
#from xgboost import XGBClassifier

#Graph
from itertools import cycle
import matplotlib.pyplot as plt

#Memory usage 
import gc

color_cycle = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
# loadin data

DIR='/kaggle/input/m5-forecasting-accuracy'
cal_df = pd.read_csv(f'{DIR}/calendar.csv')
sale_df = pd.read_csv(f'{DIR}/sales_train_validation.csv')
price_df = pd.read_csv(f'{DIR}/sell_prices.csv')
submission_df = pd.read_csv(f'{DIR}/sample_submission.csv')
#Pour faire simple, tableau qui rassemble les informations sur les évenments et leur date dans les magasins et durant l'année. 
cal_df.sample(2)
#Tableau rassemblant les ventes de tous les items en fonction des jours de ventes et des magasins 
sale_df.sample(2)

x=1913/365
print("l'étude de vente dans les magasins s'étend sur 1913 jours soit {'x'} ans",x)
#Le tableau rassemble les inforamtions sur les prix des différents items 
price_df.sample(2)
submission_df.head()
cal_df.info()
##Ajustement de tableau Calendar.


#Pour réduire la consommation de RAM 
#Reduction de int64 a int8 ou int16 lorsque c'est nécessaire sur 

cal_df[["month", "snap_CA", "snap_TX", "snap_WI", "wday"]] = cal_df[["month", "snap_CA", "snap_TX", "snap_WI", "wday"]].astype("int8")
cal_df[["wm_yr_wk", "year"]] = cal_df[["wm_yr_wk", "year"]].astype("int16") 
cal_df["date"] = cal_df["date"].astype("datetime64")
#Suppréssion des valeurs NA
nan_features = ['event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']
for feature in nan_features:
    cal_df[feature].fillna('unknown', inplace = True)

cal_df[["weekday", "event_name_1", "event_type_1", "event_name_2", "event_type_2"]] = cal_df[["weekday", "event_name_1", "event_type_1", "event_name_2", "event_type_2"]] .astype("category")
#int16 sur le tableau sale
sale_df.loc[:, "d_1":] = sale_df.loc[:, "d_1":].astype("int16")


#Réduction de la mémoire dans le tableau price 
price_df['sell_price']=price_df['sell_price'].astype("float16")
price_df['wm_yr_wk']=price_df['wm_yr_wk'].astype("int16")

sale_df.head(2)
d_cols = [c for c in sale_df.columns if 'd_' in c] # sales data colonne

# Différente étape 
# 1. Selectionner un item
# 2. Mettre l'id en index
# 3. Transformaer en colonne
# 4. Plot les données
sale_df.loc[sale_df['id'] == 'FOODS_1_001_CA_1_validation'] \
    .set_index('id')[d_cols] \
    .T \
    .plot(figsize=(15, 5),
          title='FOODS_1_001_CA_1 sales by "d" number')
plt.legend('')
plt.show()
sale_df
exemple = sale_df.loc[sale_df['cat_id'] == 'HOBBIES']
exemple

#Charge un Tableau avec les ventes de l'item choisis

exemple = sale_df.loc[sale_df['id'] == 'FOODS_1_001_CA_1_validation'][d_cols].T 
exemple = exemple.rename(columns={1612:'FOODS_3_090_CA_3'}) # Renomer correctement la colonne créer
exemple = exemple.reset_index().rename(columns={'index': 'd'}) # mettre 'd' en index
exemple = exemple.merge(cal_df, how='left', validate='1:1') #Merge le tableau sale et cal avec les indexs


exemple2 = sale_df.loc[sale_df['id'] == 'HOBBIES_1_001_CA_1_validation'][d_cols].T 
exemple2 = exemple2.rename(columns={0:'HOBBIES_1_001_CA_1'}) # Renomer correctement la colonne créer
exemple2 = exemple2.reset_index().rename(columns={'index': 'd'}) # mettre 'd' en index
exemple2 = exemple2.merge(cal_df, how='left', validate='1:1') #Merge le tableau sale et cal avec les indexs

exemple3 = sale_df.loc[sale_df['id'] == 'HOUSEHOLD_1_002_CA_1_validation'][d_cols].T 
exemple3 = exemple3.rename(columns={566:'HOUSEHOLD_1_002_CA_1'}) # Renomer correctement la colonne créer
exemple3 = exemple3.reset_index().rename(columns={'index': 'd'}) # mettre 'd' en index
exemple3 = exemple3.merge(cal_df, how='left', validate='1:1') #Merge le tableau sale et cal avec les indexs


exemple.set_index('date')['FOODS_3_090_CA_3'] \
    .plot(figsize=(15, 5),
          title='FOODS_3_090_CA_3 vente en focntion des dates')
plt.show()

exemples = ['FOODS_3_090_CA_3','HOBBIES_1_001_CA_1','HOUSEHOLD_1_002_CA_1']
exemple_df = [exemple, exemple2, exemple3]
for i in [0, 1, 2]:
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 3))
    exemple_df[i].groupby('wday').mean()[exemples[i]]\
        .plot(kind='line',
              title='average sale: day of week',
              lw=5,
              color='red',
              ax=ax1)
    exemple_df[i].groupby('month').mean()[exemples[i]] \
        .plot(kind='line',
              title='average sale: month',
              lw=5,
              color='blue',
              ax=ax2)
    exemple_df[i].groupby('year').mean()[exemples[i]] \
        .plot(kind='line',
              lw=5,
              title='average sale: year',
              color='orange',
              ax=ax3)
    fig.suptitle(f'Tendance pour l item: {exemples[i]}',
                 size=20,
                 y=1.1)
    plt.tight_layout()
    plt.show()
twenty_examples = sale_df.sample(20, random_state=529) \
        .set_index('id')[d_cols] \
    .T \
    .merge(cal_df.set_index('d')['date'],
           left_index=True,
           right_index=True,
            validate='1:1') \
    .set_index('date')
fig, axs = plt.subplots(10, 2, figsize=(15, 20))
axs = axs.flatten()
ax_idx = 0
for item in twenty_examples.columns:
    twenty_examples[item].plot(title=item,
                              color=next(color_cycle),
                              ax=axs[ax_idx])
    ax_idx += 1
plt.tight_layout()
plt.show()
#Ajout de L'id complet dans le tableau price
price_df.loc[:, "id"] = price_df.loc[:, "item_id"] + "_" + price_df.loc[:, "store_id"] + "_validation"

def make_dataframe():
    df = pd.merge(price_df, sale_df, on="id")
    return df
df1=make_dataframe()
df1.head()
cal_df.head(5)
#creation d'un tableau des ventes d'un item en fonction du jour de vente
df_id= sale_df.drop(columns=["item_id", "dept_id", "cat_id", "state_id","store_id", "id"]).T
df_id.index = cal_df["date"][:1913]
df_id.columns = sale_df["id"]

df_id.head()
df_long = df_id.stack().reset_index(1)#.Sack permet de les mettres en colonnes // et .reset_index(1) permet de de mettre la colonne date dans une colonne variable (plus en index ! tu as compris ????)

df_long.columns = ["id", "value"]#rename des colonnes
df_long
cal_df.head(2)
df = pd.merge(df_long.reset_index(), cal_df, on='date')

df= pd.merge(df, price_df, on=["id", "wm_yr_wk"])
df
df.info()
price_df.head(2)
price_df = pd.concat([price_df, price_df["item_id"].str.split("_", expand=True)], axis=1)#création des variables 'cat_id' et 'dept_id'
price_df = price_df.rename(columns={0:"cat_id", 1:"dept_id"})#Rename
price_df[["store_id", "item_id", "cat_id", "dept_id"]] = price_df[["store_id","item_id", "cat_id", "dept_id"]].astype("category")#Ajustement Memory Usage
price_df = price_df.drop(columns=2)#Drop colonne inutile
price_df.head(2)
def make_dataframe():

    df_id = sale_df.drop(columns=["item_id", "dept_id", "cat_id", "state_id","store_id", "id"]).T
    df_id.index = cal_df["date"][:1913]
    df_id.columns = sale_df["id"]
    



    df_long = df_id.stack().reset_index(1)
    df_long.columns = ["id", "value"]

    del df_id
    gc.collect()
    
    df = pd.merge(pd.merge(df_long.reset_index(), cal_df, on="date"), price_df, on=["id", "wm_yr_wk"])
    df = df.drop(columns=["d"])
#     df[["cat_id", "store_id", "item_id", "id", "dept_id"]] = df[["cat_id"", store_id", "item_id", "id", "dept_id"]].astype("category")


    del df_long
    gc.collect()

    return df

df = make_dataframe()
df.head()
df.info()
df
