# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data_train=pd.read_csv('/kaggle/input/cs-challenge/training_set.csv')

data_test=pd.read_csv('/kaggle/input/cs-challenge/test_set.csv')

df_train=data_train.copy()

df_test=data_test.copy()
def fill_nan(df) :

    '''Remplacement des données manquantes. On suppose la continuité des grandeurs sur une même éolienne

    car phénomène physique'''

    

    df_clean=pd.DataFrame()

    for mac_code in ['WT1', 'WT2', 'WT3', 'WT4'] :

        wt=df[df["MAC_CODE"]==mac_code].sort_values("Date_time").fillna(method="ffill")

        df_clean=pd.concat([df_clean,wt]).sort_values("ID")

    

    return df_clean



df_clean=fill_nan(df_train)
def variance_threshold(df, threshold=2.0) :

    '''Sélectionne les colonnes de df dont la variance >= threshold. Elimine les colonnes constantes. '''

    return np.array(df.columns)[df.std()>=threshold]

    

def speed_transform(df) :

    '''On élève les vitesses aux cubes car puissance récupérable dans éolienne proportionnelle à la vitesse turbine au cube.'''

    speed_columns=["Rotor_speed", "Rotor_speed_min", "Rotor_speed_max", "Generator_converter_speed","Generator_converter_speed_min","Generator_converter_speed_max", "Generator_speed", "Generator_speed_min", "Generator_speed_max" ]

    for column in speed_columns :

        df[column+"^3"] = (np.array(df[column]))**3

        df=df.drop(column, axis=1)

    return df



def selectKBest(df,k=10) :

    '''Etude de corrélation avec la colonne target et sélection des k meilleures colonnes'''

    corr_target={np.abs(df[column].corr(df["TARGET"])) : str(column) for column in df.columns}

    corr=list(corr_target.keys())

    corr.sort(reverse=True)

    

    selected_features=[corr_target[element] for element in corr[:k]]

    

    return selected_features

 

def remove_duplicates(df, selected_features) :

    '''Suppression des colonnes fortement corrélées entre elles. Elles font doublons.'''

    

    for k in range(2,3) :

        selected=selected_features[k]

        corr_selected={column : np.abs(df[selected].corr(df[column])) for column in df.columns}

        duplicates=[column for column in df.columns if corr_selected[column] >=0.95]

        

        if "TARGET" in duplicates :

            duplicates.remove("TARGET")

            

        duplicates.remove(selected)

        df=df.drop(duplicates, axis=1)

    

    return list(df.columns)

    

def features_selection(df) :

    

    df=df.drop(["ID", "MAC_CODE"], axis=1) #Suppression des colonnes inutiles

    selected_features=variance_threshold(df) #Suppression des données constantes

    df=df[selected_features]                

    df=speed_transform(df)               #Elévation des vitesses au cube

    selected_features=selectKBest(df)

    df=df[selected_features]

    selected_features=remove_duplicates(df, selected_features)

    print("Selected features : ", selected_features)

    

    return selected_features, df[selected_features]

    

    

selected_features, df_clean = features_selection(df_clean)
from sklearn.model_selection import train_test_split

#Dernier pré-traitement des données : la normalisation et séparation du dataset d'entraînement en train et test



def normalize(df) :

    ''' Normalisation des données pour qu'une colonne ne soit pas prépondérante sur une autre.'''

    

    for column in df.columns :

        df[column]=(df[column]-df[column].mean())/df[column].std()

    

    return df



Y=df_clean["TARGET"]

X=normalize(df_clean.drop("TARGET", axis=1))



X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.20, random_state=40)
from sklearn.linear_model import Ridge, SGDRegressor

from sklearn.neural_network import MLPRegressor

from sklearn.ensemble import RandomForestRegressor

from  sklearn.metrics import mean_absolute_error



def ridge_regression(X_train, X_test, Y_train, Y_test) :

    reg=Ridge(alpha=30000)

    reg.fit(X_train, Y_train)

    

    #Evaluation

    Y_pred_train=reg.predict(X_train)

    Y_pred_test=reg.predict(X_test)

    

    print("MAE train : ", mean_absolute_error(Y_pred_train, Y_train))

    print("MAE test : ", mean_absolute_error(Y_pred_test, Y_test))

    

    return reg 



def SGD_regression(X_train, X_test, Y_train, Y_test) :

    reg=SGDRegressor(alpha=0.26)

    reg.fit(X_train, Y_train)

    

    #Evaluation

    Y_pred_train=reg.predict(X_train)

    Y_pred_test=reg.predict(X_test)

    

    print("MAE train : ", mean_absolute_error(Y_pred_train, Y_train))

    print("MAE test : ", mean_absolute_error(Y_pred_test, Y_test))

    

    return reg



def random_forest_regression(X_train, X_test, Y_train, Y_test) :

    reg=RandomForestRegressor(n_estimators=50)

    reg.fit(X_train, Y_train)

    

    #Evaluation

    Y_pred_train=reg.predict(X_train)

    Y_pred_test=reg.predict(X_test)

    

    print("MAE train : ", mean_absolute_error(Y_pred_train, Y_train))

    print("MAE test : ", mean_absolute_error(Y_pred_test, Y_test))

    

    return reg



def MLP_regression(X_train, X_test, Y_train, Y_test) :

    reg=MLPRegressor(hidden_layer_sizes=(10,5,5), max_iter=200, random_state=5)

    reg.fit(X_train, Y_train)

    

    #Evaluation

    Y_pred_train=reg.predict(X_train)

    Y_pred_test=reg.predict(X_test)

    

    print("MAE train : ", mean_absolute_error(Y_pred_train, Y_train))

    print("MAE test : ", mean_absolute_error(Y_pred_test, Y_test))

    

    return reg



reg=random_forest_regression(X_train, X_test, Y_train, Y_test)
def preprocessing_validation(df, selected_features) :

    selected_features.remove("TARGET")

    df=fill_nan(df)

    df=speed_transform(df)

    ids_val=df["ID"]

    X_realworld=normalize(df[selected_features])

    

    return ids_val, X_realworld, df



def prediction_to_csv(filename,ids, model, X_real) :

    results=pd.DataFrame()

    results["ID"]=ids

    results["TARGET"]=model.predict(X_real)

    

    results.to_csv(str(filename)+".csv", index=False)

    

ids_val, X_realworld, df_new=preprocessing_validation(df_test, selected_features)

prediction_to_csv("results_RF9", ids_val, reg, X_realworld)