import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import statsmodels.formula.api as smf

from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix, auc

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

from sklearn.preprocessing import OneHotEncoder

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.ensemble import BaggingClassifier



from sklearn.tree import DecisionTreeClassifier

from xgboost import XGBClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.compose import ColumnTransformer

conjunto = pd.read_csv("../input/hotel-booking-demand/hotel_bookings.csv")

conjunto.head()



def model(algo):

    algo_model = algo.fit(X_train, y_train)

    global y_prob, y_pred

    y_prob = algo.predict_proba(X_test)[:,1]

    y_pred = algo_model.predict(X_test)



    print('Accuracy Score: {}\n\nConfusion Matrix:\n {}'

      .format(accuracy_score(y_test,y_pred), confusion_matrix(y_test,y_pred),roc_auc_score(y_test,y_pred)))



# Press the green button in the gutter to run the script.

if __name__ == '__main__':

    trazar = False

    try:

        if trazar:

            print(conjunto.head())

    except Exception:

        print("Ha ocurrido una excepcion 1")





    #información de la tabla

    try:

        if trazar:

            print(conjunto.info())

    except Exception:

        print("Ha ocurrido una excepcion 2")



    #Comprobamos el ratio de reservas canceladas

    try:

        if trazar:

            print("Reservas canceladas")

            print(conjunto["is_canceled"].value_counts())

    except Exception:

        print("Ha ocurrido una excepcion 3")



    #sacamos una lista con las correlaciones de las distintas columnas respecto al resultado de la reserva

    try:

        correlaciones = conjunto.corr()

        if trazar:

            print(correlaciones["is_canceled"].sort_values(ascending=False))

    except Exception:

        print("Ha ocurrido una excepcion 4")



    try:

        ##Limpieza de nulls (imprescindible)

        nulls = conjunto.isnull().sum()

        nulls[nulls > 0]



        conjunto.iloc[:, 23].fillna(conjunto.iloc[:, 23].mean(), inplace=True)

        conjunto.iloc[:, 10].fillna(conjunto.iloc[:, 10].mean(), inplace=True)



        nulls = conjunto.isnull().sum()

        nulls[nulls > 0]



    except Exception:

        print("Exceptions 5")

    #Investigamos por país a ver que sale

    try:

        conjunto = conjunto.drop(['stays_in_weekend_nights', 'arrival_date_day_of_month', 'children',

                          'arrival_date_week_number', 'company','reservation_status_date'], axis=1)

        if trazar:

            print("Insertar comentario")

            print(conjunto)

            print("Valores por país");

            print(conjunto["country"].value_counts())



        if trazar:

            print("Forma del conjunto antes de deshacernos de valores sin país: ", conjunto.shape)

        conjunto = conjunto[conjunto['country'].notna()]

        if trazar:

            print("Forma del conjunto después de deshacernos de valores sin país : ", conjunto.shape)



        if trazar:

            print("Recuento de valores por país")

            print(conjunto["country"].value_counts())



        print("-----------------------------------------------------")



        conjunto = conjunto.drop(['country'], axis=1)

        print(conjunto)



        print("++++++++++++++++++++++++++++++++++++++++++++++++++++")



        conjunto = conjunto.drop(['reservation_status'], axis=1)



        #print(conjunto.columns)

        #conjunto = conjunto.drop(['customer_type'], axis=1)

        #conjunto = conjunto.drop(['lead_time'],axis=1)

        ####conjunto = conjunto.drop(['days_in_waiting_list'],axis=1)

        #conjunto = conjunto.drop(['deposit_type'],axis=1)

        #conjunto = conjunto.drop(['arrival_date_week_number'],axis=1)

        #conjunto = conjunto.drop(['stays_in_week_nights'],axis=1)

        #conjunto = conjunto.drop(['babies'],axis='columns') #babies

        ####conjunto = conjunto.drop(['agent'],axis='columns')

        ####conjunto = conjunto.drop(['is_repeated_guest'],axis=1)

        #conjunto = conjunto.drop(['meal'],axis=1)

        #conjunto = conjunto.drop(['required_car_parking_spaces'],axis=1)



        #conjunto2 = conjunto.extract(conjunto['hotel'] == 'Resort Hotel')



    except Exception as exx:

        print(exx.__str__())

        print("Ha ocurrido una excepción 6")



    #Partiendo el conjunto

    try:

        X = (conjunto.loc[:, conjunto.columns != 'is_canceled'])

        y = (conjunto.loc[:, conjunto.columns == 'is_canceled'])



        x_columns = X.columns



        object_column_name = X.select_dtypes('object').columns

        if trazar:

           print(object_column_name)



        object_column_index = X.columns.get_indexer(X.select_dtypes('object').columns)

        if trazar:

            print(object_column_index)

        if trazar:

            print(X.shape)



        columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), object_column_index)], remainder='passthrough')



        X = columnTransformer.fit_transform(X)

        if trazar:

            print(X.shape)



        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=25)

    except Exception:

        print("Ha ocurrido una excepción 7")





    try:

        print('Random Forest\n')

        model(RandomForestClassifier(n_estimators=220,n_jobs=6,max_features=0.25))

        #model(RandomForestClassifier(n_jobs=6,max_features=0.25))



        #oob_score no sirve para mucho en este caso, es mejor usar max_features

        #conjunto = np.append(conjunto,np.asmatrix(y_pred),axis=1)



    except Exception as exx:

        print(exx.__str__())