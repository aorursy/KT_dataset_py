import numpy as np
import os
%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt

import pandas as pd
from pandas import read_csv
filename = '../input/all_stocks_5yr.csv'
stock = read_csv(filename)
print("Die Tabelle zeigt den Aufbau unserer Datei")
stock.head() 
ticker_name = 'AMZN'
stock_a = stock[stock['Name'] == ticker_name]
stock_a.shape #(Anzahl der Zeilen, Anzahl der Spalten)
stock.info() 
stock_a.describe()
stock_a['Tagesveraenderung'] = ((stock['high'] - stock['low'] )/ stock['low'])*100

stock_a['VeraenderungZumVorherigenTag'] = (abs(stock_a['close'].shift() - stock_a['close'] )/ stock['close'])*100
print
stock_a.head()
stock_a.hist(bins=50, figsize=(20,15))
plt.show()
stock_a.plot(kind="line", x="date", y="close", figsize=(15, 10))
corr_matrix = stock_a.corr()
corr_matrix["close"].sort_values(ascending=False)
from pandas.plotting import scatter_matrix

attributes = ["high", "low", "open", "Tagesveraenderung", "VeraenderungZumVorherigenTag", "volume"]

scatter_matrix(stock_a[attributes], figsize=(20, 15))
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline
corr = stock_a[["high", "low", "open", "Tagesveraenderung", "VeraenderungZumVorherigenTag", "volume"]].corr()

# Generierung einer Maske
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# set up the matplotlib figure
f, ax = plt.subplots(figsize=(18, 12))

# Auswahl der Farbpalette
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Heatmap mit Maske darstellen und Achsenlänge bestimmen
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3,
            square=True, 
            linewidths=.5, cbar_kws={"shrink": .5}, ax=ax);
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,Normalizer
X_stock_a = stock_a.drop(['date', 'Name','close'], axis=1)
y_stock_a = stock_a['close']

X_stock_train, X_stock_test, y_stock_train, y_stock_test = train_test_split(X_stock_a, y_stock_a, test_size=0.2, 
                                                                            random_state=42)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer,StandardScaler
data_pipeline = Pipeline([
        ('imputer', Imputer(missing_values="NaN",strategy="median")), #fehlenden werte werden durch Median ersetzt
        ('scaler',StandardScaler())
#        ('normalizer', Normalizer()),
    ])
from sklearn.preprocessing import Imputer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler,Normalizer

from sklearn.pipeline import Pipeline

Lr_pipeline_nor = Pipeline([
        ('imputer', Imputer(missing_values="NaN",strategy="median")), #fehlenden werte werden durch Median ersetzt
        ('normalizer',Normalizer()),
        ('lr', LinearRegression())
        
    ])

Lr_pipeline_nor.fit(X_stock_train, y_stock_train)
from sklearn.svm import SVR
from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline

svr_pipeline_nor = Pipeline([
        ('imputer', Imputer(missing_values="NaN",strategy="median")), #fehlenden werte werden durch Median ersetzt
        ('normalizer',Normalizer()),
        ('svr', SVR(kernel="linear"))
        
    ])

svr_pipeline_nor.fit(X_stock_train, y_stock_train)
from sklearn.preprocessing import Imputer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline



Lr_pipeline_std = Pipeline([
        ('imputer', Imputer(missing_values="NaN",strategy="median")), #fehlenden werte werden durch Median ersetzt
        ('scaler',StandardScaler()),
        ('lr', LinearRegression())
        
    ])

Lr_pipeline_std.fit(X_stock_train, y_stock_train)
from sklearn.svm import SVR
from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline

svr_pipeline_std = Pipeline([
        ('imputer', Imputer(missing_values="NaN",strategy="median")), #fehlenden werte werden durch Median ersetzt
        ('scaler',StandardScaler()),
        ('svr', SVR(kernel="linear"))
        
    ])

svr_pipeline_std.fit(X_stock_train, y_stock_train)
from sklearn.metrics import mean_absolute_error

#Lineare Regression mit Normalisierung und Standardisierung 
lr_stock_predictions_nor = Lr_pipeline_nor.predict(X_stock_test)
lr_mae_nor = mean_absolute_error(y_stock_test, lr_stock_predictions_nor)
print('Lr MAE with Normalization', lr_mae_nor)

lr_stock_predictions_std = Lr_pipeline_std.predict(X_stock_test)
lr_mae_std = mean_absolute_error(y_stock_test, lr_stock_predictions_std)
print('Lr MAE with standardization', lr_mae_std)

#SVM mit Normalisierung und Standardisierung
svm_stock_predictions_nor = svr_pipeline_nor.predict(X_stock_test)
svm_mae_nor = mean_absolute_error(y_stock_test, svm_stock_predictions_nor)
print('SVM MAE with Normalization', svm_mae_nor)

svm_stock_predictions_std = svr_pipeline_std.predict(X_stock_test)
svm_mae_std = mean_absolute_error(y_stock_test, svm_stock_predictions_std)
print('SVM MAE with standardization', svm_mae_std)
import pandas as pd
import numpy as np

#Vorhersage und Ausgabe der RMSE
from sklearn.metrics import mean_squared_error

from sklearn.metrics import mean_squared_error



#Linear Regression mit Normalisierung und Standardisierung
lr_stock_predictions_nor = Lr_pipeline_nor.predict(X_stock_test)
lr_mse_nor = mean_squared_error(y_stock_test, lr_stock_predictions_nor)
lr_rmse_nor = np.sqrt(lr_mse_nor)
print('Lr RMSE mit Normalisierung', lr_rmse_nor)

lr_stock_predictions_std = Lr_pipeline_std.predict(X_stock_test)
lr_mse_std = mean_squared_error(y_stock_test, lr_stock_predictions_std)
lr_rmse_std = np.sqrt(lr_mse_std)
print('Lr RMSE mit Standardisierung', lr_rmse_std)

#SVM mit Normalisierung und Standardisierung
svm_stock_predictions_nor = svr_pipeline_nor.predict(X_stock_test)
svm_mse_nor = mean_squared_error(y_stock_test, svm_stock_predictions_nor)
svm_rmse_nor = np.sqrt(svm_mse_nor)
print('SVM RMSE mit Normalisierung', svm_rmse_nor)

svm_stock_predictions_std = svr_pipeline_std.predict(X_stock_test)
svm_mse_std = mean_squared_error(y_stock_test, svm_stock_predictions_std)
svm_rmse_std = np.sqrt(svm_mse_std)
print('SVM RMSE mit Standardisierung', svm_rmse_std)



lr_std = ['1',"Linear Regression mit Standardisierung",np.round(lr_rmse_std,3),np.round(lr_mae_std,3)]
lr_nor = ['2',"Linear Regression mit Normalisierung",np.round(lr_rmse_nor,3),np.round(lr_mae_nor,3)]

svm_std = ['5',"SVM mit Standardisierung",np.round(svm_rmse_std,3),np.round(svm_mae_std,3)]
svm_nor = ['6',"SVM mit Normalisierung",np.round(svm_rmse_nor,3),np.round(svm_mae_nor,3)]



linear_model_result= pd.DataFrame([lr_std,lr_nor,svm_std,svm_nor],columns=[ "ExpID", "Model", "RMSE","MAE"])

linear_model_result
#Funktion zur Ausgabe sämtlicher Modelle
from sklearn.preprocessing import Imputer
    
def allModelsResultForAllStocks():
    
    best_result_per_ticker = pd.DataFrame(columns=['Ticker','Model','RMSE'])
    ticker_list = np.unique(stock["Name"])
    best_result_per_ticker = list()
    for ticker_name in ticker_list:
        result = pd.DataFrame(columns=['Ticker','Model','RMSE'])
        stock_a = stock[stock['Name'] == ticker_name]
        #Weitere Feautures werden hinzugefügt 
        # 1 Preisveränderungen über den Tag
        stock_a['Tagesveraenderung'] = ((stock['high'] - stock['low'] )/ stock['low'])*100

        #2 Preisveränderungen zum voherigen Tag
        stock_a['VeraenderungZumVorherigenTag'] = (abs(stock_a['close'].shift() - stock_a['close'] )/ stock['close'])*100

        X_stock_a = stock_a.drop(['date', 'Name','close'], axis=1)
        y_stock_a = stock_a['close']

        
        imputer = Imputer(missing_values='NaN', strategy='median') #Fehlende Werte werden durch den Median ersetz
        
        imputer.fit_transform(X_stock_a)
       
        X_stock_train, X_stock_test, y_stock_train, y_stock_test = train_test_split(X_stock_a, y_stock_a, test_size=0.2, 
                                                                                random_state=42)


        Lr_pipeline_std.fit(X_stock_train, y_stock_train)
        Lr_pipeline_nor.fit(X_stock_train, y_stock_train)

        svr_pipeline_nor.fit(X_stock_train, y_stock_train)
        svr_pipeline_std.fit(X_stock_train, y_stock_train)

        
        # Vorhersage und Berechnung des RSME für alle Modelle

        #Linear Regression mit Normalisierung und Standartisierung
        lr_stock_predictions_nor = Lr_pipeline_nor.predict(X_stock_test)
        lr_mse_nor = mean_squared_error(y_stock_test, lr_stock_predictions_nor)
        lr_rmse_nor = np.sqrt(lr_mse_nor)
        rmse_row =   [ticker_name,'Lr RMSE mit Normalisierung', lr_rmse_nor]

        result.loc[-1] = rmse_row  # Hinzufügen einer Reihe zur Ausgabe
        result.index = result.index + 1  # Iterationsstufe +1
     
    
        lr_stock_predictions_std = Lr_pipeline_std.predict(X_stock_test)
        lr_mse_std = mean_squared_error(y_stock_test, lr_stock_predictions_std)
        lr_rmse_std = np.sqrt(lr_mse_std)
        rmse_row =   [ticker_name,'Lr RMSE mit Standardisierung', lr_rmse_std]
    
    

        result.loc[-1] = rmse_row  # Hinzufügen einer Reihe zur Ausgabe
        result.index = result.index + 1  # Iterationsstufe +1

        #SVM mit Normalisierung und Standartisierung
        svm_stock_predictions_nor = svr_pipeline_nor.predict(X_stock_test)
        svm_mse_nor = mean_squared_error(y_stock_test, svm_stock_predictions_nor)
        svm_rmse_nor = np.sqrt(svm_mse_nor)
        rmse_row =   [ticker_name,'SVM RMSE mit Normalisierung', svm_rmse_nor]
        

        result.loc[-1] = rmse_row  # Hinzufügen einer Reihe zur Ausgabe
        result.index = result.index + 1  # Iterationsstufe +1

        svm_stock_predictions_std = svr_pipeline_std.predict(X_stock_test)
        svm_mse_std = mean_squared_error(y_stock_test, svm_stock_predictions_std)
        svm_rmse_std = np.sqrt(svm_mse_std)
        rmse_row =   [ticker_name,'SVM RMSE mit Standardisierung', svm_rmse_std]
    
        result.loc[-1] = rmse_row  # adding a row
        result.index = result.index + 1  # shifting index


       
        result = result.sort_values(by = ['RMSE'])
        
       
        best_result_per_ticker.append(np.array(result.iloc[0, :]))
       


    best_result_per_ticker_df = pd.DataFrame(data=best_result_per_ticker, columns=['Ticker','Model','RMSE'])
    
    
    return best_result_per_ticker_df

best_result_per_ticker = allModelsResultForAllStocks()
def classify (meanValue):
    if meanValue <=1.5:
        return 'Low'
    elif meanValue >1.5 and  meanValue <=2.5:
        return 'Medium'
    elif meanValue >2.5:
        return 'High'
def linearModel(ticker):
    stock_a = stock[stock['Name'] == ticker]
    #Neue Features hinzufügen 
    #1 Preis-Tagesveraenderung 
    stock_a['Tagesveraenderung'] = ((stock['high'] - stock['low'] )/ stock['low'])*100

    #2 Preis-VeraenderungZumVorherigenTag 
    stock_a['VeraenderungZumVorherigenTag'] = (abs(stock_a['close'].shift() - stock_a['close'] )/ stock['close'])*100

    X_stock_a = stock_a.drop(['date', 'Name','close'], axis=1)
    y_stock_a = stock_a['close']

    Lr_pipeline_std.fit(X_stock_a, y_stock_a)
    
    model = Lr_pipeline_std.named_steps['lr']
    
    return model,stock_a

#Alle 500 Aktien werden für Training eingesetzt
ticker_list = np.unique(stock['Name'])

df = pd.DataFrame(columns=['TICKER','CLASS','Coef for open','Coef for high','Coef for low','Coef for volume','Coef for change within day','Coef for change from prev day'])
for ticker in ticker_list:
    
    model,stock_a = linearModel(ticker)    
    
    print("Mean value:",stock_a["Tagesveraenderung"].mean())
    #adding target class 
    stock_features = np.concatenate((np.asarray([ticker,classify(stock_a["Tagesveraenderung"].mean())]),model.coef_))
    
    df.loc[-1] = stock_features  # adding a row
    df.index = df.index + 1  # shifting index
    df = df.sort_index() 
   
#print(df)

#Abspeichern von Feature Coeffizienten und Target Klassen aller 500 Aktien 
df.to_csv('coeff1.csv', mode='a',header=['TICKER','CLASS','Coef for open','Coef for high','Coef for low','Coef for volume','Coef for change within day','Coef for change from prev day'])

# Hochladen der libraries
import numpy as np
from sklearn.model_selection import train_test_split

X_class = np.array(df.ix[:, 2:8]) 
y_class = np.array(df['CLASS']) 


# Teilung in train and test
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X_class, y_class, test_size=0.2, random_state=42)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
# instantiate learning model (k = 3)
knn = KNeighborsClassifier(n_neighbors=3)

# fitting the model
knn.fit(X_train_class, y_train_class)

# predict the response
pred = knn.predict(X_test_class)

# evaluate accuracy
print ("Accuracy of KNN ", accuracy_score(y_test_class, pred))
from sklearn.cluster import KMeans

X_class = np.array(df.ix[:, 2:8]) 	# end index is exclusive

k_mean = KMeans()

#K-mean++ bestimmt die Clusteranzahl nun selbst 
k_mean_model = k_mean.fit(X_class)

print("Number of clusters",k_mean_model.n_clusters)
df_cluster = df.drop(['CLASS'], axis=1)

#Selecting features from dataframe , there are 6 features 
X_cluster = np.array(df_cluster.ix[:, 1:7])

y_pred = k_mean_model.predict(X_cluster)

pred_df = pd.DataFrame({'labels': y_pred, 'companies': df_cluster.ix[:, 0]})
#Cluster assignment for the stocks 
pred_df
window = 150
sharpes = []
returns = []
ignore = ["APTV"] #Werte dieser Aktie müssen komplett ignoriert werden  
for name in stock["Name"].unique(): #einbauen einer Schleife
    if name not in ignore: #aussotieren der Firmen mit invalidem Datensatz
        stock_prices = stock[stock["Name"] == name]  #
        stock_prices = stock_prices.set_index("date", drop=True)
        stock_prices.index = [pd.Timestamp(x) for x in stock_prices.index]
        daily_returns = (stock_prices["close"] / stock_prices["close"].shift()).add(-1).dropna() 
        # deklinieren der daily return durch die Ratios auf Grundlage der "closings" am Ende der Zeile werden durch 
        # die Funktionen "shift und add" alle spalten durchlaufen / die Funktion "dropna" lässt vernachlässigt alle Zeilen mit NA Daten 
        mean = daily_returns.rolling(window=window).mean().dropna() #Errechnung des Mittelwertes mit der Funktion "rolling"über die Zeitperiode "window"
        #durch die rolling Funktion werden look forward bias verhindert im anschließenden back test
        std = daily_returns.rolling(window=window).std().dropna() #Berechnung der Stadartabweichung, ebenfalls mit der Funktion "rolling"
        sharpe = mean / std #Berrechnung der Ratio wird später als "weight" verwendet
        returns.append(daily_returns.rename(name)) #befüllen der Listen sharpes und returns mit den Werten von daily returns bzw. sharpe
        sharpes.append(sharpe.rename(name))
        print("Name: {}; First Date: {}".format(name, daily_returns.index[0])) #printen der Anfangsdaten um herauszufinden welche Unternehmen zu Liste "ignore" hinzugefügt werden müssen 
def weights_generator(n): #Aufbauen der Methode "weight_generator", um die einzelnen Unternehmen nach unserer Ratio zu ranken
    non_normalized_weights = np.array([i*1/n for i in range(n)]) 
    return non_normalized_weights / np.sum(non_normalized_weights)
n = 20 #Definieren der Anzahl der Firmen, welche in das Portfolio aufgenommen werden
weight_vector = weights_generator(n) #Erstellen von Vektoren mit den Gewichten
sharpe_df = pd.concat(sharpes,axis=1) #Verbinden von den Zeilen sharpe  in sharpe_df
sharpe_df = sharpe_df.replace(to_replace=np.nan, value=-10000) #Eliminieren der "nan" Zeilen/ Vermeidung von Fehlerproduktion im Code
return_df = pd.concat(returns, axis=1) #Verbinden von den Zeilen return in return_df
mean_return = return_df.mean(axis=0) #Erstellen eines Mittelwertes über die gesamte Spalte einer Firma
std_return = return_df.std(axis=0) #Erstellen der Standardabweichung über den gesamten Zeitlauf
plt.figure(figsize=(14,9)) #Festlegung der Größe des Diagramms
plt.plot(std_return.values, mean_return.values, 'o') #abrufen der Plot-Funktion mit den Parametern return und Standardabweichung
print("Max Standard Deviation: {}".format(std_return.idxmax())) #ausgeben des Unternehmens mit der maximalen Standardabweichung
print("Max Expected Return: {}".format(mean_return.idxmax())) #ausgeben des Unternehmens mit dem maximalen Return
print("Min Standard Deviation: {}".format(std_return.idxmin())) #ausgeben des Unternehmens mit der minimalen Standardabweichung
print("Min Expected Return: {}".format(mean_return.idxmin())) #ausgeben des Unternehmens mit dem minimalen Return
rows = {}
for index, row in sharpe_df.iterrows(): 
    picks = row.sort_values().iloc[-n:]
    new_row = pd.Series(data=0, index=row.index)
    for weight_index, pick in enumerate(picks.index):
        new_row.loc[pick] = weight_vector[weight_index]
    rows[index] = new_row
weights = pd.DataFrame.from_dict(rows, orient="index")
mean_weight = weights.mean()
plt.figure(figsize=(14,9))
plt.bar(x = mean_weight.sort_values().index[-20:-10], height = mean_weight.sort_values()[-20:-10]*100)
plt.bar(x = mean_weight.sort_values().index[-10:], height = mean_weight.sort_values()[-10:]*100)
weighted_return_df = weights.mul(return_df, axis=1).replace(to_replace=np.nan, value=0).iloc[window-1:]
plt.figure(figsize=(14,9))
plt.plot(weighted_return_df.sum(axis=1).add(1).cumprod())