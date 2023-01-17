from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
rcParams['font.size']=13
import matplotlib.pyplot as plt

import pandas as pd # para leer datos
import sklearn.ensemble # para el random forest
import sklearn.model_selection # para split train-test
import sklearn.metrics # para calcular el f1-score
from datetime import datetime
data1 = pd.read_csv('../input/dataset1/johns_hopkins_csse/johns-hopkins-covid-19-daily-dashboard-cases-over-time.csv')
#data2 = pd.read_csv('../input/dataset1/johns_hopkins_csse/johns-hopkins-covid-19-daily-dashboard-cases-by-country.csv')
data3 = pd.read_csv('../input/dataset2/google_mobility/regional-mobility.csv')
data3.head()
data1.head()
dates = data3.iloc[12103:12164,2].to_numpy()
y_values = data3.iloc[12103:12164,3:9].to_numpy()
x_values = [datetime.strptime(d,"%Y-%m-%d").date() for d in dates]
index = ['retail','grocery_and_pharmacy','parks','transit_stations','workplaces','residential']

plt.figure(figsize = (14,8))
for g in range(6):
    plt.subplot(2,3,g+1)
    plt.plot_date(x_values, y_values[:,g], '-', label =index[g])
    plt.xlabel('Fecha')
    plt.ylabel('porcentaje' )
    plt.xticks(rotation=30, fontsize =9)  
    plt.legend()

plt.suptitle('Movilidad Colombia')
plt.show()

paises_data1 = []
paises_data2 = []

for n in range(181):
    paises_data1.append(data1.iloc[20+(n*100),0])

paises_data1 = np.array(paises_data1)

for jj in range(len(paises_data1)):
    for ii in range(len(data3)):
        if (data3.iloc[ii,0]== paises_data1[jj]):
            if (data3.iloc[ii,1] == 'Total'):
                if (data3.iloc[ii,2] == '2020-03-05'):        
                    paises_data2.append(data3.iloc[ii,0])
                    
print(len(paises_data2))     
latam = ['Argentina','Barbados','Belize','Bolivia','Brazil','Chile','Colombia','Costa Rica','Dominican Republic','Ecuador','El Salvador','Guatemala','Haiti','Honduras','Mexico','Nicaragua','Panama','Paraguay','Peru','Uruguay','Venezuela']

for ll,mm in enumerate(paises_data2):
    for kk,ii in enumerate(latam):
        if (ii==mm):
            paises_data2.remove(ii)
print(paises_data2)
data_final = []
  
for k in range(len(paises_data2)):
    
    quincena1 = []
    quincena2 = []
    quincena3 = []
    quincena4 = []

    for i in range(len(data1)):

        if (data1.iloc[i,0] == paises_data2[k]):
            if (data1.iloc[i,1] == '2020-03-20'):
                quincena1.append(data1.iloc[i,8])
                quincena1.insert(0,data1.iloc[i,0])
            
            if (data1.iloc[i,1] == '2020-04-05'):
                quincena2.append(data1.iloc[i,8])
                quincena2.insert(0,data1.iloc[i,0])
            if (data1.iloc[i,1] == '2020-04-20'):
                quincena3.append(data1.iloc[i,8])
                quincena3.insert(0,data1.iloc[i,0])
                
                
            if (data1.iloc[i,1] == '2020-04-26'):
                quincena4.append(data1.iloc[i,8])
                quincena4.insert(0,data1.iloc[i,0])

    for i in range(len(data3)):
        if (data3.iloc[i,0] == paises_data2[k]):
            if (data3.iloc[i,1] == 'Total'):
                if (data3.iloc[i,2] == '2020-03-05'):
                    quincena1.extend(data3.iloc[i,3:].values.tolist())

            if (data3.iloc[i,1] == 'Total'):
                if (data3.iloc[i,2] == '2020-03-20'):
                    quincena2.extend(data3.iloc[i,3:].values.tolist())

            if (data3.iloc[i,1] == 'Total'):
                if (data3.iloc[i,2] == '2020-04-05'):
                    quincena3.extend(data3.iloc[i,3:].values.tolist())
                    
            if (data3.iloc[i,1] == 'Total'):
                if (data3.iloc[i,2] == '2020-04-11'):
                    quincena4.extend(data3.iloc[i,3:].values.tolist())
    
    data_final.append(quincena1)
    data_final.append(quincena2)
    data_final.append(quincena3)   
    data_final.append(quincena4) 
    
print(len(data_final))   
data_final2 = np.array(data_final)

nan_rows = []

for i in range(len(data_final2)):
    for j in range(8):
        if (data_final2[i,j]=='nan'):
            nan_rows.append(i)
data_final3 = data_final2

for i in nan_rows[::-1]:
    data_final3 = np.delete(data_final3, (i), axis=0)
df2 = pd.DataFrame(data=data_final3, columns=['pais', 'incident_rate','retail','grocery_and_pharmacy','parks','transit_stations','workplaces','residential'])
target = np.zeros(len(df2))
for ii in range(len(df2)):
    if (float(df2.iloc[ii,1])<10.0):
        target[ii]=0
    elif (float(df2.iloc[ii,1])>70.0):
        target[ii]=2
    else:
        target[ii]=1      

Y = df2['incident_rate']
X = df2.drop(['pais','incident_rate'],axis=1)
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
                                    X, target, test_size=0.25)
n_trees = np.arange(1,100,2)
f1_train = []
f1_test = []
feature_importance = np.zeros((len(n_trees), 6))

for i, n_tree in enumerate(n_trees):
    clf = sklearn.ensemble.RandomForestClassifier(n_estimators=n_tree, max_features='sqrt')
    clf.fit(X_train, y_train)
    f1_train.append(sklearn.metrics.f1_score(y_train, clf.predict(X_train), average='micro'))
    f1_test.append(sklearn.metrics.f1_score(y_test, clf.predict(X_test), average='micro'))
    feature_importance[i, :] = clf.feature_importances_
plt.figure(figsize =(4,4))
plt.scatter(n_trees, f1_train, label = 'Train')
plt.scatter(n_trees, f1_test, label = 'Test')
plt.ylim(0,1.2)
plt.xlabel('Num. Trees')
plt.ylabel('f1 score')
plt.legend()
plt.show()
# Grafica los features mas importantes
clf = sklearn.ensemble.RandomForestClassifier(n_estimators=20, max_features='sqrt')
clf.fit(X_train, y_train)
avg_importance = np.average(feature_importance, axis=0)
a = pd.Series(avg_importance, index = index)
print(a)
a.nlargest().plot(kind='barh')
plt.xlabel('Average Feature Importance')

a = Y.astype(float).to_numpy()

plt.figure(figsize = (4,4))
_ = plt.hist(a, bins = 300)
plt.xlim(-1,15)
plt.xlabel("Incident Rate")
plt.ylabel("histograma")
plt.title("Mediana: {}".format(np.median(a)))


target = np.zeros(len(df2))
for ii in range(len(df2)):
    if (float(df2.iloc[ii,1])<5.86):
        target[ii]=0
    else:
        target[ii]=1      

Y = df2['incident_rate']
X = df2.drop(['pais','incident_rate'],axis=1)
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
                                    X, target, test_size=0.25)

n_trees = np.arange(1,100,2)
f1_train = []
f1_test = []
feature_importance = np.zeros((len(n_trees), 6))

for i, n_tree in enumerate(n_trees):
    clf = sklearn.ensemble.RandomForestClassifier(n_estimators=n_tree, max_features='sqrt')
    clf.fit(X_train, y_train)
    f1_train.append(sklearn.metrics.f1_score(y_train, clf.predict(X_train), average='micro'))
    f1_test.append(sklearn.metrics.f1_score(y_test, clf.predict(X_test), average='micro'))
    feature_importance[i, :] = clf.feature_importances_
plt.figure(figsize =(4,4))
plt.scatter(n_trees, f1_train, label = 'Train')
plt.scatter(n_trees, f1_test, label = 'Test')
plt.ylim(0,1.2)
plt.xlabel('Num. Trees')
plt.ylabel('f1 score')
plt.legend()
plt.show()
# Grafica los features mas importantes
clf = sklearn.ensemble.RandomForestClassifier(n_estimators=20, max_features='sqrt')
clf.fit(X_train, y_train)
avg_importance = np.average(feature_importance, axis=0)
a = pd.Series(avg_importance, index = index)
print(a)
a.nlargest().plot(kind='barh')
plt.xlabel('Average Feature Importance')

latam = ['Argentina','Barbados','Belize','Bolivia','Brazil','Chile','Colombia','Costa Rica','Dominican Republic','Ecuador','El Salvador','Guatemala','Haiti','Honduras','Mexico','Nicaragua','Panama','Paraguay','Peru','Uruguay','Venezuela']
data_final_latam = []
   
for k in range(len(latam)):
    
    quincena1 = []
    quincena2 = []
    quincena3 = []

    for i in range(len(data1)):

        if (data1.iloc[i,0] == latam[k]):
            if (data1.iloc[i,1] == '2020-03-20'):
                quincena1.append(data1.iloc[i,8])
                quincena1.insert(0,data1.iloc[i,0])
            
            if (data1.iloc[i,1] == '2020-04-05'):
                quincena2.append(data1.iloc[i,8])
                quincena2.insert(0,data1.iloc[i,0])
            if (data1.iloc[i,1] == '2020-04-20'):
                quincena3.append(data1.iloc[i,8])
                quincena3.insert(0,data1.iloc[i,0])

    for i in range(len(data3)):
        if (data3.iloc[i,0] == latam[k]):
            if (data3.iloc[i,1] == 'Total'):
                if (data3.iloc[i,2] == '2020-03-05'):
                    quincena1.extend(data3.iloc[i,3:].values.tolist())

            if (data3.iloc[i,1] == 'Total'):
                if (data3.iloc[i,2] == '2020-03-20'):
                    quincena2.extend(data3.iloc[i,3:].values.tolist())

            if (data3.iloc[i,1] == 'Total'):
                if (data3.iloc[i,2] == '2020-04-05'):
                    quincena3.extend(data3.iloc[i,3:].values.tolist())

    data_final_latam.append(quincena1)
    data_final_latam.append(quincena2)
    data_final_latam.append(quincena3)      
data_final2_latam = np.array(data_final_latam)

nan_rows_latam = []

for i in range(len(data_final2_latam)):
    for j in range(8):
        if (data_final2_latam[i,j]=='nan'):
            nan_rows_latam.append(i)

data_final3_latam = data_final2_latam

for i in nan_rows_latam[::-1]:
    data_final3_latam = np.delete(data_final3_latam, (i), axis=0)
df2_latam = pd.DataFrame(data=data_final3_latam, columns=['pais', 'incident_rate','retail','grocery_and_pharmacy','parks','transit_stations','workplaces','residential'])
target_latam = np.zeros(len(df2_latam))
for ii in range(len(df2_latam)):
    if (float(df2_latam.iloc[ii,1])<10.0):
        target_latam[ii]=0
    elif (float(df2_latam.iloc[ii,1])>70.0):
        target_latam[ii]=2
    else:
        target_latam[ii]=1
        
Y_latam = df2_latam['incident_rate']
X = df2_latam.drop(['pais','incident_rate'],axis=1)

target_latam
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
                                    X, target_latam, test_size=0.30)
n_trees = np.arange(1,100,2)
f1_train = []
f1_test = []
feature_importance = np.zeros((len(n_trees), 6))

for i, n_tree in enumerate(n_trees):
    clf = sklearn.ensemble.RandomForestClassifier(n_estimators=n_tree, max_features='sqrt')
    clf.fit(X_train, y_train)
    f1_train.append(sklearn.metrics.f1_score(y_train, clf.predict(X_train), average='micro'))
    f1_test.append(sklearn.metrics.f1_score(y_test, clf.predict(X_test), average='micro'))
    feature_importance[i, :] = clf.feature_importances_
plt.figure(figsize =(4,4))
plt.scatter(n_trees, f1_train, label = 'Train')
plt.scatter(n_trees, f1_test, label = 'Test')
plt.ylim(0,1.2)
plt.xlabel('Num. Trees')
plt.ylabel('f1 score')
plt.legend()
plt.show()
# Grafica los features mas importantes
clf = sklearn.ensemble.RandomForestClassifier(n_estimators=50, max_features='sqrt')
clf.fit(X_train, y_train)
avg_importance = np.average(feature_importance, axis=0)
a = pd.Series(avg_importance, index = index)
print(a)
a.nlargest().plot(kind='barh')
plt.xlabel('Average Feature Importance')

a_latam = Y_latam.astype(float).to_numpy()

plt.figure(figsize = (4,4))
_ = plt.hist(a_latam, bins = 100)
plt.xlim(-1,15)
plt.xlabel("Incident Rate")
plt.ylabel("histograma")
plt.title("Mediana: {}".format(np.median(a_latam)))
target_latam = np.zeros(len(df2_latam))
for ii in range(len(df2_latam)):
    if (float(df2_latam.iloc[ii,1])<2.08):
        target_latam[ii]=0
    else:
        target_latam[ii]=1
        
Y_latam = df2_latam['incident_rate']
X = df2_latam.drop(['pais','incident_rate'],axis=1)

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
                                    X, target_latam, test_size=0.30)
n_trees = np.arange(1,100,2)
f1_train = []
f1_test = []
feature_importance = np.zeros((len(n_trees), 6))

for i, n_tree in enumerate(n_trees):
    clf = sklearn.ensemble.RandomForestClassifier(n_estimators=n_tree, max_features='sqrt')
    clf.fit(X_train, y_train)
    f1_train.append(sklearn.metrics.f1_score(y_train, clf.predict(X_train), average='micro'))
    f1_test.append(sklearn.metrics.f1_score(y_test, clf.predict(X_test), average='micro'))
    feature_importance[i, :] = clf.feature_importances_
plt.figure(figsize =(4,4))
plt.scatter(n_trees, f1_train, label = 'Train')
plt.scatter(n_trees, f1_test, label = 'Test')
plt.ylim(0,1.2)
plt.xlabel('Num. Trees')
plt.ylabel('f1 score')
plt.legend()
plt.show()
# Grafica los features mas importantes
clf = sklearn.ensemble.RandomForestClassifier(n_estimators=50, max_features='sqrt')
clf.fit(X_train, y_train)
avg_importance = np.average(feature_importance, axis=0)
a = pd.Series(avg_importance, index )
print(a)
a.nlargest().plot(kind='barh')
plt.xlabel('Average Feature Importance')

