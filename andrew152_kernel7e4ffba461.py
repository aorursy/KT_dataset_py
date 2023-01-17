import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        if filename == "AirQualityUCI.csv":
            inputFile = os.path.join(dirname,filename)

            
#am observat ca nu il incarca bine, pentru ca delimitatorul nu este , ci ;

data = pd.read_csv(inputFile, delimiter=";", decimal=",", sep=",")
data.info()        
#_,indexer = pd.factorize(data["Date"])
#data["Date"] = indexer.get_indexer(data["Date"])
data["Date"] = list(range(data.shape[0]))

data["Date"]

#trebuie sa transformam timpul in int64 si data in datetime in loc de object, alegem sa salvam doar ora, rezultatele(y) o sa le alegem sa fie noxe/ora
data = data.set_index("Date")

data['Time'] = pd.to_datetime(data['Time'], format= '%H.%M.%S').dt.hour  
                              
#data.index = pd.to_datetime(data.index)


#ultimele doua coloane sunt unnamed, si nici pe site-ul de unde am extras datasetul nu se prezinta detalii despre ele
data = data.drop(["Unnamed: 15", "Unnamed: 16"], axis = 1)

#verificam cate valori null avem in dataset
data.isnull().sum()
#stergem valorile de null, acestea fiind parazite, adevaratele valori de NaN sunt marcate in dataset drept -200
data = data.dropna()
data.shape
#in acest dataset valorile de NaN se marcheaza drept -200, le inlocuim sa putem sa le analizam
data = data.replace(to_replace = -200, value = np.NaN)

data.isnull().sum()

import matplotlib.pyplot as plt
import seaborn as sns

sns.heatmap(data.isnull(),yticklabels=False,cbar=False,cmap='viridis')
plt.show()
#observam ca in coloana NMHC mai mult de 90% din date sunt NaN. NaN sunt in numar prea mare, prin urmare daca am inlocui aceste valori, am creste gradul de eroare, de aceea, stergem coloana
data = data.drop("NMHC(GT)", axis = 1)

#restul valorilor de null le inlocuim cu valoarea mediana
for i in data.columns:
    data[i] = data[i].fillna(data[i].median())

data.isnull().any()
#verificam daca au mai ramas valori de NaN

#folosim NOx(GT) pe post de rezultat(y), indicand valoarea reala a noxelor (Nitric oxide (NO) Nitrogen dioxide (NO2)) din aer / ora

y = data['NOx(GT)'].astype(float)
X = data.drop(['NOx(GT)', 'Time'], axis = 1).astype(float)

X.info()
print(y)
#FEATURE SELECTION
#am decis sa folosesc un filtru anova pentru SelectKBest pentru a face feature selection

from sklearn.feature_selection import SelectKBest, f_regression
from scipy.stats.stats import pearsonr

anovaFilter = SelectKBest(score_func=f_regression, k=2)


X_new = anovaFilter.fit_transform(X, y)


data_usage = 2000
data = X_new[:data_usage]
data.shape
y_new = y[:data_usage]
# #data normalization
# from sklearn.preprocessing import Normalizer

# data = Normalizer().fit_transform(data)
#Quantum Potential function
def quantumPotential(point):
    #point,width = point_and_width
    number_of_points = data.shape[0]
    sum1 = 0
    sum2 = 0
    for j in range(number_of_points):
        dist_to_j = np.sqrt((point[0] - data[j][0])**2 + (point[1] - data[j][1]) **2)
        sum1 += (dist_to_j**2) * np.exp(-(dist_to_j**2) / (2.0 * width**2))
        sum2 += np.exp(-(dist_to_j**2) / (2.0 * width**2))
    if sum2 == 0:
        sum2 += 0.000000001
    potential = (1.0 / (2.0 * width**2)) * sum1 / sum2
    return potential

from scipy.optimize import minimize
# def quantumClustering(data, width, treshold):
#     number_of_points = data.shape[0]
#     for i in range(number_of_points):
#         optimized = minimize(quantumPotential, data[i], method="BFGS")
#     print(optimized.x)
#Quantum Clustering Function

width = 2

Opt = np.zeros((data.shape[0],data.shape[1]), dtype=np.float64)

for i in range(data_usage):
    result = minimize(quantumPotential, data[i], method="BFGS")
    Opt[i] = result.x
    
    #width = result.x[2]
    
    
pltX,pltY = np.hsplit(data[:data_usage], 2) 
optX,optY = np.hsplit(Opt[:data_usage], 2)

fig = plt.figure(figsize=(5,4), dpi=80)
plt.plot(pltX, pltY, 'o')
plt.plot(optX, optY, 'x')
plt.grid('on')
plt.show()
#marking outliers
def markOutliers(clusterer):
    print(nr_clusters)
#     if nr_clusters != 1:
#         threshold = nr_points / ((nr_clusters-1)**2)
#     else:
#         threshold = 0
    labels = clusterer.labels_
    reverse_labels = {}
    reverse_labels[-1] = []
    print(nr_points)
    for i in range(nr_points):
        if labels[i] in reverse_labels:
            reverse_labels[labels[i]].append(data[i])
        else:
            reverse_labels[labels[i]] = [data[i]]
            
    average_per_cluster = 0
    null_clusters = 0
    for i in range(nr_clusters):
        if len(reverse_labels[i]) == 0:
            null_clusters+=1
        average_per_cluster += len(reverse_labels[i])
    average_per_cluster /= (nr_clusters - null_clusters)
    threshold = average_per_cluster / 1.7
    print("TH, AVERAGE: ",threshold, average_per_cluster)
    for i in range(nr_clusters):
        if len(reverse_labels[i]) < threshold:
            reverse_labels[-1].extend(reverse_labels[i])
            print("Cluster ",i,": ",len(reverse_labels[i]))
            reverse_labels[i] = []
           
    np_reverse_labels = {}
    for i in range(-1,nr_clusters):
        np_reverse_labels[i] = np.asarray(reverse_labels[i])
        print(np_reverse_labels[i].shape)
    
    return np_reverse_labels
import matplotlib
from sklearn.cluster import MeanShift

cmap = matplotlib.cm.get_cmap('hsv')


clusterer = MeanShift(max_iter = 8, cluster_all = False).fit(Opt)
nr_clusters = len(clusterer.cluster_centers_)
nr_points = data.shape[0]
reverse_labels = markOutliers(clusterer)
print("Number of clusters: ", nr_clusters)
print("Number of outliers: ", len(reverse_labels[-1]))

fig = plt.figure(figsize=(5,4), dpi=80)
labelX,labelY = np.hsplit(reverse_labels[-1], 2)
plt.plot(labelX, labelY, 'x', label = 'Outliers')
for i in range(nr_clusters):
    print(float(i)/nr_clusters)
    labelX,labelY = np.hsplit(reverse_labels[i], 2)
    plt.plot(labelX, labelY,'o', color = cmap(float(i)/nr_clusters))
plt.grid('on')
plt.show()
import matplotlib
from sklearn.cluster import MeanShift

cmap = matplotlib.cm.get_cmap('hsv')


clusterer = MeanShift(max_iter = 8, cluster_all = False).fit(data)
nr_clusters = len(clusterer.cluster_centers_)
nr_points = data.shape[0]
reverse_labels = markOutliers(clusterer)
print("Number of clusters: ", nr_clusters)
print("Number of outliers: ", len(reverse_labels[-1]))

fig = plt.figure(figsize=(5,4), dpi=80)
labelX,labelY = np.hsplit(reverse_labels[-1], 2)
plt.plot(labelX, labelY, 'x', label = 'Outliers')
for i in range(nr_clusters):
    print(float(i)/nr_clusters)
    labelX,labelY = np.hsplit(reverse_labels[i], 2)
    plt.plot(labelX, labelY,'o', color = cmap(float(i)/nr_clusters))
plt.grid('on')
plt.show()
width = 2
affinitySize = 500
data = data[:1000]

Opt = np.zeros((data.shape[0],data.shape[1]), dtype=np.float64)

for i in range(data.shape[0]):
    result = minimize(quantumPotential, data[i], method="BFGS")
    Opt[i] = result.x
    
    #width = result.x[2]
    
    
pltX,pltY = np.hsplit(data[:data_usage], 2) 
optX,optY = np.hsplit(Opt[:data_usage], 2)

fig = plt.figure(figsize=(5,4), dpi=80)
plt.plot(pltX, pltY, 'o')
plt.plot(optX, optY, 'x')
plt.grid('on')
plt.show()
from sklearn.cluster import AffinityPropagation

cmap = matplotlib.cm.get_cmap('hsv')

affinitySize = 500
Opt1 = Opt[:affinitySize]

clusterer = AffinityPropagation(damping=0.75, 
                                max_iter=30 * data_usage, 
                                convergence_iter=10, #int(15/width),  
                                affinity='euclidean', 
                                verbose=False, 
                                random_state=None).fit(Opt1)

nr_clusters = len(clusterer.cluster_centers_)
if nr_clusters:
    nr_points = Opt1.shape[0]
    reverse_labels = markOutliers(clusterer)
    print("Number of clusters: ", nr_clusters)
    print("Number of outliers: ", len(reverse_labels[-1]))

    fig = plt.figure(figsize=(5,4), dpi=80)
    labelX,labelY = np.hsplit(reverse_labels[-1], 2)
    plt.plot(labelX, labelY, 'x', label = 'Outliers')
    for i in range(nr_clusters):
        print(float(i)/nr_clusters)
        labelX,labelY = np.hsplit(reverse_labels[i], 2)
        plt.plot(labelX, labelY,'o', color = cmap(float(i)/nr_clusters))
    plt.grid('on')
    plt.show()
from sklearn.cluster import AffinityPropagation

cmap = matplotlib.cm.get_cmap('hsv')

data1 = data[:affinitySize]
clusterer = AffinityPropagation(damping=0.75, 
                                max_iter=20 * data_usage, 
                                convergence_iter=10,  
                                affinity='euclidean', 
                                verbose=False, 
                                random_state=None).fit(data1)

nr_clusters = len(clusterer.cluster_centers_)
if nr_clusters:
    nr_points = data1.shape[0]
    reverse_labels = markOutliers(clusterer)
    print("Number of clusters: ", nr_clusters)
    print("Number of outliers: ", len(reverse_labels[-1]))

    fig = plt.figure(figsize=(5,4), dpi=80)
    labelX,labelY = np.hsplit(reverse_labels[-1], 2)
    plt.plot(labelX, labelY, 'x', label = 'Outliers')
    for i in range(nr_clusters):
        print(float(i)/nr_clusters)
        labelX,labelY = np.hsplit(reverse_labels[i], 2)
        plt.plot(labelX, labelY,'o', color = cmap(float(i)/nr_clusters))
    plt.grid('on')
    plt.show()
##### TESTARE CUANTUM CLUSTERING PE REGRESII
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


Train, Test, y_train, y_test = train_test_split(Opt, y_new, test_size = 0.2)



scaler = MinMaxScaler()
Train = scaler.fit_transform(Train)
Test = scaler.transform(Test)
MAE = []
MSE = []
R2 = []
irange = range(2, 20)
for i in irange:
    mlp = MLPRegressor(solver='lbfgs',
                        activation='relu',
                        alpha=1e-5, 
                        hidden_layer_sizes=(i,1),
                        random_state=1,
                        max_iter=1000,
                        #learning_rate='adaptive'
                       )
    mlp.fit(Train, y_train)
    y_predicted = mlp.predict(Test)
    
    MAE.append(mean_absolute_error(y_test, y_predicted))
    MSE.append(mean_squared_error(y_test, y_predicted))
    R2.append(r2_score(y_test, y_predicted))
    
fig,(M1,M2,R) = plt.subplots(3)
#M1.figure(figsize=(5,4), dpi=80)
M1.plot(irange, MAE, color = cmap(0), label = "Mean Absolute Error")
M2.plot(irange, MSE, color = cmap(0.5), label = "Mean Squared Error")
R.plot(irange, R2, color = cmap(1), label = "R2 Score")
plt.legend()
plt.xlabel('Hidden layer number')
plt.grid('on')
plt.show()

print("Mean Absolute Error", MAE[6])
print("Mean Squared Error", MSE[6])
print("R2 Score", R2[6])
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor

ada = AdaBoostRegressor(base_estimator=DecisionTreeRegressor(max_depth=10), 
                        n_estimators=50, 
                        learning_rate=1.0, 
                        loss='square', 
                        random_state=None)

ada.fit(Train, y_train)
y_pred = ada.predict(Test)

print("Mean Absolute Error", mean_absolute_error(y_test, y_predicted))
print("Mean Squared Error", mean_squared_error(y_test, y_predicted))
print("R2 Score", r2_score(y_test, y_predicted))


Train, Test, y_train, y_test = train_test_split(data, y_new, test_size = 0.2)


scaler = MinMaxScaler()
Train = scaler.fit_transform(Train)
Test = scaler.transform(Test)
MAE = []
MSE = []
R2 = []
irange = range(2, 20)
for i in irange:
    mlp = MLPRegressor(solver='lbfgs',
                        activation='relu',
                        alpha=1e-5, 
                        hidden_layer_sizes=(i,1),
                        random_state=1,
                        max_iter=1000,
                        #learning_rate='adaptive'
                       )
    mlp.fit(Train, y_train)
    y_predicted = mlp.predict(Test)
    
    MAE.append(mean_absolute_error(y_test, y_predicted))
    MSE.append(mean_squared_error(y_test, y_predicted))
    R2.append(r2_score(y_test, y_predicted))
    
fig,(M1,M2,R) = plt.subplots(3)
#M1.figure(figsize=(5,4), dpi=80)
M1.plot(irange, MAE, color = cmap(0), label = "Mean Absolute Error")
M2.plot(irange, MSE, color = cmap(0.5), label = "Mean Squared Error")
R.plot(irange, R2, color = cmap(1), label = "R2 Score")
plt.legend()
plt.xlabel('Hidden layer number')
plt.grid('on')
plt.show()

print("Mean Absolute Error", MAE[6])
print("Mean Squared Error", MSE[6])
print("R2 Score", R2[6])
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor

ada = AdaBoostRegressor(base_estimator=DecisionTreeRegressor(max_depth=10), 
                        n_estimators=50, 
                        learning_rate=1.0, 
                        loss='square', 
                        random_state=None)

ada.fit(Train, y_train)
y_pred = ada.predict(Test)

print("Mean Absolute Error", mean_absolute_error(y_test, y_predicted))
print("Mean Squared Error", mean_squared_error(y_test, y_predicted))
print("R2 Score", r2_score(y_test, y_predicted))