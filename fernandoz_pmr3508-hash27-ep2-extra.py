import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import sklearn as skl
#import pandas as pd

#housing = pd.read_csv("../input/california-housing/housing.csv")

import pandas as pd

test = pd.read_csv("../input/atividade-3-pmr3508/test.csv")

housing = pd.read_csv("../input/atividade-3-pmr3508/train.csv")
aux = housing
housing.shape
housing.head()
housing = housing.dropna()
housing.hist(bins=20, figsize=(25,20))
housing = housing.drop(columns=["Id"])
y = housing.median_house_value 

X = housing.drop(columns=["median_house_value"])

X.shape
print('Extremos dos valores')

print(' ')

print('longitude')

print('Min ',min(X['longitude'])) 

print('Max ',max(X['longitude']))

print(' ')

print('latitude')

print('Min ',min(X['latitude']))

print('Max ',max(X['latitude']))

print(' ')

print('median_age')

print('Min ',min(X['median_age']))

print('Max ',max(X['median_age']))

print(' ')

print('total_rooms')

print('Min ',min(X['total_rooms']))

print('Max ',max(X['total_rooms']))

print(' ')

print('total_bedrooms')

print('Min ',min(X['total_bedrooms']))

print('Max ',max(X['total_bedrooms']))

print(' ')

print('population')

print('Min ',min(X['population']))

print('Max ',max(X['population']))

print(' ')

print('households')

print('Min ',min(X['households']))

print('Max ',max(X['households']))

print(' ')

print('median_income')

print('Min ',min(X['median_income']))

print('Max ',max(X['median_income']))



print(' ')

print('median_house_value')

print('Min ',min(housing['median_house_value']))

print('Max ',max(housing['median_house_value']))



import seaborn

plt.figure(figsize=(10,10))

plt.title("Matriz de correlação")

seaborn.heatmap(housing.corr(), annot=True, linewidths=0.2)
v_correl = [0.045,0.140,0.11, 0.13 ,0.05, 0.025, 0.065 , 0.69] #correlacoes em valroes absolutos

media = (0.045 + 0.140 +0.11 + 0.13 + 0.05 + 0.025 + 0.065 + 0.69)/8.0

print('Media do modulo das correlacoes em relacao a MEDIAN_HOUSE_VALUE = ',media)

i=1

v_irrelevantes = []

while i<8 :

    if(v_correl[i] < 0.05): #Se a correlacao for menor do que 10%

        v_irrelevantes.insert(i,i-1)

    i = i + 1

print(v_irrelevantes)
v_correl = [0.045,0.140,0.11, 0.13 ,0.05, 0.025, 0.065 , 0.69]

media = (0.045 + 0.140 +0.11 + 0.13 + 0.05 + 0.025 + 0.065 + 0.69)/8.0

print('Media do modulo das correlacoes em relacao a MEDIAN_HOUSE_VALUE = ',media)

i=1

v_irrelevantes = []

while i<8 :

    if(v_correl[i] < media): #Se a correlacao for menor do que 10%

        v_irrelevantes.insert(i,i-1)

    i = i + 1

print(v_irrelevantes)
aux["NEW_persons_per_room"] = X["population"]/X["total_rooms"]

aux["NEW_persons_per_bedroom"] = X["population"]/X["total_bedrooms"]
aux["NEW_house_age/income"] = X["median_age"]/X["median_income"]
import seaborn

plt.figure(figsize=(20,20))

plt.title("Matriz de correlação")

seaborn.heatmap(aux.corr(), annot=True, linewidths=0.2)
aux.plot(kind='scatter',x='median_house_value',y='total_rooms',color='red')

plt.show()
aux.plot(kind='scatter',x='median_house_value',y='NEW_persons_per_room',color='red')

plt.show()
trai1 = X

trai1.drop(columns=["total_bedrooms"])
trai2 = X

trai2["house_age/income"] = X["median_age"]/X["median_income"]
trai3 = X

trai3.drop(columns=["total_rooms"])

trai3.drop(columns=["total_bedrooms"])
trai4 = X

trai4["house_age/income"] = X["median_age"]/X["median_income"]

trai4.drop(columns=["total_rooms"])

trai4.drop(columns=["total_bedrooms"])

trai4.drop(columns=["longitude"])

trai4.drop(columns=["latitude"])

trai4.drop(columns=["median_age"])

def RF_cv_select_pred(num,train,y):  

    trainY = y

    from sklearn.ensemble import RandomForestRegressor

    forest = RandomForestRegressor(n_estimators=num, criterion='mse', min_samples_split=5, 

                          min_samples_leaf=5, min_weight_fraction_leaf=0.0, max_features="sqrt", 

                         random_state=None, verbose=0, warm_start=True)

    #Usando os dados

    forest.fit(train, trainY)

    score = forest.score(train, trainY)

    print (score)

    predictions = forest.predict(test)

    return score
RF_cv_select_pred(30,X,y)
i=1

arr_res = []

while i<150 :

    rf = RF_cv_select_pred(i,trai1,y)

    arr_res.insert(i, rf)

    print(i)

    i = i + 5

arr = np.array(arr_res)

Index_Max_Accuracy = np.where(arr == np.amax(arr))

print('List of Indices of maximum element :', Index_Max_Accuracy[0])

# Get the maximum element from a Numpy array

print('Max accuracy is :', np.amax(arr))
i=1

arr_res = []

while i<150 :

    rf = RF_cv_select_pred(i,trai2,y)

    arr_res.insert(i, rf)

    print(i)

    i = i + 5

arr = np.array(arr_res)

Index_Max_Accuracy = np.where(arr == np.amax(arr))

print('List of Indices of maximum element :', Index_Max_Accuracy[0])

# Get the maximum element from a Numpy array

print('Max accuracy is :', np.amax(arr))
i=1

arr_res = []

while i<150 :

    rf = RF_cv_select_pred(i,trai3,y)

    arr_res.insert(i, rf)

    print(i)

    i = i + 5

arr = np.array(arr_res)

Index_Max_Accuracy = np.where(arr == np.amax(arr))

print('List of Indices of maximum element :', Index_Max_Accuracy[0])

# Get the maximum element from a Numpy array

print('Max accuracy is :', np.amax(arr))
i=1

arr_res = []

while i<150 :

    rf = RF_cv_select_pred(i,trai4,y)

    arr_res.insert(i, rf)

    print(i)

    i = i + 5

arr = np.array(arr_res)

Index_Max_Accuracy = np.where(arr == np.amax(arr))

print('List of Indices of maximum element :', Index_Max_Accuracy[0])

# Get the maximum element from a Numpy array

print('Max accuracy is :', np.amax(arr))
def adaboost_neural(num,train,y):  

    trainY = y   

    from sklearn.ensemble import AdaBoostRegressor

    from sklearn import neural_network



    neural_net = neural_network.MLPRegressor(hidden_layer_sizes=(100,),

                                       activation='relu', solver='adam',

                                       learning_rate='adaptive', max_iter=800,

                                       learning_rate_init=0.01, warm_start = True, alpha=0.01)

    adaboost1 = AdaBoostRegressor(base_estimator=neural_net, n_estimators=num, learning_rate=0.01, random_state=None)

    adaboost1.fit(train, trainY)

    neural = adaboost1.score(train, trainY)*100

    print ("AdaBoost on Neural Network score: ", str(neural), "%")

    return neural
def adaboost_RF(num,train,y):  

    trainY = y   

    from sklearn.ensemble import AdaBoostRegressor

    from sklearn.ensemble import RandomForestRegressor

    forest = RandomForestRegressor(n_estimators=num, criterion='mse', min_samples_split=5, 

                          min_samples_leaf=5, min_weight_fraction_leaf=0.0, max_features="sqrt", 

                         random_state=None, verbose=0, warm_start=True)

    

    adaboost2 = AdaBoostRegressor(base_estimator=forest, n_estimators=num, learning_rate=0.01, random_state=None)

    adaboost2.fit(train, trainY)

    RF = adaboost2.score(train, trainY)*100

    print ("AdaBoost on Random Forest score: ", str(RF), "%")

    return RF
adaboost_neural(7,trai1,y)

adaboost_neural(7,trai2,y)

adaboost_neural(7,trai3,y)

adaboost_neural(7,trai4,y)
i=35

arr_res = []

while i<50 :

    rf = adaboost_RF(i,trai1,y)

    arr_res.insert(i, rf)

    print(i)

    i = i + 5

arr = np.array(arr_res)

Index_Max_Accuracy = np.where(arr == np.amax(arr))

print('List of Indices of maximum element :', Index_Max_Accuracy[0])

# Get the maximum element from a Numpy array

print('Max accuracy is :', np.amax(arr))
i=35

arr_res = []

while i<50 :

    rf = adaboost_RF(i,trai2,y)

    arr_res.insert(i, rf)

    print(i)

    i = i + 5

arr = np.array(arr_res)

Index_Max_Accuracy = np.where(arr == np.amax(arr))

print('List of Indices of maximum element :', Index_Max_Accuracy[0])

# Get the maximum element from a Numpy array

print('Max accuracy is :', np.amax(arr))
i=35

arr_res = []

while i<50 :

    rf = adaboost_RF(i,trai3,y)

    arr_res.insert(i, rf)

    print(i)

    i = i + 5

arr = np.array(arr_res)

Index_Max_Accuracy = np.where(arr == np.amax(arr))

print('List of Indices of maximum element :', Index_Max_Accuracy[0])

# Get the maximum element from a Numpy array

print('Max accuracy is :', np.amax(arr))
i=35

arr_res = []

while i<50 :

    rf = adaboost_RF(i,trai4,y)

    arr_res.insert(i, rf)

    print(i)

    i = i + 5

arr = np.array(arr_res)

Index_Max_Accuracy = np.where(arr == np.amax(arr))

print('List of Indices of maximum element :', Index_Max_Accuracy[0])

# Get the maximum element from a Numpy array

print('Max accuracy is :', np.amax(arr))
def linear(train,y):  

    trainY = y   

    from sklearn.linear_model import LinearRegression

    LR = LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=None)

    LR.fit(train, trainY)

    lin = LR.score(train, trainY)*100

    print ("Linear: ", str(lin), "%")

    return lin
print(linear(trai1,y))

print(linear(trai2,y))

print(linear(trai3,y))

print(linear(trai4,y))