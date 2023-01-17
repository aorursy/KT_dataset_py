"""
Libraries
"""
import numpy as np
import pandas as pd
import scipy
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
"""
Load data from csv
"""
path_to_data = "C:\\Users\\Rafael Beni\\Documents\\python\\uea\\data\\train.csv"
data = pd.read_csv(path_to_data)

"""
Compute the mean and std to be used for later scaling
"""
X = data.drop(['id', 'quality'], axis=1).copy()
scaler = StandardScaler()
scaler.fit(X)
del X
"""
Split dataset
"""
y = data['quality'].copy()
x = data.drop(['id', 'quality'], axis=1).copy()

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.25)


"""
Scale features
"""
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

#for a in ['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol','white']:
#    print(a, scipy.stats.pearsonr(X_train_scaled[a], y))
"""
Produce the combinations of hidden layers' neurons, as in: 
https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw

Conditions:
1: number of layers = 2
2: minimal number of neurons per layer = number of outputs
3: maximum number of neurons per layer = number of inputs
"""
#2
min_neurons = 10
#3
max_neurons = len(x.columns)
#Aqui são inseridas as redes para o teste.
combinations = [(70,30,15),(70,140,30,15),(30,15),(140,70,70),(140,140,140),(200,100,50),
               (10,10,10,10),(40,20,10),(15,15,15,15),(17,15,15)]
    

print(combinations)
"""
Create networks

For every combination produced on last step, create every single possible MPLC
"""
NNs = []

for c in combinations:
    for a in [0.0001, 0.001, 0.01,  0.1]:
        for s in ['lbfgs', 'sgd', 'adam']:
            for activation in ['identity', 'logistic', 'tanh', 'relu']:
                for iter in [100, 200, 300]:
                    NNs.append(MLPClassifier(activation=activation, alpha=a, batch_size='auto', beta_1=0.9,
                           beta_2=0.999, early_stopping=True, epsilon=0.00000001,
                           hidden_layer_sizes=c, learning_rate='constant',
                           learning_rate_init=0.03, max_iter=1000, momentum=0.9,
                           nesterovs_momentum=True, power_t=0.5, random_state=None,
                           shuffle=True, solver=s, tol=0.0001, validation_fraction=0.1,
                           verbose=False, warm_start=False))

print("Number of nets created: ", len(NNs))

"""
Train networks
"""
import winsound
duration = 500  # millisecond
freq = 1000  # Hz

for i, net in enumerate(NNs):
    print(i+1)
    net.fit(X_train_scaled, Y_train)
        
"""
Testando as redes
"""
resultado_teste = []
for i, net in enumerate(NNs):
    resultado_teste.append(net.predict(X_test_scaled))
        
"""
Calculando os F-Score
"""
resultado = []
for i in range(0, len(resultado_teste)):
    resultado.append({'f-score':f1_score(Y_test, resultado_teste[i], average='micro'), 'rede': NNs[i]})
        
"""
Get best NN and predict on final dataset
"""
#load dataset
path_to_test_dataset = "C:\\Users\\Rafael Beni\\Documents\\python\\uea\\data\\test.csv"
data = pd.read_csv(path_to_test_dataset)

#Split dataset
X_t = data.drop(['id'], axis=1).copy()
    
#Scale test data
X_scaled = scaler.transform(X_t)

ordered_results = pd.DataFrame(resultado).sort_values(by=['f-score'], ascending=False)
for i in range(5):
    print(ordered_results.iloc[i]['f-score'])
    NN = ordered_results.iloc[i]['rede']

    #Predict
    res = NN.predict(X_scaled)
    
    #Make CSV
    df = pd.DataFrame(np.array(res).reshape(len(res)), columns = ["quality"])
    df = df.assign(id=data['id'])
    df = df.reset_index(drop=True)
    df.to_csv("C:\\Users\\Rafael Beni\\Documents\\python\\uea\\output\\output-" + str(ordered_results.iloc[i]['f-score']) + ".csv" ,  index = False)

winsound.Beep(freq, duration)
for i in range(5):
    print(ordered_results.iloc[i]['f-score'], ordered_results.iloc[i]['rede'])
    
#fit again
NNagain = ordered_results.iloc[0]['rede'].fit(np.concatenate((X_train_scaled,X_test_scaled), axis=0), np.concatenate((Y_train,Y_test), axis=0))
"""
Get best NN and predict on final dataset
"""

#load dataset
path_to_test_dataset = "C:\\Users\\Rafael Beni\\Documents\\python\\uea\\data\\test.csv"
data = pd.read_csv(path_to_test_dataset)
data

#Split dataset
X_t = data.drop(['id'], axis=1).copy()

#Select best NN
#best = pd.DataFrame(NNagain).sort_values(by=['f-score'], ascending=False).iloc[0]
NN = NNagain
print(NN)

#Scale test data
X_scaled = scaler.fit(X_train).transform(X_t)

#Predict
res = NN.predict(X_scaled)
len(res)

#Make CSV
df = pd.DataFrame(np.array(res).reshape(len(res)), columns = ["quality"])
df = df.assign(id=data['id'])
df = df.reset_index(drop=True)
df.to_csv("C:\\Users\\Rafael Beni\\Documents\\python\\uea\\output-0.605.csv" ,  index = False)
df
