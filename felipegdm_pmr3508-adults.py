import pandas as pd
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

%matplotlib inline
plt.style.use('bmh')
plt.rcParams['figure.figsize'] = (12, 8)
adult = pd.read_csv("../input/uci-adults/train_data.csv",
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")
adult.head()
def plot_score(data_frame, labels = None, tag = 'Target', title = 'Teste de desempenho de algoritmo kNN', \
               J = 1, K = 40, cv_ =10, color = 'ro'):

    if labels != None:
        data_frame = data_frame[labels]
            
    ndata_frame = data_frame.fillna(0)
    #print(ndata_frame.shape)
        
    xdata = ndata_frame.drop(tag, axis = 1)
    ydata = ndata_frame[tag]
    
    mean_score = []

    for i in range(J, K):
        print(i, end=' ')
        knn = KNeighborsClassifier(n_neighbors=i, p=2, metric = 'minkowski', n_jobs = -1)
        scores = cross_val_score(knn, xdata, ydata, cv=cv_)
        mean_score.append( np.mean(scores) )
        
    print()
    plt.style.use('bmh')
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.plot(range(J, K), mean_score, color)
    plt.xlabel('Número k de vizinhos')
    plt.ylabel('Score')    
    plt.title(title)
def normalize(dataframe, tag = 'Target'):
    
    norm_families = pd.DataFrame()
    nplot = dataframe.copy()
    
    data_mean = []
    data_std  = []
    
    for header in list(nplot)[:-1]:

        deviation = dataframe[header].std()
        mean      = dataframe[header].mean()

        #  Se o desvio padrão é mt baixo, os dados são demasiadamente semelhantes
        # e portanto inúteis para o classificador
        if 0.005 < deviation:
            norm_series = dataframe[header].apply(lambda x: (x - mean)/deviation)
            norm_families = pd.concat( [norm_families, norm_series], axis = 1)
            
            data_mean.append(mean)
            data_std.append(deviation)   
    
    norm_families = pd.concat( [norm_families, dataframe[tag]], axis = 1 )
    
    return norm_families, data_mean, data_std
def train(data, header, tag = 'income', k = 50, i = 0, f = 10, passo = 0.2, p_=2):
    
    output = []
    
    testing = data.copy()
    print('Testing for {0}'.format(header))
    for n in np.arange(i, f, passo):

        test = testing.copy()
        test[header] = test[header].apply(lambda x: n*x)
        score = get_score(test, tag = tag, k = k, p_ = p_)
        #score = get_score(test, tag = tag)
        
        print(round(n, 2), end = ' ')
        
        output.append( score )

    print()
    return (output.index( np.max(output) )*passo + i)
def apply_weight(families, w, tag = 'Target'):

    target = families[tag]
    families = families.drop(tag, axis = 1)
    
    if len( list(families) ) != len(w):
        raise ValueError("Data size {0} and {1} weights given" \
                         .format(len( list(families) ), len(w)))
    
    for i, header in enumerate( list(families) ):
        families[header] = families[header].apply(lambda x: w[i]*x )        
        
    return pd.concat( [families, target], axis=1)
def get_score(data_frame, tag = 'income', k = 50, cv_ = 10, p_ = 2):

    ndata_frame = data_frame.fillna(0)
    
    xdata = ndata_frame.drop(tag, axis = 1)
    ydata = ndata_frame[tag]
    
    knn = KNeighborsClassifier(n_neighbors = k, p = p_, metric = 'minkowski', n_jobs = -1)
    scores = cross_val_score(knn, xdata, ydata, cv=cv_)
        
    return np.mean(scores)
headers = ["age",
           "education.num",
           "capital.gain", 
           "capital.loss", 
           "hours.per.week", 
           "income"]
local = adult[headers]
list(adult)
norm_adults, means, stds = normalize(local, tag = 'income')
norm_adults = norm_adults.fillna(0)
#plot_score(norm_adults, J = 10, K = 50, tag = 'income')
#ws = [1 for i in range( len(list(norm_adults)) - 1)]
#wted_data = apply_weight(norm_adults, ws, tag = 'income')

#for i in range( len(list(norm_adults)) - 1):
    
#    wted_data = apply_weight(norm_adults.fillna(0), ws, tag = 'income')
#    new_w = train(wted_data, list(norm_adults)[i] , tag = 'income')
#    print('\n{0}'.format(str(new_w)))
#    ws[i] = new_w
#    print(ws)
#    print()
pesos_top = [0.4, 1.0, 9.2, 5.0, 0.6]
wted_data = apply_weight(norm_adults, pesos_top, tag = 'income')
get_score(wted_data, k=35, tag='income')
test_input = pd.read_csv("../input/uci-adults/test_data.csv",
        sep=r',',
        engine='python',
        na_values='0')
fields = list(norm_adults)[:-1]

test = test_input.copy()
test = test[fields]

for i, header in enumerate(fields):
    test[header] = test[header].apply(lambda x: (x-means[i])/stds[i])
    
test = test.fillna(0)
Xtrain = apply_weight(norm_adults.fillna(0), pesos_top, tag = 'income')

Xtrain = Xtrain.drop('income', axis=1)
Ytrain = norm_adults['income']

knn = KNeighborsClassifier(n_neighbors = 36,  n_jobs = -1)
knn.fit(Xtrain, Ytrain)
Ytest = knn.predict(test)
output = pd.concat([ test_input['Id'].fillna(0).astype(int) , pd.Series(Ytest, name='income')], axis = 1)
output.to_csv('./sumit.csv', index = False)
