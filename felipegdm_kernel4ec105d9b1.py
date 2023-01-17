import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy.core.multiarray

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
%matplotlib inline

plt.style.use('bmh')
plt.rcParams['figure.figsize'] = (12, 8)
def plot_score(data_frame, labels = None, title = 'Teste de desempenho de algoritmo kNN', \
               J = 1, K = 40, cv_ =10, color = 'ro'):

    if labels != None:
        data_frame = data_frame[labels]
            
    ndata_frame = data_frame.fillna(0)
    #print(ndata_frame.shape)
        
    xdata = ndata_frame.drop('Target', axis = 1)
    ydata = ndata_frame.Target
    
    mean_score = []

    for i in range(J, K):
        knn = KNeighborsClassifier(n_neighbors=i, p=2, metric = 'minkowski', n_jobs = -1)
        scores = cross_val_score(knn, xdata, ydata, cv=cv_)
        mean_score.append( np.mean(scores) )
        
    plt.style.use('bmh')
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.plot(range(J, K), mean_score, color)
    plt.xlabel('Número k de vizinhos')
    plt.ylabel('Score')    
    plt.title(title)
def train(data, tag, k = 50, i = 0, f = 10, passo = 0.2, p_=2):
    
    output = []
    
    testing = data.copy()
    print('Testing for {0}'.format(tag))
    for n in np.arange(i, f, passo):

        test = testing.copy()
        test[tag] = test[tag].apply(lambda x: n*x)
        score = get_score(test, k = k, p_ = p_)
        
        print(round(n, 2), end = ' ')
        
        output.append( score )

    print()
    return (output.index( np.max(output) )*passo + i)
def apply_weight(families, w):

    target = families['Target']
    families = families.drop('Target', axis = 1)
    
    if len( list(families) ) != len(w):
        raise ValueError("Data size {0} and {1} weights given" \
                         .format(len( list(families) ), len(w)))
    
    for i, header in enumerate( list(families) ):
        families[header] = families[header].apply(lambda x: w[i]*x )        
        
    return pd.concat( [families, target], axis=1)
def get_score(data_frame, k = 50, cv_ = 10, p_ = 2):

    ndata_frame = data_frame.fillna(0)
        
    xdata = ndata_frame.drop('Target', axis = 1)
    ydata = ndata_frame.Target
    
    knn = KNeighborsClassifier(n_neighbors = k, p = p_, metric = 'minkowski', n_jobs = -1)
    scores = cross_val_score(knn, xdata, ydata, cv=cv_)
        
    return np.mean(scores)
def normalize(dataframe):
    
    norm_families = pd.DataFrame()
    nplot = fplot.copy()
    
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
    
    norm_families = pd.concat( [norm_families, dataframe['Target']], axis = 1 )
    
    return norm_families, data_mean, data_std
families = pd.read_csv("../input/train.csv",
        sep=r',',
        engine='python',
        na_values='0')
families.head()
families['Target'].value_counts().plot(kind = 'bar')
nfamilies = families.fillna(0).replace('no', 0).replace('yes', 1)

xfamilies = nfamilies.drop('Target', axis = 1).drop('Id', axis=1).drop('idhogar', axis=1)
yfamilies = nfamilies.Target
xfamilies.head()
mean_score_0 = []

J = 10
K = 35

for i in range(J, K):
    knn = KNeighborsClassifier(n_neighbors=i)
    scores = cross_val_score(knn, xfamilies, yfamilies, cv=5)
    mean_score_0.append( np.mean(scores) )
plt.style.use('bmh')
plt.rcParams['figure.figsize'] = (12, 8)

#plt.plot(range(J, K), mean_score_0, 'bo')
fields = ['v2a1', 
          'rooms', 
          'refrig', 
          'tamhog', 
          'escolari', 
          'hhsize', 
          'SQBdependency', 
          'SQBmeaned', 
          'SQBescolari',
          'SQBhogar_total',
          'SQBedjefe',
          'SQBhogar_nin',
          'SQBovercrowding',
          'Target']
plt.style.use('bmh')
plt.rcParams['figure.figsize'] = (12, 8)

#plot_score(families.fillna(0), fields, 'Teste #1',  J=10, K=50)
x_data = []
local = families.copy()

for i in range(1, 21):
    sub_set = local[local.SQBescolari == (i**2)]
    size = sub_set.shape[0]

    data = []

    total = 0
    
    for j in range(1, 5):
        try:
            data.append(sub_set.Target.value_counts()[j])
            total += sub_set.Target.value_counts()[j]
        except KeyError:
            data.append(0)
    
    x_data.append(data)
norm_data = [ [x_data[i][j]/np.sum( x_data[i] ) for i in range(0, 20) ] for j in range(0, 4) ]
for i in range(1, 5):
    plt.plot(range(1, 21), norm_data[i-1], '-o', label = str(i))
    plt.title('Relação entre escolaridade e risco de pobreza')
    plt.xlabel('Anos de estudo')
    plt.ylabel('Classificação de risco')
    plt.legend()
max_income = np.max(local['v2a1'])
max_income
x_data = []
local = families.copy()

max_income = np.max(local['v2a1'])
min_income = np.min(local['v2a1'])

step = (max_income - min_income)/20

for i in np.arange(min_income, max_income, step):
    
    sub_set = local[ (i < local.v2a1) & (local.v2a1 < i + step) ]
    size = sub_set.shape[0]
    
    data = []

    total = 0
    
    if size != 0:
        for j in range(1, 5):
            try:
                data.append(sub_set.Target.value_counts()[j])
                total += sub_set.Target.value_counts()[j]
            except KeyError:
                data.append(0)
            
    else:
        data = [0, 0, 0, 0]       
    
    x_data.append(data)
ises = []

for i in range(0, 20):
    if np.sum( x_data[i] ) != 0:
        ises.append(i)

norm_data = [ [x_data[i][j]/np.sum( x_data[i] ) for i in ises ] for j in range(0, 4) ]
for i in range(1, 5):
    plt.plot(ises, norm_data[i-1], '-o', label = str(i))
    plt.title('Relação entre renda e risco de pobreza')
    plt.xlabel('Intervalo de renda')
    plt.ylabel('Classificação de risco')
    plt.legend()
local['v2a1'].hist(bins=100)
fields = ['v2a1', 
          'rooms', 
          'tamhog', 
          'escolari', 
          'hhsize', 
          'SQBdependency', 
          'SQBmeaned', 
          'SQBescolari',
          'SQBhogar_total',
          'SQBedjefe',
          'SQBhogar_nin',
          'SQBovercrowding',
          'Target']
fplot = families[fields]
#plot_score(fplot.fillna(0), fields, J=20, K=60) 
#plot_score(fplot.fillna(0), fields, J=40, K=80) 
nplot = fplot.copy()
norm_families, means, stds = normalize(nplot)
#plot_score(norm_families.fillna(0), title = 'Teste normalizado', J = 20, K = 60)
#plot_score(norm_families.fillna(0), title = 'Teste normalizado', J = 40, K = 80)
#ws = [1 for i in range( len(list(norm_families)) - 1)]
#wted_data = apply_weight(norm_families, ws)

#for i in range( len(list(norm_families)) - 1):
#    wted_data = apply_weight(norm_families.fillna(0), ws)
#    new_w = train(wted_data, list(norm_families)[i] )
#    print('\n{0}'.format(str(new_w)))
#    ws[i] = new_w
#    print(ws)
#    print(get_score(wted_data))
#    print()
pesos_top = [2.4, 0.0, 5.6, 1.4, 0.0, 3.6, 1.0, 1.2, 8.6, 1.0, 1.0, 1.0]
super_top = apply_weight(norm_families, pesos_top)
get_score(super_top)
#pesos_top = [1.0, 0.2, 6.4, 1.0, 1.4, 1.6, 1.0, 1.0, 3.2, 1.0, 1.0, 1.0]
#super_top = apply_weight(norm_families, pesos_top)
#get_score(super_top)
test_input = pd.read_csv("../input/test.csv",
        sep=r',',
        engine='python',
        na_values='0')
fields = list(norm_families)[:-1]

test = test_input.copy()
test = test[fields]

for i, header in enumerate(fields):
    test[header] = test[header].apply(lambda x: (x-means[i])/stds[i])
    
test = test.fillna(0)
Xtrain = apply_weight(norm_families.fillna(0), pesos_top)

Xtrain = Xtrain.drop('Target', axis=1)
Ytrain = norm_families['Target']

knn = KNeighborsClassifier(n_neighbors = 50,  n_jobs = -1)
knn.fit(Xtrain, Ytrain)
Ytest = knn.predict(test)
output = pd.concat([test_input['Id'], pd.Series(Ytest, name='Target')], axis = 1)
output.head()
output.to_csv('./sumit.csv', index = False)
