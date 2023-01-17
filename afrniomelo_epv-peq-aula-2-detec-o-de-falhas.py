# comandos mágicos que não se comunicam com a linguagem Python e sim diretamente com o kernel do Jupyter

# começam com %



%load_ext autoreload

%autoreload 2



%matplotlib inline
# importando os principais módulos que usaremos ao longo da aula



# a versão 0.3.4, lançada em 14/09/2020, não funcionou neste kernel 

!pip install pyreadr==v0.3.3



import pyreadr

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
%%time



# troque aqui pela localização do dataset na sua máquina

PATH = '/kaggle/input/tennessee-eastman-process-simulation-dataset/'



train_normal_path = PATH+'TEP_FaultFree_Training.RData'

train_faulty_path = PATH+'TEP_Faulty_Training.RData'



test_normal_path = PATH+'TEP_FaultFree_Testing.RData'

test_faulty_path = PATH+'TEP_Faulty_Testing.RData'



train_normal_complete = pyreadr.read_r(train_normal_path)['fault_free_training']

#train_faulty_complete = pyreadr.read_r(train_fault_path)['faulty_training']



#test_normal_complete = pyreadr.read_r(test_normal_path)['fault_free_testing']

test_faulty_complete = pyreadr.read_r(test_faulty_path)['faulty_testing']
train_normal_complete
test_faulty_complete
df_train = train_normal_complete[train_normal_complete.simulationRun==1].iloc[:,3:]



df_test = test_faulty_complete[(test_faulty_complete.simulationRun==1)&

                               (test_faulty_complete.faultNumber==1)].iloc[:,3:]
fig, ax = plt.subplots(13,4,figsize=(30,90))



for i in range(df_train.shape[1]):

    

    x = df_train.iloc[:,i]

    

    mean  = x.mean()

    std = x.std(ddof=1)

    

    LCL = mean-3*std

    UCL = mean+3*std

    

    x.plot(ax=ax.ravel()[i]) 



    ax.ravel()[i].legend();

    

    ax.ravel()[i].axhline(mean,c='k')

    ax.ravel()[i].axhline(LCL,ls='--',c='r')

    ax.ravel()[i].axhline(UCL,ls='--',c='r')
fig, ax = plt.subplots(13,4,figsize=(30,70))



for i in range(df_train.shape[1]):

    

    x = df_train.iloc[:,i]

    x_ts = df_test.iloc[:,i]



    mean  = x.mean()

    std = x.std(ddof=1)

    

    LCL = mean-3*std

    UCL = mean+3*std

    

    x_ts.plot(ax=ax.ravel()[i]) 



    ax.ravel()[i].legend();

    

    ax.ravel()[i].axhline(mean,c='k')

    ax.ravel()[i].axhline(LCL,ls='--',c='r')

    ax.ravel()[i].axhline(UCL,ls='--',c='r')

    

    ax.ravel()[i].axvline(160,c='g')
# adaptado de https://stackoverflow.com/a/38705297/11439214



from scipy.stats import multivariate_normal

from mpl_toolkits.mplot3d import Axes3D



#Parameters to set

mu_x = 0

variance_x = 1



mu_y = 0

variance_y = 5



#Create grid and multivariate normal

x = np.linspace(-3,3,500)

y = np.linspace(-6,6,500)

X, Y = np.meshgrid(x,y)

pos = np.empty(X.shape + (2,))

pos[:, :, 0] = X; pos[:, :, 1] = Y

rv = multivariate_normal([mu_x, mu_y], [[variance_x, 0], [0, variance_y]])



fig = plt.figure(figsize=(22,6))



#Plotting 3d

ax = fig.add_subplot(121, projection='3d')

ax.plot_surface(X, Y, rv.pdf(pos), cmap='viridis', linewidth=0)

ax.set_xlabel('$x_1$')

ax.set_ylabel('$x_2$')

ax.set_zlabel('$p(\mathbf{x})$')



# Plotting contour

ax = fig.add_subplot(122)

ax.contourf(X, Y, rv.pdf(pos) ,cmap='viridis')

ax.set_xlabel('$x_1$')

ax.set_ylabel('$x_2$');
# exemplo adaptado de https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html



rng = np.random.RandomState(1)

X = np.dot(rng.rand(2, 2), rng.randn(2, 200)).T



plt.scatter(X[:, 0], X[:, 1])

plt.axis('equal')

plt.xlabel('x1')

plt.ylabel('x2');
import sklearn.decomposition 



pca = sklearn.decomposition.PCA(n_components=2)

pca.fit(X)
print('componentes:')

print(pca.components_)



print('\nvariâncias explicadas:')

print(pca.explained_variance_)
def draw_vector(v0, v1, ax=None):

    ax = ax or plt.gca()

    arrowprops=dict(arrowstyle='->',

                    linewidth=2,

                    shrinkA=0, shrinkB=0)

    ax.annotate('', v1, v0, arrowprops=arrowprops)
plt.scatter(X[:, 0], X[:, 1], alpha=0.3)



for length, vector in zip(pca.explained_variance_, pca.components_):

    v = vector * 3 * np.sqrt(length)

    draw_vector(pca.mean_, pca.mean_ + v)

    

plt.axis('equal')

plt.xlabel('x1')

plt.ylabel('x2');
T = pca.transform(X)
plt.scatter(T[:, 0], T[:, 1], alpha=0.3)

plt.axis('equal')

plt.xlabel('t1')

plt.ylabel('t2');
pca.components_.round(3)
pca = sklearn.decomposition.PCA(n_components=1)

pca.fit(X)



T = pca.transform(X)
display(X.shape)

display(T.shape)
plt.plot(T, np.zeros(len(T)),'.')

plt.gca().get_yaxis().set_visible(False)

plt.xlabel('t1');
X_reconstruido = pca.inverse_transform(T)
plt.scatter(X[:, 0], X[:, 1], alpha=0.2, label = 'original')

plt.scatter(X_reconstruido[:, 0], X_reconstruido[:, 1], alpha=0.8, label='reconstrução')

plt.legend()

plt.axis('equal')

plt.xlabel('x1')

plt.ylabel('x2');
display(X.shape)

display(T.shape)

display(X_reconstruido.shape)
class PCA():

   

    ###############



    def __init__ (self, a = 0.9):

        '''

        Construtor: função chamada toda vez que um objeto PCA é inicializado

        '''

        # se 0<=a<1,  'a' indica a fraçao de variancia explicada desejada

        # se a>=1,    'a' indica o numero de componentes desejado

        self.a = a

   

    ###############



    def fit(self, X, conf_Q = 0.99, conf_T2 = 0.99, plot = True):

        '''

        Função para treino do modelo

        '''       

        # guardando médias e desvios-padrão do treino

        self.mu_train = X.mean(axis=0)

        self.std_train = X.std(axis=0)        

       

        # normalizando dados de treino

        X = np.array(((X - self.mu_train)/self.std_train))

       

        # calculando a matriz de covariâncias dos dados

        Cx = np.cov(X, rowvar=False)

        

        # aplicando decomposição em autovalores e autovetores

        self.L, self.P = np.linalg.eig(Cx)

        

        # frações da variância explicada

        fv = self.L/np.sum(self.L)

        

        # frações da variância explicada acumuladas

        fva = np.cumsum(self.L)/sum(self.L)

       

        # definindo número de componentes

        if self.a>0 and self.a<1:

            self.a = np.where(fva>self.a)[0][0]+1 

            

        # calculando limites de detecção



        # limite da estatística T^2

        from scipy.stats import f

        F = f.ppf(conf_T2, self.a, X.shape[0]-self.a)

        self.T2_lim = ((self.a*(X.shape[0]**2-1))/(X.shape[0]*(X.shape[0]-self.a)))*F

        

        # limite da estatística Q

        theta = [np.sum(self.L[self.a:]**(i)) for i in (1,2,3)]

        ho = 1-((2*theta[0]*theta[2])/(3*(theta[1]**2)))

        from scipy.stats import norm

        nalpha = norm.ppf(conf_Q)

        self.Q_lim = (theta[0]*(((nalpha*np.sqrt(2*theta[1]*ho**2))/theta[0])+1+

                                ((theta[1]*ho*(ho-1))/theta[0]**2))**(1/ho))

        

        # plotando variâncias explicadas

        if plot:

            fig, ax = plt.subplots()

            ax.bar(np.arange(len(fv)),fv)

            ax.plot(np.arange(len(fv)),fva)

            ax.set_xlabel('Número de componentes')

            ax.set_ylabel('Variância dos dados')

            ax.set_title('PCA - Variância Explicada');



    ###############

            

    def predict(self, X):

        '''

        Função para teste do modelo

        '''

            

        # normalizando dados de teste (usando os parâmetros do treino!)

        X = np.array((X - self.mu_train)/self.std_train)



        # calculando estatística T^2

        T = X@self.P[:,:self.a]

        self.T2 = np.array([T[i,:]@np.linalg.inv(np.diag(self.L[:self.a]))@T[i,:].T for i in range(X.shape[0])])



        # calculando estatística Q

        e = X - X@self.P[:,:self.a]@self.P[:,:self.a].T

        self.Q  = np.array([e[i,:]@e[i,:].T for i in range(X.shape[0])])

        

        # calculando contribuições para Q

        self.c = np.absolute(X*e) 

                

    ###############



    def plot_control_charts(self, fault = None):

        '''

        Função para plotar cartas de controle

        '''        

        fig, ax = plt.subplots(1,2, figsize=(15,3))



        ax[0].semilogy(self.T2,'.')

        ax[0].axhline(self.T2_lim,ls='--',c='r');

        ax[0].set_title('Carta de Controle $T^2$')

        

        ax[1].semilogy(self.Q,'.')

        ax[1].axhline(self.Q_lim,ls='--',c='r')

        ax[1].set_title('Carta de Controle Q')

 

        if fault is not None:

            ax[0].axvline(fault, c='k')

            ax[1].axvline(fault, c='k')



    ###############

            

    def plot_contributions(self, fault = None, 

                           index = None, 

                           columns = None):

        '''

        Função para plotar mapas de contribuição

        '''

        fig, ax = plt.subplots(figsize=(20, 6))

        

        c = pd.DataFrame(self.c, 

                         index = index,

                         columns = columns)

    

        sns.heatmap(c, ax = ax, 

                    yticklabels=int(self.c.shape[0]/10),

                    cmap = plt.cm.Blues);

        

        ax.set_title('Contribuições parciais para Q')

        

        if fault is not None:

            ax.axhline(y=c.index[fault],

                       ls='--', c='k')
df_train = train_normal_complete[(train_normal_complete.simulationRun>=1)&

                                 (train_normal_complete.simulationRun<5)].iloc[:,3:]



df_test = train_normal_complete[(train_normal_complete.simulationRun>5)&

                                (train_normal_complete.simulationRun<10)].iloc[:,3:]
pca = PCA(a = 0.9)

pca.fit(df_train)
pca.a
pca.predict(df_test)
pca.plot_control_charts()

plt.suptitle('IDV(0)');
print('Taxa de falsos alarmes\n--------------')



print(f'T2: {(pca.T2>pca.T2_lim).sum()/pca.T2.shape[0]}')

print(f'Q: {(pca.Q>pca.Q_lim).sum()/pca.Q.shape[0]}')
IDV = 1



df_test = test_faulty_complete[(test_faulty_complete.faultNumber==IDV) & 

                               (test_faulty_complete.simulationRun==1)].iloc[:,3:]



pca.predict(df_test)



pca.plot_control_charts(fault=160)

plt.suptitle(f'IDV({IDV})');



pca.plot_contributions(fault=160, columns = df_test.columns)
print(f'Taxas de detecção de falhas - IDV({IDV})\n--------------')



print(f'T2: {(pca.T2[160:]>pca.T2_lim).sum()/pca.T2[160:].shape[0]}')

print(f'Q: {(pca.Q[160:]>pca.Q_lim).sum()/pca.Q[160:].shape[0]}')
IDV = 4



df_test = test_faulty_complete[(test_faulty_complete.faultNumber==IDV) & 

                               (test_faulty_complete.simulationRun==1)].iloc[:,3:]



pca.predict(df_test)



pca.plot_control_charts(fault=160)

plt.suptitle(f'IDV({IDV})');



pca.plot_contributions(fault=160, columns = df_test.columns)



print(f'Taxas de detecção de falhas - IDV({IDV})\n--------------')



print(f'T2: {(pca.T2[160:]>pca.T2_lim).sum()/pca.T2[160:].shape[0]}')

print(f'Q: {(pca.Q[160:]>pca.Q_lim).sum()/pca.Q[160:].shape[0]}')
IDV = 11



df_test = test_faulty_complete[(test_faulty_complete.faultNumber==IDV) & 

                               (test_faulty_complete.simulationRun==1)].iloc[:,3:]



pca.predict(df_test)



pca.plot_control_charts(fault=160)

plt.suptitle(f'IDV({IDV})');



pca.plot_contributions(fault=160, columns = df_test.columns)



print(f'Taxas de detecção de falhas - IDV({IDV})\n--------------')



print(f'T2: {(pca.T2[160:]>pca.T2_lim).sum()/pca.T2[160:].shape[0]}')

print(f'Q: {(pca.Q[160:]>pca.Q_lim).sum()/pca.Q[160:].shape[0]}')
def apply_lag (df, lag = 1):

       

    from statsmodels.tsa.tsatools import lagmat

    array_lagged = lagmat(df, maxlag=lag,

                          trim="forward", original='in')[lag:,:]  

    new_columns = []

    for l in range(lag):

        new_columns.append(df.columns+'_lag'+str(l+1))

    columns_lagged = df.columns.append(new_columns)

    index_lagged = df.index[lag:]

    df_lagged = pd.DataFrame(array_lagged, index=index_lagged,

                             columns=columns_lagged)

       

    return df_lagged 
exemplo = pd.DataFrame([[1.,10.],[2.,20.],[3.,30.],[4.,40.]],columns=['A','B'])

exemplo
apply_lag(exemplo)
apply_lag(exemplo, lag=2)
lag = 1



pca.fit(apply_lag(df_train, lag=lag))



IDV = 11



df_test = test_faulty_complete[(test_faulty_complete.faultNumber==IDV) & 

                               (test_faulty_complete.simulationRun==1)].iloc[:,3:]



pca.predict(apply_lag(df_test, lag=lag))



pca.plot_control_charts(fault=160-lag)

plt.suptitle(f'IDV({IDV})');



pca.plot_contributions(fault=160-lag, columns = apply_lag(df_test, lag=lag).columns)



print(f'Taxas de detecção de falhas - IDV({IDV})\n--------------')



print(f'T2: {(pca.T2[160-lag:]>pca.T2_lim).sum()/pca.T2[160-lag:].shape[0]}')

print(f'Q: {(pca.Q[160-lag:]>pca.Q_lim).sum()/pca.Q[160-lag:].shape[0]}')
def filter_noise_ma (df, W = 5):



    import copy

    

    new_df = copy.deepcopy(df)



    for column in df:

        new_df[column] = new_df[column].rolling(W).mean()

        

    return new_df.drop(df.index[:W])
fig, ax = plt.subplots(1,5, figsize=(20,3))



df_test['xmv_10'].plot(ax=ax[0])

ax[0].set_title('Sem filtro')



i = 1



for W in [5,10,15,20]:

    filter_noise_ma(pd.DataFrame(df_test['xmv_10']), W=W).plot(ax=ax[i], legend=False)

    ax[i].set_title(f'W={W}')

    i+=1
W = 5



pca.fit(filter_noise_ma(df_train, W=W))



IDV = 11



df_test = test_faulty_complete[(test_faulty_complete.faultNumber==IDV) & 

                               (test_faulty_complete.simulationRun==1)].iloc[:,3:]



pca.predict(filter_noise_ma(df_test, W=W))



pca.plot_control_charts(fault=160-W)

plt.suptitle(f'IDV({IDV})');



pca.plot_contributions(fault=160-W, columns = filter_noise_ma(df_test, W=W).columns)



print(f'Taxas de detecção de falhas - IDV({IDV})\n--------------')



print(f'T2: {(pca.T2[160-W:]>pca.T2_lim).sum()/pca.T2[160-W:].shape[0]}')

print(f'Q: {(pca.Q[160-W:]>pca.Q_lim).sum()/pca.Q[160-W:].shape[0]}')