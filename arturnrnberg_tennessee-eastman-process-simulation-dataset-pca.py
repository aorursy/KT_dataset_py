%load_ext autoreload

%autoreload 2



%matplotlib inline
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



test_faulty_complete = pyreadr.read_r(test_faulty_path)['faulty_testing']
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

        

        # calculando T2 e Q para dados de treino

        

        # calculando estatística T^2

        T = X@self.P[:,:self.a]

        self.T2_train = np.array([T[i,:]@np.linalg.inv(np.diag(self.L[:self.a]))@T[i,:].T for i in range(X.shape[0])])



        # calculando estatística Q

        e = X - X@self.P[:,:self.a]@self.P[:,:self.a].T

        self.Q_train  = np.array([e[i,:]@e[i,:].T for i in range(X.shape[0])])

        

        

        # plotando variâncias explicadas

        if plot:

            fig, ax = plt.subplots()

            ax.bar(np.arange(len(fv)),fv)

            ax.plot(np.arange(len(fv)),fva)

            ax.set_xlabel('Número de componentes')

            ax.set_ylabel('Variância dos dados')

            ax.set_title('PCA - Variância Explicada');

            

        ###############



    def plot_train_control_charts(self, fault = None):

        '''

        Função para plotar cartas de controle

        '''        

        fig, ax = plt.subplots(1,2, figsize=(15,3))



        ax[0].semilogy(self.T2_train,'.')

        ax[0].axhline(self.T2_lim,ls='--',c='r');

        ax[0].set_title('Carta de Controle $T^2$')

        

        ax[1].semilogy(self.Q_train,'.')

        ax[1].axhline(self.Q_lim,ls='--',c='r')

        ax[1].set_title('Carta de Controle Q')

 

        if fault is not None:

            ax[0].axvline(fault, c='k')

            ax[1].axvline(fault, c='k')

    

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
pca.plot_train_control_charts()
print('Taxa de falsos alarmes no treino\n--------------')



print(f'T2: {(pca.T2_train>pca.T2_lim).sum()/pca.T2_train.shape[0]}')

print(f'Q: {(pca.Q_train>pca.Q_lim).sum()/pca.Q_train.shape[0]}')
pca.predict(df_test)
pca.plot_control_charts()

plt.suptitle('IDV(0)');
print('Taxa de falsos alarmes\n--------------')



print(f'T2: {(pca.T2>pca.T2_lim).sum()/pca.T2.shape[0]}')

print(f'Q: {(pca.Q>pca.Q_lim).sum()/pca.Q.shape[0]}')
df_train = train_normal_complete[(train_normal_complete.simulationRun<=10)].iloc[:,3:]
pca = PCA()

pca.fit(df_train)
def apply_lag(df, lag=1):

       

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



def filter_noise_ma(df, W=5):

    

    import copy

    

    new_df = copy.deepcopy(df)

    

    for column in df:

        new_df[column] = new_df[column].rolling(W).mean()

        

    return new_df.drop(df.index[:W])



def t2_q(pca, fault_number=1, simulation=1, filter_noise=False, W_noise=5, lag=False, lag_columns=1):

    

    df_test = test_faulty_complete[(test_faulty_complete.faultNumber==fault_number) & 

                               (test_faulty_complete.simulationRun==simulation)].iloc[:,3:]

    

    if filter_noise:

        df_test = filter_noise_ma(df_test, W=W_noise)

        

    if lag:

        df_test = apply_lag(df_test, lag_columns)

    

    pca.predict(df_test)

    

    return {'T2': (pca.T2[160:]>pca.T2_lim).sum()/pca.T2[160:].shape[0],

            'Q': (pca.Q[160:]>pca.Q_lim).sum()/pca.Q[160:].shape[0]}
idv_dict = dict()



for i in range(1, 21):

    idv_dict[f'IDV({i})'] = t2_q(pca, fault_number=i, simulation=1)

    pca.plot_control_charts(fault=160)

    plt.suptitle(f'IDV({i})');

    plt.show()

    pca.plot_contributions(fault=160, columns = df_test.columns)

    plt.show()

    

idv_df = pd.DataFrame(idv_dict).T
idv_df[(idv_df.T2 < 0.5) | (idv_df.Q < 0.5)]
pca = PCA()

pca.fit(filter_noise_ma(df_train))
idv_dict = dict()



for i in range(1, 21):

    idv_dict[f'IDV({i})'] = t2_q(pca, fault_number=i, simulation=1, filter_noise=True)

    pca.plot_control_charts(fault=160)

    plt.suptitle(f'IDV({i})');

    plt.show()

    pca.plot_contributions(fault=160, columns = df_test.columns)

    plt.show()

    

idv_df = pd.DataFrame(idv_dict).T
idv_df[(idv_df.T2 < 0.5) | (idv_df.Q < 0.5)]
pca = PCA()

pca.fit(apply_lag(df_train))
idv_dict = dict()



for i in range(1, 21):

    idv_dict[f'IDV({i})'] = t2_q(pca, fault_number=i, simulation=1, lag=True)

    pca.plot_control_charts(fault=160)

    plt.suptitle(f'IDV({i})');

    plt.show()

    pca.plot_contributions(fault=160, columns = apply_lag(df_train).columns)

    plt.show()

    

idv_df = pd.DataFrame(idv_dict).T
idv_df[(idv_df.T2 < 0.5) | (idv_df.Q < 0.5)]