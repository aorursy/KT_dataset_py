# define libs
import matplotlib.pyplot as plt                                    # visualization
from mpl_toolkits.mplot3d import Axes3D 
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib.collections import PolyCollection
from matplotlib.colors import colorConverter
import seaborn as sns    
import numpy as np # vector and matrizes

%matplotlib inline
sns.set()
class functions_pertinence:
    
        def __init__(self,):
            """
            funções de pertinência
            """
    
        def triangular(self, a,m,b,x, max_degree=None):
            """
            define pertinence y of the triangular function from x values
            a - primeiro valor conjunto
            m - valor mediano do conjunto
            b - último valor do conjunto
            x - valor a ser calculado
            """

            if x <= a or x >= b:
                return 0

            if x == m and max_degree != None:
                return max_degree

            if x > a and x <= m:
                return ((x-a)/(m-a))

            if x > m and x < b:
                return ((b-x)/(b-m))
            
        def trapezoidal(self, a,m,n,b,x,max_degree = None):
            """
            define pertinence y of the trapezoidal function from x values
            a - primeiro valor conjunto
            m - primeiro valor com maximo valor de pertinência
            n - primeiro valor com maximo valor de pertinência
            b - último valor do conjunto
            x - valor a ser calculado
            """
                
            if max_degree is None:
                max_degree = 1

            if x <= a or x >= b:
                return 0
            
            if x >= m and x <= n:
                return max_degree

            if x > a and x < m:
                return (x-a)/(m-a)
    
            if x > n and x < b:
                return (b-x)/(b-n)
        
        def gaussiana(self, x, m,sigma,max_degree = None):
            """
            define pertinence y of the trapezoidal function from x values
            m: valor médio
            sigma: siga
            
            x: valor a ser calculado
            """
            return np.exp(-((x-m)**2)/(sigma**2))
            
    
            
            
# gerando a estrutura pra armazernar o conjunto fyzzye as principais operações necessárias para manipulção
class fuzzy:
    
    def __init__(self,size_set):
        self.x = np.zeros(size_set)
        self.y = np.zeros(size_set)
        self.set = np.zeros((size_set,2))
        self.size = size_set
        self.calculo = functions_pertinence()
        
        
    def set_x(self, conjunto):
        """
        seta o conjunto passado ao conjunto x - dominio
        """
        for i in range(self.x.shape[0]):
            self.x[i] = conjunto[i]
    
    def merge(self,verbose=False):
        """
        junta o conjunto x com y em um unico array
        """
        for i in range(self.x.shape[0]):
            self.set[i][0] = self.x[i]
            self.set[i][1] = self.y[i]
            
        if verbose:
            print(self.set)
            
    def set_y(self, conjunto):
        """
        seta o conjunto passado ao conjunto y - pertinencias
        """
        for i in range(self.y.shape[0]):
            self.y[i] = conjunto[i]
            
        self.merge()
        
    
    def triangular(self, a,m,b, max_degree=None):
        """
        define o conjunto x com a funcão de pertinẽncia triangular
        """
        for i in range(self.x.shape[0]):
            self.y[i] = self.calculo.triangular(a,m,b,self.x[i])
        self.merge()
        
    def trapezoidal(self, a,m,n,b, max_degree=None):
        """
        define o conjunto x com a funcão de pertinência trapezoidal
        """
        for i in range(self.x.shape[0]):
            self.y[i] = self.calculo.trapezoidal(a,m,n,b,self.x[i])
        self.merge()
    
    def gaussiana(self, m,sigma=0.5, max_degree=None):
        """
        define o conjunto x com a funcão de pertinência trapezoidal
        """
        
        for i in range(self.x.shape[0]):
            self.y[i] = self.calculo.gaussiana(self.x[i],m,sigma)
        self.merge()
    
        
    def get_y(self, x):
        """
        obtem o valor da pertinencia do valor x passado - WARNING obter posição do x
        """
        for i in range(self.x.shape[0]):
            if x == self.x[i]:
                return self.y[i]
        return 0
        
    
    def view_set(self,):
        print(self.set)
# visualização
def plot_sets(conjuntos, labels, title=None, position=None):
    """
     - função para plotagem dos conjuntos 2d
     -- plota os conjuntos em um mesmo plano
    """
    
    color = ['--bo','--go', '--ro', '--yo', '--po']
    facecolors = [cm.jet(x) for x in np.random.rand(20)]
    
    fig, ax = plt.subplots(figsize=(8,5))

    indice = 0
    for i in conjuntos:
            plt.plot(i[:,0], i[:,1], color[indice], label=labels[indice])
            plt.fill_between(i[:,0], i[:,1], facecolors=facecolors[indice], alpha=0.4)
            indice += 1 
            
    if position:
        legend = ax.legend(loc=position, shadow=True, fontsize='x-large')
    else:
        legend = ax.legend(loc='upper left', shadow=True, fontsize='x-large')

    # Put a nicer background color on the legend.
    legend.get_frame().set_facecolor('#00FFCC')
    plt.title(title)
    plt.grid(True)
# definição dos conjuntos fuzzy - representação por NUMPY ARRAY
A = np.arange(1,12)
B = np.arange(5,16)

# A = np.arange(1,50)
# B = np.arange( 55,100)

print('A', A)
print('B', B)
# definindo meus conjuntos fuzzy
vendas_baixa = fuzzy(A.size)
vendas_media = fuzzy(B.size)
# setando o domínio
vendas_baixa.set_x(A)
vendas_media.set_x(B)
vendas_baixa.x
# setando o domínio
vendas_baixa.triangular(a=A[0], m=A[5], b=A[len(A)-1])
vendas_media.triangular(a=B[0], m=B[5], b=B[len(A)-1]) # trapezoidal

# visualizando os conjuntos
plot_sets(np.array([vendas_baixa.set,vendas_media.set]), ['Baixa', 'Média'], ' Vendas')

vendas_baixa.trapezoidal(a=A[0], m=A[3], n=A[7],b=A[len(A)-1])
vendas_media.trapezoidal(a=B[0], m=B[3], n=B[7], b=B[len(A)-1]) # trapezoidal
plot_sets(np.array([vendas_baixa.set,vendas_media.set]), ['Baixa', 'Média'], 'Vendas')
vendas_baixa.gaussiana(m=A[int(len(A)/2)], sigma=0.1)
vendas_media.gaussiana(m=B[int(len(B)/2)]) 
plot_sets(np.array([vendas_baixa.set,vendas_media.set]), ['Baixa', 'Média'], 'Vendas')