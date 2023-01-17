# Libs
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

print('A', A)
print('B', B)

# definindo meus conjuntos fuzzy
vendas_baixa = fuzzy(A.size)
vendas_media = fuzzy(B.size)

# setando o domínio
vendas_baixa.set_x(A)
vendas_media.set_x(B)

# calcula a pertinência dos conjuntos com a função de pertinência tringular 
vendas_baixa.triangular(a=A[0], m=A[5], b=A[len(A)-1])
vendas_media.triangular(a=B[0], m=B[5], b=B[len(A)-1]) # trapezoidal

# visualizando os conjuntos
plot_sets(np.array([vendas_baixa.set,vendas_media.set]), ['Baixa', 'Média'], ' Vendas')
def get_coord_alpha_cut(a,b,m,pert):
    """
    return new a' and b' do alphacut
    """
    new_a_row = (pert * (m-a)) + a
    new_b_row = ((pert * (b-m))-b) * -1
    
    return new_a_row, m, new_b_row
vendas_baixa.set[0]
alpha_cut = 0.5

a_a,a_m, a_b = get_coord_alpha_cut(a=vendas_baixa.set[0],
                                   b=vendas_baixa.set[len(vendas_baixa.x)-1],
                                   m=vendas_baixa.set[int((len(vendas_baixa.x)-1)/2)],
                                   pert=alpha_cut)

b_a,b_m, b_b = get_coord_alpha_cut(a=vendas_media.set[0],
                                   b=vendas_media.set[len(vendas_media.x)-1],
                                   m=vendas_media.set[int((len(vendas_media.x)-1)/2)],
                                   pert=alpha_cut)


region_alpha_cut_baixa = np.array([a_a,a_m,a_b,a_a], dtype=np.float32)

region_alpha_cut_media = np.array([b_a,b_m,b_b,b_a], dtype=np.float32)

print('alphacut of A >\n', region_alpha_cut_baixa)
print('alphacut of B >\n', region_alpha_cut_media)
# cojunto dos cojuntos
conjuntos = np.array([vendas_baixa.set,vendas_media.set,region_alpha_cut_baixa,region_alpha_cut_media])

# plot sets
plot_sets(conjuntos, ['Baixa','Media','a cut B','a cut M'], ' Vendas ')
def intervalo_soma(a,b):
    """
    soma [1:soma de a+b, m, soma de b+c]
    """
    value_min = (a[0,0] + b[0,0]) 
    value_max = (a[len(a)-1,0]  + b[len(b)-1,0])
    m = (value_min + value_max) / 2
    
    soma = np.array([ [value_min, min(a[0,1],b[0,1])],
                       [m, 1],
                      [value_max, min(a[len(a)-1,1], b[len(b)-1,1])],
                     [value_min, min(a[0,1],b[0,1])],
                    ], dtype=np.float32)
    return soma
# calcule sets sum alpha cut of a and b
conjunto_soma = intervalo_soma(np.copy(region_alpha_cut_baixa[0:len(region_alpha_cut_baixa)-1]),
                               np.copy(region_alpha_cut_media[0:len(region_alpha_cut_baixa)-1]))
conjunto_soma
# cojunto dos cojuntos
conjuntos = np.array([vendas_baixa.set,vendas_media.set, conjunto_soma])

# plot sets
plot_sets(conjuntos, ['Baixa','Media', 'Soma'], ' Vendas ', 'lower right' )
def intervalo_subtracion(a,b):
    """
    subtracion [1:subtracion de a+b, m, subtracion de b+c]
    a = a[0,0]
    b = a[len(a)-1,0]
    c = b[0,0]
    d = b[len(b)-1,0]
    """
    value_min = abs(a[0,0] - b[len(b)-1,0]) 
    value_max = abs(b[0,0] - a[len(a)-1,0])
    m = abs(value_min + value_max) / 2
    
    subtracion = np.array([ [value_min, min(a[0,1],b[0,1])],
                            [m, 1],
                            [value_max, min(a[len(a)-1,1], b[len(b)-1,1])],
                            [value_min, min(a[0,1],b[0,1])],
                          ],dtype=np.float32)
    return subtracion
intervalo_subtracion =  intervalo_subtracion(np.copy(region_alpha_cut_baixa[0:len(region_alpha_cut_baixa)-1]),
                                             np.copy(region_alpha_cut_media[0:len(region_alpha_cut_media)-1]))

# cojunto dos cojuntos
conjuntos = np.array([vendas_baixa.set,vendas_media.set, intervalo_subtracion])

# plot sets
plot_sets(conjuntos, ['Baixa','Media', 'sub'], ' Vendas ', 'upper right' )
def intervalo_multiplication(a,b):
    """
    a = a[0,0]
    b = a[len(a)-1,0]
    c = b[0,0]
    d = b[len(b)-1,0]
    """
    
    value_min = min((a[0,0]*b[0,0]), (a[0,0]*b[len(b)-1,0]), (a[len(a)-1,0]*b[0,0]), (a[len(a)-1,0]*b[len(b)-1,0])) 
    value_max = max((a[0,0]*b[0,0]), (a[0,0]*b[len(b)-1,0]), (a[len(a)-1,0]*b[0,0]), (a[len(a)-1,0]*b[len(b)-1,0])) 
    m = abs(value_min + value_max) / 2
    
    mul = np.array([ [value_min, min(a[0,1],b[0,1])],
                     [m, 1],
                     [value_max, min(a[len(a)-1,1], b[len(b)-1,1])],
                     [value_min, min(a[0,1],b[0,1])]
                   ], dtype=np.float32)
    return mul
intervalo_mul =  intervalo_multiplication(np.copy(region_alpha_cut_baixa[0:len(region_alpha_cut_baixa)-1]),
                                             np.copy(region_alpha_cut_media[0:len(region_alpha_cut_media)-1]))
intervalo_mul

# cojunto dos cojuntos
conjuntos = np.array([vendas_baixa.set,vendas_media.set, intervalo_mul])

# plot sets
plot_sets(conjuntos, ['Baixa','Media', 'mul'], ' Vendas ', 'lower right' )
def intervalo_divisao(a,b):
    """
    a = a[0,0]
    b = a[len(a)-1,0]
    c = b[0,0]
    d = b[len(b)-1,0]
    """
    
    value_min = min((a[0,0]/b[0,0]), (a[0,0]/b[len(b)-1,0]), (a[len(a)-1,0]/b[0,0]), (a[len(a)-1,0]/b[len(b)-1,0])) 
    value_max = max((a[0,0]/b[0,0]), (a[0,0]/b[len(b)-1,0]), (a[len(a)-1,0]/b[0,0]), (a[len(a)-1,0]/b[len(b)-1,0])) 
    m = abs(value_min + value_max) / 2
    
    div = np.array([ [value_min, min(a[0,1],b[0,1])],
                     [m, 1],
                     [value_max, min(a[len(a)-1,1], b[len(b)-1,1])],
                     [value_min, min(a[0,1],b[0,1])]
                   ], dtype=np.float32)
    return div
intervalo_div =  intervalo_divisao(np.copy(region_alpha_cut_baixa[0:len(region_alpha_cut_baixa)-1]),
                                   np.copy(region_alpha_cut_media[0:len(region_alpha_cut_media)-1]))
intervalo_div

# cojunto dos cojuntos
conjuntos = np.array([vendas_baixa.set,vendas_media.set, intervalo_div])

# plot sets
plot_sets(conjuntos, ['Baixa','Media', 'div'], ' Vendas ', 'upper right' )