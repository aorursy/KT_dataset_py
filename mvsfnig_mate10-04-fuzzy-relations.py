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
sns.set_style("whitegrid")
# gerando a estrutura pra armazernar o conjunto fyzzye as principais operações necessárias para manipulção
class fuzzy:
    
    def __init__(self,size_set):
        self.x = np.zeros(size_set)
        self.y = np.zeros(size_set)
        self.set = np.zeros((size_set,2))
        self.size = size_set
        
    def set_x(self, conjunto):
        """
        seta o conjunto passado ao conjunto x - dominio
        """
        for i in range(self.x.shape[0]):
            self.x[i] = conjunto[i]
            
    def set_y(self, conjunto):
        """
        seta o conjunto passado ao conjunto y - pertinencias
        """
        for i in range(self.y.shape[0]):
            self.y[i] = conjunto[i]
            
    
    def merge(self,verbose=False):
        """
        junta o conjunto x com y em um unico array
        """
        for i in range(self.x.shape[0]):
            self.set[i][0] = self.x[i]
            self.set[i][1] = self.y[i]
            
        if verbose:
            print(self.set)
        
    def get_y(self, x):
        """
        obtem o valor da pertinencia do valor x passado
        """
        for i in range(self.x.shape[0]):
            if x == self.x[i]:
                return self.y[i]
        return 0
    
    def view_set(self,):
        print(self.set)
# definição dos conjuntos fuzzy - representação por NUMPY ARRAY
A = np.array([[1,0], [2,0.2], [3,0.4], [4,0.6], [5,0.8], [6,1], [7,0.8], [8,0.6],[9,0.4],[10,0.2], [11,0]], dtype=np.float32) # (11, 2)
B = np.array([[5,0], [6,0.2], [7,0.4], [8,0.6], [9,0.8], [10,1], [11,0.8], [12,0.6], [13,0.4], [14,0.2], [15,0]], dtype=np.float32)
# instanciando os conjuntos
a_fuzzy = fuzzy(A.shape[0])
b_fuzzy = fuzzy(B.shape[0])

# definindo os valores de dominio e pertinencia
a_fuzzy.set_x(A[:,0])
a_fuzzy.set_y(A[:,1])

b_fuzzy.set_x(B[:,0])
b_fuzzy.set_y(B[:,1])

# merge sets
a_fuzzy.merge()
b_fuzzy.merge()
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
# visualizando os conjuntos
plot_sets(np.array([a_fuzzy.set,b_fuzzy.set]), ['A', 'B'], 'Conjuntos Fuzzy A e B')

def produto_cartesiano(A,B):
    """
    1. faz o produto carteziano entre o valor das cordenadas
    2. adiciona o minimo valor da pertinencia a coordenada Z
    """
    produto_cartesiano = np.zeros([A.x.shape[0],B.x.shape[0],3], dtype=np.float64)
    
    for i in range(A.x.shape[0]):
        for j in range(B.x.shape[0]):
            
            produto_cartesiano[i][j][0] = A.x[i]
            produto_cartesiano[i][j][1] = B.x[j]
            produto_cartesiano[i][j][2] = min(A.y[i],B.y[j])
            
    return produto_cartesiano
# obtenho um array 3d a partir dos meus conjuntos fuzzy A e B
new_set_3d = produto_cartesiano(a_fuzzy,b_fuzzy)
def plot_conjuntos_3D(X,Y,Z):
    sns
    fig = plt.figure(figsize=(12,8))
    ax = fig.gca(projection='3d')

    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, alpha=0.9, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    # Customize the z axis.
    # ax.set_zlim(-1.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
plot_conjuntos_3D(new_set_3d[:,:,0],new_set_3d[:,:,1],new_set_3d[:,:,2])
def plotar_conjunto_projetado(X, Y, Z):
    
        fig = plt.figure(figsize=(12,8))
        ax = fig.gca(projection='3d')
        
        ax.plot_surface(X, Y, Z, alpha=0.5) #, 

        cset = ax.contourf(X, Y, Z, zdir='z', offset=0, cmap=cm.coolwarm)
        cset = ax.contourf(X, Y, Z, zdir='x', offset=0, cmap=cm.coolwarm)
        cset = ax.contourf(X, Y, Z, zdir='y', offset=0, cmap=cm.coolwarm)

        ax.set_xlabel('X')
        ax.set_xlim(-1, 12)
        ax.set_ylabel('Y')
        ax.set_ylim(0, 20)
        ax.set_zlabel('Z')
        ax.view_init(10, 40)
        ax.set_zlim(0, 1)
        
# obtendo as novas coordenadas selecionado a maior pertinência
plotar_conjunto_projetado(np.copy(new_set_3d[:,:,0]),
                np.copy(new_set_3d[:,:,1]),
                np.copy(new_set_3d[:,:,2]))
def get_extensao_cilindrica(A,B):
    """
    a coordenada Y, vai ser extraida do conjunto B, que A e B ja foram criado como X e Y
    """
    EC = np.zeros([A.x.shape[0],B.x.shape[0],3], dtype=np.float64)
    
    for i in range(A.x.shape[0]):
        for j in range(B.x.shape[0]):
            
            EC[i][j][0] = A.x[i] # coordenada x
            EC[i][j][1] = B.x[j] # coordenada y
            EC[i][j][2] = A.y[i] # pertinencia de A 
            
    return EC   
def plotar_extensao_cilindrica(X, Y, Z):
    
        fig = plt.figure(figsize=(12,8))
        ax = fig.gca(projection='3d')

        ax.plot_surface(X, Y, Z, alpha=0.8) #, 

        cset = ax.contourf(X, Y, Z, zdir='y', offset=0, cmap=cm.coolwarm)

        ax.set_xlabel('X')
        ax.set_xlim(0, 12)
        
        ax.set_ylabel('Y')
        ax.set_ylim(0, 20)
        
        ax.set_zlabel('Z')
        ax.set_zlim(0, 1)
ec = get_extensao_cilindrica(a_fuzzy, b_fuzzy)

plotar_extensao_cilindrica(ec[:,:,0],ec[:,:,1],ec[:,:,2])
