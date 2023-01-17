import numpy as np               # linear algebra - vectors and matrices
import matplotlib.pyplot as plt  # visualization
import matplotlib.cm as cm       # visualization
import seaborn as sns            # visualization
%matplotlib inline
sns.set()
def get_pertinence_triangle(a,m,b,x, max_degree=None):
    """
    valores do conjuntos x
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

# obter o M do conjunto triangular
def get_m(conj):
    return float((conj[0] + conj[len(conj)-1])/2)
# conjuntos x crisp
A = list(range(1, 6))
B = list(range(3, 8))
print('A : ', A)
print('B : ', B)
# gerando a estrutura pra armazernar o conjunto fyzzy
class fuzzy:
    
    def __init__(self,size_set):
        self.x = np.zeros(size_set)
        self.y = np.zeros(size_set)
        self.set = np.zeros((size_set,2))
        self.size = size_set
        
    def set_x(self, conjunto):
        for i in range(self.x.shape[0]):
            self.x[i] = conjunto[i]
    
    def merge(self,verbose=False):
        
        for i in range(self.x.shape[0]):
            self.set[i][0] = self.x[i]
            self.set[i][1] = self.y[i]
            
        if verbose:
            print(self.set)
        
    def get_y(self, x):
        for i in range(self.x.shape[0]):
            if x == self.x[i]:
                return self.y[i]
        return 0
    
    def view_set(self,):
        print(self.set)
    
        
        
# instanciando o conjunto
a_fuzzy = fuzzy(len(A))
b_fuzzy = fuzzy(len(B))
# obtendo as pertinências para os valores de x a adicionando no conjunto a_fuzzy
for i in range(len(A)):
    a_fuzzy.x[i] = A[i]
    a_fuzzy.y[i] = get_pertinence_triangle(A[0], get_m(A), A[len(A)-1], A[i])

for i in range(len(B)):
    b_fuzzy.x[i] = B[i]
    b_fuzzy.y[i] = get_pertinence_triangle(B[0], get_m(B), B[len(B)-1], B[i])

a_fuzzy.merge(True)
b_fuzzy.merge(True)
# função para plotagem dos conjuntos    
def plot_sets(conjuntos, labels, title=None, position=None, color_sets=None):
    """
     - função para plotagem dos conjuntos 2d
     -- plota os conjuntos em um mesmo plano
    """
    
    color = ['--bo','--go', '--ro', '--yo', '--po']
    facecolors = [cm.jet(x) for x in np.random.rand(20)]
    
    fig, ax = plt.subplots(figsize=(8, 5))

    indice = 0
    for i in conjuntos:
            plt.plot(i[:,0], i[:,1], color[indice], label=labels[indice])
            if color_sets:
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


conjuntos = np.array([a_fuzzy.set,b_fuzzy.set])
plot_sets(conjuntos, ['A', 'B'], 'Conjuntos Fuzzy A e B')
# função de uniao com operador máximo
def intersection_fuzzy(cfa,cfb):
    
    chaves = list(set(cfa.x) & set(cfb.x))
    
    fuzzy_inter = fuzzy(len(chaves))
    fuzzy_inter.set_x(chaves)
    
    indice = 0
    for i in chaves:
        
        fuzzy_inter.y[indice] = min(cfa.get_y(i), cfb.get_y(i))
        indice += 1
    fuzzy_inter.merge()    
    return fuzzy_inter

fuzzy_inter = intersection_fuzzy(a_fuzzy, b_fuzzy)
fuzzy_inter.merge()
conjuntos = np.array([a_fuzzy.set, b_fuzzy.set,fuzzy_inter.set])
plot_sets(conjuntos, ['A', 'B', 'A ∩ B'], 'Intersecção entre A e B')
# função de uniao com operador mínimo

def union_fuzzy(cfa,cfb):
    chaves = list(set(cfa.x).union(cfb.x))
    
    fuzzy_union = fuzzy(len(chaves))
    fuzzy_union.set_x(chaves)
    
    indice = 0
    for i in chaves:
        fuzzy_union.y[indice] = max(cfa.get_y(i), cfb.get_y(i))
        indice += 1
    
    fuzzy_union.merge()
    return fuzzy_union

fuzzy_union = union_fuzzy(a_fuzzy, b_fuzzy)
fuzzy_union.merge()
conjuntos = np.array([a_fuzzy.set, b_fuzzy.set,fuzzy_union.set])
plot_sets(conjuntos, ['A', 'B', 'A U B'], 'União entre A e B')
def complemento_fuzzy(conjunto):
    complemento = fuzzy(conjunto.size)
    complemento.set_x(conjunto.x)
    for i in range(conjunto.size):
        complemento.y[i] = 1-conjunto.y[i]
    complemento.merge()
    return complemento
comp_a = complemento_fuzzy(a_fuzzy)
comp_b = complemento_fuzzy(b_fuzzy)
comp_a.y
conjuntos = np.array([a_fuzzy.set, comp_a.set, b_fuzzy.set, comp_b.set])
plot_sets(conjuntos, ['A', "'A", 'B', "'B"], 'Complemento de A e B')
def intersection_fuzzy_produto_algebrico(cfa,cfb):
    
    chaves = list(set(cfa.x) & set(cfb.x))
    
    fuzzy_pa = fuzzy(len(chaves))
    fuzzy_pa.set_x(chaves)
    
    indice = 0
    for i in chaves:
        
        fuzzy_pa.y[indice] = (cfa.get_y(i) * cfb.get_y(i))
        indice += 1
    
    fuzzy_pa.merge()
    return fuzzy_pa
    
fuzzy_pa = intersection_fuzzy_produto_algebrico(a_fuzzy, b_fuzzy)

plot_sets(np.array([a_fuzzy.set, b_fuzzy.set, fuzzy_pa.set]),
                   ['A', 'B', "PA"],
                   'Produto Algébrico A e B')
def intersection_fuzzy_diferenca_limitada(cfa,cfb):
    
    chaves = list(set(cfa.x) & set(cfb.x))
    
    new_fuzzy = fuzzy(len(chaves))
    new_fuzzy.set_x(chaves)
    
    indice = 0
    for i in chaves:
        
        new_fuzzy.y[indice] = max(0, ((cfa.get_y(i) + cfb.get_y(i))-1))
        indice += 1
    
    new_fuzzy.merge()
    return new_fuzzy
    
fuzzy_dl = intersection_fuzzy_diferenca_limitada(a_fuzzy, b_fuzzy)

plot_sets(np.array([a_fuzzy.set, b_fuzzy.set, fuzzy_dl.set]),
                   ['A', 'B', "DL"],
                   'Diferença Limitada entre A e B')
def intersection_fuzzy_drastica(cfa,cfb):
    
    chaves = list(set(cfa.x) & set(cfb.x))
    
    new_fuzzy = fuzzy(len(chaves))
    new_fuzzy.set_x(chaves)
    
    indice = 0
    for i in chaves:
        
        if cfa.get_y(i) == 1:
            new_fuzzy.y[indice] = cfb.get_y(i)
            
        elif cfb.get_y(i) == 1:
            new_fuzzy.y[indice] = cfa.get_y(i)
        else:
            new_fuzzy.y[indice] = 0
        
        indice += 1
    
    new_fuzzy.merge()
    return new_fuzzy
    
fuzzy_cd = intersection_fuzzy_drastica(a_fuzzy, b_fuzzy)

plot_sets(np.array([a_fuzzy.set, b_fuzzy.set, fuzzy_cd.set]),
                   ['A', 'B', "DL"],
                   'Conjunção Drástica entre A e B')

def soma_algebrica(x, y):
    return x + y - (x*y)

def union_fuzzy_soma_algebrica(cfa,cfb):
    
    chaves = list(set(cfa.x).union(cfb.x))
    
    new_fuzzy = fuzzy(len(chaves))
    new_fuzzy.set_x(chaves)
    
    indice = 0
    
    for i in chaves:
        
        if i in cfa.x and i in cfb.x: # se tiverem a mesma chave
            new_fuzzy.y[indice] = soma_algebrica(cfa.get_y(i), cfb.get_y(i))
            
        else:
            if i in cfa.x:
                new_fuzzy.y[indice] = cfa.get_y(i)
            else:
                new_fuzzy.y[indice] = cfb.get_y(i)
        indice += 1

    
    new_fuzzy.merge()
    return new_fuzzy
    
fuzzy_sa = union_fuzzy_soma_algebrica(a_fuzzy, b_fuzzy)

plot_sets(np.array([a_fuzzy.set, b_fuzzy.set, fuzzy_sa.set]),
                   ['A', 'B', "SA"],
                   'Soma Algébrica entre A e B')



def soma_limitada(x, y):
    return min(1, (x+y))

def union_fuzzy_soma_limitada(cfa,cfb):
    
    chaves = list(set(cfa.x).union(cfb.x))
    
    new_fuzzy = fuzzy(len(chaves))
    new_fuzzy.set_x(chaves)
    
    indice = 0
    
    for i in chaves:
        
        if i in cfa.x and i in cfb.x: # se tiverem a mesma chave
            new_fuzzy.y[indice] = soma_limitada(cfa.get_y(i), cfb.get_y(i))
            
        else:
            if i in cfa.x:
                new_fuzzy.y[indice] = cfa.get_y(i)
            else:
                new_fuzzy.y[indice] = cfb.get_y(i)
        indice += 1

    new_fuzzy.merge()
    return new_fuzzy
    
fuzzy_sl = union_fuzzy_soma_limitada(a_fuzzy, b_fuzzy)

plot_sets(np.array([a_fuzzy.set, b_fuzzy.set, fuzzy_sl.set]),
                   ['A', 'B', "SA"],
                   'Soma Limitada entre A e B')


def soma_drastica(x, y):
    if x == 0:
        return y
    elif y == 0:
        return x
    return 1

def union_fuzzy_soma_soma_drastica(cfa,cfb):
    
    chaves = list(set(cfa.x).union(cfb.x))
    
    new_fuzzy = fuzzy(len(chaves))
    new_fuzzy.set_x(chaves)
    
    indice = 0
    
    for i in chaves:
        
        if i in cfa.x and i in cfb.x: # se tiverem a mesma chave
            new_fuzzy.y[indice] = soma_drastica(cfa.get_y(i), cfb.get_y(i))
            
        else:
            if i in cfa.x:
                new_fuzzy.y[indice] = cfa.get_y(i)
            else:
                new_fuzzy.y[indice] = cfb.get_y(i)
        indice += 1

    new_fuzzy.merge()
    return new_fuzzy
    
fuzzy_sd = union_fuzzy_soma_soma_drastica(a_fuzzy, b_fuzzy)

plot_sets(np.array([a_fuzzy.set, b_fuzzy.set, fuzzy_sd.set]),
                   ['A', 'B', "SD"],
                   'Soma Drástica entre A e B')

