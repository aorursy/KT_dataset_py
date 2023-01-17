%load_ext autoreload
%autoreload 2

%matplotlib inline
from fastai.imports import *
from fastai.structured import *
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from IPython.display import display
from sklearn import metrics
df_raw = pd.read_feather('../input/bulldozersraw/bulldozers-raw')
df_trn, y_trn, nas = proc_df(df_raw, 'SalePrice')
print ("SHAPES!!!")
print ("df_raw => " + str(df_raw.shape))
print ("df_trn => " + str(df_trn.shape))
print ("y_trn ==> " + str(y_trn.shape))
# Criando uma função para fazer o split do DataFrame 'a' na posição 'n'
def split_vals(a,n):
    return a[:n], a[n:]

# Qtde de registros que ficarão nos DataFrame's de validação
n_valid = 12000

# Ponto de corte do DataFrame
n_trn = len(df_trn)-n_valid

# Dividindo os DataFrame's iniciais em 2
X_train, X_valid = split_vals(df_trn, n_trn)
y_train, y_valid = split_vals(y_trn, n_trn)
raw_train, raw_valid = split_vals(df_raw, n_trn)
print ("SHAPES!!!")
print ("X_train => " + str(X_train.shape))
print ("X_valid => " + str(X_valid.shape))
print ("y_train => " + str(y_train.shape))
print ("y_valid => " + str(y_valid.shape))
print ("raw_train => " + str(raw_train.shape))
print ("raw_valid => " + str(raw_valid.shape))
### Gerando um DataFrame com apenas duas colunas
x_sub = X_train[['YearMade', 'MachineHoursCurrentMeter']]
print ("x_sub => " + str(x_sub.shape))
x_sub.head().T
# 'class' declara uma classe / objeto, neste caso chamado 'TreeEnsemble'
class TreeEnsemble():
    
    # Os métodos / funções em Python são declarados utilizando o 'def', o '__init__' é um MÉTODO MÁGICO
    # do Python (ele é o construtor da classe / objeto), existem muitos outros métodos mágicos

    def __init__(self, x, y, n_trees, sample_sz, min_leaf=5):       
        
        # Todo método dentro de uma classe deve ter um primeiro parametro obrigatório, que convencionou-se 
        # chamar de 'self'
        
        # Gerando uma semente aleatória
        np.random.seed(42)
        
        # Atribuindo os valores recebidos aos atribudos correspondentes do objeto 'TreeEnsemble'
        # O Python permite a atribuição de vários valores em uma única linha
        self.x,self.y,self.sample_sz,self.min_leaf = x,y,sample_sz,min_leaf
        
        # Chama um método da própria classe 'TreeEnsemble' para criar uma lista '[]' chamada 'trees'
        # com as 'n_trees' (Observe que não foi efetuado um tratamento caso este valor não tenha sido informado) 
        self.trees = [self.create_tree() for i in range(n_trees)]
        
        
    def create_tree(self):
        
        # Cria uma variável local chamada 'rnd_idxs' contendo os 'sample_sz' indices das linhas (já embaralhados)
        # Veja um exemplo na próxima celula
        rnd_idxs = np.random.permutation(len(self.y))[:self.sample_sz]
        
        # O retorno deste método é o resultado da execução da classe 'DecisionTree', passando como parametro
        # os atributos 'self.x' e 'self.y', MAS APENAS com as linhas (índices) que foram selecionados aleatoriamente.
        #
        # self.x.iloc[rnd_idxs] => Como 'self.x' é uma matriz é necessário utilizar o método 'iloc' para
        # referenciar as linhas informadas na lista 'rnd_idxs'
        #
        # self.y[rnd_idxs] => Como 'self.y' é um array (tem apenas uma coluna) não foi necessário utilizar o
        # 'iloc', bastando apenas referenciar as linhas informando a lista 'rnd_idxs'
        #
        # Portanto para cada arvore gerada teremos uma amostra diferente de registros
        return DecisionTree(self.x.iloc[rnd_idxs], self.y[rnd_idxs], min_leaf=self.min_leaf)
        
        
    def predict(self, x):
        
        # O método 'predict' retorna a média ..
        
        return np.mean([t.predict(x) for t in self.trees], axis=0)
# Demonstração do método 'numpy.random.permutation' 
print ("Tamanho do 'y_train' ==> " + str(len(y_train)))
print ("Exibindo os 5 primeiros indices entre '0' e 'len(y_train)' de forma aleatória.")
np.random.permutation(len(y_train))[:5]
class DecisionTree():
    
    def __init__(self, x, y, idxs=None, min_leaf=5):
        
        # Simplesmente está inicializando a classe 'DecisionTree' atribuindo os parametros recebidos aos 
        # atribudos da classe, sem qualquer tipo de tratamento adicional
        # será reimplementada logo abaixo
        
        self.x,self.y,self.idxs,self.min_leaf = x,y,idxs,min_leaf
# Criando um objeto 'TreeEnsemble'
m = TreeEnsemble(X_train, y_train, n_trees=10, sample_sz=1000, min_leaf=3)
# Exibindo uma representação da 1ª arvore do objeto 'TreeEnsemble' 
m.trees[0]
class DecisionTree():
    
    def __init__(self, x, y, idxs=None, min_leaf=5):
        
        # Trata o atributo 'idxs'
        if idxs is None:
            # Se 'idxs' não for definido, atribui um Array de inteiros de '0' até 'len(y)'
            idxs = np.arange(len(y))
            
        # Atribui os parametros recebidos / tratados aos atributos da classe 'DecisionTree'
        self.x,self.y,self.idxs,self.min_leaf = x,y,idxs,min_leaf
        
        # Atribui à 'self.n' o tamanho do array 'idxs'
        # Atribui à 'self.c' o número de colunas da matriz 'x'
        self.n,self.c = len(idxs), x.shape[1]
        
        # Atribui à 'self.val' a média dos valores de 'y'
        self.val = np.mean(y[idxs])
        
        # Atribui à 'self.score' um float infinito positivo
        self.score = float('inf')
        
        # Executa o método interno 'self.find_varsplit()'
        self.find_varsplit()
        
    # Isso apenas faz uma decisão; nós vamos fazer isso recursivo depois
    def find_varsplit(self):
        for i in range(self.c):
            self.find_better_split(i)
            
    # Nós vamos escrever isso mais tarde!
    def find_better_split(self, var_idx):
        pass
    
    # Devolve o nome da coluna na posição passada como parametro
    # Equivalente a um método 'get_split_name(var_idx)'
    @property
    def split_name(self):
        return self.x.columns[self.var_idx]
    
    # Retorna o conteúdo da coluna 'self.var_idx' de todas as linhas contidas no array 'self.idxs'
    @property
    def split_col(self):
        return self.x.values[self.idxs,self.var_idx]
    
    # Retorna 'True' ou 'False' para a comparação 
    @property
    def is_leaf(self):
        return self.score == float('inf')
    
    # Implementa outro método mágico do Python, agora responsável pela representação da classe / objeto
    # O objetivo do '__repr__' é ser inequívoco
    def __repr__(self):
        s = f'n: {self.n}; val:{self.val}'
        if not self.is_leaf:
            s += f'; score:{self.score}; split:{self.split}; var:{self.split_name}'
        return s
# Demonstração do que acontece no método 'split_name'
#     return self.x.columns[self.var_idx]
print("Todas as colunas de 'X_train' => " + str(X_train.columns))
print("Nome da 3ª coluna de 'X_train' => " + str(X_train.columns[2]))
# Demonstração do que acontece no método 'split_col'
#     return self.x.values[self.idxs,self.var_idx]
print ("Todos os valores das 3ª e 4ª linhas de 'X_train'\n" + str(X_train.values[[2,3]]))
print ("Todos os valores da 3ª coluna, das 3ª e 4ª linhas de X_train\n" + str(X_train.values[[2,3],2]))
# Refazendo o objeto 'TreeEnsemble' com apenas 30 registros
m = TreeEnsemble(X_train, y_train, n_trees=10, sample_sz=30, min_leaf=3)
# Exibindo os valores 'target' da 2ª arvore
m.trees[1].y
# Exibindo a representação da 2ª arvore (agora usando o método __repr__)
m.trees[1]
# Exibindo os valores 'target' da 3ª arvore
m.trees[2].y
# Exibindo a representação da 3ª arvore (agora usando o método __repr__)
m.trees[2]