# Dataset

from sklearn import datasets

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



from matplotlib import animation

from IPython.display import HTML

from tqdm import tqdm_notebook as tqdm

from sklearn import preprocessing
!pip install neupy



from neupy import utils

from neupy import algorithms, utils
# carregar dados

iris = datasets.load_iris()

data = iris.data

features = iris.feature_names

iris_target = iris.target

classes = iris.target_names
df_iris = pd.DataFrame(data, columns=features)

df_iris["class_id"] = iris_target



# Creating numeric class identifiers (0,1,2) 

df_iris.loc[df_iris["class_id"]==0, 'class'] = str(classes[0])

df_iris.loc[df_iris["class_id"]==1, 'class'] = str(classes[1])

df_iris.loc[df_iris["class_id"]==2, 'class'] = str(classes[2])
sns.set(style="ticks")

sns.pairplot(df_iris, hue="class")
df_iris.head()
# Seleção dos atributos numéricos e categóricos

iris_x = df_iris.filter(['sepal length (cm)', 'sepal width (cm)','petal length (cm)', 'petal width (cm)'])

species = {'setosa': 0,'versicolor': 1,'virginica': 2}

iris_class = df_iris['class'].map(species)

iris_class = iris_class.to_numpy()



# Normalização dos dados

iris_x = preprocessing.scale(iris_x)

iris_x = pd.DataFrame(iris_x, columns = ['sepal length (cm)', 'sepal width (cm)','petal length (cm)', 'petal width (cm)'])



# Visualização das estatísticas básicas

iris_x.describe()
def create_gng(n_features, n_start_nodes, epsilon_b, epsilon_n, max_age, lambda_, beta, alpha, max_nodes, verbose = True):

    """

    Parâmetros

    ----------

        n_inputs : Número de classes no conjunto de dados

        n_start_nodes : Número de neurônios inicializados

        step (epsilon_b) : move o nó vencedor em epsilon_b vezes

        neighbour_step (epsilon_n) : move os nós vizinhos do nó vencedor epsilon_n vezes

        max_edge_age : remove arestas mais antigas que max_edge_age  

        n_iter_before_neuron_added (lambda) : a cada lambda iteração um novo nó é adicionado

        error_decay_rate (beta) : taxa de deicamento para todo nó

        after_split_error_decay_rate (alpha) : decaimento de erro após a inserção de novo nó

        max_nodes : Número máximo de nós a serem adicionados    

    """

    

    return algorithms.GrowingNeuralGas(

        n_inputs=n_features,

        n_start_nodes=n_start_nodes,



        shuffle_data=True,

        verbose=verbose,



        step=epsilon_b,

        neighbour_step=epsilon_n,



        max_edge_age=max_age,

        max_nodes=max_nodes,



        n_iter_before_neuron_added=lambda_,

        after_split_error_decay_rate=alpha,

        error_decay_rate=beta,

        min_distance_for_update=0.01,

    )
## Função para pegar erros

def create_model(model_df, list_gng_models, list_modes = [] ):



    for index in range(len(list_gng_models)):

            dict_info = pd.DataFrame.from_dict({'quantisation_error': list_gng_models[index].errors.train,

                                        'iterations': [i for i in range(1, len(list_gng_models[index].errors.train)+1)],

                                        'mode': [list_modes[index] for i in range(1, len(list_gng_models[index].errors.train)+1)],

                                        'error_min':[min(list_gng_models[index].errors.train) for i in range(1, len(list_gng_models[index].errors.train)+1)]

                                        })

            model_df = model_df.append(dict_info)

    return(model_df)
# Criação do objeto 

gng_model1 = create_gng(n_features = len(iris_x.columns),

                         n_start_nodes = 2,

                         epsilon_b = 10e-2,

                         epsilon_n = 10e-4,

                         max_age = 90,

                         lambda_ = 50,

                         beta = 0.005,

                         alpha = 0.5,

                         max_nodes = 500,

                         verbose=True)



# Treino do modelo

gng_model1.train(iris_x, epochs=2000)
# Criação do objeto 

gng_model2 = create_gng(n_features=len(iris_x.columns),

                           n_start_nodes=2,

                           epsilon_b=0.05,

                           epsilon_n=0.0006,

                           max_age=100,

                           lambda_=200,

                           beta=0.05,

                           alpha=0.5,

                           max_nodes=300,

                           verbose=False)



# Treino do modelo

gng_model2.train(iris_x, epochs=2000)
# Resultado do modelo

iris_model = pd.DataFrame(columns = ['quantisation_error', 'iterations', 'mode', 'error_min'])

iris_model = create_model(model_df = iris_model, list_gng_models = [gng_model1, gng_model2], 

                          list_modes = ["gng_model1", "gng_model2"] )
## Gráfico quantização

g = sns.FacetGrid(iris_model, col="mode")

g.map(sns.lineplot, "iterations", "quantisation_error")

g.fig.set_figwidth(22)

g.fig.set_figheight(8)





ax1, ax2 = g.axes[0]



ax1.axhline(iris_model.query('mode == "gng_model1"')['error_min'][0], ls='--')

ax2.axhline(iris_model.query('mode == "gng_model2"')['error_min'][0], ls='--')





ax1.set(ylim=(0.00, 0.075))

ax2.set(ylim=(0.00, 0.075))



ax1.text(1500,0.03, "min error:" + str(round(iris_model.query('mode == "gng_model1"')['error_min'][0], 5)))

ax2.text(1500,0.03, "min error:" + str(round(iris_model.query('mode == "gng_model2"')['error_min'][0], 5)))
nodes, max_age, max_nodes = 2, 25, 50



plt.title('Dimensões das pétalas')

plt.scatter(df_iris[df_iris["class"] == 'setosa']['petal length (cm)'].values,

            df_iris[df_iris["class"] == 'setosa']['petal width (cm)'].values, label='Iris-setosa', alpha=.5)

plt.scatter(df_iris[df_iris["class"] == 'versicolor']['petal length (cm)'].values,

            df_iris[df_iris["class"] == 'versicolor']['petal width (cm)'].values, label='Iris-versicolor', alpha=.5)

plt.scatter(df_iris[df_iris["class"] == 'virginica']['petal length (cm)'].values,

            df_iris[df_iris["class"] == 'virginica']['petal width (cm)'].values, label='Iris-virginica', alpha=.5)

plt.xlabel('petal_length')

plt.ylabel('petal_width')

plt.legend()



data = df_iris[['petal length (cm)', 'petal width (cm)']].values

gng = GNG(n_inputs=nodes, max_edge_age=max_age, max_nodes=max_nodes)





gng.train(data, epochs=500)



for node_1, node_2 in gng.graph.edges:

#     print(node_2.weight)

    weights = np.concatenate([node_1.weight, node_2.weight])

    plt.plot(*weights.T, color='k', linewidth=1)

    plt.scatter(*weights.T, color='k', s=20)
fig, ax = plt.subplots(2, 4, figsize=(17, 8), sharex=True, sharey=True)



max_edge_ages = [50, 500] #default 100

n_iter_before_neuron_added = [100, 2000] # default 1000

n_nodes = [100, 2000] #default 1000



idxs = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3)]

idx = 0

for age in max_edge_ages:

    for it in n_iter_before_neuron_added:

        for node in n_nodes:

            print('idx: ', idx)

            x, y = idxs[idx][0], idxs[idx][1]

            gng = GNG(n_inputs=2, max_edge_age=age, max_nodes=node)

            gng.train(data, 500)

            ax[x][y].set_title('GNG Iris\nn: %s, a: %s, t: %s'%(node, age, it))

            ax[x][y].scatter(df_iris[df_iris["class"] == 'setosa']['petal length (cm)'].values,

                        df_iris[df_iris["class"] == 'setosa']['petal width (cm)'].values, label='Iris-setosa', alpha=.5)

            ax[x][y].scatter(df_iris[df_iris["class"] == 'versicolor']['petal length (cm)'].values,

                        df_iris[df_iris["class"] == 'versicolor']['petal width (cm)'].values, label='Iris-versicolor', alpha=.5)

            ax[x][y].scatter(df_iris[df_iris["class"] == 'virginica']['petal length (cm)'].values,

                        df_iris[df_iris["class"] == 'virginica']['petal width (cm)'].values, label='Iris-virginica', alpha=.5)

            if x == 1:

                ax[x][y].set_xlabel('petal_length')

            if y == 0:

                ax[x][y].set_ylabel('petal_width')

            ax[x][y].legend()

            

            for node_1, node_2 in gng.graph.edges:

                weights = np.concatenate([node_1.weight, node_2.weight])

                ax[x][y].plot(*weights.T, color='k', linewidth=1)

                ax[x][y].scatter(*weights.T, color='k', s=10)

            

            idx+=1