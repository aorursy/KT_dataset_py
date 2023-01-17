# cargamos las dependencias

import os
import numpy as np
import pandas as pd
# definimos seed para que obtengan similares valores a este kernel

seed = 42
np.random.seed(seed)
# Atencion: Se sugiere obviar esta parte de codigo en la primera lectura... 

class MF():

    def __init__(self, R, K, alpha, beta, iterations):
        """
        Perform matrix factorization to predict empty
        entries in a matrix.

        Arguments
        - R (ndarray)   : user-item rating matrix
        - K (int)       : number of latent dimensions
        - alpha (float) : learning rate
        - beta (float)  : regularization parameter
        """

        self.R = R
        self.num_users, self.num_items = R.shape
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.iterations = iterations

    def train(self):
        # Initialize user and item latent feature matrice
        self.P = np.random.normal(scale=1./self.K, size=(self.num_users, self.K))
        self.Q = np.random.normal(scale=1./self.K, size=(self.num_items, self.K))

        # Initialize the biases
        self.b_u = np.zeros(self.num_users)
        self.b_i = np.zeros(self.num_items)
        self.b = np.mean(self.R[np.where(self.R != 0)])

        # Create a list of training samples
        self.samples = [
            (i, j, self.R[i, j])
            for i in range(self.num_users)
            for j in range(self.num_items)
            if self.R[i, j] > 0
        ]

        # Perform stochastic gradient descent for number of iterations
        training_process = []
        for i in range(self.iterations):
            np.random.shuffle(self.samples)
            self.sgd()
            mse = self.mse()
            training_process.append((i, mse))
            if (i+1) % 10 == 0:
                print("Iteration: %d ; error = %.4f" % (i+1, mse))

        return training_process

    def mse(self):
        """
        A function to compute the total mean square error
        """
        xs, ys = self.R.nonzero()
        predicted = self.full_matrix()
        error = 0
        for x, y in zip(xs, ys):
            error += pow(self.R[x, y] - predicted[x, y], 2)
        return np.sqrt(error)

    def sgd(self):
        """
        Perform stochastic graident descent
        """
        for i, j, r in self.samples:
            # Computer prediction and error
            prediction = self.get_rating(i, j)
            e = (r - prediction)

            # Update biases
            self.b_u[i] += self.alpha * (e - self.beta * self.b_u[i])
            self.b_i[j] += self.alpha * (e - self.beta * self.b_i[j])

            # Update user and item latent feature matrices
            self.P[i, :] += self.alpha * (e * self.Q[j, :] - self.beta * self.P[i,:])
            self.Q[j, :] += self.alpha * (e * self.P[i, :] - self.beta * self.Q[j,:])

    def get_rating(self, i, j):
        """
        Get the predicted rating of user i and item j
        """
        prediction = self.b + self.b_u[i] + self.b_i[j] + self.P[i, :].dot(self.Q[j, :].T)
        return prediction

    def full_matrix(self):
        """
        Computer the full matrix using the resultant biases, P and Q
        """
        return self.b + self.b_u[:,np.newaxis] + self.b_i[np.newaxis:,] + self.P.dot(self.Q.T)
# creamos nuestro dataframe c_e

c_e = pd.DataFrame({'codCliente':np.random.randint(1,30000,10),
                    'codEstab':np.random.randint(1,74339,10),
                    'ratingMonto':np.random.random(10)})
# visualizando dimensiones y valores unicos

c_e.shape, c_e['codCliente'].nunique(), c_e['codEstab'].nunique(), c_e['ratingMonto'].nunique()
c_e.head(10)
# ahora crearemos un dataframe de cliente versus establecimientos, pero de todos contra todos, 
# esta sero uno de tipo sparse, pues existen muchisimos pares cliente-establecimiento sin ratingmonto.... 
# es decir muchos ceros y que es lo que debemos predecir

c_e_sparse = pd.pivot_table(data=c_e,
                           values='ratingMonto',
                           index='codCliente',
                           columns='codEstab',
                           fill_value=0)
# visualizando dimensiones

c_e_sparse.shape
# visualizando nuestro dataframe sparse

c_e_sparse
# ahora generamos nuestra matrix sparse a partir del dataframe anterior, 
# esto porque nuestro modelo/ clase a emplear opera sobre matrices/ array

C_E_sparse = c_e_sparse.values
# definimos nuestro modelo a entrenar, es decir fijamos algunos hiperparametros como alpha y beta,
# y la cantidad de variables latentes a obtener:K
# como nos centraremos en recomendar establecimientos, observando la matrix sparse, vemos que sus
# features se basan en codCliente y estos son 10, asi K no debe ser mayor a 10. 
# PD. Estamos efectuando una reduccion de dimensiones (aprendizaje no-supervisado)

matrix_factorization = MF(C_E_sparse, K=5,
                         alpha=0.1, beta=0.01,
                         iterations=100)
# entrenamos nuestro modelo

training = matrix_factorization.train()
# obtenemos la matrix densa de nuestra matrix sparse

matrix_final_predictions = matrix_factorization.full_matrix()
# a partir del dataframe sparse obtenemos la "lista" de los elementos de "columns" y de "index"

columns = c_e_sparse.columns
index = c_e_sparse.index
# de la matrix densa y que es en realidad nuestra matrix predicted, generamos un dataframe

c_e_predicted_1 = pd.DataFrame(matrix_final_predictions, columns=columns,
                            index=index)
# a continuacion apreciamos nuestro dataframe completo y debajo de ella la matrix sparse... 
c_e_predicted_1
c_e_sparse
# para codCliente 11285 y codEstab 770 en matrix sparse(inicial)

c_e_sparse.iloc[3,0] 
# para codCliente 11285 y codEstab 770 en matrix sparse(predecida)... los valores con la matrix sparse son similares .. es Ok

c_e_predicted_1.iloc[3,0] 
# generamos la lista de los codestab para el cliente 11285

list_estab_cliente_11285_nmf = c_e_predicted_1.iloc[3]
# ordenamos la ultima lista obtenida

list_estab_cliente_11285_nmf.sort_values(ascending=False)
# y finalmente lo que buscamos, la lita de los 5 establecimientos con mayor ratingMonto para el cliente 11285

list_estab_cliente_11285_nmf.sort_values(ascending=False).iloc[0:5]
# Referencias

# Ref. MNF e implementacion del codigo http://www.albertauyeung.com/post/python-matrix-factorization/
# Python issues: https://stackoverflow.com/
from scipy.sparse.linalg import svds
C_E_sparse.shape  
# por un tema de comodidad renombraremos a C_E_sparse como X... :)

X = C_E_sparse
X.shape
# entrenamos nuestro modelo, igual que en el caso anterior usaremos K=5
# esta operacion genera las matrices U, S(sigma) y V(vt) sobre la que comentamos anteriormente.

U, sigma, vt = svds(X, k=5)
type(U), type(sigma), type(vt)
# generamos una matrix diagonal adecuada para luego efectuar el producto UxSxV

sigma = np.diag(sigma)
# Y ahora obtenemos nuestra matrix objetivo, pero completa/densa 

c_e_predicted_2 = np.dot(np.dot(U, sigma), vt)
# generamos nuestro dataframe 

c_e_predicted_2 = pd.DataFrame(c_e_predicted_2, columns=columns, index=index)
# ahora veamos como nuestra matrix final es la version completa de nuestra sparsida
c_e_predicted_2
c_e_sparse
# igual que hicimos antes, ahora trabajaremos en el listado de los establecimientos del cliente 11285

list_estab_cliente_11285_svd = c_e_predicted_2.iloc[3] 
list_estab_cliente_11285_svd
list_estab_cliente_11285_svd.sort_values(ascending=False)
list_estab_cliente_11285_svd.sort_values(ascending=False).iloc[0:5]
# para efectos de comparar resultados, generamos los dataframes de los 5 establecimientos a recomendar
# al cliente 11285 , basado en la tecnica NMF y SVDS

Listado_5_Locales_Cliente_11285_NMF = pd.DataFrame(list_estab_cliente_11285_nmf.sort_values(ascending=False).iloc[0:5])
Listado_5_Locales_Cliente_11285_SVDS = pd.DataFrame(list_estab_cliente_11285_svd.sort_values(ascending=False).iloc[0:5])
Listado_5_Locales_Cliente_11285_NMF
Listado_5_Locales_Cliente_11285_SVDS
# https://beckernick.github.io/matrix-factorization-recommender/
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.svds.html
