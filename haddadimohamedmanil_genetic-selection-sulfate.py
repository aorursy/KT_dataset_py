import numpy as np

import time

from sklearn.model_selection import cross_val_score, KFold

from sklearn.metrics import accuracy_score, r2_score

from sklearn.decomposition import PCA

from sklearn import preprocessing as pp

import matplotlib.pyplot as plt

import pandas as pd

from sklearn.linear_model import Lasso

from sklearn.model_selection import GridSearchCV

import os
from sklearn.linear_model import LinearRegression

lin_model = LinearRegression()
data2 = pd.read_hdf('../input/cleanedv1/clean.h5')

pd.options.display.max_columns = None

# #data2 = pd.read_excel(os.path.join('C:\\Users\\Sido\\Documents', 't2.xlsx'))



# col_sulf =data2.iloc[:, (data2.columns.get_level_values(1)=='sulfate_dose')]





# data2 = data2.iloc[:, (data2.columns.get_level_values(0)=='eau_brute') | 

#                    (data2.columns.get_level_values(0)=='eau_dec_1') |

#                    (data2.columns.get_level_values(0)=='eau_dec_2') | 

#                    (data2.columns.get_level_values(0)=='eau_filtre') |

#                    (data2.columns.get_level_values(0)=='entre_res_1') |

#                    (data2.columns.get_level_values(0)=='entre_res_2') |

#                    (data2.columns.get_level_values(0)=='eau_sp1') 

#                    ]

        

# col_sulf = col_sulf.droplevel(0, axis  = 1)

# col_sulf.reset_index(inplace = True, drop=True)

# data2 = data2.droplevel(0, axis  = 1)

# data2.reset_index(inplace = True, drop=True)

# data2['sulfate_dose'] = col_sulf







X = data2.drop('sulfate_dose', axis=1)

# X_d1 = data2.iloc[:,4:9]

# X_d2 = data2.iloc[:,9:14]

# X_f  = data2.iloc[:,14:19]

# X_r1 = data2.iloc[:,19:24]

# X_r2 = data2.iloc[:,24:29]

# X_sp = data2.iloc[:,29:34]

Y = data2.loc[:, ['sulfate_dose']]

# data2.to_csv('base4_1.csv', index=False, sep=' ')

X.shape
import numpy as np

import time

from sklearn.model_selection import cross_val_score, KFold

from sklearn.metrics import f1_score, r2_score

from sklearn.decomposition import PCA

import matplotlib.pyplot as plt





class GeneticR2():

    """

    Built to be compatible with sci-kit learn library for both regression and classification models

    This is designed to help with feature selection in highly dimensional datasets

    """



    def __init__(self, mutation_rate = 0.001, iterations = 100, pool_size = 50):

        self.mutation_rate = mutation_rate

        self.iterations = iterations

        self.pool_size = pool_size

        self.pool = np.array([])

        self.iterations_results = {}

        self.kf = KFold(n_splits=5)





    def results(self):

        """

        Print best results from the fit

        """



        return (self.pool[0], [idx for idx, gene in enumerate(self.pool[0]) if gene==1])





    def plot_progress(self):

        """

        Plots the progress of the genetic algorithm

        """



        avs = [np.mean(self.iterations_results[str(x)]['scores']) for x in range(1,self.iterations+1)]

        avs0 = [np.mean(self.iterations_results[str(x)]['scores'][0]) for x in range(1,self.iterations+1)]

        plt.plot(avs, label='Pool Average Score')

        plt.plot(avs0, label='Best Solution Score')

        plt.legend()

        plt.show()





    def fit(self, model, _type, X, y, cv=True, pca=False, verbose=True):

        """

        model = sci-kit learn regression/classification model

        _type = type of model entered STR (eg.'regression' or 'classification')

        X = X input data

        y = Y output data corresponding to X

        cv = True/False for cross-validation

        pca = True/False for principal component analysis

        """



        self.__init__(self.mutation_rate, self.iterations, self.pool_size)

        

        is_array = False



        try:

            X = np.array(X)

            is_array = True

        except:

            X = X



        self.pool = np.random.randint(0,2,(self.pool_size, X.shape[1]))



        for iteration in range(1,self.iterations+1):

            s_t = time.time()

            scores = list(); fitness = list(); 

            for chromosome in self.pool:

                chosen_idx = [idx for gene, idx in zip(chromosome, range(X.shape[1])) if gene==1]



                if is_array==True: 

                    adj_X = X[:,chosen_idx]

                elif is_array==False:

                    adj_X = X.iloc[:,chosen_idx]

                    pca==False





                if pca==True:

                    adj_X = PCA(n_components=np.where(np.cumsum(PCA(n_components=adj_X.shape[1]).fit(adj_X).explained_variance_ratio_)>0.99)[0][0]+1).fit_transform(adj_X)



                if _type == 'regression':

                    if cv==True:

                        score = np.mean(cross_val_score(model, adj_X, y, scoring='r2', cv=self.kf))

                    else:

                        score = r2_score(y, model.fit(adj_X,y).predict(adj_X))



                elif _type == 'classification':

                    if cv==True:

                        score = np.mean(cross_val_score(model, adj_X, y, scoring='f1_weighted', cv=self.kf))

                    else:

                        score = f1_score(y, model.fit(adj_X,y).predict(adj_X))



                scores.append(score)

            fitness = [x/sum(scores) for x in scores]



            fitness, self.pool, scores = (list(t) for t in zip(*sorted(zip(fitness, [list(l) for l in list(self.pool)], scores),reverse=True)))

            self.iterations_results['{}'.format(iteration)] = dict()

            self.iterations_results['{}'.format(iteration)]['fitness'] = fitness

            self.iterations_results['{}'.format(iteration)]['pool'] = self.pool

            self.iterations_results['{}'.format(iteration)]['scores'] = scores



            self.pool = np.array(self.pool)



            if iteration != self.iterations+1:

                new_pool = []

                for chromosome in self.pool[1:int((len(self.pool)/2)+1)]:

                    random_split_point = np.random.randint(1,len(chromosome))

                    next_gen1 = np.concatenate((self.pool[0][:random_split_point], chromosome[random_split_point:]), axis = 0)

                    next_gen2 = np.concatenate((chromosome[:random_split_point], self.pool[0][random_split_point:]), axis = 0)

                    for idx, gene in enumerate(next_gen1):

                        if np.random.random() < self.mutation_rate:

                            next_gen1[idx] = 1 if gene==0 else 0

                    for idx, gene in enumerate(next_gen2):

                        if np.random.random() < self.mutation_rate:

                            next_gen2[idx] = 1 if gene==0 else 0

                    new_pool.append(next_gen1)

                    new_pool.append(next_gen2)

                self.pool = new_pool

            else:

                continue

            if verbose:

                if iteration % 10 == 0:

                    e_t = time.time()

                    print('Iteration {} Complete [Time Taken For Last Iteration: {} Seconds]'.format(iteration,round(e_t-s_t,2)))

           
from sklearn import preprocessing

names = X.columns

# Create the Scaler object

scaler = preprocessing.StandardScaler()

# Fit your data on the scaler object

scaled_df = scaler.fit_transform(X)

scaled_df = pd.DataFrame(scaled_df, columns=names)

X_scaled = scaled_df
poly = pp.PolynomialFeatures(2, include_bias=False)

input_df = X_scaled

output_nparray = poly.fit_transform(input_df)

target_feature_names = ['*'.join(['{}^{}'.format(pair[0],pair[1]) for pair in tuple if pair[1]!=0]) for tuple in [zip(input_df.columns,p) for p in poly.powers_]]

output_df = pd.DataFrame(output_nparray, columns = target_feature_names)

output_df
gfs = GeneticR2()





# fit the optimizer

gfs.fit(model=lin_model, _type='regression', X=output_df, y=Y) # regression model



# get results output

binary_output_of_optimal_variables, indicies_of_optimal_variables = gfs.results()



# plot results of progress

gfs.plot_progress()
Xs = output_df.iloc[:, indicies_of_optimal_variables]
Xs.columns.tolist()
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)



names = X_train.columns

# Create the Scaler object

scaler = preprocessing.StandardScaler()

# Fit your data on the scaler object

scaled_df = scaler.fit_transform(X_train)

scaled_df = pd.DataFrame(scaled_df, columns=names)

X_train = scaled_df



poly = pp.PolynomialFeatures(2, include_bias=False)

input_df = X_train

output_nparray = poly.fit_transform(input_df)

target_feature_names = ['*'.join(['{}^{}'.format(pair[0],pair[1]) for pair in tuple if pair[1]!=0]) for tuple in [zip(input_df.columns,p) for p in poly.powers_]]

output_df = pd.DataFrame(output_nparray, columns = target_feature_names)





reg = LinearRegression().fit(output_df.loc[:, Xs.columns.tolist()], Y)



#normaliser le test

names = X_test.columns

# Create the Scaler object

scaler = preprocessing.StandardScaler()

# Fit your data on the scaler object

scaled_df = scaler.fit_transform(X_test)

scaled_df = pd.DataFrame(scaled_df, columns=names)

X_test = scaled_df



poly = pp.PolynomialFeatures(2, include_bias=False)

input_df = X_test

output_nparray = poly.fit_transform(input_df)

target_feature_names = ['*'.join(['{}^{}'.format(pair[0],pair[1]) for pair in tuple if pair[1]!=0]) for tuple in [zip(input_df.columns,p) for p in poly.powers_]]

output_df = pd.DataFrame(output_nparray, columns = target_feature_names)





reg.score(output_df, y_train), reg.score(output_df, y_test), mean_squared_error(y_test, reg.predict(output_df))
data2 = pd.read_hdf('../input/cleanedv1/clean.h5')

pd.options.display.max_columns = None

X = data2.drop('sulfate_dose', axis=1)

Y = data2.loc[:, ['sulfate_dose']]

X.shape
poly = pp.PolynomialFeatures(2, include_bias=False)

input_df = X

output_nparray = poly.fit_transform(input_df)

target_feature_names = ['*'.join(['{}^{}'.format(pair[0],pair[1]) for pair in tuple if pair[1]!=0]) for tuple in [zip(input_df.columns,p) for p in poly.powers_]]

output_df = pd.DataFrame(output_nparray, columns = target_feature_names)
from sklearn import preprocessing

names = output_df.columns

# Create the Scaler object

scaler = preprocessing.StandardScaler()

# Fit your data on the scaler object

scaled_df = scaler.fit_transform(output_df)

scaled_df = pd.DataFrame(scaled_df, columns=names)

output_df = scaled_df

# poly normalisé
gfs = GeneticR2()





# fit the optimizer

gfs.fit(model=lin_model, _type='regression', X=output_df, y=Y) # regression model



# get results output

binary_output_of_optimal_variables, indicies_of_optimal_variables = gfs.results()



# plot results of progress

gfs.plot_progress()
Xs = output_df.iloc[:, indicies_of_optimal_variables]

Xs.columns.tolist()
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)



poly = pp.PolynomialFeatures(2, include_bias=False)

input_df = X_train

output_nparray = poly.fit_transform(input_df)

target_feature_names = ['*'.join(['{}^{}'.format(pair[0],pair[1]) for pair in tuple if pair[1]!=0]) for tuple in [zip(input_df.columns,p) for p in poly.powers_]]

output_df = pd.DataFrame(output_nparray, columns = target_feature_names) #X_train_poly



names = output_df.columns

# Create the Scaler object

scaler = preprocessing.StandardScaler()

# Fit your data on the scaler object

scaled_df = scaler.fit_transform(output_df)

scaled_df = pd.DataFrame(scaled_df, columns=names)

output_df = scaled_df #X_train_normalise_poly





reg = LinearRegression().fit(output_df.loc[:, Xs.columns.tolist()], Y)



poly = pp.PolynomialFeatures(2, include_bias=False)

input_df = X_test

output_nparray = poly.fit_transform(input_df)

target_feature_names = ['*'.join(['{}^{}'.format(pair[0],pair[1]) for pair in tuple if pair[1]!=0]) for tuple in [zip(input_df.columns,p) for p in poly.powers_]]

output_df = pd.DataFrame(output_nparray, columns = target_feature_names) #X_test_poly



#normaliser le test

names = output_df.columns

# Create the Scaler object

scaler = preprocessing.StandardScaler()

# Fit your data on the scaler object

scaled_df = scaler.fit_transform(output_df)

scaled_df = pd.DataFrame(scaled_df, columns=names)

output_df = scaled_df#X_test_poly_normalisé



reg.score(output_df, y_test), mean_squared_error(y_test, reg.predict(output_df))