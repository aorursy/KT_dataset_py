import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

!pip install turicreate
import numpy as np

import pandas as pd

import pandas_profiling as pp

#TuriCreate For Recommender

import turicreate as tc

#Visualize

import matplotlib.pyplot as plt

import matplotlib.ticker as ticker

import seaborn as sns

#Sparse Matrix

from scipy.sparse import csr_matrix 

#KNN

from sklearn.neighbors import NearestNeighbors

#Surprise for Recommender

sol = pd.read_excel("../input/yakkin.xlsx")
sol['have']=1
pl_tc = tc.SFrame(sol)
pl_tc.plot()
train, test = tc.recommender.util.random_split_by_user(pl_tc,

                                                       user_id='วัตุดิบ',

                                                       item_id='เมนู')
sim_32 = tc.item_similarity_recommender.create(train,

                                               user_id='วัตุดิบ',

                                               item_id='เมนู',

                                               verbose=False,

                                               only_top_k = 32,    

                                               similarity_type='jaccard'

                                               )
list_sim = sim_32.get_similar_items(k=5,verbose=True)
list_sim.print_rows(num_rows=10000,

    num_columns=4,max_column_width=30, max_row_width=100000, output_file=None)
m3_ials_matrix_linear = tc.ranking_factorization_recommender.create(train,

                                                                    user_id='เมนู',

                                                                    item_id='วัตุดิบ',



                                                                     solver ='ials',

                                                     num_factors = 80,

                                                     ials_confidence_scaling_type = 'linear'

#                                                      nmf =1

                                                     )
list_sim1 = m3_ials_matrix_linear.get_similar_users(k=5)
list_sim1.print_rows(num_rows=10000,

    num_columns=4,max_column_width=30, max_row_width=100000, output_file=None)