import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
test_ranks_model = pd.read_csv("../input/learningtorank0657/test_ranks_model.csv")['BasePrediction']

train_ranks_model = pd.read_csv("../input/learningtorank0657/train_ranks_model.csv")['BasePrediction']
from sklearn.datasets import load_svmlight_file

from joblib import Memory

from sklearn.datasets import load_svmlight_file

mem = Memory("./mycache")



@mem.cache

def get_train_data():

    data = load_svmlight_file("/kaggle/input/maderankdataset/train.txt", query_id=True)

    return data[0], data[1], data[2]





@mem.cache

def get_test_data():

    data = load_svmlight_file("/kaggle/input/maderankdataset/test.txt", query_id=True)

    return data[0], data[1], data[2]



train_X, train_y, train_Q = get_train_data()

test_X, test_y, test_Q = get_test_data()
from math import log2



def dcg_at_k(r, k):

    """Score is discounted cumulative gain (dcg)

    Args:

        r: Relevance scores (list or numpy) in rank order

            (first element is the first item)

        k: Number of results to consider

    Returns:

        Discounted cumulative gain

    """

    r = np.asfarray(r)[:k]

    if r.size:

        return np.sum(np.subtract(np.power(2, r), 1) / np.log2(np.arange(2, r.size + 2)))

    return 0.



def dcg_max_at_k(r, k):

    return dcg_at_k(sorted(r, reverse=True), k)



def ndcg_at_k(r, k):

    """Score is normalized discounted cumulative gain (ndcg)

    Args:

        r: Relevance scores (list or numpy) in rank order

            (first element is the first item)

        k: Number of results to consider

    Returns:

        Normalized discounted cumulative gain

    """

    dcg_max = dcg_max_at_k(r, k)

    if dcg_max:

        return dcg_at_k(r, k) / dcg_max

    return 1.



precalculated_dcg_deltas = {}



import itertools

for i, j, k, m in itertools.product(range(5), range(5), range(150), range(150)):

    precalculated_dcg_deltas[i, j, k, m] = abs((pow(2, i) - pow(2, j)) * (1 / log2(2 + k) - 1 / log2(2 + m)))



def dcg_delta(r, i, j):

    return precalculated_dcg_deltas[r[i], r[j], i, j]

#     return abs((pow(2, r[i]) - pow(2, r[j])) * (1 / log2(2 + i) - 1 / log2(2 + j)))
from math import exp

from scipy.special import expit

from sklearn.tree import DecisionTreeRegressor

import sklearn

from functools import partial



import multiprocessing

from multiprocessing import Pool

num_of_cores = multiprocessing.cpu_count()

print(f"num_of_cores = {num_of_cores}")





class MadeLambdaMART:

    def __init__(self, grad_steps, learning_rate=0.05, max_depth=50):

        self.learning_rate = learning_rate

        self.max_depth = max_depth

        self.grad_steps = grad_steps

        self.trees = []

        

    def get_lambdas_and_ws_per_query(self, assess_rates, boost_prediction):

        predicted_ordering = np.argsort(boost_prediction)[::-1]

        assess_rates_sorted = assess_rates[predicted_ordering]

        boost_prediction_sorted = boost_prediction[predicted_ordering]

        lambdas_per_query = np.zeros(len(assess_rates_sorted))

        ws_per_query = np.zeros(len(assess_rates_sorted))

        for i in range(len(assess_rates_sorted)):

            for j in range(i):

                if assess_rates_sorted[i] > assess_rates_sorted[j]:

                    roh = expit(boost_prediction_sorted[j] - boost_prediction_sorted[i])

                    lambda_ij = dcg_delta(assess_rates_sorted, i, j) * roh

                    lambdas_per_query[predicted_ordering[i]] -= lambda_ij

                    lambdas_per_query[predicted_ordering[j]] += lambda_ij

                    w_ij = lambda_ij * (1 - roh)

                    ws_per_query[predicted_ordering[i]] += w_ij

                    ws_per_query[predicted_ordering[j]] += w_ij

                elif assess_rates_sorted[i] < assess_rates_sorted[j]:

                    roh = expit(boost_prediction_sorted[i] - boost_prediction_sorted[j])

                    lambda_ij = dcg_delta(assess_rates_sorted, i, j) * roh

                    lambdas_per_query[predicted_ordering[i]] += lambda_ij

                    lambdas_per_query[predicted_ordering[j]] -= lambda_ij

                    w_ij = lambda_ij * (1 - roh)

                    ws_per_query[predicted_ordering[i]] += w_ij

                    ws_per_query[predicted_ordering[j]] += w_ij

        return (lambdas_per_query, ws_per_query)

    

    def get_queries_bounds(self, queries):

        query_bounds = []

        current_query_index = 0

        current_query_from = current_query_index

        current_query = queries[current_query_from]

        while (current_query_index < len(queries)):

            while (current_query_index < len(queries) and current_query == queries[current_query_index]):

                current_query_index += 1

            query_bounds.append((current_query_from, current_query_index))

            if (current_query_index < len(queries)):

                current_query_from = current_query_index

                current_query = queries[current_query_from]

        return query_bounds

    

    def update_leaf_values(self, tree, input_data, y, w, boost_prediction):

        leaf_clustered_data = tree.apply(input_data.astype(np.float32))

        for leaf in np.where(tree.children_left == sklearn.tree._tree.TREE_LEAF)[0]:

            leaf_cluster = np.where(leaf_clustered_data == leaf)

            suml = np.sum(y[leaf_cluster])

            sumd = np.sum(w[leaf_cluster])

            tree.value[leaf, 0, 0] = 0.0 if abs(sumd) < 1e-50 else ( -suml / sumd)

            boost_prediction[leaf_cluster] += tree.value[leaf, 0, 0] * self.learning_rate

    

    def calc_lambdas_and_ws_per_query(self, args, y, w, assess_rates, boost_prediction, query_max_dcgs):

        idx, (query_from, query_to) = args

        y_per_query = np.zeros(query_to - query_from)

        w_per_query = np.zeros(query_to - query_from)

        if query_max_dcgs[idx] != 0:

            y_per_query, w_per_query = self.get_lambdas_and_ws_per_query(assess_rates[query_from:query_to], 

                                                                         boost_prediction[query_from:query_to])

            y_per_query /= query_max_dcgs[idx]

            w_per_query /= query_max_dcgs[idx]

#         if (idx % 1000 == 0 and idx > 0):

#             print(f"Query number = {idx}")

        return query_from, query_to, y_per_query, w_per_query

        

            

    def fit(self, input_data, assess_rates, queries, validation_size=0, base_prediction=None):

        boost_prediction = np.zeros(len(assess_rates)) if base_prediction is None else base_prediction

        

        if validation_size != 0:

            validate_input_data = input_data[-validation_size:]

            validate_assess_rates = assess_rates[-validation_size:]

            validate_queries = queries[-validation_size:]

            validate_boost_prediction = boost_prediction[-validation_size:]

            

            input_data = input_data[:-validation_size]

            assess_rates = assess_rates[:-validation_size]

            queries = queries[:-validation_size]

            boost_prediction = boost_prediction[:-validation_size]

            

        

        query_bounds = self.get_queries_bounds(queries)

        query_max_dcgs = [dcg_max_at_k(assess_rates[query_from:query_to], 5000) for (query_from, query_to) in query_bounds]

        

        for grad_step in range(self.grad_steps):

            y = np.zeros(len(assess_rates))

            w = np.zeros(len(assess_rates))

            

            with Pool(processes=num_of_cores) as pool:

                for query_from, query_to, y_per_query, w_per_query in pool.map(

                    partial(self.calc_lambdas_and_ws_per_query, y=y, w=w, assess_rates=assess_rates, 

                            boost_prediction=boost_prediction, query_max_dcgs=query_max_dcgs), enumerate(query_bounds)):

                    y[query_from:query_to] = y_per_query

                    w[query_from:query_to] = w_per_query

            current_tree = DecisionTreeRegressor(max_depth=self.max_depth)

            # predict gradient for the "gain" function

            current_tree.fit(input_data, y)

            self.update_leaf_values(current_tree.tree_, input_data, y, w, boost_prediction)

            self.trees.append(current_tree)

                    

            if validation_size != 0 and (grad_step % 10 == 0 or grad_step == self.grad_steps - 1):

                print(f"grad_step = {grad_step}")

                ndcgs_array_train = self.validate(queries, assess_rates, boost_prediction)

                print(f"train nanmean: {np.nanmean(ndcgs_array_train)}")

                ndcgs_array_val = self.validate(validate_queries, validate_assess_rates, self.predict(validate_input_data, validate_queries))

                print(f"validation nanmean: {np.nanmean(ndcgs_array_val)}")



    def predict(self, input_data, queries, base_prediction=None):

        ranks_prediction = np.zeros(len(queries)) if base_prediction is None else base_prediction

        for tree in self.trees:

            ranks_prediction += tree.predict(input_data) * self.learning_rate

        return ranks_prediction

    

    def validate(self, queries, assess_rates, ranks_prediction):

        query_bounds = self.get_queries_bounds(queries)

        ndcgs_array = []

        for idx, (query_from, query_to) in enumerate(query_bounds):

            query_assess_rates = assess_rates[query_from:query_to]

            query_ranks_prediction = ranks_prediction[query_from:query_to]

            predicted_sorted_indexes = np.argsort(query_ranks_prediction)[::-1]

            ndcgs_array.append(ndcg_at_k(query_assess_rates[predicted_sorted_indexes], 5))

        return ndcgs_array
%%time

L = 100000

test_mart = MadeLambdaMART(250, learning_rate=0.1, max_depth=7)

test_mart.fit(train_X, train_y, train_Q, validation_size=L, base_prediction=np.array(train_ranks_model))
train_ranks_prediction = test_mart.predict(train_X, train_Q, base_prediction=np.array(train_ranks_model))
train_ranks_output = pd.DataFrame()

train_ranks_output['BasePrediction'] = train_ranks_prediction

train_ranks_output.to_csv('train_ranks_model.csv',index=False)
test_ranks_prediction = test_mart.predict(test_X, test_Q, base_prediction=np.array(test_ranks_model))
test_output = pd.DataFrame(np.zeros((len(test_Q), 2)), columns=['DocumentScore', 'QueryId'])

test_output['QueryId'] = test_Q

test_output['DocumentScore'] = test_ranks_prediction

test_output['DocumentId'] = np.arange(len(test_output)) + 1

test_output["DocumentRank"] = test_output.groupby("QueryId")["DocumentScore"].rank("min", ascending=False)

test_output.sort_values(['QueryId', 'DocumentRank'], ascending=[True, True], inplace=True)

test_output[['QueryId', 'DocumentId']].to_csv('test_results.csv',index=False)
test_ranks_output = pd.DataFrame()

test_ranks_output['BasePrediction'] = test_ranks_prediction

test_ranks_output.to_csv('test_ranks_model.csv',index=False)