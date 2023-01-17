import numpy as np

import pandas as pd

import random

from itertools import chain

from sklearn.linear_model import Perceptron

from sklearn.svm import LinearSVC

from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

from matplotlib.pyplot import figure
class DataGenerator:

    def __init__(self, region_distribution_size=125, distribution_range=(-5,5)):

        self.__region_threshold = region_distribution_size

        self.__distribution_range = list(distribution_range)

        self.__dataset = None

        self.generate_data()

    

    def generate_data(self):

        region_data = {"R13": set(), "R14": set(), "R23": set(), "R24": set()}

        while(True):

            random_point = [random.uniform(*self.__distribution_range), random.uniform(*self.__distribution_range)]

            random_point_region = self.__get_region(*random_point)

            if(random_point_region is not None):

                if(len(region_data[random_point_region])<self.__region_threshold):

                    region_data[random_point_region].add(tuple(random_point))

            if(len(region_data["R13"])==self.__region_threshold and len(region_data["R14"])==self.__region_threshold 

               and len(region_data["R23"])==self.__region_threshold and len(region_data["R24"])==self.__region_threshold):

                break

        flattened_data = list(chain(*[list(map(lambda x: list(x) + [i],region_data[i])) for i in region_data.keys()]))

        print("R13 ->",len(region_data['R13']),"\nR23 ->",len(region_data['R23']),"\nR14 ->",len(region_data['R14']),

              "\nR24 ->",len(region_data['R24']))

        self.__dataset = pd.DataFrame(flattened_data, columns=['x_coordinate', 'y_coordinate', 'region'])



    def __get_region(self,x,y)-> str:

        if(x==y): return None

        if(x+y+1==0): return None

        if(x>y):

            if(x > -1-y): return "R13"

            else: return "R14"

        else:

            if(x >-1-y): return "R23"

            else: return "R24"

    

    def get_shuffled_dataset(self, training_set_dist: int) -> (pd.DataFrame, pd.DataFrame):

        training_set = self.__dataset.groupby('region').apply(lambda group: group.sample(training_set_dist)).reset_index(drop = True)

        testing_set = self.__dataset.merge(training_set, how = 'outer' ,indicator=True).loc[lambda x : x['_merge']=='left_only'].drop('_merge', axis=1).reset_index(drop=True)

        return training_set, testing_set
data_gen = DataGenerator(region_distribution_size=125)

training_set_dist_matrix = [100, 75, 50, 40, 30, 20, 10, 5, 1]
perceptron_acc_matrix = []

for training_set_dist in training_set_dist_matrix:

    accuracy_vals = []

    print('--------------------------Training set size: {}--------------------------'.format(training_set_dist*4))

    for i in range(5):

        print('---------------Iteration: {}---------------'.format(i+1))

        training_data, testing_data = data_gen.get_shuffled_dataset(training_set_dist=training_set_dist)

        X_train = pd.DataFrame({'x1': training_data['x_coordinate'], 'x2': training_data['y_coordinate'], 

                                         'x1_sq': training_data['x_coordinate']**2, 'x2_sq': training_data['y_coordinate']**2})

        X_test = pd.DataFrame({'x1': testing_data['x_coordinate'], 'x2': testing_data['y_coordinate'], 

                                         'x1_sq': testing_data['x_coordinate']**2, 'x2_sq': testing_data['y_coordinate']**2})

        Y_train = training_data.region.apply(lambda x: 'NC1' if(x in {'R13', 'R24'}) else 'NC2' )

        Y_test = testing_data.region.apply(lambda x: 'NC1' if(x in {'R13', 'R24'}) else 'NC2' )

        clf = Perceptron()

        clf.fit(X_train, Y_train)

        print('Weights -> {}'.format(clf.coef_[0]))

        Y_pred = clf.predict(X_test)

        accuracy_vals.append(accuracy_score(Y_pred, Y_test))

    perceptron_acc_matrix.append(np.average(accuracy_vals))

    print('Average accuracy -> {}'.format(np.average(accuracy_vals)))

perceptron_df = pd.DataFrame({'training_set_size': map(lambda x: x * 4, training_set_dist_matrix), 'accuracy': perceptron_acc_matrix})
perceptron_df.plot(x='training_set_size', y = 'accuracy', kind="line", figsize=(5,4))
lin_svm_acc_matrix = []

for training_set_dist in training_set_dist_matrix:

    accuracy_vals = []

    print('--------------------------Training set size: {}--------------------------'.format(training_set_dist*4))

    for i in range(5):

        print('---------------Iteration: {}---------------'.format(i+1))

        training_data, testing_data = data_gen.get_shuffled_dataset(training_set_dist=100)

        X_train = pd.DataFrame({'x1': training_data['x_coordinate'], 'x2': training_data['y_coordinate'],

                                'x1_sq': training_data['x_coordinate']**2, 'x2_sq': training_data['y_coordinate']**2,

                                'x1x2': training_data['x_coordinate']*training_data['y_coordinate']})

        X_test = pd.DataFrame({'x1': testing_data['x_coordinate'], 'x2': testing_data['y_coordinate'],

                               'x1_sq': testing_data['x_coordinate']**2, 'x2_sq': testing_data['y_coordinate']**2,

                               'x1x2': testing_data['x_coordinate']*testing_data['y_coordinate']})

        Y_train = training_data.region.apply(lambda x: 'NC1' if(x in {'R13', 'R24'}) else 'NC2' )

        Y_test = testing_data.region.apply(lambda x: 'NC1' if(x in {'R13', 'R24'}) else 'NC2' )

        tuned_estimator = GridSearchCV(estimator=LinearSVC(), param_grid={

            'C': [0.01, 0.05, 0.1, 0.5, 1, 2, 3, 4, 5, 10, 20, 30, 50, 100, 150, 200, 1000], 'penalty': ['l2'] , 

            'max_iter': [10000000]}, cv= 10)

        tuned_estimator.fit(X_train, Y_train)

        clf = LinearSVC(**tuned_estimator.best_params_)

        clf.fit(X_train, Y_train)

        print('Tuned C -> {}\nWeights -> {}\nBias -> {}'.format(tuned_estimator.best_params_.get('C'), clf.coef_[0], clf.intercept_[0]))

        Y_pred = clf.predict(X_test)

        accuracy_vals.append(accuracy_score(Y_pred, Y_test))

    lin_svm_acc_matrix.append(np.average(accuracy_vals))

    print('Average accuracy -> {}'.format(np.average(accuracy_vals)))

lin_svm_df = pd.DataFrame({'training_set_size': map(lambda x: x * 4, training_set_dist_matrix), 'accuracy': lin_svm_acc_matrix})
lin_svm_df.plot(x='training_set_size', y = 'accuracy', kind="line", figsize=(5,4))
kernel_svm_acc_matrix = []

for training_set_dist in training_set_dist_matrix:

    accuracy_vals = []

    print('--------------------------Training set size: {}--------------------------'.format(training_set_dist*4))

    for i in range(5):

        print('---------------Iteration: {}---------------'.format(i+1))

        training_data, testing_data = data_gen.get_shuffled_dataset(training_set_dist=10)

        X_train = pd.DataFrame({'x1': training_data['x_coordinate'], 'x2': training_data['y_coordinate']})

        X_test = pd.DataFrame({'x1': testing_data['x_coordinate'], 'x2': testing_data['y_coordinate']})

        Y_train = training_data.region.apply(lambda x: 'NC1' if(x in {'R13', 'R24'}) else 'NC2' )

        Y_test = testing_data.region.apply(lambda x: 'NC1' if(x in {'R13', 'R24'}) else 'NC2' )

        tuned_estimator = GridSearchCV(estimator=SVC(), cv= 10, param_grid={

            'C': [1, 2, 3, 4, 5, 10, 20, 30, 50, 100, 150, 200, 1000], 'kernel': ['rbf']})

        tuned_estimator.fit(X_train, Y_train)

        clf = SVC(**tuned_estimator.best_params_)

        clf.fit(X_train, Y_train)

        print('Tuned C -> {}\nBias -> {}'.format(tuned_estimator.best_params_.get('C'), clf.intercept_[0]))

        Y_pred = clf.predict(X_test)

        print('Accuracy score -> {}'.format(accuracy_score(Y_pred, Y_test)))

        accuracy_vals.append(accuracy_score(Y_pred, Y_test))

    kernel_svm_acc_matrix.append(np.average(accuracy_vals))

    print('Average accuracy -> {}'.format(np.average(accuracy_vals)))

kernel_svm_df = pd.DataFrame({'training_set_size': map(lambda x: x * 4, training_set_dist_matrix), 'accuracy': kernel_svm_acc_matrix})
kernel_svm_df.plot(x='training_set_size', y = 'accuracy', kind="line", figsize=(5,4))