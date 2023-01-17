# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Pertemuan Pertama

class perceptron:



    def __init__(self, input_length, weights=None):

        if weights is None:

            self.weights = np.ones(input_length) * 0

        else:

            self.weights = weights



    @staticmethod

    def unit_step_function(x):

        if x > 0.5:

            return 1

        return 0



    def __call__(self, in_data):

        weighted_input = self.weights * in_data

        weighted_sum = weighted_input.sum()

        return perceptron.unit_step_function(weighted_sum)



p = perceptron(2, np.array([0.5, 0.5]))

for x in [np.array([0, 0]), np.array([0, 1]),

          np.array([1, 0]), np.array([1, 1])]:

    y = p(np.array(x))

    print(x,y)
# Pertemuan Kedua

from collections import Counter

class Perceptron:

    

    def __init__(self, input_length, weights=None):

        if weights==None:

            self.weights = np.random.random((input_length)) * 2 - 1

        self.learning_rate = 0.9

        

    @staticmethod

    def unit_step_function(x):

        if x < 0:

            return 0

        return 1

    

    def __call__(self, in_data):

        weighted_input = self.weights * in_data

        weighted_sum = weighted_input.sum()

        return Perceptron.unit_step_function(weighted_sum)

    

    def adjust(self,

              target_result,

              calculated_result,

              in_data):

        error = target_result - calculated_result

        for i in range(len(in_data)):

            correction = error * in_data[i] * self.learning_rate

            self.weights[i] += correction



def above_line(point, line_func):

    x, y = point

    if y > line_func(x):

        return 1

    else:

        return 0



points = np.random.randint(1, 100, (100, 2))

p = Perceptron(2)

def lin1(x):

    return x + 4

# for i in range (1000): # Penambahan perulangan for

for point in points:

    p.adjust(above_line(point, lin1),

             p(point),

             point)

    evaluation = Counter()

    for point in points:

        if p(point) == above_line(point, lin1):

            evaluation["correct"] += 1

        else:

            evaluation["wrong"] += 1



print(evaluation.most_common())