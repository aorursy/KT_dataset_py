import csv

import pandas

import numpy as np

from sklearn.linear_model import LinearRegression



def main():

    dtypes = {

        'rating': float,

        'bayes_rating': float

    }

    headers = []

    with open('../input/bgggamesdata/bgg_games_data/mechanics_data.csv') as data:

        csvreader = csv.reader(data)

        for row in csvreader:

            header = row

            for entry in header:

                if entry not in dtypes:

                    dtypes[entry] = bool

                    headers.append(entry)

            break

    data = pandas.read_csv('../input/bgggamesdata/bgg_games_data/mechanics_data.csv', dtype=dtypes)

    columns = []

    for i in range(182):

        columns.append('bayes_rating')

    Y = data[columns].values.reshape(-1, 182)

    data = data.drop(columns=['bayes_rating', 'rating'])

    X = data.values.reshape(-1,182)

    linear_regressor = LinearRegression()

    linear_regressor.fit(X, Y)

    coefs = linear_regressor.coef_[0]

    highest_coefs = [-1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000]

    best_coefs = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]

    for repeat in range(20):

        for i in range(182):

            for j in range(10):

                if coefs[i] > highest_coefs[j] and i not in best_coefs:

                    highest_coefs[j] = coefs[i]

                    best_coefs[j] = i

    output = {}

    for j in range(10):

        output[headers[best_coefs[j]]] = highest_coefs[j]

    for key in output:

        print(f'{key}: {output[key]}')

        

if __name__=='__main__':

    main()
import csv

import pandas

import numpy as np

from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt



def main():

    dtypes = {

        'rating': float,

        'bayes_rating': float

    }

    headers = []

    with open('../input/bgggamesdata/bgg_games_data/mechanics_data.csv') as data:

        csvreader = csv.reader(data)

        for row in csvreader:

            header = row

            for entry in header:

                if entry not in dtypes:

                    dtypes[entry] = bool

                    headers.append(entry)

            break

    data = pandas.read_csv('../input/bgggamesdata/bgg_games_data/mechanics_data.csv', dtype=dtypes)

    columns = []

    for i in range(182):

        columns.append('bayes_rating')

    Y = data[columns].values.reshape(-1, 182)

    data = data.drop(columns=['bayes_rating', 'rating'])

    X = data.values.reshape(-1,182)

    linear_regressor = LinearRegression()

    linear_regressor.fit(X, Y)

    Y_pred = linear_regressor.predict(X)

    plt.scatter(X, Y)

    plt.plot(X, Y_pred, color='red')

    plt.show()

    

if __name__=='__main__':

    main()