#-*- coding:utf-8 -*-

import os
import re
import csv
import subprocess
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import cross_val_score

VERSION = '12'

# NOTE:
# BASE_DIR is ../input in kaggle competition
BASE_DIR = '../input'
TRAIN_DATA_LIMIT = 30000

MAX_BUFFER = 10000


def info(message):
    def _func(func):
        def wrapper(*args):
            print("[INFO] %s" % message)
            ret = func(*args)
            print("[INFO] Done!")
            return ret
        return wrapper
    return _func


class Model():
    def __init__(self):
        self.lookup = {
            'name': {},
            'category_name': {},
            'brand_name': {},
        }

    @info('loading train data into database...')
    def load_train_data(self):
        data = []
        n_lack_data = 0

        path = os.path.join(BASE_DIR, 'train.tsv')
        with open(path) as file:
            reader = csv.reader(file, delimiter="\t")

            # skip header
            next(reader, None)

            for row in reader:
                if len(row) == 8:
                    data.append([
                        # id
                        int(row[0]),

                        # name
                        row[1],

                        # item_condition_id
                        # int(row[2]),

                        # category_name
                        row[3],

                        # brand_name
                        row[4],

                        # price
                        float(row[5]),

                        # shipping
                        # int(row[6]),

                        # item_description
                        # row[7]
                    ])

                    # prepare lookup table
                    # for name
                    # words = re.split('\s', row[1])
                    # for word in words:
                    #     if not word is None and word != "" and word not in self.lookup['name']:
                    #         self.lookup['name'][word] = len(self.lookup['name'])

                    # for category name
                    words = re.split('\s', row[3])
                    for word in words:
                        if not word is None and word != "" and word not in self.lookup['category_name']:
                            self.lookup['category_name'][word] = len(self.lookup['category_name'])

                    # for brand name
                    words = re.split('\s', row[4])
                    for word in words:
                        if not word is None and word != "" and word not in self.lookup['brand_name']:
                            self.lookup['brand_name'][word] = len(self.lookup['brand_name'])

                    if len(data) >= TRAIN_DATA_LIMIT:
                        return data

                else:
                    n_lack_data += 1

        print("%d lack data found." % n_lack_data)

        return data

    def _vectorize(self, key, value):
        """
          returns np.array
        """
        if not key in self.lookup:
            raise NameError(key)

        words = re.split('\s', value)
        x = np.zeros(len(self.lookup[key]))
        for word in words:
            if not word is None and word != "" and word in self.lookup[key]:
                idx = self.lookup[key][word]
                x[idx] += 1
        return x

    @info('preparing data...')
    def prepare_data(self, data):
        """
          returns x and y, which are indicator variable and target
        """

        x = []
        y = []
        for row in data:
            # fv_name = self._vectorize('name', row[1])
            fv_brand = self._vectorize('brand_name', row[2])
            fv_category = self._vectorize('category_name', row[3])
            fv = np.concatenate((fv_brand, fv_category))
            x.append(fv)
            y.append(row[4]) # price

        return (x, y)

    @info('training model...')
    def train(self, x, y):
        self.reg = linear_model.SGDRegressor(max_iter=100)
        self.reg.fit(x, y)

    @info('predict from file...')
    def predict_from_file(self, path):
        with open(path, "r") as file:
            reader = csv.reader(file, delimiter="\t")

            # skip header
            next(reader, None)

            ids = []
            pred = []
            buffer = []
            n_data = 0
            for row in reader:
                # number of data
                n_data += 1

                if len(buffer) < MAX_BUFFER:
                    ids.append(row[0])
                    buffer.append(
                        np.concatenate((
                            self._vectorize('brand_name', row[3]),
                            self._vectorize('category_name', row[4]),
                        ))
                    )

                else:
                    pred += self.reg.predict(buffer).tolist()

                    ids.append(row[0])
                    buffer = [
                        np.concatenate((
                            self._vectorize('brand_name', row[3]),
                            self._vectorize('category_name', row[4]),
                        ))
                    ]

            # out of tsv read loop
            if len(buffer) > 0:
                ids += [x[0] for x in buffer]
                pred += self.reg.predict(buffer).tolist()

            return pd.DataFrame([ids[0:n_data], pred[0:n_data]]).T

    @info('training model and validating...')
    def train_and_validation(self, x, y):
        reg = linear_model.SGDRegressor(max_iter=100)
        scores = cross_val_score(reg, x, y, cv=5)
        print(scores)


def main():
    model = Model()

    if not os.path.exists(os.path.join(BASE_DIR, 'train.tsv')):
        train_7z_file_path = os.path.join(BASE_DIR, 'train.tsv.7z')
        subprocess.call(['7z', 'e', train_7z_file_path, '-y', '-o' + BASE_DIR + ''])

    if not os.path.exists(os.path.join(BASE_DIR, 'test.tsv')):
        test_7z_file_path = os.path.join(BASE_DIR, 'test.tsv.7z')
        subprocess.call(['7z', 'e', test_7z_file_path, '-y', '-o' + BASE_DIR + ''])

    # train
    train_data = model.load_train_data()
    x_train, y_train = model.prepare_data(train_data)
    model.train(x_train, y_train)

    # free memory for variables used in training phase
    del train_data
    del x_train
    del y_train

    frame = model.predict_from_file(os.path.join(BASE_DIR, 'test.tsv'))
    frame.columns = ['test_id', 'price']
    frame.to_csv('submission_v' + VERSION + '.csv', index=False)

if __name__ == '__main__':
    main()
