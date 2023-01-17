% matplotlib inline



import os

import sys

import warnings



import numpy as np

import matplotlib.pyplot as plt



from pandas import DataFrame, Series, concat
import matplotlib as mpl

from matplotlib import rc

mpl.rcdefaults() # сброс настроек  



font = {'family': 'Arial',

        'weight': 'normal'}

rc('font', **font)
# -*- coding: utf-8 -*-





class Name(object):

    def __init__(self, name):

        self.name = name



    def __str__(self):

        return '%s' % self.name



    def __unicode__(self):

        return '%s' % self.name.decode('utf-8')



    def __repr__(self):

        return '%s' % self.name
from xlrd import open_workbook



default_configuration = {0: {'name': 'id_emis', 'format': 'int'},

                         1: {'name': 'depart', 'format': 'str'},

                         2: {'name': 'region', 'format': 'str'},

                         3: {'name': 'obj', 'format': 'str'},

                         4: {'name': 'objnum', 'format': 'int'},

                         5: {'name': 'channel', 'format': 'str'},

                         6: {'name': 'nomenclature', 'format': 'str'},

                         7: {'name': 'year', 'format': 'int'},

                         8: {'name': 'month', 'format': 'int'},

                         9: {'name': 'volume', 'format': 'float'}}





def get_data_from_file(data_path, configuration=default_configuration,

                       data_file_name='data.xlsx'):

    wb = open_workbook(os.path.join(data_path, data_file_name))



    column_names = [configuration[i]['name'] for i in

                    configuration.keys()]

    column_types = [configuration[i]['format'] for i in

                    configuration.keys()]

    columns_with_str = [i for i in configuration.keys()

                        if configuration[i]['format'] is 'str']

    columns_with_int = [i for i in configuration.keys()

                        if configuration[i]['format'] is 'int']



    sheet = wb.sheets()[0]

    number_of_rows = sheet.nrows

    number_of_columns = sheet.ncols

    items = []

    for row in range(1, number_of_rows):

        values = []

        for col in range(number_of_columns):

            try:

                value = (sheet.cell(row, col).value).encode('utf8')

            except:

                value = (sheet.cell(row, col).value)

            finally:

                value = Name(value)

            try:

                if col in columns_with_str:

                    value = str(value)

                elif col in columns_with_int:

                    value = int(float(str(value)))

                else:

                    value = float(str(value))

            except Exception:

                value = None

            values.append(value)

        items.append(values)

    return DataFrame(items, columns=column_names)
data = get_data_from_file(data_path="../data",

                          data_file_name='data2.xlsx')

column_names = data.columns.values
print(column_names)
def get_groups(data, by_column='region'):

    if by_column not in data.columns.values:

        raise AttributeError("Data does not have the column '{0}'".format(by_column))

    group_list = list(set(data[by_column]))

    return Series(group_list, index=range(1, len(group_list) + 1)).sort_values()





def get_group_data(data, regions=None, objs=None):

    full_regions = get_groups(data=data, by_column='region').values

    full_objs = get_groups(data=data, by_column='obj').values

    if ((regions is None) and (objs is None)):

        return data

    elif regions is None:

        full_regions_flag = True

    elif objs is None:

        full_objs_flag = True

    data_index = data.index.values

    rows = []

    if not isinstance(regions, list):

        regions = [regions]

    for region in regions:

        row = [i for i in data_index if data['region'][i] == region]

        rows.append(row)

    if not isinstance(objs, list):

        objs = [objs]

    for obj in objs:

        row = [i for i in data_index if data['obj'][i] == obj]

        rows.append(row)

    rows = [item for items in rows for item in items]

    return data.loc[rows]
# какой регион продает больше

regions = get_groups(data=data, by_column='region')

sale_region_volume = {}

for region in regions:

    data_region = get_group_data(data=data, regions=region)

    sale_region_volume[region] = sum(data_region['volume'])



# plotting

pos = np.arange(len(regions))

plt.bar(pos, sale_region_volume.values())

plt.xticks(pos + 0.3, ['Region {0}'.format(

    (sale_region_volume.keys()[i])[sale_region_volume.keys()[i].rfind(' '):]) for i in pos], 

           rotation='vertical')

plt.xlabel('Regions')

plt.ylabel('volume of sales')

plt.title('Sales of regions')

plt.show()



print("В регионе '{0}' зафиксирован наибольший объем продаж нефтепродуктов".format(

    max(sale_region_volume, key=sale_region_volume.get)))
# какая АЗС продает больше

objs = get_groups(data=data, by_column='obj')

sale_obj_volume = {}

for obj in objs:

    data_obj = get_group_data(data=data, objs=obj)

    sale_obj_volume[obj] = sum(data_obj['volume'])



# plotting

plt.figure(figsize=(40, 20))

pos = np.arange(len(objs))

plt.bar(pos, sale_obj_volume.values())

plt.xticks(pos + 0.5, ['Object {0}'.format((sale_obj_volume.keys()[i])[sale_obj_volume.keys()[i].rfind(' '):])

                       for i in pos], rotation='vertical')

plt.xlabel('Objects')

plt.ylabel('volume of sales')

plt.title('Sales of Object')

plt.show()



print("По АЗС '{0}' зафиксирован наибольший объем продаж нефтепродуктов".format(

    max(sale_obj_volume, key=sale_obj_volume.get)))
coded_data = {}

for col in column_names:

    coded_data[col] = [(item[item.rfind(" "):]) if isinstance(item, str) else item for item in data[col]]

coded_data = DataFrame(coded_data.values(), index=coded_data.keys()).T
matrix = DataFrame([[sum(get_group_data(data=data, objs=obj, regions=region)['volume']) for region in regions]

                    for obj in objs], index=objs, columns=regions).T



# plotting

fig = plt.figure()

ax = fig.add_subplot(111).matshow(matrix, cmap='Reds')

pos_x = np.arange(len(objs))

plt.xticks(pos_x, ['Object {0}'.format((matrix.columns.values[i])[matrix.columns.values[i].rfind(' '):])

                       for i in pos_x], rotation='vertical')

pos_y = np.arange(len(regions))

plt.yticks(pos_y, ['Region {0}'.format(

    (matrix.index.values[i])[matrix.index.values[i].rfind(' '):]) for i in pos_y])

fig.colorbar(ax, shrink=0.5, aspect=5)





# correlations according to region

sale_region_max_volume = {}

for region in regions:

    data_region = get_group_data(data=data, regions=region)['volume']

    sale_region_max_volume[region] = data_region.max()

new_region_order = Series(sale_region_max_volume).sort_values(ascending =False).index.values



region_encoding = Series(range(len(new_region_order)), index=new_region_order)

region_data_column = [region_encoding[value] for value in data['region']]



fig = plt.figure()

ax = fig.add_subplot(111).scatter(region_data_column, data['volume'])

pos_y = np.arange(len(regions))

plt.xticks(pos_y, ['Region {0}'.format(

    (matrix.index.values[i])[matrix.index.values[i].rfind(' '):]) for i in pos_y], rotation='vertical')

# plt.xtics(pos_y, matrix.index.values)

plt.ylabel("Sales")

plt.title("Dependency of sales volume and Region of sales")



x=region_data_column

y=list(data['volume'].values)

df = DataFrame([x, y]).T

correlation_coefficient = df.corr()[0][1]

plt.show()

print("Correlation coefficient of volume sales values with number of the region group is {cor}.".format(

    cor=correlation_coefficient))
from pandas import isnull



def exclude_nun(data):

    indexs = []

    all_ind = data.index.values

    for i in all_ind:

        if (isnull(data.loc[i])).any():

            indexs.append(i)

    data_new = data.loc[[i for i in all_ind if i not in indexs]]

    return data_new



new_coded = exclude_nun(coded_data)[['id_emis', 'depart', 'region', 'obj', 'objnum', 'channel', 'nomenclature', 'volume']]
from scipy.spatial.distance import cdist

import random



# mics

def get_argmin(matrix, exept):

    arg = matrix.argmin()

    if arg in exept:

        arg = matrix[arg].argmin()

    return arg





def greedy_clustering(data):

    dist_matrix = DataFrame(cdist(data, data, 'eu'))

    point_list = dist_matrix.index.values.tolist()

    start_point = random.choice(point_list)

    seq = [start_point]

    point_list = [el for el in point_list if el not in seq]

    print(start_point, point_list)

    clusters = []

    i = 10

    while i > 0:  # len(point_list) > 0:

        print([cdist(DataFrame(data.loc[s]).T, data.loc[point_list]) for s in seq])

        dist = DataFrame({s: {'min': (cdist(DataFrame(data.loc[s]).T, data.loc[point_list])).min(),

            'argmin': get_argmin(matrix=cdist(DataFrame(data.loc[s]).T, data.loc[point_list]), exept=seq)} for s in seq})

        

        min_to = dist.loc['min'].min()

        to = dist[dist.loc['min'].argmin()].loc['argmin']

        mean_seq = (cdist(data.loc[seq], data.loc[seq])).mean()

        if mean_seq != 0 and min_to > mean_seq:

            clusters.append(seq)

            seq = []

            print('Sequences', clusters)

        seq.append(to)

        point_list = [el for el in point_list if el not in seq]

#         print(seq, point_list)

        i -= 1

    return clusters



# new_coded = exclude_nun(coded_data)[['depart', 'region', 'obj', 'objnum', 'volume']]



# clusters = greedy_clustering(new_coded)

# print(clusters)



# # coloring and plotting

# col = {}

# for c in range(len(clusters)):

#     for el in clusters[c]:

#        col[el] = c

# print(col)

# for column in column_names:

#     plt.figure(column)

#     plt.plot(coded_data[column], c=col.values())
# Исследование временного ряда

time_series = new_coded['volume']

plt.figure('ghj')

plt.plot(time_series)

# plt.title(u"Временной ряд")



from sklearn import linear_model

import pandas as pd



reg = linear_model.LinearRegression()

length = len(time_series)

x=np.array(time_series.index.values)

y=np.array(time_series.values)

X =  x.reshape(length, 1)

Y = y.reshape(length, 1)

reg.fit(X, Y)



# xx = np.arange(list(x)[-1], list(x)[-1]+5)

# XX =  xx.reshape(len(xx), 1)

# YY = reg.predict(XX)

YY_ = reg.predict(X)

plt.plot(X, Y)

plt.plot(X, YY_)

# plt.plot(XX, YY)

plt.show()

print('Linear regression coefficient: ', reg.coef_[0][0])



import statsmodels.api as sm

rng = pd.date_range('1/1/2011', periods=length, freq='M')

ts = Series(list(time_series), index=rng)

res = sm.tsa.seasonal_decompose(ts)

res.plot()


# misc

def get_data_by_value(data, column_name, value):

    column = data[column_name]

    index_list = [i for i in column.index.values if column[i] == value]

    return data.loc[index_list]



# misc

def train_test(data, by='month'):

    if by == 'month':

        months = list(set(data[by]))

        months.sort()

        last_month = months[-1]

        test = get_data_by_value(data=data, column_name=by, value=last_month)

        test_index = test.index.values

        all_index = data.index.values

        train_index = [index for index in all_index if index not in test_index]

        train = data.loc[train_index]

    else:

        raise AttributeError('This {0} functionality is not supported'.format(by))

    return train, test



def prediction(series, period=1, fig_name='Name'):

    length = len(series)

    index = pd.date_range('1/1/2014', periods=length, freq='D')

    time_series_new = Series(time_series, index=index)



    reg1 = linear_model.LinearRegression()

    length = len(time_series)

    x=np.array(time_series.index.values)

    y=np.array(time_series.values)

    X =  x.reshape(length, 1)

    Y = y.reshape(length, 1)

    reg1.fit(X, Y)



    xx = np.arange(list(x)[-1], list(x)[-1] + period)

    XX =  xx.reshape(len(xx), 1)

    YY = reg1.predict(XX)

    YY_ = reg1.predict(X)

#     plt.figure(fig_name)

#     plt.plot(X, Y)

#     plt.plot(X, YY_)

#     plt.plot(XX, YY)

#     plt.title(fig_name)

#     plt.show()

    return X, Y, XX, YY





for period in [1, 3, 6]:

    # first approach: to make a prediction according full volume series

    X1, Y1, XX1, YY1 = prediction(new_coded['volume'], period=period, fig_name='common appraoch')

    plt.figure('common appraoch with period={}'.format(period))

    plt.plot(X1, Y1)

    plt.plot(XX1, YY1)

    plt.title('common appraoch with period={}'.format(period))

        

    # second approach: to make a prediction for each AZS

    X, Y, XX, YY = [], [], [], []

    obj_names = list(set(data['objnum'].values))

    predicted_volume_for_objs = {}

    plt.figure('Each_AZS with period={}'.format(period))

    plt.title('Each_AZS with period={}'.format(period))

    for obj_name in obj_names:

        if not (isnull(obj_name)):

            obj_data = get_data_by_value(new_coded, 'objnum', int(obj_name))

    #         train_data, test_data = train_test(obj_data)

            obj_volume = obj_data['volume']

            X_, Y_, XX_, YY_ = predicted_volume = prediction(obj_volume, period=period, fig_name='Each_AZS')

            X.append(X_)

            Y.append(Y_)

            XX.append(XX_)

            YY.append(YY_)

            predicted_volume_for_objs[obj_name] = predicted_volume

            plt.plot(X_, Y_)

            plt.plot(XX_, YY_)

        else:

            print('Passed nun')

plt.show()