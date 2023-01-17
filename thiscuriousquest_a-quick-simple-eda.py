import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(rc={'figure.figsize':(11,8)})
train_df = pd.read_csv("../input/learn-together/train.csv")

train_df.head(3)
soil_type_list = list(train_df.columns[15:-1])

soil_type_EDA_column = train_df[soil_type_list].idxmax(axis=1)

soil_type_EDA_column = soil_type_EDA_column.str.replace('Soil_Type', '')

soil_type_EDA_column = soil_type_EDA_column.astype('int32')

print(soil_type_EDA_column[:3])
_ = plt.hist(soil_type_EDA_column, bins=40)

_ = plt.xlabel('Soil Type')

_ = plt.ylabel('No of instances')

plt.show()
wilderness_area_list = list(train_df.columns[11:15])

wilderness_area_EDA_column = train_df[wilderness_area_list].idxmax(axis=1)

wilderness_area_EDA_column = wilderness_area_EDA_column.str.replace('Wilderness_Area', '')

wilderness_area_EDA_column = wilderness_area_EDA_column.astype('int32')

print(wilderness_area_EDA_column[:3])
_ = plt.hist(wilderness_area_EDA_column)

_ = plt.xlabel('Wilderness Area')

_ = plt.ylabel('No of instances')

plt.show()
features_list = list(train_df.columns[1:10])



def make_boxplot(x, y, data):

    _ = sns.boxplot(x=x, y=y, data=data)

    _ = plt.xlabel('Cover Type')

    _ = plt.ylabel('{}'.format(y))

    plt.show()
for i in features_list:

    make_boxplot('Cover_Type', i, train_df)