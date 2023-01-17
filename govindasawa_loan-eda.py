import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec
url = '/kaggle/input/loan-prediction-problem-dataset'

train_url = url + '/train_u6lujuX_CVtuZ9i.csv'

test_url = url +'test_Y3wMUE5_7gLdaTN.csv'
train_data = pd.read_csv(train_url)

print(train_data.info())
train_data.drop(['Loan_ID'], axis=1,inplace=True)
def cateInfo(data):

    categorical = set(data.select_dtypes(include='object')) - {'Loan_ID'}

    no_of_classes = {col:len(set(data[col]) - {np.nan}) for col in categorical}

    

    no_of_classes = pd.DataFrame(no_of_classes.values(), index = no_of_classes.keys(), columns=['n_classes']).sort_values(by='n_classes')

    no_of_classes.plot(kind='bar')

    plt.show()
cateInfo(train_data)
def catePlot(feature):

    plt.title(feature.name)

    feature.value_counts(normalize=True).plot(kind='bar')

    plt.show()



for col in train_data.select_dtypes(include='object'):

    catePlot(train_data[col])
def contPlot(feature):

    _, axs = plt.subplots(1,2)

    axs[0].set_title(feature.name)

    axs[0].boxplot(feature)

    axs[1].hist(feature, bins=50)

    plt.show()



for col in train_data.select_dtypes(exclude='object'):

    contPlot(train_data[col].dropna())