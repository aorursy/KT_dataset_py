# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from scipy import stats

from matplotlib import pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('../input/creditcardfraud/creditcard.csv')

df.info()
df.describe()
n = len(df)

n_fradulent = len(df[df.Class == 1])

n_genuine = len(df[df.Class == 0])

percent_fradulent = n_fradulent / n

percent_genuine = n_genuine / n



print(f'number of genuine transactions: {n_genuine} \nnumber of fradulent transactions: {n_fradulent}')

print(f'percent of genuine transactions: {percent_genuine:.2%} \npercent of fradulent transactions: {percent_fradulent:.2%}')

def plot_pdf(df, column):

    def plot_graph(values):

        hist = np.histogram(values, bins=50)

        dist = stats.rv_histogram(hist)

        color = 'orange' if values.name == 'fradulent' else 'blue'

        plt.hist(values, density=True, bins=50, color=color, alpha=0.3)

        

        freq, values = hist

        pairs = values[:-1], values[1:]

        x = np.average(pairs, axis = 0)

        plt.plot(x, dist.pdf(x), label='PDF', color=color)

    fradulent = df.loc[df.Class == 1, column].rename("fradulent")

    genuine = df.loc[df.Class == 0, column].rename("genuine")

    plt.title(f'PDF for {column}')

    plot_graph(fradulent)

    plot_graph(genuine)

    plt.show()

for column in df.columns[:-1]:

    plot_pdf(df, column)
def plot_cdf(df, column):

    def plot_graph(values):

        hist = np.histogram(values, bins=50)

        dist = stats.rv_histogram(hist)

        color = 'orange' if values.name == 'fradulent' else 'blue'

        plt.hist(values, density=True, bins=50, color=color, alpha=0.3, cumulative=True)

        

        freq, values = hist

        pairs = values[:-1], values[1:]

        x = np.average(pairs, axis = 0)

        plt.plot(x, dist.cdf(x), label='CDF', color=color)

    fradulent = df.loc[df.Class == 1, column].rename("fradulent")

    genuine = df.loc[df.Class == 0, column].rename("genuine")

    plt.title(f'CDF for {column}')

    plot_graph(fradulent)

    plot_graph(genuine)

    plt.show()

for column in df.columns[:-1]:

    plot_cdf(df, column)
def test_statistic(table, column, func):

    fradulent_n = table.loc[table.Class == 1]['Class'].count()

    fradulent = table.loc[table.Class == 1, column]

    genuine = table.loc[table.Class == 0, column]

    diff = abs(func(fradulent) - func(genuine))

    diffs = []

    for _ in range(100):

        df_shuffled = table.sample(frac=1)

        estimated_diff = abs(func(df_shuffled[:fradulent_n][column]) - func(df_shuffled[fradulent_n:][column]))

        diffs.append(estimated_diff)

    #print(diff)

    

    hist = np.histogram(diffs, bins=10)

    dist = stats.rv_histogram(hist)

    freq, values = hist

    pairs = values[:-1], values[1:]

    x = np.average(pairs, axis = 0)

    

    plt.plot(x, dist.cdf(x), label='CDF')

    plt.title(f'difference test for {column}')

    plt.axvline(diff, color = 'r')

    plt.show()

    

    return column, diff, dist.sf(diff)



for column in df.columns[:-1]:

    test_statistic(df, column, np.mean)
def test_statistic(table, column, func, n, iter):

    fradulent = table.loc[table.Class == 1, column]

    genuine = table.loc[table.Class == 0, column]

    diff = abs(func(fradulent) - func(genuine)) # real diff in the population

    diffs = []

    position = n // 2

    for _ in range(iter): # generate diffs between the whole population

        sub_genuine = genuine.sample(n)

        sub_fradulent = fradulent.sample(n)

        balanced_df = pd.concat([sub_genuine, sub_fradulent], ignore_index=True)

        shuffled_df = balanced_df.sample(frac=1)

        estimated_diff = abs(func(shuffled_df[:position]) - func(shuffled_df[position:]))

        diffs.append(estimated_diff)

    #print(diff)

    

    hist = np.histogram(diffs, bins=10)

    dist = stats.rv_histogram(hist)

    freq, values = hist

    pairs = values[:-1], values[1:]

    x = np.average(pairs, axis = 0)

    

    plt.plot(x, dist.cdf(x), label='CDF')

    plt.title(f'difference test for {column}')

    plt.axvline(diff, color = 'r')

    plt.show()

    

    return column, diff, dist.sf(diff)
mean_statistic = [test_statistic(df, column, np.mean, 100, 100) for column in df.columns[:-1]]

mean_pvalue = [pvalue for (feature, diff, pvalue) in mean_statistic]

mean_pvalue
std_statistic = [test_statistic(df, column, np.std, 100, 100) for column in df.columns[:-1]]

std_pvalue = [pvalue for (feature, diff, pvalue) in std_statistic]

std_pvalue
test_statistic_table = pd.DataFrame(list(zip(df.columns, mean_pvalue, std_pvalue)), columns = ['feature', 'mean_pvalue', 'std_pvalue'])

test_statistic_table
relevant_features = test_statistic_table.loc[(test_statistic_table.mean_pvalue < 0.05) & (test_statistic_table.std_pvalue < 0.05), 'feature']

relevant_features
df_corr = df.loc[:, df.columns != 'Class'].corr()

cax = plt.matshow(df_corr)

fig = plt.gcf()

fig.set_size_inches(18.5, 10.5)

fig.colorbar(cax)