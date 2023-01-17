# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# Read in all the data

df_train = pd.read_csv('../input/train.csv', nrows=500000)

df_test = pd.read_csv('../input/test.csv', nrows=500000)

cliente_tabla = pd.read_csv('../input/cliente_tabla.csv')

producto_tabla = pd.read_csv('../input/producto_tabla.csv')

town_state = pd.read_csv('../input/town_state.csv')
# What data columns do each hold?

print('Train data columns')

print(df_train.columns)



print('Test data columns')

print(df_test.columns)



print('Client table data columns')

print(cliente_tabla.columns)



print('Prodcut table data columns')

print(producto_tabla.columns)



print('Town state columns')

print(town_state.columns)
# Let's quickly merge the tables

df_train = pd.merge(df_train, cliente_tabla, on="Cliente_ID", how="left")

df_train = pd.merge(df_train, producto_tabla, on="Producto_ID", how="left")

df_train = pd.merge(df_train, town_state, on="Agencia_ID", how="left")
df_train.describe()
# Let's look at the distribution of the variable we are trying

# to describe - 'demanda_uni_equil'

# Let's create a histogram of this column of data

target = df_train['Demanda_uni_equil'].tolist()



import matplotlib.pyplot as plt

%matplotlib inline

plt.hist(target, bins=30)
# Since the distribution has large outliers we can reduce the range

plt.hist(target, bins=30, range=(0,30))

plt.title('Distribution of target values')

plt.xlabel('Demanda_uni_equil')

plt.ylabel('Count')
# Since we only have to predict the value for 'demanda_uni_equil' 

# let's try submitting a model with the mean and with the mode

# of the training data



# Two ways of printing out most common values

# As a data frame

print(df_train.Demanda_uni_equil.value_counts()[0:10])

print('\n')

# As a list using Counter

from collections import Counter

print(Counter(target).most_common(10))
# Generate a submission by replacing all the values in 

# 'Demanda_uni_equil' by the mode 2

# I will pass these files into Kaggle and see the output



#sub_mode = pd.read_csv('../input/sample_submission.csv')

#sub_mode['Demanda_uni_equil'] = 2

#sub_mode.to_csv('mode.csv', index=False)



#sub_mean = pd.read_csv('../input/sample_submission.csv')

#sub_mean['Demanda_uni_equil'] = 6.9

#sub_mean.to_csv('mean.csv', index=False)
# Does time series have any effect on the data?

# Let's first select all the data with a demand < 20

psuedo_time = df_train.loc[df_train.Demanda_uni_equil < 20].index.tolist()

target = df_train.loc[df_train.Demanda_uni_equil < 20].Demanda_uni_equil.tolist()





plt.hist2d(psuedo_time, target, bins=[50, 20])
# In order to avoid having to submit to Kaggle to figure out the

# error, let's instead write our own root mean squared logarithmic

# error function



import math

def rmsle(y, y_pred):

    assert len(y) == len(y_pred)

    terms_to_sum = [(math.log(y_pred[i] + 1) - math.log(y[i] + 1))

                   ** 2.0 for i, pred in enumerate(y_pred)]

    return ((1.0 / len(y)) * sum(terms_to_sum)) ** 0.5



# Let's now pass in all values from 3-12 to find the best naive 

# estimate.

errors = []

for num in range(3,12):

    length = len(df_train.Demanda_uni_equil)

    errors += [rmsle(df_train.Demanda_uni_equil, [num for i in 

                                                range(length)])]

print(errors)

bestNaive = np.argmin(errors)

print(bestNaive)
# Look at the other columns

semana = df_train['Semana']

print(semana.value_counts())



print("Thus we see that the first 500000 rows of the data only includes data from week 3")

print('\nWe want to sample from the entire distribution\n')



# Let's look at the 'Semana' column for the entire dataset

timing = pd.read_csv('../input/train.csv', usecols=['Semana', 

                                                   'Demanda_uni_equil'])

print('Size: ' + str(timing.shape))

#print(timing['Semana'].value_counts())



#Let's visualize this with a histogram

#plt.hist(timing['Semana'].tolist(), bins=7)
df_train = DataFrame()

chunksize = 10**7

for chunk in pd.read_csv('../input/train.csv', chunksize=chunksize):

    samplesize = len(chunk) / 10

    df_train = df_train.append(chunk.sample(n=samplesize))
# Since this is too much data to read in and process at once

# we can instead sample randomly from the dataset



import random



num_rows = 74180464

sample_size = 500000 #desired sample size

filename = "../input/train.csv"

skip = sorted(random.sample(range(num_rows),num_rows - sample_size))

df = pd.read_csv(filename, skiprows=skip)
# First how do agencies predict return rate?

# How many unique agencies (distributors) are there?



# Let's add in the column names again

df.columns = ['Semana', 'Agencia_ID', 'Canal_ID', 'Ruta_SAK', 'Cliente_ID',

       'Producto_ID', 'Venta_uni_hoy', 'Venta_hoy', 'Dev_uni_proxima',

       'Dev_proxima', 'Demanda_uni_equil']



print(len(df.Agencia_ID.unique()))



# What is the distribution for how many units/ week these agencies sell?



def return_rate(a,b):

    print(a)

    df['return_rate'] = 1

    #print(df)

    return df



agencia_groups = df.groupby(by='Agencia_ID')

#print(agencia_groups.count()) # This tells us the number of rows for each agency

agencia_sums = agencia_groups.sum() # This sums all of the units sold and returned per agency

#print(agencia_sums)

#for d, r in agencia_sums['Venta_uni_hoy']:

#    print(agencia_sums[row])

    #agencia_sums[row]['return_rate'] = agencia_sums[row]['Dev_uni_proxima'] / agencia_sums[row]['Venta_uni_hoy']



print(agencia_sums)

test = agencia_sums.aggregate(np.sum)

print(test)

#agencia_sums = agencia_sums['Venta_uni_hoy', 'Dev_uni_proxima'].map(return_rate)

#print(agencia_sums)



# What are the top agencies in terms of number of units sold?





# How can we use this information to visualize the correlation between agency and 

# return rate?
df_train = pd.read_csv('../input/train.csv', nrows=1000000)

df_train.Semana.unique()
# Why do we aggregate the products based on week and product ID? What does that

# do for us?

# Why did we do all the stemming of the words?

# Why do we create a bag of words and vectorize?

# What do we do once we have all the word roots and the number of times they appear?

# How would I create the top demand plot?

# Aggregate the products by name and count. Leaves me with those groups. 

# How do I sum? Get that column and call sum().  Then sort and graph

# Why did we do np.log1p(df)? What does this graph tell us?

# What are the actual units of the x axis?

# How did they come to the conlucsion that some products have no sales but have returns?

# Group by product name and sum the columns

# How could we statistically check if there is a difference of demand from week to week?

# What have I learned from this...my goal is to predict the output based on input