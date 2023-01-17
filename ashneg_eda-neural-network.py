import numpy as np 

import pandas as pd

import os

import seaborn as sns

import numpy as np

import matplotlib.pyplot as plt

import plotly.graph_objs as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

from plotly import tools

import plotly.plotly as py

import plotly.figure_factory as ff

from sklearn import preprocessing

import itertools

import tensorflow as tf

%matplotlib inline

print(os.listdir("../input"))
df = pd.read_excel("../input/loan.xlsx")

sol = pd.read_excel("../input/Data_Dictionary.xlsx")

# Best to save a copy in case data gets corrupted

df_copy = df.copy()
# Get idea of genral structre of dataet

df.head()
# Getting details which can't be seen directly by structre

df.info()
# Checking all the null and NaN values in dataset

print(df.isnull().sum().value_counts())
#Columns Details

df.columns
# Removing useless coloumns

df.drop(['emp_title', 'desc', 'zip_code', 'url', 'title'], axis=1, inplace=True)
# Basic details about loan

print(df['loan_amnt'].describe())

print('----------------------------------------------------------------------------------------------------------')

print(df['funded_amnt'].describe())
fig, ax = plt.subplots(1, 2, figsize=(16,5))





loan_amount = df["loan_amnt"].values

funded_amount = df["funded_amnt"].values



sns.distplot(loan_amount, ax=ax[0], color="#c42d1f")

ax[0].set_title("Requested amount by the borrower", fontsize=16)

sns.distplot(funded_amount, ax=ax[1], color="#1F8A3F")

ax[1].set_title("Amount Funded", fontsize=16)
# intrest rate distribution

df['int_round'] = df['int_rate'].round(0).astype(int)



plt.figure(figsize = (10,8))



#Exploring the Int_rate

plt.subplot(211)

g = sns.distplot(np.log(df["int_rate"]))

g.set_xlabel("", fontsize=12)

g.set_ylabel("Distribuition", fontsize=12)

g.set_title("Int Rate Log distribuition", fontsize=20)



plt.show()
# Getting info about 

print('Types:\t', df["loan_status"].unique())

print('----------------------------------------------------------------------------------------------------------\n')

print('Distribution:\n\n',df["loan_status"].describe())

print('----------------------------------------------------------------------------------------------------------\n')

print('No. of bad loans present:\t',df.loan_status.str.contains(r'Charged Off').sum())

print('Percentage of bad loans:\t',(df.loan_status.str.contains(r'Charged Off').sum()/39717)*100)
#correlation matrix

df_correlations = df.corr()

# f, ax = plt.subplots(figsize=(12, 10))

# sns.heatmap(corrmat, vmax=.8, square=True);







trace = go.Heatmap(z=df_correlations.values,

                   x=df_correlations.columns,

                   y=df_correlations.columns,

                  colorscale=[[0.0, 'rgb(165,0,38)'], 

                              [0.1111111111111111, 'rgb(215,48,39)'], 

                              [0.2222222222222222, 'rgb(244,109,67)'], 

                              [0.3333333333333333, 'rgb(253,174,97)'], 

                              [0.4444444444444444, 'rgb(254,224,144)'], 

                              [0.5555555555555556, 'rgb(224,243,248)'], 

                              [0.6666666666666666, 'rgb(171,217,233)'], 

                              [0.7777777777777778, 'rgb(116,173,209)'], 

                              [0.8888888888888888, 'rgb(69,117,180)'], 

                              [1.0, 'rgb(49,54,149)']],

            colorbar = dict(

            title = 'Level of Correlation',

            titleside = 'top',

            tickmode = 'array',

            tickvals = [-0.52,0.2,0.95],

            ticktext = ['Negative Correlation','Low Correlation','Positive Correlation'],

            ticks = 'outside'

        )

                  )





layout = {"title": "Correlation Heatmap"}

sns.heatmap(df_correlations, vmax=.8, square=True);

# data=[trace]



# fig = dict(data=data, layout=layout)

# iplot(fig, filename='labelled-heatmap')
df['Default_Binary'] = int(0)

for index, value in df.loan_status.iteritems():

    if value == 'Default':

        df.set_value(index,'Default_Binary',int(1))

    if value == 'Charged Off':

        df.set_value(index, 'Default_Binary',int(1))

    if value == 'Late (31-120 days)':

        df.set_value(index, 'Default_Binary',int(1))    

    if value == 'Late (16-30 days)':

        df.set_value(index, 'Default_Binary',int(1))

    if value == 'Does not meet the credit policy. Status:Charged Off':

        df.set_value(index, 'Default_Binary',int(1))
df['Purpose_Cat'] = int(0) 

for index, value in df.purpose.iteritems():

    if value == 'debt_consolidation':

        df.set_value(index,'Purpose_Cat',int(1))

    if value == 'credit_card':

        df.set_value(index, 'Purpose_Cat',int(2))

    if value == 'home_improvement':

        df.set_value(index, 'Purpose_Cat',int(3))    

    if value == 'other':

        df.set_value(index, 'Purpose_Cat',int(4))    

    if value == 'major_purchase':

        df.set_value(index,'Purpose_Cat',int(5))

    if value == 'small_business':

        df.set_value(index, 'Purpose_Cat',int(6))

    if value == 'car':

        df.set_value(index, 'Purpose_Cat',int(7))    

    if value == 'medical':

        df.set_value(index, 'Purpose_Cat',int(8))   

    if value == 'moving':

        df.set_value(index, 'Purpose_Cat',int(9))    

    if value == 'vacation':

        df.set_value(index,'Purpose_Cat',int(10))

    if value == 'house':

        df.set_value(index, 'Purpose_Cat',int(11))

    if value == 'wedding':

        df.set_value(index, 'Purpose_Cat',int(12))    

    if value == 'renewable_energy':

        df.set_value(index, 'Purpose_Cat',int(13))     

    if value == 'educational':

        df.set_value(index, 'Purpose_Cat',int(14))
df_train = pd.get_dummies(df.purpose).astype(int)



df_train.columns = ['debt_consolidation','credit_card','home_improvement',

                     'other','major_purchase','small_business','car','medical',

                     'moving','vacation','house','wedding','renewable_energy','educational']



# Also add the target column we created at first

df_train['Default_Binary'] = df['Default_Binary']

df_train.head()
x = np.array(df.int_rate.values).reshape(-1,1) 

min_max_scaler = preprocessing.MinMaxScaler()

x_scaled = min_max_scaler.fit_transform(x)

df['int_rate_scaled'] = pd.DataFrame(x_scaled)

print (df.int_rate_scaled[0:5])
x = np.array(df.funded_amnt.values).reshape(-1,1) 

min_max_scaler = preprocessing.MinMaxScaler()

x_scaled = min_max_scaler.fit_transform(x)

df['funded_amnt_scaled'] = pd.DataFrame(x_scaled)

print (df.funded_amnt_scaled[0:5])
df_train['int_rate_scaled'] = df['int_rate_scaled']

df_train['funded_amnt_scaled'] = df['funded_amnt_scaled']
training_set = df_train[0:23000] # Train on first 500k rows

testing_set = df_train[23001:30000] # Test on next 400k rows

prediction_set = df_train[30001:] # Predict on final ~87k rows



COLUMNS = ['debt_consolidation','credit_card','home_improvement',

           'other','major_purchase','small_business','car','medical',

           'moving','vacation','house','wedding','renewable_energy','educational',

           'funded_amnt_scaled','int_rate_scaled','Default_Binary']   



FEATURES = ['debt_consolidation','credit_card','home_improvement',

           'other','major_purchase','small_business','car','medical',

           'moving','vacation','house','wedding','renewable_energy','educational',

           'funded_amnt_scaled','int_rate_scaled'] 



#CONTINUOUS_COLUMNS = ['funded_amnt_scaled','int_rate_scaled'] 

#CATEGORICAL_COLUMNS = ['Purpose_Cat']



LABEL = 'Default_Binary'



def input_fn(data_set):

    ### Simple Version ######

    feature_cols = {k: tf.constant(data_set[k].values) for k in FEATURES} # Working method for continous data DO NOT DELETE 

    labels = tf.constant(data_set[LABEL].values)

    return feature_cols, labels
learning_rate = 0.01

feature_cols = [tf.contrib.layers.real_valued_column(k)

              for k in FEATURES]

#config = tf.contrib.learn.RunConfig(keep_checkpoint_max=1) ######## DO NOT DELETE

regressor = tf.contrib.learn.DNNRegressor(

                    feature_columns=feature_cols,

                    optimizer=tf.train.GradientDescentOptimizer(learning_rate=learning_rate),

                    hidden_units=[10, 20, 10], )



regressor.fit(input_fn=lambda: input_fn(training_set), steps=500)
# Score accuracy

ev = regressor.evaluate(input_fn=lambda: input_fn(testing_set), steps=1)

loss_score = ev["loss"]

print("Loss: {0:f}".format(loss_score))
y = regressor.predict(input_fn=lambda: input_fn(prediction_set))

predictions = list(itertools.islice(y, 87378))

plt.plot(prediction_set.int_rate_scaled, predictions, 'ro')

plt.ylabel("Model Prediction Value")

plt.xlabel("Interest Rate of Loan (Scaled between 0-1)")

plt.show()
plt.plot(prediction_set.funded_amnt_scaled, predictions, 'ro')

plt.ylabel("Model Prediction Value")

plt.xlabel("Funded Amount of Loan (Scaled between 0-1)")

plt.show()