# Importing libraries



# overall libraries

import pandas as pd

import numpy as np

import warnings

warnings.filterwarnings('ignore')

from collections import OrderedDict

from IPython.core.pylabtools import figsize

import re



# plotting libraries

import seaborn as sns

sns.set_style('white')

import matplotlib.pyplot as plt

import matplotlib.lines as mlines

%matplotlib inline



# bayesian libraries

import pymc3 as pm

import arviz as az

import theano

import theano.tensor as T

floatX = theano.config.floatX

import itertools

from pymc3.theanof import set_tt_rng, MRG_RandomStreams



# sklearn libraries

import sklearn

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import scale

from sklearn.model_selection import train_test_split

from sklearn.datasets import make_moons

from sklearn.metrics import log_loss, roc_auc_score, roc_curve, auc, precision_recall_curve, confusion_matrix
# setting the display so you can see all the columns and all the rows



pd.set_option("max_columns", None)

pd.set_option("max_rows", None)
# creating the DataFrame



df = pd.read_excel('../input/covid19/dataset.xlsx', encoding='utf8')
# Checking how the df imported



df.head()
# Removing unwanted columns for the task 1



df_task1 = df.drop(['Patient addmited to regular ward (1=yes, 0=no)', 'Patient addmited to semi-intensive unit (1=yes, 0=no)', 'Patient addmited to intensive care unit (1=yes, 0=no)'], axis = 1)

df_task1.head()
# Checking the unique values for the SARS-Cov-2 exam result (our target for this task)



df_task1['SARS-Cov-2 exam result'].unique()
# Replacing negative to 0 an positive to 1 and then checking if it worked



df_task1['SARS-Cov-2 exam result'] = df_task1['SARS-Cov-2 exam result'].replace({'negative': 0, 'positive': 1})

df_task1['SARS-Cov-2 exam result'].unique()
# checking the categorical variables



df_task1.select_dtypes(include = ['object']).columns
# replacing the values to make them numerical, I am doing them by hand to make sure all the exams make sense.

# This is possible because there aren't many categorical variables.

# To do this, I checked all the variables unique values and created an unique dictionary



df_task1.loc[:,'Respiratory Syncytial Virus':'Parainfluenza 2'] = df_task1.loc[:,'Respiratory Syncytial Virus':'Parainfluenza 2'].replace({'not_detected':0, 'detected':1})

df_task1.loc[:,'Influenza B, rapid test':'Strepto A'] = df_task1.loc[:,'Influenza B, rapid test':'Strepto A'].replace({'negative':0, 'positive':1})

df_task1['Urine - Esterase'] = df_task1['Urine - Esterase'].replace({'absent':0})

df_task1['Urine - Aspect'] = df_task1['Urine - Aspect'].replace({'clear':0, 'cloudy':2, 'altered_coloring':3, 'lightly_cloudy':1})

df_task1['Urine - pH'] = df_task1['Urine - pH'].replace({'6.5':6.5, '6.0':6.0,'5.0':5.0, '7.0':7.0, '5':5, '5.5':5.5,

       '7.5':7.5, '6':6, '8.0':8.0})

df_task1['Urine - Hemoglobin'] = df_task1['Urine - Hemoglobin'].replace({'absent':0, 'present':1})

df_task1.loc[:,'Urine - Bile pigments':'Urine - Nitrite'] = df_task1.loc[:,'Urine - Bile pigments':'Urine - Nitrite'].replace({'absent':0})

df_task1.loc[:,'Urine - Urobilinogen':'Urine - Protein'] = df_task1.loc[:,'Urine - Urobilinogen':'Urine - Protein'].replace({'absent':0, 'normal':1})

df_task1['Urine - Hemoglobin'] = df_task1['Urine - Hemoglobin'].replace({'absent':0, 'present':1, 'not_done':np.nan})

df_task1['Urine - Leukocytes'] = df_task1['Urine - Leukocytes'].replace({'38000':38000, '5942000':5942000, '32000':32000, '22000':22000,'<1000': 900, '3000': 3000,'16000':16000, '7000':7000, '5300':5300, '1000':1000, '4000':4000, '5000':5000, '10600':106000, '6000':6000, '2500':2500, '2600':2600, '23000':23000, '124000':124000, '8000':8000, '29000':29000, '2000':2000,'624000':642000, '40000':40000, '3310000':3310000, '229000':229000, '19000':19000, '28000':28000, '10000':10000,'4600':4600, '77000':77000, '43000':43000})

df_task1['Urine - Crystals'] = df_task1['Urine - Crystals'].replace({'Ausentes':0, 'Urato Amorfo --+':1, 'Oxalato de Cálcio +++':3,'Oxalato de Cálcio -++':2, 'Urato Amorfo +++':4})

df_task1.loc[:,'Urine - Hyaline cylinders':'Urine - Yeasts'] = df_task1.loc[:,'Urine - Hyaline cylinders':'Urine - Yeasts'].replace({'absent':0})

df_task1['Urine - Color'] = df_task1['Urine - Color'].replace({'light_yellow':0, 'yellow':1, 'orange':2, 'citrus_yellow':1})

df_task1 = df_task1.replace('not_done', np.NaN)

df_task1 = df_task1.replace('Não Realizado', np.NaN)
# Dropping the patient ID column



df_task1 = df_task1.drop('Patient ID', axis = 1)
# checking if all of the categorical variables were treated



df_task1.select_dtypes(include = ['object']).columns
# checking how the data is distribuited in the dataframe



df_task1.info()
# let's see what are the two columns that are working with int



df_task1.select_dtypes(include = ['int64']).columns
# let's create a rank of missing values



null_count = df_task1.isnull().sum().sort_values(ascending=False)

null_percentage = null_count / len(df_task1)

null_rank = pd.DataFrame(data=[null_count, null_percentage],index=['null_count', 'null_ratio']).T

null_rank
# dropping columns that don't have any content in it



df_task1 = df_task1.drop(['Mycoplasma pneumoniae','Urine - Nitrite', 'Urine - Sugar', 'Partial thromboplastin time (PTT) ', 'Prothrombin time (PT), Activity', 'D-Dimer'], axis = 1)
# let's see the min and max values of the variables to fill their missing values



df_task1.describe().round(2)
# filling missing values with 0



df_task1[['Urine - Leukocytes', 'Urine - pH']] = df_task1[['Urine - Leukocytes', 'Urine - pH']].fillna(0)
# filling missing values with -1



df_task1[['Patient age quantile', 'SARS-Cov-2 exam result', 'Respiratory Syncytial Virus', 'Influenza A', 'Influenza B', 'Parainfluenza 1', 'CoronavirusNL63', 'Rhinovirus/Enterovirus', 'Coronavirus HKU1', 'Parainfluenza 3', 'Chlamydophila pneumoniae', 'Adenovirus', 'Parainfluenza 4', 'Coronavirus229E', 'CoronavirusOC43', 'Inf A H1N1 2009', 'Bordetella pertussis', 'Metapneumovirus', 'Parainfluenza 2', 'Influenza B, rapid test', 'Influenza A, rapid test', 'Strepto A', 'Fio2 (venous blood gas analysis)','Myeloblasts', 'Urine - Esterase', 'Urine - Hemoglobin', 'Urine - Bile pigments', 'Urine - Ketone Bodies', 'Urine - Protein', 'Urine - Crystals', 'Urine - Hyaline cylinders', 'Urine - Granular cylinders', 'Urine - Yeasts', 'Urine - Color']] = df_task1[['Patient age quantile', 'SARS-Cov-2 exam result', 'Respiratory Syncytial Virus', 'Influenza A', 'Influenza B', 'Parainfluenza 1', 'CoronavirusNL63', 'Rhinovirus/Enterovirus', 'Coronavirus HKU1', 'Parainfluenza 3', 'Chlamydophila pneumoniae', 'Adenovirus', 'Parainfluenza 4', 'Coronavirus229E', 'CoronavirusOC43', 'Inf A H1N1 2009', 'Bordetella pertussis', 'Metapneumovirus', 'Parainfluenza 2', 'Influenza B, rapid test', 'Influenza A, rapid test', 'Strepto A', 'Fio2 (venous blood gas analysis)','Myeloblasts', 'Urine - Esterase', 'Urine - Hemoglobin', 'Urine - Bile pigments', 'Urine - Ketone Bodies', 'Urine - Protein', 'Urine - Crystals', 'Urine - Hyaline cylinders', 'Urine - Granular cylinders', 'Urine - Yeasts', 'Urine - Color']].fillna(-1)
# filling all the other missing values with 99



df_task1 = df_task1.fillna(99)
# let's see if there is still any missing values left



null_count = df_task1.isnull().sum().sort_values(ascending=False)

null_percentage = null_count / len(df_task1)

null_rank = pd.DataFrame(data=[null_count, null_percentage],index=['null_count', 'null_ratio']).T

null_rank
# let's now see the description of the dataframe again, because i am pretty sure we will have to apply some sort of normalization technique on it



df_task1.describe()
# let's see visually how our variables are behaving



#df_task1.hist(bins = 50, figsize=(40,40))

#plt.show()
# creating a scaler and using it, disconsidering the target column



scaler = MinMaxScaler()

exam = pd.DataFrame(df_task1['SARS-Cov-2 exam result'], columns = ['SARS-Cov-2 exam result'])

df_scaled = pd.DataFrame(scaler.fit_transform(df_task1.drop('SARS-Cov-2 exam result', axis = 1)), columns = (df_task1.drop('SARS-Cov-2 exam result', axis = 1).columns))
# concatenating all the columns again



df_total = pd.concat([exam, df_scaled], axis = 1)
# checking if the concatening worked



df_total.head()
# doing a correlation rank to see how the exams work with the result of the exam



df_total.corr()['SARS-Cov-2 exam result'].sort_values(ascending=False)
# Let's remove all special characters and spaces from the column names

# We will also make them lowercase



df_total.columns=df_total.columns.str.replace(r'\(|\)|:|,|;|\.|’|”|“|\?|%|>|<|(|)|\\','')

df_total.columns=df_total.columns.str.replace(r'/','')

df_total.columns=df_total.columns.str.replace(' ','')

df_total.columns=df_total.columns.str.replace('"','')

df_total.columns=df_total.columns.str.replace('-','')

df_total.columns=df_total.columns.str.lower()
# let's get a list of all the columns so we can start working on our model



list(df_total.columns)
# renaming the columns



df_total = df_total.rename(columns={"Meancorpuscularhemoglobinconcentration\xa0MCHC": "Meancorpuscularhemoglobinconcentrationxa0MCHC", "Gammaglutamyltransferase\xa0": "Gammaglutamyltransferasexa0", "Ionizedcalcium\xa0": "Ionizedcalciumxa0", "Creatinephosphokinase\xa0CPK\xa0" : "Creatinephosphokinasexa0CPKxa0", 'rods#': 'rods'})
# Creating X and y



X = df_total.drop('sarscov2examresult', axis = 1)

y = df_total['sarscov2examresult']
# let's split the X and y into a test and train set

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=.3, random_state = 42)
def construct_nn(ann_input, ann_output):

    n_hidden = 7



    # Initialize random weights between each layer

    init_1 = np.random.randn(X.shape[1], n_hidden).astype(floatX)

    init_2 = np.random.randn(n_hidden, n_hidden).astype(floatX)

    init_out = np.random.randn(n_hidden).astype(floatX)



    with pm.Model() as neural_network:

        ann_input = pm.Data('ann_input', X_train)

        ann_output = pm.Data('ann_output', Y_train)



        # Weights from input to hidden layer

        weights_in_1 = pm.Normal('w_in_1', 0, sigma=1,

                                 shape=(X.shape[1], n_hidden),

                                 testval=init_1)



        # Weights from 1st to 2nd layer

        weights_1_2 = pm.Normal('w_1_2', 0, sigma=1,

                                shape=(n_hidden, n_hidden),

                                testval=init_2)



        # Weights from hidden layer to output

        weights_2_out = pm.Normal('w_2_out', 0, sigma=1,

                                  shape=(n_hidden,),

                                  testval=init_out)



        # Build neural-network using tanh activation function

        act_1 = pm.math.tanh(pm.math.dot(ann_input,

                                         weights_in_1))

        act_2 = pm.math.tanh(pm.math.dot(act_1,

                                         weights_1_2))

        act_out = pm.math.sigmoid(pm.math.dot(act_2,

                                              weights_2_out))



        # Binary classification -> Bernoulli likelihood

        out = pm.Bernoulli('out',

                           act_out,

                           observed=ann_output,

                           total_size=Y_train.shape[0]

                          )

    return neural_network



# using the model on the train data

neural_network = construct_nn(X_train, Y_train)
# set the package-level random number generator

set_tt_rng(MRG_RandomStreams(42))
%%time



#  ADVI variational inference algorithm

with neural_network:

    inference = pm.ADVI()

    approx = pm.fit(n=30000, method=inference)
# lets predict on the hold-out set using a posterior predictive check (PPC)



trace = approx.sample(draws=5000)
# We can get predicted probability from model

neural_network.out.distribution.p
# create symbolic input

x = T.matrix('X')

n = T.iscalar('n')

x.tag.test_value = np.empty_like(X_train[:10])

n.tag.test_value = 100

_sample_proba = approx.sample_node(neural_network.out.distribution.p,

                                   size=n,

                                   more_replacements={neural_network['ann_input']: x})

# It is time to compile the function

# No updates are needed for Approximation random generator

# Efficient vectorized form of sampling is used

sample_proba = theano.function([x, n], _sample_proba)



# Create bechmark functions

def production_step1():

    pm.set_data(new_data={'ann_input': X_test, 'ann_output': Y_test}, model=neural_network)

    ppc = pm.sample_posterior_predictive(trace, samples=500, progressbar=False, model=neural_network)



    # Use probability of > 0.5 to assume prediction of class 1

    pred = ppc['out'].mean(axis=0) > 0.5



def production_step2():

    sample_proba(X_test, 500).mean(0) > 0.5
%timeit production_step1()

# checking the performance of the first function
%timeit production_step2()

# checking the performance of the second function
# let's create the prediction



pred = sample_proba(X_test, 500).mean(0) > 0.5
# let's see the accuracy of our model



print('Accuracy = {}%'.format((Y_test == pred).mean() * 100))
# Confusion Matrix



def plot_confusion_matrix(y_test, y_pred):

    cm = confusion_matrix(y_test, y_pred)

    df = pd.DataFrame(cm.T, index=["has disease", "no disease"], columns=["has disease", "no disease"])

    ax = sns.heatmap(df, annot=True)

    ax.set_xlabel("Predicted label")

    ax.set_ylabel("True label")

    return ax



plot_confusion_matrix(Y_test, pred)