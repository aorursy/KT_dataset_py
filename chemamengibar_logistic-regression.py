import os

import sys



import warnings

warnings.filterwarnings('ignore')



import gc,json, csv, time, string, itertools, copy

import numpy as np

import pandas as pd

#import datetime as dt



from sklearn import linear_model

from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, accuracy_score

from sklearn.model_selection import train_test_split

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import PolynomialFeatures





import scipy.stats as st

import statsmodels.api as sm



import matplotlib.pyplot as plt

DIR_DATA = '../input/'



inputFilePath = DIR_DATA + 'beer-consumption_sao-paulo_2015.csv'
df_a = pd.read_csv( filepath_or_buffer=inputFilePath, sep=',',  quoting=0,  decimal="," )

df_a.columns = ['date', 'temp_mean_c', 'temp_min_c', 'temp_max_c', 'prec', 'is_weekend', 'beer_lt']



df_a['beer_lt'] = df_a['beer_lt'].astype('float')

df_a['counter'] = 1
df_a.head()
# Checking for missing values 

df_a.isnull().sum()
def isOverTarget( row ):

    return 1 if (row['beer_lt'] >= 30.0 ) else 0



df_a['target_beer_lt'] = df_a.apply (lambda row: isOverTarget(row), axis=1)

df_a.head()
grouped = df_a.groupby([ 'is_weekend']).sum()

grouped
#table=pd.crosstab(df_a.beer_lt, df_a.is_weekend)

#table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)

#plt.figure(figsize=(15,8))

#plt.title('')

#plt.xlabel('Liters')

#plt.ylabel('Is Weekend')

#plt.show()
boolean_col = 'target_beer_lt'

cols = ['temp_max_c', 'is_weekend']

#cols = ['temp_min_c', 'temp_max_c', 'prec', 'is_weekend']



x = df_a[cols].values.astype('float')

y = df_a[boolean_col].values
#X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)
#test_df = df_a.sample(frac=0.3)

#test_df = test_df.reset_index()



#train_df = df_a.drop(test_df.index)

#train_df = train_df.reset_index()
numSplitElements = int( len(df_a) * 0.7 )



xTrain = x[: numSplitElements ]

xTest = x[numSplitElements:]



yTrain = y[:numSplitElements]

yTest = y[numSplitElements:]
st.chisqprob = lambda chisq, df: st.chi2.sf(chisq, df)



model = sm.Logit( yTrain, xTrain )

#model = sm.Logit( train_df[boolean_col], train_df[cols].astype(float) ) # @issue_1



result = model.fit()

#result = model.fit_regularized(alpha = 1, disp = False)



result.summary( xname=cols, yname=boolean_col,title='Logit Model', alpha=1)# @issue_3:solution
result.pvalues
# Predict with the same number of feature columns

result.predict( xTest[0] )
def calculate_accuracy( predictions, real):

    correct = 0

    for i in range(len(predictions)):

        if round(predictions[i]) == round(real[i]):

            correct += 1

    return correct * 1.0 / len(predictions)
# predict train

train_predictions = result.predict(xTrain)

train_accuracy = calculate_accuracy( train_predictions, yTrain )

print(">>> train accuracy: ", train_accuracy * 100 )
# predict test

test_predictions  = result.predict(xTest)

test_accuracy = calculate_accuracy( test_predictions, yTest )

print(">>> test accuracy: ", test_accuracy * 100 )
for i in range(len(test_predictions)):  # @issue_2:solution

    test_predictions[i] = round(test_predictions[i])



# testing accuracy

accuracy_score( yTest, test_predictions )
test_cm = confusion_matrix( yTest, test_predictions, labels = [1.0, 0.0]) # @issue_2

print(">>> test confusion matrix: \n", test_cm ) 
labels = ['0', '1']

cm = test_cm

fig = plt.figure()

ax = fig.add_subplot(111)

cax = ax.matshow(cm, cmap='viridis') # RdBu, jet, viridis, cubehelix OR plt.cm.get_cmap('Blues', 6)

plt.title('Confusion matrix')

fig.colorbar(cax)

ax.set_xticklabels([''] + labels)

ax.set_yticklabels([''] + labels)



# Cell Text

r=0

c=0

for listItem in cm:

    for cellItem in listItem: 

        ax.text(c,r, cellItem, va='center', ha='center', color='r')

        c+=1

    c=0

    r+=1

   

plt.xlabel('Actual label')

plt.ylabel('Predicted label')

plt.show()