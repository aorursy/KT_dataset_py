# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import classification_report,confusion_matrix

from plotly import __version__

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import cufflinks as cf

init_notebook_mode(connected=True)

cf.go_offline()

from sklearn.tree import DecisionTreeClassifier

from IPython.display import Image  

from sklearn.externals.six import StringIO  

from sklearn.tree import export_graphviz

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_excel('/kaggle/input/covid19/dataset.xlsx')

df.head(5)
df_cleaned = pd.concat([df['SARS-Cov-2 exam result'], df['Patient age quantile'], df['Eosinophils'],

                       df['Leukocytes'], df['Lymphocytes']],

                      axis = 1)
df_cleaned.head(5)
df_cleaned['SARS-Cov-2 exam result'] = df_cleaned['SARS-Cov-2 exam result'].map(lambda r: 1 if r == 'positive' else 0 )
def clean_nan(col, datafr):

    i = 0

    while i < len(col):

        if pd.isnull(col[i]):

            datafr.drop(i, axis = 0, inplace = True)

        i+=1

clean_nan(df_cleaned['Eosinophils'], df_cleaned)
df_cleaned.head(10)
sns.set_style('darkgrid')
sns.countplot(x ='Patient age quantile',data=df_cleaned,hue='SARS-Cov-2 exam result' )
sns.countplot(x ='Patient age quantile',data=df_cleaned[df_cleaned['Patient age quantile'] == 0],hue='SARS-Cov-2 exam result' )
#Here we see that the 0 age quantile doesn't have any covid-positive case, so we'll drop it

null_covid = df_cleaned[df_cleaned['Patient age quantile'] == 0]

for i in null_covid.index:

    df_cleaned.drop(i, axis = 0, inplace = True)
sns.countplot(x ='Patient age quantile',data=df_cleaned,hue='SARS-Cov-2 exam result' )
##Seting a threshold for accuracy



n_covid = len(df_cleaned[df_cleaned['SARS-Cov-2 exam result'] == 0].index)

y_covid = len(df_cleaned[df_cleaned['SARS-Cov-2 exam result'] == 1].index)

print('Min Accuracy for 0: ', n_covid/(n_covid + y_covid))

print('Min Accuracy for 1: ', y_covid/(n_covid + y_covid)) 
sns.pairplot(df_cleaned, hue = 'SARS-Cov-2 exam result')
sns.heatmap(df_cleaned.drop('SARS-Cov-2 exam result', axis = 1).corr(),annot=True)



#We'll only see the "intra-feature" correlation in this graph
#It seems that the features(above) and the features and the target(below) are weakly pair-correlated

sns.heatmap(df_cleaned.corr(),annot=True)
#Splitting the train/test sets

X_train, X_test, y_train, y_test = train_test_split(df_cleaned.drop(['SARS-Cov-2 exam result','Lymphocytes'], axis = 1),

                                                    df_cleaned['SARS-Cov-2 exam result'],

                                                    test_size=0.30, random_state = 42)
#Choosing a good k value

error_rate = []

for i in range(1,30):

    knn_model = KNeighborsClassifier(n_neighbors = i)

    knn_model.fit(X_train, y_train)

    predictions = knn_model.predict(X_test)

    error_rate.append(np.mean(predictions != y_test))

newdf = pd.DataFrame(error_rate, columns = ['Error Rate'])

newdf['IDX'] = range(1,30)
newdf.plot.line(x = 'IDX', y = 'Error Rate', figsize = (12,6))

#From the plot above, we can see that K = 5 is the minimum K that minimizes the error rate, dropping it to 0.11
#Final KNN model



knn_model = KNeighborsClassifier(n_neighbors = 5)

knn_model.fit(X_train, y_train)
predictions = knn_model.predict(X_test)

print('WITH K = 5')

print('\n')

print(confusion_matrix(predictions, y_test))

print('\n')

print(classification_report(predictions,y_test))



# By  the data below, we can see that this isn't a good model from the 'minimize the False-Negatives' perspective.

# Out of 173 patients, we predicted 4 False-Negative and 15 False-Positives
df_cleaned[df_cleaned['SARS-Cov-2 exam result']==1]['Eosinophils'].sort_values()
df_cleaned[df_cleaned['SARS-Cov-2 exam result']==1]['Leukocytes'].sort_values()
#The number of patients with a 'Leukocytes' level greater than 1.0 is 66. We could classify all this patients as 

#covid-negative, with a 1,5% porcentage of false-negatives

len(df_cleaned[df_cleaned['Leukocytes'] > 1.0])
#The number of patients with a 'Leukocytes' level greater than 1.0 is 71. We could classify all this patients as 

#covid-negative, with a 1,4% porcentage of false-negatives

len(df_cleaned[df_cleaned['Eosinophils'] > 1.0])
#Finally, let's count how many patients had both levels greater than 1.0

(df_cleaned[df_cleaned['Leukocytes'] > 1.0]['Eosinophils'] > 1.0).sort_values()