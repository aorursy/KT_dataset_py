import matplotlib.pyplot as plt

import numpy as np 

import pandas as pd

import os

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.feature_selection import RFECV

plt.style.use('seaborn-whitegrid')

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df=pd.read_csv('/kaggle/input/predicting-a-pulsar-star/pulsar_stars.csv')

print('Done!')
def data_desc(df):

    print('number of collected data: ',df.shape[0])

    print('number of variables: ',df.shape[1])

    print('-'*10)

    print('dataset variables names: ')

    print(df.columns)

    print('-'*10)

    print('variables data-type: ')

    print(df.dtypes)

    print('-'*10)

    c=df.isnull().sum()

    print('missing data')

    print(c[c>0])

data_desc(df)
df[df['target_class']==1][' Mean of the integrated profile'].plot.kde(label='positive',color='blue')

df[df['target_class']==0][' Mean of the integrated profile'].plot.kde(label='negative',color='red')

plt.title('Comparing the Means of the integrated profile for positive and negative cases')

plt.ylabel('frequency')

plt.xlabel('Mean of the integrated profile')

plt.legend()

plt.show()
df[df['target_class']==1][' Standard deviation of the integrated profile'].plot.kde(label='positive',color='blue')

df[df['target_class']==0][' Standard deviation of the integrated profile'].plot.kde(label='negative',color='red')

plt.title('Comparing the  Standard deviations of the integrated profile for positive and negative cases')

plt.ylabel('freuency')

plt.xlabel('Standard deviation of the integrated profile')

plt.legend()

plt.show()
df[df['target_class']==1][' Excess kurtosis of the integrated profile'].plot.kde(label='positive',color='blue')

df[df['target_class']==0][' Excess kurtosis of the integrated profile'].plot.kde(label='negative',color='red')

plt.title('Comparing the  Excess kurtosis of the integrated profile for positive and negative cases')

plt.ylabel('frequency')

plt.xlabel(' Excess kurtosis of the integrated profile')

plt.legend()

plt.show()
df[df['target_class']==1][' Skewness of the integrated profile'].plot.kde(label='positive',color='blue')

df[df['target_class']==0][' Skewness of the integrated profile'].plot.kde(label='negative',color='red')

plt.title('Comparing the  Skewness of the integrated profile for positive and negative cases')

plt.ylabel('frequency')

plt.xlabel(' Skewness of the integrated profile')

plt.legend()

plt.show()
df[df['target_class']==1][' Mean of the DM-SNR curve'].plot.kde(label='positive',color='blue')

df[df['target_class']==0][' Mean of the DM-SNR curve'].plot.kde(label='negative',color='red')

plt.title('Comparing the  Mean of the DM-SNR curve for positive and negative cases')

plt.ylabel('frequency')

plt.xlabel(' Mean of the DM-SNR curve')

plt.legend()

plt.show()
df[df['target_class']==1][' Standard deviation of the DM-SNR curve'].plot.kde(label='positive',color='blue')

df[df['target_class']==0][' Standard deviation of the DM-SNR curve'].plot.kde(label='negative',color='red')

plt.title('Comparing the Standard deviations of the DM-SNR curve for positive and negative cases')

plt.ylabel('frequency')

plt.xlabel(' Standard deviation of the DM-SNR curve')

plt.legend()

plt.show()
df[df['target_class']==1][' Excess kurtosis of the DM-SNR curve'].plot.kde(label='positive',color='blue')

df[df['target_class']==0][' Excess kurtosis of the DM-SNR curve'].plot.kde(label='negative',color='red')

plt.title('Comparing the Excess kurtosis of the DM-SNR curve for positive and negative cases')

plt.ylabel('frequency')

plt.xlabel(' Excess kurtosis of the DM-SNR curve')

plt.legend()

plt.show()
df[df['target_class']==1][' Skewness of the DM-SNR curve'].plot.kde(label='positive',color='blue')

df[df['target_class']==0][' Skewness of the DM-SNR curve'].plot.kde(label='negative',color='red')

plt.title('Comparing the Skewness of the DM-SNR curve for positive and negative cases')

plt.ylabel('frequency')

plt.xlabel(' Skewness of the DM-SNR curve')

plt.legend()

plt.show()
def select_features(df):

    all_x=df.drop('target_class',axis=1)

    all_y=df['target_class']

    rf=RandomForestClassifier(random_state=1)

    selector=RFECV(rf,cv=10)

    selector.fit(all_x,all_y)

    best_columns = list(all_x.columns[selector.support_])

    print("Best Columns \n"+"-"*12+"\n{}\n".format(best_columns))

    

    return best_columns

cols=select_features(df)
def select_model(df,features):

    all_x=df[features]

    all_y=df['target_class']

    models=[

        {

            'name':'Logistic Regression',

            'estimator':LogisticRegression(),

            'hyberparameters':{

                'solver':["newton-cg", "lbfgs", "liblinear"]

            }

        },

        {

            'name':'K-Neighbors Classifier',

            'estimator':KNeighborsClassifier(),

            'hyberparameters':{

               "n_neighbors": range(1,20,2),

               "weights": ["distance", "uniform"],

               "algorithm": ["ball_tree", "kd_tree", "brute"],

               "p": [1,2]

            }

        },

        {

            'name':'Random Forest Classifier',

            'estimator':RandomForestClassifier(),

            'hyberparameters':{

                "n_estimators": [4, 6, 9],

                "criterion": ["entropy", "gini"],

                "max_depth": [2, 5, 10],

                "max_features": ["log2", "sqrt"],

                "min_samples_leaf": [1, 5, 8],

                "min_samples_split": [2, 3, 5]

            }

        },

        {

            'name':'Decision Tree',

            'estimator':DecisionTreeClassifier(),

            'hyberparameters':{

                'max_depth':np.arange(1, 21),

                'min_samples_leaf':[1, 5, 10, 20, 50, 100]

            }

        }

    ]

    for i in models:

        print(i['name'])

        gs=GridSearchCV(i['estimator'],

                        param_grid=i['hyberparameters'],

                        scoring='accuracy',

                        cv=10)

        gs.fit(all_x,all_y)

        i['best_params']=gs.best_params_

        i['best_score']=gs.best_score_

        i['best_estimator']=gs.best_estimator_

        print("Best Score: {}".format(i["best_score"]))

        print("Best Parameters: {}\n".format(i["best_params"]))

select_model(df,cols)