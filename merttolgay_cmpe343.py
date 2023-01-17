import datetime, warnings, scipy 
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
from sklearn import metrics, linear_model
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
import random
flights = pd.read_csv('../input/flights.csv', low_memory=False)
airports = pd.read_csv('../input/airports.csv')
airlines = pd.read_csv('../input/airlines.csv')
flights.head()
airports.head()
airlines
# Filter data to keys of interest
keys = ['MONTH', 'DAY', 'DAY_OF_WEEK', 'YEAR','AIRLINE', 'ORIGIN_AIRPORT',
       'DESTINATION_AIRPORT', 'SCHEDULED_DEPARTURE', 'DEPARTURE_TIME',
       'DEPARTURE_DELAY', 'SCHEDULED_TIME',
       'ELAPSED_TIME', 'AIR_TIME', 'DISTANCE', 
       'SCHEDULED_ARRIVAL', 'ARRIVAL_TIME', 'ARRIVAL_DELAY']
flights_df = flights[keys]
flights_df.head(10)
flights_df.info()
flights_df = flights_df.dropna()
flights_df.head()
# - Filter [ORIGIN_AIRPORT] by LAX (Los Angeles International Airport)
df = flights_df[flights_df['ORIGIN_AIRPORT'] == 'LAX']
df[:10]
airline_comps = airlines.set_index('IATA_CODE')['AIRLINE'].to_dict()
airlines
# - Calculate how many flights happened from LAX by [AIRLINE] 's

#__________________________________________________________________
# function that extract statistical parameters from a grouby objet:
def get_stats(group):
    return {'count': group.count()}
#_______________________________________________________________
# Creation of a dataframe with statitical infos on each airline:
global_stats = df['DEPARTURE_DELAY'].groupby(df['AIRLINE']).apply(get_stats).unstack()
global_stats = global_stats.sort_values('count')
global_stats
# - Select the [AIRLINE] that performs the most flights
# - Filter [AIRLINE] by selected airline that performs the most flights
df2 = df[df['AIRLINE'] == 'WN']
df2.head()
# - Check for missing values and 
#      if missing values are too much fill them with mean of columns
#      else you can drop those missing values rows
print(df2.isnull().sum())
df2.dropna(inplace=True)
df2.shape

df2.info()
delay_cutoff = 15

df2 = df2.reset_index(drop=True)
mapping = df2['DEPARTURE_DELAY'].values > delay_cutoff
df2['Delayed'] = np.where(df2['DEPARTURE_DELAY'] > 0, 1, 0)
df2.head()
df2.drop(['YEAR', 'SCHEDULED_TIME', 'ELAPSED_TIME'], axis = 1, inplace = True)
df2['Delayed'].value_counts()
correlations = df2.corr()
%matplotlib inline
import seaborn as sns
ax = sns.heatmap(correlations, xticklabels=correlations.columns.values,
            yticklabels=correlations.columns.values)
correlations
corr = df2.corr()#Lists all pairs of highly collinear variables
indices = np.where(corr > 0.8)
indices = [(corr.columns[x], corr.columns[y]) for x, y in zip(*indices)
                                        if x != y and x < y]
indices
#Converting categorical variables to numeric for correlation analysis and further use in prediction
df2.DESTINATION_AIRPORT = df2.DESTINATION_AIRPORT.astype("category")
df2.DESTINATION_AIRPORT = df2.DESTINATION_AIRPORT.cat.codes
df2.DAY = df2.DAY.astype("category")
df2.DAY = df2.DAY.cat.codes

df2.DAY_OF_WEEK = df2.DAY_OF_WEEK.astype("category")
df2.DAY_OF_WEEK = df2.DAY_OF_WEEK.cat.codes

df2.MONTH = df2.MONTH.astype("category")
df2.MONTH = df2.MONTH.cat.codes
keys = ['MONTH', 'DAY', 'DAY_OF_WEEK',
       'DESTINATION_AIRPORT', 'DEPARTURE_TIME',
       'DEPARTURE_DELAY', 'DISTANCE', 
        'ARRIVAL_TIME', 'Delayed']
feature_df = df2[keys]
feature_df.head()
from sklearn.metrics import precision_recall_fscore_support
from sklearn.utils.multiclass import unique_labels


def classification_report_to_dataframe(ground_truth, predictions):
    """
    Saves the classification report to dataframe using the pandas module.
    :param ground_truth: list: the true labels
    :param predictions: list: the predicted labels
    :return: dataframe
    """
    import pandas as pd

    # get unique labels / classes
    # - assuming all labels are in the sample at least once
    labels = unique_labels(ground_truth, predictions)

    # get results
    precision, recall, f_score, support = precision_recall_fscore_support(ground_truth,
                                                                          predictions,
                                                                          labels=labels,
                                                                          average=None)
    # a pandas way:
    results_pd = pd.DataFrame({"class": labels,
                               "precision": precision,
                               "recall": recall,
                               "fscore": f_score
                               })

    return results_pd
from time import time
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from collections import defaultdict
from sklearn.metrics import confusion_matrix

classifiers = ['NB', 'SVM', 'RF']


def machine_learning_algorithm(X_train_transformed, 
                               X_test_transformed, y_train, y_test):
    
    results = {}
    for classifier in classifiers:
        if classifier == 'SVM':
            print("-------------------------------------------------")
            print("Implementing Support Vector Machine")
            print("-------------------------------------------------")
            clf = SVC()

            t0 = time()
            clf.fit(X_train_transformed, y_train)
            print ('Training time:', round(time()-t0, 3), 's')

            t0 = time()
            pred = clf.predict(X_test_transformed)
            print ('Predicting time:', round(time()-t0, 3), 's')
            acc_svm = accuracy_score(y_test, pred)
            print ('Average Accuracy:',accuracy_score(y_test, pred)*100,'%')
            print ('Classification Report:') 
            results_svm = classification_report_to_dataframe(y_test, pred)
            results_svm.index = results_svm['class']
            results_svm.drop(results_svm.columns[[0]], axis = 1, inplace = True)
            print (results_svm)
            results['SVM'] = results_svm

            
        elif classifier == 'NB':
            print("-------------------------------------------------")
            print("Implementing Naive Bayes")
            print("-------------------------------------------------")
            clf = GaussianNB()

            t0 = time()
            clf.fit(X_train_transformed, y_train)
            print ('Training time:', round(time()-t0, 3), 's')

            t0 = time()
            pred = clf.predict(X_test_transformed)
            print ('Predicting time:', round(time()-t0, 3), 's')
            acc_nb = accuracy_score(y_test, pred)
            print ('Average Accuracy:',accuracy_score(y_test, pred)*100,'%')
            print ('Classification Report:') 
            results_nb = classification_report_to_dataframe(y_test, pred)
            results_nb.index = results_nb['class']
            results_nb.drop(results_nb.columns[[0]], axis = 1, inplace = True)
            print (results_nb)
            results['NB'] = results_nb
            
        
        elif classifier == 'RF':
            print("-------------------------------------------------")
            print("Implementing Random Forest")
            print("-------------------------------------------------")
            clf = RandomForestClassifier(n_estimators = 500)

            t0 = time()
            clf.fit(X_train_transformed, y_train)
            print ('Training time:', round(time()-t0, 3), 's')

            t0 = time()
            pred = clf.predict(X_test_transformed)
            print ('Predicting time:', round(time()-t0, 3), 's')
            acc_rf = accuracy_score(y_test, pred)
            print ('Average Accuracy:',accuracy_score(y_test, pred)*100,'%')
            print ('Classification Report:') 
            results_rf = classification_report_to_dataframe(y_test, pred)
            results_rf.index = results_rf['class']
            results_rf.drop(results_rf.columns[[0]], axis = 1, inplace = True)
            print (results_rf)
            results['RF'] = results_rf

            print()
            print ('Feature Importance:')
            
            importances = clf.feature_importances_
            indices = np.argsort(importances)[::-1]

            # Print the feature ranking
            feature_importances = pd.DataFrame(clf.feature_importances_,
                                               index = X_train_transformed.columns,
                                                columns=['importance']).sort_values('importance',
                                                                                    ascending=False)
            print (feature_importances.head())
            
    return (results)
def data_preperation_for_machine_learning(data):
    X = data.iloc[:,data.columns != 'Delayed']
    y = data['Delayed']

    # test_size is the percentage of events assigned to the test set
    # (remainder go into training) 
    # 60% Training and 40% Test dataset 
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                                               test_size=0.4, random_state=42)
    

    
    return (X_train, X_test, y_train, y_test)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

(X_train_transformed, 
     X_test_transformed, 
     y_train, y_test) = data_preperation_for_machine_learning(feature_df)
metrics = machine_learning_algorithm(X_train_transformed,X_test_transformed,y_train,y_test)
