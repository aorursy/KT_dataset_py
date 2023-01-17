#General Imports

import numpy as np # linear algebra

import pandas as pd # data processing

data = pd.read_csv('../input/creditcard.csv')

data.describe()
data.columns
data.shape
data.isnull().any().sum()
LEGAL, FRAUD = range(2)



# sampler name constants

IMBALANCE, UNDER_RANDOM, OVER_RANDOM, OVER_SMOTE =  (

    'Imbalance', 'Random Under Sampler', 

    'Random Over Sampler','Smote'

)

n = data.Class.count()



frauds = data.Class == FRAUD

legals = data.Class == LEGAL

n_frauds = frauds.sum()

n_legals = legals.sum()



print('Total transactions:', n)

print('Legal transactions: {1} ({0:.4f}%).'

      ''.format(n_legals/n*100, n_legals))

print('Fraudulent transactions: {1} ({0:.4f}%).'

      ''.format(n_frauds/n*100, n_frauds))
from collections import Counter



X = data.drop('Class', axis=1)

y = data.Class

c = Counter(y)

print('Original distribution '

      'Legal ({0}) - Fraud ({1}))'.format(c[LEGAL], c[FRAUD]))
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4)

n = y_test.count()

n_frauds = y_test.sum()

"Fraud transactions: {0} ({1:.2}%)".format(n_frauds, n_frauds/n * 100)

from time import time

from imblearn.over_sampling import RandomOverSampler, SMOTE

from imblearn.under_sampling import RandomUnderSampler



samplers = {

    UNDER_RANDOM: RandomUnderSampler,

    OVER_RANDOM: RandomOverSampler,

    OVER_SMOTE: SMOTE,

}



samples = {IMBALANCE: (X_train, y_train)}

durations = {IMBALANCE: 0}



for name, sampler in samplers.items():

    start2 = time()

    smp = sampler(random_state=0)

    samples[name] = X_sample, y_sample = smp.fit_resample(X_train, y_train)

    durations[name] = time() - start2



    print('{0} tooks {1:.2} seconds'.format(name, durations[name]))

    print('Distribution is', Counter(y_sample))



    

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import balanced_accuracy_score

models = dict()



common_args = {'n_estimators': 100, 

               'max_depth': 3,

               'random_state': 0, 'oob_score': True}

args = {

    IMBALANCE: {

        'class_weight': 'balanced_subsample' 

         }

}

models = dict()

samples[IMBALANCE+' No Handling'] = samples[IMBALANCE]

    

for name, sample in samples.items():

    specific_args = args.get(name, {})

    start = time()

    rfc = RandomForestClassifier(**common_args, **specific_args)

    rfc.fit(*sample)

    

    total_time = time() - start

    

    X_sample, y_sample = sample

    y_pred = rfc.predict(X_test)

    

    score = balanced_accuracy_score(y_test, y_pred)

    # store models and results

    models[name] = rfc, y_pred, score, total_time

    print('{0}\nSample Size: {3:.0f}\n'

          'Trained time: {1:.2f} seconds\n'

          'Balance Score: {2:.2f}%\n'

          'Oob Score: {4:.2f}%\n'.format(

            name, total_time, score*100, 

            len(y_sample), rfc.oob_score_*100))

MODEL, PREDICTS, SCORE, DURATION = range(4)

print('Out of Bag Scores')

for name in models.keys():

    score = models[name][MODEL].oob_score_

    

    print(' - {0} is {1:.2f}%'.format(name, score*100))





from sklearn.metrics import confusion_matrix, classification_report

"""  

           LEGAL                    FRAUD

   PASS     lp(True Negative)       fp(False Negative)       

   BLOCK    lb(False Positive)      fb(True Positive)

   

    recall = fb / (fb+fp)        true pos / (true pos + false neg)

    precision = fb / (fb+lb)     true pos / (true pos + false pos)

    

    """

target_names = ['Legals', 'Frauds']

for name, model in models.items():

    # tn, fp, fn, tp

    # legal_passed, legal_blocked, fraud_passed, fraud_blocked 

    y_pred = model[PREDICTS]

    lp, lb, fp, fb = confusion_matrix(y_test, 

                                      y_pred).ravel()

    

    print('{0}:\nLegal Passed: {1}\n'

    'Legal Blocked: {2}\n'

    'Fraud Blocked: {4}\n'

    'Fraud Passed: {3}'

    ''.format(name, lp, lb, fp, fb))

    

    print('Detect {0:.2f}% Frauds and Block {1:.2f}% of legals\n'

          ''.format(fb/(fp+fb)*100, lb/(lb+lp)*100))

    

    print(classification_report(y_test, y_pred, target_names=target_names))
