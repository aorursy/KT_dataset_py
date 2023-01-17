import numpy as np

import pandas as pd





from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import GridSearchCV

import catboost as cb



import matplotlib.pyplot as plt







from pylab import rcParams



rcParams['figure.figsize'] = 18, 8









files = []



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        files.append(os.path.join(dirname, filename))



files = sorted(files)



files
cv_count = 10
train = pd.read_csv(files[2])



test = pd.read_csv(files[1])



submission = pd.read_csv(files[0])



targets = pd.read_csv(files[4])



train.head()
train_df = train.drop(['sig_id'], axis = 1)
colnames = list(targets.columns)



colnames[:10]
cat_features_index = [0,1,2]





params = {'learning_rate' : [0.04, 0.08, 0.1, 0.12, 0.13, 0.15],

         'iterations': [50]}





results = np.empty((targets.shape[1] - 1, len(params['learning_rate'])))







cb_model = GridSearchCV(cb.CatBoostClassifier(loss_function = 'Logloss'), params,  cv = cv_count, scoring = 'neg_log_loss')



for ind in range(1, 7):

    

    cb_model.fit(train_df, targets.iloc[:,ind],  cat_features = cat_features_index, verbose = False)

    

    time_for_fitting = cb_model.cv_results_['mean_fit_time'].sum() * cv_count/60



    print(f'{colnames[ind]} fitted about {time_for_fitting} mins')

    

    

    ans = -cb_model.cv_results_['mean_test_score']

    

    results[ind-1, :] = ans



    plt.plot(params['learning_rate'], ans, color = 'red', linewidth = 2, zorder = 1)

    plt.scatter(params['learning_rate'], ans, color = 'blue', linewidth = 4, zorder = 2)

    plt.xlabel('learning rate')

    plt.ylabel('average logloss')

    plt.title(f'Logloss by learning rate ({colnames[ind]})')



    plt.show()

results.shape
am = np.argmin(results, axis = 1)



best_rates = [params['learning_rate'][i] for i in am]



best_rates
np.save('results.npy', results)