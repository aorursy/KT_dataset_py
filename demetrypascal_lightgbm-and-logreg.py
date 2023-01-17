import numpy as np

import pandas as pd

import json





from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import GridSearchCV

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline



from sklearn.linear_model import LogisticRegression





import lightgbm as lgb



import matplotlib.pyplot as plt





from pylab import rcParams



rcParams['figure.figsize'] = 18, 8







cv_count = 20



categories_count = 15











files = []



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        files.append(os.path.join(dirname, filename))



files = sorted(files)



files
errors = pd.read_csv(files[13])



errors.head(15)
plt.scatter(errors['positives'], errors['score'])



plt.xlabel('Count of positive samples in each category')

plt.ylabel('Score (lower is better)')



plt.title('Logistic RFE regression errors by category positive size')
categs = errors['category'].values[:categories_count]



categs
with open(files[14], 'r') as file:

    main_data = json.load(file)



print(main_data.keys())



print(main_data['start_predictors'][:10])



print()



for i, (key, val) in enumerate(main_data['better_PCAs'].items()):

    if i < 4:

        print(f'{key}: {val}')
train = pd.read_csv(files[2])



test = pd.read_csv(files[1])



submission = pd.read_csv(files[0])



targets = pd.read_csv(files[4])



train.head()
targets = targets[train['cp_type']!= 'ctl_vehicle']



train = train[train['cp_type']!= 'ctl_vehicle']



train.shape
new_train = train.loc[:, main_data['start_predictors']]

new_test = test.loc[:, main_data['start_predictors']]



new_train.head()
new_train.shape
pipe = Pipeline([

    ('scaler', StandardScaler()), 

    ('pca', PCA(n_components = 0.8, svd_solver = 'full'))])



new_train = pipe.fit_transform(new_train)



new_test = pipe.transform(new_test)





cols = [f'PC{i+1}' for i in range(56)]



new_train = pd.DataFrame(new_train, columns = cols)

new_test = pd.DataFrame(new_test, columns = cols)



new_train.head()
pipe['pca'].explained_variance_ratio_.sum()
predictors = main_data['better_PCAs']
for categ, preds in predictors.items():

    

    if type(preds) != type([]):

        preds = [preds]

    

    #print(categ)

    

    model = LogisticRegression(penalty = 'none')

    

    model.fit(new_train.loc[:, preds].values.reshape(-1, len(preds)), targets[categ].values)# targets[categ].values.reshape(-1, 1))

    

    submission[categ] = 1 - model.predict_proba(new_test.loc[:, preds])
submission.head()
ctl_inds = test['cp_type'] == 'ctl_vehicle' # for future submission



drops = ['sig_id', 'cp_type', 'cp_time', 'cp_dose']



train = train.drop(drops, 1)



test = test.drop(drops, 1)



train.head()


iterations = 100



params = {"max_depth": [20],

              "learning_rate" : [0.002, 0.005, 0.01, 0.02, 0.03],

              "n_estimators": [iterations]

             }



results = np.empty((len(categs), len(params['learning_rate'])))





cb_model = GridSearchCV(lgb.LGBMClassifier(silent=True, loss_function = 'Logloss'), params,  cv = cv_count, scoring = 'neg_log_loss')



for i, categ in enumerate(categs):

    

    cb_model.fit(train, targets[categ], verbose = False)

    

    time_for_fitting = cb_model.cv_results_['mean_fit_time'].sum() * cv_count/60



    print(f'{categ} fitted about {time_for_fitting} mins')

    

    

    ans = -cb_model.cv_results_['mean_test_score']

    

    results[i, :] = ans



    plt.plot(params['learning_rate'], ans, color = 'red', linewidth = 2, zorder = 1)

    plt.scatter(params['learning_rate'], ans, color = 'blue', linewidth = 4, zorder = 2)

    plt.xlabel('learning rate')

    plt.ylabel('average logloss')

    plt.title(f'Logloss by learning rate ({categ})')



    plt.show()
am = np.argmin(results, axis = 1)



best_rates = [params['learning_rate'][i] for i in am]



best_rates
for rate, categ in zip(best_rates, categs):

    

    cb_model = lgb.LGBMClassifier(loss_function = 'Logloss', n_estimators = iterations, learning_rate = rate, max_depth = 20)

    

    cb_model.fit(train, targets[categ], verbose = False)

    

    #print(cb_model.predict_proba(test))

    

    submission[categ] = pd.Series(cb_model.predict_proba(test)[:,1])

    
submission.loc[ctl_inds, submission.columns[1:]] = 0



submission.head(10)
submission[submission.loc[:,submission.columns[1:]]>1].fillna(1).mean().mean()
targets = pd.read_csv(files[4])

test = pd.read_csv(files[1])
np.sum(targets.columns != submission.columns)
np.sum(test['sig_id'].values != submission['sig_id'].values)
submission.to_csv('submission.csv', index = False)