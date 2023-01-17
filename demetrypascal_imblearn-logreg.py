import numpy as np

import pandas as pd



from collections import Counter



from sklearn.model_selection import StratifiedKFold, cross_val_score

from sklearn.preprocessing import StandardScaler



from sklearn.pipeline import Pipeline



from imblearn.over_sampling import SMOTE





from sklearn.linear_model import LogisticRegression

from sklearn.metrics import log_loss





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

train = pd.read_csv(files[2])



test = pd.read_csv(files[1])



submission = pd.read_csv(files[0])



targets = pd.read_csv(files[4])



train.head()
valid_inds = np.arange(train.shape[0])[train['cp_type'] != 'ctl_vehicle']



print(len(valid_inds))

                                   

X = train.iloc[valid_inds, 4:]



y = targets.iloc[valid_inds, 1:]



y.head()
def show_cv_mean(X, y):



    results = []



    for i, categ in enumerate(y.columns):



        pipe = Pipeline([('scale', StandardScaler()), ('LogReg', LogisticRegression())])



        positives = np.sum(y[categ])



        if positives > 10:



            cv = StratifiedKFold(3)# if np.sum(y[categ]) > 10 else 10



            scores = cross_val_score(pipe, X, y[categ], cv = cv, scoring = 'neg_log_loss')



            results.append(-scores.mean())



        #else:

        #    pipe.fit(X, y[categ])

        #    ans = log_loss(pipe.predict_proba(X), y[categ])

        #    results.append(ans)

        #    print(ans)



    print(np.array(results).mean())
show_cv_mean(X, y)
uniqs = []

dic = {}

answers = []



for i in range(y.shape[0]):

    

    flag = False

    

    arr0 = y.iloc[i,:] 

    

    for j, arr in enumerate(uniqs):

        

        if np.sum(arr0 != arr) == 0:

            answers.append(j)

            dic[j] += 1

            flag = True

            break

    

    if not flag:

        uniqs.append(arr0)

        answers.append(len(uniqs)-1)

        dic[len(uniqs)-1] = 1

        

len(uniqs)
counts = np.array(sorted(dic.values()))



counts
answers = np.array(answers)
not_corrected = [key for key, value in dic.items() if value == 1]



good_answers_index = np.array([val not in not_corrected for val in answers])
X1 = X.iloc[good_answers_index,:]

y1 = answers[good_answers_index]



X1.shape
X2 = X.iloc[~good_answers_index,:]

y2 = answers[~good_answers_index]



X2.shape
multiply_count = 50

max_total = 1000



def value_correct(value):

    if value > 7000:

        return value

    return min(value*multiply_count, max_total)



sampling_strategy = {key: value_correct(value)  for key, value in dic.items() if value > 1}



sampling_strategy
over = SMOTE(sampling_strategy = sampling_strategy, k_neighbors = 5)

#under = RandomUnderSampler(sampling_strategy = sampling_strategy)

#steps = [('o', over), ('u', under)]

#pipeline = Pipeline(steps=steps)
X1, y1 = over.fit_resample(X1, y1)
X1.shape
counter = Counter(y1)

print(counter)
X = pd.concat([X1, X2])

y = np.concatenate((y1, y2))



X.shape
y.shape
Y = pd.DataFrame([uniqs[i] for i in y], columns = list(targets.columns)[1:])



Y.shape
sums = Y.sum(axis = 0)

print(sums.mean(), sums.median(), sums.max())
X.to_csv('big_train.csv', index = False)

Y.to_csv('big_target.csv', index = False)
#show_cv_mean(X, Y)