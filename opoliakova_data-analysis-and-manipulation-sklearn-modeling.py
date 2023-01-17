import pandas as pd

import numpy as np



from scipy.stats import ttest_ind



import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")
data_df = [train_df, test_df]
train_df.head()
train_df.dtypes
for df in data_df:

    df.crew = df.crew.astype('category')

    df.experiment = df.experiment.astype('category')

    df.seat = df.seat.astype('category')

    

train_df.event = train_df.event.astype('category')
train_df.isnull().sum().sum(), test_df.isnull().sum().sum()
train_df[['event', 'crew']].groupby(['event', 'crew'], as_index=False).size()
train_df.experiment.unique().categories, test_df.experiment.unique().categories
train_df[['event', 'seat']].groupby(['event', 'seat'], as_index=False).size()
#Dropping categorical columns

for df in data_df:

    df.drop(columns=['crew'], inplace = True)

    df.drop(columns=['experiment'], inplace = True)

    df.drop(columns=['seat'], inplace = True)
for predictor in test_df.columns[1:]:

    g = sns.FacetGrid(train_df, col='event')

    g.map(plt.hist, predictor, bins=100)
#Dropping numerical columns

for df in data_df:

    df.drop(columns=['time'], inplace = True)

    df.drop(columns=['ecg'], inplace = True)

    df.drop(columns=['r'], inplace = True)

    df.drop(columns=['gsr'], inplace = True)
for predictor in test_df.columns[1:]:

    g = sns.FacetGrid(train_df, col='event')

    g.set(yscale="log")

    g.map(plt.hist, predictor, bins=100)
for predictor in test_df.columns[1:]:

    train_df.boxplot(column=predictor, by='event', showmeans=True, showfliers=False)
#Helper function

def get_subset(event, predictor):

    return train_df.loc[train_df.event == event, predictor]



#Helper lists

all_predictors = test_df.columns[1:]

events = ['A', 'B', 'C', 'D']
distinctive_A_predictors = []



for predictor in all_predictors:

    p_vals = [ttest_ind(get_subset('A', predictor), get_subset(e, predictor), equal_var = False)[1] for e in events[1:]]

    is_distinct = [p < 0.05 for p in p_vals]

    if (all(is_distinct)):

        distinctive_A_predictors.append(predictor)



distinctive_A_predictors
distinctive_predictors = []



for predictor in all_predictors:

    p_vals = [ttest_ind(get_subset(events[e1], predictor), get_subset(events[e2], predictor), equal_var = False)[1] for e1 in range(3) for e2 in range(e1+1,4)]

    is_distinct = [p < 0.05 for p in p_vals]

    if (all(is_distinct)):

        distinctive_predictors.append(predictor)



distinctive_predictors
corrmat = train_df.corr()

plt.subplots(figsize=(12,9))

sns.heatmap(corrmat, vmax=0.9, square=True)
abs_corrmat = train_df[distinctive_predictors].corr().abs()

corrlist = abs_corrmat.unstack().sort_values(ascending=False).iloc[len(distinctive_predictors)::2]

corrlist[corrlist > 0.7]
uncorrelated_predictors = distinctive_predictors.copy()

uncorrelated_predictors.remove('eeg_p4')

uncorrelated_predictors.remove('eeg_fp2')
selected_predictors = uncorrelated_predictors
from sklearn.model_selection import train_test_split



from sklearn.linear_model import LogisticRegression

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

from sklearn.ensemble import RandomForestClassifier



from sklearn.metrics import log_loss
X = train_df[selected_predictors]

Y = train_df.event

X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=.2, random_state=42)

X_train.shape, Y_train.shape, X_val.shape, Y_val.shape 
losses = pd.DataFrame(columns=['log_loss','model'])
qda = QuadraticDiscriminantAnalysis()

qda.fit(X_train, Y_train)

qda_prob = qda.predict_proba(X_val)

qda_loss = log_loss(Y_val, qda_prob, labels=qda.classes_)

losses.loc["QDA"] = [qda_loss, qda]



qda_loss
lda = LinearDiscriminantAnalysis()

lda.fit(X_train, Y_train)

lda_prob = lda.predict_proba(X_val)

lda_loss = log_loss(Y_val, lda_prob, labels=lda.classes_)

losses.loc["LDA"] = [lda_loss, lda]



lda_loss
for c_deg in range(-5,5):

    logreg = LogisticRegression(multi_class = 'multinomial', solver='saga', penalty='l1', C=3**c_deg, max_iter=200)

    logreg.fit(X_train, Y_train)

    logreg_prob = logreg.predict_proba(X_val)

    logreg_loss = log_loss(Y_val, logreg_prob, labels=logreg.classes_)

    losses.loc["Log Regression C=" + str(3**c_deg)] = [logreg_loss, logreg]

    

    print(3**c_deg, '-', logreg_loss)
for n_est in range(25,51,5):

    rforest = RandomForestClassifier(n_estimators = n_est , class_weight="balanced", n_jobs=2)

    rforest.fit(X_train, Y_train)

    rforest_prob = rforest.predict_proba(X_val)

    rforest_loss = log_loss(Y_val, rforest_prob, labels=rforest.classes_)

    losses.loc["Rand forest n_est=" + n_est] = [rforest_loss, rforest]

    

    print(n, '-', rforest_loss)
losses.head()
losses.log_loss.plot.line()
best_model = losses.model[losses.log_loss.argmin()]
test_res = best_model.predict_proba(test_df[selected_predictors])

res_df = pd.DataFrame({"A" : test_res[:,0], 

                       "B" : test_res[:,1], 

                       "C" : test_res[:,2], 

                       "D" : test_res[:,3]})

res_df.to_csv('submission.csv', index_label='id')