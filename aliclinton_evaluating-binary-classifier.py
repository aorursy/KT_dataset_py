import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import sklearn

from sklearn.preprocessing import *

from sklearn.metrics import *

from sklearn.ensemble import *

from sklearn.linear_model import *

from sklearn.svm import * 

from sklearn.model_selection import *

from imblearn.over_sampling import *

from sklearn.neighbors import *

from sklearn.neural_network import *

from scipy.stats import t

from collections import defaultdict

import copy

import csv
filename='german_credit_data.csv'

df = pd.read_csv(filename,sep=',')
df.head()
# pension age based on gender

pension = (df['Age'] >= 65) & (df['Sex'] == ('female')) | (df['Age'] >= 65) & (df['Sex'] == ('male'))



# discriminate based on quantiles of credit amount

df['Credit amount'].quantile([.25,.75])



credit = []

for i in df['Credit amount']:

    if i < df['Credit amount'].quantile(.25):

        credit.append('low')

    elif i < df['Credit amount'].quantile(.75):

        credit.append('med')

    else:

        credit.append('high')

# discriminate based on duration of loans in month if less than 12 then it's a short term loan, long term otherwise.



duration = []

for i in df['Duration']:

    if i < 12:

        duration.append('short')

    else:

        duration.append('long')



# clean whitespace on column names to not fuck shit up

df.columns = df.columns.str.replace(' ', '')

# create new dataframe

df_clean = pd.DataFrame(columns=['sex','housing','job','purpose','sav','chq','amt','duration','pension'])

df_clean = df_clean.fillna(0)



# factorize all categorical variables, take the log of credit amount because it's better to work in the log scale for money

df_clean['sex'] = pd.factorize(df.Sex)[0]

df_clean['housing'] = pd.factorize(df.Housing)[0]

df_clean['job'] = pd.factorize(df.Job)[0]

df_clean['purpose'] = pd.factorize(df.Purpose)[0]

df_clean['sav'] = pd.factorize(df.Savingaccounts)[0] + 1

df_clean['chq'] = pd.factorize(df.Checkingaccount)[0] + 1

df_clean['pension'] = pd.factorize(pension)[0]

df_clean['amt'] = pd.factorize(credit)[0]

df_clean['duration'] = pd.factorize(duration)[0]



# good risk = 0, bad risk = 1

label = pd.factorize(df["Risk"].values)[0]

df_clean.head()
# in order to replicate the experiment

seed = 1992

# split data for training and holdout test set

data_train, data_test, label_train, label_test = train_test_split(df_clean, label, test_size = 0.20, random_state=seed)



# SMOTE to oversample and undersample

sm = SMOTE(random_state=seed)

data_train, label_train = sm.fit_resample(data_train, label_train)



# using stratified k fold in order to get equal proportion of data

skf = StratifiedKFold(n_splits=4, random_state=seed, shuffle=True)
# ml models

l = LogisticRegression(C=0.8, random_state = seed, solver='liblinear')

r = RandomForestClassifier(n_estimators=6,max_depth=31, random_state= seed)

s = SVC(kernel='poly', degree=3, gamma=2, probability=True, random_state = seed)

k = KNeighborsClassifier(n_neighbors=7)

a = MLPClassifier(solver='sgd', activation='relu', alpha=1e-5,hidden_layer_sizes=(9,), random_state=seed)



# One hot encoding (dummy variables)

enc = OneHotEncoder(handle_unknown='ignore')

enc.fit(data_train)

X_train_one_hot = enc.transform(data_train)

X_test_one_hot = enc.transform(data_test)



X_train = []

X_test = []

y_train = []

y_test = []



# Stratified K-fold

for train_index, test_index in skf.split(X_train_one_hot, label_train):

    train_data, test_data = X_train_one_hot[train_index], X_train_one_hot[test_index]

    X_train.append(train_data)

    X_test.append(test_data)

    

    train_label, test_label = label_train[train_index], label_train[test_index]

    y_train.append(train_label)

    y_test.append(test_label)    
# testing 0 1 loss

l_test_error = [];

s_test_error = [];

r_test_error = [];

k_test_error = [];

a_test_error = [];

l_pred = []

s_pred = []

r_pred = []

k_pred = []

a_pred = []





for i in range(4):

    l.fit(X_train[i],y_train[i])

    s.fit(X_train[i],y_train[i])

    r.fit(X_train[i],y_train[i])

    k.fit(X_train[i],y_train[i])   

    a.fit(X_train[i],y_train[i])  



    l_pred.append(l.predict(X_test[i]))

    l_test_error.append(balanced_accuracy_score(y_test[i],l_pred[i]))

    s_pred.append(s.predict(X_test[i])) 

    s_test_error.append(balanced_accuracy_score(y_test[i],s_pred[i]))

    

    r_pred.append(r.predict(X_test[i]))

    r_test_error.append(balanced_accuracy_score(y_test[i],r_pred[i]))    

    k_pred.append(k.predict(X_test[i]))

    k_test_error.append(balanced_accuracy_score(y_test[i],k_pred[i]))  

    a_pred.append(a.predict(X_test[i]))

    a_test_error.append(balanced_accuracy_score(y_test[i],a_pred[i]))



index = []

index.append(l_test_error.index(max(l_test_error)))

index.append(s_test_error.index(max(s_test_error)))

index.append(r_test_error.index(max(r_test_error)))

index.append(k_test_error.index(max(k_test_error)))

index.append(a_test_error.index(max(a_test_error)))
l.fit(X_train[index[0]],y_train[index[0]])

s.fit(X_train[index[1]],y_train[index[1]])

r.fit(X_train[index[2]],y_train[index[2]])

k.fit(X_train[index[3]],y_train[index[3]])   

a.fit(X_train[index[4]],y_train[index[4]])  



l_test_pred = l.predict(X_test_one_hot)

s_test_pred = s.predict(X_test_one_hot)

r_test_pred = r.predict(X_test_one_hot)

k_test_pred = k.predict(X_test_one_hot)

a_test_pred = a.predict(X_test_one_hot)



empirical_risk = []



empirical_risk.append(zero_one_loss(label_test,l_test_pred))

empirical_risk.append(zero_one_loss(label_test,s_test_pred))

empirical_risk.append(zero_one_loss(label_test,r_test_pred))

empirical_risk.append(zero_one_loss(label_test,k_test_pred))

empirical_risk.append(zero_one_loss(label_test,a_test_pred))





m = X_test_one_hot.shape[0]



print(empirical_risk)
delta = 0.05

hoeffding = np.sqrt(float(np.log(2/delta))/(float(2)*m))

lower = empirical_risk - hoeffding

upper = empirical_risk + hoeffding

hoeffding_ci = [lower,upper]





plt.errorbar(["h1","h2","h3","h4","h5"], empirical_risk, yerr=hoeffding, linestyle='',fmt="o")

plt.title("Hoeffding's 95% Confidence Interval")

axes = plt.gca()

axes.set_ylim([0,0.6])

plt.xlabel('Classifier')

plt.ylabel('Empirical Risk')

plt.show()
delta = 0.05/5

hoeffding = np.sqrt(float(np.log(2/delta))/(float(2)*m))

lower = empirical_risk - hoeffding

upper = empirical_risk + hoeffding

hoeffding_ci = [lower,upper]





plt.errorbar(["h1","h2","h3","h4","h5"], empirical_risk, yerr=hoeffding, linestyle='',fmt="o")

plt.title("Hoeffding's Simultaneous 95% Confidence Interval")

axes = plt.gca()

axes.set_ylim([0,0.6])

plt.xlabel('Classifier')

plt.ylabel('Empirical Risk')

plt.show()
delta = 0.01/5

hoeffding = np.sqrt(float(np.log(2/delta))/(float(2)*m))

lower = empirical_risk - hoeffding

upper = empirical_risk + hoeffding

hoeffding_ci = [lower,upper]





plt.errorbar(["h1","h2","h3","h4","h5"], empirical_risk, yerr=hoeffding, linestyle='',fmt="o")

plt.title("Hoeffding's Simultaneous 99% Confidence Interval")

axes = plt.gca()

axes.set_ylim([0,0.5])

plt.xlabel('Classifier')

plt.ylabel('Empirical Risk')

plt.show()
print(avg_prec)


avg_prec = []

avg_prec.append(average_precision_score(label_test,l.predict_proba(X_test_one_hot)[:,1]))

avg_prec.append(average_precision_score(label_test,s.predict_proba(X_test_one_hot)[:,1]))

avg_prec.append(average_precision_score(label_test,r.predict_proba(X_test_one_hot)[:,1]))

avg_prec.append(average_precision_score(label_test,k.predict_proba(X_test_one_hot)[:,1]))

avg_prec.append(average_precision_score(label_test,a.predict_proba(X_test_one_hot)[:,1]))





## bootstrap for confidence interval

n_bootstraps = 10000

bootstrap_prec = []

for i in range(5):

    bootstrap_prec.append([])



rng = np.random.RandomState(seed)

for i in range(n_bootstraps):

    indices = rng.randint(0, len(label_test) - 1, len(label_test))

    bootstrap_prec[0].append(average_precision_score(label_test[indices],l.predict_proba(X_test_one_hot[indices,])[:,1], pos_label = 0))

    bootstrap_prec[1].append(average_precision_score(label_test[indices],s.predict_proba(X_test_one_hot[indices,])[:,1], pos_label = 0))

    bootstrap_prec[2].append(average_precision_score(label_test[indices],r.predict_proba(X_test_one_hot[indices,])[:,1], pos_label = 0))

    bootstrap_prec[3].append(average_precision_score(label_test[indices],k.predict_proba(X_test_one_hot[indices,])[:,1], pos_label = 0))

    bootstrap_prec[4].append(average_precision_score(label_test[indices],a.predict_proba(X_test_one_hot[indices,])[:,1], pos_label = 0))



# get standard errors to build the t student confidence interval via bootstrapping

std_error = []

for i in range(5):

    std_error.append(np.sqrt(np.var(bootstrap_prec[i])))



alpha = 0.05

t_stat = t.ppf((1-alpha+1)/2, df = n_bootstraps - 1)

upper_lower_bound = t_stat*std_error[0]



plt.errorbar(["h1","h2","h3","h4","h5"], avg_prec, yerr=upper_lower_bound, linestyle='',fmt="o")

plt.title("Average Precision Score 95% Confidence Interval")

axes = plt.gca()

axes.set_ylim([0.3,0.8])

plt.xlabel('Classifier')

plt.ylabel('Average Precision Score')

plt.show()
alpha = 0.01

t_stat = t.ppf((1-alpha+1)/2, df = n_bootstraps - 1)

upper_lower_bound = t_stat*std_error[0]



plt.errorbar(["h1","h2","h3","h4","h5"], avg_prec, yerr=upper_lower_bound, linestyle='',fmt="o")

plt.title("Average Precision Score 99% Confidence Interval")

axes = plt.gca()

axes.set_ylim([0.3,0.8])

plt.xlabel('Classifier')

plt.ylabel('Average Precision Score')

plt.show()
alpha = 0.05/5

t_stat = t.ppf((1-alpha+1)/2, df = n_bootstraps - 1)

upper_lower_bound = t_stat*std_error[0]



plt.errorbar(["h1","h2","h3","h4","h5"], avg_prec, yerr=upper_lower_bound, linestyle='',fmt="o")

plt.title("Average Precision Score Simultaneous 95% Confidence Interval")

axes = plt.gca()

axes.set_ylim([0.3,0.9])

plt.xlabel('Classifier')

plt.ylabel('Average Precision Score')

plt.show()
alpha = 0.01/5

t_stat = t.ppf((1-alpha+1)/2, df = n_bootstraps - 1)

upper_lower_bound = t_stat*std_error[0]



plt.errorbar(["h1","h2","h3","h4","h5"], avg_prec, yerr=upper_lower_bound, linestyle='',fmt="o")

plt.title("Average Precision Score Simultaneous 99% Confidence Interval")

axes = plt.gca()

axes.set_ylim([0.3,0.9])

plt.xlabel('Classifier')

plt.ylabel('Average Precision Score')

plt.show()