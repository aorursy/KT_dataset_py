import pandas as pd

import numpy as np

import xgboost as xgb

import matplotlib.pyplot as plt

import seaborn as sns



sns.set_style('whitegrid')



df = pd.read_csv('../input/creditcard.csv')

df = df.sample(frac=1, random_state=123) #Shuffle samples



class_imb = df.Class.sum()/len(df) #Record class imbalance



#Pie plot

labels = ['Genuine Transactions', 'Fraudulent Transactions']

sizes = [(len(df)-df.Class.sum()), df.Class.sum()]

colors = ['blue', 'red']



plt.pie(sizes, labels=labels, colors=colors,

        autopct='%1.1f%%', shadow=True, startangle=140)

plt.title('Genuine Transactions vs. Fraudulent Transactions')

plt.axis('equal')

plt.show()
#Define our features, omitting time

features = ['V{}'.format(x) for x in range(1, 29)] + ['Amount']

#Set cutoff at 50%

cutoff=int(.5*len(df))

#Split data into training and test 

train = df[:cutoff]

test = df[cutoff:]



#Split up into X's and Y's

X_train = train[features]

Y_train = train.Class



X_test = test[features]

Y_test = test.Class
#record results:

results1 = pd.DataFrame()

results1['true_value'] = np.array(Y_test)

results1['predicted'] = 0

results1['correct'] = np.where(results1.predicted == results1.true_value, 1, 0)

results1['tI_error'] = np.where((results1.predicted == 1) & (results1.true_value == 0), 1, 0)

results1['tII_error'] = np.where((results1.predicted == 0) & (results1.true_value == 1), 1, 0)



labels = ['False Negatives (type II errors)', 'Correctly Identified']

sizes = [results1.tII_error.sum(), results1.correct.sum()]

colors = ['red', 'blue']



plt.pie(sizes, labels=labels, colors=colors,

        autopct='%1.1f%%', shadow=True, startangle=140)

plt.title('Model 1 Performance: Error Breakdown')

plt.axis('equal')

plt.show()



results1['Amount'] = np.array(X_test['Amount'])

#type I error cost, as defined

results1['cost_I'] = np.where(results1.tI_error == 1, (25 + 10 + (.01*results1.Amount)), 0)

#type II error cost, as defined

results1['cost_II'] = np.where(results1.tII_error == 1, results1.Amount, 0)

results1['total_cost'] = results1[['cost_I', 'cost_II']].sum(axis=1)

total_cost_I = results1.cost_I.sum()

total_cost_II = results1.cost_II.sum()

total_cost = total_cost_I + total_cost_II

        

fig, ax = plt.subplots(figsize=(8, 5))

plt.bar(

    ['cost due to type I errors', 'cost due to type II errors', 'total_cost'], 

    [total_cost_I, total_cost_II, total_cost],

    color=['yellow', 'orange', 'green']

);

plt.title('Contributions to Cost');

plt.ylabel('Cost in Dollars');

plt.show();



print("The total cost of this model on our test set is: ${}".format(str(total_cost)))
#Set up our data and parameters

dtrain = xgb.DMatrix(X_train, label=Y_train)

param = {

     'colsample_bytree': 0.4,

     'eta': 14/100,

     'max_depth': 3,

     'nthread': 4,

     'objective': 'binary:logistic',

     'scale_pos_weight': .5/class_imb,

     'silent': 1

}



plst = param.items()



#Train the model (cue 'Eye of the Tiger')

num_round = 500

bst = xgb.train(plst, dtrain, num_round, verbose_eval=False)



#Grab predictions at p=.5 threshold

dtest = xgb.DMatrix(X_test)

Y_ = bst.predict(dtest)

Y_ = np.where(Y_>.5, 1, 0)
#record results:

results2 = pd.DataFrame()

results2['true_value'] = np.array(Y_test)

results2['predicted'] = Y_

results2['correct'] = np.where(results2.predicted == results2.true_value, 1, 0)

results2['tI_error'] = np.where((results2.predicted == 1) & (results2.true_value == 0), 1, 0)

results2['tII_error'] = np.where((results2.predicted == 0) & (results2.true_value == 1), 1, 0)



fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))



ax1.set_title('Model 2 Performance: Error Breakdown')

lbl = ['Incorrectly Identified', 

       'Correctly Identified ({}%)'.format(str(100*results2.correct.sum()/len(results2))[:5])

]

sz = [results2.tII_error.sum() + results2.tI_error.sum(), results2.correct.sum()]

clrs = ['red', 'blue']

ax1.pie(sz, labels=lbl, colors=clrs, shadow=True, startangle=90)



ax2.set_title('Error Types')

labels = [

    'Type I Errors (False Positives): {}'.format(results2.tI_error.sum()),

    'Type II Errors (False Negatives): {}'.format(results2.tII_error.sum())

]

sizes = [

    results2.tI_error.sum(),

    results2.tII_error.sum()

]



ax2.pie(sizes, labels=labels, colors=['cyan', 'purple'], startangle=215, 

        autopct='%1.1f%%')

plt.subplots_adjust(wspace=1.5)

plt.show()



results2['Amount'] = np.array(X_test['Amount'])

#type I error cost, as defined

results2['cost_I'] = np.where(results2.tI_error == 1, (25 + 10 + (.01*results2.Amount)), 0)

#type II error cost, as defined

results2['cost_II'] = np.where(results2.tII_error == 1, results2.Amount, 0)

results2['total_cost'] = results2[['cost_I', 'cost_II']].sum(axis=1)

total_cost_I = results2.cost_I.sum()

total_cost_II = results2.cost_II.sum()

total_cost = total_cost_I + total_cost_II

        

fig, ax = plt.subplots(figsize=(8, 5))

plt.bar(

    ['cost due to type I errors', 'cost due to type II errors', 'total cost'], 

    [total_cost_I, total_cost_II, total_cost],

    color=['cyan', 'purple', 'green']

);

plt.title('Contributions to Cost');

plt.ylabel('Cost in Dollars');

plt.show();



print("The total cost of this model on our test set is: ${}".format(str(total_cost)[:7]))