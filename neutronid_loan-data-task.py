# imports

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from statsmodels.discrete import discrete_model

from sklearn.model_selection import train_test_split

from sklearn.ensemble import AdaBoostClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score

from tabulate import tabulate
# path to data file

datafile = '../input/loan_data.csv'



# read csv to df

loandata = pd.read_csv(datafile, sep=',', index_col=0)



# list all possible variables

#list(loandata)
# label of numeric variable

hist_numeric = 'duration'

# number of bins for histogram

hist_bins = 50



# split data

loan_yes = loandata.where(loandata['y']==1).dropna()

loan_no = loandata.where(loandata['y']==0).dropna()



plt.figure()

# plot histograms

plt.hist(loan_yes[hist_numeric], hist_bins, facecolor='b', alpha=0.5, label='Yes')

plt.hist(loan_no[hist_numeric], hist_bins, facecolor='r', alpha=0.5, label='No')



plt.xlabel(hist_numeric)

plt.legend()

plt.title('Histogram of ' + hist_numeric)



plt.show()
# toggle whether campaign is a categorical or numerical value

camp_categorical = False



# p-value threshold for feature selection

pval_thres = 1e-5
# remove day and month columns from the analysis

loandata = loandata.drop(['day', 'month'], axis=1)
# list of vars that are categorical and need encoding

enccats = [

        'job',

        'marital',

        'education',

        'default',

        'housing',

        'loan',

        'contact',

        'poutcome'

        ]

# add campaign to categories if selected

if camp_categorical:

    enccats.append('campaign')

    

# count possible values of each category

catcount = []

for k in enccats:

    catcount.append(np.shape(loandata[k].unique())[0])

    

# one-hot-encoding of categorical vars

encdata = pd.get_dummies(loandata, columns=enccats)
# select one of each categories

fix_multicoll = [

        'job_unemployed',

        'marital_married',

        'education_tertiary',

        'default_no',

        'housing_no',

        'loan_yes',

        'contact_cellular',

        'poutcome_failure'

        ]

if camp_categorical:

    fix_multicoll.append('campaign_1')



# first value to occur is removed

#fix_multicoll = []

#for k in enccats:

#    fix_multicoll.append(k + '_' + loandata[k].unique()[0])



# remove the columns accordingly

encdata_fmc = encdata.drop(fix_multicoll, axis=1)



# if campaign is categorical remove campaign IDs that only occur once as these cannot be used

if camp_categorical:

#    count occurences of unique campaign IDs

    campcount = loandata['campaign'].value_counts()

#     to array

    campcount = np.array([campcount.index,campcount.values]).T

#    select all that only occur once

    campsingle = campcount[campcount[:,1]<2,0]

#    list of labels

    fix_campaign = []

    for k in campsingle:

        fix_campaign.append('campaign_' + str(k))

#    remove columns accordingly

    encdata_fmc = encdata_fmc.drop(fix_campaign, axis=1)
# output data

lrdata_out = encdata_fmc['y']



# input data

lrdata_in = encdata_fmc.drop('y', axis=1)



# list of all possible features after encoding

features = list(lrdata_in)



# results of each iteration of logistic regression stored for later analysis

store_results = []



# features that were initially removed to avoid multicollinearity are reintroduced when encoded features corresponding to the same variable are removed

reintroduce = fix_multicoll



print('Reducing features...')



rsquaredval = []

removed = []

# iterate logistic regression and feature reduction

removing_features = True

while removing_features:

    removing_features = False

    

    # data of features in use

    lrdata_in = encdata.filter(features, axis=1)

    

    # model for logistic regression

    logit_model = discrete_model.Logit(lrdata_out, lrdata_in)



    # fit model with regularized data

    logit_result = logit_model.fit_regularized(disp=False)

    # store results

    store_results.append(logit_result)

    rsquaredval.append(logit_result.prsquared)



    # p-values of all features

    pvaldict = dict(logit_result.pvalues)



    # features with highest p-value removed (least significant)

    remove_feature = max(pvaldict, key=pvaldict.get)

    if pvaldict[remove_feature] > pval_thres:

        features.remove(remove_feature)

        removed.append(remove_feature)

        #print(remove_feature + ' removed')

        removing_features = True

        

    # enoded features corresponding to each categorical variable are counted and initially removed features are added once possible

    searchstr = ";".join(features)

    for k in range(len(enccats)):

        # count encoded features of each categorical

        tempcount = searchstr.count(enccats[k]+'_')

        # if possible, add initially removed feature

        if (tempcount < catcount[k]-1) and not reintroduce[k] == '':

            features.append(reintroduce[k])

            #print(reintroduce[k] + ' brought back')

            reintroduce[k] = ''



# for re-runs            

reintroduce = fix_multicoll
# plot the change in pesudo R^2 over iterations 

# pseudo R^2 should not decrese significantly when removing features, only for observation in this implementation

plt.figure()

plt.plot(np.r_[0:len(rsquaredval)], rsquaredval)



plt.ylim(0, 1.1*max(rsquaredval))

plt.xlabel('Iterations')

plt.title(r'Evolution of pseudo $R^2$')



plt.show()
# removed features

removed
# Results of first logistic regression

store_results[0].summary()
# Results of last logistic regression

logit_result.summary()
# selected features

features
# split data into training and validation datasets

train_in, valid_in, train_out, valid_out = train_test_split(encdata.filter(features, axis=1), encdata['y'], test_size=0.3, random_state=0)



# set AdaBoosted decision tree classifier

classifier = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3), algorithm="SAMME", n_estimators=100)



# train classifier

score_out = classifier.fit(train_in, train_out).decision_function(valid_in)



# predict output with validation data

predict_out = classifier.predict(valid_in)

# accuracy score

print('Accuracy score ' + '%.4f' % accuracy_score(valid_out, predict_out) + '\n')



# confusion matrix

confusion = confusion_matrix(valid_out, predict_out)



print('Confusion matrix')

print(tabulate([['Predicted yes', confusion[0,0], confusion[0,1]], ['Predicted no', confusion[1,0], confusion[1,1]]], headers=['', 'Actual yes', 'Actual no']))
