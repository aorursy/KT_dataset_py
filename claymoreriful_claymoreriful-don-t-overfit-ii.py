from IPython.display import display, HTML



display(HTML(data="""

<style>

    div#notebook-container    { width: 95%; }

    div#menubar-container     { width: 65%; }

    div#maintoolbar-container { width: 99%; }

</style>

"""))
import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.decomposition import FastICA, PCA, FactorAnalysis

from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import ExtraTreesClassifier

from scipy.stats import norm

import warnings 

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegressionCV

from sklearn.neighbors import KNeighborsClassifier

from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, DotProduct

from sklearn.naive_bayes import GaussianNB



warnings.filterwarnings('ignore')



pd.set_option('display.max_colwidth', -1)

pd.set_option('display.max_rows', -1)

pd.set_option('display.expand_frame_repr', False)

pd.options.display.max_columns = None
train_data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')

train_label = train_data.pop('target')



test_id = test_data.pop('id')

train_data.drop('id', axis=1, inplace=True)



stats_data = pd.concat([train_data, test_data])



print('Train rows:', train_data.shape)

print('Test rows:', test_data.shape)

print('Stats rows:', stats_data.shape)
# percent missing values

desc = stats_data.describe().T

desc['missing %'] = 1- (desc['count'] / len(stats_data))

#display(desc)
# Scale, PCA, ICA, FA

COMPONENTS = 100



# Standardize the input data

scaler = StandardScaler().fit(stats_data)

train_scaled = scaler.transform(train_data)

stats_scaled = scaler.transform(stats_data)



# PCA

pca = PCA(n_components=COMPONENTS).fit(stats_scaled)

train_pca = pca.transform(train_scaled)

#print("PCA: Calculated Eigenvectors:\n", pca.components_.T)

display("PCA: Variance for each Dimension:", pca.explained_variance_ratio_)

plt.title('PCA: First 2 Eigenvectors')

plt.scatter(train_pca[:, 0], train_pca[:, 1], c=train_label)

plt.show()



# ICA

ica = FastICA(n_components=COMPONENTS).fit(stats_scaled)

train_ica = ica.transform(train_scaled)  

#print("ICA: Calculated Eigenvectors:\n", ica.components_.T)

plt.title('ICA: First 2 Eigenvectors')

plt.scatter(train_ica[:, 0], train_ica[:, 1], c=train_label)

plt.show()



# FA

fa = FactorAnalysis(n_components=COMPONENTS).fit(stats_scaled)

train_fa = fa.transform(train_scaled)

#print("FactorAnalysis: Calculated Eigenvectors:\n", factor.components_.T)

plt.title('FA: First 2 Eigenvectors')

plt.scatter(train_fa[:, 0], train_fa[:, 1], c=train_label)

plt.show()
# Correlations between variables

plt.subplot(121)

sns.heatmap(train_data.corr()).set_title('Train Data Correlation')

plt.subplot(122)

sns.heatmap(stats_data.corr()).set_title('Stats Data Correlation')

plt.show()
# Correlations with Target

absCorrWithDep = []

for var in train_data.columns:

    a = np.corrcoef(train_data[var], train_label)

    a = a[0][1]

    dic = {}

    dic.update(column=var, cor=a)

    absCorrWithDep.append(dic)

absCorrWithDep = pd.DataFrame(absCorrWithDep)



sns.distplot(absCorrWithDep['cor'], hist=True, kde=True, color = 'blue', bins = 30).set_title('Train Data Correlation with Target')

plt.show()



cors = pd.DataFrame(absCorrWithDep['cor'].values)

cors.columns = ['Correlation']

cors['Index'] = range(0, len(cors))

cors['AbsCorr'] = abs(cors['Correlation'])

cors = cors.reindex(columns=['Index', 'Correlation', 'AbsCorr'])

cors = cors.sort_values(by='AbsCorr', ascending=False)



display(cors.head())



plt.plot(cors['AbsCorr'].values)

plt.ylabel('AbsCorr')

plt.show()



print("len(AbsCorr >= 0.2)", len(cors[cors['AbsCorr'] >= 0.2]))

print("len(AbsCorr >= 0.15)", len(cors[cors['AbsCorr'] >= 0.15]))

print("len(AbsCorr >= 0.1)", len(cors[cors['AbsCorr'] >= 0.1]))



cor_list = cors['Index'].values

display("Top 15 Indices in order:", cor_list[0:15])
# Tree Based

# forest of trees to evaluate the importance of features

# fit a number of randomized decision trees on various sub-samples of the dataset and use averaging to rank order features



#orderedParams['DTree'] = {}

orderedParams = {}

orderedImportances = {}



selForestFit = ExtraTreesClassifier(n_estimators = 100).fit(train_scaled, train_label)

importances = selForestFit.feature_importances_



r = []

for a,b in enumerate(importances):

    r.append([a,b])

r = pd.DataFrame(r)

r.columns = ['Index', 'Tree_Importance']

r = r.sort_values(by='Tree_Importance', ascending=False)

display(r.head())



plt.plot(r['Tree_Importance'].values)

plt.ylabel('Tree_Importance')

plt.show()



print("len(Tree_Importance >= 0.0055)", len(r[r['Tree_Importance'] >= 0.0055]))

print("len(Tree_Importance >= 0.005)", len(r[r['Tree_Importance'] >= 0.005]))

print("len(Tree_Importance >= 0.004)", len(r[r['Tree_Importance'] >= 0.004]))



tree_list = r['Index'].values

display("Top 15 Indices in order:", tree_list[0:15])
# Input: 

# an input array with normal distributed float values

# a target array with 0-1 values, denoting an event or not. 1 = event

# an integer that denotes the number of bins



# Output DataFrame: 

# BinNr, Min_Value, Max_Value, Count

# Event, Event_Rate, NonEvent, NonEvent_Rate, Dist_Event, Dist_NonEvent, WoE, IV



def normDist_WoE (frame, target, nrOfBins):

    ranges = []

    for i in range(0, nrOfBins + 1):

        ranges.append(norm.ppf(i / nrOfBins, loc=0, scale=1))

        

    df = pd.DataFrame(frame)

    df.columns = ['val']

    df['target'] = target.astype(int)

    df['bin'] = 0

    for i in range(1, nrOfBins + 1):

        df.loc[(df['val'] >= ranges[i-1]) & (df['val'] < ranges[i]), 'bin'] = i

    

    result = []

    for i in range(1, nrOfBins + 1):

        dic = {}

        dic.update(BinNr=i, Min_Value=ranges[i-1], Max_Value=ranges[i], Count=len(df[df['bin'] == i]),

                   Event=len(df[(df['bin'] == i) & (df['target'] == 1)]), 

                   NonEvent=len(df[(df['bin'] == i) & (df['target'] != 1)]))

        result.append(dic)

    result = pd.DataFrame(result)

    result = result.reindex(columns=(['BinNr', 'Min_Value', 'Max_Value', 'Count', 'Event', 'NonEvent']))



    result['Event_Rate'] = result['Event'] / result['Count']

    result['NonEvent_Rate'] = result['NonEvent'] / result['Count']

    result['Dist_Event'] = result['Event'] / sum(result['Event'])

    result['Dist_NonEvent'] = result['NonEvent'] / sum(result['NonEvent'])

    result['WoE'] = np.log(result['Dist_Event'] / result['Dist_NonEvent'])

    result['IV'] = (result['Dist_Event'] - result['Dist_NonEvent']) * np.log(result['Dist_Event'] /result['Dist_NonEvent'])

    

    return (sum(result['IV']), result)
# Hypothesis:

## Variables are used together and then encoded with an activation function

# Test:

## Bin the variables into X ranges. For each of them, use WeightOfEvidence and IV

## If the variables are run through a monotone function prior to their combination and the activation function, 

## then we may be able to notice something. 

### Try X = 5



# - All variables are normal distributions with a standard deviation of 1

# - All variables are centered around 0

#   -> Cumulative distribution function -> Percent pint function (inverse of cdf) -> percentiles

#   -> Same bin-intervals for all variables



BIN_SIZE = 5



woe_table = pd.DataFrame()

iv_table = []

for i in train_data.columns:

    curr_iv, curr_woe = normDist_WoE(train_data[i], train_label, BIN_SIZE)

    iv_table.append([i, curr_iv])

    curr_woe['Index'] = i

    curr_woe = curr_woe.reindex(columns=['Index'] + curr_woe.columns[:-1].tolist())

    curr_woe = pd.DataFrame(curr_woe)

    woe_table = pd.concat([woe_table, curr_woe])

    

iv_table = pd.DataFrame(iv_table)

iv_table.columns = ['Index', 'IV']

iv_table = iv_table.sort_values(by='IV', ascending=False)



display(iv_table.head())



plt.plot(iv_table['IV'].values)

plt.ylabel('IV')

plt.show()



print("len(IV >= 0.2)", len(iv_table[iv_table['IV'] >= 0.2]))

print("len(IV >= 0.15)", len(iv_table[iv_table['IV'] >= 0.15]))

print("len(IV >= 0.1)", len(iv_table[iv_table['IV'] >= 0.1]))

#display(woe_table)



iv_list = iv_table['Index'].values

display("Top 15 Indices in order:", iv_list[0:15])
# Combine the 3 methods of selecting the top influential Indices



c = cor_list[0:15]

t = tree_list[0:15]

i = iv_list[0:15]

three_top_15_lists = [*c, *t, *i]

three_top_15_lists = list(map(str, three_top_15_lists))

three_top_15_lists = np.array(three_top_15_lists)



unique_elements, counts_elements = np.unique(three_top_15_lists, return_counts=True)



three_top_15_lists = pd.DataFrame({'Index' :unique_elements , 'Count' : counts_elements})

three_top_15_lists = three_top_15_lists.sort_values(by='Count', ascending=False)



display("3x Top Indices:", three_top_15_lists[three_top_15_lists['Count'] == 3].Index.values)

display("2x Top Indices:", three_top_15_lists[three_top_15_lists['Count'] == 2].Index.values)

display("1x Top Indices:", three_top_15_lists[three_top_15_lists['Count'] == 1].Index.values)



# Note: Running PCA, ICA, FA on the reduced sets (filtered by top indices) did not provide any improved results - compared to before
# Prepare split and score for modelling 

train_data_filtered = train_data[[*three_top_15_lists[three_top_15_lists['Count'] == 3].Index.values, 

                                  *three_top_15_lists[three_top_15_lists['Count'] == 2].Index.values]]



test_data_filtered = test_data[[*three_top_15_lists[three_top_15_lists['Count'] == 3].Index.values, 

                                *three_top_15_lists[three_top_15_lists['Count'] == 2].Index.values]]



train_label_int = train_label.astype(int) # for easier classification



display(train_data_filtered.head())

display(test_data_filtered.head())



train_X, val_X, train_y, val_y = train_test_split(train_data_filtered, train_label_int, test_size=0.1, random_state = None, shuffle = True)
# One in ten rule

## -> limit to Nr. of events / 10 -> learnable parameters

## -> 250 train data /2 (0/1 labels) -> ~12 parameters

## -> only use the 3x and 2x top indices



# Logistic Regression

logit = LogisticRegressionCV(solver='liblinear', cv=5, dual=False, penalty='l1', multi_class='ovr')

logit.fit(train_X, train_y)

score = logit.score(val_X, val_y)

print("Logistic Regression:", score)

print(logit.predict(val_X))

display([p[1] for p in logit.predict_proba(val_X)])



# Nearest Neighbors

clf = KNeighborsClassifier(n_neighbors=3, weights='distance', algorithm='auto', p=2)

clf.fit(train_X, train_y)

score = clf.score(val_X, val_y)

print("KNN:", score)

#result = clf.predict(val_X)

#result_prob = clf.predict_proba(val_X)



# Gaussian Process

## Kernels: RBF, Matern (generalized RBF), Rational quadratic, dot-product

for k in [RBF(), Matern(), RationalQuadratic(), DotProduct()]:

    gpc = GaussianProcessClassifier(kernel=k, n_restarts_optimizer=5, max_iter_predict=200)

    gpc.fit(train_X, train_y)

    score = gpc.score(val_X, val_y)

    print("GaussianProcess", k, score)

    #result = gpc.predict(val_X)

    #result_prob = gpc.predict_proba(val_X)

#print(gpc.predict(val_X)) #  Dot Product

#display([p[1] for p in gpc.predict_proba(val_X)])



# Naive Bayes

nb = GaussianNB()

nb.fit(train_X, train_y)

score = nb.score(val_X, val_y)

print("Naive Bayes:", score)

print(nb.predict(val_X))

display([p[1] for p in nb.predict_proba(val_X)])
# Noted score values:

#  GaussianProcess RBF:               [0.56, 0.68, 0.6,  0.52, 0.84, 0.6,  0.8]

#  GaussianProcess Matern:            [0.56, 0.68, 0.6,  0.8,  0.84, 0.6,  0.8]

#  GaussianProcess RationalQuadratic: [0.68, 0.68, 0.8,  0.8,  0.84, 0.6,  0.8]

#  GaussianProcess DotProduct:        [0.76, 0.72, 0.84, 0.76, 0.8,  0.68, 0.76]

#  KNN:                               [0.68, 0.68, 0.8,  0.64, 0.8,  0.52, 0.68]

#  Logistic Regression:               [0.76, 0.72, 0.84, 0.76, 0.84, 0.64, 0.76]

#  Naive Bayes:                       [0.68, 0.6,  0.72, 0.84, 0.84, 0.64, 0.8]

 

# Conclusion:

# - KNN is worse than Logistic Regression and Naive Bayes

# - DotProduct is the best GaussianProcess. Same average than RationalQuadratic but a little less variance -> more stable

 

# - Logistic Regression and DotProduct are very similar but Logistic Regression is the simpler concept -> Razor says to pic Logit in this case

# -> This holds true for assigning very similar probabilities

 

# Decision:

# => Use the median of Logistic Regression and Naive Bayes for the final probability
# 1) Re-Train Logistic and Naive Bayes with the full training set

# 2) Predict the test set

# 3) Write output into submission.csv

logit = LogisticRegressionCV(solver='liblinear', cv=5, dual=False, penalty='l1', multi_class='ovr')

logit.fit(train_data_filtered, train_label_int)

logit_prediction = [p[1] for p in logit.predict_proba(test_data_filtered)]



nb = GaussianNB()

nb.fit(train_data_filtered, train_label_int)

nb_prediction = [p[1] for p in nb.predict_proba(test_data_filtered)]



sub_preds = [(i + j) / 2 for i, j in zip(logit_prediction, nb_prediction)]

submission = [[a, b] for a, b in zip(test_id.values, sub_preds)]

submission = pd.DataFrame(submission, columns=['id', 'target'])



display(submission.head())

print("len(submission)", len(submission))

print("len(target ~ 1)", len(submission[submission.target >= 0.5]))

print("len(target ~ 0)", len(submission[submission.target < 0.5]))



sns.distplot(submission['target'], hist=True, kde=True, color = 'blue', bins = 25).set_title('Predictions')

plt.show()



submission.to_csv("submission.csv", index=False)