# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

from scipy import stats

import os, sys

pd.options.mode.chained_assignment = None # subpress some warnning



import warnings

warnings.filterwarnings("ignore")
from IPython.display import HTML



HTML('''<script>

code_show=true; 

function code_toggle() {

 if (code_show){

 $('div.input').hide();

 } else {

 $('div.input').show();

 }

 code_show = !code_show

} 

$( document ).ready(code_toggle);

</script>

<form action="javascript:code_toggle()"><input type="submit" value="Click here to toggle on/off the raw code."></form>''')
full_data = pd.read_csv('../input/UCI_Credit_Card.csv')

full_data.rename(columns={'default.payment.next.month': 'dpnm'}, inplace=True)

print('data_origin.shape: ', full_data.shape)

print('data_target_value_counts:')

print(full_data['dpnm'].value_counts())

full_data.head()
data_origin = full_data
from imblearn.over_sampling import SMOTE
X_sampled, Y_sampled = SMOTE().fit_sample(full_data.drop('dpnm', axis=1), full_data.dpnm)

XY_sampled = np.append(X_sampled, Y_sampled.reshape(Y_sampled.shape[0], 1), axis=1)



cata_variables = ['ID', 'AGE', 'EDUCATION', 'SEX', 'MARRIAGE', 'PAY_0','PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'dpnm']

XY_sampled_m = pd.DataFrame(XY_sampled, columns=full_data.columns)

XY_sampled_m[cata_variables] = XY_sampled_m[cata_variables].astype(int)

data_origin = XY_sampled_m



print('data_dimension after sampling: ')

print(data_origin.shape)
from termcolor import colored

import numpy as np

import seaborn as sns

import matplotlib.pylab as plt

from sklearn.feature_selection import RFE

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import MinMaxScaler

print(colored('DATA_FEATURES: ', 'yellow'))

pd.DataFrame(data_origin.columns).T
def describe_factor(x_):

    """

    describe data features, trying to find nan

    :param x: dataframe

    return: level conclusion

    """

    level_count = dict()

    

    for lvl in x_.unique():

        if pd.isnull(lvl):

            level_count["NaN"] = x_.isnull().sum()

        else:

            level_count[lvl] = np.sum(x_==lvl)

    return level_count



print('Describe and reform category data:')

print('\n')

print(colored('Sex:', 'red'))

print(describe_factor(data_origin['SEX']))

print(colored('Education: ', 'red'))

print(describe_factor(data_origin["EDUCATION"]))

data_origin["EDUCATION"] = data_origin["EDUCATION"].map({0: np.NaN, 1:1, 2:2, 3:3, 4:np.NaN, 

    5: np.NaN, 6: np.NaN})

print(colored('For Education, (0, 5, 6) should be setted to be NA for further analysis, then;', 'yellow'))

print(describe_factor(data_origin["EDUCATION"]))

print(colored('Marriage:', 'red'))

print(describe_factor(data_origin['MARRIAGE']))

data_origin.MARRIAGE = data_origin.MARRIAGE.map({0:np.NaN, 1:1, 2:2, 3:3})

print(colored('For Marriage, (0) should be setted to be NA for further analysis, then;', 'yellow'))

print(describe_factor(data_origin.MARRIAGE))
print("Others are quantitative")

print('\n')

print('#'*8, ' CHECK NULL:  ', '#'*8)

print(data_origin.isnull().sum())
data_origin["EDUCATION"][data_origin["EDUCATION"].isnull()] = data_origin["EDUCATION"].mode().values

data_origin["MARRIAGE"][data_origin["MARRIAGE"].isnull()] = data_origin["MARRIAGE"].mode().values



print('After imputation, check null: ')

print('the number of "NAN": ', data_origin.isnull().sum().sum())
corr = data_origin.drop(['ID'], axis=1).corr()



# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})

plt.show()
lr_raw_model = LogisticRegression()

mmscale = MinMaxScaler()

X_scaled = mmscale.fit_transform(full_data.iloc[:,1:-1])

rfe_lr = RFE(lr_raw_model, 1)

fit = rfe_lr.fit(X_scaled, full_data.dpnm)

print("Num Features:",fit.n_features_)

print("Selected Features:",fit.support_)

print("Feature Ranking: ",fit.ranking_)

print('\n')
from sklearn.model_selection import GridSearchCV

from sklearn.tree import DecisionTreeClassifier,export_graphviz

import numpy as np

import math

from scipy import stats

from sklearn.utils.multiclass import type_of_target



class WOE:

    def __init__(self):

        self._WOE_MIN = -20

        self._WOE_MAX = 20



    def woe(self, X, y, event=1):

        '''

        Calculate woe of each feature category and information value

        :param X: 2-D numpy array explanatory features which should be discreted already

        :param y: 1-D numpy array target variable which should be binary

        :param event: value of binary stands for the event to predict

        :return: numpy array of woe dictionaries, each dictionary contains woe values for categories of each feature

                 numpy array of information value of each feature

        '''

        self.check_target_binary(y)



        res_woe = []

        res_iv = []

        for i in range(0, X.shape[-1]):

            x = X[:, i]

            woe_dict, iv1 = self.woe_single_x(x, y, event)

            res_woe.append(woe_dict)

            res_iv.append(iv1)

        return np.array(res_woe), np.array(res_iv)



    def woe_single_x(self, x, y, event=1):

        '''

        calculate woe and information for a single feature

        :param x: 1-D numpy starnds for single feature

        :param y: 1-D numpy array target variable

        :param event: value of binary stands for the event to predict

        :return: dictionary contains woe values for categories of this feature

                 information value of this feature

        '''

        self.check_target_binary(y)



        event_total, non_event_total = self.count_binary(y, event=event)

        x_labels = np.unique(x)

        woe_dict = {}

        iv = 0

        for x1 in x_labels:

            y1 = y[np.where(x == x1)[0]]

            event_count, non_event_count = self.count_binary(y1, event=event)

            rate_event = 1.0 * (event_count + 1.0) / (event_total + 2.0)

            rate_non_event = 1.0 * (non_event_count + 1.0) / (non_event_total + 2.0)

            woe1 = math.log(rate_event / rate_non_event)

            woe_dict[x1] = woe1

            iv += (rate_event - rate_non_event) * woe1

        return woe_dict, iv



    def woe_replace(self, X, woe_arr):

        '''

        replace the explanatory feature categories with its woe value

        :param X: 2-D numpy array explanatory features which should be discreted already

        :param woe_arr: numpy array of woe dictionaries, each dictionary contains woe values for categories of each feature

        :return: the new numpy array in which woe values filled

        '''

        if X.shape[-1] != woe_arr.shape[-1]:

            raise ValueError('WOE dict array length must be equal with features length')



        res = np.copy(X).astype(float)

        idx = 0

        for woe_dict in woe_arr:

            for k in woe_dict.keys():

                woe = woe_dict[k]

                res[:, idx][np.where(res[:, idx] == k)[0]] = woe * 1.0

            idx += 1



        return res



    def combined_iv(self, X, y, masks, event=1):

        '''

        calcute the information vlaue of combination features

        :param X: 2-D numpy array explanatory features which should be discreted already

        :param y: 1-D numpy array target variable

        :param masks: 1-D numpy array of masks stands for which features are included in combination,

                      e.g. np.array([0,0,1,1,1,0,0,0,0,0,1]), the length should be same as features length

        :param event: value of binary stands for the event to predict

        :return: woe dictionary and information value of combined features

        '''

        if masks.shape[-1] != X.shape[-1]:

            raise ValueError('Masks array length must be equal with features length')



        x = X[:, np.where(masks == 1)[0]]

        tmp = []

        for i in range(x.shape[0]):

            tmp.append(self.combine(x[i, :]))



        dumy = np.array(tmp)

        # dumy_labels = np.unique(dumy)

        woe, iv = self.woe_single_x(dumy, y, event)

        return woe, iv



    def combine(self, list):

        res = ''

        for item in list:

            res += str(item)

        return res



    def count_binary(self, a, event=1):

        event_count = (a == event).sum()

        non_event_count = a.shape[-1] - event_count

        return event_count, non_event_count



    def check_target_binary(self, y):

        '''

        check if the target variable is binary, raise error if not.

        :param y:

        :return:

        '''

        y_type = type_of_target(y)

        if y_type not in ['binary']:

            raise ValueError('Label type must be binary')



    def feature_discretion(self, X):

        '''

        Discrete the continuous features of input data X, and keep other features unchanged.

        :param X : numpy array

        :return: the numpy array in which all continuous features are discreted

        '''

        temp = []

        for i in range(0, X.shape[-1]):

            x = X[:, i]

            x_type = type_of_target(x)

            if x_type == 'continuous':

                x1 = self.discrete(x)

                temp.append(x1)

            else:

                temp.append(x)

        return np.array(temp).T



    def discrete(self, x):

        '''

        Discrete the input 1-D numpy array using 5 equal percentiles

        :param x: 1-D numpy array

        :return: discreted 1-D numpy array

        '''

        res = np.array([0] * x.shape[-1], dtype=int)

        for i in range(5):

            point1 = stats.scoreatpercentile(x, i * 20)

            point2 = stats.scoreatpercentile(x, (i + 1) * 20)

            x1 = x[np.where((x >= point1) & (x <= point2))]

            mask = np.in1d(x, x1)

            res[mask] = (i + 1)

        return res



    @property

    def WOE_MIN(self):

        return self._WOE_MIN

    @WOE_MIN.setter

    def WOE_MIN(self, woe_min):

        self._WOE_MIN = woe_min

    @property

    def WOE_MAX(self):

        return self._WOE_MAX

    @WOE_MAX.setter

    def WOE_MAX(self, woe_max):

        self._WOE_MAX = woe_max
# using tree to discretize continous variables 'LIMIT_BAL' as an example.



reg_labels = ['LIMIT_BAL', 'AGE', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1',

       'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']

data_cut_dt = data_origin.copy()

y = data_cut_dt['dpnm']  

dt_model_ori = DecisionTreeClassifier(random_state=9, criterion='gini')

params_to_try = {'max_leaf_nodes': [5, 10, 15]}



grid_dt = GridSearchCV(dt_model_ori, params_to_try, n_jobs=-1, verbose=0)

grid_dt.fit(pd.DataFrame(data_origin['LIMIT_BAL']),y)

grid_dt.best_estimator_
# using decision tree to discretize the continuous variables.



woe_c = WOE() # function from tools





max_iv = []   

group_numbers_to_try = [4, 5, 8, 9, 10, 12, 15]



print(colored('the resulted rolling differences or deriavtes of each IV for each label are: ', 'yellow'))



for label in reg_labels:

    print('\n')

    print(colored('label: {}'.format(label), 'red'))

    

    iv_i = []

    X_to_use = pd.DataFrame(data_cut_dt[label])

    

    for g_n in group_numbers_to_try:

        dt_model = DecisionTreeClassifier(random_state=9, max_leaf_nodes=g_n, criterion='gini')

        dt_model.fit(X_to_use, y)

        X_pred_group_number = dt_model.apply(X_to_use, check_input=True)

        iv_i.append(woe_c.woe_single_x(x=np.array(X_pred_group_number), y=y)[1]) #calculate iv for each feature and each tried value

        

    max_iv_i = np.argmax(iv_i) # the position of max iv for each feature

    max_iv.append(max_iv_i)

    print(np.diff(iv_i) / np.diff(group_numbers_to_try))
# after selecting the best cut_lines mannually.



cut_bins = [5, 4, 10, 12, 15, 10, 10, 12, 8, 8, 12, 10, 8, 12]



for i, label in enumerate(reg_labels):

    X_to_use = pd.DataFrame(data_cut_dt[label])

    

    dt_model = DecisionTreeClassifier(random_state=9, max_leaf_nodes=cut_bins[i], criterion='gini')

    dt_model.fit(X_to_use, y)

    X_pred_group_number = dt_model.apply(X_to_use, check_input=True)

    X_pred_group_number_uniq = np.unique(X_pred_group_number)

    

    for j in range(len(X_pred_group_number_uniq)):

        data_cut_dt[label][X_pred_group_number == X_pred_group_number_uniq[j]] = j # convert label into 0,1,2,3,...
data_cut_dt.head()
woes, iv = woe_c.woe(X=np.array(data_cut_dt.iloc[:,1:-1]), y=y)



print('IV for each feature after discretization: ')

pd.DataFrame(iv, index=data_cut_dt.iloc[:,1:-1].columns)
from sklearn.cluster import KMeans
def kmeans_cut(data_origin, labels, k):

    """

    Using k-means to discretize continuous features

    :param data_origin: original data_set

    :param labels: column names which need discretization

    :param k: the number of groups to be divided

    :return: centers, cutting_points and transformed data

    """

    data = data_origin.copy()



    for label in labels:

        k_model = KMeans(n_clusters=k)

        k_model.fit(pd.DataFrame(data[label]))

        cut_centers = pd.DataFrame(k_model.cluster_centers_).sort_values(0)

        cutting_points = cut_centers.rolling(2).mean().iloc[1:]  # 相邻两项求重点，作为边界点



        # get minimum point of cutting points

        if data[label].min() > 0:

            cutting_points = [0] + list(cutting_points[0]) + [data[label].max() + 1]

        else:

            cutting_points = [data[label].min()] + list(cutting_points[0]) + [data[label].max() + 10]



        data_cut_group_label = pd.cut(data[label], bins=cutting_points, labels=range(k), right=False)

        data[label] = data_cut_group_label

    return cut_centers, cutting_points, data
# trying k-means distreriztion , see any difference



data_cut_kms = data_origin.copy()



print(colored('the resulted rolling differences or deriavtes of each IV for each label are: ', 'yellow'))



for i, label in enumerate(reg_labels):

    c =[]

    c.append(label) # convert label to be a list

    

    centers, cutting_points, resulted_data = kmeans_cut(data_cut_kms, c, k=cut_bins[i])

    data_cut_kms = resulted_data.copy()

    

    print('\n')

    print('#'*10, label,':')

    print(cutting_points)
data_cut_kms.head()
woes, iv = woe_c.woe(X=np.array(data_cut_kms.iloc[:,1:-1]), y=y)



print('IV for each feature after discretization: ')

pd.DataFrame(iv, index=data_cut_kms.iloc[:,1:-1].columns)
data_cut_dt.drop(['ID','SEX', 'MARRIAGE', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6'],

                 axis=1, inplace=True)



# delete SEX, MARRIAGE, BILL_AMOUNT

data_cut_X = data_cut_dt.iloc[:,:-1]

data_cut_Y = data_cut_dt.iloc[:,-1]
data_cut_kms.drop(['ID','AGE', 'SEX', 'MARRIAGE', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6'], 

                  axis=1, inplace=True)



# delete SEX, MARRIAGE, BILL_AMOUNT

data_cut_X = data_cut_kms.iloc[:,:-1]

data_cut_Y = data_cut_kms.iloc[:,-1]
from sklearn.model_selection import StratifiedShuffleSplit
def split_data(size, data_x, data_y):

    """

    split full data into training and testing data

    :param size: percentile for testing data

    :param data_x: dataframe of full data

    :param data_y: target

    """

    sss = StratifiedShuffleSplit(n_splits=2, test_size=size, random_state=9)

    split1, split2 = sss.split(data_x, data_y)

    x_train, x_test = data_x.iloc[split1[0]], data_x.iloc[split1[1]]

    Y_train, Y_test = data_y[split1[0]], data_y[split1[1]]

    return x_train, x_test, Y_train, Y_test
# eliminate negative integers



pay_status = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']



for pay in pay_status:

    pay_value = data_cut_X[pay]

    data_cut_X[pay][pay_value == -1] = 9

    data_cut_X[pay][pay_value == -2] = 10



# now split

X_train, X_test, y_train, y_test = split_data(0.2, data_cut_X, data_cut_Y) # use 20% percentile testing data here
print(colored('FULL_X_DATA: ', 'green'))

print(data_cut_X.shape)

print(colored('X_TRAIN.shape: ', 'green'))

print(X_train.shape)

print(colored('X_TEST.shape: ', 'green'))

print(X_test.shape)
from sklearn.preprocessing import OneHotEncoder
# one_hot encoder



one_hot = OneHotEncoder(sparse=True)

one_hot.fit(data_cut_X)

X_train_oh = one_hot.transform(X_train)

X_test_oh = one_hot.transform(X_test)



#using dummy to get column names

X_dummy = pd.get_dummies(X_train, columns=X_train.columns)

names = X_dummy.columns

X_train_oh = pd.DataFrame(X_train_oh.toarray(), columns=names, index=X_train.index)

X_test_oh = pd.DataFrame(X_test_oh.toarray(), columns=names, index=X_test.index)
print(colored('After encodering: ', 'yellow'))

print(colored('X_TRAIN.shape: ', 'green'))

print(X_train_oh.shape)

print(colored('X_TEST.shape: ', 'green'))

print(X_test_oh.shape)
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.svm import LinearSVC

import matplotlib.pylab as plt

from sklearn.metrics import roc_auc_score, roc_curve, precision_score, recall_score, accuracy_score, auc, confusion_matrix, f1_score
# classifier to be tried.



clfs = {'LogisticRegression':LogisticRegressionCV(Cs=10, scoring='recall', penalty='l1', solver='liblinear'),

       'RandomForest': RandomForestClassifier(n_estimators=100),

       'GradientBoosting': GradientBoostingClassifier(learning_rate= 0.05, max_depth= 6,

                                                        n_estimators=200, max_features = 0.3,

                                                        min_samples_leaf = 5)}
# original_oh = pd.read_csv(DATA_PATH + 'original_data_oh.csv').iloc[:,1:]



cols = ['model', 'auc', 'precision_score', 'recall_score', 'f1_score', 'accuracy', 'train_score']

models_report = pd.DataFrame(columns=cols)

feature_importance = pd.DataFrame()

conf_matrix = dict()



for clf, clf_name in zip(clfs.values(), clfs.keys()):

    # fit model

    clf.fit(X_train_oh, y_train)

    y_pred = clf.predict(X_test_oh)

#     y_original_pred = clf.predict(original_oh)

    y_score = pd.DataFrame(clf.predict_proba(X_test_oh)).iloc[:,1]

#     y_original_score = pd.DataFrame(clf.predict_proba(original_oh)).iloc[:,1]



    print('Computing{}'.format(clf_name))

    

    # add features importance

    if (clf_name == 'RandomForest') | (clf_name == 'GradientBoosting'):

        tmp_fi = pd.Series(clf.feature_importances_)

        feature_importance[clf_name] = tmp_fi

    

    # calculate required metrics

    tmp = pd.Series({

                    'model': clf_name,

                    'auc': roc_auc_score(y_test, y_score),

                    'precision_score':precision_score(y_test, y_pred),

                    'recall_score':recall_score(y_test, y_pred),

                    'f1_score': f1_score(y_test, y_pred),

                    'accuracy': accuracy_score(y_test, y_pred),

                    'train_score': clf.score(X_train_oh, y_train),

#                     'original_recall': recall_score(full_data.dpnm, y_original_pred),

#                     'original_auc': roc_auc_score(full_data.dpnm, y_original_score),

                    })

    models_report = models_report.append(tmp, ignore_index = True)

    conf_matrix[clf_name] = pd.crosstab(y_test, y_pred, rownames=['True'], colnames= ['Predicted'], margins=False)

    

    # plot roc

    fpr, tpr, _ = roc_curve(y_test, y_score, drop_intermediate = False, pos_label = 1)

    auc_value = auc(fpr, tpr)

    

    plt.figure(1, figsize = (9,9))

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.title('Receiver operating characteristic example')

    plt.plot(fpr, tpr, lw=2, label=clf_name + '(area = %0.2f)' % auc_value)

    plt.legend(loc="lower right")

    

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

plt.show()
print('models_report: ')

models_report
fi = feature_importance

features = X_train_oh.columns

fi.index = features

fi = fi.head(15) # Only take the 15 most important metrics

fi = fi.sort_values('RandomForest', ascending=False)

fi = (fi / fi.sum(axis=0)) * 100

fi.plot.barh(title = 'Feature importances for Tree algorithms', figsize = (6,9))

plt.show()
# fit logistic regression model



lrcv = LogisticRegressionCV(Cs=10, class_weight='balanced', scoring='recall', penalty='l1', solver='liblinear')

lrcv.fit(X_train_oh, y_train)

lrcv.C_  # c selected



print('scores for training data: ')

print(lrcv.score(X_train_oh, y_train))

print('scores for testing data: ')

print(lrcv.score(X_test_oh, y_test))
def roc_curve_plot(classifier, data, y, event=1):

    """

    plot rov curve and also calculate auc

    :param classifier: defined and fitted classifier

    :param data: data for calculating, dataframe

    :param y: true target value for data

    :return: auc value and plot

    """



    y_pred = classifier.decision_function(data)



    fpr, tpr, thresholds = roc_curve(y, y_pred, pos_label=event)

    auc_value = auc(fpr, tpr)



    # plot roc curve and caluculate auc

    plt.figure(figsize=(10, 10))

    lw = 2

    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % auc_value)

    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

    plt.xlim([0.0, 1.0])

    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.title('Receiver operating characteristic example')

    plt.legend(loc="lower right")

    plt.show()

    return auc_value

# metrics for model performance examnation

auc_v = roc_curve_plot(lrcv, X_train_oh, y_train)
def ks_plot(y_true, y_prob):

    """

    plot ks-curve and calculate ks value

    :param y_true: true values of target y of samples

    :param y_prob: predicted probability of been positive of samples

    :return: ks curve plot ad ks value

    """

    df = pd.concat((pd.DataFrame(np.array(y_prob[:, 0]), columns=['y_prob']),

                    pd.DataFrame(np.array(y_true), columns=['y_true'])),

                    axis=1)

    df_sorted = df.sort_values(by='y_prob')  # sort by predicted probability of samples

    y_prob_sorted = df_sorted.iloc[:, 0]

    y_true_sorted = df_sorted.iloc[:, 1]

    total_good_count = y_true.value_counts()[0]

    total_bad_count = y_true.value_counts()[1]



    cut_points = np.linspace(y_prob.min(), y_prob.max(), 31)  # thresholds

    good_event_cdf = []

    bad_event_cdf = []



    # calculate cdf for good and bad event respectively.

    for i, tr in enumerate(cut_points):

        selected_data = y_true_sorted[y_prob_sorted <= tr]

        good_event_count = sum(selected_data == 0)  # count good_event

        bad_event_count = sum(selected_data == 1)  # count bad_event

        good_event_cdf.append(good_event_count / total_good_count)

        bad_event_cdf.append(bad_event_count / total_bad_count)



    # calculate ks value

    good_bad_diff = np.array(bad_event_cdf) - np.array(good_event_cdf)

    ks_value = max(good_bad_diff)

    ks_position = np.argmax(good_bad_diff).astype(int)



    # plot curve

    plt.figure(figsize=(10, 10))

    lw = 2

    plt.plot(cut_points, good_event_cdf, color='darkorange', lw=lw, label='good_event_cdf')

    plt.plot(cut_points, bad_event_cdf, color='green', lw=lw, label='bad_event_cdf')

    plt.plot([cut_points[ks_position], cut_points[ks_position]],

             [good_event_cdf[ks_position], bad_event_cdf[ks_position]],

             color='navy', lw=lw, linestyle='--')

    plt.xlim([0.0, 1.3])

    plt.ylim([0.0, 1.2])

    plt.xlabel('thresholds')

    plt.ylabel('cumulative rate')

    plt.title('K-S plot')

    plt.legend(loc="lower right")

    plt.show()



    return ks_value

# plot ks



ks_plot(y_test, lrcv.predict_proba(X_test_oh))
print(colored('resultes of metrics for logistic regression: ', 'yellow'))

print('accuracy_score: ')

print(accuracy_score(y_test,lrcv.predict(X_test_oh)))

print('recall_score: ')

print(recall_score(y_test,lrcv.predict(X_test_oh), pos_label=1))  #  tp / (tp + fn)

print('precision_score: ')

print(precision_score(y_test,lrcv.predict(X_test_oh), pos_label=1))  #  tp / (tp + fp)

print('f1_score: ')

print(f1_score(y_test, lrcv.predict(X_test_oh)))

print('confusion_matrix: ')

confusion_matrix(y_test,lrcv.predict(X_test_oh))

# tn, fp, fn, tp = confusion_matrix(y_test,lrcv.predict(X_test_oh)).ravel()

# print(tn, fp, fn, tp)
# from sklearn.metrics import precision_recall_curve

# from sklearn.model_selection import cross_val_predict



# y_scores = pd.DataFrame(cross_val_predict(lrcv, X_train_oh, y_train, cv=3, method='decision_function'))

# precisions, recalls, thresholds = precision_recall_curve(y_train, y_scores)
# from sklearn.metrics import precision_recall_curve

# from model.tools import *

# plot_precision_recall_vs_threshold(precisions, recalls, thresholds)

# plt.show()
# replace discretized data with woe



woes, iv = woe_c.woe(X=np.array(data_cut_X), y=y)

data_woe = pd.DataFrame(woe_c.woe_replace(X=np.array(data_cut_X), woe_arr=woes), index=data_cut_X.index, columns=data_cut_X.columns)



# split data

X_train, X_test, y_train, y_test = split_data(0.3, data_woe, data_cut_Y)
lrcv_woe = LogisticRegressionCV(Cs=10, class_weight='balanced', scoring='recall', penalty='l1', solver='liblinear')

lrcv_woe.fit(X_train, y_train)

lrcv_woe.C_  # C selected

print('scores for training data: ')

print(lrcv_woe.score(X_train, y_train))

print('scores for testing data: ')

print(lrcv_woe.score(X_test, y_test))
# plot ks



auc_v = roc_curve_plot(lrcv_woe, X_test, y_test)
ks_plot(y_test, lrcv_woe.predict_proba(X_test))
print(colored('resultes of metrics for WOE logistic regression: ', 'yellow'))

print('WOE model')

print('accuracy_score: ')

print(accuracy_score(y_test,lrcv_woe.predict(X_test)))

print('recall_score: ')

print(recall_score(y_test,lrcv_woe.predict(X_test), pos_label=1))  #  tp / (tp + fn)

print('precision_score: ')

print(precision_score(y_test,lrcv_woe.predict(X_test), pos_label=1))  #  tp / (tp + fp)

print('f1_score: ')

print(f1_score(y_test, lrcv_woe.predict(X_test)))

print('confusion_matrix: ')

confusion_matrix(y_test,lrcv_woe.predict(X_test))