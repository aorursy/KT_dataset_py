# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.tree import DecisionTreeClassifier



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#loading the csv

data_df = pd.read_csv('../input/HR_comma_sep.csv',index_col=False)

data_df.head()
# A utility function to clean and modify data easily



def prepareData(df,clean=True,ignore=[],quantise=True,threshold_for_quantisation = 10,categories_to_quantise = 5,threshold_for_quantisation_string=5):

    '''

    prepareData will quantise any string based features. It also convert linear datas into bins

    ex:-

    input

    age:1,2,3,4,5,6

    sex:male,female,male,female,female,female

    result

    age:0,0,0,1,1,1

    sex:1,0,1,0,0,0

    '''

    df = df.copy()

    headers = df.dtypes.index

    headers = list(set(headers) - set(ignore))

    for i in headers:

        if df[i].dtypes == 'int64' or df[i].dtypes == 'float64':

            if len(df[i].value_counts()) > threshold_for_quantisation and quantise:

                maxElem = int(df[i].max())

                if maxElem == 0:

                    maxElem = 1

                minElem = int(df[i].min())

                d = (maxElem - minElem)/categories_to_quantise

                bins = [minElem + i*d for i in range(categories_to_quantise + 1)]

                group_names = [i for i in range(categories_to_quantise)]

                df[i] = pd.cut(df[i],bins,labels=group_names)

        else:

            if len(df[i].value_counts()) > threshold_for_quantisation_string:

                print("manually convert this parameter:",i)

            else:

                j = 0

                for k in df[i].value_counts().index:

                    j = j + 1

                    df[i].replace(k, j,inplace=True)

    if clean:

        df = df.dropna()

    return df
#checking how data is like where mean and std lies

data_df.describe()
#checking individual parameters for patterns

datasize = data_df['satisfaction_level'].count()

data_df['satisfaction_level'].plot.kde()

#most employs rate there satisfaction level above 0.5 maxing at 0.75. which is expected as most of us alwasys rates our employee as 0.75
#just keeping the analysis going no good feature obserrved in this case

data_df['average_montly_hours'].plot.kde()
#Our goal is to get find how left is correalted with individual aspects of employ for that we need to prepare data

print(data_df.dtypes)

new_data_df = prepareData(data_df,quantise=False,threshold_for_quantisation_string=10)

print(new_data_df.dtypes)

new_data_df.head()
#corelations between data

threshold = 0.3

x_cols = [col for col in new_data_df.columns if col not in ['logerror'] if new_data_df[col].dtype=='float64' or new_data_df[col].dtype=='int64']

print(x_cols)

labels = []

values = []

for col1 in x_cols:

    for col2 in x_cols:

        labels.append(col1+'_'+col2)

        values.append(np.corrcoef(new_data_df[col1].values, new_data_df[col2].values)[0,1])

for i in range(len(labels)):

    if values[i] > threshold or values[i] < -1*threshold:

        if values[i] != 1.0:

            print(labels[i]+':'+str(values[i]))
#drawing vorelation in a more fancy way

new_data_df_2 = new_data_df[x_cols]

corrmat = new_data_df.corr(method='spearman')

f, ax = plt.subplots(figsize=(8, 8))



# Draw the heatmap using seaborn

sns.heatmap(corrmat, vmax=1., square=True,annot=True)

plt.title("Important variables correlation map", fontsize=15)

plt.show()
#as we can see from from above analysis it is clear that relavent correlations are 

#satisfaction_level_left:-0.388374983424

#last_evaluation_number_project:0.349332588516

#last_evaluation_average_montly_hours:0.339741799838

#number_project_average_montly_hours:0.417210634402

def make_pivot (param1, param2):

    df = data_df

    df_slice = df[[param1, param2]]

    slice_pivot = df_slice.pivot_table(index=[param1], columns=[param2],aggfunc=np.size, fill_value=0)

    p_chart = slice_pivot.div(slice_pivot.sum(axis=1), axis=0).plot.bar(stacked=True)

    return slice_pivot

    return p_chart
make_pivot('salary','left')
# so low salaried employs are most likely to leave that is within what we should get

data_df['satisfaction_level'] = pd.cut(data_df['satisfaction_level'],[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
make_pivot('satisfaction_level','left')

#we see a strange behaviour here people with 0.1 to 0.2 satisfaction level are much less likely to leave
make_pivot('sales','left')

#i dont see any direct correlation here
make_pivot('promotion_last_5years','left')

# as we can see people with more promotions are less likely to leave
data_df['last_evaluation'] = pd.cut(data_df['last_evaluation'],[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])

make_pivot('last_evaluation','left')

# In this case I thought people with high evaluation will be more likely to leave and least evaluation will stay but in this case

# people with 0.4-0.5 evaluatin are most likely to leave
# lets try making some predictions on basis of given data first we will re-intialise the data I did many changes to

# original data so I want to first remove all those and then start

data_df = pd.read_csv('../input/HR_comma_sep.csv',index_col=False)

#spliting data i think 1000 data will be a good enough starting point

split = 5000

train_df = data_df[:-split]

test_df = data_df[-split:]

#reconfirming that our test and train data are in good shape

print('test:',test_df.shape)

print('train:',train_df.shape)
# all seems good now we can take two path we can either feed everthing to model and model chose which features it thinks is important or we can feed in only those features that we think are correlated

# let model do the heavy lifting

# but first lets do some preprocessing

new_train_df = prepareData(train_df,threshold_for_quantisation_string=10,categories_to_quantise=5)

new_test_df = prepareData(test_df,threshold_for_quantisation_string=10,categories_to_quantise=5)

print(new_train_df.head())

print(new_test_df.head())
x_train = new_train_df.copy()

del x_train['left']

y_train = new_train_df['left']

dtree = DecisionTreeClassifier()

dtree.fit(x_train,y_train)
'''

import graphviz 

from sklearn import tree

dot_data = tree.export_graphviz(dtree, out_file=None)

graph = graphviz.Source(dot_data)

graph

'''

#was just looking at the tree its complicated you can uncomment this part to get insight into the classification model
x_test = new_test_df.copy()

del x_test['left']

y_test = new_test_df['left']

dtree.score(x_test,y_test)

#wow thats too much how can that be. But facts seems to checkout
### XGBModel

# incooperating XGBModel

import xgboost as xgb

from sklearn.metrics import roc_auc_score, r2_score, accuracy_score, classification_report 

#I am trying XGB for first time so I am trying to get insight into it there are many things I will printout to see how all steps work

#yohan's notebook for references the accuracy is not as much as yohan's because I used bin's for my data.

xgb_params = {

    'n_trees': 100, 

    'eta': 0.1,

    'max_depth': 7,

    'subsample': 0.75,

    'colsample_bytree': 0.75,

    'objective': 'binary:logistic',

    'scale_pos_weight': float(len(y_train)-sum(y_train)) / sum(y_train),

    'eval_metric': 'auc',

    'silent': 1

}



dtrain_xgb = xgb.DMatrix(x_train.values, y_train.values)

dtest_xgb = xgb.DMatrix(x_test.values, y_test.values)



cv_result_xgb = xgb.cv(xgb_params, 

                   dtrain_xgb, 

                   num_boost_round=5000,

                   nfold = 5,

                   stratified=True,

                   early_stopping_rounds=50,

                   verbose_eval=100, 

                   show_stdv=True

                  )

num_boost_rounds_xgb = len(cv_result_xgb)

print('num_boost_rounds=' + str(num_boost_rounds_xgb))

# train model

model_xgb = xgb.train(dict(xgb_params, silent=0), 

                      dtrain_xgb, 

                      num_boost_round=num_boost_rounds_xgb)



### Visualizations about the training process:

plt.figure(figsize=(15,5))

# Features importance

plt.subplot(1,2,1)

features_score_xgb = pd.Series(model_xgb.get_fscore()).sort_values(ascending=False)

sns.barplot(x=features_score_xgb.values, 

            y=features_score_xgb.index.values, 

            orient='h', color='b')

# CV scores

plt.subplot(1,2,2)

train_scores = cv_result_xgb['train-auc-mean']

train_stds = cv_result_xgb['train-auc-std']

plt.plot(train_scores, color='blue')

plt.fill_between(range(len(cv_result_xgb)), 

                 train_scores - train_stds, 

                 train_scores + train_stds, 

                 alpha=0.1, color='blue')

test_scores = cv_result_xgb['test-auc-mean']

test_stds = cv_result_xgb['test-auc-std']

plt.plot(test_scores, color='red')

plt.fill_between(range(len(cv_result_xgb)), 

                 test_scores - test_stds, 

                 test_scores + test_stds, 

                 alpha=0.1, color='red')

plt.title('Train and test cv scores (AUC)')

plt.ylim(0.96,1)

plt.show()



### Evaluation

threshold = 0.5

y_pred_xgb = model_xgb.predict(dtest_xgb)

y_cl_xgb = [1 if x > threshold else 0 for x in y_pred_xgb]

print('Threshold:', threshold)

print('Accuracy:  {:.2f} %'.format(accuracy_score(y_test, y_cl_xgb)*100))

print('R2:        {:.4f}'.format(r2_score(y_test, y_cl_xgb)))

print('AUC:       {:.4f}'.format(roc_auc_score(y_test, y_cl_xgb)))

mis = sum(np.abs(y_test - np.array(y_cl_xgb)))

print('Misclass.: {} (~{:.2f} %) out of {}'.format(mis, 

                                                   float(mis)/len(y_test)*100, 

                                                   len(y_test)))

print(classification_report(y_test, 

                            y_cl_xgb, 

                            labels=[0,1], 

                            target_names=['stay', 'left'], 

                            digits=4))