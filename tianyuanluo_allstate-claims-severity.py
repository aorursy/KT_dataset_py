# import models

import os, sys, re



import numpy as np

import pandas as pd

from pandas import Series, DataFrame



import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline

sns.set()

train = pd.read_csv(os.path.join('../input','allstate-claims-severity', 'train.csv'))



print(train.shape)

train.head()
submit = pd.read_csv(os.path.join('../input','allstate-claims-severity','test.csv'))

print(submit.shape)
def selector(n_row, percent):

    sel = np.random.rand(n_row)

    return sel <= percent



train = train.loc[selector(train.shape[0], 0.5), :]

train = train.reset_index().drop('index', axis = 1)

train.shape
# id column is not useful for modeling, so drop it.

train.drop('id', axis = 1, inplace = True)

train.columns
# Group features into categorical or continuous features

feat_group = {'fea_cat': [_ for _ in train.columns if re.match(r'cat.*', _)],

             'fea_cont': [_ for _ in train.columns if re.match(r'cont.*', _)]}



print(f"There are {len(feat_group['fea_cat'])} categorical features, and {len(feat_group['fea_cont'])} continuous features.")
# Check if any cat features contain NaN's

count_nan = train[feat_group['fea_cat']].count() - train.shape[0]

count_nan.value_counts()
# Also check if there is any NaN's in the submit dataset

count_nan = submit[feat_group['fea_cat']].count() - submit.shape[0]

count_nan.value_counts()
# Check the variaty of each categorical features.

cat_var = {}

for cat in feat_group['fea_cat']:

    cat_var[cat] = train[cat].unique().size
cat_var = Series(cat_var)

cat_var.value_counts().sort_index()
# For those features containing > 10 categories, group the minorities.

cat_fea_group = {_: [] for _ in cat_var[cat_var > 10].index}

cat_fea_group
# Compose a function to display frequency distribution table and chart.

def freq_dist(target, df):

    dist_df = DataFrame(np.concatenate((df[target].value_counts().values.astype(np.float_).reshape(-1,1), 

               (df[target].value_counts().values.astype(np.float_) / df.shape[0]).reshape(-1,1),

                                         (df[target].value_counts().values.astype(np.float_) / df.shape[0]).cumsum().reshape(-1,1)),

              axis = 1), 

          columns = ['Frequency', 'Percentage','Cul_Percent'],

         index = df[target].value_counts().index).sort_values(by = 'Percentage', ascending = False)

    dist_df = pd.merge(dist_df, DataFrame(train.groupby(target).mean()['loss']), left_index = True, right_index = True, how = 'left')

    dist_df.columns= ['Frequency', 'Percentage','Cul_Percent','Loss_Mean']

    print(dist_df)

    

    fig, axis = plt.subplots(1,2, figsize = (18,6))

    

    axis[0].bar(dist_df.index, "Percentage", data = dist_df)

    axis[0].set_title(' '.join((target, "Frequency Plot")), fontsize = 15)

    axis[0].set_ylim(top = 1)

    

    axis[1].bar(dist_df.index, 'Loss_Mean', data = dist_df, color = 'brown')

    axis[1].set_title(' '.join((target, "vs. Loss (Mean)")), fontsize = 15)

    

    plt.show()

    return dist_df
cat99_distrib = freq_dist('cat99', train)
cat_fea_group['cat99'] = cat99_distrib[cat99_distrib['Cul_Percent'] > 0.85].index

cat_fea_group['cat99']
cat100_distrib = freq_dist('cat100', train)
cat_fea_group['cat100'] = cat100_distrib[cat100_distrib['Percentage'] < 0.03].index

cat_fea_group['cat100']
cat101_distrib = freq_dist('cat101', train)
cat_fea_group['cat101'] = cat101_distrib[cat101_distrib['Cul_Percent'] > 0.60].index

cat_fea_group['cat101']
cat103_distrib = freq_dist('cat103', train)
cat_fea_group['cat103'] = cat103_distrib[cat103_distrib['Percentage'] < 0.05].index

cat_fea_group['cat103']
cat104_distrib = freq_dist('cat104', train)
cat_fea_group['cat104'] = cat104_distrib[cat104_distrib['Percentage'] < 0.02].index

cat_fea_group['cat104']
cat105_distrib = freq_dist('cat105', train)
cat_fea_group['cat105'] = cat105_distrib[cat105_distrib['Percentage'] < 0.2].index

cat_fea_group['cat105']
cat106_distrib = freq_dist('cat106', train)
cat_fea_group['cat106'] = cat106_distrib[cat106_distrib['Percentage'] < 0.02].index

cat_fea_group['cat106']
cat107_distrib = freq_dist('cat107', train)
cat_fea_group['cat107'] = cat107_distrib[cat107_distrib['Percentage'] < 0.04].index

cat_fea_group['cat107']
target = 'cat108'

cat108_distrib = freq_dist(target, train)
cat_fea_group[target] = cat108_distrib[cat108_distrib['Percentage'] < 0.03].index

cat_fea_group[target]
target = 'cat109'

cat109_distrib = freq_dist(target, train)
cat_fea_group[target] = cat109_distrib[cat109_distrib['Percentage'] < 0.12].index

cat_fea_group[target]
target = 'cat110'

cat110_distrib = freq_dist(target, train)
# There is no obvious gap between frequencies, therefore no category grouping here.

cat_fea_group[target] = []

cat_fea_group[target]
target = 'cat111'

cat111_distrib = freq_dist(target, train)
cat_fea_group[target] = cat111_distrib[cat111_distrib['Percentage'] < 0.05].index

cat_fea_group[target]
target = 'cat112'

cat112_distrib = freq_dist(target, train)
# Not going to group this feature

cat_fea_group[target] = []

cat_fea_group[target]
target = 'cat113'

cat113_distrib = freq_dist(target, train)
target = 'cat114'

cat114_distrib = freq_dist(target, train)
cat_fea_group[target] = cat114_distrib[cat114_distrib['Percentage'] < 0.1].index

cat_fea_group[target]
target = 'cat115'

cat115_distrib = freq_dist(target, train)
cat_fea_group[target] = cat115_distrib[cat115_distrib['Percentage'] < 0.01].index

cat_fea_group[target]
target = 'cat116'

cat116_distrib = freq_dist(target, train)
# No obvious frequency gap between categories. No grouping conducted.

cat_fea_group[target] = []
cat_fea_group
# Delete the keys with no values assigned.

try:

    del cat_fea_group['cat110']

    del cat_fea_group['cat112']

    del cat_fea_group['cat113']

    del cat_fea_group['cat116']

except:

    pass
def fea_group(df, fea_dist):

    for feature in fea_dist:

        df.loc[df[feature].isin(fea_dist[feature].values.tolist()), feature] = 'Others'

        

def new_fea_group(df1, df2):

    for col in feat_group['fea_cat']:

        df2.loc[~df2[col].isin(df1[col].value_counts().index.tolist()), col] = df1[col].value_counts().index[0]

        
# Group minor categories

fea_group(train, cat_fea_group)
from sklearn.preprocessing import OneHotEncoder



ohe = OneHotEncoder(drop = 'first')
# One Hot Encoding all categorical features

train_ohe = ohe.fit_transform(train[feat_group['fea_cat']])

ohe_columns = ohe.get_feature_names(feat_group['fea_cat'])
feat_group['fea_cont']
train[feat_group['fea_cont']].describe()
# Display value distribution for each feature



fig, ax = plt.subplots(len(feat_group['fea_cont']) // 2, 2, figsize = (16, 5 * (len(feat_group['fea_cont']) // 2)))

for idx, item in enumerate(feat_group['fea_cont']):

    ax[idx//2, idx % 2].violinplot(train[item], showmedians = True)

    ax[idx//2, idx % 2].set_title(item, fontsize = 20)

    
# draw a heatmap to display the correlations between continuous features.

plt.figure(figsize = (10,10))

plt.title("Correlation Heatmap of Continuous Features", fontsize = 20)

sns.heatmap(train[feat_group['fea_cont']].corr(), cmap = 'RdBu_r')
# draw a heatmap to display the correlations between continuous features.

plt.figure(figsize = (10,10))

plt.title("Correlation Heatmap of Continuous Features", fontsize = 20)

sns.heatmap(train[['cont2','cont3','cont4','cont5','cont8','cont12','cont13','cont14']].corr(), cmap = 'RdBu_r')
feat_group['fea_cont'] =  ['cont2','cont3','cont4','cont5','cont8','cont12','cont13','cont14']
from sklearn.preprocessing import StandardScaler



sdc = StandardScaler()
def standard_scaler(df, feat_dist):

    df.loc[:, feat_dist['fea_cont']] = sdc.fit_transform(df[feat_dist['fea_cont']])

    
standard_scaler(train, feat_group)
fig, ax = plt.subplots(len(feat_group['fea_cont']) // 2, 2, figsize = (16, 5 * (len(feat_group['fea_cont']) // 2)))

for idx, item in enumerate(feat_group['fea_cont']):

    ax[idx//2, idx % 2].violinplot(train[item], showmedians = True)

    ax[idx//2, idx % 2].set_title(item, fontsize = 20)
train['loss'].describe()
print(train['loss'].describe())

plt.violinplot(train['loss'], showmedians = True)
transformed_loss = np.log1p(train['loss'].values)



from sklearn.preprocessing import StandardScaler



sdc_loss = StandardScaler()



transformed_loss = sdc_loss.fit_transform(transformed_loss.reshape(-1,1))



plt.violinplot(transformed_loss, showmedians = True)
import scipy.sparse as ssp

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(ssp.hstack((train[feat_group['fea_cont']],

                                                               train_ohe)),

                                                    transformed_loss, test_size = 0.3)

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(x_train, y_train)
print('R2 Score for Training Dataset: ',lr.score(x_train,y_train).round(2), '\n',

      "R2 Score for Testing Dataset: ", lr.score(x_test,y_test).round(2))
print("MAE for Training Dataset: ", mean_absolute_error(np.exp(sdc_loss.inverse_transform(y_train)), 

                                                        np.exp(sdc_loss.inverse_transform(lr.predict(x_train.toarray())))),

      '\n',

     "MAE for Testing Dataset: ", mean_absolute_error(np.exp(sdc_loss.inverse_transform(y_test)),

                                                      np.exp(sdc_loss.inverse_transform(lr.predict(x_test.toarray())))))
from sklearn.linear_model import Ridge



lr_ridge = Ridge(alpha = 5000)
lr_ridge.fit(x_train, y_train)



print('R2 Score for Training Dataset: ',lr_ridge.score(x_train,y_train).round(2), '\n',

      "R2 Score for Testing Dataset: ", lr_ridge.score(x_test,y_test).round(2))



print("MAE for Training Dataset: ", mean_absolute_error(np.exp(sdc_loss.inverse_transform(y_train)), 

                                                        np.exp(sdc_loss.inverse_transform(lr_ridge.predict(x_train.toarray())))),

      '\n',

     "MAE for Testing Dataset: ", mean_absolute_error(np.exp(sdc_loss.inverse_transform(y_test)),

                                                      np.exp(sdc_loss.inverse_transform(lr_ridge.predict(x_test.toarray())))))
from sklearn.linear_model import Lasso



lr_lasso = Lasso(alpha = 2**-12)
lr_lasso.fit(x_train, y_train)



print('R2 Score for Training Dataset: ',lr_lasso.score(x_train,y_train).round(2), '\n',

      "R2 Score for Testing Dataset: ", lr_lasso.score(x_test,y_test).round(2))



print("MAE for Training Dataset: ", mean_absolute_error(np.exp(sdc_loss.inverse_transform(y_train)), 

                                                        np.exp(sdc_loss.inverse_transform(lr_lasso.predict(x_train.toarray())))),

      '\n',

     "MAE for Testing Dataset: ", mean_absolute_error(np.exp(sdc_loss.inverse_transform(y_test)),

                                                      np.exp(sdc_loss.inverse_transform(lr_lasso.predict(x_test.toarray())))))
from sklearn.model_selection import GridSearchCV

from sklearn.experimental import enable_hist_gradient_boosting

from sklearn.ensemble import HistGradientBoostingRegressor

gbr = HistGradientBoostingRegressor()
parameters = {'learning_rate': [0.1, 0.05],

              'scoring' : ['mae'],

              'max_iter': [100],

              'max_depth': [5]

          ##    'l2_regularization': [2**-1, 2**-5]

             }



gbr_cv = GridSearchCV(gbr, parameters, cv = 3, scoring = 'neg_mean_absolute_error')
gbr_cv.fit(x_train.toarray(), y_train.ravel())

gbr_cv.best_estimator_
print("MAE for Training Dataset: ", mean_absolute_error(np.exp(sdc_loss.inverse_transform(y_train)), 

                                                        np.exp(sdc_loss.inverse_transform(gbr_cv.predict(x_train.toarray())))),

      '\n',

     "MAE for Testing Dataset: ", mean_absolute_error(np.exp(sdc_loss.inverse_transform(y_test)),

                                                      np.exp(sdc_loss.inverse_transform(gbr_cv.predict(x_test.toarray())))))
# 1. Process the categorical features



fea_group(submit, cat_fea_group)

new_fea_group(train, submit)



submit_ohe = ohe.transform(submit.loc[:, feat_group['fea_cat']])

submit_ohe_columns = ohe.get_feature_names(feat_group['fea_cat'])



# 2. Process the continuous features



feat_group['fea_cont'] =  ['cont2','cont3','cont4','cont5','cont8','cont12','cont13','cont14']

standard_scaler(submit, feat_group)



# 3. Form the test dataset

submit2 = ssp.hstack((submit[feat_group['fea_cont']], submit_ohe))
submit_predict = np.exp(sdc_loss.inverse_transform(gbr_cv.predict(submit2.toarray())))
submission = pd.DataFrame({'id': submit['id'].values, 'loss': submit_predict})
submission.to_csv('submission.csv', sep = ',', header = True, index = False)