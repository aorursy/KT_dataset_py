from math import sqrt

import itertools

import pandas as pd

import numpy as np

from scipy import stats

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV, cross_val_predict

from sklearn.metrics import r2_score, mean_squared_error

from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeRegressor, plot_tree

from sklearn.ensemble import RandomForestRegressor

from sklearn.base import BaseEstimator

import xgboost
df = pd.read_csv('/kaggle/input/fivethirtyeight-candy-power-ranking-dataset/candy-data.csv')

df['competitorname'] = df.competitorname.apply(lambda l: l.replace('Õ', '\''))
df[df.columns[1:-3]].agg(['sum','count'])
df[['sugarpercent', 'pricepercent']].describe().T
df.sort_values('winpercent', ascending=False)[:10]
df.sort_values('winpercent')[:10]
plt.figure(figsize=(15,10))

sns.heatmap(df.corr(), linewidth=0.5, center=0, cmap="YlGn", annot=True)
corr_df = pd.DataFrame([(f1, f2, df.corr()[f1].loc[f2]) for (f1, f2) in itertools.combinations(df.columns[1:].sort_values(), 2)],

             columns=['feature1', 'feature2', 'corr']).sort_values('corr')



corr_df = corr_df.iloc[(-corr_df['corr'].abs()).argsort()].reset_index(drop=True)

corr_df[:10]
corr_df[(corr_df.feature1 == 'winpercent') | (corr_df.feature2 == 'winpercent')]
df[['competitorname', 'chocolate', 'winpercent']].sort_values('winpercent', ascending=False)[:20]
fig = plt.figure()

sns.boxplot(data=df, x="chocolate", y="winpercent", palette="YlGn")
print('p-value = {0:.10%}'.format(stats.ttest_ind(df[df.chocolate == 0].winpercent, 

                                               df[df.chocolate == 1].winpercent)[1]))
df[(df.bar == 1) & (df.chocolate == 0)]
fig = plt.figure(figsize=(15,5))



ax = fig.add_subplot(1,2,1)

ax.set(title='Overall')

sns.heatmap(df[['bar', 'winpercent']].corr(), linewidth=0.5, center=0, cmap="YlGn", annot=True)



ax = fig.add_subplot(1,2,2)

ax.set(title='Chocolate only')

sns.heatmap(df[df.chocolate == 1][['bar', 'winpercent']].corr(), linewidth=0.5, center=0, cmap="YlGn", annot=True)
fig = plt.figure()

plt.title('Chocolate only')

sns.boxplot(data=df[df.chocolate == 1], x="bar", y="winpercent", palette="YlGn")
print('p-value = {0:.2%}'.format(stats.ttest_ind(df[(df.bar == 0) & (df.chocolate == 1)].winpercent, 

                                                  df[(df.bar == 1) & (df.chocolate == 1)].winpercent)[1]))
df[['chocolate', 'peanutyalmondy', 'competitorname']].groupby(['chocolate', 'peanutyalmondy'], as_index=False).count()
plt.figure()

sns.boxplot(data=df[df.chocolate == 1], x="peanutyalmondy", y="winpercent", palette="YlGn")
print('p-value = {0:.2%}'.format(stats.ttest_ind(df[(df.peanutyalmondy == 0) & (df.chocolate == 1)].winpercent, 

                                                  df[(df.peanutyalmondy == 1) & (df.chocolate == 1)].winpercent)[1]))
df[['chocolate', 'fruity', 'competitorname']].groupby(['chocolate', 'fruity'], as_index=False).count()
plt.figure()

sns.boxplot(data=df, x="fruity", y="winpercent", palette="YlGn")
print('p-value = {0:.2%}'.format(stats.ttest_ind(df[df.fruity == 0].winpercent, 

                                                 df[df.fruity == 1].winpercent)[1]))
plt.figure()

sns.boxplot(data=df, x="hard", y="winpercent", palette="YlGn")
print('p-value = {0:.2%}'.format(stats.ttest_ind(df[df.hard == 0].winpercent, 

                                                 df[df.hard == 1].winpercent)[1]))
df[['chocolate', 'hard', 'competitorname']].groupby(['chocolate', 'hard'], as_index=False).count()
fig = plt.figure()

sns.boxplot(data=df[df.chocolate == 0], x="hard", y="winpercent", palette="YlGn")
print('p-value = {0:.2%}'.format(stats.ttest_ind(df[(df.hard == 0) & (df.chocolate == 0)].winpercent, 

                                                  df[(df.hard == 1) & (df.chocolate == 0)].winpercent)[1]))
fig = plt.figure()

sns.boxplot(data=df, x="pluribus", y="winpercent", palette="YlGn")
print('p-value = {0:.2%}'.format(stats.ttest_ind(df[df.pluribus == 0].winpercent, 

                                                 df[df.pluribus == 1].winpercent)[1]))
fig = plt.figure()

sns.boxplot(data=df, x="caramel", y="winpercent", palette="YlGn")
print('p-value = {0:.2%}'.format(stats.ttest_ind(df[df.caramel == 0].winpercent, 

                                                 df[df.caramel == 1].winpercent)[1]))
def bars_winpercent(df, x, title, cnt_bars):

    # plot one continuous variable against winpercent

    

    factor = cnt_bars - 1

    x = '%s_rounded' % x

    price = pd.DataFrame({x: np.round(df.pricepercent*factor)/factor, 'winpercent': df.winpercent})



    fig = plt.figure(figsize=(15,5))

    fig.suptitle(title)



    ax = fig.add_subplot(1,2,1)

    sns.barplot(x=x, 

                y='winpercent',

                data=price.groupby(x, as_index=False).mean().sort_values(x),

                palette='YlGn', ax=ax)

    ax.set(ylabel='Mean winpercent')



    ax = fig.add_subplot(1,2,2)

    ax.set(ylabel='Count')

    sns.barplot(x=x, 

                y='winpercent',

                data=price.groupby(x, as_index=False).count().sort_values(x),

                palette='YlGn', ax=ax)

    ax.set(ylabel='Count')
bars_winpercent(df[df.chocolate == 1], 'pricepercent', 'Chocolate only', 5)

bars_winpercent(df[df.fruity == 1], 'pricepercent', 'Fruity only', 5)
bars_winpercent(df[df.chocolate == 1], 'sugarpercent', 'Chocolate only', 5)

bars_winpercent(df[df.fruity == 1], 'sugarpercent', 'Fruity only', 5)
t1 = pd.pivot_table(df[['chocolate', 'peanutyalmondy', 'caramel', 'winpercent']],

               values=['winpercent'],

               columns=['chocolate', 'peanutyalmondy', 'caramel'],

               aggfunc=['mean', 'count'],

               fill_value=0

              ).sort_values('mean', ascending=False).reset_index().drop('level_0', axis=1)
t2 = pd.pivot_table(df[['chocolate', 'peanutyalmondy', 'winpercent']],

               values=['winpercent'], 

               columns=['chocolate', 'peanutyalmondy'],

               aggfunc=['mean', 'count'],

               fill_value=0

              ).sort_values('mean', ascending=False).reset_index().drop('level_0', axis=1)
t3 = pd.pivot_table(df[['chocolate', 'caramel', 'winpercent']],

               values=['winpercent'],

               columns=['chocolate', 'caramel'],

               aggfunc=['mean', 'count'],

               fill_value=0

              ).sort_values('mean', ascending=False).reset_index().drop('level_0', axis=1)
t1.append(t2, sort=False).append(t3, sort=False).sort_values('mean', ascending=False).reset_index(drop=True)
df_p1_c0 = df[(df.chocolate == 1) & (df.peanutyalmondy == 1) & (df.caramel == 0)]

df_p1_c0_minus = df[[idx not in df_p1_c0.index for idx in df.index]]



print('p-value = {0:.5%}'.format(stats.ttest_ind(df_p1_c0_minus.winpercent, 

                                                 df_p1_c0.winpercent)[1]))
df_p0_c1 = df[(df.chocolate == 1) & (df.peanutyalmondy == 0) & (df.caramel == 1)]

df_p0_c1_minus = df[[idx not in df_p0_c1.index for idx in df.index]]



print('p-value = {0:.2%}'.format(stats.ttest_ind(df_p0_c1_minus.winpercent, 

                                                 df_p0_c1.winpercent)[1]))
print('p-value = {0:.2%}'.format(stats.ttest_ind(df_p1_c0.winpercent, 

                                                 df_p0_c1.winpercent)[1]))
df_p1_c1 = df[(df.chocolate == 1) & (df.peanutyalmondy == 1) & (df.caramel == 1)]

df_p1_c1_minus = df[[idx not in df_p1_c1.index for idx in df.index]]



print('p-value = {0:.2%}'.format(stats.ttest_ind(df_p1_c1_minus.winpercent, 

                                                 df_p1_c1.winpercent)[1]))
features = df.columns[1:-1]

target = 'winpercent'



class MeanEstimator(BaseEstimator):

    

    mean = None

    

    def fit(self, X, y):

        if isinstance(y, pd.Series) or isinstance(y, pd.DataFrame):

            self.mean = np.mean(y).iloc[0]

        else:

            self.mean = np.mean(y)

        

    def predict(self, X):

        if self.mean is None:

            raise ValueError('Estimator is not fitted yet')

        return np.ones(X.shape[0])*self.mean

model_mean = MeanEstimator()

pred_mean = cross_val_predict(model_mean, df[features], df[[target]], cv=4, n_jobs=4)

    

dict(r2=r2_score(df[[target]], pred_mean), 

     rmse=sqrt(mean_squared_error(df[[target]], pred_mean))

    )
r2_score(df[[target]], np.ones(df.shape[0])*np.mean(df[[target]]).iloc[0])
param_grid = dict(

    max_depth=[3, 4, 5], 

    learning_rate=[0.05, 0.1, 0.2], 

    n_estimators=[32, 33, 34],

    min_child_weight=[5, 6, 7],

    subsample=[0.4, 0.5, 0.6],

)



clf = xgboost.XGBRegressor(objective='reg:squarederror')

model_xgb = GridSearchCV(estimator=clf, param_grid=param_grid, n_jobs=4, iid=True, refit=True, cv=4, scoring='r2')



model_xgb.fit(df[features], df[[target]])



pred_xgb = cross_val_predict(model_xgb.best_estimator_, df[features], df[[target]], cv=4, n_jobs=4)



dict(r2=r2_score(df[[target]], pred_xgb), rmse=sqrt(mean_squared_error(df[[target]], pred_xgb)))
model_xgb.best_params_
imp = pd.DataFrame({'features': features, 'importance': model_xgb.best_estimator_.feature_importances_})

imp.sort_values('importance', ascending=False, inplace=True)

sns.barplot(x='importance', y='features', data=imp, palette="YlGn_r")
features = df.columns[1:-1]

target = 'winpercent'



param_grid = {

    'max_depth': [None, 10, 15, 20],

    'max_features': ['auto', 'sqrt', 'log2'],

    'min_samples_leaf': [6, 7, 8, 9],

    'min_samples_split': [10, 11, 12, 13],

    'n_estimators': [130, 140, 150, 160]

}



clf_rf = RandomForestRegressor()

model_rf = GridSearchCV(estimator=clf_rf, param_grid=param_grid, n_jobs=4, iid=True, refit=True, cv=4, scoring='r2')



model_rf.fit(df[features], np.ravel(df[[target]]))



pred_rf = cross_val_predict(model_rf.best_estimator_, df[features], df[[target]], cv=4, n_jobs=4)



dict(r2=r2_score(df[[target]], pred_rf), rmse=sqrt(mean_squared_error(df[[target]], pred_rf)))
model_rf.best_params_
imp = pd.DataFrame({'features': features, 'importance': model_rf.best_estimator_.feature_importances_})

imp.sort_values('importance', ascending=False, inplace=True)

sns.barplot(x='importance', y='features', data=imp, palette="YlGn_r")
df_rf = df.copy()

df_rf['pred_rf'] = pred_rf



sns_dt = df_rf.sort_values('pred_rf', ascending=False).reset_index(drop=True).winpercent.expanding().mean()



ax = sns.lineplot(x=sns_dt.index, y=sns_dt, color='green', palette="YlGn")

ax.set(xlabel='Candies (ordered by predicted winpercent)', ylabel='Winpercent (cumulative mean)')

plt.show()
features_no_choc = df.columns[2:-1]



param_grid = dict(

    max_depth=[3, 4, 5], 

    learning_rate=[0.05, 0.1, 0.2], 

    n_estimators=[32, 33, 34],

    min_child_weight=[5, 6, 7],

    subsample=[0.4, 0.5, 0.6],

)



clf = xgboost.XGBRegressor(objective='reg:squarederror')

model_xgb_no_choc = GridSearchCV(estimator=clf, param_grid=param_grid, n_jobs=4, iid=True, refit=True, cv=4, scoring='r2')



model_xgb_no_choc.fit(df[features_no_choc], df[[target]])



pred_xgb_no_choc = cross_val_predict(model_xgb_no_choc.best_estimator_, df[features_no_choc], df[[target]], cv=4, n_jobs=4)



dict(r2=r2_score(df[[target]], pred_xgb_no_choc), 

     rmse=sqrt(mean_squared_error(df[[target]], pred_xgb_no_choc)))
imp = pd.DataFrame({'features': features_no_choc, 'importance': model_xgb_no_choc.best_estimator_.feature_importances_})

imp.sort_values('importance', ascending=False, inplace=True)

sns.barplot(x='importance', y='features', data=imp, palette="YlGn_r")