import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
import seaborn as sns; #sns.set()
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, cross_validate, KFold
%matplotlib inline
import warnings
pd.options.mode.chained_assignment = None  # default='warn'
# warnings.filterwarnings('ignore') # disable all warnings
warnings.filterwarnings(action='once') # warn only once
data = pd.read_csv('../input/who_suicide_statistics.csv')
data.head()
data.describe()
data.astype('object').describe()
na_stats = pd.DataFrame([],columns=['count_na','prop_na'])
na_stats['count_na'] = (data.isna().sum())
na_stats['prop_na'] = (data.isna().sum()/data.shape[0])
na_stats
nan_suicides = data.suicides_no.isnull().groupby([data['country']]).sum().astype(int).reset_index(name='count')
nan_population = data.population.isnull().groupby([data['country']]).sum().astype(int).reset_index(name='count')
count_by_country = pd.DataFrame(data.groupby(data['country'])['suicides_no'].count())
count_by_country = count_by_country.reset_index()

prop = pd.DataFrame([], columns = ['country', 'prop_suicides_nan', 'prop_population_nan'])
prop['prop_suicides_nan'] = nan_suicides['count']/count_by_country['suicides_no']
prop['prop_population_nan'] = nan_population['count']/count_by_country['suicides_no']
prop['country'] = nan_suicides['country']

# Only show countries that have some missing data.
prop[prop['prop_suicides_nan'] > 0].sort_values(by=['prop_suicides_nan'], ascending=False)
data_clean = data.dropna()
data_clean.head()
target_country = "United States"
fig, ax = plt.subplots(figsize=(12,6))
ax.set_title( 'Suicides by age ({})'.format(target_country))
p = sns.scatterplot(x="year", y="suicides_no", data=data_clean,hue='age',style='sex')
target_country = "United States"
fig, ax = plt.subplots(figsize=(12,6))
ax.set_title( 'Suicides by sex ({})'.format(target_country))
sui_by_sex = data_clean[data_clean["country"].str.contains(target_country)].groupby(['sex','year'],as_index=False).sum()
p = sns.scatterplot(x="year", y="suicides_no", data=sui_by_sex,hue='sex')
agemap = {}
i = 0
for x in data.age.unique():
    agemap[x] = i
    i+=1

# since there are only two values here, we can do a mapping for gender. If >2 types listed, we could do a one-hot encoding instead.
gendermap = {}
i = 0
for x in data.sex.unique():
    gendermap[x] = i
    i+=1
    
data_clean['age_id'] = data['age'].map(agemap)
data_clean['sex_id'] = data['sex'].map(gendermap)

x = data_clean.drop(['sex','age'], axis = 1)
x = pd.get_dummies(x)
y = x[['suicides_no']]
x = x.drop('suicides_no',axis=1)
corr = data_clean.corr()
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, vmax=.3, cmap=cmap, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})
# Separate the suicides and population values by gender
data_by_gender = data_clean.loc[data_clean['sex_id'] == 1]
data_by_females = data_clean.loc[data_clean['sex_id'] == 0]
#data_by_gender['f_suicides_no'] = data_by_gender.merge(data_by_females, on=['country', 'year', 'age'])[['suicides_no_y']]
data_by_gender.rename(columns={'suicides_no':'m_suicides_no'}, inplace=True)
data_by_gender.rename(columns={'population':'m_population'}, inplace=True)

# Wrangle the data into the right format without redundant columns
data_by_gender = data_by_gender.merge(data_by_females, on=['country', 'year', 'age'])
data_by_gender = data_by_gender.drop(labels=['age','age_id_y','sex_x','sex_y','sex_id_y','sex_id_x'],axis=1)
data_by_gender.rename(columns={'population':'f_population', 'age_id_x': 'age_id'}, inplace=True)
data_by_gender.rename(columns={'suicides_no':'f_suicides_no'}, inplace=True)
data_by_gender.head()
x_s = pd.get_dummies(data_by_gender)
y_s = x_s[['f_suicides_no','m_suicides_no']]
x_s = x_s.drop(['f_suicides_no','m_suicides_no'],axis=1)

# Just doing a sanity check to make sure the data looks the way we want it.
x_s.head()
corr_by_gender = data_by_gender.corr()
sns.heatmap(corr_by_gender, vmax=.3, cmap=cmap, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})
# split into train/test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=314)

# and again by gender
x_s_train, x_s_test, y_s_train, y_s_test = train_test_split(x_s, y_s, test_size=0.25, random_state=314)
reg = LinearRegression().fit(x_train, y_train)
y_hat = reg.predict(x_test)
y_hat = pd.DataFrame(y_hat,columns=['suicides_no'])

#compute metrics

mse = (metrics.mean_squared_error(y_pred=y_hat, y_true=y_test ))
r2 = metrics.r2_score(y_pred=y_hat, y_true=y_test)
fig, ax = plt.subplots(figsize=(12,6))
sns.scatterplot(x=y_test['suicides_no'],y=y_hat['suicides_no'])
ax.set(xlabel = 'Actual y', ylabel="Predicted y")
plt.show()
print('The r-squared value is: {}, and MSE is: {}'.format(r2, mse))
kf = KFold(5,shuffle=True)
reg_cv = LinearRegression()
las_cv = Lasso()
kn_cv = KNeighborsRegressor(n_neighbors=30)
reg_cv_model = cross_validate(reg_cv, X=x, y=y, cv=kf, scoring=('neg_mean_squared_error', 'r2'), return_train_score=False)
las_cv_model = cross_validate(las_cv, X=x, y=y, cv=kf, scoring=('neg_mean_squared_error', 'r2'), return_train_score=False)
kn_cv_model = cross_validate(kn_cv, X=x, y=y, cv=kf, scoring=('neg_mean_squared_error', 'r2'), return_train_score=False)

reg_cv_scores = cross_val_score(reg_cv, X=x, y=y, cv=kf)
las_cv_scores = cross_val_score(las_cv, X=x, y=y, cv=kf)
kn_cv_scores = cross_val_score(kn_cv, X=x, y=y, cv=kf)
print('Linear Regression: The average r-squared value from {}-fold cross-validation is: {}'.format(kf.n_splits, reg_cv_scores.mean()), ('\nLasso: The average r-squared value from {}-fold cross-validation is: {}'.format(kf.n_splits, las_cv_scores.mean())), ('\nk-NN Regressor: The average r-squared value from {}-fold cross-validation is: {}'.format(kf.n_splits, kn_cv_scores.mean())))

x_usa = data_clean.drop(['sex','age'], axis = 1)
x_usa = x_usa[x_usa['country'].str.contains("United States")]
x_usa = x_usa.drop('country',axis=1)
x_usa = pd.get_dummies(x_usa)
y_usa = x_usa[['suicides_no']]
x_usa = x_usa.drop('suicides_no',axis=1);
x_usa_train, x_usa_test, y_usa_train, y_usa_test = train_test_split(x_usa, y_usa, test_size=0.25, random_state=314)
reg_usa = LinearRegression().fit(x_usa_train, y_usa_train)
y_usa_hat = reg_usa.predict(x_usa_test)
y_usa_hat = pd.DataFrame(y_usa_hat,columns=['suicides_no'])

#compute metrics
mse_usa = (metrics.mean_squared_error(y_pred=y_usa_hat, y_true=y_usa_test ))
r2_usa = metrics.r2_score(y_pred=y_usa_hat, y_true=y_usa_test)

#fig_usa, ax_usa = plt.subplots(figsize=(12,6))

fig, ax = plt.subplots()
ax.scatter(x=y_usa_test['suicides_no'],y=y_usa_hat['suicides_no'])
plt.xlabel('Actual y')
plt.ylabel('Predicted y')
plt.show()
#ax_usa.set(xlabel = 'Actual y', ylabel="Predicted y")

print('The r-squared value is: {}, and MSE is: {}'.format(r2_usa, mse_usa))
kf = KFold(5,shuffle=True)
reg_cv = LinearRegression()
las_cv = Lasso()
kn_cv = KNeighborsRegressor(n_neighbors=30)
reg_cv_model = cross_validate(reg_cv, X=x_usa, y=y_usa, cv=kf, scoring=('neg_mean_squared_error', 'r2'), return_train_score=False)
las_cv_model = cross_validate(las_cv, X=x_usa, y=y_usa, cv=kf, scoring=('neg_mean_squared_error', 'r2'), return_train_score=False)
kn_cv_model = cross_validate(kn_cv, X=x_usa, y=y_usa, cv=kf, scoring=('neg_mean_squared_error', 'r2'), return_train_score=False)

reg_cv_scores = cross_val_score(reg_cv, X=x_usa, y=y_usa, cv=kf)
las_cv_scores = cross_val_score(las_cv, X=x_usa, y=y_usa, cv=kf)
kn_cv_scores = cross_val_score(kn_cv, X=x_usa, y=y_usa, cv=kf)
print('Linear Regression: The average r-squared value from {}-fold cross-validation is: {}'.format(kf.n_splits, reg_cv_scores.mean()), ('\nLasso: The average r-squared value from {}-fold cross-validation is: {}'.format(kf.n_splits, las_cv_scores.mean())), ('\nk-NN Regressor: The average r-squared value from {}-fold cross-validation is: {}'.format(kf.n_splits, kn_cv_scores.mean())))
# Pick a target gender, m or f
target = 'f_suicides_no'
reg = LinearRegression().fit(x_s_train, y_s_train[target])
y_s_hat = reg.predict(x_s_test)
y_s_hat = pd.DataFrame(y_s_hat,columns=[target])

#compute metrics

mse = (metrics.mean_squared_error(y_pred=y_s_hat[target], y_true=y_s_test[target] ))
r2_s = metrics.r2_score(y_pred=y_s_hat, y_true=y_s_test[target])
fig, ax = plt.subplots(figsize=(12,6))
sns.scatterplot(x=y_s_test[target],y=y_s_hat[target])
ax.set(xlabel = 'Actual y', ylabel="Predicted y")
plt.show()
print('The r-squared value is: {}, and MSE is: {}'.format(r2, mse))
kf = KFold(5,shuffle=True)
reg_cv = LinearRegression()
las_cv = Lasso()
kn_cv = KNeighborsRegressor(n_neighbors=30)
reg_cv_model = cross_validate(reg_cv, X=x_s, y=y_s[target], cv=kf, scoring=('neg_mean_squared_error', 'r2'), return_train_score=False)
las_cv_model = cross_validate(las_cv, X=x_s, y=y_s[target], cv=kf, scoring=('neg_mean_squared_error', 'r2'), return_train_score=False)
kn_cv_model = cross_validate(kn_cv, X=x_s, y=y_s[target], cv=kf, scoring=('neg_mean_squared_error', 'r2'), return_train_score=False)

reg_cv_scores_s = cross_val_score(reg_cv, X=x_s, y=y_s[target], cv=kf)
las_cv_scores_s = cross_val_score(las_cv, X=x_s, y=y_s[target], cv=kf)
kn_cv_scores_s = cross_val_score(kn_cv, X=x_s, y=y_s[target], cv=kf)
print('Linear Regression: The average r-squared value from {}-fold cross-validation is: {}'.format(kf.n_splits, reg_cv_scores_s.mean()), ('\nLasso: The average r-squared value from {}-fold cross-validation is: {}'.format(kf.n_splits, las_cv_scores_s.mean())), ('\nk-NN Regressor: The average r-squared value from {}-fold cross-validation is: {}'.format(kf.n_splits, kn_cv_scores_s.mean())))
print('The results are much better. We can more accurately predict {} by {} percent over our attempt to predict suicides for both genders.'.format(target, np.round((reg_cv_scores_s.mean()/reg_cv_scores.mean())*100 - 100,2)))

