import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
! pip install empiricaldist
from empiricaldist import Pmf,Cdf
from scipy.stats import linregress
import statsmodels.formula.api as smf
import missingno as msno
from scipy.stats import chi2_contingency
from scipy.stats import ttest_ind
import shap
import xgboost as xgb
def pmf_cdf_plots(dataset):
    for c in dataset.columns:
        pmf = Pmf.from_seq(dataset[c].tolist(), normalize=True)
        cdf = Cdf.from_seq(dataset[c].tolist())

        plt.figure(figsize = [10,5])# larger figure size for subplots 

        plt.subplot(1,2,1)
        pmf.bar()
        plt.xlabel(c)
        plt.ylabel("PMF")

        plt.subplot(1,2,2)
        cdf.plot()
        plt.xlabel(c)
        plt.ylabel("CDF")

        plt.show()
        
def scatter_plots(dataset):
    float_cols = [i for i in dataset.columns if dataset[i].dtypes == 'float64' and i != 'median_house_value']
    for i in float_cols:
        i_jitter = dataset[i]+np.random.normal(0,2,size=len(dataset))
        j_jitter = dataset['median_house_value']+np.random.normal(0,2,size=len(dataset))
        plt.plot(i_jitter,j_jitter,'o',data=dataset, alpha=0.1, markersize=1)
        plt.xlabel(i)
        plt.ylabel('median house value')
        plt.show()
        
def violin_plots(dataset):
    float_cols = [i for i in dataset.columns if dataset[i].dtypes == 'float64' and i != 'median_house_value']
    for i in float_cols+['median_house_value']:
        my_order = dataset.groupby(by=["ocean_proximity"])[i].mean().sort_values().iloc[::-1].index.tolist()
        sns.violinplot(x='ocean_proximity',y=i, data=dataset, inner=None, order=my_order)
        plt.show()
        
def box_plots(dataset):
    float_cols = [i for i in dataset.columns if dataset[i].dtypes == 'float64' and i != 'median_house_value']
    for i in float_cols+['median_house_value']:
        my_order = dataset.groupby(by=["ocean_proximity"])[i].mean().sort_values().iloc[::-1].index.tolist()
        sns.boxplot(x='ocean_proximity',y=i, data=dataset, whis=10, order=my_order)
        plt.show()
        
def reg_plots(dataset):
    float_cols = [i for i in dataset.columns if dataset[i].dtypes == 'float64' and i != 'median_house_value']
    for i in float_cols:
        results = smf.ols('median_house_value~'+i, data=dataset).fit()
        pred12 = results.predict(dataset[i])
        i_jitter = dataset[i]+np.random.normal(0,2,size=len(dataset))
        j_jitter = dataset['median_house_value']+np.random.normal(0,2,size=len(dataset))
        plt.plot(i_jitter,j_jitter,'o',data=dataset, alpha=0.1, markersize=1)

    #    plt.plot(dataset[i],dataset['median_house_value'], 'o', alpha=0.05, markersize=1)
        plt.plot(dataset[i], pred12, label='median_house_value')
        plt.xlabel(i)
        plt.ylabel('predicted median house value')
        plt.legend()
        plt.show()
dataset = pd.read_csv('../input/california-housing-prices/housing.csv')
print(dataset.head())
print(dataset.info())
dataset.shape
print(dataset.columns)
print(round(dataset.describe(),2))
print(round(dataset.describe(include='object')))
cormat = dataset.corr()
plt.figure(figsize=(10,10))
sns.heatmap(round(dataset.corr(),2), annot=True, cmap="RdYlGn")
dataset['bedrooms_to_rooms'] = dataset['total_rooms']/dataset['total_bedrooms']
dataset['households_to_population'] = dataset['population']/dataset['households']
#dataset.reset_index(inplace=True)
dataset['per_capita_income'] = dataset['median_income']/dataset['population']
dataset['per_household_income'] = dataset['median_income']/dataset['households']
drop_cols = ['total_rooms','total_bedrooms','population', 'households', 'median_income','median_house_value']
df = dataset.drop(drop_cols, axis=1)
cormat = df.corr()
plt.figure(figsize=(10,10))
sns.heatmap(round(df.corr(),2), annot=True, cmap="RdYlGn")
scaled_df=df.groupby('ocean_proximity').transform(lambda x: (x - x.mean()) / x.std())
scaled_df['ocean_proximity'] = dataset['ocean_proximity']
scaled_df['median_house_value'] = dataset['median_house_value']
msno.matrix(scaled_df)
msno.heatmap(scaled_df)
msno.bar(scaled_df)
dataset
scaled_df
dataset.columns
pmf_cdf_plots(scaled_df)
sns.pairplot(scaled_df, hue = 'ocean_proximity')
cormat = scaled_df.corr()
plt.figure(figsize=(10,10))
sns.heatmap(round(scaled_df.corr(),2), annot=True, cmap="RdYlGn")
ocean_proximity_count = scaled_df.ocean_proximity.value_counts().sort_values()
ocean_proximity_count.plot(kind='bar', figsize=(8,8))
print(round(scaled_df.groupby("ocean_proximity").mean(),2))
float_cols
scatter_plots(scaled_df)
violin_plots(scaled_df)
box_plots(scaled_df)
reg_plots(scaled_df)
print(dataset.columns)
for i in ['bedrooms_to_rooms','households_to_population', 'households_to_population', 'per_capita_income']:
    results = smf.ols('median_house_value~'+i, data=dataset).fit()
    pred12 = results.predict(dataset[i])
#    i_jitter = dataset[i]+np.random.normal(0,2,size=len(dataset))
#   j_jitter = dataset['median_house_value']+np.random.normal(0,2,size=len(dataset))
    plt.plot(i,'median_house_value','o',data=dataset, alpha=0.1, markersize=1)
    plt.plot(dataset[i], pred12, label='median_house_value')
    plt.xlabel(i)
    plt.ylabel('predicted median house value')
    plt.show()
for i in ['bedrooms_to_rooms','households_to_population', 'median_house_value','households_to_population', 'per_capita_income']:
    dataset[i].hist(bins=50, alpha=0.5)
    plt.xscale('log')
    plt.show()
for i in ['bedrooms_to_rooms','households_to_population','households_to_population', 'per_capita_income']:
    splot = sns.regplot(x=i, y='median_house_value', 
                        data=dataset,
                        scatter_kws={'alpha':0.15},
                        fit_reg=True)
    splot.set(xscale="log", yscale='log')
    plt.show()
data = pd.get_dummies(scaled_df, prefix_sep='_', columns=['ocean_proximity'], drop_first=True)
data
#data.drop('index',axis=1 ,inplace=True)
data['y_response'] = data['median_house_value']
data.drop('median_house_value', axis=1, inplace=True)
data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
#trans = StandardScaler()
#data = trans.fit_transform(data)
#X, y = data[:,1:-1], data[:,-1]
X, y = data.iloc[:,1:-1], data.iloc[:,-1]
#X, y = data[:,:-1], data[:,-1]
y
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=123)
DM_train = xgb.DMatrix(data=X_train, label=y_train)
DM_test = xgb.DMatrix(data=X_test, label=y_test)
params = {
    # Parameters that we are going to tune.
    'max_depth':6,
    'min_child_weight': 1,
    'eta':.3,
    'subsample': 1,
    'colsample_bytree': 1,
    # Other parameters
    'objective':'reg:linear',
    'eval_metric':'mae',
    
}
model = xgb.train(
                params,
                DM_train,
                num_boost_round=999,
                evals=[(DM_test,"Test")],
                early_stopping_rounds = 10,
)
cv_results = xgb.cv(
        params,
        DM_train,
        num_boost_round=999,
        seed=42,
        nfold=5,
        metrics={'mae'},
        early_stopping_rounds=10,
        as_pandas = True
)
cv_results
cv_results[['train-mae-mean','test-mae-mean']].plot()
cv_results[['train-mae-std','test-mae-std']].plot()
gridsearch_params = [
    (max_depth, min_child_weight)
    for max_depth in range(9,12)
    for min_child_weight in range(5,8)
]
gridsearch_params
min_mae = float("Inf")
best_params = None

for max_depth, min_child_weight in gridsearch_params:
    print("CV with max_depth={}, min_child_weight={}".format(max_depth, min_child_weight))
    
    # Update our parameter
    params['max_depth'] = max_depth
    params['min_child_weight'] = min_child_weight
    
    # Run CV
    cv_results = xgb.cv(
    params,
    DM_train,
    num_boost_round=999,
    seed=42,
    nfold=5,
    metrics={'mae'},
    early_stopping_rounds = 10
    )
    
    # update best MAE
    mean_mae = cv_results['test-mae-mean'].min()
    boost_rounds = cv_results['test-mae-mean'].argmin()
    print("\tMAE {} for {} rounds".format(mean_mae, boost_rounds))
    if mean_mae < min_mae:
        min_mae = mean_mae
        best_params = (max_depth, min_child_weight)
        
print("Best params: {}, {}, MAE:{}".format(best_params[0], best_params[1], min_mae))
    
params['max_depth'] = 10
params['min_child_weight'] = 6
gridsearch_params = [
    (subsample, colsample)
    for subsample in [i/10. for i in range(7,11)]
    for colsample in [i/10. for i in range(7,11)]
]
gridsearch_params
min_mae = float("Inf")
best_params = None
# We start by the largest values and go down to the smallest
for subsample, colsample in reversed(gridsearch_params):
    print("CV with subsample={}, colsample={}".format(
                             subsample,
                             colsample))
    # We update our parameters
    params['subsample'] = subsample
    params['colsample_bytree'] = colsample
    # Run CV
    cv_results = xgb.cv(
        params,
        DM_train,
        num_boost_round=999,
        seed=42,
        nfold=5,
        metrics={'mae'},
        early_stopping_rounds=10
    )
    # Update best score
    mean_mae = cv_results['test-mae-mean'].min()
    boost_rounds = cv_results['test-mae-mean'].argmin()
    print("\tMAE {} for {} rounds".format(mean_mae, boost_rounds))
    if mean_mae < min_mae:
        min_mae = mean_mae
        best_params = (subsample,colsample)
print("Best params: {}, {}, MAE: {}".format(best_params[0], best_params[1], min_mae))

params['subsample'] = 1.0
params['colsample_bytree'] = 1.0
# This can take some time…
min_mae = float("Inf")
best_params = None
for eta in [.3, .2, .1, .05, .01, .005]:
    print("CV with eta={}".format(eta))
    # We update our parameters
    params['eta'] = eta
    # Run and time CV
    cv_results = xgb.cv(
            params,
            DM_train,
            num_boost_round=999,
            seed=42,
            nfold=5,
            metrics=['mae'],
            early_stopping_rounds=10
          )
    # Update best score
    mean_mae = cv_results['test-mae-mean'].min()
    boost_rounds = cv_results['test-mae-mean'].argmin()
    print("\tMAE {} for {} rounds\n".format(mean_mae, boost_rounds))
    if mean_mae < min_mae:
        min_mae = mean_mae
        best_params = eta
print("Best params: {}, MAE: {}".format(best_params, min_mae))
params['eta'] = 0.05
params
model = xgb.train(
    params,
    DM_train,
    num_boost_round=999,
    evals=[(DM_test, "Test")],
    early_stopping_rounds=10
)
y_hat = model.predict(DM_test)
y_test
y_hat
i_jitter = y_hat+np.random.normal(0,2,size=len(y_hat))
j_jitter = y_test.values+np.random.normal(0,2,size=len(y_test))
plt.plot(i_jitter,j_jitter,'o',data=dataset, alpha=0.1, markersize=1)
plt.xlabel('Predicted')
plt.ylabel('median house value')
plt.show()
i_jitter = y_hat+np.random.normal(0,2,size=len(y_hat))
residuals = y_test.values-y_hat
plt.plot(y_hat, residuals, 'o',markersize=1)
sns.distplot(residuals)
sns.distplot(y_hat)
sns.distplot(y_test)
xgb.plot_importance(model)
output_df=pd.DataFrame({
    'actual_value':y_test.tolist(),
    'predicted_value': y_hat
})
output_df.to_csv('./output.csv')
