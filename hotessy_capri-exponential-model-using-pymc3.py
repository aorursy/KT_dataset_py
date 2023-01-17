import numpy as np 

import pandas as pd 

from pathlib import Path



import pymc3 as pm

import arviz as az



import cufflinks as cf

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression



from sklearn.metrics import r2_score, mean_squared_error

from ipywidgets import interact, interact_manual, fixed
pd.set_option('display.max_rows', 500)

pd.set_option('use_inf_as_na', True)

cf.set_config_file(offline=True, theme='pearl');

path = Path("../input/novel-corona-virus-2019-dataset/")
master_df = pd.read_csv(path/'covid_19_data.csv')
recovered_df = (pd.read_csv(path/'time_series_covid_19_recovered.csv')

                .drop(columns=['Lat', 'Long'])

                .groupby('Country/Region')

                .sum())



deaths_df = (pd.read_csv(path/'time_series_covid_19_deaths.csv')

             .drop(columns=['Lat', 'Long'])

             .groupby('Country/Region')

             .sum())



confirmed_df = (pd.read_csv(path/'time_series_covid_19_confirmed.csv')

                .drop(columns=['Lat', 'Long'])

                .groupby('Country/Region')

                .sum())
sorted_country_list = confirmed_df.sort_values(by=confirmed_df.columns[-1], ascending=False).index.to_list()
@interact(country=sorted_country_list, threshold=(1, 1000, 10), fit=True)

def log_lin_visualise(country, fit, threshold=100):

    

    y = confirmed_df.filter(items=[country], axis=0).values.squeeze(0)

    y = np.log(y[y > threshold])

    x = np.arange(1, y.shape[0] + 1)

    

    plt.figure(figsize=(10, 5))

    plt.plot(x, y, label='Observed')

    

    if fit:



        lr = LinearRegression(normalize=True)

        lr.fit(X=x.reshape(-1, 1) , y=y)

        α, β =  lr.intercept_, lr.coef_



        y_fitted = lr.predict(X=x.reshape(-1, 1))



        print("Solving linear regression using OLS ... ")

        print(f"* r2_score = {round(r2_score(y, y_fitted), 2)}")

        print(f"* mean_squared_error = {round(mean_squared_error(y, y_fitted), 2)}")



        plt.plot(x, y_fitted, label=f"{round(α, 2)} + {round(β[0], 2)}*x")



    plt.xlabel(f'Days Since {threshold}th Case')

    plt.ylabel('Natural Logarithm of Confirmed Cases')

    plt.legend()

    plt.title(country)

    plt.show()

    plt.close()
country = 'US'

threshold = 100



y = confirmed_df.filter(items=[country], axis=0).values.squeeze(0)

y = np.log(y[y > threshold])

x = np.arange(1, y.shape[0] + 1)





with pm.Model() as unpooled_model:

    

    

    # priors

    α = pm.Normal(name='α', mu=int(np.log(threshold)), sd=10)

    β = pm.Normal(name='β')

    

    # error

    σ = pm.HalfNormal(name='σ', sd=10)

    

    # expected value

    μ = pm.Deterministic(name='μ', var= α + β*x)

    

    # liklihood == 'prior_predictive'

    pm.Normal(name=country, mu=μ, sd=σ, observed=y)

    

    pm.model_to_graphviz().save('unpooled_model.png')

    

pm.model_to_graphviz(unpooled_model)
## shape-sanity checks



# ((β.random()*x) + α.random()).shape

# Y[4].shape



# idx = [0, 0, 1, 1, 1, 4]

# α.random()[idx]
with unpooled_model:

    

    # sampling liklihood

    prior = pm.sample_prior_predictive()

    

    # posterior

    trace = pm.sample()

    

    # predictions == 'posterior_predictive'

    pred = pm.sample_posterior_predictive(trace)
unpooled = az.from_pymc3(trace=trace, prior=prior, posterior_predictive=pred, model=unpooled_model)

prior_vars = ['α', 'β', 'σ']

unpooled
unpooled.posterior['α'].shape
summary_df = az.summary(unpooled)

summary_df.to_csv(f'unpooled_{country}_summary.csv')
az.plot_trace(data=unpooled, var_names=prior_vars);
az.plot_posterior(data=unpooled, var_names=prior_vars, group='posterior');
az.plot_posterior(data=unpooled, var_names=prior_vars, group='prior');
az.plot_ppc(data=unpooled, group='posterior');
# pm.find_MAP(model=unpooled_model)
mean_priors = az.summary(unpooled).filter(items=prior_vars, axis=0)['mean']

y_fitted = mean_priors['α'] + mean_priors['β']*x



mean_priors
ci = 0.95





az.plot_hpd(x=x, y=unpooled.posterior_predictive[country], credible_interval=ci)#, plot_kwargs={'label': f'Predictions ({ci})'});

# az.plot_hpd(x=x, y=unpooled.posterior['μ'], color='C2', credible_interval=ci)#, plot_kwargs={'label': f'Expectation ({ci})'});

plt.plot(x, y, label='Observed');

plt.plot(x, y_fitted, label=f"{mean_priors['α']} + {mean_priors['β']}*x", color='k');



plt.xlabel(f'Days Since {threshold}th Case')

plt.ylabel('Natural Logarithm of Confirmed Cases')

plt.title(country)

plt.legend()

plt.show()

plt.close()
az.r2_score(y_true=y, y_pred=unpooled.posterior_predictive[country].data.reshape(-1, y.shape[0]))
# az.plot_dist(unpooled.log_likelihood[country]);

# az.plot_dist(unpooled.prior_predictive[country]);
## Mean of predicted values



y_fitted = unpooled.posterior_predictive[country].data.reshape(-1, y.shape[0]).mean(axis=0)

print(f"r2_score = {round(r2_score(y, y_fitted), 2)}")

print(f"mean_squared_error = {round(mean_squared_error(y, y_fitted), 2)}")
## Mean of expected values



# y_fitted = unpooled.posterior['μ'].data.reshape(-1, y.shape[0]).mean(axis=0)

# print(f"r2_score = {round(r2_score(y, y_fitted), 2)}")

# print(f"mean_squared_error = {round(mean_squared_error(y, y_fitted), 2)}")
# Data for all counties needs to be in similar dimensions.



cumulative_threshold = 1e4 # to select countries

threshold = 100 # to select start_date



high_confirmed_df = confirmed_df[confirmed_df.iloc[:, -1] > cumulative_threshold]

high_confirmed_df = high_confirmed_df.where(cond=lambda x: x > threshold, other=np.nan) # or -1
# Dropping for now

# high_confirmed_df = high_confirmed_df.drop(index=['China', 'Korea, South'])
majority = high_confirmed_df.shape[0]/2



high_confirmed_df.isna().sum().iplot(hline=majority,

                                     xTitle='Date', yTitle=f'Countries below {threshold} cases', 

                                     title='Threshold Cases for Highly Infected Countries');
cross_over_date = (high_confirmed_df.isna().sum() < majority).idxmax()

cross_over_date
thresholded_high_confirmed_df = high_confirmed_df.iloc[:, high_confirmed_df.columns.to_list().index(cross_over_date):]

thresholded_high_confirmed_df = thresholded_high_confirmed_df.dropna(how='any', axis=0)

thresholded_high_confirmed_df = thresholded_high_confirmed_df.applymap(np.log)



print(thresholded_high_confirmed_df.shape)
n_countries = thresholded_high_confirmed_df.shape[0]

date_points = thresholded_high_confirmed_df.shape[1]

idx = np.repeat(a=np.arange(n_countries), repeats=date_points)
x = np.arange(1, date_points + 1)

X = np.stack([x]*n_countries, axis=0)



Y = thresholded_high_confirmed_df.values



print(X.shape, Y.shape)
# vectorised implementation

'''

Here, 

    μ.shape = (440,)

    σ[idx].shape = (440,) 

    Y.flatten().shape = (440,)

Data for each country is repeated, so first 40 points (index 0 to 39) are of first country, 

second 40 points (index 40 to 79) are of second country and so on.

'''





def split_and_pack(labels, data, axis=2):

    '''

    Helper function to unpack values from a vectorised output

    '''

    splits = np.split(ary=data, axis=axis, indices_or_sections=len(labels))

    d = dict(zip(labels, data))

    return d

    



with pm.Model() as pooled_model:



    

    X = pm.Data(name='X', value=X)

    Y = pm.Data(name='Y', value=Y.flatten())

    

    # priors

    α = pm.Normal(name='α', mu=int(np.log(threshold)), sd=10, shape=n_countries)

    β = pm.Normal(name='β', sd=5, shape=n_countries)

    

    # error

    σ = pm.HalfNormal(name='σ', sd=10, shape=n_countries)

    

    # expected value

    

    μ = pm.Deterministic(name='μ', var= (α + β*X.T).T.flatten())

    

    # liklihood == 'prior_predictive'

    pm.Normal(name='pooled', mu=μ, sd=σ[idx], observed=Y)

    

    pm.model_to_graphviz().save('pooled_model.png')

    

pm.model_to_graphviz(pooled_model)
# non-vectorised implementation

# '''

# All three parameters (mu, sd, observed) have to be of the same shape / broadcasteble

# Here, 

#     μ[i].shape = (40,)

#     σ[i].shape = (1,) # broadcasted to (40,)

#     Y[i].shape = (40,)

# '''



# with pm.Model() as pooled_model:



#     X = pm.Data(name='X', value=X)

#     Y = pm.Data(name='Y', value=Y) # not using it since we will iterate over the dataframe

    

#     # priors

#     α = pm.Normal(name='α', mu=int(np.log(threshold)), sd=10, shape=n_countries)

#     β = pm.Normal(name='β', sd=5, shape=n_countries)

    

#     # error

#     σ = pm.HalfNormal(name='σ', sd=10, shape=n_countries)

    

#     # expected value -- Deterministic matrix μ

#     ## transpose is necessary because operations are performed row-major style

# #     μ = pm.Deterministic(name='μ', var= (α + β*X.T).T) 

    

#     # liklihood == 'prior_predictive'

#     for i, (index, row) in enumerate(thresholded_high_confirmed_df.iterrows()):

        

#         # use with Deterministic matrix μ

# #       pm.Normal(name=index, mu=μ[i], sd=σ[i], observed=row.values) 

        

# #         μ = α[i] + β[i]*x

#         μ = pm.Deterministic(name=f'μ_{index}', var= α[i] + β[i]*X[i])  # to save μ in the trace

#         pm.Normal(name=index, mu=μ, sd=σ[i], observed=Y[i])  # or row.values

            

# pm.model_to_graphviz(pooled_model)
## shape-sanity checks



# ((β.random()*X.T) + α.random()).T.shape

# Y[4].shape



# idx = [0, 0, 1, 1, 1, 4]

# α.random()[idx]
with pooled_model:

    

    # sampling liklihood

    prior = pm.sample_prior_predictive()

    

    # posterior

    trace = pm.sample()

    

    # predictions == 'posterior_predictive'

    pred = pm.sample_posterior_predictive(trace)
pooled = az.from_pymc3(trace=trace, prior=prior, posterior_predictive=pred, model=pooled_model)

prior_vars = ['α', 'β', 'σ']

pooled
# pm.find_MAP(model=pooled_model)
pooled.posterior['α'].shape # chains x samples x n_countries
summary_df = az.summary(pooled) 

summary_df.to_csv('pooled_summary.csv')
countries = thresholded_high_confirmed_df.index.to_list()

means = {var : split_and_pack(data=summary_df.filter(like=var, axis=0)['mean'], labels=countries, axis=0) for var in prior_vars }

means_df = pd.DataFrame(means)



means_df
# az.plot_trace(data=pooled, var_names=prior_vars);
## Plotting posterior will give a combined view of all countries

# az.plot_posterior(data=pooled, var_names=prior_vars, group='posterior');
# az.plot_posterior(data=pooled, var_names=prior_vars, group='prior');
az.plot_forest(data=pooled, var_names=prior_vars, combined=True, credible_interval=0.99);
# posterior_predictive = split_and_pack(data=pooled.posterior_predictive['pooled'].data, labels=countries, axis=2) # not working??

posterior_predictive = dict(zip(countries, np.split(ary=pooled.posterior_predictive['pooled'].data, axis=2, indices_or_sections=len(countries))))



X_countrywise = split_and_pack(data=pooled.constant_data['X'].data, labels=countries, axis=0)



# Y_countrywise = split_and_pack(data=pooled.constant_data['Y'].data, labels=countries, axis=0)

Y_countrywise = dict(zip(countries, np.split(ary=pooled.constant_data['Y'].data, axis=0, indices_or_sections=len(countries))))
@interact(country=countries)

def plot_countrywise_posterior(country):

    

    ci = 0.95

    

    plt.figure(figsize=(15, 10))



    az.plot_hpd(x=x, y=posterior_predictive[country], credible_interval=ci);

    plt.plot(X_countrywise[country], Y_countrywise[country], label='Observed');

    

    y_fitted = means_df['α'][country] + means_df['β'][country]*X_countrywise[country]

    

    plt.plot(X_countrywise[country], y_fitted, label=f"{means_df['α'][country]} + {means_df['β'][country]}*x", color='k');



    plt.xlabel(f'Days Since {threshold}th Case')

    plt.ylabel('Natural Logarithm of Confirmed Cases')

    plt.title(country)

    plt.legend()

    plt.show()

    plt.close()

    

    bayesian_r2 = az.r2_score(y_true=Y_countrywise[country], y_pred=posterior_predictive[country].reshape(-1, Y_countrywise[country].shape[0]))

    y_fitted_bayesian = posterior_predictive[country].reshape(-1, Y_countrywise[country].shape[0]).mean(axis=0)

    print(f"Bayesian r2_score = {round(bayesian_r2['r2'], 2)}")

    print(f"Bayesian r2_score std = {round(bayesian_r2['r2_std'], 2)}")

    print('\n')

    print(f"Point(mean) r2_score = {round(r2_score(Y_countrywise[country], y_fitted_bayesian), 2)}")

    print(f"Point(mean) mean_squared_error = {round(mean_squared_error(Y_countrywise[country], y_fitted_bayesian), 2)}")
x = np.arange(1, date_points + 1)

X = np.stack([x]*n_countries, axis=0)



Y = thresholded_high_confirmed_df.values



print(X.shape, Y.shape)
# non-vectorised implementation

'''

All three parameters (mu, sd, observed) have to be of the same shape / broadcasteble

Here, 

    μ[i].shape = (40,)

    σ[i].shape = (1,) # broadcasted to (40,)

    Y[i].shape = (40,)

'''



with pm.Model() as hierarchical_model:



    X = pm.Data(name='X', value=X)

    Y = pm.Data(name='Y', value=Y) # not using it since we will iterate over the dataframe

    

    # hyper-priors

    α_μ = pm.Normal(name='α_μ', mu=int(np.log(threshold)), sd=10)

    α_σ = pm.HalfNormal(name='α_σ', sd=10)

    

    β_μ = pm.Normal(name='β_μ', sd=10)

    β_σ = pm.HalfNormal(name='β_σ', sd=10)

    

    # priors

    α = pm.Normal(name='α', mu=α_μ, sd=α_σ, shape=n_countries)

    β = pm.Normal(name='β', mu=β_μ, sd=β_σ, shape=n_countries)

    

    # error

    σ = pm.HalfNormal(name='σ', sd=10, shape=n_countries)

    

    # liklihood

    for i, (index, row) in enumerate(thresholded_high_confirmed_df.iterrows()):

        

        μ = pm.Deterministic(name=f'μ_{index}', var= α[i] + β[i]*X[i])

        pm.Normal(name=index, mu=μ, sd=σ[i], observed=Y[i])  # or observed=row.values

        

    pm.model_to_graphviz().save('hierarchical_model.png')

            

pm.model_to_graphviz(hierarchical_model)
with hierarchical_model:

    

    # sampling liklihood

    prior = pm.sample_prior_predictive()

    

    # posterior

    trace = pm.sample()

    

    # predictions == 'posterior_predictive'

    pred = pm.sample_posterior_predictive(trace)
hierarchical = az.from_pymc3(trace=trace, prior=prior, posterior_predictive=pred, model=hierarchical_model)

prior_vars = ['α', 'β', 'σ']

hierarchical
# pm.find_MAP(model=hierarchical_model)
hierarchical.posterior['α'].shape # chains x samples x n_countries
summary_df = az.summary(hierarchical) 

summary_df.to_csv('hierarchical_summary.csv')
countries = thresholded_high_confirmed_df.index.to_list()

means = {var : split_and_pack(data=summary_df.filter(like=f"{var}[", axis=0)['mean'], labels=countries, axis=0) for var in prior_vars }

means_df = pd.DataFrame(means)



means_df
# az.plot_trace(data=hierarchical, var_names=prior_vars);
## Plotting posterior will give a combined view of all countries

# az.plot_posterior(data=hierarchical, var_names=prior_vars, group='posterior');
# az.plot_posterior(data=hierarchical, var_names=prior_vars, group='prior');
az.plot_forest(data=hierarchical, var_names=prior_vars, combined=True, credible_interval=0.99);
X_countrywise = split_and_pack(data=hierarchical.constant_data['X'].data, labels=countries, axis=0)

Y_countrywise = split_and_pack(data=hierarchical.constant_data['Y'].data, labels=countries, axis=0)
@interact(country=countries)

def plot_countrywise_posterior(country):

    

    ci = 0.95

    

    plt.figure(figsize=(15, 10))



    az.plot_hpd(x=X_countrywise[country], y=hierarchical.posterior_predictive[country], credible_interval=ci);

    plt.plot(X_countrywise[country], Y_countrywise[country], label='Observed');

    

    y_fitted = means_df['α'][country] + means_df['β'][country]*X_countrywise[country]

    

    plt.plot(X_countrywise[country], y_fitted, label=f"{means_df['α'][country]} + {means_df['β'][country]}*x", color='k');



    plt.xlabel(f'Days Since {threshold}th Case')

    plt.ylabel('Natural Logarithm of Confirmed Cases')

    plt.title(country)

    plt.legend()

    plt.show()

    plt.close()

    

    bayesian_r2 = az.r2_score(y_true=Y_countrywise[country], y_pred=hierarchical.posterior_predictive[country].data.reshape(-1, Y_countrywise[country].shape[0]))

    y_fitted_bayesian = hierarchical.posterior_predictive[country].data.reshape(-1, Y_countrywise[country].shape[0]).mean(axis=0)

    print(f"Bayesian r2_score = {round(bayesian_r2['r2'], 2)}")

    print(f"Bayesian r2_score std = {round(bayesian_r2['r2_std'], 2)}")

    print('\n')

    print(f"Point(mean) r2_score = {round(r2_score(Y_countrywise[country], y_fitted_bayesian), 2)}")

    print(f"Point(mean) mean_squared_error = {round(mean_squared_error(Y_countrywise[country], y_fitted_bayesian), 2)}")
datasets = {'unpooled' : unpooled,

           'pooled' : pooled,

           'hierarchical': hierarchical} 
az.plot_forest(data=list(datasets.values()), model_names=list(datasets.keys()), var_names=prior_vars, credible_interval=0.99, combined=True);

plt.savefig('comparison.png')