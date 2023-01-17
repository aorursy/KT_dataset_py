from pathlib import Path 

import arviz as az

import numpy as np

import pandas as pd

import pymc3 as pm

import seaborn as sns

import theano.tensor as tt

from sklearn.preprocessing import StandardScaler

import matplotlib.patches as mpatches

import cufflinks as cf

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (16, 8)

cf.set_config_file(theme='pearl', offline=True)
path = Path("../input/bayesian-methods-for-hackers/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/")

challenger = (pd.read_csv(path/'Chapter2_MorePyMC'/'data'/'challenger_data.csv', parse_dates=['Date'], index_col='Date')

              .sort_index())

challenger.tail()
## Cleaning Data



df = challenger.dropna(subset=['Damage Incident'])

df = df.drop(index=pd.to_datetime('1986-01-28')) # Challenger Accident row

df['Damage Incident'] = df['Damage Incident'].astype(int)
df.plot.scatter(x='Temperature', y='Damage Incident', s=200);
p_threshold = 0.5

feature = ['Temperature']
with pm.Model() as log_reg_model:



    X_transformed = StandardScaler().fit_transform(df[feature].values)



#     X = pm.Data('X', df[feature].values)

    X = pm.Data('X', X_transformed)

    y = pm.Data('y', df['Damage Incident'].values)



    # priors

    α = pm.Normal('α', sigma=5)

    β = pm.Normal('β', sigma=5, shape=X.shape.eval()[1])

    

    # decision boundary

    lo_threshold = np.log(p_threshold/(1 - p_threshold))



    if X.ndim > 1 and X.shape.eval()[1] > 1:

        

        # (-2, 2) range taken because x are standardised to (0, 1)

        x_temp = np.expand_dims(np.linspace(-2, 2, 100), axis=1) 

        X_temp = np.repeat(x_temp, repeats=X.shape.eval()[1] - 1, axis=1)

        Σ = pm.math.matrix_dot(X_temp, β[:-1])

    else:

        Σ = 0

        

    # values of the decision boundary for the last feature 

    x_k = (lo_threshold - α - Σ)/β[-1] 

#     X_bd = np.stack([X_temp, x_k], axis=1)

    x_k = pm.Deterministic('x_k', x_k)



    # log-odds

    π = pm.Deterministic('π', α + pm.math.dot(X, β))



    # logistic transformation

    ß = pm.Deterministic('ß', pm.math.sigmoid(π))



    # liklihood

    pm.Bernoulli('liklihood', p=ß, observed=y)

    pm.model_to_graphviz().save('log_reg_model_challenger.png')

    

pm.model_to_graphviz(log_reg_model)
with log_reg_model:

    trace = pm.sample()

    ppc = pm.sample_posterior_predictive(trace)
sampled_vars = ['β', 'α']

log_reg = az.from_pymc3(model=log_reg_model, trace=trace, posterior_predictive=ppc)
az.plot_trace(log_reg, var_names=sampled_vars);
# for legend

handles = {}



# Data Points

x = log_reg.constant_data['X'].data.squeeze(-1)

y = log_reg.constant_data['y'].data

scatter = plt.scatter(x=x, y=y, c=y, s=150);

handles.update(dict(zip([0, 1], scatter.legend_elements()[0])))





# Decison Boundary

δ_hpd = az.hpd(log_reg.posterior['x_k'], credible_interval=0.95).mean(axis=0);

plt.fill_betweenx(y=[0, 1], x1=δ_hpd[0], x2=δ_hpd[1], color='C1', alpha=0.2);

handles['Decision Boundary Spread'] = mpatches.Patch(color=f'C1', alpha=0.2)



# Mean Decision Boundary

line = plt.vlines(x=log_reg.posterior['x_k'].mean(), ymin=0.0, ymax=1.0, color='C1');

# handles['Mean Decision Boundary'] = mpatches.Patch(color=f'C1')

handles.update({'Mean Decision Boundary': line})







# Labelling

plt.xlabel(f'{feature[0]} Z Score');

plt.ylabel('Damage Incident');

plt.title('Decision Boundary');

plt.legend(handles=list(handles.values()), labels=list(handles.keys()));

# plt.legend()
# az.plot_hpd(x=x, y=log_reg.posterior['ß'], credible_interval=0.95, fill_kwargs={'alpha':0.2}, color='C1');
iris = pd.read_csv('../input/iris/Iris.csv', index_col='Id')

iris.head()
# Choosing only two categories



df = iris.query("Species in ['Iris-setosa', 'Iris-versicolor']").reset_index(drop=True)

df.head()
idx, group_names = df['Species'].factorize()
features = df.set_index('Species').columns.to_list()

features
df.boxplot(by='Species');
p_threshold = 0.5

feature = ['SepalLengthCm', 'SepalWidthCm']
## SAME MODEL AS EXAMPLE I



with pm.Model() as log_reg_model:



    X_transformed = StandardScaler().fit_transform(df[feature].values)



#     X = pm.Data('X', df[feature].values)

    X = pm.Data('X', X_transformed)

    y = pm.Data('y', idx)



    # priors

    α = pm.Normal('α', sigma=5)

    β = pm.Normal('β', sigma=5, shape=X.shape.eval()[1])

    

    # decision boundary

    lo_threshold = np.log(p_threshold/(1 - p_threshold))



    if X.ndim > 1 and X.shape.eval()[1] > 1:

        

        # (-2, 2) range taken because x are standardised to (0, 1)

        x_temp = np.expand_dims(np.linspace(-2, 2, 100), axis=1) 

        X_temp = np.repeat(x_temp, repeats=X.shape.eval()[1] - 1, axis=1)

        Σ = pm.math.matrix_dot(X_temp, β[:-1])

    else:

        Σ = 0

        

    # values of the decision boundary for the last feature 

    x_k = (lo_threshold - α - Σ)/β[-1] 

#     X_bd = np.stack([X_temp, x_k], axis=1)

    x_k = pm.Deterministic('x_k', x_k)



    # log-odds

    π = pm.Deterministic('π', α + pm.math.dot(X, β))



    # logistic transformation

    ß = pm.Deterministic('ß', pm.math.sigmoid(π))



    # liklihood

    pm.Bernoulli('liklihood', p=ß, observed=y)

    pm.model_to_graphviz().save('log_reg_model_iris.png')

    

pm.model_to_graphviz(log_reg_model)
with log_reg_model:

    trace = pm.sample()

    ppc = pm.sample_posterior_predictive(trace)
sampled_vars = ['β', 'α']

log_reg = az.from_pymc3(model=log_reg_model, trace=trace, posterior_predictive=ppc)
az.plot_trace(log_reg, var_names=sampled_vars);
# for legend

handles = {}



# Decison Boundary

hpd = az.plot_hpd(x=np.linspace(-2, 2, 100), y=log_reg.posterior['x_k'], 

                  credible_interval=0.95, fill_kwargs={'alpha':0.2}, color='C1');



handles['Decision Boundary Spread'] = mpatches.Patch(color=f'C1', alpha=0.2)



# Mean Decison Boundary

mean_x_k = log_reg.posterior['x_k'].mean(dim=['chain', 'draw'])

line = plt.plot(np.linspace(-2, 2, 100), mean_x_k, c='C1')

handles['Mean Decision Boundary'] = line[0]





# Data Points

x1, x2 = np.split(log_reg.constant_data['X'].data, axis=1, indices_or_sections=len(feature))

scatter = plt.scatter(x=x1, y=x2, c=idx);

handles.update(dict(zip(group_names.to_list(), scatter.legend_elements()[0])))



# Labelling

plt.xlabel(f'{feature[0]} Z Score');

plt.ylabel(f'{feature[1]} Z Score');

plt.title('Decision Boundary');

plt.legend(handles=list(handles.values()), labels=list(handles.keys()));