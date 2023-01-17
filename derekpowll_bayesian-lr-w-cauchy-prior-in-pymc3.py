!pip install scikit-misc



import re

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pymc3 as pm

import random

import matplotlib

import plotnine

from plotnine import ggplot, aes, geom_point, geom_jitter, geom_smooth, geom_histogram, geom_line, geom_errorbar, stat_smooth, geom_ribbon

from plotnine.facets import facet_wrap

import seaborn as sns

import theano.tensor as tt

from scipy.special import expit

from scipy.special import logit

from scipy.stats import cauchy

from sklearn.metrics import roc_auc_score

from skmisc.loess import loess

import warnings

warnings.filterwarnings("ignore") # plotnine is causing all kinds of matplotlib warnings





## Define custom functions



invlogit = lambda x: 1/(1 + tt.exp(-x))



def trace_predict(trace, X):

    y_hat = np.apply_along_axis(np.mean, 1, expit(trace['alpha'] + np.dot(X, np.transpose(trace['beta']) )) )

    return(y_hat)





# Define prediction helper function

# for more help see: https://discourse.pymc.io/t/how-to-predict-new-values-on-hold-out-data/2568

def posterior_predict(trace, model, n=1000, progressbar=True):

    with model:

        ppc = pm.sample_posterior_predictive(trace,n, progressbar=progressbar)

    

    return(np.mean(np.array(ppc['y_obs']), axis=0))





## I much prefer the syntax of tidyr gather() and spread() to pandas' pivot() and melt()

def gather( df, key, value, cols ):

    id_vars = [ col for col in df.columns if col not in cols ]

    id_values = cols

    var_name = key

    value_name = value

    return pd.melt( df, id_vars, id_values, var_name, value_name )





def spread( df, index, columns, values ):

    return df.pivot(index, columns, values).reset_index(level=index).rename_axis(None,axis=1)





## define custom plotting functions



def fit_loess(df, transform_logit=False):

    l = loess(df["value"],df["target"])

    l.fit()

    pred_obj = l.predict(df["value"],stderror=True)

    conf = pred_obj.confidence()

    

    yhat = pred_obj.values

    ll = conf.lower

    ul = conf.upper

    

    df["loess"] = np.clip(yhat,.001,.999)

    df["ll"] = np.clip(ll,.001,.999)

    df["ul"] = np.clip(ul,.001,.999)

    

    if transform_logit:

        df["logit_loess"] = logit(df["loess"])

        df["logit_ll"] = logit(df["ll"])

        df["logit_ul"] = logit(df["ul"])

    

    return(df)





def plot_loess(df, features):

    

    z = gather(df[["id","target"]+features], "feature", "value", features)

    z = z.groupby("feature").apply(fit_loess, transform_logit=True)

    z["feature"] = pd.to_numeric(z["feature"])



    plot = (

        ggplot(z, aes("value","logit_loess",ymin="logit_ll",ymax="logit_ul")) + 

        geom_point(alpha=.5) + 

        geom_line(alpha=.5) + 

        geom_ribbon(alpha=.33) + 

        facet_wrap("~feature")

    )

    return(plot)





## Load data



df = pd.read_csv("../input/train.csv")

y = np.asarray(df.target)

X = np.array(df.ix[:, 2:302])

df2 = pd.read_csv('../input/test.csv')

df2.head()

X2 = np.array(df2.ix[:, 1:301])



print("training shape: ", X.shape)

print("test shape: ", X2.shape)
random.seed(432532) # comment out for new random samples



rand_feats = [str(x) for x in random.sample(range(0,300), 12)]

dfp = gather(df[["id","target"]+rand_feats], "feature", "value", rand_feats)

sns.set(style="ticks")



sns.pairplot(spread(dfp, "id", "feature", "value").drop("id",1))
plotnine.options.figure_size = (12,9)



random.seed(432532) # comment out for new random samples



rand_feats = [str(x) for x in random.sample(range(0,300), 12)]

plot_loess(df,rand_feats)
def make_model(X, y, cauchy_scale):

    model = pm.Model()



    with model:



        # Priors for unknown model parameters

        alpha = pm.Normal('alpha', mu=0, sd=3)

        beta = pm.Cauchy('beta', alpha=0, beta=cauchy_scale, shape=X.shape[1])

        mu = pm.math.dot(X, beta)



        # Likelihood (sampling distribution) of observations

        y_obs = pm.Bernoulli('y_obs', p=invlogit(alpha + mu),  observed=y)

    

    model.name = "linear_c_"+str(cauchy_scale)

    return(model)



shape_par = .0175



prior_df = pd.DataFrame({"value": np.arange(-2,2,.01)})

prior_df = prior_df.assign(dens = cauchy.pdf(prior_df["value"],0,shape_par))

cauchy_samples = cauchy.rvs(0, shape_par, 10000)



print("percent non-zero coefs :", 1-np.mean((cauchy_samples < .1) & (cauchy_samples > -.1)))



plotnine.options.figure_size = (8,6)

ggplot(prior_df, aes(x="value", y="dens")) + geom_line()

cauchy_scale_pars = [.01,.015,.0175,.020,.0225,.025,.03,.04,.05, .1]



models = []

traces = []

model_dict = dict()



for scale_val in cauchy_scale_pars:

    model = make_model(X,y, scale_val)

    with model:

        trace = pm.sample(1000,

                          tune = 500,

                          init= "adapt_diag", 

                          cores = 1, 

                          progressbar = False, 

                          chains = 2, 

                          nuts_kwargs=dict(target_accept=.95),

                          random_seed = 12345

                         )

    

    traces.append(trace)

    models.append(model)

    model_dict[model] = trace

    
# compare models with LOO

comp = pm.stats.compare(model_dict, ic="LOO", method='BB-pseudo-BMA')



# generate posterior predictions for original data

for i in range(0,len(traces)):

    y_hat = trace_predict(traces[i], X)

    print("scale = ",cauchy_scale_pars[i],", training AUCROC:",roc_auc_score(y,y_hat))

    

# print comparisons

comp
# can throw out .1 and .05 as way out-of-bounds

comp_abridged = pm.stats.compare(dict(zip(models[0:-2], traces[0:-2])), ic="LOO", method='BB-pseudo-BMA')

comp_abridged
model1 = models[5]

trace1 = traces[5]

model1.name
coefs = pm.summary(trace1, varnames=["beta"], alpha=.10)



top_coefs = (coefs

             .assign(abs_est = abs(coefs["mean"]), non_zero = np.sign(coefs["hpd_5"]) == np.sign( coefs["hpd_95"]))

             .sort_values("abs_est", ascending=False)

            ).head(20)



top_coefs
plotnine.options.figure_size = (12,9)



regex = re.compile("__(.*)")

top_feats = [regex.search(x)[1] for x in list(top_coefs.index)]



plot_loess(df, top_feats)
# # generate test predictions and create submission file

def generate_submission(trace, file_suffix=""):



    test_predictions = trace_predict(trace, X2)



    submission  = pd.DataFrame({'id':df2.id, 'target':test_predictions})

    submission.to_csv("submission_"+file_suffix+".csv", index = False)

    return(None)



for model in model_dict.keys():

    generate_submission(model_dict[model], model.name)
# grab potentially reasonable models

comp_MA = pm.stats.compare(dict(zip(models[0:-3], traces[0:-3])), ic="LOO", method='BB-pseudo-BMA')



# do prediction from averaged model

ppc_w = pm.sample_posterior_predictive_w(traces[0:-3], 4000, [make_model(X2,np.zeros(19750),c) for c in cauchy_scale_pars[0:-3]],

                        weights=comp_MA.weight.sort_index(ascending=True),

                        progressbar=True)

                        

y_hatMA = np.mean(np.array(ppc_w['y_obs']), axis=0)

submission  = pd.DataFrame({'id':df2.id, 'target':y_hatMA})

submission.to_csv('submission_MA.csv', index = False)
non_lin_feats = ["276","91","240","246","253","255","268","118","240","7","167","65","33"]



plot_loess(df, non_lin_feats)
def make_polymodel(X, y):

    

    with pm.Model() as model:

        

        # Priors for unknown model parameters

        alpha = pm.Normal('alpha', mu=0, sd=3)

        beta1 = pm.Cauchy('beta', alpha=0, beta=.07, shape=X.shape[1])

        beta2 = pm.Cauchy('beta^2', alpha=0, beta=.07, shape=X.shape[1])

        beta3 = pm.Cauchy('beta^3', alpha=0, beta=.07, shape=X.shape[1])

        

        mu1 = pm.math.dot(X, beta1)

        mu2 = pm.math.dot(np.power(X,2), beta2)

        mu3 = pm.math.dot(np.power(X,3), beta3)

        

        p = invlogit(alpha + mu1 + mu2 + mu3)

        

        # Likelihood (sampling distribution) of observations

        y_obs = pm.Bernoulli('y_obs', p=p,  observed=y)

        

    return(model)


