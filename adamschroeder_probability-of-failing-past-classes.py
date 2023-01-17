import pymc3 as pm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import scipy.stats as stats
import theano.tensor as tt

%matplotlib inline
df = pd.read_csv("../input/student-por.csv")
df['failed'] = 0
df.loc[df.failures>0, 'failed'] = 1 # if student had more than zero failures in the past, add 1 to total_fa. column
print("100 current students had failures in past classes")
df.failed.value_counts()
plt.figure(figsize=(10,5))
sns.barplot(df.failed, df.G3)
plt.title("Failed versus average grade")
plt.ylabel("Average final grade")
plt.xlabel("Failed or not in past classes")
plt.grid()
plt.show()
print("Apparently, for students that had past failures in other classes, their avg. final grade in portuguese class ~ 8.5")
plt.scatter(df.G3, df.failed)
plt.ylabel("Failed or Not")
plt.xlabel("Final Grade")
plt.show()
print("We can see that past failures in other classes (1) has some correlation with lower final grade \
in this portuguese class.")
final_grade = np.array(df.G3) # final grade
failed_past_classes = np.array(df.failed)  # failed past classes (1) or not (0)?

# We have to set the values of beta and alpha to 0. The reason for this is that if beta and alpha are very large, 
# they make p equal to 1 or 0. Unfortunately, pm.Bernoulli does not like probabilities of exactly 0 or 1, though 
# they are mathematically well-defined probabilities. So by setting the coefficient values to 0, we set the variable 
# p to be a reasonable starting value. This has no effect on our results, nor does it mean we are including any 
# additional information in our prior. It is simply a computational caveat in PyMC3

with pm.Model() as model:
    # when τ=0.001 (precision), the variance is 1/τ (AKA, σ**2 or std**2), which is 1000
    beta = pm.Normal("beta", mu=0, tau=0.001, testval=0)  
    alpha = pm.Normal("alpha", mu=0, tau=0.001, testval=0)
    p = pm.Deterministic("p", 1.0/(1. + tt.exp((beta*final_grade) + alpha))) # p(fg)= 1/ (1+e**((β*final_grade)+α))
np.random.seed(seed=13)
norm = pm.Normal.dist(mu=0, sd=31.622).random(size=500)
sns.kdeplot(norm)
plt.show()
with model:
    observed = pm.Bernoulli("bernoulli_obs", p, observed=failed_past_classes)
    start = pm.find_MAP()
    step = pm.Metropolis()
    trace = pm.sample(120000, step=step, start=start)
    burned_trace = trace[100000::2]
pm.summary(burned_trace, varnames=['alpha','beta'])
pm.traceplot(burned_trace)
plt.show()
pm.plot_posterior(burned_trace,
                  varnames=['beta','alpha'], 
                  color='#87ceeb')
plt.show()
alpha_samples = burned_trace["alpha"][:, None]  # best to make them 1d
beta_samples = burned_trace["beta"][:, None]

def logistic(x, beta, alpha=0):
    return 1.0 / (1.0 + np.exp(np.dot(beta, x) + alpha)) #same as (1.0/(1. + tt.exp(beta*temperature + alpha)))

t = np.linspace(final_grade.min() - 1, final_grade.max()+1, 50)[:, None]
p_t = logistic(t.T, beta_samples, alpha_samples) #t.T changes shape from (50,1) to (1,50)

mean_prob_t = p_t.mean(axis=0)
from scipy.stats.mstats import mquantiles

plt.figure(figsize=(12.5, 4))

# vectorized bottom and top 2.5% quantiles for "confidence interval"
qs = mquantiles(p_t, [0.025, 0.975], axis=0)
plt.fill_between(t[:, 0], *qs, alpha=0.7,
                 color="#7A68A6")

plt.plot(t[:, 0], qs[0], label="95% CI", color="#7A68A6", alpha=0.7)

plt.plot(t, mean_prob_t, lw=1, ls="--", color="k",
         label="average posterior \nprobability of past failure")

plt.xlim(t.min(), t.max())
plt.ylim(-0.02, 1.02)
plt.legend(loc="lower left")
plt.scatter(final_grade, failed_past_classes, color="k", s=50, alpha=0.5)
plt.xlabel("Final grade")

plt.ylabel("Probability estimate")
plt.title("Posterior probability estimates given final grade")
plt.grid()
plt.show()
plt.figure(figsize=(12.5, 2.5))

prob_6 = logistic(6, beta_samples, alpha_samples)

# plt.xlim(0.995, 1) # expand this if temperature is higher
plt.hist(prob_6, bins=1000, normed=True, histtype='stepfilled')
plt.title("Posterior distribution of probability of past failures, given $final grade = 6$")
plt.xlabel("probability of past failures in other classes occurring with this student");