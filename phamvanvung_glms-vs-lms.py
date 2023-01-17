import numpy as np

import pandas as pd
# Generate uniform random stress level for 200 days

def stressLevels():

    return np.random.randint(1, 10, 200)

# Generate uniform random sleep qualities from previous nights

def sleptQualities():

    return np.random.randint(1, 10, 200)

# Generate randomly (50/50 chance) if that day is work day

def workDays():

    return np.array([1 if x >= 0.5 else 0 for x in np.random.random(200)])

# Generate lambda => expected bernoully parameber for an yes

def lambdaVal(stressLevel, sleepQuality, workDay):

    x = (stressLevel - sleepQuality - workDay * 2)

    return (1/(1 + np.exp(-x)))/2.0
# Generate simulated data

stresses = stressLevels()

sleeps = sleptQualities()

works = workDays()

lambdas = np.array([lambdaVal(stresses[i], sleeps[i], works[i]) for i in range(200)])
fig = pd.DataFrame(lambdas).hist()
# Generate number of coffees that you might drink on a day => maximum is 5

coffees = np.array([np.random.binomial(5, ld) for ld in lambdas])
fig = pd.DataFrame(coffees).hist()
# It is observable from the previous picture that number of coffees (outcomes) follow a bernoulli distribution => Linear regression will not work so we will try and see

from sklearn.linear_model import LinearRegression

df = pd.DataFrame()

df['stresses'] = stresses

df['sleeps'] = sleeps

df['works'] = works

X = df.values

y = coffees
model = LinearRegression()

model.fit(X, y)
r_sq = model.score(X, y)

print("coefficient of determination:", r_sq)
print('Intercept: ', model.intercept_)

print('slope: ', model.coef_)
predicts = model.predict(X)
fig = pd.DataFrame(predicts).hist()
import statsmodels.api as sm
binomial_model = sm.GLM(y, X, family=sm.families.Poisson())
binomial_results = binomial_model.fit()

print(binomial_results.summary())
print("Parameters; ", binomial_results.params)
yhat = binomial_results.mu
yhat
pd.DataFrame(yhat).hist()