import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy.special import expit, logit
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, median_absolute_error
from sklearn.model_selection import RepeatedKFold

np.random.seed(1)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
# load data

df = pd.read_pickle('../input/district5nyc/d5_full.pkl')
df = df[df.notnull().all(axis=1)]
nschools = df.reset_index()['DBN'].nunique()
print("load {} schools".format(nschools))

df.head()
# load model

print("load model")
with open('../input/district5nyc/full_model.pkl', mode='rb') as f:
    model = pickle.load(f)
# prepare data

def get_inputs(df):
    base_df = df[[  # explanatory variables
        'Charter School?',
        'Percent Asian',
        'Percent Black',
        'Percent Hispanic',
        'Percent Other',
        'Percent English Language Learners',
        'Percent Students with Disabilities',
        'Economic Need Index',
        'Percent of Students Chronically Absent',
        
        'Mean Scale Score - ELA',
        '% Level 2 - ELA',
        '% Level 3 - ELA',
        '% Level 4 - ELA',
        'Mean Scale Score - Math',
        '% Level 2 - Math',
        '% Level 3 - Math',
        '% Level 4 - Math',
    ]]

    # transform the variables (apply the PCA)
    n_components = 8
    pca = PCA(n_components)
    transformed = pca.fit_transform(base_df)
    transformed = pd.DataFrame(transformed, index=base_df.index, columns=["PC{}".format(i+1) for i in range(n_components)])

    # add a constant column (needed for our model with statsmodels)
    inputs = transformed
    inputs.insert(0, 'Constant', 1.0)

    return inputs


def get_outputs(df):
    outputs = logit(df['% SHSAT Testers'])
    return outputs
print("prepare data")

inputs = get_inputs(df)
outputs = get_outputs(df)
# fit first model

results = model.fit()  # full model
predictions = model.predict(results.params, exog=inputs)
predictions = pd.Series(predictions, index=df.index, name='% SHSAT Testers')
mae = median_absolute_error(outputs, predictions)
mse = mean_squared_error(outputs, predictions)

print("Initial fit")
print()
print("Median Absolute Error: {:.4f}".format(mae))
print("Mean Squared Error: {:.4f}".format(mse))
# fit second model

years = df.reset_index()['Year']
years.index = df.index

inputs2 = pd.DataFrame(index=df.index)
inputs2['constant'] = 1.0
inputs2['year_2016'] = (years == 2016).astype(int)
inputs2['prediction'] = predictions

model2 = sm.RLM(outputs, inputs2)
results2 = model2.fit()
predictions2 = results2.fittedvalues
mae = median_absolute_error(outputs, predictions2)
mse = mean_squared_error(outputs, predictions2)

print("Final fit")
print()
print("Median Absolute Error: {:.4f}".format(mae))
print("Mean Squared Error: {:.4f}".format(mse))
pct_residuals = expit(outputs) - expit(predictions2)
pivoted = pct_residuals.reset_index().pivot(index='DBN', columns='Year', values=0)

y1 = 2015
y2 = 2016
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 4))

# calculate diffs, diff mean and diff std
diffs = (pivoted[y2] - pivoted[y1]).dropna()
mean = diffs.mean()
std = diffs.std()

# first plot
ax1.plot([-0.25, 0.25], [-0.25, 0.25], '--', color='gray')
ax1.plot(pivoted[y1], pivoted[y2], '.')
ax1.set_xlabel("Gap in %s" % y1)
ax1.set_ylabel("Gap in %s" % y2)
ax1.set_xlim(-0.25, 0.25)
ax1.set_ylim(-0.25, 0.25)
ax1.set_title("Comparison of the gaps from 2015 to 2016")

# second plot
sns.distplot(diffs, ax=ax2, rug=True)
ax2.set_title("Distribution of Error ({:.2f} ±{:.2f})".format(mean, std))
ax2.set_xlabel("Gap Difference (2015 to 2016)")
ax2.set_ylabel("Count");
df.loc['84M341']
pivoted = pivoted.drop('84M341')  # drop outlier
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 4))

# calculate diffs, diff mean and diff std
diffs = (pivoted[y2] - pivoted[y1]).dropna()
mean = diffs.mean()
std = diffs.std()

# first plot
ax1.plot([-0.25, 0.25], [-0.25, 0.25], '--', color='gray')
ax1.plot(pivoted[y1], pivoted[y2], '.')
ax1.set_xlabel("Gap in %s" % y1)
ax1.set_ylabel("Gap in %s" % y2)
ax1.set_xlim(-0.25, 0.25)
ax1.set_ylim(-0.25, 0.25)
ax1.set_title("Comparison of the gap from 2015 to 2016")

# second plot
sns.distplot(diffs, ax=ax2, rug=True)
ax2.set_title("Distribution of Error ({:.2f} ±{:.2f})".format(mean, std))
ax2.set_xlabel("Gap Difference (2015 to 2016)")
ax2.set_ylabel("Count");