import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import expit, logit
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

np.random.seed(1)
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', None)
# load
df = pd.read_pickle('../input/passnyc-model-1/schools2017.pkl')

# select columns
df = df[[        
    'Mean Scale Score - ELA',
    '% Level 2 - ELA',
    '% Level 3 - ELA',
    '% Level 4 - ELA',
    'Mean Scale Score - Math',
    '% Level 2 - Math',
    '% Level 3 - Math',
    '% Level 4 - Math',
    
    '# Students in HS Admissions',
    '# SHSAT Testers',
    '% SHSAT Testers',
]].copy()
print(df.shape[0], "schools")

# drop schools with missing test data
df = df[df.loc[:, 'Mean Scale Score - ELA':'% Level 4 - Math'].notnull().all(axis=1)]
print(df.shape[0], "schools after dropping missing test data")

# schools with 0-5 SHSAT testers have this value set to NaN
applicantsok = df['# SHSAT Testers'].notnull()

# convert percentages to the (0, 1) range
bad_pct_c = [
    '% Level 2 - ELA',
    '% Level 3 - ELA',
    '% Level 4 - ELA',
    '% Level 2 - Math',
    '% Level 3 - Math',
    '% Level 4 - Math',
]
df.loc[:, bad_pct_c] = df.loc[:, bad_pct_c] / 100.0

# standardize score columns (algorithm stability)
score_c = ['Mean Scale Score - ELA', 'Mean Scale Score - Math']
df.loc[:, score_c] = scale(df.loc[:, score_c])
base_df = df[[  # explanatory variables
    'Mean Scale Score - ELA',
    '% Level 2 - ELA',
    '% Level 3 - ELA',
    '% Level 4 - ELA',
    'Mean Scale Score - Math',
    '% Level 2 - Math',
    '% Level 3 - Math',
    '% Level 4 - Math',
]]

n_components = 4  # from local cross-validation
pca = PCA(n_components)
transformed = pca.fit_transform(base_df)
transformed = pd.DataFrame(transformed, index=base_df.index, columns=["PC{}".format(i+1) for i in range(n_components)])
transformed.head()

inputs = transformed
inputs.insert(0, 'Constant', 1.0)
#data
inputs_fit = inputs[applicantsok]
outputs_fit = logit(df['% SHSAT Testers'][applicantsok])
inputs_predict = inputs

# fit
model = sm.RLM(outputs_fit, inputs_fit, M=sm.robust.norms.HuberT())
results = model.fit()

# predict
predictions = model.predict(results.params, exog=inputs_predict)
predictions = pd.Series(predictions, index=inputs_predict.index)
predictions.name = 'Predictions'
xs = expit(predictions[applicantsok])  # expit function is the inverse of the logit
ys = expit(outputs_fit)

plt.figure(figsize=(12, 8))
plt.scatter(xs, ys, s=10)
plt.plot([0, 1], [0, 1], '--')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.title("Percentage of SHSAT Applicants")
plt.xlabel("Expected Percentage")
plt.ylabel("Actual Precentage");
def from_counts(shsat_counts, hs_counts):
    return pd.DataFrame({
        'logit': logit(shsat_counts / hs_counts),
        'pct': shsat_counts / hs_counts,
        'cnt': shsat_counts,
    })

def from_logits(shsat_logits, hs_counts):
    return pd.DataFrame({
        'logit': shsat_logits,
        'pct': expit(shsat_logits),
        'cnt': expit(shsat_logits) * hs_counts,
    })
# actual values  ---

# schools with 0 to 5 applicants
hs_counts = df[~applicantsok]['# Students in HS Admissions']
max_v = from_counts(5, hs_counts)

# schools with 6 or more applicants
shsat_counts = df[applicantsok]['# SHSAT Testers']
hs_counts = df[applicantsok]['# Students in HS Admissions']
pontual_v = from_counts(shsat_counts, hs_counts)


# expected values  ---

shsat_logits = predictions
hs_counts = df['# Students in HS Admissions']
expected_v = from_logits(shsat_logits, hs_counts)


# differences  ---

max_diff = (max_v - expected_v).dropna()
pontual_diff = (pontual_v - expected_v).dropna()


# join everything  ---

everything = pd.DataFrame({
    'Actual #': pd.concat([max_v, pontual_v])['cnt'],
    'Estimated #': expected_v['cnt'],
    'Difference #': pd.concat([max_diff, pontual_diff])['cnt'],
            
    'Actual %': pd.concat([max_v, pontual_v])['pct'],
    'Estimated %': expected_v['pct'],
    'Difference %': pd.concat([max_diff, pontual_diff])['pct'],
})
everything['Difference Ratio'] = everything['Difference #'] / everything['Actual #']

everything = everything.sort_index()
everything.head().style. \
    format('{:.0f}', subset=pd.IndexSlice[:, 'Actual #':'Difference #']). \
    format('{:.0%}', subset=pd.IndexSlice[:, 'Actual %':'Difference %']). \
    format('{:.2f}', subset=['Difference Ratio'])
# load dataframes
# 'vis' are easier to visualize

schools = pd.read_pickle('../input/passnyc-model-1/schools2017.pkl')
schools = schools[schools.loc[:, 'Mean Scale Score - ELA':'% Level 4 - Math'].notnull().all(axis=1)]  # drop na
score_c = ['Mean Scale Score - ELA', 'Mean Scale Score - Math']  # standardize columns
schools.loc[:, score_c] = scale(schools.loc[:, score_c])  # standardize columns

schools_vis = pd.read_pickle('../input/passnyc-model-1/schools2017_vis.pkl')
results_vis = pd.read_pickle('../input/passnyc-model-1/model1_results_vis.pkl')
vis = schools_vis.join(results_vis, how='inner')
index = everything.sort_values('Difference Ratio').index
vis.reindex(index).head()
vis.loc['84M353']