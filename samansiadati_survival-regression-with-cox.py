!pip install lifelines
from lifelines.datasets import load_rossi
rossi = load_rossi()
rossi
from lifelines import CoxPHFitter
cph = CoxPHFitter()
cph.fit(rossi, duration_col='week', event_col='arrest')
cph.print_summary()  # access the individual results using cph.summary
cph.plot()
cph.plot_partial_effects_on_outcome(covariates='prio', values=[0, 2, 4, 6, 8, 10], cmap='coolwarm')
cph.fit(rossi, duration_col='week', event_col='arrest', formula="fin + wexp + age * prio")
cph.print_summary()
X = rossi
cph.predict_survival_function(X)
cph.predict_median(X)
cph.predict_partial_hazard(X)
cph_spline = CoxPHFitter(baseline_estimation_method="spline", n_baseline_knots=5)
cph_spline.fit(rossi, 'week', event_col='arrest')
cph_semi = CoxPHFitter().fit(rossi, 'week', event_col='arrest')
cph_piecewise = CoxPHFitter(baseline_estimation_method="piecewise", breakpoints=[20, 35]).fit(rossi, 'week', event_col='arrest')

ax = cph_spline.baseline_cumulative_hazard_.plot()
cph_semi.baseline_cumulative_hazard_.plot(ax=ax, drawstyle="steps-post")
cph_piecewise.baseline_cumulative_hazard_.plot(ax=ax)
from lifelines import WeibullAFTFitter
aft = WeibullAFTFitter()
aft.fit(rossi, duration_col='week', event_col='arrest')

aft.print_summary(3)  # access the results using aft.summary
print(aft.median_survival_time_)
print(aft.mean_survival_time_)
from matplotlib import pyplot as plt
wft = WeibullAFTFitter().fit(rossi, 'week', 'arrest', ancillary=True)
wft.plot()
import numpy as np
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

times = np.arange(0, 100)
wft_model_rho = WeibullAFTFitter().fit(rossi, 'week', 'arrest', ancillary=True, timeline=times)
wft_model_rho.plot_partial_effects_on_outcome('prio', range(0, 16, 3), cmap='coolwarm', ax=ax[0])
ax[0].set_title("Modelling rho_")

wft_not_model_rho = WeibullAFTFitter().fit(rossi, 'week', 'arrest', ancillary=False, timeline=times)
wft_not_model_rho.plot_partial_effects_on_outcome('prio', range(0, 16, 3), cmap='coolwarm', ax=ax[1])
ax[1].set_title("Not modelling rho_");
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 4))

# modeling rho == solid line
wft_model_rho.plot_partial_effects_on_outcome('prio', range(0, 16, 5), cmap='coolwarm', ax=ax, lw=2, plot_baseline=False)

# not modeling rho == dashed line
wft_not_model_rho.plot_partial_effects_on_outcome('prio', range(0, 16, 5), cmap='coolwarm', ax=ax, ls='--', lw=2, plot_baseline=False)

ax.get_legend().remove()
X = rossi.loc[:10]

aft.predict_cumulative_hazard(X, ancillary=X)
aft.predict_survival_function(X, ancillary=X)
aft.predict_median(X, ancillary=X)
aft.predict_percentile(X, p=0.9, ancillary=X)
aft.predict_expectation(X, ancillary=X)
aft_with_elastic_penalty = WeibullAFTFitter(penalizer=1e-4, l1_ratio=1.0)
aft_with_elastic_penalty.fit(rossi, 'week', 'arrest')
aft_with_elastic_penalty.predict_median(rossi)

aft_with_elastic_penalty.print_summary(columns=['coef', 'exp(coef)'])
from lifelines import LogLogisticAFTFitter
from lifelines import LogNormalAFTFitter
llf = LogLogisticAFTFitter().fit(rossi, 'week', 'arrest')
lnf = LogNormalAFTFitter().fit(rossi, 'week', 'arrest')
from lifelines import GeneralizedGammaRegressionFitter
df = load_rossi()
df['Intercept'] = 1.

# this will regress df against all 3 parameters
ggf = GeneralizedGammaRegressionFitter(penalizer=1.).fit(df, 'week', 'arrest')
ggf.print_summary()
# If we want fine control over the parameters <-> covariates.
# The values in the dict become can be formulas, or column names in lists:
regressors = {
    'mu_': rossi.columns.difference(['arrest', 'week']),
    'sigma_': ["age", "Intercept"],
    'lambda_': 'age + 1',
}

ggf = GeneralizedGammaRegressionFitter(penalizer=0.0001).fit(df, 'week', 'arrest', regressors=regressors)
ggf.print_summary()
from lifelines import LogLogisticAFTFitter, WeibullAFTFitter, LogNormalAFTFitter
wf = WeibullAFTFitter().fit(rossi, 'week', 'arrest')
print(llf.AIC_)  # 1377.877
print(lnf.AIC_)  # 1384.469
print(wf.AIC_)   # 1377.833, slightly the best model.
# with some heterogeneity in the ancillary parameters
ancillary = rossi[['prio']]
llf = LogLogisticAFTFitter().fit(rossi, 'week', 'arrest', ancillary=ancillary)
lnf = LogNormalAFTFitter().fit(rossi, 'week', 'arrest', ancillary=ancillary)
wf = WeibullAFTFitter().fit(rossi, 'week', 'arrest', ancillary=ancillary)
print(llf.AIC_) # 1377.89, the best model here, but not the overall best.
print(lnf.AIC_) # 1380.79
print(wf.AIC_)  # 1379.21
from lifelines.datasets import load_diabetes
df = load_diabetes()

df['gender'] = df['gender'] == 'male'

print(df.head())
wf = WeibullAFTFitter().fit_interval_censoring(df, lower_bound_col='left', upper_bound_col='right')
wf.print_summary()
from lifelines import AalenAdditiveFitter
from lifelines.datasets import load_dd
data = load_dd()
data.head()
import patsy
X = patsy.dmatrix('un_continent_name + regime + start_year', data, return_type='dataframe')
X = X.rename(columns={'Intercept': 'baseline'})
print(X.columns.tolist())
aaf = AalenAdditiveFitter(coef_penalizer=1.0, fit_intercept=False)
X['T'] = data['duration']
X['E'] = data['observed']
aaf.fit(X, 'T', event_col='E')
aaf.cumulative_hazards_.head()
aaf.plot(columns=['regime[T.Presidential Dem]', 'baseline', 'un_continent_name[T.Europe]'], iloc=slice(1,15))
ix = (data['ctryname'] == 'Canada') & (data['start_year'] == 2006)
harper = X.loc[ix]
print("Harper's unique data point:")
print(harper)
ax = plt.subplot(2,1,1)
aaf.predict_cumulative_hazard(harper).plot(ax=ax)

ax = plt.subplot(2,1,2)
aaf.predict_survival_function(harper).plot(ax=ax);
rossi = load_rossi().sample(frac=1.0)
train_rossi = rossi.iloc[:400]
test_rossi = rossi.iloc[400:]

cph_l2 = CoxPHFitter(penalizer=0.1, l1_ratio=0.).fit(train_rossi, 'week', 'arrest')
cph_l1 = CoxPHFitter(penalizer=0.1, l1_ratio=1.).fit(train_rossi, 'week', 'arrest')

print(cph_l2.score(test_rossi))
print(cph_l1.score(test_rossi)) # better model
rossi = load_rossi()

cph_l2 = CoxPHFitter(penalizer=0.1, l1_ratio=0.).fit(rossi, 'week', 'arrest')
cph_l1 = CoxPHFitter(penalizer=0.1, l1_ratio=1.).fit(rossi, 'week', 'arrest')

print(cph_l2.AIC_partial_) # lower is better
print(cph_l1.AIC_partial_)
cph = CoxPHFitter()
cph.fit(rossi, duration_col="week", event_col="arrest")

# fours ways to view the c-index:
# method one
cph.print_summary()

# method two
print(cph.concordance_index_)

# method three
print(cph.score(rossi, scoring_method="concordance_index"))

# method four
from lifelines.utils import concordance_index
print(concordance_index(rossi['week'], -cph.predict_partial_hazard(rossi), rossi['arrest']))
from lifelines.datasets import load_regression_dataset
from lifelines.utils import k_fold_cross_validation
regression_dataset = load_regression_dataset()
cph = CoxPHFitter()
scores = k_fold_cross_validation(cph, regression_dataset, 'T', event_col='E', k=3)
print(scores)
#[-2.9896, -3.08810, -3.02747]

scores = k_fold_cross_validation(cph, regression_dataset, 'T', event_col='E', k=3, scoring_method="concordance_index")
print(scores)
# [0.5449, 0.5587, 0.6179]
from lifelines.calibration import survival_probability_calibration
regression_dataset = load_rossi()
cph = CoxPHFitter(baseline_estimation_method="spline", n_baseline_knots=3)
cph.fit(rossi, "week", "arrest")


survival_probability_calibration(cph, rossi, t0=25)
# all regression models can be used here, WeibullAFTFitter is used for illustration
wf = WeibullAFTFitter().fit(rossi, "week", "arrest")

# filter down to just censored subjects to predict remaining survival
censored_subjects = rossi.loc[~rossi['arrest'].astype(bool)]
censored_subjects_last_obs = censored_subjects['week']

# predict new survival function
wf.predict_survival_function(censored_subjects, conditional_after=censored_subjects_last_obs)

# predict median remaining life
wf.predict_median(censored_subjects, conditional_after=censored_subjects_last_obs)