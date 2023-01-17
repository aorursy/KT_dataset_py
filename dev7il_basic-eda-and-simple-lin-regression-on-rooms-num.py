%matplotlib inline
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import seaborn as sns
plt.rcParams['figure.figsize'] = [16, 12]
df = pd.read_csv("../input/train.csv")
df.head()
df = df.set_index('ID')
df.head()
# Check if there are null values (e.g. NaN)
df.isnull().values.any()
# Check how big is dataset, how many and of what type features it has, what is target etc.
df.info()
df.describe()
df['medv_bins'] = pd.cut(df.medv, bins=5, include_lowest=True)
df.medv_bins.head()
df['lstat_bins'] = pd.cut(df.lstat, bins=[0, 7, 17, 38], labels=['richest', 'ordinary', 'poorest'], include_lowest=True)
df.lstat_bins.head()
f, (ax_viol, ax_box) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.4, .6)})

sns.boxplot(x='medv_bins', y='crim', data=df, ax=ax_viol)
sns.swarmplot(x='medv_bins', y='crim', hue='lstat_bins', data=df, ax=ax_box)

ax_viol.set(xlabel='')
f, (ax_viol, ax_box) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.4, .6)})

sns.boxplot(x='medv_bins', y='crim', data=df[df.crim < 27], ax=ax_viol)
sns.swarmplot(x='medv_bins', y='crim', hue='lstat_bins', data=df[df.crim < 27], ax=ax_box)

ax_viol.set(xlabel='')
corr = df.corr()['medv'].abs().sort_values(ascending=False)
corr[1:6]
sns.pairplot(data=df, x_vars=corr.index[1:6], y_vars='medv')
sns.scatterplot(x='rm', y='medv', data=df)
f, (ax_rm, ax_medv) = plt.subplots(ncols=2)
sns.boxplot(y='rm', data=df, ax=ax_rm)
sns.boxplot(y='medv', data=df, ax=ax_medv)
# Remove outliers...
# ...from medv
medv_Q1 = df.medv.quantile(0.25)
medv_Q3 = df.medv.quantile(0.75)
medv_IQR = medv_Q3 - medv_Q1
medv_outliers = (df.medv < (medv_Q1 - 1.5 * medv_IQR)) | (df.medv > (medv_Q3 + 1.5 * medv_IQR))

# ...from rm
rm_Q1 = df.rm.quantile(0.25)
rm_Q3 = df.rm.quantile(0.75)
rm_IQR = rm_Q3 - rm_Q1
rm_outliers = (df.rm < (rm_Q1 - 1.5 * rm_IQR)) | (df.rm > (rm_Q3 + 1.5 * rm_IQR))

df['outlier'] = medv_outliers | rm_outliers
sns.scatterplot(x='rm', y='medv', hue='outlier', data=df)

df_nooutliers = df[df.outlier == False]
# Two dims: number of rooms feature and bias
x = df_nooutliers.rm.values
y = df_nooutliers.medv.values

X = np.ones((df_nooutliers.shape[0], 2))
X[:, 0] = x # Merge biases with features

N = len(x)
print('Number of examples (N): ', N)
# Where it comes from?! See in ISL book, chapter 3!
B = inv(X.T @ X) @ X.T @ y
print("    Slope: ", B[0])
print("Intercept: ", B[1])
# Scatter data points
ax = sns.scatterplot(x='rm', y='medv', data=df_nooutliers)

# Plot regression line
X_ = np.array([
    [df_nooutliers.rm.min() - .5, 1],
    [df_nooutliers.rm.max() + .5, 1]
])
y_ = X_ @ B
ax.plot(X_[:, 0], y_, color='r')
y_ = X @ B                          # Predictions
e = y - y_                          # Residuals
rss = e @ e.T                       # Residual sum of squares
rse = np.sqrt(rss/(N-2))            # Residual standard error
se = rse * np.sqrt(inv(X.T @ X))    # Standard error
se = np.array([se[0, 0], se[1, 1]])

print("    Slope std. error: ", se[0])
print("Intercept std. error: ", se[1])
# How to understand this? See: https://stattrek.com/estimation/confidence-interval.aspx#sixsteps
conf_level = 0.95
alpha = 1 - conf_level
prob = 1 - alpha/2
critical_value = stats.t.ppf(prob, N - 2) # N - 2 degrees of freedom
margin_of_error = critical_value * se
conf_interval = B - margin_of_error, B + margin_of_error

print("Confidence interval lower bound: ", conf_interval[0])
print("Confidence interval upper bound: ", conf_interval[1])
t = B[0] / se[0]
p = 1 - stats.t.cdf(t, N - 2)
print("p-value: ", p, " Statistical significance:", p < 0.05)
print(stats.linregress(X[:, 0], y))
print("    Slope conf. interval", stats.t.interval(conf_level, N-2, loc=B[0], scale=se[0]))
print("Intercept conf. interval", stats.t.interval(conf_level, N-2, loc=B[1], scale=se[1]))
x_mean = np.mean(x)

sxx = (x - x_mean) @ (x - x_mean).T
se_x = rse * np.sqrt(1/N + (x - x_mean)**2/sxx)

margin_of_error_x = critical_value * se_x
upper_confidence_band = y_ + margin_of_error_x
lower_confidence_band = y_ - margin_of_error_x
# Scatter data points
ax = sns.scatterplot(x='rm', y='medv', data=df_nooutliers, s=60)

# Plot regression line
ax.plot(np.sort(x), np.sort(y_), color='r')

# Draw confidence bands
ax.fill_between(np.sort(x), np.sort(upper_confidence_band), np.sort(lower_confidence_band), color='r', alpha=0.2)
sns.regplot(x='rm', y='medv', data=df_nooutliers, line_kws={"color": 'r'})
# Load data frame
df_test = pd.read_csv("../input/test.csv")
df_test = df_test.set_index('ID')

# Create features matrix
X_test = np.ones((df_test.shape[0], 2))
X_test[:, 0] = df_test['rm'].values # Merge biases with features

# Evaluate model!
df_test['medv'] = X_test @ B
df_test.head()
# Save submission
df_test.to_csv('simple_lin_reg.csv', columns=['medv'])