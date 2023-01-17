#Importing necessary libraries

import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
from numpy.random import seed
from numpy.random import randn
import seaborn as sns
from PyAstronomy import pyasl

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import ExtraTreesClassifier

sns.set_style('whitegrid')

%matplotlib inline
a=[1,2,3,4,5,6]
computed_mean = np.mean(a)
print(f"The computed mean is {computed_mean}")

interval = stats.t.interval(0.95, len(a)-1, loc=np.mean(a), scale=stats.sem(a))
print(f"The mean has a confidence limit of {interval}")
b = [2.2,2.21,2.22,2.23,2.24,2.25]
computed_mean = np.mean(b)
print(f"The computed mean is {computed_mean}")

interval = stats.t.interval(0.95, len(b)-1, loc=np.mean(b), scale=stats.sem(b))
print(f"The mean has a confidence limit of {interval}")
data1 = np.random.normal(0, 1, size=50)
data2 = np.random.normal(2, 1, size=50)
true_mu = 0
# Checking for data1
onesample_results = stats.ttest_1samp(data1, true_mu)

onesample_results
twosample_results = stats.ttest_ind(data1, data2)
twosample_results
x1 = [0.0571, 0.0813, 0.0831, 0.0976, 0.0817, 0.0859, 0.0735, 0.0659, 0.0923, 0.0836]
x2 = [0.0873, 0.0662, 0.0672, 0.0819, 0.0749, 0.0649, 0.0835, 0.0725]
x3 = [0.0974, 0.1352, 0.0817, 0.1016, 0.0968, 0.1064, 0.105]
x4 = [0.1033, 0.0915, 0.0781, 0.0685, 0.0677, 0.0697, 0.0764, 0.0689]
x5 = [0.0703, 0.1026, 0.0956, 0.0973, 0.1039, 0.1045]

stats.f_oneway(x1, x2, x3, x4, x5)
x1 = [0.0571, 0.0813, 0.0831, 0.0976, 0.0817, 0.0859, 0.0735, 0.0659, 0.0923, 0.0836]
x2 = [0.0873, 0.0662, 0.0672, 0.0819, 0.0749, 0.0649, 0.0835, 0.0725]
x3 = [0.0974, 0.1352, 0.0817, 0.1016, 0.0968, 0.1064, 0.105]
x4 = [0.1033, 0.0915, 0.0781, 0.0685, 0.0677, 0.0697, 0.0764, 0.0689]
x5 = [0.0703, 0.1026, 0.0956, 0.0973, 0.1039, 0.1045]
stats.bartlett(x1, x2, x3, x4, x5)
stats.levene(x1, x2, x3, x4, x5)
# Dummy data
x = np.random.normal(0, 2, 10000)   # create random values based on a normal distribution
# The histogram plot
pd.Series(x).hist()
plt.title('Gaussian Distribution')
print( 'Excess kurtosis of normal distribution (should be 0): {}'.format( stats.kurtosis(x) ))
print( 'Skewness of normal distribution (should be 0): {}'.format( stats.skew(x) ))
# Dummy data
weibull_x = np.random.weibull(10., 10000)   # create random values based on a weibull distribution.
# The histogram plot
pd.Series(weibull_x).hist()
plt.title('Weibull Distribution')
print( 'Excess kurtosis of weibull distribution: {}'.format( stats.kurtosis(weibull_x) ))
print( 'Skewness of weibull distribution: {}'.format( stats.skew(weibull_x) ))
# seed the random number generator
seed(1)
# generate univariate observations
data = 5 * randn(100) + 50
# summarize
print('mean=%.3f stdv=%.3f' % (np.mean(data), np.std(data)))
plt.hist(data);
from statsmodels.graphics.gofplots import qqplot
qqplot(data, line='s');
stat, p = stats.shapiro(data)
print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpret
alpha = 0.05
if p > alpha:
    print('Sample looks Gaussian (fail to reject H0)')
else:
    print('Sample does not look Gaussian (reject H0)')
stat, p = stats.normaltest(data)
print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpret
alpha = 0.05
if p > alpha:
    print('Sample looks Gaussian (fail to reject H0)')
else:
    print('Sample does not look Gaussian (reject H0)')
result = stats.anderson(data)
print('Statistic: %.3f' % result.statistic)
p = 0
for i in range(len(result.critical_values)):
    sl, cv = result.significance_level[i], result.critical_values[i]
    if result.statistic < result.critical_values[i]:
        print('%.3f: %.3f, data looks normal (fail to reject H0)' % (sl, cv))
    else:
        print('%.3f: %.3f, data does not look normal (reject H0)' % (sl, cv))
observed_values=np.array([18,21,16,7,15])
expected_values=np.array([22,19,44,8,16])

stats.chisquare(observed_values, f_exp=expected_values)
np.random.seed(12345678)  #fix random seed to get the same result
n1 = 200  # size of first sample
n2 = 300  # size of second sample
rvs1 = stats.norm.rvs(size=n1, loc=0., scale=1)
rvs2 = stats.norm.rvs(size=n2, loc=0.5, scale=1.5)
stats.ks_2samp(rvs1, rvs2)
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

rng = np.random.RandomState(42)

# Generate train data
X = 0.3 * rng.randn(100, 2)
X_train = np.r_[X + 2, X - 2]

# Generate some regular novel observations
X = 0.3 * rng.randn(20, 2)
X_test = np.r_[X + 2, X - 2]

# Generate some abnormal novel observations
X_outliers = rng.uniform(low=-4, high=4, size=(20, 2))

# fit the model
clf = IsolationForest(max_samples=100,random_state=rng)
clf.fit(X_train)
y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)
y_pred_outliers = clf.predict(X_outliers)

# plot the line, the samples, and the nearest vectors to the plane
xx, yy = np.meshgrid(np.linspace(-5, 5, 50), np.linspace(-5, 5, 50))
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.figure(figsize=(15,8))

plt.title("IsolationForest")
plt.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)

b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c='white',
                 s=20, edgecolor='k')
b2 = plt.scatter(X_test[:, 0], X_test[:, 1], c='green',
                 s=20, edgecolor='k')
c = plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c='red',
                s=20, edgecolor='k')
plt.axis('tight')
plt.xlim((-5, 5))
plt.ylim((-5, 5))
plt.legend([b1, b2, c],
           ["training observations",
            "new regular observations", "new abnormal observations"],
           loc="upper left");
from scipy.stats import t, zscore

def grubbs(X, test='two-tailed', alpha=0.05):

    '''
    Performs Grubbs' test for outliers recursively until the null hypothesis is
    true.

    Parameters
    ----------
    X : ndarray
        A numpy array to be tested for outliers.
    test : str
        Describes the types of outliers to look for. Can be 'min' (look for
        small outliers), 'max' (look for large outliers), or 'two-tailed' (look
        for both).
    alpha : float
        The significance level.

    Returns
    -------
    X : ndarray
        The original array with outliers removed.
    outliers : ndarray
        An array of outliers.
    '''
    print("Original data:",X)
    Z = zscore(X, ddof=1)  # Z-score
    N = len(X)  # number of samples

    # calculate extreme index and the critical t value based on the test
    if test == 'two-tailed':
        extreme_ix = lambda Z: np.abs(Z).argmax()
        t_crit = lambda N: t.isf(alpha / (2.*N), N-2)
    elif test == 'max':
        extreme_ix = lambda Z: Z.argmax()
        t_crit = lambda N: t.isf(alpha / N, N-2)
    elif test == 'min':
        extreme_ix = lambda Z: Z.argmin()
        t_crit = lambda N: t.isf(alpha / N, N-2)
    else:
        raise ValueError("Test must be 'min', 'max', or 'two-tailed'")

    # compute the threshold
    thresh = lambda N: (N - 1.) / np.sqrt(N) * \
        np.sqrt(t_crit(N)**2 / (N - 2 + t_crit(N)**2))

    # create array to store outliers
    outliers = np.array([])

    # loop throught the array and remove any outliers
    while abs(Z[extreme_ix(Z)]) > thresh(N):

        # update the outliers
        outliers = np.r_[outliers, X[extreme_ix(Z)]]
        # remove outlier from array
        X = np.delete(X, extreme_ix(Z))
        # repeat Z score
        Z = zscore(X, ddof=1)
        N = len(X)
    print("Cleaned Data",X ,"Outlier:",outliers)
    print("---")
    return X, outliers



# setup some test arrays
X = np.arange(-5, 6)
X1 = np.r_[X, 100]
X2 = np.r_[X, -100]

# test the two-tailed case
Y, out = grubbs(X1)
assert out == 100
Y, out = grubbs(X2)
assert out == -100

# test the max case
Y, out = grubbs(X1, test='max')
assert out == 100
Y, out = grubbs(X2, test='max')
assert len(out) == 0

# test the min case
Y, out = grubbs(X1, test='min')
assert len(out) == 0
Y, out = grubbs(X2, test='min')
assert out == -100
from sklearn.datasets import load_boston
boston = load_boston()
x = boston.data
y = boston.target
columns = boston.feature_names
#create the dataframe
boston_df = pd.DataFrame(boston.data)
boston_df.columns = columns
boston_df.head()
boston_df.shape
sns.boxplot(x=boston_df['DIS'])
fig, ax = plt.subplots(figsize=(16,8))
ax.scatter(boston_df['INDUS'], boston_df['TAX'])
ax.set_xlabel('Proportion of non-retail business acres per town')
ax.set_ylabel('Full-value property-tax rate per $10,000')
plt.show()
z = np.abs(stats.zscore(boston_df))
print(z)
threshold = 3
print(np.where(z > 3))
print(z[55][1])
z_filtered_df= boston_df[(z < 3).all(axis=1)]
z_filtered_df.shape
Q1 = boston_df.quantile(0.25)
Q3 = boston_df.quantile(0.75)
IQR = Q3 - Q1
print(IQR)
filtered_df =boston_df[~((boston_df < (Q1 - 1.5 * IQR))|(boston_df > (Q3 + 1.5 * IQR))).any(axis=1)]
filtered_df.head()
filtered_df.shape
x = np.array([float(x) for x in "-0.25 0.68 0.94 1.15 1.20 1.26 1.26 1.34 1.38 1.43 1.49 1.49 \
          1.55 1.56 1.58 1.65 1.69 1.70 1.76 1.77 1.81 1.91 1.94 1.96 \
          1.99 2.06 2.09 2.10 2.14 2.15 2.23 2.24 2.26 2.35 2.37 2.40 \
          2.47 2.54 2.62 2.64 2.90 2.92 2.92 2.93 3.21 3.26 3.30 3.59 \
          3.68 4.30 4.64 5.34 5.42 6.01".split()])

# Apply the generalized ESD
r = pyasl.generalizedESD(x, 10, 0.05, fullOutput=True)

print("Number of outliers: ", r[0])
print("Indices of outliers: ", r[1])
print("        R      Lambda")
for i in range(len(r[2])):
    print("%2d  %8.5f  %8.5f" % ((i+1), r[2][i], r[3][i]))

# Plot the "data"
plt.plot(x, 'b.')
# and mark the outliers.
for i in range(r[0]):
    plt.plot(r[1][i], x[r[1][i]], 'rp')
plt.show()
# Get some data
x = np.random.normal(0.,0.1,50)

# Introduce outliers
x[27] = 1.0
x[43] = -0.66

# Run distance based outlier detection
r = pyasl.pointDistGESD(x, 5)

print("Number of outliers detected: ", r[0])
print("Indices of these outliers: ", r[1])

plt.plot(x, 'b.')
for i in range(len(r[1])):
    plt.plot(r[1][i], x[r[1][i]], 'rp')
plt.show()
# Generate some "data"
x = np.arange(100)
y = np.random.normal(x*0.067, 1.0, len(x))

# Introduce an outliers
y[14] = -5.0
y[67] = +9.8

# Find outliers based on a linear (deg = 1) fit.
# Assign outlier status to all points deviating by
# more than 3.0 standard deviations from the fit,
# and show a control plot.
iin, iout = pyasl.polyResOutlier(x, y, deg=1, stdlim=3.0, controlPlot=True)

# What about the outliers
print("Number of outliers: ", len(iout))
print("Indices of outliers: ", iout)

# Remove outliers
xnew, ynew = x[iin], y[iin]

# Plot result (outlier in red)
plt.plot(x, y, 'r.')
plt.plot(xnew, ynew, 'bp');
iin, iout = pyasl.slidingPolyResOutlier(x, y, 20, deg=1, stdlim=3.0, controlPlot=True)

# What about the outliers
print("Number of outliers: ", len(iout))
print("Indices of outliers: ", iout)

# Remove outliers
xnew, ynew = x[iin], y[iin]

# Plot result (outlier in red)
plt.plot(x, y, 'r.')
plt.plot(xnew, ynew, 'bp')

dataframe = pd.read_csv('../input/diabetes.csv')
arr = dataframe.values
X = arr[:,0:8]
y = arr[:,8]
# feature extraction
test = SelectKBest(score_func=chi2, k=4)
fit = test.fit(X, y)
# summarize scores
np.set_printoptions(precision=3)
print(fit.scores_)
features = fit.transform(X)
# summarize selected features
print(features[0:5,:])
dataframe[['Glucose','Insulin','BMI','Age']].head()
X = arr[:,0:8]
y = arr[:,8]
model = LogisticRegression()
rfe = RFE(model, 3)
fit = rfe.fit(X, y)
print("Num Features:", fit.n_features_)
print("Selected Features:", fit.support_)
print("Feature Ranking: %s", fit.ranking_)
# The best 3 features
dataframe.iloc[:,:-1].columns[fit.support_]
X = arr[:,0:8]
y = arr[:,8]
# feature extraction
pca = PCA(n_components=3)
fit = pca.fit(X)
# summarize components
print("Explained Variance:", fit.explained_variance_ratio_)
print(fit.components_)
X = arr[:,0:8]
y = arr[:,8]
# feature extraction
lda = LDA(n_components=3)
fit = lda.fit(X,y)
# summarize components
print("Explained Variance:", fit.explained_variance_ratio_)
X = arr[:,0:8]
y = arr[:,8]
# feature extraction
model = ExtraTreesClassifier()
model.fit(X, y)
print(model.feature_importances_)
dataframe.columns