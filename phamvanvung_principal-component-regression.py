import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from scipy.signal import savgol_filter

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler

from sklearn import linear_model

from sklearn.model_selection import cross_val_predict

from sklearn.metrics import mean_squared_error, r2_score
data = pd.read_csv("../input/peach_spectrabrixvalues.csv")
data.head()
# Get the data

X = data.values[:, 1:]

y = data['Brix']

wl = np.arange(1100, 2300, 2) # wavelengths

# Plot absorbance spectra

with plt.style.context('ggplot'):

    plt.plot(wl, X.T)

    plt.xlabel("Wavelength (nm)")

    plt.ylabel('Absorbance')

plt.show()

# Step 1: PCA on input data

# Define the PCA object

pca = PCA()

# Preprocessing (1): First derivative

d1X = savgol_filter(X, 25, polyorder=5, deriv=1)
# number of components

pc = 3

# Preprocessing (2): Standardize features by removing the mean and scaling to unit variance

Xstd = StandardScaler().fit_transform(d1X[:, :])



# Run PCA producing the reduced variable Xreg and select first PC components

Xreg = pca.fit_transform(Xstd)[:, :pc]
# Create linear regression object

regr = linear_model.LinearRegression()



# Fit

regr.fit(Xreg, y)
# Calibration

y_c = regr.predict(Xreg)
# Cross-validation

y_cv = cross_val_predict(regr, Xreg, y, cv=10)
# Calcuate scores for calibration and cross-validation

score_c = r2_score(y, y_c)
score_cv = r2_score(y, y_cv)
# calculate the mean square error for calibration and corss validation

mse_c = mean_squared_error(y, y_c)

mse_cv = mean_squared_error(y, y_cv)
y_cv
print(score_c)
print(score_cv)
print(mse_c)
print(mse_cv)
def pcr(X, y, pc):

    '''

        Principal Component Regression in Python

    '''

    # Step 1: PCA on the input data

    # Define the PCA object

    pca = PCA()

    # Preprocessing (1): first derivative

    d1X = savgol_filter(X, 25, polyorder=5, deriv=1)

    

    # Preprocessing (2): standardize features by removing the mean and saling to unit variance

    Xstd = StandardScaler().fit_transform(d1X[:, :])

    

    # Run PCA producing the reduced variable Xreg and select the first pc components

    Xreg = pca.fit_transform(Xstd)[:, :pc]

    

    # Step 2: Regression on selected components

    # Create linear regression object

    regr = linear_model.LinearRegression()

    

    # Fit

    regr.fit(Xreg, y)

    

    # Calibration

    y_c = regr.predict(Xreg)

    

    # Cross-validation

    y_cv = cross_val_predict(regr, Xreg, y, cv=10)

    

    # Calcualte scores for calibration and cross-validation

    score_c = r2_score(y, y_c)

    score_cv = r2_score(y, y_cv)

    

    # Calculate mean square error for calibration and cross validation

    mse_c = mean_squared_error(y, y_c)

    mse_cv = mean_squared_error(y, y_cv)

    

    return (y_cv, score_c, score_cv, mse_c, mse_cv)

    
r2s_c = []

r2s_cv = []

mses_c = []

mses_cv = []

for pc in range(1, 20):

    y_cv, score_c, score_cv, mse_c, mse_cv = pcr(X, y, pc)

    r2s_c.append(score_c)

    r2s_cv.append(score_cv)

    mses_c.append(mse_c)

    mses_cv.append(mse_cv)
xticks = np.arange(1, 20).astype('uint8')

plt.plot(xticks, r2s_c, '-o', label='Calibration', color='red')

plt.plot(xticks, r2s_cv, '-o', label='Cross Validation', color='blue')

plt.xticks(xticks)

plt.xlabel('Number of PC included')

plt.ylabel('R-squared')

plt.legend()
xticks = np.arange(1, 20).astype('uint8')

plt.plot(xticks, mses_c, '-o', label='Calibration', color='red')

plt.plot(xticks, mses_cv, '-o', label='Cross Validation', color='blue')

plt.xticks(xticks)

plt.xlabel('Number of PC included')

plt.ylabel('mse')

plt.legend()
predicted, r2r, r2cv, mser, mscv = pcr(X, y, pc=6)
# Regression plot

z = np.polyfit(y, predicted, 1)

with plt.style.context('ggplot'):

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.scatter(y, predicted, c='red', edgecolors='k')

    ax.plot(y, z[1] + z[0]*y, c='blue', linewidth=1)

    ax.plot(y, y, color='green', linewidth='1')

    rpd = y.std()/np.sqrt(mscv)

    plt.title('$R^{2}$ (CV): %0.4f, RPD: %0.4f'%(r2cv, rpd))

    

    plt.xlabel('Measured $^{\circ}$Brix')

    plt.ylabel('Predicted $^{\circ}$Brix')

    plt.show()