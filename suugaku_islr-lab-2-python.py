# Load the standard Python data science packages

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()



# Load the LinearRegression class from scikit-learn's linear_model module

from sklearn.linear_model import LinearRegression



# Load the stats module from scipy so we can code the functions to compute model statistics

from scipy import stats



# Load StatsModels API

# Note that if we wish to use R-style formulas, then we would load statsmodels.formula.api instead

import statsmodels.api as sm

import statsmodels.formula.api as smf
# Load the corrected Boston housing data set

# Create a multi-index on TOWN and TRACT columns

# Exclude the TOWNNO, LON, LAT, and MEDV columns when loading the set

boston_filepath = "../input/corrected-boston-housing/boston_corrected.csv"

boston = pd.read_csv(boston_filepath, index_col = ["TOWN", "TRACT"], 

                     usecols = ["TOWN", "TRACT", "CMEDV", "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"])
boston.head()
boston.isnull().any()
# Create a LinearRegression object

reg = LinearRegression()

# Need to extract the two columns we are using and then reshape them so that

# X has the shape (n_samples, n_features),

# Y has the shape (n_samples, n_targets)

# When using -1 as one of the reshape arguments, NumPy will infer the value

# from the length of the array and the values for the other dimensions

X = boston["LSTAT"].values.reshape(-1, 1)

y = boston["CMEDV"].values.reshape(-1, 1)

# Fit linear regression model using X = LSTAT, y = CMEDV

reg.fit(X, y);
print("Model coefficients:", reg.coef_)

print("Model intercept:", reg.intercept_)
# In the score function, X is an array of the test samples

# It needs to have shape = (num_samples, num_features)

# y is an array of the true values for the given X

reg.score(X, y)
def detailed_linear_regression(X, y):

    """

    Assume X is array-like with shape (num_samples, num_features)

    Assume y is array-like with shape (num_samples, num_targets)

    Computes the least-squares regression model and returns a dictionary consisting of

    the fitted linear regression object; a series with the residual standard error,

    R^2 value, and the overall F-statistic with corresponding p-value; and a dataframe

    with columns for the parameters, and their corresponding standard errors,

    t-statistics, and p-values.

    """

    # Create a linear regression object and fit it using x and y

    reg = LinearRegression()

    reg.fit(X, y)

    

    # Store the parameters (regression intercept and coefficients) and predictions

    params = np.append(reg.intercept_, reg.coef_)

    predictions = reg.predict(X)

    

    # Create matrix with shape (num_samples, num_features + 1)

    # Where the first column is all 1s and then there is one column for the values

    # of each feature/predictor

    X_mat = np.append(np.ones((X.shape[0], 1)), X, axis = 1)

    

    # Compute residual sum of squares

    RSS = np.sum((y - predictions)**2)

    

    # Compute total sum of squares

    TSS = np.sum((np.mean(y) - y)**2)

    

    # Estimate the variance of the y-values

    obs_var = RSS/(X_mat.shape[0] - X_mat.shape[1])

    

    # Residual standard error is square root of variance of y-values

    RSE = obs_var**0.5

    

    # Variances of the parameter estimates are on the diagonal of the 

    # variance-covariance matrix of the parameter estimates

    var_beta = obs_var*(np.linalg.inv(np.matmul(X_mat.T, X_mat)).diagonal())

    

    # Standard error is square root of variance

    se_beta = np.sqrt(var_beta)

    

    # t-statistic for beta_i is beta_i/se_i, 

    # where se_i is the standard error for beta_i

    t_stats_beta = params/se_beta

    

    # Compute p-values for each parameter using a t-distribution with

    # (num_samples - 1) degrees of freedom

    beta_p_values = [2 * (1 - stats.t.cdf(np.abs(t_i), X_mat.shape[0] - 1))

                    for t_i in t_stats_beta]

    

    # Compute value of overall F-statistic, to measure how likely our

    # coefficient estimate are, assuming there is no relationship between

    # the predictors and the response

    F_overall = ((TSS - RSS)/(X_mat.shape[1] - 1))/(RSS/(X_mat.shape[0] - X_mat.shape[1]))

    F_p_value = stats.f.sf(F_overall, X_mat.shape[1] - 1, X_mat.shape[0] - X_mat.shape[1])

    

    # Construct dataframe for the overall model statistics:

    # RSE, R^2, F-statistic, p-value for F-statistic

    oa_model_stats = pd.Series({"Residual standard error": RSE, "R-squared": reg.score(X, y),

                                "F-statistic": F_overall, "F-test p-value": F_p_value})

    

    # Construct dataframe for parameter statistics:

    # coefficients, standard errors, t-statistic, p-values for t-statistics

    param_stats = pd.DataFrame({"Coefficient": params, "Standard Error": se_beta,

                                "t-value": t_stats_beta, "Prob(>|t|)": beta_p_values})

    return {"model": reg, "param_stats": param_stats, "oa_stats": oa_model_stats}
detailed_reg = detailed_linear_regression(X, y)
np.round(detailed_reg["param_stats"], 4)
np.round(detailed_reg["oa_stats"], 4)
# Use the terms exog (exogenous) and endog (endogenous) for X and y,

# respectively to match the language used in the StatsModels documentation

# Need to manually add a column for the intercept, as StatsModels does not

# include it by default when performing ordinary least-squares regression

exog = sm.add_constant(boston["LSTAT"])

endog = boston["CMEDV"]



# Generate the model

mod = sm.OLS(endog, exog)



# Fit the model

res = mod.fit()



#Print out model summary

print(res.summary())
def param_conf_int(X, y, level = 0.95):

    """

    Assume X is array-like with shape (num_samples, num_features)

    Assume y is array-like with shape (num_samples, num_targets)

    Assume level, if given, is a float with 0 < level < 1

    Computes confidence intervals at the given confidence level for each parameter

    in the linear regression model relating the predictors X to the response y

    Returns a dataframe with the endpoints of the confidence interval for each parameter

    """

    # Store parameters and corresponding stats for easy access

    detailed_reg = detailed_linear_regression(X, y)

    param_stats = detailed_reg["param_stats"]

    conf_intervals = pd.DataFrame()

    # Degrees of freedom = num_samples - (num_features + 1)

    df = X.shape[0] - (X.shape[1] + 1)

    a, b = str(round((1 - level)*100/2, 2)) + "%", str(round((1 + level)*100/2, 2)) + "%"

    # Loop through each parameter

    for index in param_stats.index:

        coeff = param_stats.loc[index, "Coefficient"]

        std_err = param_stats.loc[index, "Standard Error"]

        # alpha = level of confidence

        # df = degrees of freedom = num_samples - number of parameters

        # loc = center of t-interval = estimated coefficient value

        # scale = standard error in coefficient estimate

        conf_intervals = conf_intervals.append(pd.DataFrame([stats.t.interval(level, df, loc = coeff, scale = std_err)],

                                                            columns = [a, b]), ignore_index = True)

    return conf_intervals
param_conf_int(X, y)
res.conf_int()
param_conf_int(X, y, level = 0.99)
res.conf_int(alpha = 0.01)
def predict_intervals(X, y, X_star, level = 0.95, kind = "confidence"):

    """

    Assume X is array-like with shape (num_samples, num_features)

    Assume y is array-like with shape (num_samples, num_targets)

    Assume X_star is array-like with shape (num_predictions, num_features) with x-values for which we want predictions

    Assume level, if given, is a float with 0 < level < 1

    Assume kind, if given is either the string "confidence" or "prediction" for the kind of interval

    Computes confidence intervals at the given confidence level for each parameter

    in the linear regression model relating the predictors X to the response y

    Returns a dataframe with the endpoints of the confidence interval for each parameter

    """

    # Store parameters and corresponding stats for easy access

    detailed_reg = detailed_linear_regression(X, y)

    predictions = detailed_reg["model"].predict(X_star)

    RSE = detailed_reg["oa_stats"]["Residual standard error"]

    intervals = pd.DataFrame()

    # Degrees of freedom = num_samples - (num_features + 1)

    df = X.shape[0] - (X.shape[1] + 1)

    a, b = str(round((1 - level)*100/2, 2)) + "%", str(round((1 + level)*100/2, 2)) + "%"

    x_bar = X.mean()

    x_tss = np.sum((X - x_bar)**2)

    # Loop through each x-value being used for prediction

    for i in range(len(predictions)) :

        prediction = predictions[i, 0]

        x_star = X_star[i, 0]

        conf_error = RSE * (1/X.shape[0] + (x_star - x_bar)**2/x_tss)**0.5

        predict_error = (RSE**2 + conf_error**2)**0.5

        # alpha = level of confidence

        # df = degrees of freedom = num_samples - number of parameters

        # loc = center of t-interval = predicted value from linear regression model

        # scale = standard error in predicted value estimate

        if (kind == "confidence"):

            lower, upper = stats.t.interval(level, df, loc = prediction, scale = conf_error)

            intervals = intervals.append(pd.Series({"prediction": prediction, a: lower, b: upper}),

                                         ignore_index = True)

        elif (kind == "prediction"):

            lower, upper = stats.t.interval(level, df, loc = prediction, scale = predict_error)

            intervals = intervals.append(pd.Series({"prediction": prediction, a: lower, b: upper}),

                                         ignore_index = True)

    return intervals
predict_intervals(X, y, np.array([5, 10, 15]).reshape((-1, 1)), level = 0.95, kind = "confidence")
predict_intervals(X, y, np.array([5, 10, 15]).reshape((-1, 1)), level = 0.99, kind = "confidence")
predict_intervals(X, y, np.array([5, 10, 15]).reshape((-1, 1)), level = 0.95, kind = "prediction")
predict_intervals(X, y, np.array([5, 10, 15]).reshape((-1, 1)), level = 0.99, kind = "prediction")
reg_predictions = res.get_prediction(np.array([[1, 5], [1, 10], [1, 15]]))
pd.DataFrame(reg_predictions.conf_int(alpha = 0.05), columns = ["2.5%", "97.5%"])
# Produce 99% prediction intervals for the predicted values of CMEDV

pd.DataFrame(reg_predictions.conf_int(obs = True, alpha = 0.01), columns = ["0.5%", "99.5%"])
class ExtendedLinearRegression(LinearRegression):

    

    def detailed_linear_regression(self, X, y):

        """

        Assume X is array-like with shape (num_samples, num_features)

        Assume y is array-like with shape (num_samples, num_targets)

        include_intercept is a boolean where True means X does not already have a column

        for the intercept

        Computes the least-squares regression model and returns a dictionary consisting of

        the fitted linear regression object; a series with the residual standard error,

        R^2 value, and the overall F-statistic with corresponding p-value; and a dataframe

        with columns for the parameters, and their corresponding standard errors,

        t-statistics, and p-values.

        """

        # Create a linear regression object and fit it using x and y

        self.training_X, self.training_y = X, y

        self.fit(X, y)

    

        # Store the parameters (regression intercept and coefficients) and predictions

        self.params = np.append(self.intercept_, self.coef_)

        predictions = self.predict(X)

    

        # Create matrix with shape (num_samples, num_features + 1)

        # Where the first column is all 1s and then there is one column for the values

        # of each feature/predictor

        X_mat = np.append(np.ones((X.shape[0], 1)), X, axis = 1)

    

        # Compute residual sum of squares

        self.RSS = np.sum((y - predictions)**2)

    

        # Compute total sum of squares

        self.TSS = np.sum((np.mean(y) - y)**2)

    

        # Estimate the variance of the y-values

        obs_var = self.RSS/(X_mat.shape[0] - X_mat.shape[1])

    

        # Residual standard error is square root of variance of y-values

        self.RSE = obs_var**0.5

    

        # Variances of the parameter estimates are on the diagonal of the 

        # variance-covariance matrix of the parameter estimates

        self.var_beta_mat = obs_var*(np.linalg.inv(np.matmul(X_mat.T, X_mat)))

        self.var_beta = self.var_beta_mat.diagonal()

    

        # Standard error is square root of variance

        self.se_beta = np.sqrt(self.var_beta)

    

        # t-statistic for beta_i is beta_i/se_i, 

        # where se_i is the standard error for beta_i

        t_stats_beta = self.params/self.se_beta

    

        # Compute p-values for each parameter using a t-distribution with

        # (num_samples - 1) degrees of freedom

        beta_p_values = [2 * (1 - stats.t.cdf(np.abs(t_i), X_mat.shape[0] - 1)) for t_i in t_stats_beta]

    

        # Compute value of overall F-statistic, to measure how likely our

        # coefficient estimate are, assuming there is no relationship between

        # the predictors and the response

        self.F_overall = ((self.TSS - self.RSS)/(X_mat.shape[1] - 1))/(self.RSS/(X_mat.shape[0] - X_mat.shape[1]))

        self.F_p_value = stats.f.sf(self.F_overall, X_mat.shape[1] - 1, X_mat.shape[0] - X_mat.shape[1])

    

        # Construct dataframe for the overall model statistics:

        # RSE, R^2, F-statistic, p-value for F-statistic

        oa_model_stats = pd.Series({"Residual standard error": self.RSE, "R-squared": self.score(X, y), "F-statistic": self.F_overall, "F-test p-value": self.F_p_value})

    

        # Construct dataframe for parameter statistics:

        # coefficients, standard errors, t-statistic, p-values for t-statistics

        param_stats = pd.DataFrame({"Coefficient": self.params, "Standard Error": self.se_beta, "t-value": t_stats_beta, "Prob(>|t|)": beta_p_values})

        return {"model": self, "param_stats": param_stats, "oa_stats": oa_model_stats}

    

    def param_conf_int(self, level = 0.95):

        """

        Assume level, if given, is a float with 0 < level < 1

        Computes confidence intervals at the given confidence level for each parameter

        in the linear regression model relating the predictors X to the response y

        Returns a dataframe with the endpoints of the confidence interval for each parameter

        """

        conf_intervals = pd.DataFrame()

        # Degrees of freedom = num_samples - (num_features + 1)

        df = self.training_X.shape[0] - (self.training_X.shape[1] + 1)

        a, b = str(round((1 - level)*100/2, 2)) + "%", str(round((1 + level)*100/2, 2)) + "%"

        # Loop through each parameter

        for i in range(len(self.params)):

            coeff = self.params[i]

            std_err = self.se_beta[i]

            # alpha = level of confidence

            # df = degrees of freedom = num_samples - number of parameters

            # loc = center of t-interval = estimated coefficient value

            # scale = standard error in coefficient estimate

            conf_intervals = conf_intervals.append(pd.DataFrame([stats.t.interval(level, df, loc = coeff, scale = std_err)], columns = [a, b]), ignore_index = True)

        return conf_intervals

    

    def predict_intervals(self, X_pred, level = 0.95, kind = "confidence"):

        """

        Assume X_pred is array-like with shape (num_predictions, num_features) with x-values for which we want predictions

        Assume level, if given, is a float with 0 < level < 1

        Assume kind, if given is either the string "confidence" or "prediction" for the kind of interval

        Computes confidence intervals at the given confidence level for each parameter

        in the linear regression model relating the predictors X to the response y

        Returns a dataframe with the endpoints of the confidence interval for each parameter

        """

        # Store predictions for easy access

        predictions = self.predict(X_pred)

        intervals = pd.DataFrame()

        # Degrees of freedom = num_samples - (num_features + 1)

        df = self.training_X.shape[0] - (self.training_X.shape[1] + 1)

        a, b = str(round((1 - level)*100/2, 2)) + "%", str(round((1 + level)*100/2, 2)) + "%"

        # Loop through each x-value being used for prediction

        for i in range(len(predictions)):

            prediction = predictions[i]

            # Need to append the leading 1 since our matrix of regression parameter

            # Estimates has first row the estimate for the constant

            x_star = np.append(np.ones(1), X_pred[i])

            conf_error = np.matmul(np.matmul(x_star.T, self.var_beta_mat), x_star)**0.5

            predict_error = (self.RSE**2 + conf_error**2)**0.5

            # alpha = level of confidence

            # df = degrees of freedom = num_samples - number of parameters

            # loc = center of t-interval = predicted value from linear regression model

            # scale = standard error in predicted value estimate

            if (kind == "confidence"):

                lower, upper = stats.t.interval(level, df, loc = prediction, scale = conf_error)

                intervals = intervals.append(pd.Series({"prediction": prediction[0], a: lower[0], b: upper[0]}), ignore_index = True) 

            elif(kind == "prediction"):

                lower, upper = stats.t.interval(level, df, loc = prediction, scale = predict_error)

                intervals = intervals.append(pd.Series({"prediction": prediction[0], a: lower[0], b: upper[0]}), ignore_index = True)

        return intervals
extended_reg = ExtendedLinearRegression()

detailed_regression_stats = extended_reg.detailed_linear_regression(X, y)

np.round(detailed_regression_stats["param_stats"], 4)
np.round(detailed_regression_stats["oa_stats"], 4)
extended_reg.predict_intervals(np.array([5, 10, 15]).reshape((-1, 1)), level = 0.99, kind = "prediction")
# Plot scatterplot with regression line and default 95% confidence interval for regression estimate

# Set the marker transparancy to 0.25 in order to more clearly see the regression line

# Make regression line orange so it is more visible

sns.regplot(x = "LSTAT", y = "CMEDV", data = boston, scatter_kws = {"alpha":0.25}, line_kws = {"color":"orange"})
# Plot scatterplot with regression line and 99% confidence interval for regression estimate

# Set the marker transparancy to 0.25 in order to more clearly see the regression line

# Make regression line orange so it is more visible

sns.regplot(x = "LSTAT", y = "CMEDV", data = boston, ci = 99, scatter_kws = {"alpha":0.25}, line_kws = {"color":"orange"})
# Set the marker transparency to 0.25 in order to improve visibility

sns.residplot(x = "LSTAT", y = "CMEDV", data = boston, scatter_kws = {"alpha":0.25})
# Generate a range of x-values to feed into the regression model for producing the

# line that gets passed to the plot function

x = np.linspace(0, 40, num = 100).reshape(-1, 1)

predictions = reg.predict(x)

fig = plt.figure()

ax = plt.axes()

# Plot the regression line in orange with a line width of 3 to increase visibility

ax.plot(x, predictions, color = "orange", linewidth = 3)

# Plot the scatter plot using an alpha value of 0.25 for the markers to reduce clutter

ax.scatter(boston["LSTAT"], boston["CMEDV"], alpha = 0.25)

# Give the scatterplot some labels that are more descriptive

ax.set(xlabel = "LSTAT", ylabel = "CMEDV", xlim = (0, 40))
# Generating residual plot by hand using scikit-learn

# Compute predicted values

predicted_cmedv = reg.predict(X)

# Compute residuals

residuals = y - predicted_cmedv

fig = plt.figure()

ax = plt.axes()

# Plot residuals versus fitted values

ax.scatter(predicted_cmedv, residuals, alpha = 0.25)

# Plot orange dashed horizontal line y = 0

ax.axhline(y = 0, color = "orange", linestyle = "--")

# Give the plot some descriptive axis labels

ax.set(xlabel = "Fitted value of CMEDV", ylabel = "Residual value")
# Generating residual plot by hand using StatsModels

# Compute predicted values

predicted_cmedv = res.predict()

# Compute residuals

residuals = res.resid

fig = plt.figure()

ax = plt.axes()

# Plot residuals versus fitted values

ax.scatter(predicted_cmedv, residuals, alpha = 0.25)

# Plot orange dashed horizontal line y = 0

ax.axhline(y = 0, color = "orange", linestyle = "--")

# Give the plot some descriptive axis labels

ax.set(xlabel = "Fitted value of CMEDV", ylabel = "Residual value")
# Appened leading column of ones to the matrix of predictors

design_mat = np.append(np.ones((X.shape[0], 1)), X, axis = 1)

# Compute hat matrix

hat_mat = design_mat @ np.linalg.inv(design_mat.T @ design_mat) @ design_mat.T

# Leverage values are the diagonal of the hat matrix

leverage_vals = hat_mat.diagonal()

residuals = (y - reg.predict(X)).flatten()

residual_standard_error = (np.sum(residuals**2) / (design_mat.shape[0] - design_mat.shape[1]))**0.5

# Compute studentized residuals

studentized_residuals = residuals/(residual_standard_error*(1 - leverage_vals)**0.5)

fig = plt.figure()

ax = plt.axes()

# Plot studentized residuals versus fitted values

ax.scatter(predicted_cmedv, studentized_residuals, alpha = 0.25)

# Plot orange dashed horizontal line y = 0

ax.axhline(y = 0, color = "orange", linestyle = "--")

# Give the plot some descriptive axis labels

ax.set(xlabel = "Fitted value of CMEDV", ylabel = "Studentized residual value")
fig = plt.figure()

ax = plt.axes()

# Plot leverage values for each observation

ax.scatter(np.arange(design_mat.shape[0]), leverage_vals, alpha = 0.25)

# Plot orange dashed horizontal line y = (p + 1)/n, the average leverage for all observations

ax.axhline(y = design_mat.shape[1]/design_mat.shape[0], color = "orange", linestyle = "--")

# Give the plot some descriptive axis labels

ax.set(xlabel = "Index", ylabel = "Leverage value", ylim = (0, leverage_vals.max()*1.1))
leverage_vals.argmax()
# Create an ExtendedLinearRegression object

reg = ExtendedLinearRegression()

# Need to extract the columns we are using and then reshape them so that

# X has the shape (n_samples, n_features),

# Y has the shape (n_samples, n_targets)

# When using -1 as one of the reshape arguments, NumPy will infer the value

# from the length of the array and the values for the other dimensions

X = boston.loc[:, ["LSTAT", "AGE"]].values

y = boston["CMEDV"].values.reshape(-1, 1)

# Fit linear regression model using X = LSTAT, y = CMEDV

detailed_regression_stats = reg.detailed_linear_regression(X, y)
np.round(detailed_regression_stats["param_stats"], 4)
np.round(detailed_regression_stats["oa_stats"], 4)
# Use the terms exog (exogenous) and endog (endogenous) for X and y,

# respectively to match the language used in the StatsModels documentation

# Need to manually add a column for the intercept, as StatsModels does not

# include it by default when performing ordinary least-squares regression

exog = sm.add_constant(boston.loc[:, ["LSTAT", "AGE"]])

endog = boston["CMEDV"]



# Generate the model

mod = sm.OLS(endog, exog)



# Fit the model

res = mod.fit()



#Print out model summary

print(res.summary())
# Create an ExtendedLinearRegression object

reg = ExtendedLinearRegression()

# Need to extract the columns we are using and then reshape them so that

# X has the shape (n_samples, n_features),

# Y has the shape (n_samples, n_targets)

# When using -1 as one of the reshape arguments, NumPy will infer the value

# from the length of the array and the values for the other dimensions

X = boston.drop(columns = ["CMEDV"]).values

y = boston["CMEDV"].values.reshape(-1, 1)

# Fit linear regression model using X = LSTAT, y = CMEDV

detailed_regression_stats = reg.detailed_linear_regression(X, y)
np.round(detailed_regression_stats["param_stats"], 4)
np.round(detailed_regression_stats["oa_stats"], 4)
# Use the terms exog (exogenous) and endog (endogenous) for X and y,

# respectively to match the language used in the StatsModels documentation

# Need to manually add a column for the intercept, as StatsModels does not

# include it by default when performing ordinary least-squares regression

exog = sm.add_constant(boston.drop(columns = ["CMEDV"]))

endog = boston["CMEDV"]



# Generate the model

mod = sm.OLS(endog, exog)



# Fit the model

res = mod.fit()



#Print out model summary

print(res.summary())
def vif(predictors):

    """

    Assumes predictors is a Pandas dataframe with at least two columns

    Returns a Pandas series containing the variance inflation factor for each column variable

    """

    columns = predictors.columns

    vif_series = pd.Series()

    for col_name in columns:

        X = predictors.drop(columns = [col_name]).values

        y = predictors[col_name].values.reshape(-1, 1)

        reg = LinearRegression().fit(X, y)

        r_sq = reg.score(X, y)

        vif_series[col_name] = 1/(1 - r_sq)

    return vif_series
vif(boston.drop(columns = ["CMEDV"]))
# Create an ExtendedLinearRegression object

reg = ExtendedLinearRegression()

# Need to extract the columns we are using and then reshape them so that

# X has the shape (n_samples, n_features),

# Y has the shape (n_samples, n_targets)

# When using -1 as one of the reshape arguments, NumPy will infer the value

# from the length of the array and the values for the other dimensions

X = boston.drop(columns = ["CMEDV", "AGE"]).values

y = boston["CMEDV"].values.reshape(-1, 1)

# Fit linear regression model using X = LSTAT, y = CMEDV

detailed_regression_stats = reg.detailed_linear_regression(X, y)
np.round(detailed_regression_stats["param_stats"], 4)
np.round(detailed_regression_stats["oa_stats"], 4)
# Use the terms exog (exogenous) and endog (endogenous) for X and y,

# respectively to match the language used in the StatsModels documentation

# Need to manually add a column for the intercept, as StatsModels does not

# include it by default when performing ordinary least-squares regression

exog = sm.add_constant(boston.drop(columns = ["CMEDV", "AGE"]))

endog = boston["CMEDV"]



# Generate the model

mod = sm.OLS(endog, exog)



# Fit the model

res = mod.fit()



#Print out model summary

print(res.summary())
# Using patsy to include interaction terms via R-style formulas



# Generate a linear regression model with LSTAT, AGE, and an interaction term between

# them to predict CMEDV

mod = smf.ols(formula = "CMEDV ~ LSTAT*AGE", data = boston)

res = mod.fit()

print(res.summary())
# Creating a column forthe interaction terms by hand



# Use the terms exog (exogenous) and endog (endogenous) for X and y,

# respectively to match the language used in the StatsModels documentation

# Need to manually add a column for the intercept, as StatsModels does not

# include it by default when performing ordinary least-squares regression

exog = sm.add_constant(boston.loc[:, ["LSTAT", "AGE"]].assign(LSTAT_AGE = boston["LSTAT"] * boston["AGE"]))

endog = boston["CMEDV"]



# Generate the model

mod = sm.OLS(endog, exog)



# Fit the model

res = mod.fit()



#Print out model summary

print(res.summary())
from sklearn.preprocessing import PolynomialFeatures

from sklearn.pipeline import Pipeline



# Create a pipeline which first transforms the data to include up to second degree terms

# Setting interaction_only to True indicates that we only want interaction terms

# and excludes higher powers of the individual features

# Since the transformed data includes the 0-degree (i.e. constant = 1) feature

# an intercept is not necessary in the linear regression

model = Pipeline([("poly", PolynomialFeatures(degree = 2, interaction_only = True)),

                 ("linear", LinearRegression(fit_intercept = False))])

X = boston.loc[:, ["LSTAT", "AGE"]].values

y = boston["CMEDV"].values.reshape((-1, 1))

model = model.fit(X, y)

print(model.named_steps["linear"].coef_)
from sklearn.preprocessing import PolynomialFeatures

# Create an ExtendedLinearRegression object

reg = ExtendedLinearRegression()

# Create a 2nd degree Polynomial features transformer which only includes interaction terms

poly = PolynomialFeatures(degree = 2, interaction_only = True)

# Need to extract the columns we are using and then reshape them so that

# X has the shape (n_samples, n_features),

# Y has the shape (n_samples, n_targets)

# When using -1 as one of the reshape arguments, NumPy will infer the value

# from the length of the array and the values for the other dimensions

# Transform X using the PolynomialFeatures transformer

# Exclude the intercept column so it plays nicely with how I've written the ExtendedLinearRegression class

X = poly.fit_transform(boston.loc[:,["LSTAT", "AGE"]].values)[:, 1:]

y = boston["CMEDV"].values.reshape(-1, 1)

# Fit linear regression model using X = LSTAT, y = CMEDV

detailed_regression_stats = reg.detailed_linear_regression(X, y)
np.round(detailed_regression_stats["param_stats"], 4)
np.round(detailed_regression_stats["oa_stats"], 4)
# Creating a column for LSTAT**2 by hand



# Use the terms exog (exogenous) and endog (endogenous) for X and y,

# respectively to match the language used in the StatsModels documentation

# Need to manually add a column for the intercept, as StatsModels does not

# include it by default when performing ordinary least-squares regression

exog = sm.add_constant(boston.loc[:, ["LSTAT"]].assign(LSTAT_sq = np.square(boston["LSTAT"])))

endog = boston["CMEDV"]



# Generate the model

mod_square = sm.OLS(endog, exog)



# Fit the model

res_square = mod_square.fit()

#Print out model summary

print(res_square.summary())
# Using patsy to include the term LSTAT**2 via R-style formulas



mod_square = smf.ols(formula = "CMEDV ~ LSTAT + np.square(LSTAT)", data = boston)

res_square = mod_square.fit()

print(res_square.summary())
# Using PolynomialFeatures transformer with scikit-learn to include the term LSTAT**2



from sklearn.preprocessing import PolynomialFeatures

# Create an ExtendedLinearRegression object

reg = ExtendedLinearRegression()

# Create a 2nd degree Polynomial features transformer

poly = PolynomialFeatures(degree = 2)

# Need to extract the columns we are using and then reshape them so that

# X has the shape (n_samples, n_features),

# Y has the shape (n_samples, n_targets)

# When using -1 as one of the reshape arguments, NumPy will infer the value

# from the length of the array and the values for the other dimensions

# Transform X using the PolynomialFeatures transformer

# Exclude the intercept column so it plays nicely with how I've written the ExtendedLinearRegression class

X = poly.fit_transform(boston.loc[:,["LSTAT"]].values)[:, 1:]

y = boston["CMEDV"].values.reshape(-1, 1)

# Fit linear regression model using X = LSTAT, y = CMEDV

detailed_regression_stats = reg.detailed_linear_regression(X, y)
np.round(detailed_regression_stats["param_stats"], 4)
np.round(detailed_regression_stats["oa_stats"], 4)
mod_square = smf.ols(formula = "CMEDV ~ LSTAT + np.square(LSTAT)", data = boston).fit()

mod = smf.ols(formula = "CMEDV ~ LSTAT", data = boston).fit()

anova_table = sm.stats.anova_lm(mod, mod_square)

anova_table
# Plot scatterplot with regression line and default 95% confidence interval for regression estimate

# Set the marker transparancy to 0.25 in order to more clearly see the regression line

# Make regression line orange so it is more visible

sns.regplot(x = "LSTAT", y = "CMEDV", data = boston, order = 2, scatter_kws = {"alpha":0.25}, line_kws = {"color":"orange"})
# Set the marker transparency to 0.25 in order to improve visibility

sns.residplot(x = "LSTAT", y = "CMEDV", data = boston, order = 2, scatter_kws = {"alpha":0.25})
poly = PolynomialFeatures(degree = 2)



# Generate a range of x-values to feed into the regression model for producing the

# line that gets passed to the plot function

x = np.linspace(0, 40, num = 100).reshape(-1, 1)

# Need to transform the x array to properly feed into the predict function

transformed = poly.fit_transform(x)[:, 1:]

predictions = reg.predict(transformed)

fig, axes = plt.subplots(nrows = 2, figsize = (10, 10), gridspec_kw = {})

# Plot the regression line in orange with a line width of 3 to increase visibility

axes[0].plot(x, predictions, color = "orange", linewidth = 3)

# Plot the scatter plot using an alpha value of 0.25 for the markers to reduce clutter

axes[0].scatter(boston["LSTAT"], boston["CMEDV"], alpha = 0.25)

# Give the scatterplot some labels that are more descriptive

axes[0].set(xlabel = "LSTAT", ylabel = "CMEDV", xlim = (0, 40))



# Generating residual plot by hand using scikit-learn

# Compute predicted values

predicted_cmedv = reg.predict(X)

# Compute residuals

residuals = y - predicted_cmedv

# Plot residuals versus fitted values

axes[1].scatter(predicted_cmedv, residuals, alpha = 0.25)

# Plot orange dashed horizontal line y = 0

axes[1].axhline(y = 0, color = "orange", linestyle = "--")

# Give the plot some descriptive axis labels

axes[1].set(xlabel = "Fitted value of CMEDV", ylabel = "Residual value")
# Creating a fifth-order polynomial fit by hand



# Use the terms exog (exogenous) and endog (endogenous) for X and y,

# respectively to match the language used in the StatsModels documentation

# Need to manually add a column for the intercept, as StatsModels does not

# include it by default when performing ordinary least-squares regression

exog = sm.add_constant(boston.loc[:, ["LSTAT"]].assign(LSTAT_2 = boston["LSTAT"]**2,

                                                      LSTAT_3 = boston["LSTAT"]**3,

                                                      LSTAT_4 = boston["LSTAT"]**4,

                                                      LSTAT_5 = boston["LSTAT"]**5))

endog = boston["CMEDV"]



# Generate the model

mod_quint = sm.OLS(endog, exog)



# Fit the model

res_quint = mod_quint.fit()



#Print out model summary

print(res_quint.summary())
# Using PolynomialFeatures transformer with scikit-learn to create fifth-order polynomial fit



from sklearn.preprocessing import PolynomialFeatures

# Create an ExtendedLinearRegression object

reg = ExtendedLinearRegression()

# Create a 2nd degree Polynomial features transformer

poly = PolynomialFeatures(degree = 5)

# Need to extract the columns we are using and then reshape them so that

# X has the shape (n_samples, n_features),

# Y has the shape (n_samples, n_targets)

# When using -1 as one of the reshape arguments, NumPy will infer the value

# from the length of the array and the values for the other dimensions

# Transform X using the PolynomialFeatures transformer

# Exclude the intercept column so it plays nicely with how I've written the ExtendedLinearRegression class

X = poly.fit_transform(boston.loc[:,["LSTAT"]].values)[:, 1:]

y = boston["CMEDV"].values.reshape(-1, 1)

# Fit linear regression model using X = LSTAT, y = CMEDV

detailed_regression_stats = reg.detailed_linear_regression(X, y)
np.round(detailed_regression_stats["param_stats"], 6)
np.round(detailed_regression_stats["oa_stats"], 6)
# Using PolynomialFeatures transformer to create fifth-order polynomial fit with StatsModels



poly = PolynomialFeatures(degree = 5)

# Use the terms exog (exogenous) and endog (endogenous) for X and y,

# respectively to match the language used in the StatsModels documentation

# No need to include a column for intercept in this case, since it is

# included when applying the fit_transform function

# Make sure to reshape the LSTAT column values to play nicely with fit_transform

exog = poly.fit_transform(boston["LSTAT"].values.reshape((-1, 1)))

endog = boston["CMEDV"]



# Generate the model

mod_quint = sm.OLS(endog, exog)



# Fit the model

res_quint = mod_quint.fit()



#Print out model summary

print(res_quint.summary())
# Using Python string manipulation alongside patsy to create fifth-order polynomial fit



# Create string for the higher-order polynomial terms

poly_terms = "+".join(["I(LSTAT**{0})".format(i) for i in range(2, 6)])

# Join this string with the rest of the formula I wish to use

my_formula = "CMEDV ~ LSTAT + " + poly_terms

mod_quint = smf.ols(formula = my_formula, data = boston)

res_quint = mod_quint.fit()

print(res_quint.summary())
# Creating a column for log(RM) by hand, using StatsModels

# Here log refers to the natural logarithm



# Use the terms exog (exogenous) and endog (endogenous) for X and y,

# respectively to match the language used in the StatsModels documentation

# Need to manually add a column for the intercept, as StatsModels does not

# include it by default when performing ordinary least-squares regression

exog = sm.add_constant(np.log(boston["RM"].rename("log(RM)")))

endog = boston["CMEDV"]



# Generate the model

mod_log = sm.OLS(endog, exog)



# Fit the model

res_log = mod_log.fit()

#Print out model summary

print(res_log.summary())
# Using patsy to include the term log(RM) via R-style formulas



mod_log = smf.ols(formula = "CMEDV ~ np.log(RM)", data = boston)

res_log = mod_log.fit()

print(res_log.summary())
# Creating a column for log(RM) by hand, using scikit-learn



# Create an ExtendedLinearRegression object

reg = ExtendedLinearRegression()

# Need to extract the columns we are using and then reshape them so that

# X has the shape (n_samples, n_features),

# Y has the shape (n_samples, n_targets)

# When using -1 as one of the reshape arguments, NumPy will infer the value

# from the length of the array and the values for the other dimensions

# Transform the RM column using np.log

X = np.log(boston["RM"]).values.reshape(-1, 1)

y = boston["CMEDV"].values.reshape(-1, 1)

# Fit linear regression model using X = LSTAT, y = CMEDV

detailed_regression_stats = reg.detailed_linear_regression(X, y)
np.round(detailed_regression_stats["param_stats"], 4)
np.round(detailed_regression_stats["oa_stats"], 4)
# Load the Carseats data set

# Use the unnamed zeroth column as the index

carseats_filepath = "../input/islr-carseats/Carseats.csv"

carseats = pd.read_csv(carseats_filepath, index_col = ["Unnamed: 0"])
carseats.head()
carseats.isnull().any()
# Using patsy to include the perform multiple regression using the Carseats data

# Include interaction terms for Income:Advertising and Price:Age

# Note that there are some qualitative predictors



# Create string for the names of all of the columns

all_columns = "+".join(carseats.columns.drop("Sales"))

# Join this string with the rest of the formula I wish to use

my_formula = "Sales ~" + all_columns + "+ Income:Advertising + Price:Age"

mod = smf.ols(formula = my_formula, data = carseats)

res = mod.fit()

print(res.summary())
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder

# Create an ExtendedLinearRegression object

reg = ExtendedLinearRegression()

# Create a 2nd degree Polynomial features transformer which only includes interaction terms

poly = PolynomialFeatures(degree = 2, interaction_only = True)

# Create columns for interaction terms Income:Advertising and Price:Age

income_advert = pd.Series(poly.fit_transform(carseats.loc[:, ["Income", "Advertising"]])[:, -1], name = "Income:Advertising")

price_age = pd.Series(poly.fit_transform(carseats.loc[:, ["Price", "Age"]])[:, -1], name = "Price:Age")

# Encode categorical predictors using OneHotEncoder

# Set the categories and drop the first category when encoding to use reduced-rank coding

# This then replicates the default behavior of how Patsy and R do categorical encoding

enc = OneHotEncoder(categories = [["Bad", "Medium", "Good"], ["No", "Yes"], ["No", "Yes"]], drop = "first")

cat_pred = enc.fit_transform(carseats.loc[:, ["ShelveLoc", "Urban", "US"]]).toarray()

cat_pred = pd.DataFrame(cat_pred, columns = ["ShelveLocMedium", "ShelveLocGood", "UrbanYes", "USYes"])

quant_pred = carseats.loc[:, ["CompPrice", "Income", "Advertising", "Population", "Price", "Age", "Education"]].reset_index(drop = True)



# Combine all of the columns into a single dataframe of predictors

# Note that we needed to reset the index for quant_pred in order to have it align with the indices

# for the other columns when joining

# We could avoid this if we worked purely with the underlying NumPy arrays

X = cat_pred.join([quant_pred, income_advert, price_age])

y = carseats["Sales"].values.reshape(-1, 1)

detailed_regression_stats = reg.detailed_linear_regression(X, y)
np.round(detailed_regression_stats["param_stats"], 4)
np.round(detailed_regression_stats["oa_stats"], 4)
# List of columns to remember which column corresponds to each coefficient

X.columns