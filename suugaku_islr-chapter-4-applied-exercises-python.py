# Import standard Python data science libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()



# Import classes from scikit-learn for logistic regression, LDA, QDA, and KNN classification

# Import convenience function for computing confusion matrices 

# Import OneHotEncoder and StandardScaler for data pre-processing

# Import Pipeline, ColumnTransformer to encapsulate pre-processing heterogenous data and fitting

# into a single estimator

from sklearn.linear_model import LogisticRegression

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer



# Load StatsModels API

# Note that if we wish to use R-style formulas, then we would use the StatsModels Formula API

import statsmodels.api as sm

import statsmodels.formula.api as smf
weekly_filepath = "../input/islr-weekly/Weekly.csv"

weekly = pd.read_csv(weekly_filepath)

weekly.head()
weekly.describe()
weekly["Direction"].value_counts()
sns.pairplot(weekly, hue = "Direction");
fig = plt.figure(figsize = (10, 8))

ax = plt.axes()

ax.scatter(x = weekly.index, y = weekly["Volume"], alpha = 0.5)

ax.set(xlabel = "Week", ylabel = "Shares traded (in billions)");
weekly.corr()
# Using the Logit class from StatsModels

# First encode response numerically

endog = (weekly["Direction"] == "Up").astype("int64")

exog = sm.add_constant(weekly.drop(columns = ["Direction", "Year", "Today"]))

logit_mod = sm.Logit(endog, exog)

logit_res = logit_mod.fit()

print(logit_res.summary())
pd.DataFrame({"Estimate": logit_res.params, "Std. Error": logit_res.bse, "z value": logit_res.tvalues,

             "Pr(>|z|)": logit_res.pvalues})
mat = pd.DataFrame(logit_res.pred_table(), columns = ["Down", "Up"], index = ["Down", "Up"])

fig, ax = plt.subplots()

sns.heatmap(mat, annot = True, cbar = False, ax = ax, fmt = "g", square = True, annot_kws = {"fontsize": "x-large"})

ax.set(xlabel = "predicted label", ylabel = "true label");
logit_preds = pd.Series(logit_res.predict()).apply(lambda x: "Up" if (x > 0.5) else "Down")

(logit_preds == weekly["Direction"]).mean()
train_mask = (weekly["Year"] < 2009)
# Using the Logit class from StatsModels

# First encode response numerically

train_endog = (weekly.loc[train_mask, "Direction"] == "Up").astype("int64")

train_exog = sm.add_constant(weekly.loc[train_mask, "Lag2"])

logit_mod = sm.Logit(train_endog, train_exog)

logit_res = logit_mod.fit()

print(logit_res.summary())
test_exog = sm.add_constant(weekly.loc[~train_mask, "Lag2"])

test_endog = weekly.loc[~train_mask, "Direction"]

logit_test_probs = logit_res.predict(test_exog)

logit_test_preds = pd.Series(logit_test_probs).apply(lambda x: "Up" if (x > 0.5) else "Down")
mat = pd.DataFrame(confusion_matrix(test_endog, logit_test_preds), columns = ["Down", "Up"], index = ["Down", "Up"])

fig, ax = plt.subplots()

sns.heatmap(mat, annot = True, cbar = False, ax = ax, fmt = "g", square = True, annot_kws = {"fontsize": "x-large"})

ax.set(xlabel = "predicted label", ylabel = "true label");
(logit_test_preds == weekly.loc[~train_mask, "Direction"]).mean()
(weekly.loc[~train_mask, "Direction"] == "Up").mean()
# First separate out the training and test sets

X_train = weekly.loc[train_mask, "Lag2"].to_frame()

y_train = weekly.loc[train_mask, "Direction"]

X_test = weekly.loc[~train_mask, "Lag2"].to_frame()

y_test = weekly.loc[~train_mask, "Direction"]



# Fit the LDA model using the training set

lda_clf = LinearDiscriminantAnalysis()

lda_clf.fit(X_train, y_train)
y_pred = lda_clf.predict(X_test)

mat = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots()

sns.heatmap(mat, annot = True, cbar = False, ax = ax, fmt = "g", square = True, annot_kws = {"fontsize": "x-large"},

           xticklabels = lda_clf.classes_, yticklabels = lda_clf.classes_)

ax.set(xlabel = "predicted label", ylabel = "true label");
lda_clf.score(X_test, y_test)
# First separate out the training and test sets

X_train = weekly.loc[train_mask, "Lag2"].to_frame()

y_train = weekly.loc[train_mask, "Direction"]

X_test = weekly.loc[~train_mask, "Lag2"].to_frame()

y_test = weekly.loc[~train_mask, "Direction"]



# Fit the QDA model using the training set

qda_clf = QuadraticDiscriminantAnalysis()

qda_clf.fit(X_train, y_train)
y_pred = qda_clf.predict(X_test)

mat = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots()

sns.heatmap(mat, annot = True, cbar = False, ax = ax, fmt = "g", square = True, annot_kws = {"fontsize": "x-large"},

           xticklabels = qda_clf.classes_, yticklabels = qda_clf.classes_)

ax.set(xlabel = "predicted label", ylabel = "true label");
qda_clf.score(X_test, y_test)
# First separate out the training and test sets

X_train = weekly.loc[train_mask, "Lag2"].to_frame()

y_train = weekly.loc[train_mask, "Direction"]

X_test = weekly.loc[~train_mask, "Lag2"].to_frame()

y_test = weekly.loc[~train_mask, "Direction"]



# Set NumPy random seed for consistency and reproducibility of our results

np.random.seed(312)



# Fit the QDA model using the training set

knn_1_clf = KNeighborsClassifier(n_neighbors = 1)

knn_1_clf.fit(X_train, y_train)
y_pred = knn_1_clf.predict(X_test)

mat = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots()

sns.heatmap(mat, annot = True, cbar = False, ax = ax, fmt = "g", square = True, annot_kws = {"fontsize": "x-large"},

           xticklabels = qda_clf.classes_, yticklabels = qda_clf.classes_)

ax.set(xlabel = "predicted label", ylabel = "true label");
knn_1_clf.score(X_test, y_test)
weighted_lag_avg = 0.4*weekly["Lag1"] + 0.35*weekly["Lag2"] + 0.15*weekly["Lag3"] + 0.05*weekly["Lag4"] + 0.05*weekly["Lag5"]

weekly["weighted_lag_avg"] = weighted_lag_avg

weekly.head()
weekly[["Today", "weighted_lag_avg"]].corr()
# Using the Logit class from StatsModels

# First encode response numerically

train_endog = (weekly.loc[train_mask, "Direction"] == "Up").astype("int64")

train_exog = sm.add_constant(weekly.loc[train_mask, "weighted_lag_avg"])

logit_mod = sm.Logit(train_endog, train_exog)

logit_res = logit_mod.fit()

print(logit_res.summary())
test_exog = sm.add_constant(weekly.loc[~train_mask, "weighted_lag_avg"])

test_endog = weekly.loc[~train_mask, "Direction"]

logit_test_probs = logit_res.predict(test_exog)

logit_test_preds = pd.Series(logit_test_probs).apply(lambda x: "Up" if (x > 0.5) else "Down")
mat = pd.DataFrame(confusion_matrix(test_endog, logit_test_preds), columns = ["Down", "Up"], index = ["Down", "Up"])

fig, ax = plt.subplots()

sns.heatmap(mat, annot = True, cbar = False, ax = ax, fmt = "g", square = True, annot_kws = {"fontsize": "x-large"})

ax.set(xlabel = "predicted label", ylabel = "true label");
(logit_test_preds == weekly.loc[~train_mask, "Direction"]).mean()
# First separate out the training and test sets

X_train = weekly.loc[train_mask, "weighted_lag_avg"].to_frame()

y_train = weekly.loc[train_mask, "Direction"]

X_test = weekly.loc[~train_mask, "weighted_lag_avg"].to_frame()

y_test = weekly.loc[~train_mask, "Direction"]



# Fit the LDA model using the training set

lda_clf = LinearDiscriminantAnalysis()

lda_clf.fit(X_train, y_train)
y_pred = lda_clf.predict(X_test)

mat = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots()

sns.heatmap(mat, annot = True, cbar = False, ax = ax, fmt = "g", square = True, annot_kws = {"fontsize": "x-large"},

           xticklabels = lda_clf.classes_, yticklabels = lda_clf.classes_)

ax.set(xlabel = "predicted label", ylabel = "true label");
lda_clf.score(X_test, y_test)
# First separate out the training and test sets

X_train = weekly.loc[train_mask, "weighted_lag_avg"].to_frame()

y_train = weekly.loc[train_mask, "Direction"]

X_test = weekly.loc[~train_mask, "weighted_lag_avg"].to_frame()

y_test = weekly.loc[~train_mask, "Direction"]



# Fit the QDA model using the training set

qda_clf = QuadraticDiscriminantAnalysis()

qda_clf.fit(X_train, y_train)
y_pred = qda_clf.predict(X_test)

mat = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots()

sns.heatmap(mat, annot = True, cbar = False, ax = ax, fmt = "g", square = True, annot_kws = {"fontsize": "x-large"},

           xticklabels = qda_clf.classes_, yticklabels = qda_clf.classes_)

ax.set(xlabel = "predicted label", ylabel = "true label");
qda_clf.score(X_test, y_test)
qda_clf.predict_proba(X_test).max()
qda_predict_proba = qda_clf.predict_proba(X_test)

# Columns in the 2-dimensional array qda_predict_proba correspond to posterior probabilities

# for the classes, as found in qda_clf.classes_

# In this case, qda_clf.classes_ is the list ["Down", "Up"], so the index 1 column contains the 

# posterior probabilities for the class "Up"

y_pred_60 = pd.Series(qda_predict_proba[:, 1]).apply(lambda x: "Up" if (x > 0.6) else "Down")

mat = confusion_matrix(y_test, y_pred_60)

fig, ax = plt.subplots()

sns.heatmap(mat, annot = True, cbar = False, ax = ax, fmt = "g", square = True, annot_kws = {"fontsize": "x-large"},

           xticklabels = qda_clf.classes_, yticklabels = qda_clf.classes_)

ax.set(xlabel = "predicted label", ylabel = "true label");
(y_pred_60.values == y_test.values).mean()
# First separate out the training and test sets

X_train = weekly.loc[train_mask, "weighted_lag_avg"].to_frame()

y_train = weekly.loc[train_mask, "Direction"]

X_test = weekly.loc[~train_mask, "weighted_lag_avg"].to_frame()

y_test = weekly.loc[~train_mask, "Direction"]



# Set NumPy random seed for consistency and reproducibility of our results

np.random.seed(312)



# Fit the QDA model using the training set

knn_1_clf = KNeighborsClassifier(n_neighbors = 1)

knn_1_clf.fit(X_train, y_train)
y_pred = knn_1_clf.predict(X_test)

mat = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots()

sns.heatmap(mat, annot = True, cbar = False, ax = ax, fmt = "g", square = True, annot_kws = {"fontsize": "x-large"},

           xticklabels = qda_clf.classes_, yticklabels = qda_clf.classes_)

ax.set(xlabel = "predicted label", ylabel = "true label");
knn_1_clf.score(X_test, y_test)
# Set NumPy random seed for consistency and reproducibility of our results

np.random.seed(312)



# Fit the QDA model using the training set

knn_3_clf = KNeighborsClassifier(n_neighbors = 3)

knn_3_clf.fit(X_train, y_train)
y_pred = knn_3_clf.predict(X_test)

mat = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots()

sns.heatmap(mat, annot = True, cbar = False, ax = ax, fmt = "g", square = True, annot_kws = {"fontsize": "x-large"},

           xticklabels = qda_clf.classes_, yticklabels = qda_clf.classes_)

ax.set(xlabel = "predicted label", ylabel = "true label");
knn_3_clf.score(X_test, y_test)
# Set NumPy random seed for consistency and reproducibility of our results

np.random.seed(312)



# Fit the QDA model using the training set

knn_5_clf = KNeighborsClassifier(n_neighbors = 5)

knn_5_clf.fit(X_train, y_train)
y_pred = knn_5_clf.predict(X_test)

mat = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots()

sns.heatmap(mat, annot = True, cbar = False, ax = ax, fmt = "g", square = True, annot_kws = {"fontsize": "x-large"},

           xticklabels = qda_clf.classes_, yticklabels = qda_clf.classes_)

ax.set(xlabel = "predicted label", ylabel = "true label");
knn_5_clf.score(X_test, y_test)
auto_filepath = "../input/ISLR-Auto/Auto.csv"

auto = pd.read_csv(auto_filepath, na_values = ["?"]).dropna()

auto.head()
auto["origin"] = auto["origin"].map({1: "American", 2: "European", 3: "Japanese"})

auto.head()
mpg_med = (auto["mpg"] > auto["mpg"].median()).map({False: "Below", True: "Above"})

auto["mpg_med"] = mpg_med

auto.head()
fig, axes = plt.subplots(nrows = 2, ncols = 3, figsize = (16, 12))

sns.boxplot(x = "mpg_med", y = "cylinders", data = auto, ax = axes[0, 0])

sns.boxplot(x = "mpg_med", y = "displacement", data = auto, ax = axes[0, 1])

sns.boxplot(x = "mpg_med", y = "horsepower", data = auto, ax = axes[0, 2])

sns.boxplot(x = "mpg_med", y = "weight", data = auto, ax = axes[1, 0])

sns.boxplot(x = "mpg_med", y = "acceleration", data = auto, ax = axes[1, 1])

sns.boxplot(x = "mpg_med", y = "year", data = auto, ax = axes[1, 2])

fig.suptitle("Boxplots for cars with above and below median mpg", size = "xx-large", y = 0.925);
fig, axes = plt.subplots(nrows = 2, ncols = 3, figsize = (16, 12))

sns.distplot(auto.loc[auto["mpg_med"] == "Below", "cylinders"], ax = axes[0, 0])

sns.distplot(auto.loc[auto["mpg_med"] == "Above", "cylinders"], ax = axes[0, 0], color = "orange")

sns.distplot(auto.loc[auto["mpg_med"] == "Below", "displacement"], ax = axes[0, 1])

sns.distplot(auto.loc[auto["mpg_med"] == "Above", "displacement"], ax = axes[0, 1], color = "orange")

sns.distplot(auto.loc[auto["mpg_med"] == "Below", "horsepower"], ax = axes[0, 2])

sns.distplot(auto.loc[auto["mpg_med"] == "Above", "horsepower"], ax = axes[0, 2], color = "orange")

sns.distplot(auto.loc[auto["mpg_med"] == "Below", "weight"], ax = axes[1, 0])

sns.distplot(auto.loc[auto["mpg_med"] == "Above", "weight"], ax = axes[1, 0], color = "orange")

sns.distplot(auto.loc[auto["mpg_med"] == "Below", "acceleration"], ax = axes[1, 1])

sns.distplot(auto.loc[auto["mpg_med"] == "Above", "acceleration"], ax = axes[1, 1], color = "orange")

sns.distplot(auto.loc[auto["mpg_med"] == "Below", "year"], ax = axes[1, 2])

sns.distplot(auto.loc[auto["mpg_med"] == "Above", "year"], ax = axes[1, 2], color = "orange")

fig.suptitle("Histograms and KDE plots for cars with above (orange) and below (blue) median mpg",

             size = "xx-large", y = 0.925);
fig, ax = plt.subplots(figsize = (10, 8))

sns.scatterplot(x = "year", y = "mpg", data = auto, ax = ax)

ax.axhline(y = auto["mpg"].median(), color = "orange", linewidth = 3);
fig, ax = plt.subplots(figsize = (10, 8))

sns.violinplot(x = "origin", y = "mpg", data = auto, ax = ax)

sns.swarmplot(x = "origin", y = "mpg", data = auto, ax = ax, color = ".25")

ax.axhline(y = auto["mpg"].median(), color = "red", linewidth = 3);
X_train, X_test, y_train, y_test = train_test_split(auto, auto["mpg_med"], test_size = 0.25, random_state = 312)
categorical_features = ["origin"]

categorical_transformer = Pipeline([("onehot", OneHotEncoder(drop = "first"))])

numerical_features = ["cylinders", "displacement", "horsepower", "weight", "year"]

preprocessor = ColumnTransformer([("cat", categorical_transformer, categorical_features),

                                 ("num", "passthrough", numerical_features)])

clf = Pipeline([("preprocessor", preprocessor), ("classifier", LinearDiscriminantAnalysis())])
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

mat = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots()

sns.heatmap(mat, annot = True, cbar = False, ax = ax, fmt = "g", square = True, annot_kws = {"fontsize": "x-large"},

           xticklabels = clf["classifier"].classes_, yticklabels = clf["classifier"].classes_)

ax.set(xlabel = "predicted label", ylabel = "true label");
1 - clf.score(X_test, y_test)
# Exclude the origin and year columns

categorical_features = []

categorical_transformer = Pipeline([("onehot", OneHotEncoder(drop = "first"))])

numerical_features = ["cylinders", "displacement", "horsepower", "weight"]

preprocessor = ColumnTransformer([("cat", categorical_transformer, categorical_features),

                                 ("num", "passthrough", numerical_features)])

clf = Pipeline([("preprocessor", preprocessor), ("classifier", LinearDiscriminantAnalysis())])
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

mat = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots()

sns.heatmap(mat, annot = True, cbar = False, ax = ax, fmt = "g", square = True, annot_kws = {"fontsize": "x-large"},

           xticklabels = clf["classifier"].classes_, yticklabels = clf["classifier"].classes_)

ax.set(xlabel = "predicted label", ylabel = "true label");
1 - clf.score(X_test, y_test)
categorical_features = ["origin"]

categorical_transformer = Pipeline([("onehot", OneHotEncoder(drop = "first"))])

numerical_features = ["cylinders", "displacement", "horsepower", "weight", "year"]

preprocessor = ColumnTransformer([("cat", categorical_transformer, categorical_features),

                                 ("num", "passthrough", numerical_features)])

clf = Pipeline([("preprocessor", preprocessor), ("classifier", QuadraticDiscriminantAnalysis())])
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

mat = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots()

sns.heatmap(mat, annot = True, cbar = False, ax = ax, fmt = "g", square = True, annot_kws = {"fontsize": "x-large"},

           xticklabels = clf["classifier"].classes_, yticklabels = clf["classifier"].classes_)

ax.set(xlabel = "predicted label", ylabel = "true label");
1 - clf.score(X_test, y_test)
# Exclude the origin and year columns

categorical_features = []

categorical_transformer = Pipeline([("onehot", OneHotEncoder(drop = "first"))])

numerical_features = ["cylinders", "displacement", "horsepower", "weight"]

preprocessor = ColumnTransformer([("cat", categorical_transformer, categorical_features),

                                 ("num", "passthrough", numerical_features)])

clf = Pipeline([("preprocessor", preprocessor), ("classifier", QuadraticDiscriminantAnalysis())])
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

mat = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots()

sns.heatmap(mat, annot = True, cbar = False, ax = ax, fmt = "g", square = True, annot_kws = {"fontsize": "x-large"},

           xticklabels = clf["classifier"].classes_, yticklabels = clf["classifier"].classes_)

ax.set(xlabel = "predicted label", ylabel = "true label");
1 - clf.score(X_test, y_test)
categorical_features = ["origin"]

categorical_transformer = Pipeline([("onehot", OneHotEncoder(drop = "first"))])

numerical_features = ["cylinders", "displacement", "horsepower", "weight", "year"]

preprocessor = ColumnTransformer([("cat", categorical_transformer, categorical_features),

                                 ("num", "passthrough", numerical_features)])

clf = Pipeline([("preprocessor", preprocessor), 

                ("classifier", LogisticRegression(penalty = "none", solver = "lbfgs"))])
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

mat = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots()

sns.heatmap(mat, annot = True, cbar = False, ax = ax, fmt = "g", square = True, annot_kws = {"fontsize": "x-large"},

           xticklabels = clf["classifier"].classes_, yticklabels = clf["classifier"].classes_)

ax.set(xlabel = "predicted label", ylabel = "true label");
1 - clf.score(X_test, y_test)
# Exclude the origin and year columns

categorical_features = []

categorical_transformer = Pipeline([("onehot", OneHotEncoder(drop = "first"))])

numerical_features = ["cylinders", "displacement", "horsepower", "weight"]

preprocessor = ColumnTransformer([("cat", categorical_transformer, categorical_features),

                                 ("num", "passthrough", numerical_features)])

clf = Pipeline([("preprocessor", preprocessor), 

                ("classifier", LogisticRegression(penalty = "none", solver = "lbfgs", max_iter = 500))])
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

mat = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots()

sns.heatmap(mat, annot = True, cbar = False, ax = ax, fmt = "g", square = True, annot_kws = {"fontsize": "x-large"},

           xticklabels = clf["classifier"].classes_, yticklabels = clf["classifier"].classes_)

ax.set(xlabel = "predicted label", ylabel = "true label");
1 - clf.score(X_test, y_test)
numerical_features = ["cylinders", "displacement", "horsepower", "weight", "year"]

numerical_transformer = Pipeline([("standardize", StandardScaler())])

preprocessor = ColumnTransformer([("num", numerical_transformer, numerical_features)])



np.random.seed(312)

k_vals = list(range(1, 21, 2))

knn_errors = {}

confusion_matrices = {}

for k in k_vals:

    clf = Pipeline([("preprocessor", preprocessor), ("classifier", KNeighborsClassifier(n_neighbors = k))])

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    confusion_matrices[k] = confusion_matrix(y_test, y_pred)

    knn_errors[k] = 1 - clf.score(X_test, y_test)

pd.Series(knn_errors)
mat = confusion_matrices[pd.Series(knn_errors).idxmin()]

fig, ax = plt.subplots()

sns.heatmap(mat, annot = True, cbar = False, ax = ax, fmt = "g", square = True, annot_kws = {"fontsize": "x-large"},

           xticklabels = clf["classifier"].classes_, yticklabels = clf["classifier"].classes_)

ax.set(xlabel = "predicted label", ylabel = "true label");
numerical_features = ["cylinders", "displacement", "horsepower", "weight"]

numerical_transformer = Pipeline([("standardize", StandardScaler())])

preprocessor = ColumnTransformer([("num", numerical_transformer, numerical_features)])



np.random.seed(312)

k_vals = list(range(1, 21, 2))

knn_errors = {}

confusion_matrices = {}

for k in k_vals:

    clf = Pipeline([("preprocessor", preprocessor), ("classifier", KNeighborsClassifier(n_neighbors = k))])

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    confusion_matrices[k] = confusion_matrix(y_test, y_pred)

    knn_errors[k] = 1 - clf.score(X_test, y_test)

pd.Series(knn_errors)
mat = confusion_matrices[pd.Series(knn_errors).idxmin()]

fig, ax = plt.subplots()

sns.heatmap(mat, annot = True, cbar = False, ax = ax, fmt = "g", square = True, annot_kws = {"fontsize": "x-large"},

           xticklabels = clf["classifier"].classes_, yticklabels = clf["classifier"].classes_)

ax.set(xlabel = "predicted label", ylabel = "true label");
def Power():

    print(2**3)
Power()
def Power2(x, a):

    print(x**a)
Power2(3, 8)
Power2(10, 3)
Power2(8, 17)
Power2(131, 3)
def Power3(x, a):

    return x**a
result = Power3(5, 3)
result
x = list(range(1, 11))

y = [Power3(i, 2) for i in x]

fig, ax = plt.subplots(figsize = (10, 8))

ax.plot(x, y, "bo")

ax.set(xlabel = "x", ylabel = "f(x)", title = "Plot of x vs x^2");
x = list(range(1, 11))

y = [Power3(i, 2) for i in x]

fig, ax = plt.subplots(figsize = (10, 8))

ax.semilogy(x, y, "bo")

ax.set(xlabel = "x", ylabel = "f(x)", title = "Log-scale plot of x vs x^2");
def PlotPower(x, a, x_scale = None, y_scale = None):

    """

    Assumes x is array-like, a is a float

    If given, assumes x_scale, y_scale are strings that can be passed to the

    matplotlib Axes.set_xscale() and Axes.set_yscale() functions

    """

    y = [Power3(i, a) for i in x]

    fig, ax = plt.subplots(figsize = (10, 8))

    ax.plot(x, y, "bo")

    if x_scale is not None:

        ax.set_xscale(x_scale)

    if y_scale is not None:

        ax.set_yscale(y_scale)
PlotPower(np.arange(1, 11), 3)
PlotPower(np.arange(1, 11), 3, x_scale = "log", y_scale = "log")
boston_filepath = "../input/corrected-boston-housing/boston_corrected.csv"

index_cols = ["TOWN", "TRACT"]

data_cols = ["TOWN", "TRACT", "CMEDV", "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX",

            "PTRATIO", "B", "LSTAT"]

boston = pd.read_csv(boston_filepath, index_col = index_cols, usecols = data_cols)

boston.head()
crim_med = (boston["CRIM"] > boston["CRIM"].median()).map({False: "Below", True: "Above"})

boston["crim_med"] = crim_med

boston.head()
boston.corr()["CRIM"]
fig, axes = plt.subplots(nrows = 2, ncols = 3, figsize = (16, 12))

sns.boxplot(x = "crim_med", y = "CMEDV", data = boston, ax = axes[0, 0])

axes[0, 0].set_ylabel("Median home value ($1000)")

sns.boxplot(x = "crim_med", y = "ZN", data = boston, ax = axes[0, 1])

axes[0, 1].set_ylabel("Proportion of land zoned for 25000+ sq ft lots")

sns.boxplot(x = "crim_med", y = "INDUS", data = boston, ax = axes[0, 2])

axes[0, 2].set_ylabel("Proportion of non-retail business acres")

sns.boxplot(x = "crim_med", y = "NOX", data = boston, ax = axes[1, 0])

axes[1, 0].set_ylabel("Nitric oxides concentration (parts per 10 million)")

sns.boxplot(x = "crim_med", y = "RM", data = boston, ax = axes[1, 1])

axes[1, 1].set_ylabel("Average rooms per home")

sns.boxplot(x = "crim_med", y = "AGE", data = boston, ax = axes[1, 2])

axes[1, 2].set_ylabel("Proportion of homes built before 1940")

fig.suptitle("Boxplots for towns with above and below median crime rate", size = "xx-large", y = 0.925);
fig, axes = plt.subplots(nrows = 2, ncols = 3, figsize = (16, 12))

sns.boxplot(x = "crim_med", y = "DIS", data = boston, ax = axes[0, 0])

axes[0, 0].set_ylabel("Weighted distance to Boston employment centers")

sns.boxplot(x = "crim_med", y = "RAD", data = boston, ax = axes[0, 1])

axes[0, 1].set_ylabel("Index of accessibility to radial highways")

sns.boxplot(x = "crim_med", y = "TAX", data = boston, ax = axes[0, 2])

axes[0, 2].set_ylabel("Property tax rate (USD per $10000)")

sns.boxplot(x = "crim_med", y = "PTRATIO", data = boston, ax = axes[1, 0])

axes[1, 0].set_ylabel("Pupil-teacher ratio")

sns.boxplot(x = "crim_med", y = "B", data = boston, ax = axes[1, 1])

axes[1, 1].set_ylabel("1000*(Proportion of black residents - 0.63)^2")

sns.boxplot(x = "crim_med", y = "LSTAT", data = boston, ax = axes[1, 2])

axes[1, 2].set_ylabel("Proportion lower socioeconomic status population")

fig.suptitle("Boxplots for towns with above and below median crime rate", size = "xx-large", y = 0.925);
X_train, X_test, y_train, y_test = train_test_split(boston, boston["crim_med"], test_size = 0.25, 

                                                    random_state = 312)
not_chas = boston.columns.drop(["CHAS", "CRIM", "crim_med"])

moderate_corr = boston.corr().loc[(boston.corr()["CRIM"].abs() > 0.3), "CRIM"].index.drop("CRIM")
clf = LogisticRegression(penalty = "none", solver = "lbfgs", max_iter = 10000)

clf.fit(X_train[not_chas], y_train)

y_pred = clf.predict(X_test[not_chas])

mat = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots()

sns.heatmap(mat, annot = True, cbar = False, ax = ax, fmt = "g", square = True, annot_kws = {"fontsize": "x-large"},

           xticklabels = clf.classes_, yticklabels = clf.classes_)

ax.set(xlabel = "predicted label", ylabel = "true label");
1 - clf.score(X_test[not_chas], y_test)
clf = LogisticRegression(penalty = "none", solver = "lbfgs", max_iter = 10000)

clf.fit(X_train[moderate_corr], y_train)

y_pred = clf.predict(X_test[moderate_corr])

mat = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots()

sns.heatmap(mat, annot = True, cbar = False, ax = ax, fmt = "g", square = True, annot_kws = {"fontsize": "x-large"},

           xticklabels = clf.classes_, yticklabels = clf.classes_)

ax.set(xlabel = "predicted label", ylabel = "true label");
1 - clf.score(X_test[moderate_corr], y_test)
clf = LinearDiscriminantAnalysis()

clf.fit(X_train[not_chas], y_train)

y_pred = clf.predict(X_test[not_chas])

mat = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots()

sns.heatmap(mat, annot = True, cbar = False, ax = ax, fmt = "g", square = True, annot_kws = {"fontsize": "x-large"},

           xticklabels = clf.classes_, yticklabels = clf.classes_)

ax.set(xlabel = "predicted label", ylabel = "true label");
1 - clf.score(X_test[not_chas], y_test)
clf = LinearDiscriminantAnalysis()

clf.fit(X_train[moderate_corr], y_train)

y_pred = clf.predict(X_test[moderate_corr])

mat = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots()

sns.heatmap(mat, annot = True, cbar = False, ax = ax, fmt = "g", square = True, annot_kws = {"fontsize": "x-large"},

           xticklabels = clf.classes_, yticklabels = clf.classes_)

ax.set(xlabel = "predicted label", ylabel = "true label");
1 - clf.score(X_test[moderate_corr], y_test)
clf = QuadraticDiscriminantAnalysis()

clf.fit(X_train[not_chas], y_train)

y_pred = clf.predict(X_test[not_chas])

mat = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots()

sns.heatmap(mat, annot = True, cbar = False, ax = ax, fmt = "g", square = True, annot_kws = {"fontsize": "x-large"},

           xticklabels = clf.classes_, yticklabels = clf.classes_)

ax.set(xlabel = "predicted label", ylabel = "true label");
1 - clf.score(X_test[not_chas], y_test)
clf = QuadraticDiscriminantAnalysis()

clf.fit(X_train[moderate_corr], y_train)

y_pred = clf.predict(X_test[moderate_corr])

mat = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots()

sns.heatmap(mat, annot = True, cbar = False, ax = ax, fmt = "g", square = True, annot_kws = {"fontsize": "x-large"},

           xticklabels = clf.classes_, yticklabels = clf.classes_)

ax.set(xlabel = "predicted label", ylabel = "true label");
1 - clf.score(X_test[moderate_corr], y_test)
np.random.seed(312)

k_vals = list(range(1, 21, 2))

knn_errors = {}

confusion_matrices = {}

for k in k_vals:

    clf = Pipeline([("standardize", StandardScaler()), ("classifier", KNeighborsClassifier(n_neighbors = k))])

    clf.fit(X_train[not_chas], y_train)

    y_pred = clf.predict(X_test[not_chas])

    confusion_matrices[k] = confusion_matrix(y_test, y_pred)

    knn_errors[k] = 1 - clf.score(X_test[not_chas], y_test)

pd.Series(knn_errors)
mat = confusion_matrices[1]

fig, ax = plt.subplots()

sns.heatmap(mat, annot = True, cbar = False, ax = ax, fmt = "g", square = True, annot_kws = {"fontsize": "x-large"},

           xticklabels = clf.classes_, yticklabels = clf.classes_)

ax.set(xlabel = "predicted label", ylabel = "true label");
mat = confusion_matrices[3]

fig, ax = plt.subplots()

sns.heatmap(mat, annot = True, cbar = False, ax = ax, fmt = "g", square = True, annot_kws = {"fontsize": "x-large"},

           xticklabels = clf.classes_, yticklabels = clf.classes_)

ax.set(xlabel = "predicted label", ylabel = "true label");
np.random.seed(312)

k_vals = list(range(1, 21, 2))

knn_errors = {}

confusion_matrices = {}

for k in k_vals:

    clf = Pipeline([("standardize", StandardScaler()), ("classifier", KNeighborsClassifier(n_neighbors = k))])

    clf.fit(X_train[moderate_corr], y_train)

    y_pred = clf.predict(X_test[moderate_corr])

    confusion_matrices[k] = confusion_matrix(y_test, y_pred)

    knn_errors[k] = 1 - clf.score(X_test[moderate_corr], y_test)

pd.Series(knn_errors)
mat = confusion_matrices[1]

fig, ax = plt.subplots()

sns.heatmap(mat, annot = True, cbar = False, ax = ax, fmt = "g", square = True, annot_kws = {"fontsize": "x-large"},

           xticklabels = clf.classes_, yticklabels = clf.classes_)

ax.set(xlabel = "predicted label", ylabel = "true label");