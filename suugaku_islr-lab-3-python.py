import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()



# Import classes from scikit-learn for logistic regression, LDA, QDA, and KNN classification

# Also import convenience function for computing confusion matrices

from sklearn.linear_model import LogisticRegression

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import confusion_matrix

from sklearn.preprocessing import OneHotEncoder



# Load StatsModels API

# Note that if we wish to use R-style formulas, then we would load statsmodels.formula.api instead

import statsmodels.api as sm

import statsmodels.formula.api as smf
Smarket_filepath = "../input/Smarket.csv"

Smarket = pd.read_csv(Smarket_filepath, index_col = "Unnamed: 0")

Smarket.head()
# Check for missing values

Smarket.isna().any()
Smarket.shape
Smarket.describe()
Smarket["Direction"].value_counts()
sns.pairplot(Smarket, hue = "Direction")
Smarket.corr()
fig = plt.figure(figsize = (10, 8))

ax = plt.axes()

ax.scatter(x = Smarket.index, y = Smarket["Volume"], alpha = 0.5)

ax.set(xlabel = "Day", ylabel = "Shares traded (in billions)");
# Using the Logit class from StatsModels

# First encode the response numerically

endog = (Smarket["Direction"] == "Up").astype("int64")

exog = sm.add_constant(Smarket.drop(columns = ["Direction", "Year", "Today"]))

logit_mod = sm.Logit(endog, exog)

logit_res = logit_mod.fit()

print(logit_res.summary())
# Using the GLM class from StatsModels

# First encode the response numerically

endog = (Smarket["Direction"] == "Up").astype("int64")

exog = sm.add_constant(Smarket.drop(columns = ["Direction", "Year", "Today"]))

glm_mod = sm.GLM(endog, exog, family = sm.families.Binomial())

glm_res = glm_mod.fit()

print(glm_res.summary())
logit_res.params
pd.DataFrame({"Estimate":logit_res.params, "Std. Error":logit_res.bse, "z value":logit_res.tvalues,

             "Pr(>|z|)":logit_res.pvalues})
X = Smarket.drop(columns = ["Direction", "Year", "Today"])

y = Smarket["Direction"]



# By default scikit-learn uses an l2 regularization penalty, which we don't want for

# vanilla logistic regression as described in the book

log_reg = LogisticRegression(penalty = "none", solver = "lbfgs")

log_reg.fit(X, y)

params = np.concatenate((log_reg.intercept_, log_reg.coef_.flatten()))

pd.DataFrame(data = {"Coef. Est.":params}, index = X.columns.insert(0, "intercept"))
# Predictions from using Logit

logit_probs = logit_res.predict()

logit_probs[0:10]
# Predictions from using GLM with the Binomial family

glm_probs = glm_res.predict()

glm_probs[0:10]
log_reg_probs = log_reg.predict_proba(X)

pd.DataFrame(log_reg_probs, columns = log_reg.classes_).head(10)
log_reg_pred = log_reg.predict(X)

pd.DataFrame(np.hstack((log_reg_probs, log_reg_pred.reshape(-1,1))), 

             columns = np.concatenate((log_reg.classes_, ["Prediction"]))).head(10)
pd.DataFrame(logit_res.pred_table(), columns = ["PredDown", "PredUp"], index = ["ActualDown", "ActualUp"])
pd.DataFrame(confusion_matrix(y, log_reg_pred), 

             columns = ["PredDown", "PredUp"], index = ["ActualDown", "ActualUp"])
logit_preds = pd.Series(logit_probs).apply(lambda x: "Up" if (x > 0.5) else "Down")

(logit_preds == Smarket["Direction"].values).mean()
log_reg.score(X, y)
train_mask = (Smarket["Year"] < 2005)
# Recall that to to element-wise logical operators for boolean indexing in Pandas, we need to use

# | for or, & for and, ~ for not

Smarket_2005 = Smarket[~train_mask]

Smarket_2005.head()
# Using the Logit class from StatsModels, and training only on the training set

# First encode the response numerically

train_endog = (Smarket.loc[train_mask, "Direction"] == "Up").astype("int64")

train_exog = sm.add_constant(Smarket[train_mask].drop(columns = ["Direction", "Year", "Today"]))

logit_mod = sm.Logit(train_endog, train_exog)

logit_res = logit_mod.fit()
test_exog = sm.add_constant(Smarket[~train_mask].drop(columns = ["Direction", "Year", "Today"]))

test_endog = Smarket.loc[~train_mask, "Direction"]

logit_test_probs = logit_res.predict(test_exog)

logit_test_preds = pd.Series(logit_test_probs).apply(lambda x: "Up" if (x > 0.5) else "Down")

pd.DataFrame(confusion_matrix(test_endog, logit_test_preds), 

             columns = ["PredDown", "PredUp"], index = ["ActualDown", "ActualUp"])
(logit_test_preds == Smarket_2005["Direction"].values).mean()
(logit_test_preds != Smarket_2005["Direction"].values).mean()
# Using LogisticRegression from scikit-learn

# First separate out the training and test sets

X_train = Smarket[train_mask].drop(columns = ["Direction", "Year", "Today"])

y_train = Smarket.loc[train_mask, "Direction"]

X_test = Smarket[~train_mask].drop(columns = ["Direction", "Year", "Today"])

y_test = Smarket.loc[~train_mask, "Direction"]



# Fit the model using the training set

log_reg = LogisticRegression(penalty = "none", solver = "lbfgs")

log_reg.fit(X_train, y_train)



#Test the model using the held-out test set

log_reg_pred = log_reg.predict(X_test)

pd.DataFrame(confusion_matrix(y_test, log_reg_pred), 

             columns = ["PredDown", "PredUp"], index = ["ActualDown", "ActualUp"])
log_reg.score(X_test, y_test)
1 - log_reg.score(X_test, y_test)
# Using the Logit class from StatsModels, and training only on the training set

# Separate out the training and test sets

# Then encode the response numerically

train_endog = (Smarket.loc[train_mask, "Direction"] == "Up").astype("int64")

train_exog = sm.add_constant(Smarket.loc[train_mask, ["Lag1", "Lag2"]])

test_exog = sm.add_constant(Smarket.loc[~train_mask, ["Lag1", "Lag2"]])

test_endog = Smarket.loc[~train_mask, "Direction"]



# Fit logistic regression model using the training set

logit_mod = sm.Logit(train_endog, train_exog)

logit_res = logit_mod.fit()



# Test the model using the held-out test set

logit_test_probs = logit_res.predict(test_exog)

logit_test_preds = pd.Series(logit_test_probs).apply(lambda x: "Up" if (x > 0.5) else "Down")

pd.DataFrame(confusion_matrix(test_endog, logit_test_preds), 

             columns = ["PredDown", "PredUp"], index = ["ActualDown", "ActualUp"])
(logit_test_preds == Smarket_2005["Direction"].values).mean()
# Using LogisticRegression from scikit-learn

# First separate out the training and test sets

X_train = Smarket.loc[train_mask, ["Lag1", "Lag2"]]

y_train = Smarket.loc[train_mask, "Direction"]

X_test = Smarket.loc[~train_mask, ["Lag1", "Lag2"]]

y_test = Smarket.loc[~train_mask, "Direction"]



# Fit the model using the training set

log_reg = LogisticRegression(penalty = "none", solver = "lbfgs")

log_reg.fit(X_train, y_train)



#Test the model using the held-out test set

log_reg_pred = log_reg.predict(X_test)

pd.DataFrame(confusion_matrix(y_test, log_reg_pred), 

             columns = ["PredDown", "PredUp"], index = ["ActualDown", "ActualUp"])
log_reg.score(X_test, y_test)
(Smarket_2005["Direction"] == "Up").mean()
df = pd.DataFrame({"Lag1": [1.2, 1.5], "Lag2":[1.1, -0.8]})
# Using StatsModels to generate predicted probabilities for particular values of Lag1 and Lag2

logit_res.predict(sm.add_constant(df))
# Using scikit-learn to generate predicted probabilities for particular values of Lag1 and Lag2

pd.DataFrame(log_reg.predict_proba(df), columns = log_reg.classes_)
# Using scikit-learn to directly computed predicted class labels

log_reg.predict(df)
# First separate out the training and test sets

X_train = Smarket.loc[train_mask, ["Lag1", "Lag2"]]

y_train = Smarket.loc[train_mask, "Direction"]

X_test = Smarket.loc[~train_mask, ["Lag1", "Lag2"]]

y_test = Smarket.loc[~train_mask, "Direction"]



# Fit the LDA model using the training set

lda_clf = LinearDiscriminantAnalysis(store_covariance = True)

lda_clf.fit(X_train, y_train)
# Prior probabilities for the classes

pd.Series(lda_clf.priors_, index = lda_clf.classes_)
# Group means for each predictor within each class

pd.DataFrame(lda_clf.means_, index = lda_clf.classes_, columns = X_train.columns)
# Coefficients for linear discriminants

pd.Series(lda_clf.coef_.flatten(), index = X_train.columns)
# Scalings for linear discriminants

pd.Series(lda_clf.scalings_.flatten(), index = X_train.columns)
# Computing the coefficients for Lag1 and Lag2 using the log-ratio formula above

# These coefficients match with the coefficient values obtained when using the least-squares solver

np.linalg.inv(lda_clf.covariance_) @ (lda_clf.means_[1] - lda_clf.means_[0])
# Relating the coefficients with the scaling values

# These are the coefficients when using the singular value decomposition

# or eigenvalue decomposition solver

orig_coefs = np.dot(lda_clf.means_, lda_clf.scalings_).dot(lda_clf.scalings_.T)

log_ratio_coefs = orig_coefs[1, :] - orig_coefs[0, :]

log_ratio_coefs
lda_scores = lda_clf.decision_function(X_train)

sns.distplot(lda_scores, kde = False, axlabel = "log-ratio score")
lda_pred = lda_clf.predict(X_test)

pd.DataFrame(confusion_matrix(y_test, lda_pred), 

             columns = ["PredDown", "PredUp"], index = ["ActualDown", "ActualUp"])
lda_clf.score(X_test, y_test)
lda_pred_probs = pd.DataFrame(lda_clf.predict_proba(X_test), columns = lda_clf.classes_)

lda_pred_probs.head()
pd.Series(lda_pred).head()
(lda_pred_probs["Down"] > 0.9).sum()
lda_pred_probs["Down"].max()
lda_pred_probs.max()
# First separate out the training and test sets

X_train = Smarket.loc[train_mask, ["Lag1", "Lag2"]]

y_train = Smarket.loc[train_mask, "Direction"]

X_test = Smarket.loc[~train_mask, ["Lag1", "Lag2"]]

y_test = Smarket.loc[~train_mask, "Direction"]



# Fit the QDA model using the training set

qda_clf = QuadraticDiscriminantAnalysis()

qda_clf.fit(X_train, y_train)
# Prior probabilities for the classes

pd.Series(qda_clf.priors_, index = qda_clf.classes_)
# Group means for each predictor within each class

pd.DataFrame(qda_clf.means_, index = qda_clf.classes_, columns = X_train.columns)
qda_pred = qda_clf.predict(X_test)

pd.DataFrame(confusion_matrix(y_test, qda_pred), 

             columns = ["PredDown", "PredUp"], index = ["ActualDown", "ActualUp"])
qda_clf.score(X_test, y_test)
qda_pred_probs = pd.DataFrame(qda_clf.predict_proba(X_test), columns = lda_clf.classes_)

(qda_pred_probs["Down"] > 0.9).sum()
qda_pred_probs.max()
# First separate out the training and test sets

X_train = Smarket.loc[train_mask, ["Lag1", "Lag2"]]

y_train = Smarket.loc[train_mask, "Direction"]

X_test = Smarket.loc[~train_mask, ["Lag1", "Lag2"]]

y_test = Smarket.loc[~train_mask, "Direction"]



# Set NumPy random seed for consistency and reproducibility for our results

np.random.seed(1)



# Fit the KNN model using the training set and K = 1

knn_1_clf = KNeighborsClassifier(n_neighbors = 1)

knn_1_clf.fit(X_train, y_train)
knn_1_pred = knn_1_clf.predict(X_test)

pd.DataFrame(confusion_matrix(y_test, knn_1_pred), 

             columns = ["PredDown", "PredUp"], index = ["ActualDown", "ActualUp"])
knn_1_clf.score(X_test, y_test)
# Fit the KNN model using the training set and K = 3

knn_3_clf = KNeighborsClassifier(n_neighbors = 3)

knn_3_clf.fit(X_train, y_train)

knn_3_pred = knn_3_clf.predict(X_test)

pd.DataFrame(confusion_matrix(y_test, knn_3_pred), 

             columns = ["PredDown", "PredUp"], index = ["ActualDown", "ActualUp"])
knn_3_clf.score(X_test, y_test)
knn_scores = []

for k in range(1, 16, 2):

    knn_clf = KNeighborsClassifier(n_neighbors = k)

    knn_clf.fit(X_train, y_train)

    knn_scores.append(knn_clf.score(X_test, y_test))

fig = plt.figure()

ax = plt.axes()

ax.plot(range(1, 16, 2), knn_scores)

ax.set(xlabel = "n_neighbors", ylabel = "test accuracy");
caravan_filepath = "../input/Caravan.csv"

Caravan = pd.read_csv(caravan_filepath, index_col = "Unnamed: 0")

Caravan.head()
Caravan["Purchase"].value_counts()
(Caravan["Purchase"] == "Yes").mean()
standardized_X = Caravan.drop(columns = ["Purchase"]).transform(lambda x: (x - x.mean())/x.std())
Caravan.mean().head()
standardized_X.mean().head()
Caravan.var().head()
standardized_X.var().head()
test_mask = range(1, 1001)

X_train = standardized_X.drop(index = test_mask)

X_test = standardized_X.loc[test_mask, ]

y_train = Caravan.drop(index = test_mask)["Purchase"]

y_test = Caravan.loc[test_mask, "Purchase"]
# Set NumPy random seed for consistency and reproducibility for our results

np.random.seed(1)



# Fit the KNN model using the training set and K = 1

knn_1_clf = KNeighborsClassifier(n_neighbors = 1)

knn_1_clf.fit(X_train, y_train)
1 - knn_1_clf.score(X_test, y_test)
(y_test != "No").mean()
knn_1_pred = knn_1_clf.predict(X_test)

pd.DataFrame(confusion_matrix(y_test, knn_1_pred), 

             columns = ["PredNo", "PredYes"], index = ["ActualNo", "ActualYes"])
# Using K = 3

knn_3_clf = KNeighborsClassifier(n_neighbors = 3)

knn_3_clf.fit(X_train, y_train)

knn_3_pred = knn_3_clf.predict(X_test)

pd.DataFrame(confusion_matrix(y_test, knn_3_pred), 

             columns = ["PredNo", "PredYes"], index = ["ActualNo", "ActualYes"])
# Using K = 5

knn_5_clf = KNeighborsClassifier(n_neighbors = 5)

knn_5_clf.fit(X_train, y_train)

knn_5_pred = knn_5_clf.predict(X_test)

pd.DataFrame(confusion_matrix(y_test, knn_5_pred), 

             columns = ["PredNo", "PredYes"], index = ["ActualNo", "ActualYes"])
# Using the Logit class from StatsModels, and training only on the training set

# Separate out the training and test sets

# Then encode the response numerically

train_endog = (y_train == "Yes").astype("int64")

train_exog = sm.add_constant(X_train)

# Need to use has_constant = "add" to force StatsModels to include a constant term;

# X_test coincidentally happens to have a column which is already constant

test_exog = sm.add_constant(X_test, has_constant = "add")

test_endog = y_test



# Fit logistic regression model using the training set

logit_mod = sm.Logit(train_endog, train_exog)

logit_res = logit_mod.fit()



# Test the model using the held-out test set

logit_test_probs = logit_res.predict(test_exog)

logit_test_preds = pd.Series(logit_test_probs).apply(lambda x: "Yes" if (x > 0.5) else "No")

pd.DataFrame(confusion_matrix(test_endog, logit_test_preds), 

             columns = ["PredNo", "PredYes"], index = ["ActualNo", "ActualYes"])
logit_test_preds = pd.Series(logit_test_probs).apply(lambda x: "Yes" if (x > 0.25) else "No")

pd.DataFrame(confusion_matrix(test_endog, logit_test_preds), 

             columns = ["PredNo", "PredYes"], index = ["ActualNo", "ActualYes"])
# Fit the model using the training set

log_reg = LogisticRegression(penalty = "none", solver = "newton-cg")

log_reg.fit(X_train, y_train)



#Test the model using the held-out test set

log_reg_pred = log_reg.predict(X_test)

pd.DataFrame(confusion_matrix(y_test, log_reg_pred), 

             columns = ["PredNo", "PredYes"], index = ["ActualNo", "ActualYes"])
log_reg_prob = pd.DataFrame(log_reg.predict_proba(X_test), columns = log_reg.classes_)

log_reg_pred = log_reg_prob["Yes"].apply(lambda x: "Yes" if (x > 0.25) else "No")

pd.DataFrame(confusion_matrix(y_test, log_reg_pred), 

             columns = ["PredNo", "PredYes"], index = ["ActualNo", "ActualYes"])