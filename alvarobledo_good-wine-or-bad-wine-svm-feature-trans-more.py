%matplotlib inline

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
df = pd.read_csv('../input/winequality-red.csv')
df.head()
df.info()
df['quality'].sort_values().unique()
def plot_wine_quality_histogram(quality):
    unique_vals = df['quality'].sort_values().unique()
    plt.xlabel("Quality")
    plt.ylabel("Count")
    plt.hist(quality.values, bins=np.append(unique_vals, 9), align='left')
plot_wine_quality_histogram(df['quality'])
def plot_features_correlation(df):
    plt.figure(figsize=(7.5,6))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    sns.set(font_scale=1)
    corr_mat = df.corr()
    ax = sns.heatmap(data=corr_mat, annot=True, fmt='0.1f', vmin=-1.0, vmax=1.0, center=0.0, xticklabels=corr_mat.columns, yticklabels=corr_mat.columns, cmap="Blues")
    colorbar = ax.collections[0].colorbar
    colorbar.set_ticks([-1, -0.5, 0, 0.5, 1])
plot_features_correlation(df)
y = df.quality
X = df.drop('quality', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
scaler = preprocessing.StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
def scores_results(y_train, y_test, y_pred_train, y_pred_test):
    #this function will provide us with accuracy and mse scores for training and test sets
    y_pred_train_round = np.round(y_pred_train)
    y_pred_test_round = np.round(y_pred_test)
    accuracy = [accuracy_score(y_train, y_pred_train_round), accuracy_score(y_test, y_pred_test_round)]
    mse_with_rounding = [mean_squared_error(y_train, y_pred_train_round), mean_squared_error(y_test, y_pred_test_round)]
    results = pd.DataFrame(list(zip(accuracy, mse_with_rounding)), columns = ['accuracy score', 'mse (after rounding)'], index = ['train', 'test'])
    return results

def baseline(X_train_scaled, X_test_scaled, y_train, y_test):
    #dummy regressor which always predicts the mean of the dataset
    from sklearn.dummy import DummyRegressor
    baseline = DummyRegressor(strategy='mean')
    baseline.fit(X_train_scaled, y_train)
    y_pred_train = baseline.predict(X_train_scaled)
    y_pred_test = baseline.predict(X_test_scaled)
    return scores_results(y_train, y_test, y_pred_train, y_pred_test)

baseline(X_train_scaled, X_test_scaled, y_train, y_test)
def linear_reg(X_train_scaled, X_test_scaled, y_train, y_test):
    # basic linear regression
    from sklearn.linear_model import LinearRegression
    lm = LinearRegression()
    lm.fit(X_train_scaled, y_train)
    y_pred_train = lm.predict(X_train_scaled)
    y_pred_test = lm.predict(X_test_scaled)
    global metrics_lr #store this for a later comparison between different methods
    metrics_lr = [accuracy_score(y_test, np.round(y_pred_test)), mean_squared_error(y_test, y_pred_test), r2_score(y_test, y_pred_test)]
    return scores_results(y_train, y_test, y_pred_train, y_pred_test)
linear_reg(X_train_scaled, X_test_scaled, y_train, y_test)
def lasso_reg(X_train_scaled, X_test_scaled, y_train, y_test):
    from sklearn.linear_model import LassoCV
    n_alphas = 5000
    alpha_vals = np.logspace(-6, 0, n_alphas)
    lr = LassoCV(alphas=alpha_vals, cv=10, random_state=0)
    lr.fit(X_train_scaled, y_train)
    y_pred_train = lr.predict(X_train_scaled)
    y_pred_test = lr.predict(X_test_scaled)
    metrics_lasso = [accuracy_score(y_test, np.round(y_pred_test)), mean_squared_error(y_test, y_pred_test), r2_score(y_test, y_pred_test)]
    return metrics_lasso
def elastic_net_reg(X_train_scaled, X_test_scaled, y_train, y_test):
    from sklearn.linear_model import ElasticNetCV
    n_alphas = 300
    l1_ratio = [.1, .3, .5, .7, .9]
    rr = ElasticNetCV(n_alphas=n_alphas, l1_ratio=l1_ratio, cv=10, random_state=0)
    rr.fit(X_train_scaled, y_train)
    y_pred_train = rr.predict(X_train_scaled)
    y_pred_test = rr.predict(X_test_scaled)
    metrics_en = [accuracy_score(y_test, np.round(y_pred_test)), mean_squared_error(y_test, y_pred_test), r2_score(y_test, y_pred_test)]
    return metrics_en
def ridge_reg(X_train_scaled, X_test_scaled, y_train, y_test):
    from sklearn.linear_model import RidgeCV
    n_alphas = 100
    alpha_vals = np.logspace(-1, 3, n_alphas)
    rr = RidgeCV(alphas=alpha_vals, cv=10)
    rr.fit(X_train_scaled, y_train)
    y_pred_train = rr.predict(X_train_scaled)
    y_pred_test = rr.predict(X_test_scaled)
    metrics_ridge = [accuracy_score(y_test, np.round(y_pred_test)), mean_squared_error(y_test, y_pred_test), r2_score(y_test, y_pred_test)]
    return metrics_ridge

metrics_lasso = lasso_reg(X_train_scaled, X_test_scaled, y_train, y_test)
metrics_en = elastic_net_reg(X_train_scaled, X_test_scaled, y_train, y_test)
metrics_ridge = ridge_reg(X_train_scaled, X_test_scaled, y_train, y_test)
finalscores = pd.DataFrame(list(zip(metrics_lr, metrics_lasso, metrics_en, metrics_ridge)), columns = ['lr', 'lasso', 'el net', 'ridge'], index = ['acc','mse','r2'])
finalscores
from sklearn.preprocessing import PolynomialFeatures
X_deg2 = PolynomialFeatures(degree=2).fit_transform(X) #this has now 78 feautures
X_deg3 = PolynomialFeatures(degree=3).fit_transform(X) #this has now 170 features

X_train_deg2, X_test_deg2, y_train_deg2, y_test_deg2 = train_test_split(X_deg2, y, test_size=0.2, random_state=0, stratify=y)
X_train_deg3, X_test_deg3, y_train_deg3, y_test_deg3 = train_test_split(X_deg3, y, test_size=0.2, random_state=0, stratify=y)

scaler_deg2 = preprocessing.StandardScaler().fit(X_train_deg2)
scaler_deg3 = preprocessing.StandardScaler().fit(X_train_deg3)

X_train_scaled_deg2 = scaler_deg2.transform(X_train_deg2)
X_train_scaled_deg3 = scaler_deg3.transform(X_train_deg3)

X_test_scaled_deg2 = scaler_deg2.transform(X_test_deg2)
X_test_scaled_deg3 = scaler_deg3.transform(X_test_deg3)
from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.filterwarnings(action='ignore', category=ConvergenceWarning) #this is just to disable some anoying warnings that sklearn forces to always come up

linear_reg(X_train_scaled_deg2, X_test_scaled_deg2, y_train_deg2, y_test_deg2)
metrics_lasso = lasso_reg(X_train_scaled_deg2, X_test_scaled_deg2, y_train_deg2, y_test_deg2)
metrics_en = elastic_net_reg(X_train_scaled_deg2, X_test_scaled_deg2, y_train_deg2, y_test_deg2)
metrics_ridge = ridge_reg(X_train_scaled_deg2, X_test_scaled_deg2, y_train_deg2, y_test_deg2)

finalscores = pd.DataFrame(list(zip(metrics_lr, metrics_lasso, metrics_en, metrics_ridge)), columns = ['lr', 'lasso', 'el net', 'ridge'], index = ['acc','mse','r2'])
finalscores
linear_reg(X_train_scaled_deg3, X_test_scaled_deg3, y_train_deg3, y_test_deg3)
metrics_lasso = lasso_reg(X_train_scaled_deg3, X_test_scaled_deg3, y_train_deg3, y_test_deg3)
metrics_en = elastic_net_reg(X_train_scaled_deg3, X_test_scaled_deg3, y_train_deg3, y_test_deg3)
metrics_ridge = ridge_reg(X_train_scaled_deg3, X_test_scaled_deg3, y_train_deg3, y_test_deg3)

finalscores = pd.DataFrame(list(zip(metrics_lr, metrics_lasso, metrics_en, metrics_ridge)), columns = ['lr', 'lasso', 'el net', 'ridge'], index = ['acc','mse','r2'])
finalscores
def svm_reg(X_train_scaled, X_test_scaled, y_train, y_test):
    from sklearn.svm import SVR
    parameters = [{'C': [0.1, 1, 10],
                   'epsilon': [0.01, 0.1],
                    'gamma': [0.01, 0.1, 0.3, 0.5, 1]}]
    clf2 = SVR(kernel = 'rbf')
    clf = GridSearchCV(clf2, parameters, cv=10)
    clf.fit(X_train_scaled, y_train)
    y_pred_train = clf.predict(X_train_scaled)
    y_pred_test = clf.predict(X_test_scaled)
    best_parameters = clf.best_params_
    print ('best parameters:', best_parameters)
    return scores_results(y_train, y_test, y_pred_train, y_pred_test)
svm_reg(X_train_scaled, X_test_scaled, y_train, y_test)
def nn_reg(X_train_scaled, X_test_scaled, y_train, y_test):
    from sklearn.neural_network import MLPRegressor
    parameters = [{'hidden_layer_sizes': [3, 5, 10, 100],
                   'alpha': [0.01, 1, 10, 100],
                   'activation': ['relu','logistic','tanh', 'identity']}]
    nn = MLPRegressor(solver='lbfgs', random_state=0)
    nn = GridSearchCV(nn, parameters, cv = 10)
    nn.fit(X_train_scaled, y_train)
    y_pred_train = nn.predict(X_train_scaled)
    y_pred_test = nn.predict(X_test_scaled)
    best_parameters = nn.best_params_
    print ('best parameters:', best_parameters)
    return scores_results(y_train, y_test, y_pred_train, y_pred_test)
nn_reg(X_train_scaled, X_test_scaled, y_train, y_test)
