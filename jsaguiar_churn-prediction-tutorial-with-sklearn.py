import gc
import warnings
import numpy as np
import pandas as pd
# Sklearn imports
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
warnings.simplefilter(action='ignore', category=FutureWarning)
df = pd.read_csv("../input/WA_Fn-UseC_-Telco-Customer-Churn.csv")
df.head(3)
df['TotalCharges'] = df['TotalCharges'].replace(" ", 0).astype('float32')
df.drop(['customerID'],axis=1, inplace=True)
categorical_cols = [c for c in df.columns if df[c].dtype == 'object'
                    or c == 'SeniorCitizen']
df_categorical = df[categorical_cols].copy()
for col in categorical_cols:
    if df_categorical[col].nunique() == 2:
        df_categorical[col], _ = pd.factorize(df_categorical[col])
    else:
        df_categorical = pd.get_dummies(df_categorical, columns=[col])

df_categorical.head(3)
def distplot(feature, frame, color='g'):
    plt.figure(figsize=(8,3))
    plt.title("Distribution for {}".format(feature))
    ax = sns.distplot(frame[feature], color= color)

numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
df[numerical_cols].describe()
for feat in numerical_cols: distplot(feat, df)
df_std = pd.DataFrame(StandardScaler().fit_transform(df[numerical_cols].astype('float64')),
                       columns=numerical_cols)
for feat in numerical_cols: distplot(feat, df_std, color='gray')
class Model():
    def __init__(self, classifier, frame, metrics, fixed_params = {},
                 test_size=0.2, random_seed=50):
        self.estimator = classifier
        self.seed = random_seed
        self.metrics = metrics
        self.hyperparameters = {}
        self.fixed_params = fixed_params
        self.fixed_params['random_state'] = random_seed
        if classifier == KNeighborsClassifier:
            del self.fixed_params['random_state']

        # First divide data in learning set and final test set
        self.train, self.test = train_test_split(frame, test_size=test_size, random_state= self.seed)
        self.predictors = [c for c in self.train.columns if c not in ['customerID', 'Churn']]

    def grid_search(self, fit_metric, params, num_folds=10):
        """ Save the best params to self.hyperparameters. """
        print(self.fixed_params)
        gs = GridSearchCV(self.estimator(**self.fixed_params), param_grid= params,
                          scoring=self.metrics, cv=num_folds, refit= fit_metric)
        gs.fit(self.train[self.predictors], self.train['Churn'])
        self.hyperparameters = gs.best_params_
        return [(m, gs.cv_results_['mean_test_{}'.format(m)][gs.best_index_]) for m in self.metrics]
    
    def train_and_evaluate_test(self):
        """ Train classifier on the full train set and evaluate the performance on the test set. """
        params = {**self.hyperparameters, **self.fixed_params}
        clf = self.estimator(**params).fit(self.train[self.predictors], self.train['Churn'])
        y_pred = clf.predict(self.test[self.predictors])
        y_prob = clf.predict_proba(self.test[self.predictors])[:, 1]
        results = list()
        for m in self.metrics:
            if m == 'roc_auc':
                # For calculating roc auc we need the probability of target==1
                results.append((m, roc_auc_score(self.test['Churn'], y_prob)))
            else:
                # For the other metrics we can simply use the predicted label (0 or 1)
                results.append((m, eval("{}_score".format(m))(self.test['Churn'], y_pred)))
        return results

def print_result(results, sufix = ""):
    """ Function for printing the results nicely. """
    msg = ""
    for result in results:
        msg += "| {}: {:.4f} ".format(result[0], result[1])
    print("{}- {}".format(msg, sufix))
df_processed = pd.concat([df_std, df_categorical], axis=1)
metrics = ['roc_auc', 'accuracy']
def logistic_regression(frame, grid):
    logit = Model(LogisticRegression, frame, metrics)
    print_result(logit.grid_search('roc_auc', grid), "cross-validation")
    print_result(logit.train_and_evaluate_test(), "test set")
    print("Best hyperparameters:", logit.hyperparameters)
logistic_regression(df_processed, {'C': np.logspace(-4, 4, 100, base=10)})
logit_grid = {'C': np.linspace(0.02, 3, 150)}
logistic_regression(df_processed, logit_grid)
# Grid-search following second reference suggestions
def svc_rbf(frame, grid):
    rbf = Model(SVC, frame, metrics, fixed_params= {'kernel': 'rbf', 'probability': True})
    print_result(rbf.grid_search('roc_auc', grid, num_folds=4), "cross-validation")
    print_result(rbf.train_and_evaluate_test(), "test set")
    print("Best hyperparameters:", rbf.hyperparameters)

grid_rbf =  {'C': np.logspace(-4, 1, 10, base=2), 'gamma': np.logspace(-6, 2, 10, base=2)}
svc_rbf(df_processed, grid_rbf)
def svc_linear(frame, grid):
    linear = Model(SVC, frame, metrics, fixed_params={'kernel': 'linear', 'probability': True})
    print_result(linear.grid_search('roc_auc', grid), "cross-validation")
    print_result(linear.train_and_evaluate_test(), "test set")
    print("Best hyperparameters:", linear.hyperparameters)
svc_linear(df_processed, {'C': np.logspace(-4, 1, 100, base=10)})
def svc_poly(frame, grid):
    poly_svc = Model(SVC, frame, metrics, fixed_params={'kernel': 'poly', 'probability': True})
    print_result(poly_svc.grid_search('roc_auc', grid), "cross-validation")
    print_result(poly_svc.train_and_evaluate_test(), "test set")
    print("Best hyperparameters:", poly_svc.hyperparameters)
svc_poly(df_processed, {'C': np.logspace(-5, 1, 30, base=2), 'degree': [2, 3]})
def knn_clf(frame, grid):
    knn = Model(KNeighborsClassifier, frame, metrics)
    print_result(knn.grid_search('roc_auc', grid), "cross-validation")
    print_result(knn.train_and_evaluate_test(), "test set")
    print("Best hyperparameters:", knn.hyperparameters)
knn_clf(df_processed, {'n_neighbors': [i for i in range(10, 50, 2)]})
# Remove Gender
features = ['gender']
df_processed.drop(features, axis=1, inplace=True)
logit = Model(LogisticRegression, df_processed, metrics)
print_result(logit.grid_search('roc_auc', logit_grid), "cross-validation")
# Remove services with 'no internet' label
features = ['OnlineSecurity_No internet service', 'OnlineBackup_No internet service',
           'DeviceProtection_No internet service', 'TechSupport_No internet service',
           'StreamingTV_No internet service', 'StreamingMovies_No internet service']
df_processed.drop(features, axis=1, inplace=True)
logit = Model(LogisticRegression, df_processed, metrics)
print_result(logit.grid_search('roc_auc', logit_grid), "cross-validation")
# Additional services 'No'
features = ['OnlineSecurity_No', 'OnlineBackup_No',
           'DeviceProtection_No', 'TechSupport_No',
           'StreamingTV_No', 'StreamingMovies_No']
df_processed.drop(features, axis=1, inplace=True)
logit = Model(LogisticRegression, df_processed, metrics)
print_result(logit.grid_search('roc_auc', logit_grid), "cross-validation")
# Remove PhoneService as MultipleLines has a 'No phone service' label
features = ['PhoneService']
df_processed.drop(features, axis=1, inplace=True)
logit = Model(LogisticRegression, df_processed, metrics)
print_result(logit.grid_search('roc_auc', logit_grid), "cross-validation")
print("Data shape: ", df_processed.shape)
print("Best hyperparameters:", logit.hyperparameters)
print_result(logit.train_and_evaluate_test(), "test set")
def add_polynomial_features(frame, poly_degree=2, interaction=False):
    # Generate polynomials for the three numerical features
    poly = PolynomialFeatures(degree=poly_degree, interaction_only=interaction, include_bias=False)
    poly_features = poly.fit_transform(frame[['tenure', 'MonthlyCharges', 'TotalCharges']])
    # Convert to dataframe and drop the repeated columns
    df_poly = pd.DataFrame(poly_features, columns=poly.get_feature_names())
    return pd.concat([frame, df_poly.drop(['x0', 'x1', 'x2'], axis=1)], axis=1)

# Let's try a few different options for polynomial features
for degree in range(2, 6):
    for interaction in [True, False]:
        df_poly = add_polynomial_features(df_processed, degree, interaction)
        print("Degree: {}, interaction only: {}, data shape: {}"
              .format(degree, interaction, df_poly.shape))
        logit = Model(LogisticRegression, df_poly, metrics)
        print_result(logit.grid_search('roc_auc', logit_grid), "cross-validation")
        del df_poly; gc.collect()
df_processed = add_polynomial_features(df_processed, 3, False)
logit = Model(LogisticRegression, df_processed, metrics)
print_result(logit.grid_search('roc_auc', logit_grid), "cross-validation")
print_result(logit.train_and_evaluate_test(), "test set")
# Difference between TotalCharges and the tenure multiplied by monthly charges
df_tmp = df_processed.copy()
df_tmp['charges_difference'] = df_tmp['TotalCharges'] - df_tmp['tenure']*df_tmp['MonthlyCharges']
logit = Model(LogisticRegression, df_tmp, metrics)
print_result(logit.grid_search('roc_auc', logit_grid), "cross-validation")

# Just tenure multiplied by monthly charges
df_tmp = df_processed.copy()
df_tmp['tenure*charges'] = df_tmp['tenure']*df_tmp['MonthlyCharges']
logit = Model(LogisticRegression, df_tmp, metrics)
print_result(logit.grid_search('roc_auc', logit_grid), "cross-validation")

# Ratio between the tenure multiplied by monthly charges and TotalCharges
df_tmp = df_processed.copy()
df_tmp['charges_ratio'] = df_tmp['tenure']*df_tmp['MonthlyCharges'] / (df_tmp['TotalCharges'] + 1)
logit = Model(LogisticRegression, df_tmp, metrics)
print_result(logit.grid_search('roc_auc', logit_grid), "cross-validation")
# add feature
df_processed['charges_ratio'] = df_processed['tenure']*df_processed['MonthlyCharges'] / (df_processed['TotalCharges'] + 1)
logit = Model(LogisticRegression, df_processed, metrics)
print_result(logit.grid_search('roc_auc', logit_grid), "cross-validation")
print_result(logit.train_and_evaluate_test(), "test set")
def group_and_merge(group, features):
    df_tmp = df_processed.copy()
    # Add the original column without ohe or transformations
    group_col = group + "_copy"
    df_tmp[group_col] = df[group].copy()
    # Group by the original column
    gp = df_tmp.groupby(group_col)[features].agg(['min', 'max', 'mean'])
    gp.columns = pd.Index(['{}_{}'.format(e[0], e[1]) for e in gp.columns.tolist()])
    # Merge with our dataframe and drop the copy column
    df_tmp = df_tmp.merge(gp.reset_index(), on=group_col, how='left')
    return df_tmp.drop([group_col], axis=1)

# Groups
for group in ['tenure', 'Contract', 'PaymentMethod', 'InternetService', 'MultipleLines']:
    if group == 'tenure':
        df_tmp = group_and_merge(group, ['MonthlyCharges', 'TotalCharges'])
    else:
        df_tmp = group_and_merge(group, ['tenure', 'MonthlyCharges', 'TotalCharges'])
    logit = Model(LogisticRegression, df_tmp, metrics)
    print_result(logit.grid_search('roc_auc', logit_grid), "cross-validation")
df_processed = group_and_merge('tenure', ['MonthlyCharges', 'TotalCharges'])
df_processed = group_and_merge('Contract', ['tenure', 'MonthlyCharges', 'TotalCharges'])
logistic_regression(df_processed, logit_grid)