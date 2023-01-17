import warnings

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import skew
from scipy.stats.stats import pearsonr

%config InlineBackend.figure_format = 'retina'
%matplotlib inline
# Load train dataset.
train = pd.read_csv("../input/TrainData_2006-2010.csv")
# Add numeric label for classifiers.
train['label_situacao'] = pd.Series(train.situacao).apply(lambda el: 0 if el == 'nao_eleito' else 1)

# Load test dataset.
test = pd.read_csv("../input/TestData_2014.csv")

# Delimit the scope of which variables to consider.
study_variables = ['quantidade_doacoes','quantidade_doadores','total_receita',
    'media_receita','recursos_de_outros_candidatos.comites','recursos_de_pessoas_fisicas',
    'recursos_de_pessoas_juridicas','recursos_proprios','quantidade_despesas',
    'quantidade_fornecedores','total_despesa','media_despesa','grau','estado_civil']
target = ['label_situacao']

# Separate all data for pre-processing.
all_data = pd.concat((train.loc[:,study_variables], test.loc[:,study_variables]), sort=False)
# Plot head of all_data, to check values.
all_data.head()
# Configure plot for possible skewed variables.
skewedVariables = (('quantidade_doacoes', 'quantidade_doadores', 'total_receita', 'media_receita'),
    ('quantidade_despesas', 'quantidade_fornecedores', 'total_despesa', 'media_despesa'),
    ('recursos_de_outros_candidatos.comites', 'recursos_de_pessoas_fisicas', 'recursos_de_pessoas_juridicas', 'recursos_proprios'))
skewedColors = (('skyblue', 'olive', 'red', 'yellow'),
    ('gold', 'teal', 'green', 'brown'),
    ('pink', 'grey', 'violet', 'orange'))
    
def check_skewed_dist():
    f, axes = plt.subplots(3, 4, figsize=(16, 10), sharex=True)

    for i in range(len(skewedVariables)):
        for j in range(len(skewedVariables[i])):
            sns.distplot(all_data[skewedVariables[i][j]], color=skewedColors[i][j], ax=axes[i, j])
# Configure canvas for the graphics.
matplotlib.rcParams['figure.figsize'] = (15.0, 4.0)

sns.catplot(
    x="situacao", kind="count", 
    palette="ch:.25", data=train,
    height=4.0, aspect=3
);
# Print the distribution before balancing.
check_skewed_dist()
# Log transform Skewed(> 0.75) numeric features on the dataset:
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) # Compute skewness
skewed_feats = skewed_feats[skewed_feats > 0.75]                      # Filter by skew > 0.75
skewed_feats = skewed_feats.index

all_data[skewed_feats] = np.log1p(all_data[skewed_feats])             # Log transform the skewed values
# Print the distribution after balancing.
check_skewed_dist()
# Import sklearn.preprocessing.StandardScaler
from sklearn.preprocessing import MinMaxScaler

# Initialize a scaler, then apply it to the features
scaler = MinMaxScaler() # default=(0, 1)

all_data[skewed_feats] = scaler.fit_transform(all_data[skewed_feats])
# Print the distribution after balancing.
check_skewed_dist()
# Dummies: boolean variables derived from the categorical variables
dummies_columns = ['grau','estado_civil']
all_data = pd.get_dummies(all_data, columns=dummies_columns)

# Filling NA's with the mean of the column:
all_data = all_data.fillna(all_data.mean())
# Separating data to use with sklearn:
X_train = all_data[:train.shape[0]][all_data.columns.difference(dummies_columns)]
X_test = all_data[train.shape[0]:][all_data.columns.difference(dummies_columns)]
y = train.label_situacao
X_train.head()
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score, auc, precision_recall_curve
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer
from itertools import product

def cv_average_score(basic_model, cvFolds):
    def calc_auc_pr(y_true, y_probs):
        precision, recall, _ = precision_recall_curve(y_true, y_probs)
        return auc(recall, precision)
    
    cv_results = cross_validate(
        basic_model, X_train, y, cv=cvFolds, return_train_score=False,
        scoring={
            'AUC-ROC': 'roc_auc',
            'Precision': 'precision',
            'Recall': 'recall',
            'F1-Score': 'f1',
            'AveragePrecision': 'average_precision',
            'AUC-PR': make_scorer(calc_auc_pr, needs_proba=True)
        }
    )
    
    return {
        'AUC-ROC': np.mean(cv_results['test_AUC-ROC']),
        'Precision': np.mean(cv_results['test_Precision']),
        'Recall': np.mean(cv_results['test_Recall']),
        'F1-Score': np.mean(cv_results['test_F1-Score']),
        'AveragePrecision': np.mean(cv_results['test_AveragePrecision']),
        'AUC-PR': np.mean(cv_results['test_AUC-PR']),
        'Model': basic_model.fit(X_train, y)
    }
    
def report_scores(scores):
    return f'''Scores from {scores['Name']}:
    AUC-ROC: {scores['AUC-ROC']}
    Precision: {scores['Precision']}
    Recall: {scores['Recall']}
    F1-Score: {scores['F1-Score']}
    AveragePrecision: {scores['AveragePrecision']}
    AUC-PR: {scores['AUC-PR']}'''
    
    return result
def cv_tune(func):
    def new_func(params, cvfolds, best_fit=None):
        if best_fit != None:
            return func(best_fit, cvfolds)
        results = []
        for p in product(*params):
            results.append(func(p, cvfolds))
        return best_score(results)
    
    return new_func
def graph_coefs(coefs):
    return pd.DataFrame({
        'Variável': pd.Series(X_train.columns.values),
        'Coeficiente': pd.Series(coefs)
    }).sort_values(by=['Coeficiente'], ascending=False)
def best_score(scores):
    scoreSAucROC = sorted(scores, key=lambda s: s['AUC-ROC'])
    scoreSPrec = sorted(scoreSAucROC, key=lambda s: s['Precision'])
    scoreSRec = sorted(scoreSPrec, key=lambda s: s['Recall'])
    scoreSF1 = sorted(scoreSRec, key=lambda s: s['F1-Score'])
    scoreSAvPrec = sorted(scoreSF1, key=lambda s: s['AveragePrecision'])
    scoreSAucPR = sorted(scoreSAvPrec, key=lambda s: s['AUC-PR'])
    return scoreSAucPR[-1]
@cv_tune
def tuned_logRegr(params, cvFolds):
    c, = params
    score = cv_average_score(LogisticRegression(
        max_iter=800, C=c, solver='lbfgs', penalty='l2'
    ), cvFolds)
    score['Params'] = (c)
    score['Name'] = 'LogisticRegression'
    return score

params_logRegr = type('Params', (), {
    'c': [10**n for n in range(-4, 4)]
})

best_logRegr = tuned_logRegr([
    params_logRegr.c
], 6, best_fit=(1,))
model_logRegr = best_logRegr['Model']
print(report_scores(best_logRegr))
print(f"Best Params: {best_logRegr['Params']}")
graph_coefs(model_logRegr.coef_[0])
@cv_tune
def tuned_knn(params, cvFolds):
    n, p = params
    score = cv_average_score(KNeighborsClassifier(
        n_neighbors=n, p=p
    ), cvFolds)
    score['Params'] = (n, p)
    score['Name'] = 'K-NearestNeighbors'
    return score

params_knn = type('Params', (), {
    'n_neighbors': range(1, len(X_train.columns.values), 6),
    'p': [1, 2, 5]
})

best_knn = tuned_knn([
    params_knn.n_neighbors, params_knn.p
], 6, best_fit=(19, 1))
model_knn = best_knn['Model']
print(report_scores(best_knn))
print(f"Best Params: {best_knn['Params']}")
@cv_tune
def tuned_decTree(params, cvFolds):
    md, ms, ml, mf = params
    score = cv_average_score(DecisionTreeClassifier(
        max_depth=md, min_samples_split=ms,
        min_samples_leaf=ml, max_features=mf
    ), cvFolds)
    score['Params'] = (md, ms, ml, mf)
    score['Name'] = 'DecisionTree'
    return score

params_decTree = type('Params', (), {
    'max_depth': range(1, 32, 4),
    'min_samples_split': range(int(0.1*len(X_train)), int(len(X_train)), int(0.3*len(X_train))),
    'min_samples_leaf': [0.1, 0.25, 0.5],
    'max_features': range(1, len(X_train.columns.values), 4)
})

best_decTree = tuned_decTree([
    params_decTree.max_depth, params_decTree.min_samples_split,
    params_decTree.min_samples_leaf, params_decTree.max_features
], 6, best_fit=(25, 3048, 0.1, 21))
model_decTree = best_decTree['Model']
print(report_scores(best_decTree))
print(f"Best Params: {best_decTree['Params']}")
import graphviz 

dot_data = export_graphviz(
    model_decTree, 
    out_file=None, 
    feature_names=X_train.columns.values, 
    class_names=['nao_eleito', 'eleito']
) 
graph = graphviz.Source(dot_data) 
graph
@cv_tune
def tuned_gradBoost(params, cvfolds):
    (ne, ms, ml, md) = params
    score = cv_average_score(GradientBoostingClassifier(
        learning_rate=0.1, n_estimators=ne,
        min_samples_split=ms, max_depth=md,
        subsample=0.8, min_samples_leaf=ml,
        max_features='sqrt'
    ), cvfolds)
    score['Params'] = (ne, ms, ml, md)
    score['Name'] = 'GradientBoosting'
    return score

params_gradBoost = type('Params', (), {
    'n_estimators': [40, 55, 70],
    'min_samples_split': range(int(0.005*len(X_train)), int(0.02*len(X_train)), 10),
    'min_samples_leaf': [30, 50, 70],
    'max_depth': [5, 10, 15, 20]
})

best_gradBoost = tuned_gradBoost([
    params_gradBoost.n_estimators, params_gradBoost.min_samples_split, 
    params_gradBoost.min_samples_leaf, params_gradBoost.max_depth
], 6, best_fit=(40, 78, 70, 5))
model_gradBoost = best_gradBoost['Model']
print(report_scores(best_gradBoost))
print(f"Best Params: {best_gradBoost['Params']}")
graph_coefs(model_gradBoost.feature_importances_)
@cv_tune
def tuned_randForest(params, cvFolds):
    n, ml, mf = params
    score = cv_average_score(RandomForestClassifier(
        n_estimators=n, min_samples_leaf=ml, max_features=mf
    ), cvFolds)
    score['Params'] = (n, ml, mf)
    score['Name'] = 'RandomForest'
    return score

params_randForest = type('Params', (), {
    'n_estimators': [40, 55, 70],
    'min_samples_leaf': [30, 50, 70],
    'max_features': [None, 'sqrt', 0.2]
})

best_randForest = tuned_randForest([
    params_randForest.n_estimators, params_randForest.min_samples_leaf,
    params_randForest.max_features
], 6, best_fit=(40, 50, 'sqrt'))
model_randForest = best_randForest['Model']
print(report_scores(best_randForest))
print(f"Best Params: {best_randForest['Params']}")
graph_coefs(model_randForest.feature_importances_)
@cv_tune
def tuned_adaBoost(params, cvFolds):
    n, lr = params
    score = cv_average_score(AdaBoostClassifier(
        n_estimators=n, learning_rate=lr
    ), cvFolds)
    score['Params'] = (n, lr)
    score['Name'] = 'AdaBoost'
    return score

params_adaBoost = type('Params', (), {
    'n_estimators': [40, 48, 52, 55, 58, 60, 62, 65, 70],
    'learning_rate': [0.001, 0.00625, 0.0125, 0.025, 0.1, 0.5, 1]
})

best_adaBoost = tuned_adaBoost([
    params_adaBoost.n_estimators,
    params_adaBoost.learning_rate
], 6 , best_fit=(62, 0.0125))
model_adaBoost = best_adaBoost['Model']
print(report_scores(best_adaBoost))
print(f"Best Params: {best_adaBoost['Params']}")
graph_coefs(model_adaBoost.feature_importances_)
all_scores = [best_logRegr, best_knn, best_decTree, best_gradBoost, best_randForest, best_adaBoost]

analysis_scores = pd.DataFrame({
    'Name': pd.Series(list(map(lambda x: x['Name'], all_scores))),
    'AUC-ROC': pd.Series(list(map(lambda x: x['AUC-ROC'], all_scores))), 
    'Precision': pd.Series(list(map(lambda x: x['Precision'], all_scores))),
    'Recall': pd.Series(list(map(lambda x: x['Recall'], all_scores))),
    'AveragePrecision': pd.Series(list(map(lambda x: x['AveragePrecision'], all_scores))),
    'F1-Score': pd.Series(list(map(lambda x: x['F1-Score'], all_scores))),
    'AUC-PR': pd.Series(list(map(lambda x: x['AUC-PR'], all_scores))),
})

analysis_scores
best_model = best_score(all_scores)
choosen_model = best_model['Model']
print(report_scores(best_model))
y_test = choosen_model.predict(X_test)
y_test_result = pd.Series(y_test).apply(lambda el: 'nao_eleito' if el == 0 else 'eleito')
import csv
analysis_result = pd.DataFrame({'Id': test['sequencial_candidato'], 'Predicted': y_test_result})
analysis_result.to_csv('election_prediction.csv',header=True, index=False, quoting=csv.QUOTE_ALL)
analysis_result.head()
