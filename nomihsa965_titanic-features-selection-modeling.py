import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

sns.set()
pd.set_option('display.expand_frame_repr', False)
train = pd.read_csv('/kaggle/input/titaniccleaned/train_cleaned.csv')
test = pd.read_csv('/kaggle/input/titaniccleaned/test_cleaned.csv')
passengerId = test['PassengerId']
test.drop('PassengerId', inplace=True, axis=1)

x_train = train.drop(['Survived'], axis=1)
y_train = train['Survived']
x_train.head()
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score


# Logistic Regression
lr = LogisticRegression()
lr.fit(x_train, y_train)
acc_log = round(lr.score(x_train, y_train) * 100, 2)

# Support Vector Machines
svc = SVC()
svc.fit(x_train, y_train)
acc_svc = round(svc.score(x_train, y_train) * 100, 2)

# Multi Layers Perceptron
mlp = MLPClassifier()
mlp.fit(x_train, y_train)
acc_mlp = round(mlp.score(x_train, y_train) * 100, 2)

# Decision Tree Classifier
dtc = DecisionTreeClassifier()
dtc.fit(x_train, y_train)
acc_dtc = round(mlp.score(x_train, y_train) * 100, 2)

# Random Forest Classifier
rfc = RandomForestClassifier()
rfc.fit(x_train, y_train)
acc_rfc = round(rfc.score(x_train, y_train) * 100, 2)

# Ada Boost Classifiter
abc = AdaBoostClassifier()
abc.fit(x_train, y_train)
acc_abc = round(abc.score(x_train, y_train) * 100, 2)

# Guassian NB
gnb = GaussianNB()
gnb.fit(x_train, y_train)
acc_gnb = round(gnb.score(x_train, y_train) * 100, 2)

# Bagging Classifier
bc = BaggingClassifier()
bc.fit(x_train, y_train)
acc_bc = round(bc.score(x_train, y_train) * 100, 2)

# Gradient Boosting Classifier
gbc = GradientBoostingClassifier()
gbc.fit(x_train, y_train)
acc_gbc = round(gbc.score(x_train, y_train) * 100, 2)

# K-Neighbors Classifier
knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
acc_knn = round(knn.score(x_train, y_train) * 100, 2)

# XGboost Classifier
_xgb = XGBClassifier()
_xgb.fit(x_train, y_train)
xgb_preds = _xgb.predict(x_train)
acc_xgb =  round(accuracy_score(y_train, xgb_preds) * 100, 2)
results = pd.DataFrame({
    'Model': ['Logistic Regression', 'Support Vector Machines', 'Multi Layers Perceptron (NN)',
              'Decision Tree Classifier', 'Random Forest', 'AdaBoost Classifier', 'Guassian Naive Bayes', 'Bagging Classifier', 
              'Gradient Boosting Classifier', 'K-Neighbors Classifier', 'XGB'],
    'Score': [acc_log, acc_svc, acc_mlp, acc_dtc, acc_rfc, acc_abc, acc_gnb, acc_bc, acc_gbc, acc_knn, acc_xgb]
})

result_df = results.sort_values(by='Score', ascending=False)
result_df = result_df.set_index('Score')
result_df.head(10)
import numpy as np

importances = pd.DataFrame({'feature':train.columns})

importances_rfc = pd.DataFrame({'feature':x_train.columns, 'RFC': np.round(rfc.feature_importances_, 3)})
importances = pd.merge(importances, importances_rfc, on=['feature', 'feature']).set_index('feature')

importances_gbc = pd.DataFrame({'feature':x_train.columns, 'GBC': np.round(gbc.feature_importances_, 3)})
importances = pd.merge(importances, importances_gbc, on=['feature', 'feature']).set_index('feature')

importances_xgb = _xgb.get_booster().get_score(importance_type='weight')
keys = list(feature_important.keys())
values = list(feature_important.values())
fig, axes = plt.subplots(nrows=1, ncols=3,figsize=(24, 3))


data = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(by = "score", ascending=False)

data.plot(kind='bar', title='XGBoost Classifier (weights)', ax=axes[0])
importances['RFC'].plot.bar(title='Random Forest Classifier', ax=axes[1])
importances['GBC'].plot.bar(title='Gradiant Boosting Classifier', ax=axes[2])

plt.subplots_adjust(top = 0.99, bottom=0.01, hspace=1, wspace=0.2)
plt.show()
from sklearn.model_selection import train_test_split


features = train.drop(['Survived'], axis=1)
labels = train['Survived']
x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
def print_results(results):
    
    best_acc = None
    means = results.cv_results_['mean_test_score']
    stds = results.cv_results_['std_test_score']
    params = results.cv_results_['params']
    for mean, std, params in zip(means, stds, params):
        print('{} (+/-{}) for {}'.format(round(mean, 3), round(std * 2, 3), params))
        if params == results.best_params_:
            best_acc = round(mean, 3)
    
    print('\nBEST PARAMS: {} | acc: {} (+/-{})\n'.format(results.best_params_, best_acc, round(std * 2, 3)))
    
X_train = x_train.drop(['Alone', 'Embarked', 'SibSp'], axis=1)
X_test = x_test.drop(['Alone', 'Embarked', 'SibSp'], axis=1)
rf = RandomForestClassifier()
parameters = {
    "n_estimators": [50, 100, 500],
    "max_depth": [4, 8, 16, None],
    "criterion" : ["gini", "entropy"],
    "min_samples_leaf" : [5, 10, 25, 50]
}

rfccv = GridSearchCV(rf, parameters, cv=7, n_jobs=-1)
rfccv.fit(X_train, y_train.values.ravel())
print_results(rfccv)

gbc = GradientBoostingClassifier()
parameters = {
    'n_estimators': [10, 50, 250, 500, 1000],
    'learning_rate': [0.001, 0.01, 1],
    'max_depth': [1, 5, 10, 20]
}

gbccv = GridSearchCV(gbc, parameters, cv=7, n_jobs=-1)
gbccv.fit(X_train, y_train.values.ravel())
print_results(gbccv)
_xgb = XGBClassifier(objective= 'binary:logistic', nthread=4, seed=42)

parameters = {
    'max_depth': range (2, 10, 1),
    'n_estimators': range(60, 220, 40),
    'learning_rate': [0.1, 0.01, 0.05]
}

xgbcv = GridSearchCV(estimator=_xgb, param_grid=parameters, scoring = 'roc_auc', n_jobs=-1, cv=7, verbose=True)
xgbcv.fit(x_train, y_train.values.ravel())
print_results(xgbcv)
from sklearn.ensemble import VotingClassifier 

models = {
    "rfc": {"mdl":rfccv.best_estimator_},
    "gbc": {"mdl":gbccv.best_estimator_},
    "xgb": {"mdl":xgbcv.best_estimator_}
}

estimators = [(mdl_name, models[mdl_name]['mdl']) for mdl_name, model in models.items()]

vot_hard = VotingClassifier(estimators = estimators, voting = 'hard', n_jobs=-1)
vot_hard.fit(x_train, y_train)
models['vot'] = {'mdl': vot_hard}
# evaluate_model('Voting Hard', vot_hard, x_test, y_test)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from time import time

def evaluate_model(name, model, features, labels):
    
    start = time()
    predicted_values = model.predict(features)
    end = time()
    f1 = round(f1_score(labels, predicted_values), 2)
    accuracy = round(accuracy_score(labels, predicted_values), 2)
    precision = round(precision_score(labels, predicted_values), 2)
    recall = round(recall_score(labels, predicted_values), 2)

    print("Name: {} | Accuracy: {} | Precision: {} | Recall: {} | F1: {} | Latency: {}ms".format(name,
                                                                                        accuracy,
                                                                                        precision,
                                                                                        recall,
                                                                                        f1,      
                                                                                        round(end - start, 4)))
    return f1, accuracy, precision, recall


for mdl_name, model in models.items():
    if mdl_name in ['rfc', 'gbc']:
        f1, accuracy, precision, recall = evaluate_model(mdl_name, model['mdl'], X_test, y_test)
    else:
        f1, accuracy, precision, recall = evaluate_model(mdl_name, model['mdl'], x_test, y_test)
    models[mdl_name]['f1'] = f1
    models[mdl_name]['acc'] = accuracy
    models[mdl_name]['prec'] = precision
    models[mdl_name]['rec'] = recall
performances = pd.DataFrame({'Performace Measures':['Accuracy', 'Precision', 'Recall', 'F1']})

for name, model in models.items():
    name = pd.DataFrame({'Performace Measures':['Accuracy', 'Precision', 'Recall', 'F1'], 
                         name: [model['acc'], model['prec'], model['rec'], model['f1']]})
    performances = pd.merge(performances, name, 
                            on=['Performace Measures', 'Performace Measures']).set_index('Performace Measures')

performances

from sklearn.metrics import precision_recall_curve

# getting the probabilities of our predictions
y_scores = models['rfc']['mdl'].predict_proba(X_train)
y_scores = y_scores[:,1]

precision, recall, threshold = precision_recall_curve(y_train, y_scores)
def plot_precision_and_recall(precision, recall, threshold):
    plt.plot(threshold, precision[:-1], "r-", label="precision", linewidth=5)
    plt.plot(threshold, recall[:-1], "b", label="recall", linewidth=5)
    plt.xlabel("threshold", fontsize=19)
    plt.legend(loc="upper right", fontsize=19)
    plt.ylim([0, 1])

plt.figure(figsize=(14, 7))
plot_precision_and_recall(precision, recall, threshold)
plt.show()
models
def ensemble(models, x_features, X_features):
    threshold = 0.3
    df = pd.DataFrame(columns=['rfc', 'gbc', 'xgb'])
    f1_com = 0
    for mdl_name, model in models.items():
        if mdl_name in ['rfc', 'gbc']:
            pred = model['mdl'].predict(X_features)
        else:
            pred = model['mdl'].predict(x_features)
        df[mdl_name] = pred * model['f1']
        f1_com += model['f1']
        
    df['Survived'] = round((df['rfc']+ df['gbc'] + df['xgb']) / f1_com, 2)

    df.loc[df['Survived'] <= threshold, 'Survived'] = 0,
    df.loc[df['Survived'] > threshold, 'Survived'] = 1,
    return df['Survived']


# getting the probabilities of our predictions
predictions = ensemble(models, x_train, X_train)

accuracy = round(accuracy_score(y_train, predictions), 2)
precision = round(precision_score(y_train, predictions), 2)
recall = round(recall_score(y_train, predictions), 2)
print("Accuracy: {} | Precision: {} | Recall: {}".format(accuracy, precision, recall))
pred = esemble(models, test)

data = list(zip(passengerId, pred))
submission = pd.DataFrame(data, columns=['PassengerId', 'Survived'])
submission['Survived'] = submission['Survived'].astype(int)
submission.to_csv('./submission.csv', index=False)
pred = models['xgb']['mdl'].predict(test)
data = list(zip(passengerId, pred))
submission = pd.DataFrame(data, columns=['PassengerId', 'Survived'])
submission.to_csv('./submission.csv', index=False)