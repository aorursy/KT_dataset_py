!pip install -U geometric-smote
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.combine import SMOTETomek
from gsmote import GeometricSMOTE
from xgboost import XGBClassifier
import lightgbm as lgb
import warnings
from pprint import pprint
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")
data = pd.read_csv('../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')
data.head()
data.describe()
data.info()
data.shape
sns.pairplot(data)
def log_transform(col):
    return np.log(col[0])

data['residual sugar'] = data[['residual sugar']].apply(log_transform, axis=1)
data['chlorides'] = data[['chlorides']].apply(log_transform, axis=1)
data['free sulfur dioxide'] = data[['free sulfur dioxide']].apply(log_transform, axis=1)
data['total sulfur dioxide'] = data[['total sulfur dioxide']].apply(log_transform, axis=1)
data['sulphates'] = data[['sulphates']].apply(log_transform, axis=1)
fig, ax1 = plt.subplots(4,3, figsize=(22,16))
k = 0
columns = list(data.columns)
for i in range(4):
    for j in range(3):
        if k != 11:
            sns.boxplot(data['quality'], data[columns[k]], ax = ax1[i][j])
            k += 1
plt.show()
def scale_outputs(col):
    return col[0] - 3

data['quality'] = data[['quality']].apply(scale_outputs, axis=1)
train_labels = pd.DataFrame(data.quality, columns=['quality'])
train_data = data.drop('quality', axis=1)
train_data.head()
sns.countplot(train_labels.quality)
sampler = SMOTETomek()
train_res, labels_res = sampler.fit_sample(train_data, train_labels)
sns.countplot(labels_res.quality)
geometric_smote = GeometricSMOTE()
train_resG, labels_resG = geometric_smote.fit_resample(train_data, train_labels)
sns.countplot(labels_resG.quality)
total_data = pd.concat([train_res, labels_res], axis=1)
corr = total_data.corr()
plt.figure(figsize=(10, 10))
plt.title('Correlation Matrix')
sns.heatmap(corr, cmap='YlGnBu', annot=True)
kfold = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)

#cross validation.
def cross_validate(model, train, labels):
    scores = []
    best_accuracy = 0
    print(train.shape)

    for train_ind, val_ind in kfold.split(train, labels):

        t_data = train.loc[train_ind]
        t_labels = labels.loc[train_ind]
        v_data = train.loc[val_ind]
        v_labels = labels.loc[val_ind]
    
        model.fit(t_data, t_labels)
        preds = model.predict(v_data)
        score = accuracy_score(v_labels, preds)
        scores.append(score)
        
        if score > best_accuracy:
            best_accuracy = score
            best_model = model
    
    print('Accuracy : ' + str(round(sum(scores)/len(scores), 2)))
    return best_model, best_accuracy
    
#parameter tuning.
def tune(model, params, train, labels):
    search = RandomizedSearchCV(model, params, n_iter=20, cv=6, random_state=21)
    best_model = search.fit(train, labels)
    pprint(best_model.best_estimator_.get_params())
    return best_model
params = {
    'n_estimators' : [125, 150, 175, 200],
    'max_depth' : [6, 7, 8],
    'max_features' : [4, 5, 6, 7],
    'bootstrap' : [True],
    'min_samples_leaf' : [2, 3, 4]
}

model = RandomForestClassifier()

best_model = tune(model, params, train_res, labels_res)
rfc = RandomForestClassifier(**best_model.best_estimator_.get_params())
final_model, best_accuracy = cross_validate(rfc, train_res, labels_res)
print('Accuracy of the best model : ' + str(round(best_accuracy, 2)))
params = {
    'eta' : [0.1, 0.2, 0.3, 0.4],
    'max_depth' : [4, 5, 6, 7],
    'verbosity' : [1],
    'subsample' : [0.5, 0.75, 1],
    'n_estimators' : [75, 100, 125, 150],
    'min_child_weight' : [2, 3, 4, 5],
    'objective' : ['multi:softmax'],
    'num_class' : [6]
}

model = XGBClassifier()

best_model = tune(model, params, train_res, labels_res)
xgb = XGBClassifier(**best_model.best_estimator_.get_params())
final_model, best_accuracy = cross_validate(xgb, train_res, labels_res)
print('Accuracy of the best model : ' + str(round(best_accuracy, 2)))
feature_imp = pd.DataFrame(sorted(zip(final_model.feature_importances_,train_res.columns)), columns=['Value','Feature'])

sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))
plt.title('XGB Features (avg over folds)')
plt.tight_layout()
plt.show()
params = {
    'n_estimators' : [75, 100, 125],
    'num_iterations' : [225, 250, 275, 300],
    'learning_rate' : [0.05, 0.075],
    'max_depth' : [6],
    'num_leaves' : [30, 35, 40],
    'min_data_in_leaf' : [15],
    'bagging_fraction' : [0.4, 0.5, 0.6],
    'feature_fraction' : [0.5, 0.6, 0.7],
    'lambda_l2' : [0.5, 0.75, 1],
    'subsample' : [0.5, 0.75, 1]
}

model = lgb.LGBMClassifier()

best_model = tune(model, params, train_res, labels_res)
model_lgb = lgb.LGBMClassifier(**best_model.best_estimator_.get_params())
final_model, best_accuracy = cross_validate(model_lgb, train_res, labels_res)
print('Accuracy of the best model : ' + str(round(best_accuracy, 2)))
feature_imp = pd.DataFrame(sorted(zip(final_model.feature_importances_,train_res.columns)), columns=['Value','Feature'])

sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))
plt.title('LightGBM Features (avg over folds)')
plt.tight_layout()
plt.show()
