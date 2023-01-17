import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



import warnings

warnings.filterwarnings("ignore")
path = "../input/"
df = pd.read_csv(path+'heart.csv')

df.head()
df.describe()
df.target.value_counts()
sns.countplot(x='target', data=df)

plt.show()
sns.countplot(x='sex', data=df)

plt.xlabel('Sex (0=female, 1=male)')

plt.show()
sns.countplot(x='sex', hue='target', data=df)

plt.xlabel('Sex (0=female, 1=male)')

plt.show()
percentFemale = len(df[df.sex==0])/len(df.sex)*100

percentMale = len(df[df.sex==1])/len(df.sex)*100

print(f'Percentage of Female Patients: {percentFemale:.2f}%')

print(f'Percentage of Male Patients: {percentMale:.2f}%')
percentFemaleWithDisease = len(df[(df.sex==0) & (df.target==1)])/len(df[df.sex==0])*100

percentMaleWithDisease = len(df[(df.sex==1) & (df.target==1)])/len(df[df.sex==1])*100

print(f'Percentage of Female Patients with Disease: {percentFemaleWithDisease:.2f}%')

print(f'Percentage of Male Patients with Disease: {percentMaleWithDisease:.2f}%')
sns.countplot(x='cp', hue='target', data=df)

plt.xlabel('Chest pain type')

plt.show()
sns.countplot(x='fbs', hue='target', data=df)

plt.title('Fasting blood sugar > 120 (0=false, 1=true)')

plt.show()
df.groupby('target').mean()
fig, ax = plt.subplots(figsize=(12, 12))

sns.heatmap(df.corr(), annot=True, ax=ax)
df.dtypes
df.nunique()
cat_columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'thal']
df_cat = df.astype(dict((item, 'object') for item in cat_columns))
df_cat.dtypes
df_cat.head()
pd.__version__
df_cat = pd.get_dummies(df_cat, columns=cat_columns)

df_cat.head()
from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier

import xgboost as xgb

import lightgbm as lgb
train = np.array(df.drop('target', axis=1))

y_train = np.array(df['target'])
from sklearn.model_selection import KFold, cross_val_score



n_folds = 5



def get_cv_scores(model, print_scores=True):

    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train)

    accuracy = cross_val_score(model, train, y_train, scoring="accuracy", cv = kf)

    f1_score = cross_val_score(model, train, y_train, scoring="f1", cv = kf)

    roc_auc_score = cross_val_score(model, train, y_train, scoring="roc_auc", cv = kf)

    if print_scores:

        print(f'Accuracy: {accuracy.mean():.3f} ({accuracy.std():.3f})')

        print(f'f1_score: {f1_score.mean():.3f} ({f1_score.std():.3f})')

        print(f'roc_auc_score: {roc_auc_score.mean():.3f} ({roc_auc_score.std():.3f})')

    return [accuracy, f1_score, roc_auc_score]
%%time

lr = LogisticRegression()

get_cv_scores(lr);
%%time

svm = SVC()

get_cv_scores(svm);
%%time

rf = RandomForestClassifier()

get_cv_scores(rf);
%%time

gb = GradientBoostingClassifier()

get_cv_scores(gb);
%%time

et = ExtraTreesClassifier()

get_cv_scores(et);
%%time

xgb_model = xgb.XGBClassifier()

get_cv_scores(xgb_model);
%%time

lgb_model = lgb.LGBMClassifier()

get_cv_scores(lgb_model);
train = np.array(df_cat.drop('target', axis=1))

y_train = np.array(df_cat['target'])
%%time

lr = LogisticRegression()

get_cv_scores(lr);
%%time

xgb_model = xgb.XGBClassifier()

get_cv_scores(xgb_model);
%%time

lgb_model = lgb.LGBMClassifier()

get_cv_scores(lgb_model);
train = np.array(df.drop('target', axis=1))

y_train = np.array(df['target'])
%%time

from sklearn.model_selection import RandomizedSearchCV



params = {

    'C': [0.1, 0.3, 1, 3],

    'max_iter': [50, 100, 200],

}



clf = RandomizedSearchCV(LogisticRegression(), params, cv=5, scoring='accuracy', random_state=1)

clf.fit(train, y_train)

print(clf.best_params_)
%%time

lg_tuned = LogisticRegression(**clf.best_params_)

get_cv_scores(lg_tuned)
%%time



params = {

    'max_depth': [3, 5, 7],

    'n_estimators': [100, 300, 800, 1100],

    'colsample_bytree': [0.5, 0.8, 1],

    'subsample': [0.5, 0.8, 1]

}



clf = RandomizedSearchCV(xgb.XGBClassifier(), params, cv=5, scoring='roc_auc', random_state=1)

clf.fit(train, y_train)

print(clf.best_params_)
%%time

xgb_model_tuned = xgb.XGBClassifier(**clf.best_params_)

get_cv_scores(xgb_model_tuned)
%%time

params = {

    'max_depth': [3, 5, 7, -1],

    'n_estimators': [50, 100, 300, 800, 1100],

    'colsample_bytree': [0.5, 0.8, 1],

}



clf = RandomizedSearchCV(lgb.LGBMClassifier(), params, cv=5, scoring='roc_auc', random_state=1)

clf.fit(train, y_train)

print(clf.best_params_)
%%time

lgb_model_tuned = xgb.XGBClassifier(**clf.best_params_)

get_cv_scores(lgb_model_tuned)
from sklearn.base import BaseEstimator, TransformerMixin, clone, ClassifierMixin



# based on https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard

class StackingAveragedModels(BaseEstimator, ClassifierMixin, TransformerMixin):

    def __init__(self, base_models, meta_model, n_folds=5):

        self.base_models = base_models

        self.meta_model = meta_model

        self.n_folds = n_folds

        

    def fit(self, X, y):

        """Fit all the models on the given dataset"""

        self.base_models_ = [list() for x in self.base_models]

        self.meta_model_ = clone(self.meta_model)

        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)

        

        # Train cloned base models and create out-of-fold predictions

        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))

        for i, model in enumerate(self.base_models):

            for train_index, holdout_index in kfold.split(X, y):

                instance = clone(model)

                self.base_models_[i].append(instance)

                instance.fit(X[train_index], y[train_index])

                y_pred = instance.predict(X[holdout_index])

                out_of_fold_predictions[holdout_index, i] = y_pred

        

        # Train meta-model on out-of-fold predicitions

        self.meta_model_.fit(out_of_fold_predictions, y)

        return self

    

    def predict(self, X):

        meta_features = np.column_stack([

            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)

            for base_models in self.base_models_ ])

        return self.meta_model_.predict(meta_features)

    

    def predict_proba(self, X):

        meta_features = np.column_stack([

            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)

            for base_models in self.base_models_ ])

        return self.meta_model_.predict_proba(meta_features)
%%time

stacked_averaged_model_1 = StackingAveragedModels(base_models=[GradientBoostingClassifier(), xgb.XGBClassifier(),

                                                               lgb.LGBMClassifier(),LogisticRegression()], meta_model=LogisticRegression())

stacked_averaged_model_1.fit(train, y_train)

get_cv_scores(stacked_averaged_model_1);
%%time

stacked_averaged_model_2 = StackingAveragedModels(base_models=[xgb_model_tuned, lgb_model_tuned], 

                                                  meta_model=LogisticRegression())

stacked_averaged_model_2.fit(train, y_train)

get_cv_scores(stacked_averaged_model_1);
import eli5

from eli5.sklearn import PermutationImportance



def get_feature_importance(model, X, y, feature_names):

    perm = PermutationImportance(model, random_state=42).fit(X, y)

    return eli5.show_weights(perm, feature_names=feature_names)
from sklearn.model_selection import train_test_split



train = np.array(df.drop('target', axis=1))

y_train = np.array(df['target'])



X_train, X_test, y_train, y_test = train_test_split(train ,y_train , test_size=0.2, random_state=1)
feature_names = df.drop('target', axis=1).columns.tolist()
lr = LogisticRegression(max_iter=50, C=0.3).fit(X_train, y_train)

get_feature_importance(lr, X_test, y_test, feature_names)
xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=5, colsample_bytree=0.8, subsample=0.5).fit(X_train, y_train)

get_feature_importance(xgb_model, X_test, y_test, feature_names)
lgb_model = lgb.LGBMClassifier(n_estimators=50, max_depth=3, colsample_bytree=1).fit(X_train, y_train)

get_feature_importance(lgb_model, X_test, y_test, feature_names)
!pip install git+https://github.com/SauceCat/PDPbox.git
from pdpbox import pdp, get_dataset, info_plots



from sklearn.model_selection import train_test_split



train = df.drop('target', axis=1)

y_train = df['target']



X_train, X_test, y_train, y_test = train_test_split(train ,y_train , test_size=0.2, random_state=1)
pdp_sex = pdp.pdp_isolate(model=lr, dataset=X_test, model_features=feature_names, feature='sex')



pdp.pdp_plot(pdp_sex, 'Gender', plot_lines=True, frac_to_plot=0.5)

plt.show()
pdp_sex = pdp.pdp_isolate(model=lgb_model, dataset=X_test, model_features=feature_names, feature='sex')



pdp.pdp_plot(pdp_sex, 'Gender', plot_lines=True, frac_to_plot=0.5)

plt.show()
pdp_thal = pdp.pdp_isolate(model=lr, dataset=X_test, model_features=feature_names, feature='thal')



pdp.pdp_plot(pdp_thal, 'Thal', plot_lines=True, frac_to_plot=0.5)

plt.show()
pdp_thal = pdp.pdp_isolate(model=lr, dataset=X_test, model_features=feature_names, feature='ca')



pdp.pdp_plot(pdp_thal, 'ca', plot_lines=True, frac_to_plot=0.5)

plt.show()
features_to_plot = ['age', 'sex']

inter1 = pdp.pdp_interact(model=lr, dataset=X_test, model_features=feature_names, features=features_to_plot)



pdp.pdp_interact_plot(pdp_interact_out=inter1, feature_names=features_to_plot, plot_type='contour')

plt.show()
features_to_plot = ['age', 'sex']

inter1 = pdp.pdp_interact(model=lgb_model, dataset=X_test, model_features=feature_names, features=features_to_plot)



pdp.pdp_interact_plot(pdp_interact_out=inter1, feature_names=features_to_plot, plot_type='contour')

plt.show()
features_to_plot = ['ca', 'sex']

inter1 = pdp.pdp_interact(model=lr, dataset=X_test, model_features=feature_names, features=features_to_plot)



pdp.pdp_interact_plot(pdp_interact_out=inter1, feature_names=features_to_plot, plot_type='contour')

plt.show()
features_to_plot = ['ca', 'sex']

inter1 = pdp.pdp_interact(model=lgb_model, dataset=X_test, model_features=feature_names, features=features_to_plot)



pdp.pdp_interact_plot(pdp_interact_out=inter1, feature_names=features_to_plot, plot_type='contour')

plt.show()
X_test.iloc[0:5]
y_test.iloc[0:5]
lr.predict_proba(np.array(X_test.iloc[0:5]))
lgb_model.predict_proba(np.array(X_test.iloc[0:5]))
import shap



# Create a object that can calculate shap values for our logistic regression model

explainer = shap.TreeExplainer(lgb_model)



# Calculate Shap values

shap_values = explainer.shap_values(np.array(X_test))
shap_values[:,0]
shap.initjs()

shap.force_plot(explainer.expected_value, shap_values[0,:], X_test.iloc[0,:])
shap.force_plot(explainer.expected_value, shap_values[1,:], X_test.iloc[1,:])
shap.force_plot(explainer.expected_value, shap_values[2,:], X_test.iloc[2,:])
shap.dependence_plot('sex', shap_values, X_test)
shap.dependence_plot('ca', shap_values, X_test)
shap.dependence_plot('thalach', shap_values, X_test)
shap.summary_plot(shap_values, X_test)