import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

import pandas as pd



import warnings

warnings.filterwarnings("ignore")
data = pd.read_csv("../input/bank-marketing-dataset/bank.csv")

print(data.shape)

data.head()
data_explore = data.copy()
data_explore.info()
data_explore.describe()
data_explore['pdays'].value_counts()[-1]
data_explore = data_explore.drop(columns=['pdays'], axis=1)
Q1 = data_explore.quantile(0.25)

Q3 = data_explore.quantile(0.75)

IQR = Q3 - Q1

((data_explore < (Q1 - 1.5 * IQR)) | (data_explore > (Q3 + 1.5 * IQR))).sum()
plt.figure(figsize=(10,5))

plt.subplot(1, 4, 1)

sns.boxplot(x='age', data=data_explore, orient='v')

plt.subplot(1, 4, 2)

sns.boxplot(x='balance', data=data_explore, orient='v')

plt.subplot(1, 4, 3)

sns.boxplot(x='campaign', data=data_explore, orient='v')

plt.subplot(1, 4, 4)

sns.boxplot(x='previous', data=data_explore, orient='v')

plt.tight_layout()
features = list(data_explore.columns)

cat_attrs = [ col for col in features if data_explore[col].dtype=='O' ]

cat_attrs
plt.figure(figsize=(13, 9))

for k in range(len(features)):

    plt.subplot(4, 4, k+1)

    plt.hist(data_explore[features[k]])

    plt.title(features[k], fontsize=12)

    plt.tight_layout()
plt.figure(figsize=(12, 5))

plt.hist(data_explore['job'])

plt.title('Jobs', fontsize=14)

plt.tight_layout()
plt.figure(figsize=(10, 5))

plt.hist(data_explore['month'])

plt.title('Months', fontsize=14)

plt.tight_layout()
sns.boxplot(x="deposit", y="age", hue="deposit", data=data_explore, palette="RdBu")

plt.tight_layout()
plt.figure(figsize=(10, 5))

ax = sns.countplot(x="loan", hue="deposit", data=data_explore)

for p in ax.patches:

        ax.annotate('{}'.format(p.get_height()), (p.get_x()+0.1, p.get_height()+10))

plt.xlabel("Personal Loan?", fontsize=14)

plt.show()
plt.figure(figsize=(10, 5))

ax = sns.countplot(x="housing", hue="deposit", data=data_explore)

for p in ax.patches:

        ax.annotate('{}'.format(p.get_height()), (p.get_x()+0.1, p.get_height()+10))

plt.xlabel("Housing Loan?", fontsize=14)

plt.show()
plt.figure(figsize=(10, 5))

ax = sns.countplot(x="default", hue="deposit", data=data_explore)

for p in ax.patches:

        ax.annotate('{}'.format(p.get_height()), (p.get_x()+0.1, p.get_height()+10))

plt.xlabel("Has Default on Credit?", fontsize=14)

plt.show()
plt.figure(figsize=(10, 5))

ax = sns.countplot(x="marital", hue="deposit", data=data_explore)

for p in ax.patches:

        ax.annotate('{}'.format(p.get_height()), (p.get_x()+0.1, p.get_height()+10))

plt.xlabel('Marital Status', fontsize=14)

plt.show()
plt.figure(figsize=(10, 5))

ax = sns.countplot(x="education", hue="deposit", data=data_explore)

for p in ax.patches:

        ax.annotate('{}'.format(p.get_height()), (p.get_x()+0.1, p.get_height()+50))

plt.show()
plt.figure(figsize=(15, 6))

ax = sns.countplot(x="job", hue="deposit", data=data_explore)

for p in ax.patches:

        ax.annotate('{}'.format(p.get_height()), (p.get_x()+0.1, p.get_height()+10))

plt.xlabel('Job Occupation', fontsize=14)

plt.ylabel("Count", fontsize=14)

plt.show()
plt.figure(figsize=(10, 6))

plt.subplot(1, 2, 1)

sns.boxplot(x="balance", data=data_explore, orient='v')

plt.ylim(-3000, 6000)

plt.tight_layout()

plt.subplot(1, 2, 2)

sns.boxplot(x="deposit", y="balance", data=data_explore)

plt.ylim(-3000, 6000)

plt.tight_layout()
plt.figure(figsize=(13, 6))

sns.boxplot(x="job", y="balance", hue="deposit", data=data_explore)

plt.ylim(-4000, 8000)

plt.tight_layout()
def has_loan(loans):

    a, b, c = loans

    if a=='yes' or b=='yes' or c=='yes':

        return 1

    else:

        return 0

    

data_explore['has_loans'] = data_explore[['default', 'housing', 'loan']].apply(has_loan, axis=1)

data_explore.head()
plt.figure(figsize=(10, 5))

ax = sns.countplot(x="has_loans", data=data_explore)

for p in ax.patches:

        ax.annotate('{}'.format(p.get_height()), (p.get_x()+0.1, p.get_height()+10))

plt.xlabel("Has Loan?", fontsize=14)

plt.show()
plt.figure(figsize=(12, 5))

ax = sns.countplot(x="has_loans", hue="deposit", data=data_explore)

for p in ax.patches:

        ax.annotate('{}'.format(p.get_height()), (p.get_x()+0.1, p.get_height()+10))

plt.xlabel("Has Loan?", fontsize=14)

plt.show()
data_explore_has_loan_deposit = data_explore[(data_explore['has_loans']==1) & (data_explore['deposit']=='yes')]

data_explore_has_loan_deposit.shape
plt.figure(figsize=(15, 15))

plt.subplot(3, 1, 1)

plt.title("Peoples Who Have Loan", fontsize=16)

ax = sns.countplot(x='education', hue='deposit', data=data_explore[data_explore['has_loans']==1])

for p in ax.patches:

        ax.annotate('{}'.format(p.get_height()), (p.get_x()+0.1, p.get_height()+10))



plt.subplot(3, 1, 2)

ax = sns.countplot(x='job', hue='deposit', data=data_explore[data_explore['has_loans']==1])

for p in ax.patches:

        ax.annotate('{}'.format(p.get_height()), (p.get_x()+0.1, p.get_height()+10))

        

plt.subplot(3, 1, 3)

ax = sns.boxplot(x='job', y='balance', hue='deposit', data=data_explore[data_explore['has_loans']==1], orient='v' )

for p in ax.patches:

        ax.annotate('{}'.format(p.get_height()), (p.get_x()+0.1, p.get_height()+10))

plt.ylim(-2500, 7000)

plt.show()
plt.figure(figsize=(15, 15))

plt.subplot(3, 1, 1)

plt.title("Peoples Who Don't Have Loan", fontsize=16)

ax = sns.countplot(x='education', hue='deposit', data=data_explore[data_explore['has_loans']==0])

for p in ax.patches:

        ax.annotate('{}'.format(p.get_height()), (p.get_x()+0.1, p.get_height()+10))



plt.subplot(3, 1, 2)

ax = sns.countplot(x='job', hue='deposit', data=data_explore[data_explore['has_loans']==0])

for p in ax.patches:

        ax.annotate('{}'.format(p.get_height()), (p.get_x()+0.1, p.get_height()+10))

        

plt.subplot(3, 1, 3)

ax = sns.boxplot(x='job', y='balance', hue='deposit', data=data_explore[data_explore['has_loans']==0], orient='v' )

for p in ax.patches:

        ax.annotate('{}'.format(p.get_height()), (p.get_x()+0.1, p.get_height()+10))

plt.ylim(-2500, 7000)

plt.show()
plt.figure(figsize=(10, 5))

ax = sns.countplot(x="contact", hue="deposit", data=data_explore)

for p in ax.patches:

        ax.annotate('{}'.format(p.get_height()), (p.get_x()+0.1, p.get_height()+10))

plt.ylabel("Count", fontsize=14)

plt.xlabel("Communication Type", fontsize=14)

plt.show()
print("Average contact duration with perosn who has subscribed: ", data_explore[data_explore['deposit']=='yes']['duration'].mean()/60)

print("Average contact duration with perosn who hasn't subscribed: ", data_explore[data_explore['deposit']=='no']['duration'].mean()/60)
plt.figure(figsize=(10, 6))

plt.subplot(1, 2, 1)

sns.boxplot(x="duration", data=data_explore, palette="RdBu", orient='v')

plt.ylim(top=1600)

plt.tight_layout()

plt.subplot(1, 2, 2)

sns.boxplot(x="deposit", y="duration", data=data_explore, palette="RdBu")

plt.ylim(top=1600)

plt.tight_layout()
plt.figure(figsize=(12, 10))

plt.subplot(2, 1, 1)

sns.countplot(x='campaign', hue='deposit', data=data_explore)

plt.xlim(right=10)

plt.xlabel('')

plt.subplot(2, 1, 2)

sns.countplot(x='campaign', hue='deposit', data=data_explore)

plt.xlim(left=11)

plt.ylim(top=30)

plt.xlabel('# of Campaign', fontsize=14)

plt.show()
plt.figure(figsize=(10, 5))

ax = sns.countplot(x="poutcome", hue="deposit", data=data_explore)

for p in ax.patches:

        ax.annotate('{}'.format(p.get_height()), (p.get_x()+0.1, p.get_height()+10))

plt.show()
plt.figure(figsize=(15, 6))

ax1 = plt.subplot(1, 3, 1)

sns.scatterplot(x='age', y='campaign', hue='poutcome', data=data_explore[(data_explore['deposit']=='no') & (data_explore['poutcome']!='unknown')],)

plt.title("Unsubscribed", fontsize=14)

plt.subplot(1, 3, 2, sharey=ax1)

sns.scatterplot(x='age', y='campaign', hue='poutcome', data=data_explore[(data_explore['deposit']=='yes') & (data_explore['poutcome']!='unknown')], )

plt.title("Subscribed", fontsize=14)
plt.figure(figsize=(12, 5))

ax = sns.countplot(x="month", hue="deposit", data=data_explore)

for p in ax.patches:

        ax.annotate('{}'.format(p.get_height()), (p.get_x()+0.1, p.get_height()+10))

plt.show()
from sklearn.preprocessing import LabelEncoder



label_enc = LabelEncoder()

for cat in cat_attrs:

    data_explore[cat] = label_enc.fit_transform(data_explore[cat])



data_explore.head()
corr_matrix = data_explore.corr()



plt.figure(figsize=(17, 12))

sns.heatmap(corr_matrix, mask=np.zeros_like(corr_matrix, dtype=np.bool), square=True, annot=True, cbar=False)
X = data.drop(columns=['deposit'], axis=1)

y = data['deposit'].copy()

y = y.apply(lambda x: 0 if x=='no' else 1)
list(X.columns)
feature_columns = list(X.columns)

cat_attrs = [ col for col in feature_columns if X[col].dtype=='O' ]

num_attrs = [ col for col in feature_columns if not col in cat_attrs ]

num_attrs.remove('pdays')

cat_attrs, num_attrs
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train.shape, X_test.shape, y_train.shape, y_test.shape
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.preprocessing import OneHotEncoder

from sklearn.impute import SimpleImputer

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline
class AddCustomAttribute(BaseEstimator, TransformerMixin):

    def __init__(self):

        pass

    

    def fit(self, X, y=None):

        return self

    

    def transform(self, X):

        default_idx, housing_idx, loan_idx = cat_attrs.index('default'), cat_attrs.index('housing'), cat_attrs.index('loan')

        has_loan_attr = (X[:, default_idx]=='yes') | (X[:, housing_idx]=='yes') | (X[:, loan_idx]=='yes')

        X = np.delete(X, (default_idx, housing_idx, loan_idx), axis=1)

        return np.c_[X, has_loan_attr]
cat_pipeline = Pipeline([('cat_imputer', SimpleImputer(strategy='most_frequent')),

                        ('add_attrs', AddCustomAttribute()),

                        ('encoder', OneHotEncoder(handle_unknown='ignore'))])



pre_process = ColumnTransformer([('drop_attrs', 'drop', ['pdays']),

                                 ('cat_process', cat_pipeline, cat_attrs),

                                 ('num_process', SimpleImputer(strategy='mean'), num_attrs)], remainder='passthrough')
X_train_transformed = pre_process.fit_transform(X_train)

X_test_transformed = pre_process.transform(X_test)
X_train_transformed.shape, X_test_transformed.shape
cat_attrs.remove('loan')

cat_attrs.remove('housing')

cat_attrs.remove('default')

cat_attrs.append('has_loan')



all_cat_attrs = list(pre_process.transformers_[1][1]['encoder'].get_feature_names(cat_attrs))
feature_columns = all_cat_attrs + num_attrs

len(feature_columns), feature_columns
from sklearn.model_selection import GridSearchCV, KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)



def grid_search(model, grid_param):

    print("Obtaining Best Model for {}".format(model.__class__.__name__))

    grid_search = GridSearchCV(model, grid_param, cv=kf, scoring='roc_auc', return_train_score=True, n_jobs=-1)

    grid_search.fit(X_train_transformed, y_train)

    

    print("Best Parameters: ", grid_search.best_params_)

    print("Best Scores: ", grid_search.best_score_)

    

    cvres = grid_search.cv_results_

    print("\nResults for each run of {}...".format(model.__class__.__name__))

    for train_mean_score, test_mean_score, params in zip(cvres["mean_train_score"], cvres["mean_test_score"], cvres["params"]):

        print(train_mean_score, test_mean_score, params)

        

    return grid_search.best_estimator_
from sklearn.metrics import roc_auc_score, roc_curve

from sklearn.model_selection import cross_val_score



results = dict()



np.set_printoptions(precision=4)



def plot_roc_curve(model, X=X_test_transformed, y_true=y_test):

    y_scores = model.predict(X)

    auc_score = np.round(roc_auc_score(y_true, y_scores), 4)

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)

    plt.plot(fpr, tpr, linewidth=2, label=model.__class__.__name__+"(AUC Score: "+str(auc_score)+")")

    plt.plot([0, 1], [0, 1], 'k--') # Dashed diagonal

    plt.axis([0, 1, 0, 1])

    plt.xlabel("FPR", fontsize=16)

    plt.ylabel("TPR", fontsize=16)

    plt.legend()

    



    

def performance_measures(model, store_results=True):

    

    test_acc = cross_val_score(model, X_test_transformed, y_test, cv=kf, n_jobs=-1, scoring='accuracy')

    test_acc = np.around(test_acc, decimals=4)

    mean_test_acc = np.around(np.mean(test_acc), decimals=4)

    sd_test_acc = np.around(np.std(test_acc), decimals=4)

    print("CV Test Accuracy Scores: ", test_acc)

    print("Mean Accuracy: {} (S.D = {})".format(mean_test_acc, sd_test_acc))

    

    test_f1 = cross_val_score(model, X_test_transformed, y_test, cv=kf, n_jobs=-1, scoring='f1')

    test_f1 = np.around(test_f1, decimals=4)

    mean_test_f1 = np.around(np.mean(test_f1), decimals=4)

    sd_test_f1 = np.around(np.std(test_f1), decimals=4)

    print("\nCV Test F1 Scores: ", test_f1)

    print("Mean F1: {} (S.D = {})".format(mean_test_f1, sd_test_f1))

     

    if store_results:

        results[model.__class__.__name__] = (mean_test_acc*100, sd_test_acc*100,  mean_test_f1*100, sd_test_f1*100)
from sklearn.linear_model import LogisticRegression
logistic_clf = LogisticRegression(solver='liblinear', random_state=42, n_jobs=-1)

logistic_param_grid = [{'C':[0.01, 0.1, 1, 10], 'penalty':['l1', 'l2']}]
logistic_clf = grid_search(logistic_clf, logistic_param_grid)
feature_importance = []

for feature_imp in zip(feature_columns, logistic_clf.coef_[0]):

    feature_importance.append(feature_imp)

    

feature_importance.sort(key=lambda a:a[1], reverse=True)

feature_importance[:10]
from sklearn.ensemble import RandomForestClassifier
forest_clf = RandomForestClassifier(n_estimators=250, random_state=42, n_jobs=-1)

forest_param_grid = [{'max_depth':[8, 12, 16, 20], 'max_features':[None, 'sqrt', 'auto']}]
forest_clf = grid_search(forest_clf, forest_param_grid)
forest_clf.max_depth=8

forest_clf.max_features='auto'

forest_clf.fit(X_train_transformed, y_train)
feature_importance = []

for feature_imp in zip(feature_columns, forest_clf.feature_importances_):

    feature_importance.append(feature_imp)

    

feature_importance.sort(key=lambda a:a[1], reverse=True)

feature_importance[:10]
from sklearn.ensemble import GradientBoostingClassifier
gb_clf = GradientBoostingClassifier(n_estimators=250, loss='deviance', random_state=42)

gb_param_grid = [{'max_depth':[3, 8, 16], 'max_features':[None, 'sqrt', 'auto']}]
gb_clf = grid_search(gb_clf, gb_param_grid)

gb_clf
gb_clf.max_depth=3

gb_clf.max_features='auto'

gb_clf.fit(X_train_transformed, y_train)
feature_importance = []

for feature_imp in zip(feature_columns, gb_clf.feature_importances_):

    feature_importance.append(feature_imp)

    

feature_importance.sort(key=lambda a:a[1], reverse=True)

feature_importance[:10]
from xgboost import XGBClassifier
xgb_clf = XGBClassifier(n_estimators=250, random_state=42, n_jobs=-1)

xgb_param_grid = [{'max_depth':[4, 8, 16], 'learning_rate':[0.01, 0.1, 1]}]
xgb_clf = grid_search(xgb_clf, xgb_param_grid)
feature_importance = []

for feature_imp in zip(feature_columns, xgb_clf.feature_importances_):

    feature_importance.append(feature_imp)

    

feature_importance.sort(key=lambda a:a[1], reverse=True)

feature_importance[:10]
print('\n Logistic Regression : CV Results')

performance_measures(logistic_clf)



print("--"*30)

print('\n Random Forest : CV Results')

performance_measures(forest_clf)



print("--"*30)

print('\n Gradient Boost : CV Results')

performance_measures(gb_clf)



print("--"*30)

print('\n XGBoost : CV Results')

performance_measures(xgb_clf)
models =  list(results.keys())

result = list(results.values())

test_mean_acc=[]

test_sd_acc=[]

test_mean_f1=[]

test_sd_f1=[]



for res in result:

    test_mean_acc.append(res[0])

    test_sd_acc.append(res[1])

    test_mean_f1.append(res[2])

    test_sd_f1.append(res[3])
plt.figure(figsize=(7, 4))

plot_roc_curve(logistic_clf)

plot_roc_curve(forest_clf)

plot_roc_curve(gb_clf)

plot_roc_curve(xgb_clf)

plt.title("ROC Curve", fontsize=14)

plt.show()
plt.figure(figsize=(12, 4))

x_indexes = np.arange(len(models))     

width = 0.15                            



plt.bar(x_indexes - width,  test_mean_acc, label="Mean Test Accuracy (S.D.)", width=width)

for i in range(len(x_indexes)):

    label=str(test_mean_acc[i])[:6]+" ({:.3f})".format(test_sd_acc[i])

    plt.text(x=x_indexes[i]-width, y=test_mean_acc[i]+0.3, s=label, fontsize=12)



plt.bar(x_indexes,  test_mean_f1, label="Mean F1 Score (S.D.)", width=width)

for i in range(len(x_indexes)):

    label=str(test_mean_f1[i])[:6]+"({:.3f})".format(test_sd_f1[i])

    plt.text(x=x_indexes[i], y=test_mean_f1[i]+0.1, s=label, fontsize=12)

    

plt.ylim(75, 85)

plt.ylabel("%", fontsize=14)

plt.legend(loc="upper left", fontsize=12)

plt.xticks(ticks=x_indexes, labels=models, fontsize=12)

plt.show()
feature_importance = []

for feature_imp in zip(feature_columns, xgb_clf.feature_importances_):

    feature_importance.append(feature_imp)

    

feature_importance.sort(key=lambda a:a[1], reverse=True)

feature_importance[:10]
y_train_pred = xgb_clf.predict(X_train_transformed)

y_test_pred = xgb_clf.predict(X_test_transformed)

y_pred = np.concatenate([y_train_pred, y_test_pred], axis=0)



y_true = np.concatenate([y_train, y_test], axis=0)

y_pred.shape, y_true.shape
combine_data = pd.concat([X_train, X_test], axis=0)

combine_data.shape
combine_data['deposit'] = y_true

combine_data['predictions'] = y_pred

combine_data['has_loan'] = combine_data[['default', 'housing', 'loan']].apply(has_loan, axis=1)
combine_data.head()
plt.figure(figsize=(15, 4))

plt.subplot(1, 2, 1)

ax = sns.countplot(x='deposit', data=combine_data)

for p in ax.patches:

        ax.annotate('{:.0f}'.format(p.get_height()), (p.get_x()+0.1, p.get_height()+50))

plt.title("Observed Subscibers")

plt.subplot(1, 2, 2)

ax = sns.countplot(x='predictions', data=combine_data)

for p in ax.patches:

        ax.annotate('{:.0f}'.format(p.get_height()), (p.get_x()+0.1, p.get_height()+50))

plt.title("Predicted Subscibers")

plt.show()
plt.figure(figsize=(15, 4))

plt.subplot(1, 2, 1)

ax = sns.countplot(x='has_loan', hue='deposit', data=combine_data)

for p in ax.patches:

        ax.annotate('{:.0f}'.format(p.get_height()), (p.get_x()+0.1, p.get_height()+50))

plt.title("Observed Subscibers", fontsize=14)



plt.subplot(1, 2, 2)

ax = sns.countplot(x='has_loan', hue='predictions', data=combine_data)

for p in ax.patches:

        ax.annotate('{:.0f}'.format(p.get_height()), (p.get_x()+0.1, p.get_height()+50))

plt.title("Predicted Subscibers", fontsize=14)

plt.show()
plt.figure(figsize=(14, 10))

plt.subplot(2, 1, 1)

ax = sns.countplot(x='job', hue='deposit', data=combine_data)

for p in ax.patches:

        ax.annotate('{:.0f}'.format(p.get_height()), (p.get_x()+0.1, p.get_height()+10))

plt.title("Observed Subscibers", fontsize=14)

plt.subplot(2, 1, 2)

ax = sns.countplot(x='job', hue='predictions', data=combine_data)

for p in ax.patches:

        ax.annotate('{:.0f}'.format(p.get_height()), (p.get_x()+0.1, p.get_height()+10))

plt.title("Predicted Subscibers", fontsize=14)

plt.show()
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)

ax = sns.countplot(x='education', hue='deposit', data=combine_data)

for p in ax.patches:

        ax.annotate('{:.0f}'.format(p.get_height()), (p.get_x()+0.1, p.get_height()+10))

plt.title("Observed Subscibers", fontsize=14)

plt.subplot(1, 2, 2)

ax = sns.countplot(x='education', hue='predictions', data=combine_data)

for p in ax.patches:

        ax.annotate('{:.0f}'.format(p.get_height()), (p.get_x()+0.1, p.get_height()+10))

plt.title("Predicted Subscibers", fontsize=14)

plt.show()
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)

ax = sns.countplot(x='contact', hue='deposit', data=combine_data)

plt.title("Observed Subscibers", fontsize=14)

for p in ax.patches:

        ax.annotate('{:.0f}'.format(p.get_height()), (p.get_x()+0.1, p.get_height()+10))

plt.subplot(1, 2, 2)

ax = sns.countplot(x='contact', hue='predictions', data=combine_data)

for p in ax.patches:

        ax.annotate('{:.0f}'.format(p.get_height()), (p.get_x()+0.1, p.get_height()+10))

plt.title("Predicted Subscibers", fontsize=14)

plt.show()