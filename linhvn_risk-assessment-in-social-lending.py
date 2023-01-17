import warnings

warnings.filterwarnings('ignore')



# Import our libraries we are going to use for our data analysis.

import pandas as pd

import seaborn as sns

import numpy as np

% matplotlib inline

import matplotlib.pyplot as plt

plt.style.use('ggplot')

plt.rcParams['figure.figsize'] = 10, 6

import seaborn as sns



# Other Libraries

from datetime import datetime

from dateutil.parser import parse
df = pd.read_csv('../input/loan.csv', low_memory=False)

df.info()
df.head()
df['loan_status'].value_counts()
df[['issue_d', 'loan_status']]
df = df[df['issue_d'].apply(parse) < parse('Jan-2013')]

df['loan_status'].value_counts().plot.pie(figsize=(7, 7), title='Loan Status', 

                                          shadow=True, startangle=70, autopct='%1.2f%%')
df = df[df['loan_status'].isin(['Fully Paid', 'Default', 'Charged Off'])]

print(df.info())
def good_loan_converter(status):

    return int(status == 'Fully Paid')

    

df['good_loan'] = df['loan_status'].apply(good_loan_converter)

df['good_loan'].value_counts().plot.pie(figsize=(7, 7), title='Good Loan', 

                                             shadow=True, startangle=70, autopct='%1.2f%%', explode=[0,0.25])
df['annual_inc'].describe()
df['annual_inc'].hist()
df['annual_inc_ln'] = np.log(df['annual_inc'])

df['annual_inc_ln'].hist()
df['earliest_cr_line'].describe()
df['earliest_cr_line'].min()
def credit_age_converter(x):

    return parse(x['issue_d']) - parse(x['earliest_cr_line'])



df['credit_age'] = df.apply(credit_age_converter, axis=1) / np.timedelta64(1, 'M')

df['credit_age'].hist()
df['credit_age_ln'] = np.log(df['credit_age'])

df['credit_age_ln'].hist()
df['delinq_2yrs'] =  df['delinq_2yrs'].map(lambda x: x >= 2 and 2 or x)

df['delinq_2yrs'].value_counts().plot.pie(figsize=(5, 5), title='Delinquencies', 

                                             shadow=True, startangle=70, autopct='%1.2f%%')
df['emp_length'].describe()
def emp_length_converter(x):

    r = 0

    try:

        r = int(x[:2])

    except:

        pass

    return r



df['emp_length_int'] = df['emp_length'].apply(emp_length_converter)

df['emp_length_int'].value_counts().plot.pie(figsize=(7, 7), title='Employment Length', 

                                             shadow=True, startangle=70, autopct='%1.2f%%')
dummy = pd.get_dummies(df[['home_ownership']], prefix='home')

home_labels = ['home_MORTGAGE', 'home_OWN', 'home_RENT']

df = pd.concat([df, dummy], axis=1)

df['home_ownership'].value_counts().plot.pie(figsize=(7, 7), title='Home Ownership', 

                                             shadow=True, startangle=70, autopct='%1.2f%%')
df['inq_last_6mths'] = df['inq_last_6mths'].map(lambda x: x >= 3 and 3 or x)

df['inq_last_6mths'] = df['inq_last_6mths'].fillna(0)

df['inq_last_6mths'].value_counts().plot.pie(figsize=(5, 5), title='Inquiries', 

                                             shadow=True, startangle=70, autopct='%1.2f%%')
df['loan_amnt'].hist()
dummy = pd.get_dummies(df[['purpose']], prefix='purpose')

purpose_labels = dummy.columns.get_values().tolist()

purpose_labels.remove('purpose_other')

df = pd.concat([df, dummy], axis=1)

df['purpose'].value_counts().plot.pie(figsize=(10,10), title='Purpose', 

                                             shadow=True, startangle=70, autopct='%1.2f%%')
df[['open_acc', 'total_acc']].hist(figsize=(15,5))
dummy = pd.get_dummies(df[['term']], prefix='term')

term_labels = dummy.columns.get_values().tolist()

df = pd.concat([df, dummy], axis=1)

df['term'].value_counts().plot.pie(figsize=(5, 5), title='Term', 

                                   shadow=True, startangle=70, autopct='%1.2f%%')
df['dti'] = df['dti']/100

df['dti'].hist()
df['itp'] = df['installment']/(df['annual_inc']/12)

df['itp'].hist()
df['revol_util'] = df['revol_util'].fillna(0)

df['revol_util'].hist()
df['rti'] = df['revol_bal']/(df['annual_inc']/12)

df['rti'].describe()
df['rti'].hist()
df[df['revol_bal']==0]['revol_bal'].count()
df['revol_bal'] = df['revol_bal'].map(lambda x: x < 1 and 1 or x)

df['rti_ln'] = np.log(df['revol_bal'])/np.log(df['annual_inc']/12)

df['rti_ln'].hist()
numeric_features = ['annual_inc', 'credit_age', 'delinq_2yrs', 'emp_length_int', 'inq_last_6mths', 'loan_amnt',

                    'open_acc', 'total_acc', 'dti', 'itp', 'revol_util', 'rti']

features = numeric_features + home_labels + purpose_labels + term_labels

print('There are {} features.'.format(len(features)))

y = df['good_loan']

df[features].head()
sns.heatmap(df[features+['good_loan']].corr())
df[features].isnull().any(axis=0) 
from sklearn.model_selection import train_test_split, StratifiedKFold

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, SGDClassifier

from sklearn.preprocessing import PolynomialFeatures

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import GridSearchCV, cross_val_score

from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score



X = df[features].values

X_train, X_holdout, y_train, y_holdout = train_test_split(X, y, test_size=0.3, random_state=17)

# for KNN and LR, we need to scale features

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_holdout_scaled = scaler.transform(X_holdout)



def show_accuracy(model, X, y, model_name):

    pred = model.predict(X)

    print('Accuracy: {0:0.3f}'.format(accuracy_score(y, pred)))



    # predict probabilities & keep probabilities for the positive outcome only

    probs = model.predict_proba(X)[:,1]

    tpr, fpr, _ = roc_curve(y, probs)

    roc_auc = roc_auc_score(y, probs)



    # plot the curve

    plt.plot(tpr, fpr, marker='.')

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.title('{0}: ROC curve AUC={1:0.3f}'.format(model_name, roc_auc))

    plt.show()
%%time

tree = DecisionTreeClassifier(max_depth=5, random_state=17)

tree.fit(X_train, y_train)

show_accuracy(tree, X_holdout, y_holdout, "Decision Tree")
import pydotplus #pip install pydotplus

from sklearn.tree import export_graphviz



def tree_graph_to_png(tree, feature_names, png_file_to_save):

    tree_str = export_graphviz(tree, feature_names=feature_names, 

                                     filled=True, out_file=None)

    graph = pydotplus.graph_from_dot_data(tree_str)  

    graph.write_png(png_file_to_save)



tree_graph_to_png(tree=tree, feature_names=features,

                 png_file_to_save='tree.png')
%%time

tree_params = {'max_depth': range(1,40)}

tree_grid = GridSearchCV(tree, tree_params,cv=5, n_jobs=-1, verbose=True, scoring='roc_auc')



tree_grid.fit(X_train, y_train)
tree_grid.best_params_, tree_grid.best_score_
%%time

show_accuracy(tree_grid, X_holdout, y_holdout, "Decision Tree best model")
tree_graph_to_png(tree=tree_grid.best_estimator_, feature_names=features,

                 png_file_to_save='best_tree.png')
%%time

knn = KNeighborsClassifier(n_neighbors=10)

knn.fit(X_train_scaled, y_train)

show_accuracy(knn, X_holdout_scaled, y_holdout, "KNN")
%%time

knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train_scaled, y_train)

show_accuracy(knn, X_holdout_scaled, y_holdout, "KNN")
%%time

knn_pipe = Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsClassifier(n_jobs=-1))])

knn_params = {'knn__n_neighbors': range(1, 10)}

knn_grid = GridSearchCV(knn_pipe, knn_params,

                        cv=5, n_jobs=-1, verbose=True, scoring='roc_auc')



knn_grid.fit(X_train, y_train)
knn_grid.best_params_, knn_grid.best_score_
%%time

show_accuracy(knn_grid, X_holdout, y_holdout, "KNN best model")
%%time

logit = LogisticRegression(C=1, random_state=17)

logit.fit(X_train_scaled, y_train)

show_accuracy(logit, X_holdout_scaled, y_holdout, "Logistic Regression")
%%time

poly = PolynomialFeatures(degree=3)

X_train_poly = poly.fit_transform(X_train_scaled)

X_holdout_poly = poly.fit_transform(X_holdout_scaled)



logit.fit(X_train_poly, y_train)

show_accuracy(logit, X_holdout_poly, y_holdout, "Logistic Regression Polynominal features")
%%time

sgd_logit = SGDClassifier(loss='log', n_jobs=-1, random_state=17, max_iter=5)

sgd_logit.fit(X_train_scaled, y_train)

show_accuracy(sgd_logit, X_holdout_scaled, y_holdout, "Stochastic Gradient Descent")
%%time

forest = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=17)

print('Cross validation score on training set: {0:0.3f}'.format(np.mean(cross_val_score(forest, X_train, y_train, cv=5))))

forest.fit(X_train, y_train)

show_accuracy(forest, X_holdout, y_holdout, "Random Forest")
%%time

forest = RandomForestClassifier(n_estimators=80, max_depth=40, min_samples_split=5, n_jobs=-1, random_state=17)

print('Cross validation score on training set: {0:0.3f}'.format(np.mean(cross_val_score(forest, X_train, y_train, cv=5))))

forest.fit(X_train, y_train)

show_accuracy(forest, X_holdout, y_holdout, "Random Forest")
%%time

forest_params = {'max_depth': range(1, 40)}



forest_grid = GridSearchCV(forest, forest_params,

                           cv=5, n_jobs=-1, verbose=True, scoring='roc_auc')



forest_grid.fit(X_train, y_train)
forest_grid.best_params_, forest_grid.best_score_
%%time

show_accuracy(forest_grid, X_holdout, y_holdout, "Random Forest best model")
importances = forest.feature_importances_



indices = np.argsort(importances)[::-1]

# Plot the feature importancies of the forest

num_to_plot = 10

feature_indices = [ind+1 for ind in indices[:num_to_plot]]



# Print the feature ranking

print("Feature ranking:")

  

for f in range(num_to_plot):

    print("%d. %s %f " % (f + 1, 

            features[feature_indices[f]], 

            importances[indices[f]]))



plt.figure(figsize=(20,5))

plt.title(u"Feature Importance")

bars = plt.bar(range(num_to_plot), 

               importances[indices[:num_to_plot]],

       color=([str(i/float(num_to_plot+1)) 

               for i in range(num_to_plot)]),

               align="center")

ticks = plt.xticks(range(num_to_plot), feature_indices)

plt.xlim([-1, num_to_plot])

plt.legend(bars, [u''.join(features[i]) 

                  for i in feature_indices]);
selected_features = []

for f in range(num_to_plot):

    selected_features.append(features[feature_indices[f]])



X_selected = df[selected_features].values

X_selected_train, X_selected_holdout, y_train, y_holdout = train_test_split(X_selected, y, test_size=0.3, random_state=17)

X_selected_train_scaled = scaler.fit_transform(X_train)

X_selected_holdout_scaled = scaler.transform(X_holdout)
%%time

logit = LogisticRegression(C=1, random_state=17)

logit.fit(X_selected_train_scaled, y_train)

show_accuracy(logit, X_selected_holdout_scaled, y_holdout, "Logistic Regression")