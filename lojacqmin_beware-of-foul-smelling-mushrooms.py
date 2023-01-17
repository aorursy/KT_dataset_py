import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import itertools

import scipy.stats as ss

from sklearn.model_selection import train_test_split, RandomizedSearchCV

from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.metrics import accuracy_score, classification_report

from sklearn.tree import DecisionTreeClassifier, plot_tree

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier
df = pd.read_csv("../input/mushroom-classification/mushrooms.csv")

print('Name of the attributes: \n{}'.format(df.columns))
print('Size of the dataset: {}'.format(df.shape))
description_df = df.describe()

description_df
for col in description_df:

    if description_df[col].loc['unique'] < 2:

        print(('Deleted column: {}'.format(col)))

        df = df.drop([col], axis=1)
col_names = list(df.columns)

num_missing = (df[col_names].isnull()).sum()

print('There are {} columns with missing values.'.format(len([i for i in num_missing if i > 0])))
le = LabelEncoder()

label_df = pd.DataFrame()

for col in df.columns:

    label_df[col] = le.fit_transform(df[col])
onehot_df = pd.get_dummies(df)

onehot_df.head()
print(label_df.info())

print(onehot_df.info())
binary_class = pd.Series(df['class']).value_counts().sort_index()

sns.barplot(binary_class.index, binary_class.values)

plt.ylabel("Count")

plt.xlabel("Class")

plt.title('Number of poisonous/edible mushrooms (e = edible, p = poisonous)');
plt.figure(figsize=(14,12))

sns.heatmap(label_df.corr(),linewidths=.1,cmap="BrBG", annot=True)

plt.title('Pearson Correlation of Features', y=1.05, size=15)

plt.yticks(rotation=0);
df_long = pd.melt(label_df, 'class', var_name='Characteristics')

fig, ax = plt.subplots(figsize=(12,6))

plot = sns.violinplot(ax=ax, x='Characteristics', y='value', hue='class', split=True, data=df_long, inner='quartile')

df_no_class = label_df.drop(['class'], axis=1)

plot.set_xticklabels(rotation=90, labels=list(df_no_class.columns));
def cramers_corrected_stat(x,y):

    

    """ calculate Cramers V statistic for categorial-categorial association.

        uses correction from Bergsma and Wicher, 

        Journal of the Korean Statistical Society 42 (2013): 323-328

    """

    conf_matrix=pd.crosstab(x, y)

    chi2 = ss.chi2_contingency(conf_matrix)[0]

    n = sum(conf_matrix.sum())

    phi2 = chi2/n

    r,k = conf_matrix.shape

    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    

    rcorr = r - ((r-1)**2)/(n-1)

    kcorr = k - ((k-1)**2)/(n-1)

    result = np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))

    return round(result,6)



corrM = np.zeros((len(col_names), len(col_names)))

for col1, col2 in itertools.combinations(col_names, 2):

    idx1, idx2 = col_names.index(col1), col_names.index(col2)

    corrM[idx1, idx2] = cramers_corrected_stat(label_df[col1], label_df[col2])

    corrM[idx2, idx1] = corrM[idx1, idx2]



corr = pd.DataFrame(corrM, index=col_names, columns=col_names)

corr['class'].plot(kind='bar', figsize=(8,6), title="Association between the binary class and each variable");
new_var = df[['class', 'gill-color']]

sns.catplot('class', col='gill-color', data=new_var, kind='count', height=2.5, aspect=.8, col_wrap=4, order=['e', 'p']);
new_var = df[['class', 'spore-print-color']]

sns.catplot('class', col='spore-print-color', data=new_var, kind='count', height=2.5, aspect=.8, col_wrap=4, order=['e', 'p']);
new_var = df[['class', 'odor']]

sns.catplot('class', col='odor', data=new_var, kind='count', height=2.5, aspect=.8, col_wrap=4, order=['e', 'p']);
X = onehot_df.drop(['class_e', 'class_p'], axis=1)

y = df['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
# fit the model

model = LogisticRegression()

model.fit(X_train, y_train)

# evaluate the model

y_pred = model.predict(X_test)

# evaluate predictions

print('Accuracy (all features): %.2f' % (accuracy_score(y_test, y_pred)))
# feature selection

def select_features(X_train, y_train, X_test, method):

    fs = SelectKBest(score_func=method, k=25)

    fs.fit(X_train, y_train)

    X_train_fs = fs.transform(X_train)

    X_test_fs = fs.transform(X_test)

    return X_train_fs, X_test_fs, fs
# feature selection

X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test, chi2)

# what are scores for the features

feature_scores = [tup for tup in zip(fs.scores_, X.columns)]

print('Top 10 chi-squared features: \n')

for i, j in sorted(feature_scores, key=lambda tup: tup[0], reverse=True)[:10]:

    print('%s: %f' % (j, i))

# plot the scores

plt.bar([i for i in range(len(fs.scores_))], fs.scores_)

plt.title('Chi-square Feature Importance')

plt.show()
# fit the model

model = LogisticRegression()

model.fit(X_train_fs, y_train)

# evaluate the model

y_pred = model.predict(X_test_fs)

# evaluate predictions

print('Accuracy (chi-squared features): %.2f' % (accuracy_score(y_test, y_pred)))
# feature selection

X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test, mutual_info_classif)

# what are scores for the features

feature_scores = [tup for tup in zip(fs.scores_, X.columns)]

print('Top 10 Mutual Information features: \n')

for i, j in sorted(feature_scores, key=lambda tup: tup[0], reverse=True)[:10]:

    print('%s: %f' % (j, i))

# plot the scores

plt.bar([i for i in range(len(fs.scores_))], fs.scores_)

plt.title('Mutual Information Feature Importance')

plt.show()
# fit the model

model = LogisticRegression()

model.fit(X_train_fs, y_train)

# evaluate the model

y_pred = model.predict(X_test_fs)

# evaluate predictions

print('Accuracy (Mutual Information features): %.2f' % (accuracy_score(y_test, y_pred)))
descr = label_df.describe()

descr.loc['std'].sort_values(ascending=False)
models = [SVC(kernel='rbf', random_state=0), SVC(kernel='linear', random_state=0), XGBClassifier(), LogisticRegression(), KNeighborsClassifier()]

model_names = ['SVC_rbf', 'SVC_linear', 'xgboost', 'Logistic Regression', 'KNeighbors Classifier']

for i, model in enumerate(models):

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print ('Accurancy of ' + model_names[i] + ': ' + str(accuracy_score(y_test, y_pred)))
feature_scores = [tup for tup in zip(models[3].coef_[0], X.columns)]

print('Top 10 Logistic Regression features: \n')

for i, j in sorted(feature_scores, key=lambda tup: tup[0], reverse=True)[:10]:

    print('%s: %f' % (j, i))
feature_scores = [tup for tup in zip(models[2].feature_importances_, X.columns)]

print('Top 10 xgboost features: \n')

for i, j in sorted(feature_scores, key=lambda tup: tup[0], reverse=True)[:10]:

    print('%s: %f' % (j, i))
tree_clf = DecisionTreeClassifier().fit(X_train, y_train)

print ('Accuracy of Decision Tree Classifier: ' + str(accuracy_score(y_test, tree_clf.predict(X_test))))



importances = tree_clf.feature_importances_

feature_scores = [tup for tup in zip(importances, X.columns)]

print('Top 10 Decision Tree features: \n')

sorted_scores = sorted(feature_scores, key=lambda tup: tup[0], reverse=True)

for i, j in sorted_scores[:10]:

    print('%s: %f' % (j, i))

    

# plot the scores

plt.bar([j for i, j in sorted_scores[:13]], sorted(importances, reverse=True)[:13])

plt.title('Decision Tree Feature Importance')

plt.xticks(rotation=90)

plt.show()
print('Double-click on the image to enlarge it')

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(6,6), dpi=600)

plot_tree(tree_clf,

            feature_names=list(X_train.columns), 

            class_names=['poisonous', 'edible'],

            filled=True);
RR_model = RandomForestClassifier()



parameters = {'min_samples_leaf': range(10,100,10), 

                    'n_estimators' : range(10,100,10),

                    'criterion' : ['gini', 'entropy'],

                    'max_features':['auto','sqrt','log2']

                    }



RR_model = RandomizedSearchCV(RR_model, parameters, cv=10, scoring='accuracy', n_iter=20, n_jobs=-1)

RR_model.fit(X_train,y_train)
y_pred = RR_model.predict(X_test)

print(classification_report(y_test, y_pred))
scores_df = pd.DataFrame(RR_model.cv_results_)

scores_df = scores_df.sort_values(by=['rank_test_score']).reset_index(drop='index')

scores_df.head()