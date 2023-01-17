import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import warnings



from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score, roc_curve, confusion_matrix

from mlxtend.classifier import StackingClassifier

from xgboost import XGBClassifier



warnings.filterwarnings('ignore')

defaultcolor = '#66ccff'

pd.options.display.float_format = '{:.2f}'.format

rc={'savefig.dpi': 75, 'figure.autolayout': False, 'figure.figsize': [12, 8], 'axes.labelsize': 18,\

   'axes.titlesize': 18, 'font.size': 18, 'lines.linewidth': 2.0, 'lines.markersize': 8, 'legend.fontsize': 16,\

   'xtick.labelsize': 16, 'ytick.labelsize': 16}

sns.set(style='ticks',rc=rc)

sns.set_palette('husl')
df = pd.read_csv('../input/heart.csv')

df.head()
df.describe()
fig, ax = plt.subplots()

sns.countplot(df.target, ax=ax)

for i,p in enumerate(ax.patches):

    ax.annotate('{:.2f}%'.format(df['target'].value_counts().apply(lambda x: 100*x/df['target'].value_counts().sum())[i]), (p.get_x()+0.32, p.get_height()+1)).set_fontsize(15)

ax.set_ylabel("")

ax.set_xlabel("")

ax.set_title("Target distribution");
fig, ax = plt.subplots(figsize=[15,15])

df.hist(ax=ax, bins=30, color='b');
fig, ax = plt.subplots(figsize=[20,15])

sns.heatmap(df.corr(), ax=ax, cmap='Blues', annot=True);

ax.set_title("Pearson correlation coefficients", size=20);
fig, ax = plt.subplots()

df.groupby(['age', 'target']).size().reset_index().pivot(index='age', columns='target', values=0).fillna(0).plot.bar(stacked=True, ax=ax)

ax.set_title("Distribution of the target according to the age")

ax.set_xlabel("");
fig, ax = plt.subplots()

sns.scatterplot(x='age', y='thalach', data=df[df.target==1], color='b', ax=ax)

sns.scatterplot(x='age', y='thalach', data=df[df.target==0], color='r', ax=ax)

ax.legend(['1', '0']);
sns.heatmap(df.groupby(['exang', 'cp']).size().reset_index().pivot(columns='exang', index='cp', values=0), cmap='Blues', fmt='g', annot=True);
fig, ax = plt.subplots()

sns.boxplot(x='slope', y='thalach', data=df, ax=ax);

ax.set_title("Thalach distribution by slope values");
sns.violinplot(x='cp', y='thalach', data=df);
sns.boxplot(x='slope', y='oldpeak', data=df);
X_train, X_test, y_train, y_test = train_test_split(df.drop('target', axis=1), df.target, test_size=0.2, random_state=56)
models = {

    'CART': DecisionTreeClassifier(),

    'SVC': SVC(probability=True),

    'XGB': XGBClassifier(n_jobs=-1),

    'GNB': GaussianNB(),

    'LDA': LinearDiscriminantAnalysis(),

    'LR': LogisticRegression(),

    'KNN': KNeighborsClassifier()

}
def cv_report(models, X, y):

    results = []

    for name in models.keys():

        model = models[name]

        scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

        print("Accuracy: %.3f (+/- %.3f) [%s]" %(scores.mean(), scores.std(), name))
cv_report(models, X_train, y_train)
xgb_params = {

    'max_depth': [2,3,4],

    'n_estimators': [50, 100, 400, 1000],

    'learning_rate': [0.1, 0.01, 0.05]

}



xg_grid = GridSearchCV(models['XGB'], xgb_params, cv=5)

models['XGB_Grid'] = xg_grid
cv_report(models, X_train, y_train)
lr_params = [{

                'penalty': ['l2'],

                'C': (0.1, 0.5, 1.0, 1.5, 2.0),

                'solver': ['newton-cg', 'lbfgs', 'sag'],

                'max_iter': [50, 100, 200, 500]

            },

            {

                'penalty': ['l1', 'l2'],

                'C': (0.1, 0.5, 1.0, 1.5, 2.0),

                'solver': ['liblinear', 'saga']

            }

]



lr_grid = GridSearchCV(models['LR'], lr_params, cv=5)

models['LR_Grid'] = lr_grid
cv_report(models, X_train, y_train)
models['LR_Grid'].fit(X_train, y_train)
predictions = models['LR_Grid'].predict(X_test)
print("Accuracy of the model: {:.2f}%".format(100*accuracy_score(predictions, y_test)))
fig, ax = plt.subplots()

ax.set_title("Confusion Matrix")

sns.heatmap(confusion_matrix(y_test, predictions), annot=True, cmap='Blues');