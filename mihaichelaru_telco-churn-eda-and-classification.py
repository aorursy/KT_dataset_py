import pandas as pd
import numpy as np

pd.set_option('display.width', 100)
pd.set_option('precision', 2)

df = pd.read_csv("../input/WA_Fn-UseC_-Telco-Customer-Churn.csv").drop("customerID", axis=1)
print(df.head(10))
print(df.shape)
print(df.info())
print(df.groupby('Churn').size())
cols =  pd.DataFrame({col_name : sum([int(str(elem).isspace() == True) \
                                 for elem in col]) for col_name, col in df.iteritems()}, index=[0])
cols = cols.rename(index={cols.index[0]: 'blank rows'})
print(cols)
# See null values before and after converstion of whitespace-only chars to np.nan
print(df.isnull().values.sum())
df.replace(r'^\s+$', np.nan, regex=True, inplace=True)
print(df.isnull().values.sum())
df['TotalCharges'] = pd.to_numeric(df.TotalCharges)
print(df.describe())
null_rows = df[df['TotalCharges'].isnull()]
print(null_rows[['MonthlyCharges', 'tenure', 'TotalCharges', 'Churn']])
# Look at the shape before and after to be sure they were removed
print(df.shape)
df = df[df['TotalCharges'].notnull()]
print(df.shape)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for col in df:
    if df[col].dtype == 'object':
        if df[col].nunique() == 2:
            df[col] = le.fit_transform(df[col])
        elif df[col].nunique() > 2:
            df[col] = df[col].astype('category').cat.codes
df_maj = df[df['Churn'] == 0]
df_min = df[df['Churn'] == 1]
from sklearn.utils import resample
df_min_ups = resample(df_min, replace=True, n_samples=5163)
print(df_min.shape)
df_ups = pd.concat([df_min_ups, df_maj])
print(df_ups.Churn.value_counts())
X = df_ups.iloc[:, :-1].values
y = df_ups.iloc[:,-1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
from sklearn import metrics

def print_metrics(y_pred, y_test):
    '''
    Prints accuracy, confusion matrix, and classification report for a given set
    of true and predicted values.
    '''
    print(metrics.accuracy_score(y_test, y_pred))
    print(metrics.confusion_matrix(y_test, y_pred))
    print(metrics.classification_report(y_test, y_pred))
from sklearn.tree import DecisionTreeClassifier, export_graphviz

dt_clf = DecisionTreeClassifier(max_depth=3)
dt_clf.fit(X_train, y_train)
dt_pred = dt_clf.predict(X_test)
print_metrics(dt_pred, y_test)
%matplotlib inline
import pydotplus
import matplotlib.pyplot as plt
import matplotlib.image as img

col_names = df.columns.values.tolist()[:-1]
dot_data = export_graphviz(dt_clf,
                                feature_names=col_names,
                                out_file=None,
                                filled=True,
                                rounded=True)

graph = pydotplus.graphviz.graph_from_dot_data(dot_data)
graph.write_png('telco_tree.png')
plt.figure(figsize=(20,20))
plt.imshow(img.imread(fname='telco_tree.png'))
plt.show()
import plotly.figure_factory as ff

data = []
for i in range(3):
    trace = {
        "type": 'violin',
        "x": df['Contract'][df['Churn'] == i],
        "y": df['MonthlyCharges'][df['Churn'] == i],
        "name": 'Churn' if i == 1 else 'No Churn',
        "box": {
            "visible": True
        },
        "meanline": {
            "visible": True
        }
    }
    data.append(trace)

fig1 = {
    "data": data,
    "layout": {
        "title": "Violin Plot of Contract vs Monthly Charges",
        "yaxis": {
            "zeroline": False,
            "title": "Monthly Charges ($)",
        },
        "xaxis": {
            "title": "Contract",
            "tickvals": [0, 1, 2],
            "ticktext": ['Month-to-month', 'One-year', 'Two-year']
        }
    }
}

from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
iplot(fig1, filename='violin_contract_monchar')
fig2 = ff.create_facet_grid(
    df,
    x='tenure',
    y='MonthlyCharges',
    facet_col='Churn',
)

iplot(fig2, filename='scatter_tenure_monchar')
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
Xs_train, Xs_test, ys_train, ys_test = train_test_split(X_scaled, y, test_size=0.25)
from sklearn.model_selection import GridSearchCV

class ClassifierEvaluator:
    '''
    Wrapper class for applying GridSearchCV to multiple classifiers,
    each with their own set of parameters to search through and evaluate.
    '''
    def __init__(self, classifiers, params):
        self.classifiers = classifiers
        self.params = params
        self.grid_search_results = {}

    def fit(self, X, y, cv=5, n_jobs=5, scoring=None, refit=True, return_train_score=False):
        for clf in self.classifiers.keys():
            print("Running grid search for %s." % clf)
            model = self.classifiers[clf]
            params = self.params[clf]
            grid_search = GridSearchCV(model, params, cv=cv, n_jobs=n_jobs, scoring=scoring,
                                       refit=refit, return_train_score=return_train_score)
            grid_search.fit(X, y)
            self.grid_search_results[clf] = grid_search
        print("Grid search complete.")

    def gs_cv_results(self):
        def iter_results(clf_name, scores, params):
            stat_dict = {
                'classifier' : clf_name,
                'mean_score': np.mean(scores),
                'min_score': min(scores),
                'max_score': max(scores),
                'std_score': np.std(scores),
            }
            return pd.Series({**params, **stat_dict})

        rows = []
        for clf_name, gs_run in self.grid_search_results.items():
            params = gs_run.cv_results_['params']
            scores = []
            for i in range(gs_run.cv):
                key = "split{}_test_score".format(i)
                result = gs_run.cv_results_[key]
                scores.append(result.reshape(len(params),1))

            all_scores = np.hstack(scores)
            for param, score in zip(params, all_scores):
                rows.append((iter_results(clf_name, score, param)))

        df = pd.concat(rows, axis=1, sort=False).T.sort_values(['mean_score'], ascending=False)

        columns = ['classifier', 'mean_score', 'min_score', 'max_score', 'std_score']
        columns = columns + [c for c in df.columns if c not in columns]

        return df[columns]
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

models = {
    'KNeighborsClassifier': KNeighborsClassifier(),
    'RandomForestClassifier' : RandomForestClassifier(),
    'LogisticRegression': LogisticRegression(),
}

params = {
    'KNeighborsClassifier': {'n_neighbors': [1, 3, 5, 10]},
    'RandomForestClassifier': {
                  'n_estimators': [100, 200, 400],
                  'max_depth': [None, 3, 5, 10],
                  'min_samples_split': [2, 5, 10],
                  'max_features': ['sqrt']
    },
    'LogisticRegression': {'C': [1, 10, 100]}
}

evaluator = ClassifierEvaluator(models, params)
evaluator.fit(Xs_train, ys_train)
results = evaluator.gs_cv_results()
import plotly.offline as py
import plotly.graph_objs as go

trace = go.Table(
    header=dict(values=['classifier', 'mean_score', 'C', 'max_depth', 'max_features',
                        'min_samples_split', 'n_estimators', 'n_neighbors'],
                fill = dict(color='#C2D4FF'),
                align = ['left'] * 5),
    cells=dict(values=[results.classifier, results.mean_score,
                       results.C, results.max_depth, results.max_features, results.min_samples_split,
                       results.n_estimators, results.n_neighbors],
               fill = dict(color='#F5F8FF'),
               align = ['left'] * 5))

data = [trace]
iplot(data, filename = 'telco_gridsearchcv_results')
rf = RandomForestClassifier(max_depth=None, max_features='sqrt', min_samples_split=2, n_estimators=100)
rf.fit(Xs_train, ys_train)
y_pred = rf.predict(Xs_test)
print_metrics(y_pred, ys_test)