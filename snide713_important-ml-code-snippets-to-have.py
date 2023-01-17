%load_ext autoreload

%autoreload 2



%matplotlib inline



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import math



import warnings



from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"
gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")

test = pd.read_csv("../input/titanic/test.csv")

train = pd.read_csv("../input/titanic/train.csv")

train.head()

train.describe(include='all')

train.info()
def convert_size(size_bytes):

    if size_bytes == 0:

        return "0B"

    size_name = ("Bytes", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")

    i = int(math.floor(math.log(size_bytes, 1024)))

    p = math.pow(1024, i)

    s = round(size_bytes / p, 2)

    return "%s %s" % (s, size_name[i])

convert_size(train.memory_usage().sum())
# Let’s plot the distribution of each feature

def plot_distribution(dataset, cols=5, width=20, height=15, hspace=0.2, wspace=0.5):

    plt.style.use('seaborn-whitegrid')

    fig = plt.figure(figsize=(width,height))

    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=wspace, hspace=hspace)

    rows = math.ceil(float(dataset.shape[1]) / cols)

    for i, column in enumerate(dataset.columns):

        ax = fig.add_subplot(rows, cols, i + 1)

        ax.set_title(column)

        if dataset.dtypes[column] == np.object:

            g = sns.countplot(y=column, data=dataset)

            substrings = [s.get_text()[:18] for s in g.get_yticklabels()]

            g.set(yticklabels=substrings)

            plt.xticks(rotation=25)

        else:

            g = sns.distplot(dataset[column])

            plt.xticks(rotation=25)

plot_distribution(train.drop(labels=['Cabin', 'Name', 'Ticket'], axis=1).dropna(), cols=3, width=20, height=20, hspace=0.45, wspace=0.5)
train['Survived'] = train.Survived.astype(str)



# Plot a count of the categories from each categorical feature split by our prediction class: salary - predclass.

def plot_bivariate_bar(dataset, hue, cols=5, width=20, height=15, hspace=0.2, wspace=0.5):

    dataset = dataset.select_dtypes(include=[np.object])

    plt.style.use('seaborn-whitegrid')

    fig = plt.figure(figsize=(width,height))

    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=wspace, hspace=hspace)

    rows = math.ceil(float(dataset.shape[1]) / cols)

    for i, column in enumerate(dataset.columns):

        ax = fig.add_subplot(rows, cols, i + 1)

        ax.set_title(column)

        if dataset.dtypes[column] == np.object:

            g = sns.countplot(y=column, hue=hue, data=dataset)

            substrings = [s.get_text()[:10] for s in g.get_yticklabels()]

            g.set(yticklabels=substrings)

            

plot_bivariate_bar(train.drop(labels=['Cabin', 'Name', 'Ticket'], axis=1).dropna(), hue='Survived', cols=3, width=20, height=12, hspace=0.4, wspace=0.5)
def add_datepart(df, fldname, drop=True):

    fld = df[fldname]

    if not np.issubdtype(fld.dtype, np.datetime64):

        df[fldname] = fld = pd.to_datetime(fld, 

                                     infer_datetime_format=True)

    targ_pre = re.sub('[Dd]ate$', '', fldname)

    for n in ('Year', 'Month', 'Week', 'Day', 'Dayofweek', 

            'Dayofyear', 'Is_month_end', 'Is_month_start', 

            'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 

            'Is_year_start'):

        df[targ_pre+n] = getattr(fld.dt,n.lower())

    df[targ_pre+'Elapsed'] = fld.astype(np.int64) // 10**9

    if drop: df.drop(fldname, axis=1, inplace=True)
def train_cats(df):

    for col in df.columns:

        if df[col].dtype == 'O' : 

            df[col] = df[col].astype('category').cat.as_ordered()
def fix_missing(df):

    for col in df.columns:

        if (df[col].dtype == 'int64') or (df[col].dtype == 'float64') or (df[col].dtype == 'bool'):

            if df[col].isnull().sum() != 0:

                df[col + '_na'] = df[col].isnull()

                df[col] = df[col].fillna(df[col].median())

        else:

            df[col + '_coded'] = df[col].cat.codes +1

            df.drop(columns=[col], axis=1, inplace=True)
def tree_visual(model, X_training, y_training):

    model.fit(X_training, y_training)

    # Extract single tree

    estimator = model.estimators_[0]



    from sklearn.externals.six import StringIO  

    from IPython.display import Image  

    from sklearn.tree import export_graphviz

    import pydotplus

    dot_data = StringIO()

    export_graphviz(estimator, out_file=dot_data, feature_names=X_training.columns,  

                    filled=True, rounded=True,

                    special_characters=True)

    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  

    

    return Image(graph.create_png())
total = train.isnull().sum().sort_values(ascending=False)

percent = ((train.isnull().sum()/train.isnull().count())*100).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

f, ax = plt.subplots(figsize=(15, 6))

plt.xticks(rotation='90')

sns.barplot(x=missing_data.index, y=missing_data['Percent'])

plt.xlabel('Features', fontsize=15)

plt.ylabel('Percent of missing values', fontsize=15)

plt.title('Percent missing data by feature', fontsize=15)

missing_data.head()
train['Survived'] = train.Survived.astype(int)



corr = train.select_dtypes(exclude=np.object).corr()

corr.style.background_gradient(cmap='coolwarm').set_precision(2)
def correlation_heatmap(df):

    _ , ax = plt.subplots(figsize =(20, 20))

    colormap = sns.diverging_palette(200, 10, as_cmap = True)

    

    _ = sns.heatmap(

        df.corr(), 

        cmap = colormap,

        square=True,

        cbar_kws={'shrink':.9 },

        ax=ax,

        annot=True, 

        linewidths=0.1,vmax=1.0, linecolor='white',

        annot_kws={'fontsize':12 }

    )

    

    plt.title('Pearson Correlation of Features', y=1.05, size=15)



correlation_heatmap(train.select_dtypes(exclude=np.object))
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score

data = train.select_dtypes(exclude=np.object).dropna()

k_range = list(range(1, 26))

k_scores = []

for k in k_range:

    knn = KNeighborsClassifier(n_neighbors=k)

    scores = cross_val_score(knn, data.drop('Survived', axis=1), data.Survived, cv=10, scoring='accuracy')

    mean_score = scores.mean()

    k_scores.append(mean_score)



plt.plot(k_range, k_scores)

plt.xlabel('Value of k')

plt.ylabel('Accuracy')
from sklearn.model_selection import GridSearchCV



X = data.drop('Survived', axis=1)

y = data.Survived

k_range = list(range(1, 31))

weight_options = ['uniform', 'distance']

param_grid = dict(n_neighbors=k_range, weights=weight_options)



grid = GridSearchCV(knn, param_grid, scoring='accuracy', cv=10)

grid.fit(X, y)



grid_result = pd.DataFrame(grid.cv_results_)

grid_result.head()

grid.best_score_

grid.best_params_

grid.best_estimator_