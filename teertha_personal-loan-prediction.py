import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



import numpy as np

import pandas as pd



import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns

import warnings



from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_predict

from sklearn import metrics

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.metrics import precision_recall_curve

from sklearn.metrics import auc

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier



pd.plotting.register_matplotlib_converters()

%matplotlib inline

plt.style.use('seaborn-whitegrid')

pd.set_option('display.max_columns', 500)

warnings.filterwarnings("ignore")
# Read the Data

data = pd.read_csv('../input/personal-loan-modeling/Bank_Personal_Loan_Modelling.csv')

data.head(3)
# let's explore the shape of the data. 

data.shape
# Let's Check if the data contains any missing or NaN values.

data.isnull().any()
data.info()
data.drop(['ID', 'ZIP Code'], axis = 1, inplace = True)
data.isnull().sum()
# Dividing the columns in the dataset in to numeric and categorical attributes.

cols = set(data.columns)

cols_numeric = set(['Age', 'Experience', 'Income', 'CCAvg', 'Mortgage'])

cols_categorical = list(cols - cols_numeric)

cols_categorical
for x in cols_categorical:

    data[x] = data[x].astype('category')



data.info()
data.describe().transpose()
data_num = data.select_dtypes(include='number')

data_cat = data.select_dtypes(include='category')

print(f'Numerical Attributes: {list(data_num.columns)}')

print(f'Categorical Attributes: {list(data_cat.columns)}')
# Let's construct a function that shows the summary and density distribution of a numerical attribute:

def summary(x):

    x_min = data[x].min()

    x_max = data[x].max()

    Q1 = data[x].quantile(0.25)

    Q2 = data[x].quantile(0.50)

    Q3 = data[x].quantile(0.75)

    print(f'5 Point Summary of {x.capitalize()} Attribute:\n'

          f'{x.capitalize()}(min) : {x_min}\n'

          f'Q1                    : {Q1}\n'

          f'Q2(Median)            : {Q2}\n'

          f'Q3                    : {Q3}\n'

          f'{x.capitalize()}(max) : {x_max}')



    fig = plt.figure(figsize=(16, 10))

    plt.subplots_adjust(hspace = 0.6)

    sns.set_palette('pastel')

    

    plt.subplot(221)

    ax1 = sns.distplot(data[x], color = 'r')

    plt.title(f'{x.capitalize()} Density Distribution')

    

    plt.subplot(222)

    ax2 = sns.violinplot(x = data[x], palette = 'Accent', split = True)

    plt.title(f'{x.capitalize()} Violinplot')

    

    plt.subplot(223)

    ax2 = sns.boxplot(x=data[x], palette = 'cool', width=0.7, linewidth=0.6)

    plt.title(f'{x.capitalize()} Boxplot')

    

    plt.subplot(224)

    ax3 = sns.kdeplot(data[x], cumulative=True)

    plt.title(f'{x.capitalize()} Cumulative Density Distribution')

    

    plt.show()
summary('Age')
summary('Experience')
summary('Income')
summary('CCAvg')
summary('Mortgage')
# Create a function that returns a Pie chart and a Bar Graph for the categorical variables:

def cat_view(x = 'Education'):

    """

    Function to create a Bar chart and a Pie chart for categorical variables.

    """

    from matplotlib import cm

    color1 = cm.inferno(np.linspace(.4, .8, 30))

    color2 = cm.viridis(np.linspace(.4, .8, 30))

    

    fig, ax = plt.subplots(1, 2, figsize=(16, 6))

    

     

    """

    Draw a Pie Chart on first subplot.

    """    

    s = data.groupby(x).size()



    mydata_values = s.values.tolist()

    mydata_index = s.index.tolist()



    def func(pct, allvals):

        absolute = int(pct/100.*np.sum(allvals))

        return "{:.1f}%\n({:d})".format(pct, absolute)





    wedges, texts, autotexts = ax[0].pie(mydata_values, autopct=lambda pct: func(pct, mydata_values),

                                      textprops=dict(color="w"))



    ax[0].legend(wedges, mydata_index,

              title="Index",

              loc="center left",

              bbox_to_anchor=(1, 0, 0.5, 1))



    plt.setp(autotexts, size=12, weight="bold")



    ax[0].set_title(f'{x.capitalize()} Piechart')

    

    """

    Draw a Bar Graph on second subplot.

    """

    

    df = pd.pivot_table(data, index = [x], columns = ['Personal Loan'], values = ['Income'], aggfunc = len)



    labels = df.index.tolist()

    loan_no = df.values[:, 0].tolist()

    loan_yes = df.values[:, 1].tolist()

    

    l = np.arange(len(labels))  # the label locations

    width = 0.35  # the width of the bars



    rects1 = ax[1].bar(l - width/2, loan_no, width, label='No Loan', color = color1)

    rects2 = ax[1].bar(l + width/2, loan_yes, width, label='Loan', color = color2)



    # Add some text for labels, title and custom x-axis tick labels, etc.

    ax[1].set_ylabel('Scores')

    ax[1].set_title(f'{x.capitalize()} Bar Graph')

    ax[1].set_xticks(l)

    ax[1].set_xticklabels(labels)

    ax[1].legend()

    

    def autolabel(rects):

        

        """Attach a text label above each bar in *rects*, displaying its height."""

        

        for rect in rects:

            height = rect.get_height()

            ax[1].annotate('{}'.format(height),

                        xy=(rect.get_x() + rect.get_width() / 2, height),

                        xytext=(0, 3),  # 3 points vertical offset

                        textcoords="offset points",

                        fontsize = 'large',   

                        ha='center', va='bottom')





    autolabel(rects1)

    autolabel(rects2)



    fig.tight_layout()

    plt.show()
cat_view('Family')
cat_view('Education')
cat_view('Securities Account')
cat_view('CD Account')
cat_view('Online')
splot = sns.countplot(x = 'Personal Loan', data = data)



for p in splot.patches:

    splot.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()), 

                   ha = 'center', va = 'center', xytext = (0, 5), textcoords = 'offset points')
X = data.drop('Personal Loan', axis = 1)

Y = data[['Personal Loan']]
corr = X.corr()

plt.figure(figsize=(10, 8))

g = sns.heatmap(corr, annot=True, cmap = 'summer_r', square=True, linewidth=1, cbar_kws={'fraction' : 0.02})

g.set_yticklabels(g.get_yticklabels(), rotation=0, horizontalalignment='right')

bottom, top = g.get_ylim()

g.set_ylim(bottom + 0.5, top - 0.5)

plt.show()
# Let's plot all Dependent variables to see their inter-relations.

sns.pairplot(X, diag_kind = 'kde', vars = list(data_num.columns))
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=1, stratify = Y)
from sklearn.feature_selection import mutual_info_classif

mutual_information = mutual_info_classif(X_train, y_train, n_neighbors=5, copy = True)



plt.subplots(1, figsize=(26, 1))

sns.heatmap(mutual_information[:, np.newaxis].T, cmap='Blues', cbar=False, linewidths=1, annot=True, annot_kws={"size": 20})

plt.yticks([], [])

plt.gca().set_xticklabels(X_train.columns, rotation=45, ha='right', fontsize=16)

plt.suptitle("Variable Importance (mutual_info_classif)", fontsize=22, y=1.2)

plt.gcf().subplots_adjust(wspace=0.2)
from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=42)

rf_clf.fit(X_train, y_train)



features = list(X_train.columns)

importances = rf_clf.feature_importances_

indices = np.argsort(importances)



fig, ax = plt.subplots(figsize=(10, 7))

plt.barh(range(len(indices)), importances[indices], color='b', align='center')

ax.tick_params(axis="x", labelsize=12)

ax.tick_params(axis="y", labelsize=14)

plt.yticks(range(len(indices)), [features[i] for i in indices])

plt.xlabel('Relative Importance', fontsize = 18)
# from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import OneHotEncoder

from sklearn.compose import ColumnTransformer



X_train_num = X_train.select_dtypes(include='number')

X_train_cat = X_train.select_dtypes(include='category')



num_attribs = list(X_train_num.columns)

cat_attribs = list(X_train_cat.columns)



transformer = ColumnTransformer([

        ("num", StandardScaler(), num_attribs),

        ("cat", OneHotEncoder(), cat_attribs),

    ])



X_train = transformer.fit_transform(X_train)

print(X_train.shape)

X_train[1, :]
y_train = np.array(y_train)

print(y_train.shape)
def train_model(model):

    m = model[1]

    y_train_pred = cross_val_predict(model[1], X_train, y_train, cv=5)

    cm = confusion_matrix(y_train, y_train_pred)

    print('Confusion matrix: ' + model[0])

    print(cm)

    print()

    accuracy = accuracy_score(y_train, y_train_pred)

    precision = precision_score(y_train, y_train_pred)

    recall = recall_score(y_train, y_train_pred)

    f1 = f1_score(y_train, y_train_pred)

    print(f'{model[0]} Accuracy: {accuracy}')

    print(f'{model[0]} Precision: {precision}')

    print(f'{model[0]} Recall: {recall}')

    print(f'{model[0]} f1 - score: {f1}')
train_model(('Gaussian Naive Bayes', GaussianNB()))
train_model(('Logistic Regression', LogisticRegression(solver="liblinear")))
train_model(('k Nearest Neighbor', KNeighborsClassifier(n_neighbors= 7, weights = 'distance' )))
train_model(('SVM', SVC(gamma='auto')))
train_model(('CART', DecisionTreeClassifier()))
train_model(('Random Forest', RandomForestClassifier(n_estimators=100, random_state=42)))
from sklearn.model_selection import GridSearchCV



param_grid = [

    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},

    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},

  ]



forest_reg = RandomForestClassifier(random_state=42)

# train across 5 folds, that's a total of (12+6)*5=90 rounds of training 

grid_search = GridSearchCV(forest_reg, param_grid, cv=5,

                           scoring='f1',

                           return_train_score=True)

grid_search.fit(X_train, y_train)
grid_search.best_params_
rf_clf = grid_search.best_estimator_
X_test = transformer.fit_transform(X_test)

print(X_test.shape)
y_test = np.array(y_test)

print(y_test.shape)
rf_clf.fit(X_test, y_test)
y_test_predict = rf_clf.predict(X_test)
rf_clf.score(X_test, y_test)
print(metrics.classification_report(y_test, y_test_predict, labels=[1, 0]))