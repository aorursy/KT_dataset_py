import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn.linear_model import LogisticRegression, Ridge

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import mean_squared_error, accuracy_score

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import PolynomialFeatures

from sklearn.pipeline import make_pipeline

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import warnings

warnings.filterwarnings('ignore')
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
nRowsRead = None # specify 'None' if want to read whole file

df1 = pd.read_csv('/kaggle/input/diabete.csv', delimiter=',', nrows = nRowsRead)

df1.dataframeName = 'diabete.csv'

nRow, nCol = df1.shape

print(f'There are {nRow} rows and {nCol} columns')
df1.head(5)
df1.describe()
# Correlation matrix

def plotCorrelationMatrix(df, graphWidth):

    filename = df.dataframeName

    df = df.dropna('columns') # drop columns with NaN

    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values

    if df.shape[1] < 2:

        print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')

        return

    corr = df.corr()

    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')

    corrMat = plt.matshow(corr, fignum = 1)

    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)

    plt.yticks(range(len(corr.columns)), corr.columns)

    plt.gca().xaxis.tick_bottom()

    plt.colorbar(corrMat)

    plt.title(f'Correlation Matrix for {filename}', fontsize=15)

    plt.show()

    

plotCorrelationMatrix(df1, 8)
y = df1.pop('diabete')

x = df1

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=5)

# CONFIG

alphas = [0, 0.004, 0.2, 1, 5, 20, 100]

criteria = ["entropy","gini"]

MAX_DEPTH = 10

MIN_DEG = 1

MAX_DEG = 5

res = list()



for depth in range(1, MAX_DEPTH+1):

    for c in criteria:

        model_tree = DecisionTreeClassifier(criterion=c, max_depth=depth)

        model_tree.fit(X_train, Y_train)

        predictions = model_tree.predict(X_test)

        res.append((f"Decision tree with criteria={c} and max_depth={depth}",accuracy_score(Y_test, predictions)))

for deg in range(MIN_DEG, MAX_DEG+1):

    model_tree = make_pipeline(PolynomialFeatures(degree=deg), LogisticRegression())

    model_tree.fit(X_train, Y_train)

    predictions = model_tree.predict(X_test)

    res.append((f"Logistic regression with degree={deg}",accuracy_score(Y_test, predictions)))

for deg in range(MIN_DEG, MAX_DEG+1):

    for a in alphas:

        model_reg = make_pipeline(PolynomialFeatures(degree=deg), Ridge(alpha=a))

        model_reg.fit(X_train, Y_train)

        predictions = list(map(lambda p: 0 if p>0.5 else 1, model_reg.predict(X_test)))

        res.append((f"Ridge regression with degree={deg} and alpha={a}",accuracy_score(Y_test, predictions)))



res.sort(key=lambda a: a[1])

plt.figure(figsize=(10,10))

plt.title("Accuracy of different models")

plt.plot(range(0,len(res)), list(map(lambda a: a[1], res)), 'bo-', linewidth=2, markersize=4)

plt.ylabel("Accuracy")

plt.xlabel("Model")

plt.show()

# Legend

for i in range(len(res)):

    print(f"Model {i}: {res[i]}")