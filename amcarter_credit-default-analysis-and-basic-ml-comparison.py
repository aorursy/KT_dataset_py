# standard imports



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set_style('whitegrid')
# sklearn imports

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split



# from sklearn.preprocessing import StandardScaler

from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.tree import DecisionTreeClassifier
df = pd.read_excel('data.xls', sheetname='Data', skiprows=1, index_col='ID')

df.head()
# check for nulls and correct data import type - expect all rows to be int64

df.info()
# examine distribution of values

fig = plt.figure(figsize=(10, 10))

ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=3)

ax2 = plt.subplot2grid((3, 3), (1, 0),)

ax3 = plt.subplot2grid((3, 3), (1, 1),)

ax4 = plt.subplot2grid((3, 3), (1, 2),)

ax5 = plt.subplot2grid((3, 3), (2, 0), colspan=3)

sns.distplot(df.LIMIT_BAL, ax=ax1);

sns.countplot(df.SEX, ax=ax2)

sns.countplot(df.EDUCATION, ax=ax3)

sns.countplot(df.MARRIAGE, ax=ax4)

sns.distplot(df.AGE, ax=ax5);

plt.tight_layout()

plt.suptitle('Exploratory analysis of uncleaned data',y=1.02,fontsize=16,weight='bold');
# create dictionaries of the known data types,rename final column for ease of access by name

sexdict = {1:'Male', 2:'Female'}

edudict = {1:'Grad School', 2:'University', 3:'High School'}

marriagedict = {1:'Married', 2:'Single'}

df.rename(columns={'default payment next month':'default'}, inplace=True)
# Filter the dataset to exclude undefined categorical values for education and marriage



edulist= [1,2,3]

marriagelist = [1, 2]

df = df[df['EDUCATION'].isin(edulist)]

df = df[df['MARRIAGE'].isin(marriagelist)]



# Apply names from the dictionaries above

df.SEX = df.SEX.map(sexdict)

df.EDUCATION = df.EDUCATION.map(edudict)

df.MARRIAGE = df.MARRIAGE.map(marriagedict)
# Convert the continuous distributions of age and credit limit into binned values

# Benefits ease of analysis and machine learning algorithms

agebins = np.arange(20, 80, 10)

agebinlabels = ['{}s'.format(i, j) for i, j in zip(agebins, agebins[1:])]

df['AGE_GROUP'] = pd.cut(df.AGE, bins=agebins, labels=agebinlabels, right=False)



creditlimitbins = np.arange(0, 550000, 50000)

creditbinlabels = ['{}-{}k'.format(i//1000, j//1000) for i, j in zip(creditlimitbins, creditlimitbins[1:])]

df['LIMIT_BAL_GROUP'] = pd.cut(df.LIMIT_BAL, bins=creditlimitbins, labels=creditbinlabels)
# examine distribution of values following data cleaning

fig = plt.figure(figsize=(10, 10))

ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=3)

ax2 = plt.subplot2grid((3, 3), (1, 0),)

ax3 = plt.subplot2grid((3, 3), (1, 1),)

ax4 = plt.subplot2grid((3, 3), (1, 2),)

ax5 = plt.subplot2grid((3, 3), (2, 0), colspan=3)

plotlist = ['LIMIT_BAL_GROUP', 'SEX', 'MARRIAGE', 'EDUCATION', 'AGE_GROUP']

for col, ax in zip(plotlist, fig.get_axes()):

    sns.countplot(x=col, data=df, ax=ax)

plt.suptitle('Exploratory analysis of cleaned data', y=1.01, fontsize=16, weight='bold')

plt.tight_layout()
# construct figure from pointplots

fig = plt.figure(figsize=(10, 10))

ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=3)

ax2 = plt.subplot2grid((3, 3), (1, 0), sharey=ax1)

ax3 = plt.subplot2grid((3, 3), (1, 1), sharey=ax1)

ax4 = plt.subplot2grid((3, 3), (1, 2), sharey=ax1)

ax5 = plt.subplot2grid((3, 3), (2, 0), colspan=3, sharey=ax1)

plotlist = ['LIMIT_BAL_GROUP', 'SEX', 'MARRIAGE', 'EDUCATION', 'AGE_GROUP']

for col, ax in zip(plotlist, fig.get_axes()):

    sns.pointplot(x=col, y='default', data=df, ax=ax, markers='')

    ax.set_ylabel('Correlation with default')

ax1.set_ylim(0.0, 1)

plt.suptitle('Correlation with default for each variable', y=1.01, fontsize=16, weight='bold')

plt.tight_layout()
ax1 = sns.countplot(x='PAY_2', data=df)

ax2 = ax1.twinx()

ax2 = sns.pointplot(x='PAY_2', y='default', data=df, zorder=10, ax=ax2)

ax1.grid(False)

ax2.grid(False)

plt.suptitle('Risk of default with payment delay in months');
payment_delay_cols = ['PAY_2', 'PAY_3', 'PAY_4', 'PAY_5']

fig, ax = plt.subplots(1, 4, figsize=(12, 4), sharey=True)

for i,(column, ax) in enumerate(zip(payment_delay_cols, ax)):

    sns.pointplot(x=column, y='default', data=df[df[column]!=1], ax=ax, color=sns.color_palette()[i])

plt.suptitle('Risk of default with payment delay across four past periods');
cutoff_point = {'PAY_2': 2, 'PAY_3': 2, 'PAY_4': 2, 'PAY_5': 2}

for color, column in enumerate(cutoff_point.keys()):

    df[column+'_TEST'] =  df[column].map(lambda x: 'Delay' if x >= cutoff_point[column] else 'Paid')

    sns.pointplot(x=column+'_TEST', y='default', data=df, order=['Paid', 'Delay'], color=sns.color_palette()[color])

plt.xlabel('')

plt.ylim(0, 1);

plt.suptitle('Risk of default with a timely payment or delay beyond a 2 or 3 month cut off');
def compare4models():

    """ returns a figure based from four machine learning models"""

    

    names = ["Nearest Neighbors",

             "Linear SVM",

             "Decision Tree",

             "Naive Bayes"]



    classifiers = [KNeighborsClassifier(),

                   SVC(),

                   DecisionTreeClassifier(),

                   GaussianNB()]

    

    cmaps = ['Reds',

             'Greens',

             'Blues',

             'Oranges']

    

    numrows = int(np.ceil(len(names)/2))

    fig, ax = plt.subplots(nrows=numrows, ncols=2, figsize=(8, numrows*4))

    

    for name,clf,ax,cmap in zip(names, classifiers, ax.ravel(), cmaps):

        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        score = clf.score(X_test, y_test)

        confmatrix = confusion_matrix(y_test, y_pred)

        true0,true1 = [sum(confmatrix[i]) for i in [0,1]]

        pred0,pred1 = [sum(i) for i in zip(*confmatrix)]

        ylabels=['Not default: {}'.format(true0), 'Default: {}'.format(true1)]

        xlabels=['Not default: {}'.format(pred0), 'Default: {}'.format(pred1)]

        sns.heatmap(confusion_matrix(y_test, y_pred),

                    annot=True,

                    xticklabels=xlabels,

                    yticklabels=ylabels,

                    fmt='g',

                    ax=ax,

                    vmax=len(X_test),

                    vmin=0,

                    cbar=False,

                    cmap=cmap)

        ax.set_xlabel('Truth')

        ax.set_ylabel('Predicted')

        ax.set_title('{} (Score: {})'.format(name, np.round(score, decimals=3)), size=14)

    return fig,confmatrix
# One-hot encoding to produce a sparse matrix suitable for input to sklearn

modeldata = df[['LIMIT_BAL_GROUP', 'AGE_GROUP', 'SEX', 'MARRIAGE', 'EDUCATION']]

X = pd.get_dummies(modeldata)

y = df.default

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=31)
fig,confmatrix = compare4models()

plt.suptitle('Comparison of 4 algorithms - Personal Data only', y=1.02, size=16, weight='bold')

plt.tight_layout()
# One-hot encoding to produce a sparse matrix suitable for input to sklearn

# Add extra columns with payment history data

modeldata = df[['LIMIT_BAL_GROUP', 'AGE_GROUP', 'SEX', 'MARRIAGE', 'EDUCATION',

                'PAY_2_TEST', 'PAY_3_TEST', 'PAY_4_TEST', 'PAY_5_TEST']]

X = pd.get_dummies(modeldata)

y = df.default

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=31)
compare4models()

plt.suptitle('Comparison of 4 algorithms - Personal and Payment History data', y=1.02, size=16, weight='bold')

plt.tight_layout()