import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import LinearSVC

from sklearn.model_selection import cross_val_score, train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.neural_network import MLPClassifier



import xgboost as xgb



import warnings

warnings.filterwarnings('ignore')



import os

os.listdir('../input')

brainwave_df = pd.read_csv('../input/emotions.csv')
brainwave_df.head()
brainwave_df.tail()
brainwave_df.shape
brainwave_df.dtypes
plt.figure(figsize=(12,5))

sns.countplot(x=brainwave_df.label, color='lightblue')

plt.title('Emotional sentiment class distribution', fontsize=16)

plt.ylabel('Class Counts', fontsize=16)

plt.xlabel('Class Label', fontsize=16)

plt.xticks(rotation='vertical');
label_df = brainwave_df['label']

brainwave_df.drop('label', axis = 1, inplace=True)
corr = brainwave_df.corr(method='pearson')

plt.figure(figsize=(100, 100))

corrMat = plt.matshow(corr, fignum = 1)

plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)

plt.yticks(range(len(corr.columns)), corr.columns)

plt.gca().xaxis.tick_bottom()

plt.colorbar(corrMat)

plt.title('Correlation Matrix', fontsize=15)

plt.show()
def plotScatterMatrix(df, plotSize, textSize):

    df = df.select_dtypes(include =[np.number]) # keep only numerical columns

    # Remove rows and columns that would lead to df being singular

    df = df.dropna('columns')

    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values

    columnNames = list(df)

    if len(columnNames) > 10: # reduce the number of columns for matrix inversion of kernel density plots

        columnNames = columnNames[:10]

    df = df[columnNames]

    ax = pd.plotting.scatter_matrix(df, alpha=0.75, figsize=[plotSize, plotSize], diagonal='kde')

    corrs = df.corr().values

    for i, j in zip(*plt.np.triu_indices_from(ax, k = 1)):

        ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', ha='center', va='center', size=textSize)

    plt.suptitle('Scatter and Density Plot')

    plt.show()
plotScatterMatrix(brainwave_df, 20, 10)
skew = brainwave_df.skew()

skew
%%time



pl_random_forest = Pipeline(steps=[('random_forest', RandomForestClassifier())])

scores = cross_val_score(pl_random_forest, brainwave_df, label_df, cv=10,scoring='accuracy')

print('Accuracy for RandomForest : ', scores.mean())