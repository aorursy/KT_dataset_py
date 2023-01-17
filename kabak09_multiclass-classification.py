# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



#from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



frame = pd.read_csv('../input/Iris.csv')



pd.DataFrame({'Data Type': frame.dtypes, 'Null Value': pd.isnull(frame).any(), 

              'Count': list(map(lambda column: len(frame[column].unique()), frame.columns))})
%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns



frame_dummy = pd.get_dummies(frame.drop('Id', axis=1))



plt.figure(figsize=(10,8))

sns.heatmap(frame_dummy.corr().round(decimals=5), cmap='bwr_r', annot=True, linewidths= 2,

            annot_kws={'size': 15}, cbar_kws={'ticks': [-1,-0.5,0,0.5,1]}, vmin=-1, vmax=1)



from sklearn.model_selection import train_test_split



unique_cat = set(frame['Species'].unique())

frame['Species_'] = pd.Categorical(frame['Species'], categories=unique_cat).codes



y = frame['Species_']

X = frame.drop(['Species', 'Species_', 'Id'], axis=1)



X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
from sklearn.decomposition import PCA

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler





pca_pipe = Pipeline(steps=[('norm', StandardScaler()), ('pca', PCA(n_components=2))]).fit(X_train)

pc = pca_pipe.transform(X_train)

pc_1, pc_2 = list(zip(*pc))



groups = pd.DataFrame({'pc_1' : pc_1, 'pc_2': pc_2, 'y': y_train}).groupby('y')



plt.figure(figsize=(10,8))

for group, color, name in zip(frame['Species_'].unique(), ['#0080ff', '#ff6600', '#9966ff'], 

                              frame['Species'].unique()):

    pc_group = groups.get_group(group)

    plt.scatter(pc_group.loc[:,'pc_1'], pc_group.loc[:,'pc_2'], color=color, label=name)

plt.legend()
from sklearn.model_selection import validation_curve

from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_auc_score



clf = LogisticRegression()

clf_pipe = Pipeline(steps=[('poly', PolynomialFeatures(degree=2)), ('norm', MinMaxScaler()), 

                           ('clf', clf)])



param_name='C'

param_range=np.logspace(-4,1, 50)



train_score, test_score = validation_curve(clf_pipe, X_train, y_train, scoring = 'accuracy',

                                           param_name='clf__{}'.format(param_name),

                                           param_range=param_range)



train_max = list(map(np.max, train_score))

train_min = list(map(np.min, train_score))

train_mean = list(map(np.mean, train_score))



test_max = list(map(np.max, test_score))

test_min = list(map(np.min, test_score))

test_mean = list(map(np.mean, test_score))



plt.figure(figsize=(10,8))

for min_val, max_val, mean_val, color, name in zip([train_min, test_min],

                                             [train_max, test_max],

                                             [train_mean, test_mean],

                                             ['blue', 'orange'], 

                                             ['Train Score', 'Test Score']):



    plt.plot(param_range, mean_val, color=color, label=name)

    plt.fill_between(param_range, max_val, min_val, color=color, alpha=0.2)



plt.xlabel('Hyper Parameter ({})'.format(param_name))

plt.ylabel('Score (Accuracy)')

sns.despine()





def multi_class_roc(class_code):



    clf_pipe.fit(X_train, y_train)

    return roc_auc_score(np.where(y_test==class_code, 1,0), 

                  [val[class_code] for val in clf_pipe.predict_proba(X_test)])



print('roc auc score for iris virginica = {}'.format(multi_class_roc(0)))

print('roc auc score for iris versicolor = {}'.format(multi_class_roc(1)))

print('roc auc score for iris setosa = {}'.format(multi_class_roc(2)))

print('average roc auc score = {}'.format(np.mean([multi_class_roc(0),

                                                  multi_class_roc(1),

                                                  multi_class_roc(2)])))  
plt.figure(figsize=(10,8))

for group, color, name in zip(frame['Species_'].unique(), ['#0080ff', '#ff6600', '#9966ff'], 

                              frame['Species'].unique()):

    pc_group = groups.get_group(group)

    plt.scatter(pc_group.loc[:,'pc_1'], pc_group.loc[:,'pc_2'], color=color, label=name)

plt.legend()



pc_1 = np.array(pc_1)

pc_2 = np.array(pc_2)



pc_1_min = np.min(pc_1) - 0.5

pc_1_max = np.max(pc_1) + 0.5

pc_2_min = np.min(pc_2) - 0.5

pc_2_max = np.max(pc_2) + 0.5



pc_1_ , pc_2_ = np.meshgrid(np.arange(pc_1_min, pc_1_max, 0.0025), 

                            np.arange(pc_2_min, pc_2_max, 0.0025))



clf0 = LogisticRegression(C=15)

clf_pipe0 = Pipeline(steps=[('poly', PolynomialFeatures(degree=3)), ('norm', MinMaxScaler()), 

                           ('clf', clf0)]).fit(pc, y_train)

prediction = clf_pipe0.predict(np.c_[pc_1_.ravel(), pc_2_.ravel()])

prediction = prediction.reshape(pc_1_.shape)



from matplotlib.colors import ListedColormap



cmap=ListedColormap(['#ff6600', '#0080ff', '#9966ff'])

plt.contourf(pc_1_, pc_2_, prediction, alpha=0.2, cmap=cmap)

plt.xlabel('First Principle Component')

plt.ylabel('Second Principle Component')
