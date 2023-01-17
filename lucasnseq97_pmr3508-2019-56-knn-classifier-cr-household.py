import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline



import seaborn as sns
df_train = pd.read_csv("../input/costa-rican-household-poverty-prediction/train.csv")
df_train.head()
features = ['escolari', 'SQBmeaned', 'hogar_nin', 'hogar_total', 'area1',

            'lugar1', 'cielorazo', 'pisonotiene', 'v14a', 'abastaguano', 'v2a1',

            'hacdor', 'rez_esc', 'meaneduc', 'SQBovercrowding', 'abastaguadentro',

            'tipovivi1', 'Target']
base = df_train[features]
base.head()
base = base.astype(np.float)

base.shape
corrmat = base.corr()

sns.set()

plt.figure(figsize=(13,9))

sns.heatmap(corrmat)
total = base.isnull().sum().sort_values(ascending = False)

percent = ((base.isnull().sum()/base.isnull().count())*100).sort_values(ascending = False)

missing_data = pd.concat([total, percent], axis = 1, keys = ['Total', '%'])

missing_data.head()
base = base.drop(['v2a1', 'rez_esc'], axis = 1)
base['SQBmeaned'].plot(kind = 'box')
base['meaneduc'].plot(kind = 'box')
col = 'SQBmeaned'

base[col] = base[col].fillna(base[col].describe().mean())



col = 'meaneduc'

base[col] = base[col].fillna(base[col].describe().mean())
total = base.isnull().sum().sort_values(ascending = False)

percent = ((base.isnull().sum()/base.isnull().count())*100).sort_values(ascending = False)

missing_data = pd.concat([total, percent], axis = 1, keys = ['Total', '%'])

missing_data.head()
base.head()
cols = ['escolari', 'SQBmeaned', 'area1', 'lugar1', 'hogar_nin', 'hogar_total', 'SQBovercrowding']



sns.set()

sns.pairplot(base, hue = 'Target', vars = cols)
var2 = 'escolari'

var1 = 'Target'



data = pd.concat([df_train[var2], df_train[var1]], axis=1)



f, ax = plt.subplots(figsize=(10, 8))



sns.boxplot(x=var1, y=var2, data=data)

plt.title('Boxplot of escolari over Target')
var2 = 'hogar_nin'

var1 = 'Target'



data = pd.concat([df_train[var2], df_train[var1]], axis=1)



f, ax = plt.subplots(figsize=(10, 8))



sns.boxplot(x=var1, y=var2, data=data)

plt.title('Boxplot of hogar_nin over Target')
var2 = 'SQBmeaned'

var1 = 'Target'



data = pd.concat([df_train[var2], df_train[var1]], axis=1)



f, ax = plt.subplots(figsize=(10, 8))



sns.boxplot(x=var1, y=var2, data=data)

plt.title('Boxplot of escolari over Target')
base.hist(column='cielorazo', by ='Target', figsize=(10,10), color = 'coral')
base.hist(column='pisonotiene', by ='Target', figsize=(10,10), color = 'coral')
base.hist(column='v14a', by ='Target', figsize=(10,10), color = 'coral')
# Author: https://www.kaggle.com/willkoehrsen/a-complete-introduction-and-walkthrough



def plot_categoricals(x, y, data, annotate = True):

    """Plot counts of two categoricals.

    Size is raw count for each grouping.

    Percentages are for a given value of y."""

    

    # Raw counts 

    raw_counts = pd.DataFrame(data.groupby(y)[x].value_counts(normalize = False))

    raw_counts = raw_counts.rename(columns = {x: 'raw_count'})

    

    # Calculate counts for each group of x and y

    counts = pd.DataFrame(data.groupby(y)[x].value_counts(normalize = True))

    

    # Rename the column and reset the index

    counts = counts.rename(columns = {x: 'normalized_count'}).reset_index()

    counts['percent'] = 100 * counts['normalized_count']

    

    # Add the raw count

    counts['raw_count'] = list(raw_counts['raw_count'])

    

    plt.figure(figsize = (14, 10))

    # Scatter plot sized by percent

    plt.scatter(counts[x], counts[y], edgecolor = 'k', color = 'lightgreen',

                s = 100 * np.sqrt(counts['raw_count']), marker = 'o',

                alpha = 0.6, linewidth = 1.5)

    

    if annotate:

        # Annotate the plot with text

        for i, row in counts.iterrows():

            # Put text with appropriate offsets

            plt.annotate(xy = (row[x] - (1 / counts[x].nunique()), 

                               row[y] - (0.15 / counts[y].nunique())),

                         color = 'navy',

                         s = f"{round(row['percent'], 1)}%")

        

    # Set tick marks

    plt.yticks(counts[y].unique())

    plt.xticks(counts[x].unique())

    

    # Transform min and max to evenly space in square root domain

    sqr_min = int(np.sqrt(raw_counts['raw_count'].min()))

    sqr_max = int(np.sqrt(raw_counts['raw_count'].max()))

    

    # 5 sizes for legend

    msizes = list(range(sqr_min, sqr_max,

                        int(( sqr_max - sqr_min) / 5)))

    markers = []

    

    # Markers for legend

    for size in msizes:

        markers.append(plt.scatter([], [], s = 100 * size, 

                                   label = f'{int(round(np.square(size) / 100) * 100)}', 

                                   color = 'lightgreen',

                                   alpha = 0.6, edgecolor = 'k', linewidth = 1.5))

        

    # Legend and formatting

    plt.legend(handles = markers, title = 'Counts',

               labelspacing = 3, handletextpad = 2,

               fontsize = 16,

               loc = (1.10, 0.19))

    

    plt.annotate(f'* Size represents raw count while % is for a given y value.',

                 xy = (0, 1), xycoords = 'figure points', size = 10)

    

    # Adjust axes limits

    plt.xlim((counts[x].min() - (6 / counts[x].nunique()), 

              counts[x].max() + (6 / counts[x].nunique())))

    plt.ylim((counts[y].min() - (4 / counts[y].nunique()), 

              counts[y].max() + (4 / counts[y].nunique())))

    plt.grid(None)

    plt.xlabel(f"{x}"); plt.ylabel(f"{y}"); plt.title(f"{y} vs {x}");
base.columns
plot_categoricals('hogar_nin', 'Target', base)
plot_categoricals('area1', 'Target', base)
plot_categoricals('hacdor', 'Target', base)
plot_categoricals('tipovivi1', 'Target', base)
target = base['Target']

aux1 = pd.DataFrame({'Target | %': target.value_counts(normalize=True)})

aux1
base.shape
sem_chao = base[base['pisonotiene'] == 1.0]

sem_chao.shape
batch = 32

n_batchs = int(500/batch)

base_aux = sem_chao.sample(batch)

for i in range(n_batchs):

    base_aux = pd.concat([base_aux, sem_chao.sample(batch)], axis = 0)
base = pd.concat([base, base_aux], axis = 0)
base.hist(column='pisonotiene', by ='Target', figsize=(10,10), color = 'coral')
corrmat = base.corr()

sns.set()

plt.figure(figsize=(13,10))

sns.heatmap(corrmat)
base = base.drop(['cielorazo', 'v14a', 'abastaguano', 'tipovivi1', 'hacdor'], axis = 1)
base.head()
from sklearn.preprocessing import StandardScaler
X = base.drop('Target', axis = 1)

y = base['Target']
scaler_x = StandardScaler()



X = scaler_x.fit_transform(X)
from sklearn.model_selection import cross_val_score

from sklearn.neighbors import KNeighborsClassifier
scores_mean = []

scores_std = []



k_lim_inf = 1

k_lim_sup = 30



folds = 5



k_max = None

max_acc = 0



i = 0

print('Finding best k...')

for k in range(k_lim_inf, k_lim_sup):

    

    KNNclf = KNeighborsClassifier(n_neighbors=k)

    

    score = cross_val_score(KNNclf, X, y, cv = folds)

    

    scores_mean.append(score.mean())

    scores_std.append(score.std())

    

    if scores_mean[i] > max_acc:

        k_max = k

        max_acc = scores_mean[i]

    i += 1

    if not (k%3):

        print('   K = {0} | Best CV acc = {1:2.2f}% (best k = {2})'.format(k, max_acc*100, k_max))

print('\nBest k: {}'.format(k_max))
plt.figure(figsize=(15, 7))

plt.errorbar(np.arange(k_lim_inf, k_lim_sup), scores_mean, scores_std,

             marker = 'o', markerfacecolor = 'purple' , linewidth = 3,

             markersize = 10, color = 'coral', ecolor = 'purple', elinewidth = 1.5)





yg = []

x = np.arange(0, k_lim_sup+1)

for i in range(len(x)):

    yg.append(max_acc)

plt.plot(x, yg, '--', color = 'purple', linewidth = 1)

plt.xlabel('k')

plt.ylabel('accuracy')

plt.title('KNN performed on several values of k')

plt.axis([0, k_lim_sup, min(scores_mean) - max(scores_std), max(scores_mean) + 1.5*max(scores_std)])
k = 16

KNNclf = KNeighborsClassifier(n_neighbors=k)

KNNclf.fit(X, y)
df_test = pd.read_csv("../input/costa-rican-household-poverty-prediction/test.csv")
features = ['escolari', 'SQBmeaned', 'hogar_nin', 'hogar_total', 'area1',

            'lugar1', 'pisonotiene', 'meaneduc', 'SQBovercrowding', 'abastaguadentro']
base_test = df_test[features]

base_test.head()
total = base_test.isnull().sum().sort_values(ascending = False)

percent = ((base_test.isnull().sum()/base_test.isnull().count())*100).sort_values(ascending = False)

missing_data = pd.concat([total, percent], axis = 1, keys = ['Total', '%'])

missing_data.head()
col = 'SQBmeaned'

base_test = base_test.astype(np.float)

base_test[col] = base_test[col].fillna(base_test[col].describe().mean())



col = 'meaneduc'

base_test = base_test.astype(np.float)

base_test[col] = base_test[col].fillna(base_test[col].describe().mean())
total = base_test.isnull().sum().sort_values(ascending = False)

percent = ((base_test.isnull().sum()/base_test.isnull().count())*100).sort_values(ascending = False)

missing_data = pd.concat([total, percent], axis = 1, keys = ['Total', '%'])

missing_data.head()
X_test = scaler_x.transform(base_test)
previews = KNNclf.predict(X_test)

previews = previews.astype(np.int)

previews = pd.DataFrame({'Target': previews})

previews = pd.concat([df_test['Id'], previews], axis = 1)
previews.head()
previews.to_csv('submission.csv', index = False)