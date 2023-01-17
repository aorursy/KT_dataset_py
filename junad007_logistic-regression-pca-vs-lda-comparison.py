# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import matplotlib.ticker as ticker



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/HR_comma_sep.csv')
df.head()
df.rename(columns = {'sales':'position'}, inplace = True)
df.describe()
sns.heatmap(data=df.isnull(),yticklabels=False)
fig, axs = plt.subplots(4,2,figsize=(16, 16))

axs[0,0].set_title('Left on Basis of Work Accident')

axs[0,1].set_title('Left on Basis of Promotion in last 5 years')

axs[1,0].set_title('Left on Basis of Salary')

axs[1,1].set_title('Satisfaction Level ratios')

axs[2,0].set_title('Last evaluation ratios')

axs[2,1].set_title('Time spend in company ratios')

axs[3,0].set_title('Number of projects ratios')

axs[3,1].set_title('Average monthly time spend in company ratios')

sns.countplot(x="left", data=df, hue='Work_accident',ax=axs[0,0])

sns.countplot(x ='left', data=df, hue= 'promotion_last_5years',ax=axs[0,1])

sns.countplot(x ='salary', data=df, hue= 'left' ,ax=axs[1,0])

sns.kdeplot(df[df['left']==0]['satisfaction_level'], ax=axs[1,1],label='left = 0')

sns.kdeplot(df[df['left']==1]['satisfaction_level'], ax=axs[1,1],label='left = 1')

sns.kdeplot(df[df['left']==0]['last_evaluation'], ax=axs[2,0],label='left = 0')

sns.kdeplot(df[df['left']==1]['last_evaluation'], ax=axs[2,0],label='left = 1')

sns.kdeplot(df[df['left']==0]['time_spend_company'], ax=axs[2,1],label='left = 0')

sns.kdeplot(df[df['left']==1]['time_spend_company'], ax=axs[2,1],label='left = 1')

sns.kdeplot(df[df['left']==0]['number_project'], ax=axs[3,0],label='left = 0')

sns.kdeplot(df[df['left']==1]['number_project'], ax=axs[3,0],label='left = 1')

sns.kdeplot(df[df['left']==0]['average_montly_hours'], ax=axs[3,1],label='left = 0')

sns.kdeplot(df[df['left']==1]['average_montly_hours'], ax=axs[3,1],label='left = 1')

fig.tight_layout()
df.corr()
# 'satisfaction_level' is the highest correlated value to 'left'

plt.figure(figsize=(8, 8))

sns.heatmap(df.corr(), cmap="BrBG")

plt.title('Correlation Heatmap')
# Further exploratory analysis with data gave a cluster formation

swarmPlot = sns.swarmplot(x = 'satisfaction_level', y = 'last_evaluation', data=df, hue='left', palette='Set2')

swarmPlot.set(xticklabels=[])

swarmPlot.legend(loc='upper right')

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3)
cluster_data = df[df['left']==1][['satisfaction_level','last_evaluation']]
# Fit data with kmeans and add labels to dataframe

kmeans.fit(cluster_data)

cluster_data['labels'] = kmeans.labels_

cluster_data.head()
swarmPlot =  sns.swarmplot(x="satisfaction_level", y="last_evaluation", data=cluster_data, hue='labels')

swarmPlot.set(xticklabels=[])
# Getting features and labes from data frame. Encoding String features.

X = df.drop(['left'], axis=1)

y = df['left']

X = X.join(pd.get_dummies(X['position']))

X = X.join(pd.get_dummies(X['salary']))

X = X.drop(['position','salary'], axis=1)

X.head()
#removing 1 column each of dummy data to avoid dummy trap

X = X.iloc[:,:-1].drop(X.columns[-4], axis=1)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 55)
# Fitting and Predicting

from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression(class_weight = "balanced")

logmodel.fit(X_train,y_train)

predictions = logmodel.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

print(confusion_matrix(y_test,predictions))

print('***************************************************')

print(classification_report(y_test,predictions))
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 55)
from sklearn.decomposition import PCA

pca_test = PCA()

X_pca_train = pca_test.fit_transform(X_train)

X_pca_test = pca_test.transform(X_test)

variance_component_ratios = pca_test.explained_variance_ratio_

np.set_printoptions(precision=4, suppress=True)

variance_component_ratios
predictions_pca = []

for i in range(1,7):

    pca = PCA(n_components=i)

    X_pca_train = pca.fit_transform(X_train)

    X_pca_test = pca.transform(X_test)

    log_pca_model = LogisticRegression(class_weight = "balanced")

    log_pca_model.fit(X_pca_train,y_train)

    predictions_pca.append(log_pca_model.predict(X_pca_test))
for i in range(1,7):

    print('PCA with Components =', i)

    print(classification_report(y_test,predictions_pca[i-1]))

    print('***************************************************')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 55)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda_test = LinearDiscriminantAnalysis(n_components=4)

X_lda_train = lda_test.fit_transform(X_train, y_train)

X_lda_test = lda_test.transform(X_test)

log_lda_model = LogisticRegression(class_weight = "balanced")

log_lda_model.fit(X_lda_train,y_train)

predictions_lda = log_lda_model.predict(X_lda_test)

print('LDA with Components =', 2)

print(classification_report(y_test,predictions_lda))

print('***************************************************')
from matplotlib import colors

data = np.random.rand(10, 10) * 20



# # create discrete colormap

# cmap = colors.ListedColormap(['red', 'blue'])

# bounds = [0,10,20]

# norm = colors.BoundaryNorm(bounds, cmap.N)



# fig, ax = plt.subplots()

# ax.imshow(data, cmap=cmap, norm=norm)



# # draw gridlines

# ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)

# ax.set_xticks(np.arange(-.5, 10, 1));

# ax.set_yticks(np.arange(-.5, 10, 1));



# plt.show()



swarmPlot =  sns.swarmplot(x="satisfaction_level", y="last_evaluation", data=cluster_data, hue='labels')

swarmPlot.set(xticklabels=[])

swarmPlot.grid(which='major', axis='both', color='k')
X_lda_test