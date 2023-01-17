import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.svm import LinearSVC





import os

print(os.listdir("../input"))
df=pd.read_csv('../input/train.csv')
colors = ["#0101DF", "#DF0101"]



sns.countplot('target', data=df, palette=colors)

plt.title('Class Distributions \n (0: No Transaction || 1: Transaction)', fontsize=14)
print('No Transaction', round(df['target'].value_counts()[0]/len(df) * 100,2), '% of the dataset')

print('Transaction', round(df['target'].value_counts()[1]/len(df) * 100,2), '% of the dataset')
from sklearn.preprocessing import StandardScaler



X = df.drop('target', axis=1)

X=df.drop('ID_code', axis=1)

y = df['target']

X = StandardScaler().fit_transform(X)



from sklearn.decomposition import PCA

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X)

X_pca_df = pd.DataFrame(data = X_pca)
from sklearn.model_selection import train_test_split

from sklearn.model_selection import StratifiedShuffleSplit



print('No Transaction', round(df['target'].value_counts()[0]/len(df) * 100,2), '% of the dataset')

print('Transaction', round(df['target'].value_counts()[1]/len(df) * 100,2), '% of the dataset')







sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)



for train_index, test_index in sss.split(X, y):

    print("Train:", train_index, "Test:", test_index)

    Xtrain_pca, Xtest_pca = X_pca_df.iloc[train_index], X_pca_df.iloc[test_index]

    ytrain, ytest = y.iloc[train_index], y.iloc[test_index]





# Turn into an array

Xtrain_pca = Xtrain_pca.values

Xtest_pca = Xtest_pca.values

ytrain = ytrain.values

ytest = ytest.values



# See if both the train and test label distribution are similarly distributed

train_unique_label, train_counts_label = np.unique(ytrain, return_counts=True)

test_unique_label, test_counts_label = np.unique(ytest, return_counts=True)

print('-' * 100)



print('Label Distributions: \n')

print(train_counts_label/ len(ytrain))

print(test_counts_label/ len(ytest))
def samplecompare (Xtrain,Xtest,ytrain,ytest):

    clf = LinearSVC( random_state=123)

    clf.fit(Xtrain, ytrain)

    score=clf.score(Xtest,ytest)

    

    w = clf.coef_[0]



    a = -w[0] / w[1]



    x_min=Xtrain[:,0].min()

    x_max=Xtrain[:,0].max()



    y_min=Xtrain[:,1].min()

    y_max=Xtrain[:,1].max()





    xx = np.linspace(x_min,x_max,10)

    yy = a * xx - clf.intercept_[0] / w[1]

    h0 = plt.plot(xx, yy, 'k-', label="non weighted div")



    plt.scatter(Xtrain[:, 0], Xtrain[:, 1], c = ytrain)

    plt.ylim([y_min,y_max])

    plt.xlim([x_min,x_max])

    plt.show()

    

    print('Accuaracy score of this sampling method ' +str(score))

    return
from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler(random_state=42)

Xtrain_under, ytrain_under = rus.fit_resample(Xtrain_pca, ytrain)



Xtest_under, ytest_under = rus.fit_resample(Xtest_pca, ytest)
sns.countplot(ytrain_under, palette=colors)

plt.title('Class Distributions \n (0: No Transaction || 1: Transaction)', fontsize=14)
samplecompare(Xtrain_under,Xtest_under,ytrain_under,ytest_under)
from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler(random_state=0)



Xtrain_over, ytrain_over = ros.fit_resample(Xtrain_pca, ytrain)



Xtest_over, ytest_over = ros.fit_resample(Xtest_pca, ytest)
sns.countplot(ytrain_over, palette=colors)

plt.title('Class Distributions \n (0: No Transaction || 1: Transaction)', fontsize=14)
samplecompare(Xtrain_over, Xtest_over,ytrain_over,ytest_over)
from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=42)

Xtrain_sm, ytrain_sm = sm.fit_resample(Xtrain_pca, ytrain)

Xtest_sm, ytest_sm = sm.fit_resample(Xtest_pca, ytest)
sns.countplot(ytrain_sm, palette=colors)

plt.title('Class Distributions \n (0: No Transaction || 1: Transaction)', fontsize=14)
samplecompare(Xtrain_sm,Xtest_sm,ytrain_sm,ytest_sm)
from imblearn.over_sampling import ADASYN

adn=ADASYN(random_state=1)

Xtrain_ad, ytrain_ad = adn.fit_resample(Xtrain_pca, ytrain)

Xtest_ad, ytest_ad = adn.fit_resample(Xtest_pca, ytest)
sns.countplot(ytrain_ad, palette=colors)

plt.title('Class Distributions \n (0: No Transaction || 1: Transaction)', fontsize=14)
samplecompare(Xtrain_ad,Xtest_ad,ytrain_ad,ytest_ad)