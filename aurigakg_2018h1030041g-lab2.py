# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
dataset= pd.read_csv("/kaggle/input/eval-lab-2-f464/train.csv")
dataset.head()
dataset.describe()
test_data= pd.read_csv('/kaggle/input/eval-lab-2-f464/test.csv')

test_data.info()
dataset['class'].unique()
dataset['class'].value_counts()
X = dataset[dataset.columns.drop(['id','class'])]

Y = dataset['class']



test= test_data[test_data.columns.drop(['id'])]

from sklearn.ensemble import IsolationForest

isf = IsolationForest(n_jobs=-1, random_state=1)

isf.fit(X, Y)

outlier=isf.predict(X)

X_= X[outlier==1]

Y_= Y[outlier==1]
# from sklearn.decomposition import PCA

# pca = PCA(n_components=2)

# from sklearn.preprocessing import StandardScaler

# X_new = pd.DataFrame(StandardScaler().fit_transform(X_),columns=X_.columns)

# principalComponents = pca.fit_transform(X_new)

# principalDf = pd.DataFrame(data = principalComponents

#              , columns = ['PC1', 'PC2'])

# finalDf = pd.concat([principalDf, Y], axis = 1)

# finalDf.head()
# import seaborn as sns

# import matplotlib.pyplot as plt

# fig = plt.figure(figsize = (8,8))

# ax = fig.add_subplot(1,1,1) 

# ax.set_xlabel('PC1', fontsize = 15)

# ax.set_ylabel('PC2', fontsize = 15)

# ax.set_title('2 component PCA', fontsize = 20)

# targets = [1,2,3,5,6,7]

# colors = ['r', 'g', 'b','c','m','y']

# for target, color in zip(targets,colors):

#     indicesToKeep = finalDf['class'] == target

#     ax.scatter(finalDf.loc[indicesToKeep, 'PC1']

#                , finalDf.loc[indicesToKeep, 'PC2']

#                , c = color

#                , s = 50)

# ax.legend(targets)

# ax.grid()
from imblearn.over_sampling import ADASYN

adasyn= ADASYN(n_neighbors=3, sampling_strategy='minority',random_state=42)

X1, y1 = adasyn.fit_resample(X_, Y_)
dataset['class'].value_counts().plot(kind='bar', title='Count (target)');
df = pd.DataFrame(X1,columns=X_.columns)

df['class'] = y1



df['class'].value_counts().plot(kind='bar', title='Count (target)');

X_1= df[df.columns.drop(['class'])]

Y_1= df['class']
from mlxtend.classifier import StackingClassifier

from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier

from sklearn.naive_bayes import GaussianNB





# BaggingClassifier()



m = StackingClassifier(

    classifiers=[

        XGBClassifier(max_depth=5,gamma=0.5,eta=0.8,reg_alpha=0.5,reg_lambda=0.5),

        LogisticRegression()

               ],

    use_probas=True,

    meta_classifier=  LogisticRegression()

)

m.fit(X_1, Y_1)
np.unique(m.predict(test))
id1=test_data['id']

predict_y= m.predict(test)

sol_xgb= pd.DataFrame({'id':id1,'class':predict_y})

sol_xgb.to_csv('final.csv',index=False)