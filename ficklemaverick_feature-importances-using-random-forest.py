# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn.ensemble import RandomForestClassifier

import seaborn as sns

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
trainData = pd.read_csv('../input/train.csv',index_col='ID_code')
total = trainData.isnull().sum().sort_values(ascending=False)

percent = (trainData.isnull().sum()/trainData.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head()
model = RandomForestClassifier(random_state=1, max_depth=10,n_estimators=20)
X = trainData.drop(['target'],axis=1)



y = trainData['target']



model.fit(X,y)







features = X.columns



importances = model.feature_importances_



indices = np.argsort(importances)[-20:]
plt.title('Feature Importances')

plt.barh(range(len(indices)), importances[indices], color='b', align='center')

plt.yticks(range(len(indices)), [features[i] for i in indices])

plt.xlabel('Relative Importance')

plt.show()
filteredFeatures = [features[i] for i in np.argsort(importances)[-100:]]
Xfinal = trainData.filter(filteredFeatures,axis=1)
y = trainData['target']
def plotCorrelation(df):

    corr = df.corr()

    # Generate a mask for the upper triangle

    mask = np.zeros_like(corr, dtype=np.bool)

    mask[np.triu_indices_from(mask)] = True

    

    # Set up the matplotlib figure

    f, ax = plt.subplots(figsize=(11, 9))



    # Generate a custom diverging colormap

    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    

    # Draw the heatmap with the mask and correct aspect ratio

    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plotCorrelation(Xfinal)
Xfinal.corr()
Xfinal['var81_139'] = Xfinal['var_81'] + Xfinal['var_139']
Xfinal = Xfinal.drop(['var_81','var_139'],axis=1)
plotCorrelation(Xfinal)