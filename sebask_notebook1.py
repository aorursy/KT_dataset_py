import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from subprocess import check_output



# Load and describe data

# print(check_output(["ls", "../input"]).decode("utf8"))

df = pd.read_json(open("../input/train.json", "r"))

df['n_photos'] = df['photos'].apply(len)

df['n_features'] = df['features'].apply(len)

df['ilevel_categ'] = df['interest_level'].map({'low': 1, 'medium': 2, 'high': 3})

df["n_description_words"] = df["description"].apply(lambda x: len(x.split(" ")))

df.info()
from sklearn.ensemble import RandomForestClassifier

from sklearn import metrics



# extract target variable (interest categories)

dfY = df['ilevel_categ'].copy()

del df['ilevel_categ']

# extract numerical features

numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

dfX = df.select_dtypes(include=numerics)

# delete listing id because we dont know what it means

del dfX['listing_id']



# train random forest model 

clf = RandomForestClassifier(n_estimators = 1000)

clf.fit(dfX, dfY)



# Let the model tell us the important features and plot it

importances = clf.feature_importances_

display = pd.DataFrame(dfX.columns,columns=['Feature'])

display['Importance'] = importances

print(display)

indices = np.argsort(importances)[::-1]

std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)

plt.figure()

plt.title("Feature importances")

plt.bar(range(dfX.shape[1]), importances[indices],

       color="royalblue", yerr=std[indices], align="center")

plt.xticks(range(dfX.shape[1]), indices)

plt.xlim([-1, dfX.shape[1]])

plt.show()
# View correlation matrix

corr = dfX.corr()



# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(10, 220, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3,

            square=True, xticklabels=True, yticklabels=True,

            linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)

plt.title('Correlation Matrix for Appointment Data')
# Load and wrangle test data

test = pd.read_json(open("../input/test.json", "r"))

test['n_photos'] = test['photos'].apply(len)

test['n_features'] = test['features'].apply(len)

test["n_description_words"] = test["description"].apply(lambda x: len(x.split(" ")))

# numerics only

numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

testX = test.select_dtypes(include=numerics)

listing_id = testX['listing_id'].copy()

del testX['listing_id']

predictions = clf.predict_proba(testX)

submission = pd.DataFrame(index=listing_id)

submission['high'] = predictions[:,2]

submission['medium'] = predictions[:,1]

submission['low'] = predictions[:,0]

submission.to_csv('submission1.csv')