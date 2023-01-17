import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df = pd.read_csv("../input/travel-insurance/travel insurance.csv")
df.head()
df.info()
def cleanYesNo(s):

    if s == "Yes":

        return 1

    elif s == "No":

        return 0



df["Claim0"] = df.loc[:,'Claim'].apply(cleanYesNo)
df.drop(["Claim", "Gender"], axis = 1, inplace = True)
df.groupby(["Agency"]).mean()
df.groupby(['Agency Type']).mean()
df.groupby(['Distribution Channel']).mean()
df.describe()
df[df["Duration"] <0]
df[df["Age"] > 100]
df.loc[df['Duration'] < 0, 'Duration'] = 49.317

df.loc[df['Age'] > 100, 'Age'] = 39.969981
df.describe()
print("Claimed")

print(df[df["Claim0"] == 1]["Claim0"].count())

print("Not Claimed")

print(df[df["Claim0"] == 0]["Claim0"].count())
g = sns.catplot(x="Agency",y = "Claim0", data=df)

g.fig.set_size_inches(10,5)
claimeddata = df[df["Claim0"]==1]
claimeddata['Destination'].value_counts().head(10).plot(kind='barh', figsize=(5,5))
f, axes = plt.subplots(1, 2)

f.set_size_inches(15,5)

axes[0].set_title('General')

axes[1].set_title('Agency (Claimed Policies)')

sns.countplot(y="Agency", data=df, ax = axes[0])

sns.countplot(y="Agency", data=claimeddata, ax = axes[1])
a = pd.DataFrame(df.loc[:, "Product Name"].value_counts())

b = pd.DataFrame(claimeddata.loc[:, "Product Name"].value_counts())

combined = a.join(b, lsuffix = "_general", rsuffix = "_claimed")

combined.fillna(0, inplace = True)

combined
ratio_list = []

for i in range(len(combined)):

    ratio_list.append(combined.iloc[i][1] / combined.iloc[i][0])

ratio = pd.DataFrame(ratio_list, index = np.array(combined.index))

ratio = ratio.rename(columns = {0:"Ratio"})



plt.figure(figsize=(7,7))

sns.barplot(data = ratio, y = ratio.index, x = "Ratio")
sns.heatmap(df.corr(), square=True)
X = df.drop(columns=['Claim0'])

y = df['Claim0']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
updated_X = X_train.drop(columns = ["Agency", "Agency Type", "Distribution Channel", "Product Name", "Destination"])
from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler(random_state=0)

X_resampled, y_resampled = ros.fit_resample(updated_X, y_train)
(unique, counts) = np.unique(y_resampled, return_counts=True)

(unique, counts)
X_test_updated = X_test.drop(columns = ["Agency", "Agency Type", "Distribution Channel", "Product Name", "Destination"])
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(random_state=0).fit(X_resampled, y_resampled)

y_pred = clf.predict(X_test_updated)
clf.score(X_resampled, y_resampled)
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, y_pred)
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred, target_names = ["Claimed", "Non-claimed"]))
from sklearn.model_selection import cross_validate

cv_results = cross_validate(clf, X_resampled, y_resampled, cv=10)
cv_results['test_score']