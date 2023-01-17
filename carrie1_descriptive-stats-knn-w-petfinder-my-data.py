import pandas as pd 
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("../input/train/train.csv")
df.head(5)
df.info()
import seaborn as sns
ax = sns.countplot(x="Quantity", data=df)
#focus on single-pet adoptions only
df_singles = df.loc[df['Quantity'] == 1]
ax = sns.countplot(x="Quantity", data=df_singles)
df_singles['AdoptionSpeed'].describe()
ax = sns.countplot(x="AdoptionSpeed", data=df_singles)
df_singles['Age'].describe()
df_singles['age_quartile'] = pd.qcut(df_singles['Age'], 4)
ax = sns.boxplot(x="age_quartile", y="AdoptionSpeed", data=df_singles)
ax = sns.boxplot(x="Gender", y="AdoptionSpeed", data=df_singles)
ax = sns.boxplot(x="MaturitySize", y="AdoptionSpeed", data=df_singles)
import numpy as np
df_singles['Medium'] = np.where(df_singles['MaturitySize']==2, 'Yes', 'No')
ax = sns.boxplot(x="Medium", y="AdoptionSpeed", data=df_singles)
ax = sns.boxplot(x="FurLength", y="AdoptionSpeed", data=df_singles)
df_singles['ShortHair'] = np.where(df_singles['FurLength']==1, 'Yes', 'No')
ax = sns.boxplot(x="ShortHair", y="AdoptionSpeed", data=df_singles)
ax = sns.boxplot(x="Health", y="AdoptionSpeed", data=df_singles)
ax = sns.boxplot(x="Sterilized", y="AdoptionSpeed", data=df_singles)
ax = sns.countplot(x="Sterilized", hue="age_quartile", data=df_singles)
ax = sns.boxplot(x="Dewormed", y="AdoptionSpeed", data=df_singles)
ax = sns.countplot(x="Dewormed", hue="age_quartile", data=df_singles)
ax = sns.boxplot(x="Vaccinated", y="AdoptionSpeed", data=df_singles)
ax = sns.countplot(x="Vaccinated", hue="age_quartile", data=df_singles)
ax = sns.boxplot(x="Type", y="AdoptionSpeed", data=df_singles)
import numpy as np
df_singles['Free'] = np.where(df_singles['Fee']==50, 'Yes', 'No')
ax = sns.boxplot(x="Free", y="AdoptionSpeed", data=df_singles)
df_singles['Photos'] = np.where(df_singles['PhotoAmt']==0, 'No', 'Yes')
ax = sns.boxplot(x="Photos", y="AdoptionSpeed", data=df_singles)
df_singles['Videos'] = np.where(df_singles['VideoAmt']==0, 'No', 'Yes')
ax = sns.boxplot(x="Videos", y="AdoptionSpeed", data=df_singles)
temp = df_singles.groupby('Color1')['AdoptionSpeed'].mean()
temp.sort_values()
#in what states are animals adopted the fastest on average?
temp = pd.merge(df_singles, pd.read_csv("../input/state_labels.csv"), left_on='State',right_on='StateID')
temp = temp.groupby('StateName')['AdoptionSpeed'].mean()
temp.sort_values()
#which breed on average is adopted the fastest? 
temp = pd.merge(df_singles, pd.read_csv("../input/breed_labels.csv"), left_on='Breed1',right_on='BreedID')
temp = temp.groupby('BreedName')['AdoptionSpeed'].mean()
temp.sort_values()
df_singles['Description']
df_singles['Guard'] = np.where(df_singles['Description'].str.contains("guard", case=False), '1', '0')
ax = sns.violinplot(x="Guard", y="AdoptionSpeed", data=df_singles)
df_singles['Friend'] = np.where(df_singles['Description'].str.contains("friend", case=False), '1', '0')
ax = sns.violinplot(x="Friend", y="AdoptionSpeed", data=df_singles)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
X = df_singles[['Age','Gender','Health','FurLength','MaturitySize','Type','Fee',
               'Guard']] #see how these predictors perform
y = df_singles['AdoptionSpeed']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

k_range = range(1, 51)
scores = []
features = []

for k in k_range: #trying to find optimal k
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    scores.append(metrics.accuracy_score(y_test, y_pred))

import matplotlib.pyplot as plt
plt.plot(k_range, scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Testing Accuracy')