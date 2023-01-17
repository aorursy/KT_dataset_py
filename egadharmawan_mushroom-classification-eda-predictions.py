import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv('../input/mushroom-classification/mushrooms.csv')
df.head()
df.info()
df.columns
df['class'].replace(to_replace=['e','p'], value=['edible','poisonous'],inplace=True)
df['cap-shape'].replace(to_replace=['b', 'c','f','x','k','s'], value=['bell','conical','convex','flat','knobbed','sunken'],inplace=True)
df['cap-surface'].replace(to_replace=['f','g','y','s'], value=['fibrous','grooves','scaly','smooth'],inplace=True)
df['cap-color'].replace(to_replace=['n','b','c','g','r','p','u','e','w','y'], value=['brown','buff','cinnamon','gray','green','pink','purple','red','white','yellow'],inplace=True)
df['bruises'].replace(to_replace=['t', 'f'], value=['bruises','no'],inplace=True)
df['odor'].replace(to_replace=['a','l','c','y','f','m','n','p','s'], value=['almond','anise','creosote','fishy','foul','musty','none','pungent','spicy'],inplace=True)
df['gill-attachment'].replace(to_replace=['a','d','f','n'], value=['attached','descending','free','notched'],inplace=True)
df['gill-spacing'].replace(to_replace=['c','w','d'], value=['close','crowded','distant'],inplace=True)
df['gill-size'].replace(to_replace=['b', 'n'], value=['broad','narrow'],inplace=True)
df['gill-color'].replace(to_replace=['k','n','b','h','g','r','o','p','u','e','w','y'], value=['black','brown','buff','chocolate','gray','green','orange','pink','purple','red','white','yellow'],inplace=True)
df['stalk-shape'].replace(to_replace=['e', 't'], value=['enlarging','tapering'],inplace=True)
df['stalk-root'].replace(to_replace=['b','c','u','e','z','r','?'], value=['bulbous','club','cup','equal','rhizomorphs','rooted','missing'],inplace=True)
df['stalk-surface-above-ring'].replace(to_replace=['f','y','k','s'], value=['fibrous','scaly','silky','smooth'],inplace=True)
df['stalk-surface-below-ring'].replace(to_replace=['f','y','k','s'], value=['fibrous','scaly','silky','smooth'],inplace=True)
df['stalk-color-above-ring'].replace(to_replace=['n','b','c','g','o','p','e','w','y'], value=['brown','buff','cinnamon','gray','orange','pink','red','white','yellow'],inplace=True)
df['stalk-color-below-ring'].replace(to_replace=['n','b','c','g','o','p','e','w','y'], value=['brown','buff','cinnamon','gray','orange','pink','red','white','yellow'],inplace=True)
df['veil-type'].replace(to_replace=['p', 'u'], value=['partial','universal'],inplace=True)
df['veil-color'].replace(to_replace=['n','o','w','y'], value=['brown','orange','white','yellow'],inplace=True)
df['ring-number'].replace(to_replace=['n','o','t'], value=['none','one','two'],inplace=True)
df['ring-type'].replace(to_replace=['c','e','f','l','n','p','s','z'], value=['cobwebby','evanescent','flaring','large','none','pendant','sheathing','zone'],inplace=True)
df['spore-print-color'].replace(to_replace=['k','n','b','h','r','o','u','w','y'], value=['black','brown','buff','chocolate','green','orange','purple','white','yellow'],inplace=True)
df['population'].replace(to_replace=['a','c','n','s','v','y'], value=['abundant','clustered','numerous','scattered','several','solitary'],inplace=True)
df['habitat'].replace(to_replace=['g','l','m','p','u','w','d'], value=['grasses','leaves','meadows','paths','urban','waste','woods'],inplace=True)
df.head()
corr = df.apply(lambda x : pd.factorize(x)[0]).corr(method='pearson', min_periods=1)
plt.figure(figsize=(16,16))
sns.heatmap(corr, cmap = "RdBu_r", vmax=0.9, square=True)
df['veil-type'].value_counts()
IF = corr['class'].sort_values(ascending=False).head(10).to_frame()
IF.head(8)
print(df.groupby('gill-size')['class'].value_counts())
df.groupby('gill-size')['class'].value_counts().unstack().plot.barh()
print(df.groupby('gill-spacing')['class'].value_counts())
df.groupby('gill-spacing')['class'].value_counts().unstack().plot.barh()
print(df.groupby('cap-surface')['class'].value_counts())
df.groupby('cap-surface')['class'].value_counts().unstack().plot.barh()
print(df.groupby('ring-number')['class'].value_counts())
df.groupby('ring-number')['class'].value_counts().unstack().plot.barh()
print(df.groupby('gill-attachment')['class'].value_counts())
df.groupby('gill-attachment')['class'].value_counts().unstack().plot.barh()
print(df.groupby('veil-color')['class'].value_counts())
df.groupby('veil-color')['class'].value_counts().unstack().plot.barh()
print(df.groupby('stalk-shape')['class'].value_counts())
df.groupby('stalk-shape')['class'].value_counts().unstack().plot.barh()
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
## Feature engineering

df['class'].replace(to_replace=['edible','poisonous'], value=['0','1'],inplace=True)
df.head()
# Split the data

X = df.drop('class', axis=1)
y = df['class']
# handling categorical features

X = pd.get_dummies(X)
X.head()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=44)
from sklearn.ensemble import RandomForestClassifier

# Fitting Random Forest Classification
classifier = RandomForestClassifier(n_estimators = 200)
classifier.fit(X_train, y_train)

# predict
RF_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, RF_pred)
accuracy
from sklearn.linear_model import SGDClassifier

# fit to SGDClassifier
sgd= SGDClassifier()
sgd.fit(X_train, y_train)

# predict
SGD_pred = sgd.predict(X_test)
acc = accuracy_score(y_test, SGD_pred)
print(acc)
from sklearn.linear_model import LogisticRegression

#fit to LogReg
lr = LogisticRegression()
lr.fit(X_train, y_train)

# Predict
LR_pred = lr.predict(X_test)
acc = accuracy_score(y_test, LR_pred)
print(acc)
## Distribution Comparison

f = plt.figure(figsize=(15,12))

# Basic Distribution
ax = f.add_subplot(221)
ax = sns.distplot(y_test)
ax.set_title('Basic Distribution')

# Random Forest Predicted result
ax = f.add_subplot(222)
xx = pd.DataFrame(RF_pred)
ax = sns.distplot(RF_pred, label="Predicted Values")
ax.set_title('Random Forest Predicted result')

# SGDClassifier Predicted result
ax = f.add_subplot(223)
ax = sns.distplot(SGD_pred, label="Predicted Values")
ax.set_title('SGDClassifier Predicted result')

# Logistic Regression Predicted result
ax = f.add_subplot(224)
ax = sns.distplot(LR_pred, label="Predicted Values")
ax.set_title('Logistic Regression Predicted result')