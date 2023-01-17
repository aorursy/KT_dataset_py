# Import libraries
import warnings; warnings.simplefilter('ignore')
import re
import pandas as pd
import numpy as np
import seaborn as sns
import scipy
import scipy.stats as stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.graphics.mosaicplot import mosaic
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.datasets import load_wine
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
%matplotlib inline
# courtesy Names Corpus Version 1.3 by Mark Kantrowitz (c) 1991 (http://www.cs.cmu.edu/afs/cs/project/ai-repository/ai/areas/nlp/corpora/names/)
names = pd.read_csv("../input/names/names.csv")
train = pd.read_csv("../input/titanic/train.csv")
test = pd.read_csv("../input/titanic/test.csv")
train.info()

"""
Globals
=======
"""
MEDIAN_AGE = train.Age.median()
MEDIAN_FARE = train.Fare.median()
SVM_MODEL = None
CATBOOST_MODEL = None

"""
Columns
=========
"""

# To avoid hardcoding column names:
class columns:
    pass

c = columns()
for col in train.columns.values:
    setattr(c, col, col)

setattr(c, "Female", "Female")
setattr(c, "Title", "Title")
setattr(c, "HasSib", "HasSib")
setattr(c, "HasParch", "HasParch")
setattr(c, "HasFamily", "HasFamily")
setattr(c, "FareCluster", "FareCluster")
setattr(c, "AgeCluster", "AgeCluster")
setattr(c, "CabinLevel", "CabinLevel")
setattr(c, "Sarch", "Sarch")

# Additional Constants
HAS_SIB_THRESHOLD = 0
HAS_PARCH_THRESHOLD = 0
NUM_FARE_CLUSTERS = 3
NUM_AGE_CLUSTERS = 6
MARRIED_TITLES = ["Mr", "Mrs"]
TRUE_LOVE_AGE_CUTOFF = 18
COMMON = "Common"
PRIVILAGED = "Privilaged"
MILITARY = "Military"
PUBLIC_SERVANT = "PublicServant"
# Pipeline manager
class Pipeline:
    _steps = {}
    
    def step(func):
        df = pd.DataFrame()
        df = func(df)
        try:
            df.info()
        except:
            raise "function does not return pandas.DataFrame"

        Pipeline._steps[func.__name__] = func
        return func


    def process(df: pd.DataFrame) -> pd.DataFrame:
        df_copy = df.copy()
        for f in Pipeline._steps.values():
            df_copy = f(df_copy)

        return df_copy
# Drafts a contingency table between column1 and column2 with pvalues.
# Assums column1 and column2 are categorical variables.
def contingency(df: pd.DataFrame, column1: str, column2: str) -> pd.DataFrame:
    row = sorted(pd.unique(df[column1]))
    col = sorted(pd.unique(df[column2]))
    d = {}
    columns = [(column2, k) for k in col]
    index = [(column1, m) for m in row]
    p = []
    total = []

    for k in col:
        if k not in d:
            d[k] = []
        for m in row:
            d[k].append(
                len(
                    df.query("{} == {} and {} == {}".format(column2, k, column1, m))
                )
            )

    for m in row:
        slice = df.query("{} == {}".format(column1, m))
        (_, pval) = stats.chisquare(slice[column2].value_counts())
        p.append(pval)

        total.append(len(slice))

    cs = pd.DataFrame(d)
    cs.columns = pd.MultiIndex.from_tuples(columns)
    cs.index = pd.MultiIndex.from_tuples(index)
    cs["Total"] = total
    cs["Pvalue"] = p
    
    return cs
# Plotting Helpers
def pretty_hist(df: pd.DataFrame, yaxis_title: str, column_name: str, title: str, labels: list):
    values = df.groupby([column_name])[column_name].count().values
    uniq_values = pd.unique(df[column_name].values)
    ticks = np.arange(np.min(uniq_values) - 0.5, np.max(uniq_values)+1, step=1)
    n = len(df)
    percentages = [int(k*100/n) for k in values]
    
    if len(percentages) != len(labels):
        print(len(percentages), len(labels))
    assert len(percentages) == len(labels)
        
    for i in range(len(labels)):
        labels[i] = labels[i] + " ({}%)".format(percentages[i])

    labels.insert(0, "")
    labels.append("")
    plt.hist(df[column_name], np.arange(len(uniq_values)*2), rwidth=0.3)
    plt.title(title)
    plt.xticks(ticks, labels)
    plt.ylabel(yaxis_title)
    
assert len(pd.unique(train.PassengerId.values)) == len(train)
pretty_hist(train, "Count", c.Survived, "Survived | Titanic train dataset", ["Not\nSurvived", "Survived"])
pretty_hist(train, "Count", c.Pclass, "Pclass | Titanic train dataset", ["1st", "2nd", "3rd"])
train[[c.Name]].head(10)
pretty_hist(train.assign(Sex = [1 if k == 'female' else 0 for k in train.Sex]), "Count", c.Sex, "Sex | Titanic train dataset.", ["Male", "Female"])
pretty_hist(train.assign(Missing = [1 if k else 0 for k in train.Age.isnull()]), "Count", "Missing", "Age | Titanic train dataset.", ["Not\nMissing Value", "Missing Value"])
ax = train.Age.plot.kde()
ax.set_title("Age | Titanic train dataset.")
ax.set_ylabel("Density")
ax.set_xbound(0)
_ = ax.set_xlabel("Age (year)")
p = plt.hist(train.SibSp)
plt.title("SibSb | Titanic train dataset.")
plt.ylabel("Count")
_ = plt.xlabel("Number of siblings on board.")
p = plt.hist(train.Parch)
plt.title("Parch | Titanic train dataset.")
plt.ylabel("Count")
_ = plt.xlabel("Number of parent/children accompanied on board.")
assert train.Fare.isnull().sum() == 0 # True
ax = train.Fare.plot.kde()
ax.set_title("Fare | Titanic train dataset.")
ax.set_ylabel("Density")
ax.set_xbound(0)
_ = ax.set_xlabel("Fare ($)")
train.assign(CabinLetter = train.Cabin.str.get(0)).CabinLetter.value_counts()
train.Embarked.value_counts()
train[train.Embarked.isnull()]
train['Embarked'].value_counts()
train['Embarked'].value_counts().plot.bar();

df_Survived = train[train['Survived']==1]
df_Not_Survived = train[train['Survived']==0]

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
df_Survived['Embarked'].value_counts().plot.box(ax=axes[0]); 
axes[0].set_title('Embarked Survived')
df_Not_Survived['Embarked'].value_counts().plot.box(ax=axes[1]); 
axes[1].set_title('Embarked not Survived')

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 5))
df_Survived['Embarked'].value_counts().plot.bar(ax=axes[0]); 
axes[0].set_title('the total count of Embarked for survived ')
df_Not_Survived['Embarked'].value_counts().plot.bar(ax=axes[1]); 
axes[1].set_title('the total count of Embarked for non-survived ')

train.groupby('Survived')['Embarked'].value_counts().unstack(level=1).plot.bar(stacked=True)

stats.ttest_ind(train[train.Survived == 0].Fare, train[train.Survived == 1].Fare)

fg = sns.FacetGrid(data=train, hue='Survived')
fg.map(plt.scatter, 'Age', 'Fare').add_legend()

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
df_Survived['Fare'].value_counts().plot.box(ax=axes[0], sharey=True); 
axes[0].set_title('Fare for df_Survived')
df_Not_Survived['Fare'].value_counts().plot.box(ax=axes[1]); 
axes[1].set_title('Fare for df_Not_Survived')
plt.show()

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 8), sharey=True)
df_Survived['Fare'].value_counts().plot.hist(ax=axes[0]); 
axes[0].set_title('Fare for df_Survived')
df_Not_Survived['Fare'].value_counts().plot.hist(ax=axes[1]); 
axes[1].set_title('Fare for df_Not_Survived')
plt.show()
print(train['Sex'].value_counts())
train['Sex'].value_counts().plot.bar()

train.groupby('Survived')['Sex'].value_counts().unstack(level=1).plot.bar(stacked=True)
contingency(train, c.Pclass, c.Survived)
fig, _ = mosaic(train, [c.Pclass, c.Survived], title="Pclass vs Survived | Titanic train dataset.", axes_label=True)
fig.axes[0].set_ylabel(c.Survived)
_ = fig.axes[0].set_xlabel(c.Pclass)

train.groupby('Survived')['Pclass'].value_counts().unstack(level=1).plot.bar(stacked=True)
train['Pclass'].value_counts()

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 4), sharey=True)
train[train['Pclass']==1]['Fare'].plot.box(ax=axes[0]);
axes[0].set_title('Class 1')

train[train['Pclass']==2]['Fare'].plot.box(ax=axes[1]); 
axes[1].set_title('Class 1')

train[train['Pclass']==3]['Fare'].plot.box(ax=axes[2]); 
axes[2].set_title('Class 3')
train.groupby('Pclass')['Embarked'].value_counts().unstack(level=1).plot.bar(stacked=True)
@Pipeline.step
def numerify_sex(df: pd.DataFrame) -> pd.DataFrame:
    if c.Sex not in df.columns.values:
        return df
    
    dfcopy = df.assign(Sex=df.Sex.str.title())
    return dfcopy.join(pd.get_dummies(dfcopy.Sex)).drop(["Male"], axis=1)
# Test numerify_sex
train_sex_numerified = numerify_sex(train)
assert train_sex_numerified.query("Sex == 'Male'").query("Female == 0").Female.count() == 577
assert train_sex_numerified.query("Sex == 'Female'").query("Female == 1").Female.count() == 314

contingency(train_sex_numerified, c.Female, c.Survived)
def extract_title(name: str) -> str:
    return re.search(r'(?<=\,\s)\w+', name).group(0)

def remap_title(name: str) -> str:
    result = name
    remap = {
        "Mr": COMMON,
        "Mrs": COMMON,
        "Mlle": PRIVILAGED,
        "Mme": PRIVILAGED,
        "Ms": PRIVILAGED,
        "the": PRIVILAGED,
        "Miss": PRIVILAGED,
        "Master": PRIVILAGED,
        "Rev": PUBLIC_SERVANT,
        "Dr": PUBLIC_SERVANT,
        "Don": COMMON,
        "Madame": COMMON,
        "Major": MILITARY,
        "Lady": PRIVILAGED,
        "Sir": PRIVILAGED,
        "Col": MILITARY,
        "Capt": MILITARY,
        "Royalty": PRIVILAGED,
        "Dona": PRIVILAGED,
        "Jonkheer": COMMON
    }
    if is_name(name):
        result = "NONE"

    if name in remap:
        result = remap[name]
    else:
        raise Exception("{} not assignable to title remapping.".format(name))

    return result

def is_name(name: str) -> bool:
    return names.Name.equals(name)

@Pipeline.step
def numerify_title(df: pd.DataFrame) -> pd.DataFrame:
    if c.Name not in df.columns.values:
        return df

    dfcopy = df.copy()
    if dfcopy.Name.isnull().sum() > 0:
        raise Exception("numerify_title is not implemented to handle missing data")

    dfcopy = dfcopy.assign(Title=[remap_title(extract_title(s)) for s in dfcopy.Name])

    dfcopy = dfcopy.join(pd.get_dummies(dfcopy.Title))

    return dfcopy
train_title = numerify_title(train)
f = train_title[train_title.Sex == 'female']
m = train_title[train_title.Sex == 'male']

titleS = pd.unique(train_title.Title.values)
ms = []
fs = []
tot = []

for k in titleS:
    ms.append(m[m.Title == k].Survived.sum())
    fs.append(f[f.Title == k].Survived.sum())
    tot.append(len(train_title[train_title.Title == k]))


df = pd.DataFrame()
df = df.assign(Title = titleS, MaleSurvived = ms, FemaleSurvived = fs, Total = tot)
df = df[["Title", "MaleSurvived", "FemaleSurvived", "Total"]]
df
@Pipeline.step
def recode_pclass(df: pd.DataFrame) -> pd.DataFrame:
    if c.Pclass not in df.columns.values:
        return df

    dfcopy = df.copy()
    assert dfcopy.Pclass.isnull().sum() == 0

    return dfcopy.join(pd.get_dummies(dfcopy.Pclass, prefix=c.Pclass))
# Test recode_pclass
train_pclass = recode_pclass(train)
assert train_pclass.Pclass_1.isnull().sum() == 0
assert train_pclass.Pclass_2.isnull().sum() == 0
assert train_pclass.Pclass_3.isnull().sum() == 0
assert len(train_pclass.query("Pclass == 1 and Pclass_1 == 1 and Pclass_2 == 0 and Pclass_3 == 0")) == len(train.query("Pclass == 1"))
assert len(train_pclass.query("Pclass == 2 and Pclass_1 == 0 and Pclass_2 == 1 and Pclass_3 == 0")) == len(train.query("Pclass == 2"))
assert len(train_pclass.query("Pclass == 3 and Pclass_1 == 0 and Pclass_2 == 0 and Pclass_3 == 1")) == len(train.query("Pclass == 3"))
train_pclass[[c.Pclass, "Pclass_1", "Pclass_2", "Pclass_3"]].head()
def recode_sibsp(df: pd.DataFrame) -> pd.DataFrame:
    if c.SibSp not in df.columns.values:
        return df

    dfcopy = df.copy()
    assert dfcopy.SibSp.isnull().sum() == 0

    return dfcopy.assign(HasSib = [k > HAS_SIB_THRESHOLD for k in dfcopy.SibSp])

def recode_parch(df: pd.DataFrame) -> pd.DataFrame:
    if c.Parch not in df.columns.values:
        return df

    dfcopy = df.copy()
    assert dfcopy.Parch.isnull().sum() == 0

    return dfcopy.assign(HasParch = [k > HAS_PARCH_THRESHOLD for k in dfcopy.Parch])

@Pipeline.step
def recode_has_family(df: pd.DataFrame) -> pd.DataFrame:
    if c.SibSp not in df or c.Parch not in df:
        return df

    dfcopy = df.copy()
    df_sibsp = recode_sibsp(dfcopy)
    df_parch = recode_parch(dfcopy)
    df_family = df_sibsp.merge(df_parch)
    has_family = [m or n for (m,n) in zip(df_sibsp.HasSib, df_parch.HasParch)]
    return df_family.assign(HasFamily = [1 if k else 0 for k in has_family])
train_family = recode_has_family(train)
qry = "(SibSp > {} or Parch > {}) and HasFamily == 1".format(HAS_SIB_THRESHOLD, HAS_PARCH_THRESHOLD)
assert len(train_family.query(qry)) == len(train_family.query("HasFamily == 1"))
contingency(train_family, c.HasFamily, c.Survived)
@Pipeline.step
def recode_fare(df: pd.DataFrame) -> pd.DataFrame:
    if c.Fare not in df.columns.values:
        return df

    dfcopy = df.copy()
    
    median_fare = 0
    
    if not MEDIAN_FARE:
        median_fare = dfcopy.Fare.median()
    
    dfcopy.Fare.fillna(median_fare, inplace=True)
    
    X = dfcopy.Fare.values.reshape(-1,1)
    km = KMeans(n_clusters=NUM_FARE_CLUSTERS, random_state=0)
    results = km.fit(X)
    dfcopy = dfcopy.assign(FareCluster = results.predict(X))
    
    return dfcopy.join(pd.get_dummies(dfcopy.FareCluster, prefix=c.FareCluster))

train_fare = recode_fare(train)
centers = pd.unique(train_fare.FareCluster.values)
n_centers = len(centers)
fig, axes = plt.subplots(len(centers), sharex=True, sharey=True)
for i, k in zip(range(n_centers+1), centers):
    ax = axes[i]
    train_fare[train_fare.FareCluster == i].Fare.plot.kde(ax=ax)
minS = []
maxS = []
clusterS = range(train_fare.FareCluster.min(),train_fare.FareCluster.max()+1)
for k in clusterS:
    g = train_fare[train_fare.FareCluster == k]
    minS.append(g.Fare.min())
    maxS.append(g.Fare.max())
    
df = pd.DataFrame()
df.assign(FareCluster = clusterS, MinFare=minS, MaxFare=maxS).sort_values(by="MinFare")
contingency(train_fare, c.FareCluster, c.Survived)
@Pipeline.step
def recode_age(df: pd.DataFrame) -> pd.DataFrame:
    if c.Age not in df.columns.values:
        return df

    dfcopy = df.copy()
    
    median_age = 0
    
    if not MEDIAN_AGE:
        median_age = dfcopy.Age.median()
    
    dfcopy.Age.fillna(median_age, inplace=True)
    X = dfcopy.Age.values.reshape(-1,1)
    km = KMeans(n_clusters=NUM_AGE_CLUSTERS, random_state=0)
    results = km.fit(X)
    dfcopy = dfcopy.assign(AgeCluster = results.predict(X))
    return dfcopy.join(pd.get_dummies(dfcopy.AgeCluster, prefix=c.AgeCluster))
train_age = recode_age(train)
assert train_age.AgeCluster.isnull().sum() == 0
centers = sorted(pd.unique(train_age.AgeCluster.values))
n_centers = len(centers)
fig, axes = plt.subplots(len(centers)-1, sharex=True, sharey=True)
for i, k in zip(range(0, n_centers+1), centers[1:]):
    ax = axes[i]
    train_age[train_age.AgeCluster == i].Age.plot.kde(ax=ax)
minS = []
maxS = []
clusterS = range(train_age.AgeCluster.min(),train_age.AgeCluster.max()+1)
for k in clusterS:
    g = train_age[train_age.AgeCluster == k]
    minS.append(g.Age.min())
    maxS.append(g.Age.max())
    
df = pd.DataFrame()
df.assign(AgeCluster = clusterS, MinAge=minS, MaxAge=maxS).sort_values(by="MinAge")
contingency(train_age, c.AgeCluster, c.Survived)
@Pipeline.step
def recode_embarked(df: pd.DataFrame) -> pd.DataFrame:
    if c.Embarked not in df.columns.values:
        return df
    
    dfcopy = df.copy()
    
    dfcopy.Embarked.fillna(dfcopy.Embarked.mode()[0], inplace=True)
    return dfcopy.join(pd.get_dummies(dfcopy.Embarked, prefix=c.Embarked))

train_embarked = recode_embarked(train)
train_embarked[[c.Embarked, "Embarked_C", "Embarked_Q", "Embarked_S"]].head()
def code_sarch(df: pd.DataFrame) -> pd.DataFrame:
    if c.SibSp not in df.columns.values:
        return df
    
    if c.Parch not in df.columns.values:
        return df
    
    if c.Age not in df.columns.values:
        return df
    
    dfcopy = df.copy()
    sibsp1 = dfcopy.SibSp == 1
    parch0 = dfcopy.Parch == 0
    titleS = [extract_title(k) for k in dfcopy.Name]
    title_mr_or_mrs = [k in MARRIED_TITLES for k in titleS]
    
    median_age = 0
    
    if not MEDIAN_AGE:
        median_age = dfcopy.Age.median()
    
    dfcopy.Age.fillna(median_age, inplace=True)
    
        
    age_gt_cutoff = [k > TRUE_LOVE_AGE_CUTOFF for k in dfcopy.Age]
    return dfcopy.assign(Sarch = 1 * (sibsp1 & parch0 & title_mr_or_mrs & age_gt_cutoff))
train_sarch = code_sarch(train)
contingency(train_sarch, c.Sarch, c.Survived)
train_recoded = Pipeline.process(train)
columns_to_drop = [c.PassengerId, c.Pclass, c.Name, c.Sex, c.Age, c.SibSp, c.Parch, c.Ticket, c.Fare, c.Cabin, c.Embarked, c.HasSib, c.HasParch, c.Title, c.AgeCluster, c.FareCluster]
train_recoded.drop(columns_to_drop, axis=1, inplace=True)
train_recoded.info()
X = train_recoded.drop([c.Survived], axis=1)
y = train_recoded.Survived
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
def getSVM(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame) -> GridSearchCV:
    parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
    svc = svm.SVC(gamma="scale")
    clf = GridSearchCV(svc, parameters, cv=5)
    clf.fit(X_train, y_train)
    print("score %s" % (clf.score(X_test, y_test)))
    y_pred = clf.predict(X_test)
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d")
    return clf

SVM_MODEL = getSVM(X_train, X_test, y_train, y_test)
from sklearn.neighbors import KNeighborsClassifier
def getKN():
    scores = []
    bestScore = 0
    bestModel = None
    for n in range(1, 10):
        clf = KNeighborsClassifier(n_neighbors=n)
        clf.fit(X_train, y_train)
        s = clf.score(X_test, y_test)
        if bestScore < s:
            bestScore = s
            bestModel = clf
        print("%s score %s" % (n, s))
        scores.append(s)
    plt.plot(range(1, 10), scores, 'ro')
    plt.show()
    y_pred = bestModel.predict(X_test)
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d")
    return bestModel
getKN()
from sklearn.ensemble import RandomForestClassifier
def getRFT():
    gnb = RandomForestClassifier(n_estimators=100, random_state=20)
    gnb.fit(X_train, y_train)
    print("score %s" % (gnb.score(X_test, y_test)))
    y_pred = gnb.predict(X_test)
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d")
    return gnb
getRFT()

def getRFTGridSearch():
    from sklearn.model_selection import GridSearchCV
    from sklearn.ensemble import RandomForestClassifier

    rfc = RandomForestClassifier(n_jobs=-1,max_features= 'sqrt' ,n_estimators=50, oob_score = True, random_state=20) 

    param_grid = { 
        'n_estimators': [200, 700],
        'max_features': ['auto', 'sqrt', 'log2']
    }
    CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
    CV_rfc.fit(X_train, y_train)
    print("score %s" % (CV_rfc.score(X_test, y_test)))
    forest = RandomForestClassifier(n_estimators = CV_rfc.best_params_['n_estimators'], max_features= CV_rfc.best_params_['max_features'], random_state=20)
    forest.fit(X_train, y_train)
    
    # get feature importance
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")
    indcesOfFeatureNames = [ X_train.columns[f] for f in indices ]
    for f in range(X.shape[1]):
        print("%d. feature %s (index %d )(%f)" % (f + 1, X_train.columns[f], indices[f], importances[indices[f]]))

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(X.shape[1]), importances[indices],
           color="r", yerr=std[indices], align="center")
    plt.xticks(range(X.shape[1]), indcesOfFeatureNames, rotation=70)
    plt.xlim([-1, X.shape[1]])
    plt.show()
    
    return forest
getRFTGridSearch()
def getCatBoost():
    from catboost import CatBoostClassifier
    model = CatBoostClassifier(iterations=10, learning_rate=1, depth=4, loss_function='Logloss', random_state=20)
    model.fit(X_train, y_train)
    print("score %s" % (model.score(X_test, y_test)))
    y_pred = model.predict(X_test)
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d")
    return model
CATBOOST_MODEL = getCatBoost()
test_recoded = Pipeline.process(test)
columns_to_drop = [c.PassengerId, c.Pclass, c.Name, c.Sex, c.Age, c.SibSp, c.Parch, c.Ticket, c.Fare, c.Cabin, c.Embarked, c.HasSib, c.HasParch, c.Title, c.AgeCluster, c.FareCluster]
test_recoded.drop(columns_to_drop, axis=1, inplace=True)
test_recoded.info()
y_pred = SVM_MODEL.predict(test_recoded)
submission = pd.DataFrame()
submission = submission.assign(PassengerId = test.PassengerId, Survived = y_pred)
submission.to_csv("titanic_survival_prediction.csv", index=False)
submission.head()
y_pred = CATBOOST_MODEL.predict(test_recoded)
submission = pd.DataFrame()
submission = submission.assign(PassengerId = test.PassengerId, Survived = y_pred)
submission.to_csv("titanic_survival_prediction_CATBOOST_MODEL.csv", index=False)
submission.head()
