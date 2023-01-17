import pandas as pd
import seaborn as sns
import warnings
import numpy as np
from matplotlib import pyplot as plt

warnings.filterwarnings('ignore')
sns.set(style="darkgrid")
## Read data files
train = pd.read_csv("../input/train/train.csv")
test = pd.read_csv("../input/test/test.csv")
train.head()
fig = plt.figure(figsize=(15,8))
ax = fig.add_subplot(121)
    
ax = sns.countplot(x="AdoptionSpeed", data=train, palette="YlOrRd", edgecolor="black")

ax.set_ylabel('Count')
ax.set_xlabel('Adoption Speed')
ax.set_title('Adoption Speed')

ax = fig.add_subplot(122)
    
ax = sns.countplot(x="Type", data=train, palette="YlOrRd", edgecolor="black")

ax.set_ylabel('Count')
ax.set_xlabel('Type')
ax.set_title('Count by Type (1 = Dog, 2 = Cat)')

plt.show()

fig = plt.figure(figsize=(15,8))
ax = fig.add_subplot(1,1,1)
    
ax = sns.countplot(x="AdoptionSpeed", hue="Type", data=train, palette="YlOrRd", edgecolor="black")

ax.set_ylabel('Count')
ax.set_xlabel('Adoption Speed')
ax.set_title('Adoption Speed by Type (1 = Dog, 2 = Cat)')

plt.show()
def generate_groupBY_data(col_name):
    l = train.groupby(['AdoptionSpeed', col_name])[['PetID']].count().reset_index().rename(
        columns={'PetID':'count'})
    count_pets = train.groupby(['AdoptionSpeed', col_name])[['PetID']].count().reset_index().groupby(
        [col_name]).sum()[['PetID']].reset_index()
    new_col_name = 'total_pets_by' + col_name
    count_pets.rename(columns={'PetID': new_col_name}, inplace=True)

    temp = l.merge(count_pets, on=[col_name], how='left')
    temp['fraction'] = temp['count'] * 100 / temp[new_col_name]

    temp = temp.pivot("AdoptionSpeed", col_name, "fraction")
    
    return temp
# Plots for Categorical Variables
list_cols = ['Type', 'Gender', 'Color1', 'Color2', 'Color3', 'MaturitySize', 'FurLength', 'Vaccinated', 'Dewormed',
            'Sterilized', 'Health', 'State']

cols = 2
rows = len(list_cols)

fig = plt.figure(figsize=(15,5*rows))
fig_no = 0

for col in list_cols:
    fig_no += 1
    ax = fig.add_subplot(rows, cols, fig_no)
    sns.countplot(x=col, data=train, palette="YlOrRd", edgecolor="black")
    ax.set_xlabel(col)
    ax.set_ylabel('Count')
    ax.set_title('Count of pets by ' + col)
    
    temp = generate_groupBY_data(col)
    fig_no += 1
    ax = fig.add_subplot(rows, cols, fig_no)
    sns.heatmap(temp, annot=True, cmap='YlOrRd')
    ax.set_xlabel(col)
    ax.set_ylabel('Adoption Speed')
    ax.set_title('Percentage of pets adopted by ' + col)

fig.tight_layout()
plt.show()
g = sns.catplot(x="AdoptionSpeed", y="PhotoAmt", kind="box", data=train, height=8, aspect=1.5, palette="YlOrRd",
               showfliers=False)

g.axes[0,0].set_xlabel('Adoption Speed')
g.axes[0,0].set_ylabel('Photo Amount')
g.axes[0,0].set_title('Photo Amount vs Adoption Speed (after removing outliers)')

plt.show()
### Figure 1
fig = plt.figure(figsize=(15,8))

ax = fig.add_subplot(121)
sns.kdeplot(train.loc[train['Type'] == 1]['Age'], shade=True, color="red", label='Dogs')
sns.kdeplot(train.loc[train['Type'] == 2]['Age'], shade=True, color="green", label='Cats')
ax.set_xlabel('Age')
ax.set_title('Distribution of Age by Type')

ax = fig.add_subplot(122)
sns.boxplot(x="Type", y="Age", data=train, palette="YlOrRd", showfliers=False)
ax.set_xlabel('Type')
ax.set_title('Distribution of Age by Type (after removing outliers) (1 = Dog, 2 = Cat)')

plt.show()

### Figure 2
g = sns.catplot(x="AdoptionSpeed", y="Age", kind="box", data=train, height=8, aspect=1.56, palette="YlOrRd",
                showfliers=False)
g.axes[0,0].set_xlabel('Adoption Speed')
g.axes[0,0].set_ylabel('Age')
g.axes[0,0].set_title('Age vs Adoption Speed (after removing outliers)')

### Figure 3
fig = plt.figure(figsize=(15,8))

ax = fig.add_subplot(121)
sns.boxplot(x="AdoptionSpeed", y="Age", data=train.loc[train['Type'] == 1], palette="YlOrRd", showfliers=False)
ax.set_ylim((-5,60))
ax.set_xlabel('Adoption Speed')
ax.set_title('Distribution of Age by Adoption Speed for Dog (after removing outliers)')

ax = fig.add_subplot(122)
sns.boxplot(x="AdoptionSpeed", y="Age", data=train.loc[train['Type'] == 2], palette="YlOrRd", showfliers=False)
ax.set_ylim((-5,60))
ax.set_xlabel('Adoption Speed')
ax.set_title('Distribution of Age by Adoption Speed for Cat (after removing outliers)')

plt.show()
train['Name_absent'] = train['Name'].isnull()
train['Name_absent'] = train['Name_absent'].astype(int)
train['Description_None'] = train['Description'].isnull()
train['Description_None'] = train['Description_None'].astype(int)

### Figure 1
fig = plt.figure(figsize=(15,8))

ax = fig.add_subplot(121)
sns.countplot(x="AdoptionSpeed", data=train.loc[train['Name_absent'] == 0], palette="YlOrRd", edgecolor="black")
ax.set_ylim((0,4000))
ax.set_xlabel('Adoption Speed')
ax.set_ylabel('Count')
ax.set_title('Count by Adoption Speed (with Name)')

ax = fig.add_subplot(122)
sns.countplot(x="AdoptionSpeed", data=train.loc[train['Name_absent'] == 1], palette="YlOrRd", edgecolor="black")
ax.set_ylim((0,4000))
ax.set_xlabel('Adoption Speed')
ax.set_ylabel('Count')
ax.set_title('Count by Adoption Speed (with no Name)')

plt.show()

### Figure 1
fig = plt.figure(figsize=(15,8))

ax = fig.add_subplot(121)
sns.countplot(x="AdoptionSpeed", data=train.loc[train['Description_None'] == 0], palette="YlOrRd", edgecolor="black")
ax.set_xlabel('Adoption Speed')
ax.set_ylabel('Count')
ax.set_title('Count by Adoption Speed (with Description)')

ax = fig.add_subplot(122)
sns.countplot(x="AdoptionSpeed", data=train.loc[train['Description_None'] == 1], palette="YlOrRd", edgecolor="black")
ax.set_xlabel('Adoption Speed')
ax.set_ylabel('Count')
ax.set_title('Count by Adoption Speed (with no Description)')

plt.show()
train['Name'].fillna('None', inplace=True)
train['Description'].fillna('None', inplace=True)

test['Name_absent'] = test['Name'].isnull()
test['Name_absent'] = test['Name_absent'].astype(int)
test['Description_None'] = test['Description'].isnull()
test['Description_None'] = test['Description_None'].astype(int)
test['Name'].fillna('None', inplace=True)
test['Description'].fillna('None', inplace=True)
# Extract features from description
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer(max_features=100, stop_words='english', use_idf=True)

tfidf_matrix = tfidf_vectorizer.fit_transform(list(train['Description'].values) +
                                                    list(test['Description'].values))

cols = tfidf_vectorizer.get_feature_names()
tfidf_train = pd.DataFrame(tfidf_matrix.toarray()[0:train.shape[0],], columns=cols)
tfidf_test = pd.DataFrame(tfidf_matrix.toarray()[train.shape[0]:,], columns=cols)
train = pd.concat([train, tfidf_train], axis=1)
test = pd.concat([test, tfidf_test], axis=1)
categorical_cols = ['Type', 'Breed1', 'Breed2', 'Gender', 'Color1', 'Color2', 'Color3', 'MaturitySize', 'FurLength',
                    'Vaccinated', 'Dewormed', 'Sterilized', 'Health', 'State', 'Name_absent', 'Description_None']

categorical_features_lgbm = []
for col in categorical_cols:
    categorical_features_lgbm.append('name:' + col)
    train[col] = train[col].astype('category')
    test[col] = test[col].astype('category')
X_train = train.drop(['Name', 'RescuerID', 'Description', 'PetID', 'AdoptionSpeed'], axis=1)
y_train = train['AdoptionSpeed']
X_test = test.drop(['Name', 'RescuerID', 'Description'], axis=1)
from sklearn.model_selection import KFold
from sklearn.metrics import cohen_kappa_score
import lightgbm as lgb
import time

param = {'num_leaves': 38,
         'min_data_in_leaf': 146, 
         'objective':'multiclass',
         'num_class': 5,
         'max_depth': 4,
         'learning_rate': 0.01,
         "boosting": "gbdt",
         "feature_fraction": 0.9980062052116254,
         "bagging_freq": 1,
         "bagging_fraction": 0.844212672233457,
         "bagging_seed": 11,
         "metric": 'multi_logloss',
         "lambda_l1": 0.12757257166471625,
         "random_state": 133,
         "verbosity": -1}

folds = KFold(n_splits=5, shuffle=True, random_state=15)
oof = np.zeros(len(X_train))
start = time.time()
feature_importance_df = pd.DataFrame()

for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train.values, y_train.values)):
    print("fold n°{}".format(fold_))
    trn_data = lgb.Dataset(X_train.iloc[trn_idx],
                           label=y_train.iloc[trn_idx]
                          )
    val_data = lgb.Dataset(X_train.iloc[val_idx],
                           label=y_train.iloc[val_idx]
                          )

    num_round = 10000
    clf = lgb.train(param,
                        trn_data,
                        10000,
                        valid_sets = [trn_data, val_data],
                        verbose_eval=500,
                        early_stopping_rounds = 200)
        
    oof[val_idx] = (pd.DataFrame(clf.predict(X_train.iloc[val_idx], 
                                                     num_iteration=clf.best_iteration)).idxmax(axis=1))

print("CV score: {:<8.5f}".format(cohen_kappa_score(oof, y_train, weights="quadratic")))
p = (pd.DataFrame(clf.predict(X_test.drop(['PetID'], axis=1),
                              num_iteration=clf.best_iteration)).idxmax(axis=1))
test['AdoptionSpeed'] = p
test[['PetID', 'AdoptionSpeed']].to_csv("submission_lgbm_bayes_optimization.csv", index=False)
# Feature importance
features_importance = pd.Series(clf.feature_importance(), index=X_train.columns)
features_importance = features_importance.sort_values(ascending=False)
df = features_importance.to_frame()
df['feature'] = df.index
df = df.rename(columns={0: 'importance'})

fig = plt.figure(figsize=(15,60))
ax = sns.barplot(x="importance", y="feature", data=df)
ax.set_xlabel('Importance')
ax.set_ylabel('Feature')
plt.show()
