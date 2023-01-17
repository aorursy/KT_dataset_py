import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.base as skb
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from xgboost.sklearn import XGBClassifier

data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')

all_data = pd.concat([data, test_data])
all_data.reset_index(drop=True, inplace=True)

all_data.describe(include='all')
def breakdown_desc(attrib, eff_attrib, orient='v', data=data):
    if orient == 'h':
        x = eff_attrib
        y = attrib
    else:
        x = attrib
        y = eff_attrib
    sns.barplot(x=data[x], y=data[y], order=data.groupby(by=attrib)[eff_attrib].mean().sort_values().index)
    plt.tight_layout()
all_data['Title'] = all_data.Name.str.extract('([A-Za-z]+)\\.', expand=False)

all_data.Cabin.fillna('U', inplace=True)
all_data['Deck'] = all_data.Cabin.apply(lambda x: x[0])

all_data['LName'] = all_data.Name.str.extract('([A-Za-z]+),', expand=False)

all_data['FName'] = all_data.Name.str.extract('\\(([A-Za-z]+)', expand=False)
all_data.loc[all_data.FName.isnull(), 'FName'] = all_data.Name[all_data.FName.isnull()] \
.str.extract('\\.\s([A-Za-z]+)', expand=False)

vc = all_data.LName.value_counts()
other = vc[vc < 2]
other[:] = 'Other'
all_data.LName = all_data.LName.replace(other)

vc = all_data.FName.value_counts()
other = vc[vc < 4]
other[:] = 'Other'
all_data.FName = all_data.FName.replace(other)


# all_data.drop(columns=['Name', 'Cabin', 'Ticket'], inplace=True)
all_data.drop('Name', axis=1, inplace=True)
all_data.drop('Cabin', axis=1, inplace=True)
all_data.drop('Ticket', axis=1, inplace=True)
title_med_age = all_data[~all_data.Age.isnull()].groupby(by='Title').Age.median()
sns.barplot(y=title_med_age.index, x=title_med_age.values, 
            order=title_med_age.sort_values().index).set(xlabel='Median Age')
all_data.loc[all_data.Age.isnull(), 'Age'] = title_med_age[all_data[all_data.Age.isnull()].Title].values
sns.countplot(all_data.Embarked)
all_data.Embarked.fillna('S', inplace=True)
class_med_fare = all_data[~all_data.Fare.isnull()].groupby(by='Pclass').Fare.median()

sns.barplot(x=class_med_fare.index, y=class_med_fare.values,
            order=class_med_fare.sort_values().index).set(ylabel='Median Fare')

all_data.loc[all_data.Fare.isnull(), 'Fare'] = class_med_fare[all_data[all_data.Fare.isnull()].Pclass].values
all_data.describe(include='all')
all_data.head()
breakdown_desc('Embarked', 'Survived', data=all_data[~all_data.Survived.isnull()])
emb_map = {
    'S':0,
    'Q':1,
    'C':2
}

all_data.Embarked = all_data.Embarked.map(emb_map)
breakdown_desc('Parch', 'Survived', data=all_data[~all_data.Survived.isnull()])
parch_map = {
    9: 0,
    6: 1,
    5: 2,
    4: 3,
    0: 4,
    2: 5,
    1: 6,
    3: 7
}


all_data.Parch = all_data.Parch.map(parch_map)
breakdown_desc('Pclass', 'Survived', data=all_data[~all_data.Survived.isnull()])
breakdown_desc('SibSp', 'Survived', data=all_data[~all_data.Survived.isnull()])
sibsp_map = {
    8: 0,
    5: 0,
    4: 1,
    3: 2,
    0: 3,
    2: 4,
    1: 5
}

all_data.SibSp = all_data.SibSp.map(sibsp_map)
breakdown_desc('Title', 'Survived', data=all_data[~all_data.Survived.isnull()], orient='h')
title_map = {
    # Likely to help others, mostly old
    'Rev': 0,
    'Capt': 0,
    
    # Mostly adult males
    'Mr': 1,
    'Dr': 2,
    
    # Military
    'Col': 3,
    'Major': 3,
    
    # Women and children
    'Master': 4,
    'Miss': 5,
    'Ms': 5,
    'Mlle': 5,
    'Mrs': 6,
    'Mme': 6,
    
    # Lordship titles
    'Jonkheer': 7,
    'Don': 7,
    'Dona': 7,
    'Sir': 7,
    'Lady': 7,
    'Countess': 7
}

all_data.Title = all_data.Title.map(title_map)
breakdown_desc('Deck', 'Survived', data=all_data[~all_data.Survived.isnull()])
deck_map = {
    'T': 0,
    'U': 1,
    'A': 2,
    'G': 3,
    'C': 4,
    'F': 5,
    'B': 6,
    'E': 7,
    'D': 8
}

all_data.Deck = all_data.Deck.map(deck_map)
# surv_prob = {}
# vcount = all_data.LName.value_counts()
# for name in all_data.LName.unique():
#     surv_count = np.sum(all_data[all_data.LName == name].Survived)
#     surv_prob[name] = surv_count / vcount[name]

# surv_prob_by_lname = pd.Series(surv_prob).sort_values()

# lname_map = dict(zip(surv_prob_by_lname.index, np.arange(0, len(surv_prob_by_lname))))

# all_data.LName = all_data.LName.map(lname_map)

# surv_prob = {}
# vcount = all_data.FName.value_counts()
# for name in all_data.FName.unique():
#     surv_count = np.sum(all_data[all_data.FName == name].Survived)
#     surv_prob[name] = surv_count / vcount[name]

# surv_prob_by_fname= pd.Series(surv_prob).sort_values()

# fname_map = dict(zip(surv_prob_by_fname.index, np.arange(0, len(surv_prob_by_fname))))

# all_data.FName = all_data.FName.map(fname_map)
all_data.Sex = all_data.Sex.map({'male':0, 'female':1})

all_data.LName = LabelEncoder().fit_transform(all_data.LName)
all_data.FName = LabelEncoder().fit_transform(all_data.FName)

# all_data.Fare = np.log(all_data.Fare+1)

bins = np.arange(all_data.Fare.min()-1, all_data.Fare.max(), 12)
bins[-1] = all_data.Fare.max()+1
all_data.Fare = LabelEncoder().fit_transform(pd.cut(all_data.Fare, bins))

bins = [0, 14, 18, 25, 35, 50, 60, 80]
all_data.Age = LabelEncoder().fit_transform(pd.cut(all_data.Age, bins))

to_scale = ['Age', 'Embarked', 'Fare', 'Parch', 'Pclass', 'SibSp', 'Title', 'Deck']
all_data[to_scale] = StandardScaler().fit_transform(all_data[to_scale])
all_data.head()
data_train = all_data[~all_data.Survived.isnull()].drop('PassengerId', axis=1)
data_test = all_data[all_data.Survived.isnull()].drop('Survived', axis=1)

data_train.reset_index(drop=True, inplace=True)
data_test.reset_index(drop=True, inplace=True)

data_train.to_csv('train_proc.csv', index=False)
data_test.to_csv('test_proc.csv', index=False)
sns.heatmap(data_train.corr())
import warnings
warnings.filterwarnings("ignore")  # XGB/numpy Warning

X = data_train.drop('Survived', axis=1)
y = data_train.Survived

skf = StratifiedKFold(n_splits=8)

models = [SVC(), LogisticRegression(), XGBClassifier(), RandomForestClassifier(), LinearDiscriminantAnalysis()]

model_stack = []

trained_comm = {}

iterations = 30

acc_tot = 0.

print("Training...")
for i in range(iterations):
    print('Iter:', i)
    for train_index, test_index in skf.split(X, y):
        X_train = X.iloc[train_index]
        y_train = y.iloc[train_index]
        X_test = X.iloc[test_index]
        y_test = y.iloc[test_index]

        for base in models:
            model = skb.clone(base)
            model.fit(X_train, y_train)
            acc = accuracy_score(y_test, model.predict(X_test))
            acc_tot += acc
            model_stack.append((model, acc))
print('Done.')            
print("Avg. Acc:", acc_tot / len(model_stack))
model = model_stack[3][0]
importances = model.feature_importances_
table = pd.Series(importances, index = X_test.columns).sort_values(ascending=False)
sns.barplot(x=table.values, y=table.index)
pred = np.zeros(data_test.shape[0])

X_test = data_test.drop('PassengerId', axis=1)

for model, w in model_stack:
    pred += model.predict(X_test) * w
pred = np.round(pred/len(model_stack))

df = pd.DataFrame(columns=['PassengerId', 'Survived'])
df.PassengerId = data_test.PassengerId
df.Survived =  pd.to_numeric(pred, downcast='integer')

df.to_csv('output.csv', index=False)

df.Survived.value_counts()