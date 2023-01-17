import pandas as pd
outcomes = pd.read_csv("../input/aac_shelter_outcomes.csv")

df = (outcomes
      .assign(
         age=(pd.to_datetime(outcomes['datetime']) - pd.to_datetime(outcomes['date_of_birth'])).map(lambda v: v.days)
      )
      .rename(columns={'sex_upon_outcome': 'sex', 'animal_type': 'type'})
      .loc[:, ['type', 'breed', 'color', 'sex', 'age']]
)
df = pd.get_dummies(df[['type', 'breed', 'color', 'sex']]).assign(age=df['age'] / 365)
X = df
y = outcomes.outcome_type.map(lambda v: v == "Adoption")
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from tqdm import tqdm

clf = BaggingClassifier(base_estimator=DecisionTreeClassifier(max_depth=5), n_estimators=10)

kf = KFold(n_splits=4)
scores = []

for train_index, test_index in tqdm(kf.split(X)):
    X_train, X_test = X.loc[train_index], X.loc[test_index]
    y_train, y_test = y.loc[train_index], y.loc[test_index]
    
    scores.append(
        accuracy_score(clf.fit(X_train, y_train).predict(X_test), y_test)
    )
scores
sum(scores) / len(scores)
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=10, max_features=5)

kf = KFold(n_splits=4)
scores = []

for train_index, test_index in tqdm(kf.split(X)):
    X_train, X_test = X.loc[train_index], X.loc[test_index]
    y_train, y_test = y.loc[train_index], y.loc[test_index]
    
    scores.append(
        accuracy_score(clf.fit(X_train, y_train).predict(X_test), y_test)
    )
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
