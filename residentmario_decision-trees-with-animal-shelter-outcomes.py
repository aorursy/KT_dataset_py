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
df.head()
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

clf = DecisionTreeClassifier(max_depth=3)
clf.fit(X, y)
y_hat = clf.predict(df)
accuracy_score(y, y_hat)
import graphviz
from sklearn.tree import export_graphviz
from IPython.display import display

dot_data = export_graphviz(clf, out_file=None, feature_names=X.columns, class_names=['Not Adopted', 'Adopted'], filled=True)
graph = graphviz.Source(dot_data)
display(graph)
# graph.render("shelter_outcomes") 
from sklearn.model_selection import KFold
import numpy as np

clf = DecisionTreeClassifier(max_depth=3)
for train, test in KFold(n_splits=5).split(X):
    X_train, X_test = X.loc[train], X.loc[test]
    y_train, y_test = y.loc[train], y.loc[test]
    clf.fit(X_train, y_train)
    print(accuracy_score(y_test, clf.predict(X_test)))