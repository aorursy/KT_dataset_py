import pandas as pd
df = pd.read_csv("../input/train.csv")
target_per_household = df.groupby(['idhogar'])['Target'].nunique()

no_target = len(target_per_household.loc[target_per_household == 0])
unique_target = len(target_per_household.loc[target_per_household == 1])
more_targets = len(target_per_household.loc[target_per_household > 1])
more_targets_perc = more_targets / (no_target + unique_target + more_targets)

print("No per household: {}".format(no_target))
print("1 target per household: {}".format(unique_target))
print("More targets per household: {} or {:.1f}%" .format(more_targets, more_targets_perc * 100))
df.loc[(df.idhogar == '1b31fd159'), 'meaneduc'] = 10
df.loc[(df.idhogar == 'a874b7ce7'), 'meaneduc'] = 5
df.loc[(df.idhogar == 'faaebf71a'), 'meaneduc'] = 12
df.edjefe.replace({'no': 0}, inplace=True)
df.edjefa.replace({'no': 0}, inplace=True)
df.edjefe.replace({'yes': 1}, inplace=True)
df.edjefa.replace({'yes': 1}, inplace=True)
categorical_features = df.columns.tolist()
for feature in df.describe().columns:
    categorical_features.remove(feature)

# Just for saving them
numerical_features = df.columns.tolist()
for categorical_feature in categorical_features:
    numerical_features.remove(categorical_feature)
    
categorical_features
df[categorical_features].head()
df[['edjefe', 'SQBedjefe']].head()
df[['edjefe', 'SQBedjefe']].head()
print("Number of observations {}".format(len(df)))
features_with_null = df.isna().sum().sort_values(ascending=False)
features_with_null = features_with_null.loc[features_with_null > 0]
feature_names_with_null = features_with_null.index.tolist()

features_with_null
selectable_features = numerical_features.copy()
selectable_features.remove('Target')
for feature in feature_names_with_null:
    selectable_features.remove(feature)

X = df[selectable_features]
y = df.Target
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=112, test_size=0.2)
# from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
clf= RandomForestClassifier()
clf.fit(X_train, y_train)

sorted(zip(X.columns, clf.feature_importances_ * 100), key=lambda x: -x[1])
id_with_null = df[df.meaneduc.isna()].idhogar
df[df.idhogar.isin(id_with_null)][['idhogar', 'meaneduc', 'escolari', 'age']]
df.loc[(df.idhogar == '1b31fd159'), 'meaneduc'] = 10
df.loc[(df.idhogar == 'a874b7ce7'), 'meaneduc'] = 5
df.loc[(df.idhogar == 'faaebf71a'), 'meaneduc'] = 12
df.edjefe.replace({'no': 0}, inplace=True)
df.edjefa.replace({'no': 0}, inplace=True)
df.edjefe.replace({'yes': 1}, inplace=True)
df.edjefa.replace({'yes': 1}, inplace=True)
selectable_features = numerical_features.copy()
selectable_features.append('edjefe')
selectable_features.append('edjefa')
selectable_features.remove('Target')
for feature in feature_names_with_null:
    selectable_features.remove(feature)

X = df[selectable_features]
y = df.Target
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=112, test_size=0.2)
selected_features = ['SQBedjefe', 'SQBdependency', 'overcrowding', 'qmobilephone', 'SQBage', 'rooms', 'SQBhogar_nin', 'edjefe', 'edjefa' ]

X_train_4predict = X_train[selected_features]
predictor = RandomForestClassifier()
predictor.fit(X_train_4predict, y_train)
X_test_4predict = X_test[selected_features]
y_predict = predictor.predict(X_test_4predict)
from sklearn.metrics import precision_recall_fscore_support as score

precision, recall, fscore, support = score(y_test, y_predict)

print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))
print('support: {}'.format(support))
from sklearn.metrics import f1_score
f1_score(y_test, y_predict, average='macro')
df_eval = pd.read_csv("../input/test.csv")
df_eval.edjefe.replace({'no': 0}, inplace=True)
df_eval.edjefa.replace({'no': 0}, inplace=True)
df_eval.edjefe.replace({'yes': 1}, inplace=True)
df_eval.edjefa.replace({'yes': 1}, inplace=True)
X_eval = df_eval[selected_features]
df_eval['Target'] = predictor.predict(X_eval)
df_eval[['Id', 'Target']].to_csv("sample_submission.csv", index=False)