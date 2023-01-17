import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('https://raw.githubusercontent.com/PacktWorkshops/The-Data-Science-Workshop/master/Chapter06/Dataset/bank-additional-full.csv', sep=';')
df.info()
df.head()
cat_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']

_df = pd.get_dummies(df, columns=cat_cols, prefix=cat_cols, drop_first=True)

_df.info()

_df.head()
X = _df.drop(['y'], axis=1)

X = X.values

y = df['y'].apply(lambda x: 0 if x == 'no' else 1)

y = y.values

train_X, eval_X, train_y, eval_y = train_test_split(X, y, test_size=0.3, random_state=0)

val_X, test_X, val_y, test_y = train_test_split(eval_X, eval_y, random_state=0)

lr_model = LogisticRegression()
lr_model.fit(train_X, train_y)
lr_preds = lr_model.predict(val_X)
lr_report = classification_report(val_y, lr_preds)

print(lr_report)

dt_model = DecisionTreeClassifier(max_depth= 6)
dt_model.fit(train_X, train_y)
dt_preds = dt_model.predict(val_X)
dt_report = classification_report(val_y, dt_preds)

print(dt_report)

rf_model = RandomForestClassifier(n_estimators=1000)

rf_model.fit(train_X, train_y)
rf_preds = rf_model.predict(val_X)
rf_report = classification_report(val_y, rf_preds)

print(rf_report)

print('Linear Score: {}, DecisionTree Score: {}, RandomForest Score: {}'.format(lr_model.score(val_X, val_y), dt_model.score(val_X, val_y), rf_model.score(val_X, val_y)))