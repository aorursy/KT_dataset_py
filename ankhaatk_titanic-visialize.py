# import Required Library
import numpy as np
import pandas as pd
# for Chart
from bokeh.plotting import figure
from bokeh.io import output_notebook, show
from bokeh.models import ColumnDataSource, FactorRange, ranges, LabelSet

output_notebook()
# Read Data
df = pd.read_csv("../input/train.csv") 
# Clean Bad Values
df = df.dropna() 
df[['Age']] = df[['Age']].astype(int)
df.head()
surv_counts = df.Survived.value_counts()
source = ColumnDataSource(dict(x=["Died", "Survived"],y=surv_counts.values))

plot = figure(plot_width=300, plot_height=300, title="Total Survived Stat", x_range = source.data["x"])

plot.vbar(source=source, x='x', width=0.5, top='y', name='y')
show(plot)
surv_counts = df.Survived[df.Sex == 'female'].value_counts()
source = ColumnDataSource(dict(x=["Died", "Survived"],y=surv_counts.values))

plot = figure(plot_width=300, plot_height=300, title="Total Female Survived Stat", y_axis_label = "count", x_range = source.data["x"])

plot.vbar(source=source, width=0.5, x='x', top='y')

show(plot)
surv_counts = df.Survived[df.Sex == 'male'].value_counts()
source = ColumnDataSource(dict(x=["Died", "Survived"],y=surv_counts.values))

plot = figure(plot_width=300, plot_height=300, title="Total Male Survived Stat", y_axis_label = "count", x_range = source.data["x"])

plot.vbar(source=source, width=0.5, x='x', top='y')

show(plot)
gender_counts = df.Sex[df.Survived == 1].value_counts()
source = ColumnDataSource(dict(x=gender_counts.keys().tolist(),y=gender_counts.values))

plot = figure(plot_width=300, plot_height=300, title="Gender Survived Stat", y_axis_label = "count", x_range = source.data["x"])

plot.vbar(source=source, width=0.5, x='x', top='y')

show(plot)
# Learning Add Simple Model
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.ensemble import RandomForestRegressor

train_data = pd.read_csv("../input/train.csv")
train_data[['Age']] = train_data[['Age']].fillna(value=32)

train_data[['Age']] = train_data[['Age']].astype(int)

# Replase Age Attr. 0 - kid, 1 - Young, 2- Mid, 3- Old
train_data.loc[train_data.Age < 12, "Age"] = 0 # Kid
train_data.loc[(train_data.Age >= 12) & (train_data.Age < 40), "Age"] = 1 # Young
train_data.loc[(train_data.Age >= 40) & (train_data.Age < 60), "Age"] = 2 # Mid
train_data.loc[train_data.Age >= 60, "Age"] = 3

from sklearn.preprocessing import LabelBinarizer, LabelEncoder
str_cols = train_data.columns[train_data.columns.str.contains('Sex')]
clfs = {c:LabelEncoder() for c in str_cols}
for col, clf in clfs.items():
    train_data[col] = clfs[col].fit_transform(train_data[col])

train_data.head()

# Learing
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch']

X = train_data[features]

from sklearn.ensemble import RandomForestRegressor

titanic_model = RandomForestRegressor(random_state=1)
y = train_data.Survived
titanic_model.fit(X, y)
test_data = pd.read_csv('../input/test.csv')

test_data[['Age']] = test_data[['Age']].fillna(value=32)
test_data[['Age']] = test_data[['Age']].astype(int)

# Replase Age Attr. 0 - kid, 1 - Young, 2- Mid, 3- Old
test_data.loc[test_data.Age < 12, "Age"] = 0 # Kid
test_data.loc[(test_data.Age >= 12) & (test_data.Age < 40), "Age"] = 1 # Young
test_data.loc[(test_data.Age >= 40) & (test_data.Age < 60), "Age"] = 2 # Mid
test_data.loc[test_data.Age >= 60, "Age"] = 3


str_cols = test_data.columns[test_data.columns.str.contains('Sex')]
clfs = {c:LabelEncoder() for c in str_cols}
for col, clf in clfs.items():
    test_data[col] = clfs[col].fit_transform(test_data[col])

test_data.describe()
test_X = test_data[features]

test_predictions = titanic_model.predict(test_X)
int_test_predictions = test_predictions.astype(int)


my_submission = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': int_test_predictions})
my_submission.to_csv('submission.csv', index=False)