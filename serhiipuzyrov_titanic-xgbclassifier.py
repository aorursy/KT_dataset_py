import matplotlib.pyplot as plt
from IPython.display import display, HTML
import numpy as np
import pandas as pd
import seaborn as sns
import warnings
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 500)
df_raw = pd.read_csv('../input/train.csv')
df = df_raw

print(df_raw.info())
display(df_raw.head(25))
df['Title'] = df['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
df['Title'] = df['Title'].replace(['Don', 'Mme', 'Major', 'Lady', 'Sir', 'Mlle', 'Capt', 'the Countess', 'Jonkheer', 'Col', 'Rev'], 'no')

df['Relatives'] = df.SibSp + df.Parch
df['Alone'] = df.Relatives.apply(lambda i: '0' if i>0 else '1')

q1 = df.Fare.quantile(.25)
q2 = df.Fare.quantile(.5)
q3 = df.Fare.quantile(.75)

def quantile(i):
    if i < q1:
        return '0'
    if i >= q1 and i < q2:
        return '1'    
    if i >= q2 and i < q3:
        return '2'
    if i > q3:
        return '3'

df['Fare_quantile'] = df.Fare.apply(quantile)

def parent_child(age, p):
    if age < 18:
        return 'child'
    if age >= 20 and p > 0:
        return 'parent'
    else: 
        return 'other'
df['parent_child'] = np.vectorize(parent_child)(df['Age'], df['Parch'])

display(df.head(10))
# choosing valuable data
vals_heads = ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked', 'Title', 'Relatives', 'Alone', 'Fare_quantile', 'parent_child']
df_vals = df[vals_heads]
df_plt = df[vals_heads]
df = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Title', 'Relatives', 'Alone', 'Fare_quantile', 'parent_child']]
display(df_vals.head())
#fill null records
df.Age.fillna(value=df.Age.mean(), inplace=True)
df.Embarked.fillna(value=df.Embarked.mode(), inplace=True)
df.Fare.fillna(value=df.Fare.median(), inplace=True)
df_plt.Survived = df_plt.Survived.apply(lambda i: "Survived" if i==1 else "Died")

fig = plt.figure(figsize=(20, 10))
ax1 = fig.add_subplot(331)
ax2 = fig.add_subplot(332)
ax3 = fig.add_subplot(333)
ax4 = fig.add_subplot(334)
ax5 = fig.add_subplot(335)
ax6 = fig.add_subplot(336)
ax7 = fig.add_subplot(337)
ax8 = fig.add_subplot(338)
ax9 = fig.add_subplot(339)

sns.countplot(x='Survived', data=df_plt, ax=ax1)
sns.countplot(x='Pclass', data=df_plt, hue='Survived', ax=ax2)
sns.countplot(x='Sex', data=df_plt, hue='Survived', ax=ax3)

sns.countplot(x='SibSp', data=df_plt, hue='Survived', ax=ax4)
sns.countplot(x='Parch', data=df_plt, hue='Survived', ax=ax5).legend(loc='upper right', title="Survived")
sns.countplot(x='Embarked', data=df_plt, hue='Survived', ax=ax6)

sns.countplot(x='Title', data=df_plt, hue='Survived', ax=ax7)
sns.countplot(x='Relatives', data=df_plt, hue='Survived', ax=ax8)
sns.countplot(x='Fare_quantile', data=df_plt, hue='Survived', ax=ax9)
plt.show()

g=sns.FacetGrid(df_plt, hue='Survived').map(sns.kdeplot, 'Fare', label='Survived').add_legend()
g.fig.set_size_inches(20,5)
plt.xticks(range(0, 525, 25))
plt.show()

g=sns.FacetGrid(df_plt, hue='Survived').map(sns.kdeplot, 'Age', label='Survived').add_legend()
g.fig.set_size_inches(20,5)
plt.xticks(range(0, 85, 5))
plt.show()
#Catergories to integers
df = pd.get_dummies(df)
df = df.astype('float64')
df = df.drop(['Title_no'], axis=1)
display(df.head())
headers = df.columns.values
x_train = df
y_train = df_vals.Survived

#Cross validation

from sklearn.model_selection import cross_val_score
models = [XGBClassifier(learning_rate=0.1, max_depth=5)]

for model in models:
    scores = cross_val_score(model, x_train, y_train, cv=10)
    scores.sort()
    accuracy = scores.mean()
    print(scores)
    print(str(accuracy))
display(x_train.head())
#XGBClassifier
model = XGBClassifier(learning_rate=0.1, max_depth=5)
model.fit(x_train, y_train)
df_test_raw = pd.read_csv('../input/test.csv')
print(df_test_raw.info())
display(df_test_raw.head())
df_test = df_test_raw
df_test['Title'] = df_test['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
df_test['Relatives'] = df_test.SibSp + df_test.Parch
df_test['Alone'] = df.Relatives.apply(lambda i: '0' if i>0 else '1')

def quantile(i):
    if i < q1:
        return '0'
    if i >= q1 and i < q2:
        return '1'    
    if i >= q2 and i < q3:
        return '2'
    if i > q3:
        return '3'
df_test['Fare_quantile'] = df.Fare.apply(quantile)

def parent_child(age, p):
    if age < 18:
        return 'child'
    if age >= 20 and p > 0:
        return 'parent'
    else: 
        return 'other'
df_test['parent_child'] = np.vectorize(parent_child)(df_test['Age'], df_test['Parch'])
#choosing valuable data
df_test = df_test[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Title', 'Relatives', 'Alone', 'Fare_quantile', 'parent_child']]
display(df_test.head(10))
#fill null records
df_test.Age.fillna(value=df_test.Age.mean(), inplace=True)
df_test.Embarked.fillna(value=df_test.Embarked.mode(), inplace=True)
df_test.Fare.fillna(value=df_test.Fare.mean(), inplace=True)
display(df_test.head(10))
#Catergories to integers
df_test = pd.get_dummies(df_test)

df_test = df_test[headers]

display(x_train.head())
display(df_test.head())
#XGBClassifier
df_test_raw['Survived'] = model.predict(df_test)
display(df.head(20))
#Prepare output
df=df_test_raw[['PassengerId', 'Survived']]
df = df.astype('int')
#df.to_csv('result.csv', sep=',', index=False)
print(df.head(10))
