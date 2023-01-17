import pandas as pd

full = pd.read_csv("../input/titanic-extended/full.csv")

test = pd.read_csv("../input/titanic-extended/test.csv")

train = pd.read_csv("../input/titanic-extended/train.csv")
full.head(10)

full.describe()
full.info()
full.isna().sum()
full.dropna(inplace=True, subset=['Survived'])
full.isna().sum()
full.info()
full = full[['Survived', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Class']]

full.head()
full.head()
full = pd.get_dummies(full, drop_first=True)

full.head()
full.corr()
full.isna().sum()
age_average = full['Age'].mean()

print(age_average)
full['Age'].fillna(age_average, inplace=True)

full.isna().sum()
full.Class.fillna(method='bfill', inplace=True)
y = full['Survived']

X = full.drop(columns=['Survived'])

X.head()
from sklearn import preprocessing



cols = X.columns

x = X.values

min_max_scaler = preprocessing.MinMaxScaler()

x_scaled = min_max_scaler.fit_transform(x)

X = pd.DataFrame(x_scaled, index=full.index, columns=cols)

X.head()