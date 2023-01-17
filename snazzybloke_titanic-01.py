import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pathlib
%matplotlib inline

#csv_dir = '/home/ante/PYTHON/KAGGLE/TITANIC/'
csv_dir = '../input'
df_train = pd.read_csv(csv_dir / pathlib.Path('train.csv'), index_col='PassengerId')
df_Test = pd.read_csv(csv_dir / pathlib.Path('test.csv'), index_col='PassengerId')

df_train.shape, df_Test.shape
df_train.head()
df_Test.head()
df_train.describe()
# df_train.select_dtypes(include=['object']).describe()   # works fine, same as:
# df_train.describe(include=['object'])                   # or, also:
df_train.describe(exclude='number')
df_train.isnull().sum()
df_Test.isnull().sum()
# Looks like the place of embarkment affects the survival rate:
df_train.groupby('Embarked')[['Survived']].mean()
# Most passangers have Cabin Nan (usually no cabin, unless Pclass=1)
df_train.Cabin.describe()
df_Test.Cabin.describe()
# df_train.pivot(index='Pclass', columns='Cabin')['Fare']  # fails, there are duplicates (Pclass, Cabin) entries
# Hence, need to make an aggregation, e.g. mean fare:
df_train.pivot_table(values='Fare', index='Pclass', columns='Cabin')
# number of passanger per cabin:
df_train.pivot_table(values='Name', index='Pclass', columns='Cabin', aggfunc='count')
df_train.pivot_table(index='Sex', columns='Pclass', values='Survived')
# to get pivot_table for (Sex, Age) need to discretize the Age
age_bin = pd.cut(df_train['Age'], bins=[0, 18, 80], labels=['underaged', 'adult'])
df_train.pivot_table(index='Sex', columns=age_bin, values='Survived')
df_train.pivot_table(index=['Sex', age_bin], columns='Pclass', values='Survived')
fare_bin = pd.qcut(x=df_train['Fare'], q=2, labels=['lower50%', 'higher50%'])
df_train.pivot_table('Survived', index=['Sex', age_bin], columns=[fare_bin, 'Pclass'])
df_train.pivot_table(index='Sex', columns='Pclass', aggfunc={'Fare': 'mean', 'Survived': sum})
# Problem: sex not (yet) explicitly correlated to survival:
df_train.corr()['Survived'].sort_values(ascending=False)
# majority 'male' 
df_train.groupby('Sex').count()
# from sklearn.preprocessing import LabelBinarizer
# lb = LabelBinarizer()
# df_train['Is_male'] = lb.fit_transform(df_train['Sex'])
# df_Test['Is_male'] = lb.transform(df_Test['Sex'])
# df_train.groupby('Is_male').count(), df_Test.groupby('Is_male').count()

# The above works but, say, for Logistic Regression maybe it's not ideal to have only 'Is_male'=1 contributing.
# Hence, use OneHotEncoder instead:

y_train = df_train['Survived']
df_train = df_train.drop('Survived', axis=1)
df_train['Training_set'] = True
df_Test['Training_set'] = False
df_FULL = pd.concat([df_train, df_Test])
df_FULL.isnull().sum()
df_FULL[df_FULL.Embarked.isnull()]
df_FULL[df_FULL['Fare'].between(75, 85) & (df_FULL.Pclass==1) & (df_FULL.Sex=='female') & (df_FULL.Cabin.str.startswith('B'))]
# Impossible tp tell where 62 and 830 came from. Impute the most_frequent value:
df_FULL['Embarked']= df_FULL['Embarked'].fillna(value=df_FULL['Embarked'].value_counts().index[0])
df_FULL.loc[[62, 830]]
from sklearn.preprocessing import Imputer
imputer = Imputer(strategy='median')
imputer.fit(df_FULL[['Age', 'Fare']])
df_FULL[['Age', 'Fare']] = imputer.transform(df_FULL[['Age', 'Fare']])
# OneHotEncoder for Sex, Embarked:
df_FULL = df_FULL.join(pd.get_dummies(df_FULL[['Sex', 'Embarked']]))
df_FULL.drop(['Sex','Embarked'], axis=1, inplace=True)
df_FULL.tail()
df_FULL.loc[df_FULL.Cabin.notnull() & df_FULL.Cabin.str.contains('F'), 'Cabin']
print(list(map(lambda x: x[0], df_FULL.Cabin.dropna().tolist())))
deck_list = list(map(lambda x: x[0], df_FULL.Cabin.dropna().tolist()))
# Deck as the initial of the Cabin. If no cabin, use 'X'
crit = df_FULL['Cabin'].isnull()
df_FULL['Deck'] = df_FULL['Cabin'].astype(str).str[0].where(~crit, other='X')
df_FULL.tail()
# some passengers booked more than a single cabin
df_FULL.loc[[28, 76, 89, 129]]
df_FULL = df_FULL.join(pd.get_dummies(df_FULL[['Deck']]))
df_FULL.drop("Deck", axis=1, inplace=True)
df_FULL.tail()
df_train = df_FULL[df_FULL['Training_set'] == True]
df_train.drop('Training_set', axis=1)
df_Test = df_FULL[df_FULL['Training_set'] == False]
df_Test.drop('Training_set', axis=1)
df_train.shape, df_Test.shape
df_train.describe(include='number')
df_train.describe(include='O')
X_train = df_train.select_dtypes('number')
X_Test = df_Test.select_dtypes('number')
y_train.shape, X_train.shape, X_Test.shape
df_tmp = df_train.select_dtypes('number').join(y_train)
df_tmp.corr()['Survived'].sort_values(ascending=False)
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_Pred = logreg.predict(X_Test)
acc_log = round(logreg.score(X_train, y_train) * 100, 2)
acc_log
df_lr_coeff = pd.DataFrame(X_train.columns)
df_lr_coeff.columns = ['Feature']
df_lr_coeff['Correlation'] = pd.Series(logreg.coef_[0])
df_lr_coeff.sort_values(by='Correlation', ascending=False)
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron, SGDClassifier, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import  AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
scoring = []
models= [Perceptron(), SGDClassifier(), GaussianNB(), LogisticRegression(), \
         KNeighborsClassifier(n_neighbors=3), LinearSVC(), SVC(), DecisionTreeClassifier(),\
         RandomForestClassifier(n_estimators=100), AdaBoostClassifier(n_estimators=100),\
         GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=4)]
for model in models:
    model.fit(X_train, y_train)
    y_Pred = model.predict(X_Test)
    scoring.append(model.score(X_train, y_train) * 100)
    
df_scores = pd.DataFrame({"Model": models, "Score": scoring})
df_scores.sort_values(by='Score', ascending=False)
# Estimators not based on the Decision Tree requires standardised/scaled data:
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
scoring = []
models= [Perceptron(), SGDClassifier(), GaussianNB(), LogisticRegression(), \
         KNeighborsClassifier(n_neighbors=3), LinearSVC(), SVC(), DecisionTreeClassifier(),\
         RandomForestClassifier(n_estimators=100), AdaBoostClassifier(n_estimators=100),\
         GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=4)]
for model in models:
    ssmodel = make_pipeline(StandardScaler(), model)
    ssmodel.fit(X_train, y_train)
    y_Pred = ssmodel.predict(X_Test)
    scoring.append(ssmodel.score(X_train, y_train) * 100)
    
df_scores = pd.DataFrame({"Model": models, "Score": scoring})
df_scores.sort_values(by='Score', ascending=False)
from sklearn.model_selection import RandomizedSearchCV
gbs = GradientBoostingClassifier()
grid = {'n_estimators': range(100, 300, 100),\
        'learning_rate': np.linspace(0.1, 1, 10),\
        'max_depth': np.arange(1, 10)}
rs = RandomizedSearchCV(estimator=gbs, param_distributions=grid, scoring='accuracy',\
                        n_iter=50, cv=5, n_jobs=-1)
rs.fit(X_train, y_train)
rs.best_score_, rs.best_params_
# But it looks a bit worse than the setting used above:
model = rs.best_estimator_
model.fit(X_train, y_train)
y_Pred = model.predict(X_Test)
acc = round(model.score(X_train, y_train) * 100, 2)
acc

feat_labels = df_train.select_dtypes(include='number').columns
fimportance = model.feature_importances_
indices = np.argsort(fimportance)[::-1]
for i in range(X_train.shape[1]):
    ii = indices[i]
    print("{:2d}) {:20s} {:.5f}".format(i+1, feat_labels[ii], fimportance[ii]))
# RandomForrestClassifier, 100 trees:
model = models[8]
model.fit(X_train, y_train)
y_Pred = model.predict(X_Test)
acc = round(model.score(X_train, y_train) * 100, 2)
acc
feat_labels = df_train.select_dtypes(include='number').columns
fimportance = model.feature_importances_
indices = np.argsort(fimportance)[::-1]
for i in range(X_train.shape[1]):
    ii = indices[i]
    print("{:2d}) {:20s} {:.5f}".format(i+1, feat_labels[ii], fimportance[ii]))
yT = model.predict(X_Test)
my_1st_submission = pd.DataFrame({'PassengerId': df_Test.index, 'Survived': yT})
my_1st_submission.to_csv('Titanic_RFC_01.csv', index=False)