import warnings

warnings.filterwarnings('ignore')



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline



import seaborn as sns

sns.set(style="darkgrid")





from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV

from sklearn.metrics import classification_report, confusion_matrix, auc, roc_auc_score, roc_curve, mean_squared_error

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler



from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier

from sklearn.svm import SVC, LinearSVC

from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB



from xgboost import XGBClassifier
df_train = pd.read_csv("/kaggle/input/titanic/train.csv")

df_train.name = "training dataset"

df_test = pd.read_csv("/kaggle/input/titanic/test.csv")

df_test.name = "test dataset"
print(f"No of rows in training dataset - {df_train.shape[0]}")

print(f"No of rows in test dataset - {df_test.shape[0]}")

print(f"No of columns in training dataset - {df_train.shape[1]}")

print(f"No of columns in test dataset - {df_test.shape[1]}")
def display_missing(df):    

    for col in df.columns.tolist():          

        print('{} column missing values: {}'.format(col, df[col].isnull().sum()))

    print('\n') 
for df in [df_train, df_test]:

    print('{}'.format(df.name))

    print('-----------------------------')

    display_missing(df)
df_corr = df_train.corr().abs().unstack().sort_values(kind="quicksort", ascending=False).reset_index()

df_corr.rename(columns={"level_0": "Feature 1", "level_1": "Feature 2", 0: 'Correlation Coefficient'}, inplace=True)

df_corr[df_corr['Feature 1'] == 'Age']
df_train['Age'] = df_train.groupby(['Sex', 'Pclass'])['Age'].apply(lambda x: x.fillna(x.median()))

df_test['Age'] = df_test.groupby(['Sex', 'Pclass'])['Age'].apply(lambda x: x.fillna(x.median()))
df_train['Embarked'].fillna(df_train['Embarked'].mode(dropna=True)[0], inplace=True)

df_test['Embarked'].fillna(df_train['Embarked'].mode(dropna=True)[0], inplace=True)
med_fare = df_test.groupby(['Pclass', 'Parch', 'SibSp']).Fare.median()[3][0][0]

df_test['Fare'] = df_test['Fare'].fillna(med_fare)
df_gb_survived = df_train.groupby('Survived')['Name'].agg({'count'}).reset_index().sort_values(by='count', ascending=False)

df_gb_survived['count_per'] = round(df_gb_survived['count']/len(df_train)*100, 2)



df_gb_survived['Survived'] = df_gb_survived['Survived'].apply(lambda x: 'Not Survived' if x == 0 else 'Survived')



plt.figure(figsize=(15,8))

ax = sns.barplot(x='Survived', y='count', data=df_gb_survived)



for i in ax.patches:

    vl = round((i.get_height()/len(df_train))*100,2)

    ax.annotate('{} %'.format(vl), (i.get_x()+0.4, i.get_height()),

                    ha='center', va='bottom',

                    color= 'black')



plt.title("Distribution of Target variable")

plt.show()
df_gb_sex_survived = df_train.groupby(['Survived','Sex'])['Name'].agg({'count'}).reset_index().sort_values(by='count', ascending=False)

df_gb_sex_survived['Survived'] = df_gb_sex_survived['Survived'].apply(lambda x: 'Not Survived' if x == 0 else 'Survived')

df_gb_sex_survived['Sex'] = df_gb_sex_survived['Sex'].apply(lambda x: 'Female' if x == 0 else 'Male')

#df_gb_sex_survived



plt.figure(figsize=(15,8))

ax = sns.barplot(x='Sex', y='count', hue='Survived', data=df_gb_sex_survived)



for i in ax.patches:

    vl = round((i.get_height()/len(df_train))*100,2)

    ax.annotate('{} %'.format(vl), (i.get_x()+0.2, i.get_height()),

                    ha='center', va='bottom',

                    color= 'black')



plt.title("Distribution of Target variable by Sex")

plt.show()
df_train.head(1)
df_gb_pclass_survived = df_train.groupby(['Survived','Pclass'])['Name'].agg({'count'}).reset_index().sort_values(by='count', ascending=False)

df_gb_pclass_survived



plt.figure(figsize=(15,8))

ax = sns.barplot(x='Pclass', y='count', hue='Survived', data=df_gb_pclass_survived)



for i in ax.patches:

    vl = round((i.get_height()/len(df_train))*100,2)

    ax.annotate('{} %'.format(vl), (i.get_x()+0.2, i.get_height()),

                    ha='center', va='bottom',

                    color= 'black')



plt.title("Distribution of Target variable by Pclass")

plt.show()
df_gb_sibsp_survived = df_train.groupby(['Survived','SibSp'])['Name'].agg({'count'}).reset_index().sort_values(by='count', ascending=False)

df_gb_sibsp_survived



plt.figure(figsize=(15,8))

ax = sns.barplot(x='SibSp', y='count', hue='Survived', data=df_gb_sibsp_survived)



for i in ax.patches:

    vl = round((i.get_height()/len(df_train))*100,2)

    ax.annotate('{} %'.format(vl), (i.get_x()+0.2, i.get_height()),

                    ha='center', va='bottom',

                    color= 'black')



plt.title("Distribution of Target variable by SibSp")

plt.show()
df_gb_parch_survived = df_train.groupby(['Survived','Parch'])['Name'].agg({'count'}).reset_index().sort_values(by='count', ascending=False)

df_gb_parch_survived



plt.figure(figsize=(15,8))

ax = sns.barplot(x='Parch', y='count', hue='Survived', data=df_gb_parch_survived)



for i in ax.patches:

    vl = round((i.get_height()/len(df_train))*100,2)

    ax.annotate('{} %'.format(vl), (i.get_x()+0.2, i.get_height()),

                    ha='center', va='bottom',

                    color= 'black')



plt.title("Distribution of Target variable by Parch")

plt.show()
df_train['Family_Size'] = df_train['SibSp'] + df_train['Parch'] + 1

df_test['Family_Size'] = df_test['SibSp'] + df_test['Parch'] + 1
family_map = {1: 'Alone', 2: 'Small', 3: 'Small', 4: 'Small', 5: 'Medium', 6: 'Medium', 7: 'Large', 8: 'Large', 11: 'Large'}



df_train['Family_Size_Grouped'] = df_train['Family_Size'].map(family_map)

df_test['Family_Size_Grouped'] = df_test['Family_Size'].map(family_map)
# creating new variable Is_Married and Title

for ds in [df_train, df_test]:

    ds['Title'] = ds['Name'].str.split(', ', expand=True)[1].str.split('.', expand=True)[0]

    ds['Is_Married'] = 0

    ds['Is_Married'].loc[ds['Title'] == 'Mrs'] = 1
for ds in [df_train, df_test]:

    ds['Title'] = ds['Title'].replace(['Miss', 'Mrs','Ms', 'Mlle', 'Lady', 'Mme', 'the Countess', 'Dona'], 'Miss/Mrs/Ms')

    ds['Title'] = ds['Title'].replace(['Dr', 'Col', 'Major', 'Jonkheer', 'Capt', 'Sir', 'Don', 'Rev'], 'Dr/Military/Noble/Clergy')
non_numeric_features = ['Embarked', 'Sex', 'Title', 'Family_Size_Grouped', 'Age', 'Fare']



for df in [df_train, df_test]:

    for feature in non_numeric_features:        

        df[feature] = LabelEncoder().fit_transform(df[feature])
cat_features = ['Pclass', 'Sex', 'Embarked', 'Title', 'Family_Size_Grouped']

encoded_features = []



for feature in df_train.columns:

    if feature in cat_features:

        encoded_feat = OneHotEncoder().fit_transform(df_train[feature].values.reshape(-1, 1)).toarray()

        n = df_train[feature].nunique()

        cols = ['{}_{}'.format(feature, n) for n in range(1, n + 1)]

        encoded_df = pd.DataFrame(encoded_feat, columns=cols)

        encoded_df.index = df_train.index

        encoded_features.append(encoded_df)        



df_train = pd.concat([df_train, *encoded_features[:6]], axis=1)



encoded_features = []

for feature in df_test.columns:

    if feature in cat_features:

        encoded_feat = OneHotEncoder().fit_transform(df_test[feature].values.reshape(-1, 1)).toarray()

        n = df_test[feature].nunique()

        cols = ['{}_{}'.format(feature, n) for n in range(1, n + 1)]

        encoded_df = pd.DataFrame(encoded_feat, columns=cols)

        encoded_df.index = df_test.index

        encoded_features.append(encoded_df)        



df_test = pd.concat([df_test, *encoded_features[:6]], axis=1)
drop_cols1 = ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'Cabin', 'Title', 'Ticket']

drop_cols2 = ['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'Cabin', 'Title', 'Ticket']
X = StandardScaler().fit_transform(df_train.drop(columns=drop_cols1))

y = df_train['Survived'].values

X_test = StandardScaler().fit_transform(df_test.drop(columns=drop_cols2))
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
mlist = []

mlist.append(('LogisticRegression', LogisticRegression()))

mlist.append(('Perceptron', Perceptron()))

mlist.append(('SGDClassifier', SGDClassifier()))

mlist.append(('SVC', SVC()))

mlist.append(('LinearSVC', LinearSVC()))

mlist.append(('DecisionTreeClassifier', DecisionTreeClassifier()))

mlist.append(('ExtraTreeClassifier', ExtraTreeClassifier()))

mlist.append(('RandomForestClassifier', RandomForestClassifier()))

mlist.append(('KNeighborsClassifier', KNeighborsClassifier()))

mlist.append(('GaussianNB', GaussianNB()))

mlist.append(('AdaBoostClassifier', AdaBoostClassifier(DecisionTreeClassifier())))

mlist.append(('Xgboost', XGBClassifier()))



df = pd.DataFrame(columns=['name', 'score'])

for name, m in mlist:

    m.fit(X_train, y_train)

    score = m.score(X_val, y_val)

    pred = m.predict(X_val)

    err = mean_squared_error(y_val, pred)

    df1 = pd.DataFrame(data={'name': [name], 'score': [score], 'error': [err]})

    df = pd.concat([df, df1])

    

    

df_models = df.reset_index().sort_values(by='score', ascending=False).reset_index().drop(columns=['index', 'level_0'])
df_models
plt.figure(figsize=(15,8))



ax = sns.barplot(x='name', y='score', data=df_models)



for i in ax.patches:

    ax.annotate('{}'.format(round(i.get_height()*100,2)), (i.get_x()+0.4, i.get_height()),

                    ha='center', va='bottom',

                    color= 'black')



plt.xticks(rotation='90')

plt.xlabel("Model Name")

plt.ylabel('Score')

plt.show()
estimator = XGBClassifier(

    objective= 'binary:logistic',

    nthread=4,

    seed=42

)



parameters = {

    'max_depth': range (2, 10, 1),

    'n_estimators': range(60, 220, 40),

    'learning_rate': [0.1, 0.01, 0.05]

}



grid_search = GridSearchCV(

    estimator=estimator,

    param_grid=parameters,

    scoring = 'roc_auc',

    n_jobs = 10,

    cv = 10,

    verbose=True

)



grid_search.fit(X_train, y_train)
grid_search.best_estimator_
xgb1 = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,

              colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,

              importance_type='gain', interaction_constraints='',

              learning_rate=0.05, max_delta_step=0, max_depth=5,

              min_child_weight=1,  monotone_constraints='()',

              n_estimators=100, n_jobs=4, nthread=4, num_parallel_tree=1,

              random_state=42, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,

              seed=42, subsample=1, tree_method='exact', validate_parameters=1,

              verbosity=None)

xgb1.fit(X_train, y_train)



print(xgb1.score(X_train, y_train), xgb1.score(X_val, y_val))
ypred = xgb1.predict(X_val)

print(classification_report(y_val, ypred))

print(confusion_matrix(y_val, ypred))
ypred = xgb1.predict(X_train)

print(classification_report(y_train, ypred))

print(confusion_matrix(y_train, ypred))
def generate_submission(model):

    result = model.predict(X_test)



    submission_df = pd.DataFrame(columns=['PassengerId', 'Survived'])

    submission_df['PassengerId'] = df_test['PassengerId']

    submission_df['Survived'] = result

    submission_df.to_csv('submissions_rsb_updated.csv', header=True, index=False)

    submission_df.head(10)
generate_submission(xgb1)