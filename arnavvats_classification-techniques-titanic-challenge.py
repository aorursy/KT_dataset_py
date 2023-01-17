import numpy as np

import pandas as pd

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

color = sns.color_palette()

sns.set_style('darkgrid')

df_test = pd.read_csv('../input/test.csv')

df_train = pd.read_csv('../input/train.csv')

n = df_train.shape[0]

df_train.shape, df_test.shape
df_train.head()
x = pd.concat([df_train, df_test], ignore_index = True, sort = False)

print(x.shape)

x.isnull().sum()
sns.catplot(x = 'Survived', y = 'PassengerId', data = x[:n])
x.drop('PassengerId', axis = 1, inplace = True)
sns.countplot(x = 'Pclass', hue = 'Survived', data = x[:n])
x['Pclass'] = x['Pclass'].astype('category')
plt.rcParams['figure.figsize'] = [10,5]

sns.countplot(x = 'Sex', hue = 'Survived', data = x[:n])
x['Sex'] = x['Sex'].astype('category')

x['Sex'].dtype
plt.rcParams['figure.figsize'] = [20,5]

fig, ax = plt.subplots(1,3)

df_tmp = x[:n]

tmp = df_tmp[df_tmp['Age'].isnull() == False]['Age']

ax[0].set_title('{} people with known age'.format(tmp.shape[0]))

sns.distplot(tmp, ax = ax[0])



tmp = df_tmp[(df_tmp['Age'].isnull() == False) & (df_tmp['Survived'] == 1)]['Age']

ax[1].set_title('{} people with known age who survived'.format(tmp.shape[0]))

sns.distplot(tmp, ax = ax[1])



tmp = df_tmp[(df_tmp['Age'].isnull() == False) & (df_tmp['Survived'] == 0)]['Age']

ax[2].set_title('{} people with known age who did not survive'.format(tmp.shape[0]))

sns.distplot(tmp, ax = ax[2])

plt.show()

title_list=['Mrs', 'Mr', 'Master', 'Miss', 'Major', 'Rev',

                    'Dr', 'Ms', 'Mlle','Col', 'Capt', 'Mme', 'Countess',

                    'Don', 'Jonkheer']

def get_title(name):

    title_vals = []

    for title in title_list:

        if name.find(title) != -1:

            return title

    return 'None'



titles = x[x['Age'].isnull()]['Name'].map(get_title)

set(titles)

title_series = x['Name'].map(get_title)

def set_age(row):

    title = get_title(row['Name'])

    if str(row['Age']) == 'nan':

        median_age = x[title_series == title]['Age'].median()

        row['Age'] = median_age

    return row



x = x.apply(set_age, axis = 1)

x['Age'].isnull().sum()

x['Sex'] = x['Sex'].astype('category')

x['Pclass'] = x['Pclass'].astype('category')
plt.rcParams['figure.figsize'] = [10,5]

sns.countplot(x = 'SibSp', hue = 'Survived', data = x[:n])
tmp = x[:n][x[:n]['SibSp'] == 0]

sns.countplot(x = 'Sex', hue = 'Survived', data = tmp)

child_indices = (x[:n]['Age'] <= 15) & (x[:n]['Parch'] == 0)

x[:n][child_indices & (x[:n]['Survived']== 1)].shape[0],  x[:n][child_indices & (x[:n]['Survived'] == 0)].shape[0]
child_indices_x = (x['Age'] <= 15) & (x['Parch'] == 0)

x.loc[child_indices_x , 'Parch'] = 1
sns.countplot(x = 'Parch', hue = 'Survived', data = x[:n])
tmp = x[:n][x[:n]['Parch'] == 0]

sns.countplot(x = 'Sex', hue = 'Survived', data = tmp)
def get_ticket_type(ticket):

    ticket_segmented = ticket.split(' ')

    if len(ticket_segmented) == 1:

        return 'None'

    else:

        return ticket_segmented[0]

x['Ticket_Type'] = x['Ticket'].map(get_ticket_type)
plt.rcParams['figure.figsize'] = [10,20]

sns.countplot(y = 'Ticket_Type', hue = 'Survived', data = x[:n])
x.drop('Ticket', axis = 1, inplace = True)

x['Ticket_Type'] = x['Ticket_Type'].astype('category')
plt.rcParams['figure.figsize'] = [20,5]

fig, ax = plt.subplots(1,3)

tmp = x[:n]['Fare']

ax[0].set_title('Fare distribution for all people')

sns.distplot(tmp, ax = ax[0])



tmp = x[:n][x[:n]['Survived'] == 1]['Fare']

ax[1].set_title('Fare distribution of survivors')

sns.distplot(tmp, ax = ax[1])



tmp = x[:n][x[:n]['Survived'] == 0]['Fare']

ax[2].set_title('Fare distribution of non-survivors')

sns.distplot(tmp, ax = ax[2])

plt.show()
x[x['Fare'].isnull()]
x['Fare'] = x['Fare'].fillna(0)
from math import isnan

def get_cabin_type(cabin_no):

        if type(cabin_no) == float and isnan(cabin_no):

            return 'None'

        else:

            return cabin_no[0]

x['Cabin_Type'] = x['Cabin'].map(get_cabin_type)
set(x['Cabin_Type'])
sns.countplot(x = 'Cabin_Type', hue = 'Survived', data = x[:n])
x.drop('Cabin', axis = 1, inplace = True)

x['Cabin_Type'] = x['Cabin_Type'].astype('category')
sns.countplot(x = 'Embarked', hue = 'Survived', data = x[:n])
indices = np.array(x[x['Embarked'].isnull()].index)

x.loc[indices]
x['Embarked'] = x['Embarked'].fillna('S').astype('category')
x['Title'] = x['Name'].map(get_title)
plt.rcParams['figure.figsize'] = [20,8]

sns.countplot(x = 'Title', hue = 'Survived', data = x[:n])
x.drop('Name', axis = 1, inplace = True)

x['Title'] = x['Title'].astype('category')
x['Family_Size'] = x['Parch'] + x['SibSp'] + 1

sns.countplot(x = 'Family_Size', hue = 'Survived', data = x[:n])
x[x['Family_Size'] > 4][['Fare', 'Family_Size', 'Cabin_Type']].head(10)
x['Fare_Per_Person'] = x['Fare'] / x['Family_Size']
x['Age_Class'] = x['Age'] * x['Pclass'].astype('int')
def get_age_group(age):

    if age >=65:

        return 'Very old'

    elif age >= 45:

        return 'Old'

    elif age >= 16:

        return 'Adult'

    else:

        return 'Child'

x['Age_Group'] = x['Age'].map(get_age_group)

x['Age_Group'] = x['Age_Group'].astype('category')

x.dtypes
x_raw = x

x = pd.get_dummies(x)

print('No. of columns: {}'.format(x.columns.shape[0]))

x.columns
y_train_cv = x[:n]['Survived']

x_train_cv = x[:n].drop('Survived', axis = 1)

x_test = x[n:].drop('Survived', axis = 1)
from sklearn.model_selection import train_test_split

x_train, x_cv, y_train, y_cv = train_test_split(x_train_cv, y_train_cv, test_size = 0.2 , random_state = 42)

x_train.shape, x_cv.shape
from sklearn.ensemble import RandomForestClassifier



model = RandomForestClassifier(n_estimators = 300, max_depth = 10,random_state = 0, max_features = 0.7,  n_jobs = -1)

model.fit(x_train, y_train)

model.score(x_cv, y_cv)
plt.rcParams['figure.figsize'] = [20,30]

fi_df = pd.DataFrame({'Feature': x_train.columns, 'Importance': model.feature_importances_})

fi_df = fi_df[fi_df['Importance'] > 0]

print(fi_df.shape)

sns.barplot(y = 'Feature', x = 'Importance', data = fi_df)
x_train_, x_cv_ = x_train[fi_df['Feature']], x_cv[fi_df['Feature']]

model = RandomForestClassifier(n_estimators = 300, max_depth = 10,random_state = 0, max_features = 0.7,  n_jobs = -1)

model.fit(x_train_, y_train)

model.score(x_cv_, y_cv)
corr = x_train_.corr()

sns.heatmap(corr, vmax = 0.9, square = True)
def get_redundant_pairs(df):

    pairs_to_drop = set()

    cols = df.columns

    for i in range(0, df.shape[1]):

        for j in range(0, i+1):

            pairs_to_drop.add((cols[i], cols[j]))

    return pairs_to_drop



def get_top_abs_correlations(df, c=0.8):

    au_corr = df.corr().abs().unstack()

    labels_to_drop = get_redundant_pairs(df)

    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)

    return au_corr[au_corr > c]

print(get_top_abs_correlations(corr, c = 0.87))
to_drop = ['Sex_female', 'Sex_male', 'SibSp', 'Family_Size', 'Pclass_1',

           'Cabin_Type_None','Fare','Fare_Per_Person','Ticket_Type_WE/P', 

           'Title_Capt', 'Title_Mr', 'Sex_male', 'Parch', 'Title_Master', 'Age_Group_Child']

for feature in to_drop:

    x_train_2 = x_train_.drop(feature, axis = 1)

    x_cv_2 = x_cv_.drop(feature, axis = 1)

    model = RandomForestClassifier(n_estimators = 300, max_depth = 10,random_state = 0, max_features = 0.7,  n_jobs = -1)

    model.fit(x_train_2, y_train)

    print('Feature: {}, score: {}'.format(feature,model.score(x_cv_2, y_cv)))
to_drop = ['Parch']

x_train_2 = x_train_.drop(to_drop, axis = 1)

x_cv_2 = x_cv_.drop(to_drop, axis = 1)

model = RandomForestClassifier(n_estimators = 300, max_depth = 10,random_state = 0, max_features = 0.7,  n_jobs = -1)

model.fit(x_train_2, y_train)

print('Feature: {}, score: {}'.format(feature,model.score(x_cv_2, y_cv)))

fi_df = pd.DataFrame({'Feature': x_train_2.columns, 'Importance': model.feature_importances_})

fi_df = fi_df[fi_df['Importance'] > 0.002]

rf_model = RandomForestClassifier(n_estimators = 300, max_depth = 10,random_state = 0, max_features = 0.7, bootstrap = True, n_jobs = -1)

x_train = x_train_2[fi_df['Feature']]

x_cv = x_cv_2[fi_df['Feature']]

rf_model.fit(x_train, y_train)

rf_model.score(x_cv, y_cv)
from sklearn.ensemble import ExtraTreesClassifier



et_model = ExtraTreesClassifier(n_estimators = 500, max_depth = 10, bootstrap = True, n_jobs = -1, random_state = 0)

et_model.fit(x_train, y_train)

et_model.score(x_cv, y_cv)
from sklearn.ensemble import AdaBoostClassifier

ada_model = AdaBoostClassifier(n_estimators = 800, learning_rate = 1.0, random_state = 0)

ada_model.fit(x_train, y_train)

ada_model.score(x_cv, y_cv)
from sklearn.ensemble import GradientBoostingClassifier

gb_model = GradientBoostingClassifier(n_estimators = 600, max_depth = 15, learning_rate = 1.0, tol = 0.001, random_state = 0)

gb_model.fit(x_train, y_train)

gb_model.score(x_cv, y_cv)
from sklearn import svm

sv_model = svm.SVC(random_state = 0, gamma = 'scale', probability=True)

sv_model.fit(x_train, y_train)

sv_model.score(x_cv, y_cv)
from sklearn.neighbors import KNeighborsClassifier

knn_model = KNeighborsClassifier(n_neighbors = 10)

knn_model.fit(x_train, y_train)

knn_model.score(x_cv, y_cv)
from sklearn.naive_bayes import GaussianNB

nb_model = GaussianNB()

nb_model.fit(x_train, y_train)

nb_model.score(x_cv, y_cv)
from sklearn.ensemble import VotingClassifier

final_model = VotingClassifier(estimators = [('rf', rf_model), ('ada', ada_model), ('gb', gb_model),

                                            ('svm', sv_model), ('knn', knn_model), ('nb', nb_model)],

                              voting = 'soft',

                              weights = [1.,1.,1.2, 0.6,0.7, 0.5])

final_model.fit(x_train, y_train)

final_model.score(x_cv, y_cv)
x = pd.concat([x_train, x_cv], ignore_index = True)

y = np.concatenate([y_train, y_cv])

final_model.fit(x, y)

final_model.score(x, y)
features = x.columns

x_test = x_test[features]
predictions = final_model.predict(x_test).astype('int')

submission_df = pd.DataFrame({

    'PassengerId': np.arange(892, 1310),

    'Survived': predictions

})
from IPython.display import HTML

import base64





def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  

    csv = df.to_csv(index = False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)



df = pd.DataFrame(np.random.randn(50, 4), columns=list('ABCD'))



create_download_link(submission_df)
submission_df.to_csv('./outputs.csv', index = False)