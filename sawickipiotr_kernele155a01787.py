# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

sns.set(style="whitegrid")



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Load data

test_data = pd.read_csv('/kaggle/input/titanic/test.csv')

train_dataset = pd.read_csv('/kaggle/input/titanic/train.csv')
train_dataset.head()
X_train = train_dataset.drop(columns='Survived')

X_test = test_data

y_train = train_dataset.Survived
print('Train dataset contain {} records'.format(X_train.shape[0]))

print('Test dataset contain {} records'.format(X_test.shape[0]))
# check does NULL are in X_train

X_train.isnull().sum()
# check does NULL are in X_train

X_test.isnull().sum()
# Train dataset

print('Train dataset')

column = 'Age'

print('Train dataset contains {} NULL objects out of {} in {} column ({:.2f} %)'.format(X_train.isnull().sum()[column], X_train.shape[0], column, 100*X_train.isnull().sum()[column]/X_train.shape[0]))

column = 'Fare'

print('Train dataset contains {} NULL objects out of {} in {} column ({:.2f} %)'.format(X_train.isnull().sum()[column], X_train.shape[0], column, 100*X_train.isnull().sum()[column]/X_train.shape[0]))

column = 'Cabin'

print('Train dataset contains {} NULL objects out of {} in {} column ({:.2f} %)'.format(X_train.isnull().sum()[column], X_train.shape[0], column, 100*X_train.isnull().sum()[column]/X_train.shape[0]))

column = 'Embarked'

print('Train dataset contains {} NULL objects out of {} in {} column ({:.2f} %)'.format(X_train.isnull().sum()[column], X_train.shape[0], column, 100*X_train.isnull().sum()[column]/X_train.shape[0]))



print('\nTest dataset')

column = 'Age'

print('Test dataset contains {} NULL objects out of {} in {} column ({:.2f} %)'.format(X_test.isnull().sum()[column], X_test.shape[0], column, 100*X_test.isnull().sum()[column]/X_test.shape[0]))

column = 'Fare'

print('Test dataset contains {} NULL objects out of {} in {} column ({:.2f} %)'.format(X_test.isnull().sum()[column], X_test.shape[0], column, 100*X_test.isnull().sum()[column]/X_test.shape[0]))

column = 'Cabin'

print('Test dataset contains {} NULL objects out of {} in {} column ({:.2f} %)'.format(X_test.isnull().sum()[column], X_test.shape[0], column, 100*X_test.isnull().sum()[column]/X_test.shape[0]))

column = 'Embarked'

print('Test dataset contains {} NULL objects out of {} in {} column ({:.2f} %)'.format(X_test.isnull().sum()[column], X_test.shape[0], column, 100*X_test.isnull().sum()[column]/X_test.shape[0]))
X_train.drop(columns='Cabin', inplace=True)

X_test.drop(columns='Cabin', inplace=True)
# ax = sns.countplot(x="Fare", hue='Pclass', data=X_test)

sns.distplot(X_train.Fare[X_train.Pclass == 1], color="skyblue", label='train 1st class', hist=False)

sns.distplot(X_train.Fare[X_train.Pclass == 2], color="red", label='train 2nd class', hist=False)

sns.distplot(X_train.Fare[X_train.Pclass == 3], color="orange", label='train 3rd class', hist=False)

sns.distplot(X_test.Fare[X_test.Pclass == 1], color="blue", label='test 1st class', hist=False)

sns.distplot(X_test.Fare[X_test.Pclass == 2], color="maroon", label='test 2nd class', hist=False)

sns.distplot(X_test.Fare[X_test.Pclass == 3], color="darkorange", label='test 3rd class', hist=False)



plt.legend()



plt.xlim(0, 200)

plt.ylim(0, 0.15);
X_test[X_test.Fare.isnull()]
age_filter = X_train.Age > 50

pclass_filter = X_train.Pclass == 3



age_and_pclass_filter_train = age_filter & pclass_filter

X_train[age_and_pclass_filter_train]['Fare'].describe()
age_filter = X_test.Age > 50

pclass_filter = X_test.Pclass == 3

# X_train[ | ]['Fare'].describe()

age_and_pclass_filter_test = age_filter & pclass_filter
X_test.set_value(X_test[age_and_pclass_filter_test].index.values, 'Fare',X_train[age_and_pclass_filter_train]['Fare'].mean());
X_test.isnull().sum()
X_train.isnull().sum()
X_train[X_train.Embarked.isnull()]
sns.countplot(x="Embarked",data=X_train);
sns.countplot(x="Embarked",hue = 'Pclass', data=X_train[X_train.Sex == 'female']);
sex_filter_train = X_train.Sex == 'female'

S_filter_train = X_train.Embarked == 'S'

C_filter_train = X_train.Embarked == 'C'

first_class_filter_train = X_train.Pclass == 1



sex_filter_test = X_test.Sex == 'female'

S_filter_test = X_test.Embarked == 'S'

C_filter_test = X_test.Embarked == 'C'

first_class_filter_test = X_test.Pclass == 1



S_frames = [X_train.Age[sex_filter_train & S_filter_train & first_class_filter_train], X_test.Age[sex_filter_test & C_filter_test & first_class_filter_test]]

S = pd.concat(S_frames)



C_frames = [X_train.Age[sex_filter_train & C_filter_train & first_class_filter_train], X_test.Age[sex_filter_test & C_filter_test & first_class_filter_test]]

C = pd.concat(C_frames)



ax = sns.distplot(S, color="skyblue", label='Embarked == S', hist=False)

sns.distplot(C, color="maroon", label='Embarked == C', hist=False)



plt.axvline(38) # Age of first woman

plt.axvline(62);  # Age of second woman



S_distribution = ax.get_lines()[0].get_data()

C_distribution = ax.get_lines()[1].get_data()



S_dens_for_first_woman = S_distribution[1][np.logical_and(S_distribution[0] >= 38, S_distribution[0] < 39)].mean()

S_dens_for_second_woman = S_distribution[1][np.logical_and(S_distribution[0] >= 62, S_distribution[0] < 63)].mean()



C_dens_for_first_woman = C_distribution[1][np.logical_and(C_distribution[0] >= 38, C_distribution[0] < 39)].mean()

C_dens_for_second_woman = C_distribution[1][np.logical_and(C_distribution[0] >= 62, C_distribution[0] < 63)].mean()
# Probability of port of embarkation based on all women from first class

prob_S_embark = len(S)/(len(S) + len(C))

prob_C_embark = len(C)/(len(S) + len(C))



# Probability of port of embarkation times distribution for each port



print('First, younger woman:')

print('Embarked == S, {:.4f}'.format(prob_S_embark * S_dens_for_first_woman))

print('Embarked == C, {:.4f}'.format(prob_C_embark * C_dens_for_first_woman))



print('First, older woman:')

print('Embarked == S, {:.4f}'.format(prob_S_embark * S_dens_for_second_woman))

print('Embarked == C, {:.4f}'.format(prob_C_embark * C_dens_for_second_woman))
X_train.set_value(X_train[X_train.Embarked.isnull()].index.values, 'Embarked', 'S');
X_train.head(1)
to_replace_dict = {'.*Miss\..*' : 'Miss',

                  '.*Mr\..*'    : 'Mr',

                  '.*Master\..*': 'Master',

                  '.*Mrs\..*'   : 'Mrs',

                  '.*Ms\..*'    : 'Ms',

                  '.*Mme\..*'   : 'Ms',

                  '.*Rev\..*'   : 'Rev',

                  '.*Dr\..*'    : 'Dr',

                  '.*Major\..*' : 'Major',

                  '.*Col\..*'   : 'Col',

                  '.*Capt\..*'  : 'Capt',

                  '.*Jonkheer\..*': 'Jonkheer',

                  '.*Countess\..*': 'Ms',

                  '.*Mlle\..*': 'Miss',

                  '.*Don\..*' : 'Don',

                  '.*Dona\..*': 'Dona',

                  '.*Lady.*'  : 'Mrs',

                  '.*Sir.*'   : 'Mr'}



X_train['Honorific'] = X_train['Name']

X_test['Honorific'] = X_test['Name']



X_train['Honorific'].replace(to_replace=to_replace_dict, regex=True, inplace=True)

X_test['Honorific'].replace(to_replace=to_replace_dict, regex=True, inplace=True)
set(X_train.Honorific)
frames = [X_train, X_test]

X_df = pd.concat(frames)
for val in set(to_replace_dict.values()):

    df = X_df[X_df['Honorific'] == val]

    null_obj = df.Age.isnull().sum()

    if null_obj > 0:

        print('{:9s}: {}/{} are NULL'.format(val, null_obj, len(df)))
# Following groups have missed age:

honorific_groups = ['Dr', 'Mr', 'Miss', 'Ms', 'Mrs','Master']
fig, axs = plt.subplots(ncols=3, nrows=2)

k=0

for val in honorific_groups:

    df = X_df[X_df['Honorific'] == val]

    for i in range(1,4):

        df1 = df[df['Pclass']==i]

        sns.distplot(df1['Age'][df1['Age'].notnull()], label=i, hist=True, ax=axs[k//3][k%3]).set_title(val)

#     sns.distplot(df['Age'][df['Age'].notnull()], label=val, hist=True, ax=axs[k//3][k%3]).set_title(val)

    k+=1

plt.tight_layout()
from sklearn.impute import SimpleImputer
imp_median = SimpleImputer(missing_values=np.nan, strategy='mean')



for group in ['Mr', 'Miss', 'Mrs','Master']:

    for Pclass in range(1,4):

        group_filter = X_df['Honorific'] == group

        Pclass_filter = X_df['Pclass'] == Pclass



        X_df.loc[group_filter & Pclass_filter, 'Age'] = imp_median.fit_transform(X_df[group_filter & Pclass_filter][['Age']])



for group in ['Dr', 'Ms']:

    group_filter = X_df['Honorific'] == group



    X_df.loc[group_filter, 'Age'] = imp_median.fit_transform(X_df[group_filter][['Age']])
for val in set(to_replace_dict.values()):

    df = X_df[X_df['Honorific'] == val]

    null_obj = df.Age.isnull().sum()

    if null_obj > 0:

        print('{:9s}: {}/{} are NULL'.format(val, null_obj, len(df)))
X_df.isnull().sum()
X_train = X_df.loc[X_df['PassengerId']<=891]

X_test =  X_df.loc[X_df['PassengerId']>891]



X_train.drop(columns=['Name', 'Ticket', 'Honorific'], inplace=True)

X_test.drop(columns=['Name', 'Ticket', 'Honorific'], inplace=True)



X_train = pd.get_dummies(X_train,drop_first=True)

X_test = pd.get_dummies(X_test,drop_first=True)

X_test.head()
X = X_train.drop(columns=['PassengerId']).copy()

y = y_train.copy()

X_submission = X_test.copy()
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
from sklearn import tree

from sklearn.tree.export import export_text

clf = tree.DecisionTreeClassifier()

clf.fit(X_train, y_train)



from sklearn.metrics import accuracy_score







res = abs(clf.predict(X_train) - y_train)

print(res.sum()/len(res))



res = abs(clf.predict(X_test) - y_test)

print(res.sum()/len(res))
from sklearn.ensemble import RandomForestClassifier

for i in range(1, 100):

    print('\n', i)

    random_forest = RandomForestClassifier(n_estimators=int(i))

    random_forest.fit(X_train, y_train)

    y_pred = random_forest.predict(X_test)

    random_forest.score(X_train, y_train)

    # acc_random_forest = round(random_forest.score(X_train, y_train) * 100, 2)

    # print(acc_random_forest)



    # acc_random_forest = round(random_forest.score(X_test, y_pred) * 100, 2)

    # print(acc_random_forest)



    res = abs(random_forest.predict(X_train) - y_train)

    print(res.sum()/len(res))



    res = abs(random_forest.predict(X_test) - y_test)

    print(res.sum()/len(res))
from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression()

logmodel.fit(X_train,y_train)



res = abs(logmodel.predict(X_train) - y_train)

print(res.sum()/len(res))



res = abs(logmodel.predict(X_test) - y_test)

print(res.sum()/len(res))
from sklearn.neural_network import MLPClassifier

min_train = 1

min_test = 1

for k_1 in range(2, 8):

    for k_2 in range(2, 8):

        for k_3 in range(2, 8):

            clf = MLPClassifier(solver='lbfgs', alpha=1e-6, hidden_layer_sizes=(k_1, k_2, k_3), random_state=1)



            clf.fit(X_train, y_train)

            

            res = abs(clf.predict(X_train) - y_train)

            if min_train > res.sum()/len(res):

                print('train')

                print(res.sum()/len(res))

                print(k_1, k_2, k_3)

                print('\n')

                min_train = res.sum()/len(res)

                

            res = abs(clf.predict(X_test) - y_test)

            if min_test > res.sum()/len(res):

                print('test')

                print(res.sum()/len(res))

                print(k_1, k_2, k_3)

                print('\n')

                min_test = res.sum()/len(res)
clf = MLPClassifier(solver='lbfgs', alpha=1e-6, hidden_layer_sizes=(6, 6, 3), random_state=1)



clf.fit(X_train, y_train)
X_submission.head()
output = pd.DataFrame({'PassengerId': X_submission['PassengerId'],

                      'Survived': random_forest.predict(X_submission.drop(columns=['PassengerId']))})

output.to_csv('submission.csv', index=False)