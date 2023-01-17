# load dependencies

import numpy as np

import pandas as pd



from sklearn.model_selection import GridSearchCV, cross_val_score

from sklearn.model_selection import cross_validate

from sklearn.impute import SimpleImputer

from sklearn.compose import ColumnTransformer

from sklearn.metrics import confusion_matrix

from sklearn.pipeline import make_pipeline

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import Pipeline

from sklearn import svm



import seaborn as sns

import matplotlib.pyplot as plt



from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder

# read the data

titanic_train = pd.read_csv("../input/titanic/train.csv")

titanic_test = pd.read_csv("../input/titanic/test.csv")

titanic_result = pd.read_csv("../input/titanic/gender_submission.csv")
titanic_train.head(10)
print(f'Training dataset: {titanic_train.shape[0]} data points \nTest dataset: {titanic_test.shape[0]} data points')
# features count

features_count = titanic_test.shape[1]

print(f'There are {features_count} columns including class column, thus there are {features_count - 1} features in the dataset.')
# check the data type

titanic_train.dtypes
titanic_train.describe(include='all')
# selection by survival

survived_mask = titanic_train['Survived'] == 1

all_survived = titanic_train[survived_mask]

nobody_survived = titanic_train[~survived_mask]
# find min and max age

min_age = titanic_train['Age'].min(); max_age = titanic_train['Age'].max()

print(f'The youngest person is {min_age}, the oldest is {max_age} years old')
# men and women who survived

survived_men = all_survived[all_survived['Sex'] == 'male']

survived_women = all_survived[all_survived['Sex'] == 'female']

survived_men_age = survived_men[['Sex', 'Age']]

survived_women_age = survived_women[['Sex', 'Age']]

survived_men_age.head()
def get_age_by_sex_count(dataset, age_class, append_negative = False):

    dataset = dataset.dropna()

    dataset['Age'] = dataset['Age'].astype(int)

    ranges = [i.split('-') for i in age_class]

    result = []

    for start, end in ranges:

        range_ =  dataset[(dataset['Age'] >= int(start)) & (dataset['Age'] <= int(end))]

        count = len(range_)

        if append_negative == True:

            count = -count

        result.append(count)

    return result
age_class = ['0-5', '6-10', '11-15', '16-20', '21-25', '26-30', '31-35', '36-40', '41-45', '46-50', '51-55', '56-60', '61-65', '66-70', '75-80', '80-85']

age_sex_count_men = get_age_by_sex_count(survived_men_age, age_class, append_negative = True)

age_sex_count_women = get_age_by_sex_count(survived_women_age, age_class)


age_sex_data = pd.DataFrame({'Age': age_class, 

                    'Male': age_sex_count_men, 

                    'Female': age_sex_count_women})



bar_plot = sns.barplot(x='Male', y='Age', data=age_sex_data, order=age_class, lw=0)



bar_plot = sns.barplot(x='Female', y='Age', data=age_sex_data, order=age_class, lw=0)



bar_plot.set(xlabel="Number of People", ylabel="Age-Group", title = "Population Pyramid of Suvived on Titanic")



# subset by port 

by_port = titanic_train[['Survived', 'Embarked']]

# port Cherbourg

port_c = by_port[by_port['Embarked'] == 'C']

# port Queenstown

port_q = by_port[by_port['Embarked'] == 'Q']

# port Southampton

port_s = by_port[by_port['Embarked'] == 'S']



loaded_by_port = [len(port_c), len(port_q), len(port_s)]

survived_by_port = [len(port_c[port_c['Survived'] == 1]), len(port_q[port_q['Survived'] == 1]), len(port_s[port_s['Survived'] == 1])]

loaded_survived_by_port = [loaded_by_port, survived_by_port]
# bar plot initialization

X = np.arange(3)

fig = plt.figure(figsize=(10, 10))

ax = fig.add_axes([0,0,1,1])

ax.set_title('Survival by port Embarked')

labels = ['Cherbourg', 'Queenstown', 'Southampton']

x = np.arange(len(labels))  # the label locations

ax.bar(X + 0.00, loaded_survived_by_port[0], color = 'b', width = 0.25)

ax.bar(X + 0.25, loaded_survived_by_port[1], color = 'g', width = 0.25)

ax.set_ylabel('Number of people')

ax.set_xticks(x)

ax.set_xticklabels(labels)

ax.legend(labels=['Total', 'Survived'])
# select a subset of data by class and ticket price

by_class = titanic_train[['Survived', 'Pclass', 'Fare']]
# prepare the data

first_class = by_class[by_class['Pclass'] == 1]

second_class = by_class[by_class['Pclass'] == 2]

third_class = by_class[by_class['Pclass'] == 3]



total_by_class = [len(first_class), len(second_class), len(third_class)]

survived_by_class = [len(first_class[first_class['Survived'] == 1]), len(second_class[second_class['Survived'] == 1]), len(third_class[third_class['Survived'] == 1])]



total_survived_by_class = [total_by_class, survived_by_class]
# plot 

X_class = np.arange(3)

fig_class = plt.figure(figsize=(10, 10))

ax_class = fig_class.add_axes([0,0,1,1])

ax_class.set_title('Survival by Class')

labels_class = ['1', '2', '3']

x_class = np.arange(len(labels_class))  # the label locations

ax_class.bar(X + 0.00, total_survived_by_class[0], color = 'b', width = 0.25)

ax_class.bar(X + 0.25, total_survived_by_class[1], color = 'g', width = 0.25)

ax_class.set_ylabel('Number of people')

ax_class.set_xticks(x_class)

ax_class.set_xticklabels(labels_class)

ax_class.legend(labels=['Total', 'Survived'])
# prepare selection

survived_fare_mask = by_class['Survived'] == 1

survived_fare = by_class[survived_fare_mask]

no_survival_fare = by_class[~survived_fare_mask]

survived_fare = survived_fare['Fare']
# plot

ax = sns.scatterplot(x='Pclass', y='Fare',

                hue='Survived', data=by_class, )

ax.set_title('Distribution of fare among classes')
first_class_fare_range = [first_class['Fare'].min(), first_class['Fare'].max()]

print(f'The lowest ticket price for first class is ${first_class_fare_range[0]}, the highest is ${first_class_fare_range[1]}')
all_survived_no_cabin = all_survived[all_survived['Cabin'].isnull()]['Fare'].count()

all_survived_yes_cabin = all_survived[all_survived['Cabin'].notnull()]['Fare'].count()

nobody_survived_no_cabin = nobody_survived[nobody_survived['Cabin'].isnull()]['Fare'].count()

nobody_survived_yes_cabin = nobody_survived[nobody_survived['Cabin'].notnull()]['Fare'].count()

df = pd.DataFrame({'Cabin':[all_survived_yes_cabin, nobody_survived_yes_cabin], 'No cabin':[all_survived_no_cabin, nobody_survived_no_cabin]}, index=['Survived', 'Did not survive'] )

 

# make the multiple plot

df.plot(kind='pie', subplots=True, figsize=(16,8))

# correlations

titanic_corr = titanic_train.corr()

sns.heatmap(titanic_corr, annot=True)

plt.show()
nulls = pd.concat([titanic_train.isnull().sum()], axis=1)

nulls[nulls.sum(axis=1) > 0]
titanic_train.info()
titanic_train.describe(include='all')
# creating preprocessing pipeline for numerical and categorical values

numeric_features = ['Age', 'Fare']

numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='mean')), 

                                     ('scaler', StandardScaler())])

categorical_features = ['Embarked', 'Pclass', 'Sex']

categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),

                                         ('onehotencoder', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features),

                                               ('cat', categorical_transformer, categorical_features)])
# select labels

X = titanic_train.drop('Survived', axis=1)

y = titanic_train['Survived']
# random forest

random_forest_model = make_pipeline(preprocessor, RandomForestClassifier())

param_grid = {

    'randomforestclassifier__n_estimators': [10, 100, 1000]

    }

grid_rf_clf = GridSearchCV(random_forest_model, param_grid, cv=8)

grid_rf_clf.fit(X, y);

grid_rf_clf.best_score_
# svm

svm_model = make_pipeline(preprocessor, svm.SVC())

param_grid = {

    'svc__C': [0.01, 0.1, 1.0],

    'svc__gamma': [0.001, 0.01, 0.1, 1],

    'svc__kernel': ['linear', 'poly', 'rbf']

    }

grid_svm_clf = GridSearchCV(svm_model, param_grid, cv=8)

grid_svm_clf.fit(X, y);

grid_svm_clf.best_score_
# decision tree

dt_model = make_pipeline(preprocessor, DecisionTreeClassifier())

dt_scores = cross_val_score(dt_model, X, y.values.ravel(), cv=8, scoring='accuracy')

dt_scores.max()
dt_model.fit(X,y)

predictions = dt_model.predict(titanic_test)
models = pd.DataFrame({

    'Model': ['RandomForestClassifier', 'SVC', 'DecisionTreeClassifier'],

    'Score': [grid_rf_clf.best_score_, grid_svm_clf.best_score_, dt_scores.max()]})

models.sort_values(by='Score', ascending=False)
predictions
submission = pd.DataFrame({

        "PassengerId": titanic_test["PassengerId"],

        "Survived": predictions

    })

submission.head(20)
# save to file

filename = 'titanic_predictions_1.csv'



submission.to_csv(filename,index=False)