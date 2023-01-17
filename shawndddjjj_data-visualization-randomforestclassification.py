# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')



from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier



from sklearn.externals.six import StringIO

from IPython.display import Image

from sklearn.tree import export_graphviz

from scipy.cluster.hierarchy import linkage, dendrogram
df = pd.read_csv('/kaggle/input/students-performance-in-exams/StudentsPerformance.csv')

df.columns = ['Gender', 'Race', 'EducationLevel', 'Lunch', 'PreparationCourse',

              'MathScore', 'ReadingScore', 'WritingScore']

df['AvgScore'] = round((df.MathScore + df.ReadingScore + df.WritingScore) / 3, 2)

df.head()
df.isnull().sum()
df.describe().T
df.describe(include = 'O').T
df['EducationLevel'] = pd.Categorical(df.EducationLevel,

                                      categories = ["some high school", "high school", "associate's degree", \

                                                    "some college", "bachelor's degree", "master's degree"],

                                      ordered = True)



print('Lunch Options\n ', df.Lunch.unique())

print('Preparation Courses\n', df.PreparationCourse.unique())

print('Races/Ethinicities\n', df.Race.unique())

print('Education Levels\n', df.EducationLevel.unique())
feature_list = ['Gender', 'Lunch', 'PreparationCourse', 'Race',  'EducationLevel']



for i in feature_list:

    grouped = df[[i, 'MathScore', 'WritingScore', \

                'ReadingScore', 'AvgScore']].groupby(i).mean().sort_values(

                by = 'AvgScore', ascending = False).round(2).reset_index()

    print('---- {} Average Score Summary ----'.format(i))

    display(grouped)
plt.style.use('seaborn')

fig, axes = plt.subplots(2, 3, figsize = (15, 8))



for ax, feature in zip(axes.flatten(), feature_list):

    sns.countplot(df[feature], ax = ax, palette = 'Blues')

    ax.set_xlabel('')

    ax.tick_params(labelrotation = 30)

    ax.set_title('Size on {}'.format(feature))



plt.tight_layout()

plt.show()
#general overview on the dataset

sns.pairplot(df)

plt.show()
# Scores & Education Level seperated by Sex

df_sex = df[['MathScore', 'WritingScore', 'ReadingScore', 'EducationLevel', 'Gender']].groupby(

            ['EducationLevel', 'Gender']).mean().reset_index()



fig, axes = plt.subplots(1, 2, figsize = (15, 4), sharex = 'all', sharey = 'all')

df_sex[df_sex.Gender == 'female'].plot(kind = 'bar', x = 'EducationLevel', stacked = True, ax = axes[0])

axes[0].set_title('Female Total Score by Education Level')

axes[0].tick_params(labelrotation = 30)

axes[0].set_xlabel('')

axes[0].set_ylim(0, 300)



df_sex[df_sex.Gender == 'male'].plot(kind = 'bar', x = 'EducationLevel', stacked = True, ax = axes[1])

axes[1].set_title('Male Total Score by Education Level')

axes[1].tick_params(labelrotation = 30)

axes[1].set_xlabel('')

plt.tight_layout()

plt.show()
# Scores & Test Preparation Course seperated by sex

scores = ['AvgScore', 'MathScore', 'ReadingScore', 'WritingScore']



fig, axes = plt.subplots(1, 4, figsize = (15, 3))

for ax, score in zip(axes.flatten(), scores):

    sns.barplot('PreparationCourse', score, data = df, hue = 'Gender', ax = ax)

    ax.set_ylim(0, 90)

    ax.set_xlabel('')

    ax.set_title('{} for Test Preparation Course'.format(score))

    ax.legend(loc = 2)



plt.tight_layout()

plt.show() 
# Scores & Lunch Options seperated by Sex

fig, axes = plt.subplots(1, 4, figsize = (15, 4))

for ax, score in zip(axes.flatten(), scores):

    sns.boxenplot('Lunch', score, data = df, hue = 'Gender', ax = ax)

    ax.set_ylim(0, 110)

    ax.set_xlabel('')

    ax.set_title('{} for Lunch Options'.format(score))

    ax.legend(loc = 3)



plt.tight_layout()

plt.show() 
# Scores & Race seperated by Sex

fig, axes = plt.subplots(2, 2, figsize = (15, 7))

for ax, score in zip(axes.flatten(), scores):

    sns.boxplot('Race', score, data = df, hue = 'Gender', ax = ax)

    ax.set_ylim(0, 110)

    ax.set_xlabel('')

    ax.set_title('{} for Race/Ethinicity'.format(score))

    ax.legend(loc = 4)



plt.tight_layout()

plt.show() 
# Scores & Education Level seperated by Sex

fig, axes = plt.subplots(2, 2, figsize = (15, 8))

for ax, score in zip(axes.flatten(), scores):

    sns.violinplot('EducationLevel', score, data = df, hue = 'Gender', ax = ax)

    ax.set_xlabel('')

    ax.set_title('{} for Education Level'.format(score))

    ax.legend(loc = 4)



plt.tight_layout()

plt.show()
# getting AvgScore intervals

df['AvgScore_int'] = pd.cut(df.AvgScore, bins = pd.interval_range(0, 100, 10))

feature_list.append('AvgScore_int')



# setting up LabelEncoder

coded_list = []

le = LabelEncoder()



# encoding all variables

for col in feature_list:

    df['{}_le'.format(col)] = le.fit_transform(df['{}'.format(col)])

    coded_list.append('{}_le'.format(col))



print('Encoded Feature List\n', coded_list)

display(df[coded_list].head())
# setting up model & hyperparameters for GridSearch - get the best score

X = df[coded_list].drop('AvgScore_int_le', axis = 1)

y = df['AvgScore_int_le']



hp_params = {'max_depth': range(3, 6, 1),

             'n_estimators': range(150, 350, 50),

             'criterion': ['gini', 'entropy']}



rf_sg = GridSearchCV(estimator = RandomForestClassifier(oob_score = True, random_state = 0), 

                     param_grid = hp_params, cv = 5)

rf_sg.fit(X, y)



print('Best Params: {}'.format(rf_sg.best_params_))
# selecting the best params to find the feature importance

rf_sg.best_estimator_.fit(X, y)

df_features = pd.DataFrame({'Feature': X.columns.values,

                           'Importance': rf_sg.best_estimator_.feature_importances_})



plt.figure(figsize = (10, 3))

sns.barplot(x = 'Importance', y = 'Feature', 

            data = df_features.sort_values('Importance', ascending = False), palette = 'Blues_d')

plt.show()
# randomly selecting one of the forest to visualize

estimator = rf_sg.best_estimator_.estimators_[np.random.randint(0, rf_sg.best_estimator_.n_estimators + 1)]



dot_data = StringIO()

export_graphviz(estimator, out_file = dot_data, filled = True, rounded = True, special_characters = True,

                feature_names = list(X.columns), class_names = [str(i) for i in set(y)])

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  

Image(graph.create_png())