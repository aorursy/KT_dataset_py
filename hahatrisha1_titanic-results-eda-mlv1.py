# Data Analysis and Wrangling

import pandas as pd # data processing

import numpy as np # linear algebra

import random # generate pseudo-random numbers



# Mathematical Functions

import math



# Visualization

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



# ignore warnings

import warnings

warnings.filterwarnings('ignore')
# Import Data

raw = pd.read_csv("/kaggle/input/titanic/train.csv")

validation = pd.read_csv("/kaggle/input/titanic/test.csv")
print("Shape of the raw dataset:", raw.shape)

raw.head(3)
print("Shape of the validation dataset:", validation.shape)

validation.head(3)
# Features in the Dataset

raw.columns.values
raw.info()
raw.describe()
raw.dtypes
raw.columns.values
# Distribution of categorical variables: Sex, Pclass, Embarked, SibSp, Parch

fig, ax = plt.subplots(2,3, figsize=(15,6))

fig.subplots_adjust(wspace=0.3, hspace=0.5)

sns.countplot(y=raw['Survived'], ax=ax[0,0], palette='Spectral').set_title('Survived or Not')

sns.countplot(y=raw['Sex'], ax=ax[0,1], palette='Spectral').set_title('Passengers Sex Distribution')

sns.countplot(y=raw['Pclass'], ax=ax[0,2], palette='Spectral').set_title('Passengers Class Distribution')

sns.countplot(y=raw['SibSp'], ax=ax[1,0], palette='Spectral').set_title('Siblings/Spouses Distribution')

sns.countplot(y=raw['Parch'], ax=ax[1,1], palette='Spectral').set_title('Parents/Children Distribution')

sns.countplot(y=raw['Embarked'], ax=ax[1,2], palette='Spectral').set_title('Embarkation Distribution')
# Distribution of continous variables: Fare, Age

fig, ax = plt.subplots(1, 2, tight_layout=True, figsize=(6,4))

ax[0].hist(raw['Age'], bins=8)

ax[1].hist(raw['Fare'], bins=8)

title0 = 'Passenger Age Distribution'

title1 = 'Passenger Fare Distribution'

ax[0].set_title(title0)

ax[1].set_title(title1)
# Examine the correlation between factors

corr_matrix = raw.corr()
# Plot annotated heatmap for correlation matrix

mask = np.triu(corr_matrix)

fig, ax = plt.subplots(figsize=(15,6))

sns.heatmap(corr_matrix, annot=True, mask=mask, vmin=-1, vmax=1, center=0, linewidth=0.8, cmap='coolwarm', ax=ax)

plt.title('Titanic Variables Correlation Matrix')
raw1 = raw.copy(deep=True)

data_cleaner = [raw1, validation]
raw1.isnull().sum()
validation.isnull().sum()
# Examine the correlated factors with Age: Pclass has a high correlation with Age

all_corr = corr_matrix.abs().unstack().sort_values(kind='quicksort', ascending=False)

df_all_corr = pd.DataFrame(all_corr).reset_index().rename(columns={'level_0':'Feature-A',

                                                                   'level_1':'Feature-B',

                                                                   '0': 'Correlation Coefficient'})

df_all_corr[df_all_corr['Feature-A']=='Age']
# Calculate median age of each group

print("Median age of all passengers: ", raw['Age'].median())

age_PS_median = raw.groupby(['Pclass','Sex']).median()['Age']

age_PS_median
# Replace missing Age values with median of each groups

for dataset in data_cleaner:

    dataset['Age'] = dataset.groupby(['Pclass','Sex'])['Age'].transform(lambda x: x.fillna(x.median()))
validation[validation['Fare'].isnull()]
median_fare = validation.groupby(['Pclass','SibSp','Parch'])['Fare'].median()

validation['Fare'].fillna(median_fare[3][0][0], inplace=True)
validation[validation['PassengerId']==1044]
raw1[raw1['Embarked'].isnull()]
# Replace missing Embarked values with S(Southampton)

raw1['Embarked'].fillna('S', inplace=True)
# Create Deck column

for dataset in data_cleaner:

    dataset['Deck'] = dataset['Cabin'].str[0]

    dataset['Deck'].fillna('M', inplace=True)



raw1['Deck'].value_counts()
# Transform the Deck-Pclass dataframe

df_raw1_DP = pd.DataFrame(raw1.groupby(['Deck','Pclass'])['Pclass'].count()).rename(columns={'Pclass': 'count'})

df_raw1_DP.reset_index(inplace=True)

table_DP = pd.pivot_table(df_raw1_DP, values='count', index=['Deck'], columns=['Pclass'], aggfunc=np.sum, fill_value=0)
# Calculate the percentage

totals = [i+j+k for i,j,k in zip(table_DP[1], table_DP[2], table_DP[3])]

dPclass_1 = [i/j*100 for i,j in zip(table_DP[1], totals)]

dPclass_2 = [i/j*100 for i,j in zip(table_DP[2], totals)]

dPclass_3 = [i/j*100 for i,j in zip(table_DP[3], totals)]
# Create percent stacked barplot

r = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'M', 'T']

barWidth = 0.8

plt.bar(r, dPclass_1, color='#ffd6aa', edgecolor='white', width=barWidth)

plt.bar(r, dPclass_2, bottom=dPclass_1, color='#efbad6', edgecolor='white', width=barWidth)

plt.bar(r, dPclass_3, bottom=[i+j for i,j in zip(dPclass_1,dPclass_2)], color='#dadafc', edgecolor='white', width=barWidth)

plt.title('Passenger Class Distribution in Decks')
# Examine the survival rate by passenger classes in different decks

raw1.groupby(['Deck', 'Pclass'])['Survived'].mean()
# Exaime & Plot the survival rate in different decks

df_raw1_Dsurvival = pd.DataFrame(raw1.groupby('Deck')['Survived'].mean()).rename(columns={'Survived':'Survival Rate'})

df_raw1_Dsurvival.reset_index(inplace=True)

df_raw1_Dsurvival.plot.barh(x='Deck', y='Survival Rate', color='lightsteelblue', figsize=(10,5))

plt.xlabel('survived %')

plt.xlim(0,1)

plt.title('Survival Rate in Each Deck')



for index, value in enumerate(df_raw1_Dsurvival['Survival Rate']):

    label = format(value,'.2%')

    plt.annotate(label, xy=(value, index-0.10), color='black')



plt.show()
for dataset in data_cleaner:

    dataset['Deck'].replace('T','A', inplace=True)

    dataset['Deck'].replace(['D','E'], 'DE', inplace=True)
for dataset in data_cleaner:

    # Title: split from Name

    # (e.g. Braund, Mr. Owen Harris)

    dataset['Title'] = dataset['Name'].str.split(', ', expand=True)[1].str.split('.', expand=True)[0]

    

    # FamilySize: the Person + Siblings/Spouses + Parents/Children

    dataset['FamilySize'] = 1 + dataset['SibSp'] + dataset['Parch']

    

    # Group Age into Age Bins using cut (value bins)

    # cut: bin values into discrete intervals

    dataset['AgeBin'] = pd.cut(dataset['Age'].astype(int), 8)

    

    # Group Fare into Fare Bins using qcut (frequency bins)

    # qcut: quantile-based discretization function

    dataset['FareBin'] = pd.qcut(dataset['Fare'], 10)
fig, ax = plt.subplots(2,1, figsize=(15,8))

fig.subplots_adjust(hspace=0.3)

sns.countplot(dataset['AgeBin'], ax=ax[0], palette='Blues').set_title('Age Bins Distribution')

sns.countplot(dataset['FareBin'], ax=ax[1], palette='Blues').set_title('Fare Bins Distribution')
print(raw1['Title'].unique())

print(validation['Title'].unique())
# Cleanup rare titles: combine into four categories (Mr. Mrs. Miss Master)

def replace_titles(x):

    title=x['Title']

    if title in ('Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col', 'Sir'):

        return 'Mr'

    elif title in ('Dona', 'the Countess', 'Mme', 'Lady'):

        return 'Mrs'

    elif title in ('Mlle', 'Ms'):

        return 'Miss'

    elif title == 'Dr':

        if x['Sex'] == 'Male':

            return 'Mr'

        else:

            return 'Mrs'

    else:

        return title



for dataset in data_cleaner:

    dataset['Title_group'] = dataset.apply(replace_titles, axis=1)
raw1['Title_group'].value_counts()
# Group the family size: alone, small, medium, large

family_map = {1:'Alone', 2:'Small', 3:'Small', 4:'Medium', 5:'Medium', 6:'Medium', 7:'Large', 8:'Large', 11:'Large'}



for dataset in data_cleaner:

    dataset['FamilySize_group'] = dataset['FamilySize'].map(family_map)
dataset['FamilySize_group'].value_counts()
fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(15,8))



# Plot bar chart to show the distribution of family size

sns.barplot(x=raw1['FamilySize'].value_counts().index,

            y=raw1['FamilySize'].value_counts().values,

            ax=axs[0][0], palette='Blues')

axs[0][0].set_title('Family Size Value Count')



# Show survived-or-not in each family size

sns.countplot(x='FamilySize', hue='Survived', data=raw1,

             ax=axs[0][1], palette='Blues')

axs[0][1].legend(['Not Survived', 'Survived'], loc='upper right')

axs[0][1].set_title('Survival Counts in Each Family Size')



# Plot bar chart to show the distribution of family size groups

sns.barplot(x=raw1['FamilySize_group'].value_counts().index, 

            y=raw1['FamilySize_group'].value_counts().values, 

            ax=axs[1][0], palette='Blues')

axs[1][0].set_title('Family Size Groups Value Count')



# Show survived-or-not in each family size group

sns.countplot(x='FamilySize_group', hue='Survived', data=raw1, 

              ax=axs[1][1], palette='Blues')

axs[1][1].legend(['Not Survived', 'Survived'], loc='upper right')

axs[1][1].set_title('Survival Counts in Each Family Size Group')
raw1.head(3)
predictor = ['Pclass', 'Sex', 'Title_group', 'Age', 'AgeBin', 'FamilySize_group', 'Fare', 'FareBin', 'Embarked', 'Deck']

target = ['Survived']
# Overall Survival Rate

raw_survived = raw.loc[raw['Survived'] == 1, 'PassengerId'].count()

survival_rate =  raw_survived / raw['PassengerId'].count()

print("There were {} people in the raw dataset- {} of them survived. \nThe overall survival rate is {}".format(raw.shape[0], raw_survived, '%.2f'%survival_rate))
predictor_cat = []

for x in predictor:

    if raw1[x].dtypes != 'float64':

        predictor_cat.append(x)
fig, axs = plt.subplots(ncols=2, nrows=4, figsize=(20,18))



for x, i in zip(predictor_cat, range(len(predictor_cat))):

    i += 1

    ax1 = math.floor((i-1)/2)

    ax2 = (i-1) % 2

    survival_rate = raw1.groupby([x])[target[0]].mean()

    

    sns.barplot(x=survival_rate.index, y=survival_rate, ax=axs[ax1][ax2], palette='RdBu')

    axs[ax1][ax2].set_title("Passenger Survival Rate by {}".format(x), fontsize=10)

    axs[ax1][ax2].set_xticklabels(survival_rate.index, fontsize=7)

    axs[ax1][ax2].xaxis.label.set_visible(False)
for x in predictor:

    if raw1[x].dtypes !='float64':

        survival_rate = raw1[[x, target[0]]].groupby(x, as_index=False).mean()

        print("Survival Correlation by {} is {}".format(x, survival_rate))
grid = sns.FacetGrid(raw1, col='Survived', row='Pclass', size=2.5, aspect=1.8)

grid.map(plt.hist, 'Age', alpha=0.5, bins=20)

grid.add_legend()
# Demographic group: children (age<=16), female adults, male adults, seniors (age>64)

# https://stackoverflow.com/questions/16476924/how-to-iterate-over-rows-in-a-dataframe-in-pandas

raw1['demo'] = 0

for index, row in raw1.iterrows():

    if raw1['Age'][index] <= 16:

        raw1['demo'][index] = 'child'

    elif raw1['Age'][index] > 64:

        raw1['demo'][index] = 'senior'

    elif 16 < raw1['Age'][index] <= 64 and raw1['Sex'][index] == 'male':

        raw1['demo'][index] = 'male adult'

    else:

        raw1['demo'][index] = 'female adult'
sns.catplot(x='Pclass', hue='demo', col='Survived', data=raw1,

           kind='count', height=4, aspect=0.8, palette='RdBu')

plt.ylim(0, 100)
grid = sns.FacetGrid(raw1, row='Embarked', col='Survived', size=2.5, aspect=1.8)

grid.map(sns.barplot, 'Fare', 'demo', palette='RdBu', ci=None)

grid.add_legend()
raw.to_csv('raw.csv', header=True, index=False)

raw1.to_csv('raw_clean.csv', header=True, index=False)

validation.to_csv('validation_clean.csv', header=True, index=False)
raw1.head(1)
df = raw1.drop(['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Title', 'AgeBin', 'FareBin', 'FamilySize', 'demo'], axis=1)

df.head(3)
# Convert dummy variables

df_dummies = pd.get_dummies(df)

df_dummies.head(3)
X_dummies = ['Pclass', 'Age', 'Fare', 'Sex_female', 'Sex_male',

       'Embarked_C', 'Embarked_Q', 'Embarked_S', 'Deck_A', 'Deck_B', 'Deck_C',

       'Deck_DE', 'Deck_F', 'Deck_G', 'Deck_M', 'Title_group_Master',

       'Title_group_Miss', 'Title_group_Mr', 'Title_group_Mrs',

       'FamilySize_group_Alone', 'FamilySize_group_Large',

       'FamilySize_group_Medium', 'FamilySize_group_Small']

y = ['Survived']
# Split training and testing data

from sklearn.model_selection import train_test_split

df_X_train, df_X_test, df_y_train, df_y_test = train_test_split(df_dummies[X_dummies], df_dummies[y], random_state=0)

print("The original dataset: {}. \nThe training set: {}. \nThe testing set: {}.".format(df.shape, df_X_train.shape, df_X_test.shape))
# Supervised Learning Algorithms



# 1. Generalized Linear Models

from sklearn.linear_model import LogisticRegression # - Logistic Regression

lr = LogisticRegression(C=0.01, solver='liblinear')



from sklearn.linear_model import RidgeCV # - Ridge Classification

ridge = RidgeCV()



# 2. Classification

from sklearn.tree import DecisionTreeClassifier # - Decision Tree

dtree = DecisionTreeClassifier(criterion='entropy', max_depth=5)



from sklearn import neighbors # - KNN

knn = neighbors.KNeighborsClassifier(n_neighbors=5)



from sklearn.svm import SVC # - SVM

svc = SVC(kernel='linear', probability=True)



from sklearn.naive_bayes import GaussianNB # - Naive Bayes

gnb = GaussianNB()



# 3. Gaussian Process

from sklearn import gaussian_process

gp = gaussian_process.GaussianProcessClassifier()



# 4. Ensemble Methods

from sklearn.ensemble import AdaBoostClassifier

adaboost = AdaBoostClassifier(n_estimators=100, base_estimator=None, learning_rate=1, random_state=1) # - Boosting



from sklearn.ensemble import BaggingClassifier

bagging = BaggingClassifier() # - Bagging



from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

forest = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0) # - Random Forest

extraTree = ExtraTreesClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0) # - Extra Trees







# Cross Validation

from sklearn.model_selection import cross_validate
MLA = [lr, ridge, dtree, knn, svc, gnb, gp, adaboost, bagging, forest, extraTree]
# create a table to compare algorithms' performance

MLA_columns = ['MLA Name', 'MLA Parameter', 'Test Accuracy Score', 'Processing Time']

MLA_table = pd.DataFrame(columns=MLA_columns)



# create a table of prediction results for the test dataset

MLA_predict = df_y_test
MLA_row = 0



for model in MLA:

    # get the name and parameters of the model

    MLA_name = model.__class__.__name__

    MLA_table.loc[MLA_row, 'MLA Name'] = MLA_name

    MLA_table.loc[MLA_row, 'MLA Parameter'] = str(model.get_params())

    

    # train the model

    model.fit(df_X_train, df_y_train)

    

    # make prediction

    MLA_predict[MLA_name] = model.predict(df_X_test)

    

    # use cross validation to score model

    cv_results = cross_validate(model, df_dummies[X_dummies], df_dummies[y], cv=10)

    MLA_table.loc[MLA_row, 'Test Accuracy Score'] = cv_results['test_score'].mean()

    MLA_table.loc[MLA_row, 'Processing Time'] = cv_results['fit_time'].mean()

    

    MLA_row += 1
MLA_table.sort_values(by=['Test Accuracy Score'], ascending=False, inplace=True)

MLA_table
MLA_table['Test Accuracy Score'] = [float(score) for score in MLA_table['Test Accuracy Score']]
MLA_table.plot.barh(x='MLA Name', y='Test Accuracy Score', color='lightsteelblue', figsize=(10,5))

plt.xlabel('Accuracy Score %')

plt.xlim(0.3,0.9)

plt.ylabel('Model')

plt.title('Accuracy Comparision Machine Learning Algorithm')



for index, value in enumerate(MLA_table['Test Accuracy Score']):

    label = format(value,'.2%')

    plt.annotate(label, xy=(value, index-0.10), color='black')



plt.show()
MLA_predict[['Survived', 'AdaBoostClassifier', 'BaggingClassifier', 'DecisionTreeClassifier', 'RandomForestClassifier']].head()
# Import model evaluation libraries

from sklearn import metrics

from sklearn.metrics import classification_report, confusion_matrix
# Define confusion matrix plotting

import itertools

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):

    

    """

    This function prints and plots the confusion matrix.

    Default by non-normalized confusion matrix.

    Normalization can be applied by setting 'normalize=True'

    """

    

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("normalized confusion matrix")

    else:

        print("confusion matrix without normalization")

        

    print(cm)

    

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes)

    plt.yticks(tick_marks, classes, rotation=90)

    

    fmt = '.2f' if normalize else 'd'

    thresh = cm.max()/2

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, format(cm[i,j], fmt),

                 horizontalalignment='center',

                 color='white' if cm[i,j] > thresh else 'black')

        

    plt.tight_layout()

    plt.ylabel('True Label')

    plt.xlabel('Predicted Label')
# Evaluate the Bagging model- accuracy score

metrics.accuracy_score(MLA_predict['Survived'], MLA_predict['BaggingClassifier'])
# Evaluate the Bagging model- confusion matrix

cnf_matrix = confusion_matrix(MLA_predict['Survived'], MLA_predict['BaggingClassifier'], labels=[1,0])

np.set_printoptions(precision=2)



plt.figure()

plot_confusion_matrix(cnf_matrix, classes=['survived', 'died'], normalize=False, title='Bagging Model- Confusion Matrix')
# Evaluate the AdaBoost model- accuracy score

metrics.accuracy_score(MLA_predict['Survived'], MLA_predict['AdaBoostClassifier'])
# Evaluate the AdaBoost model- confusion matrix

cnf_matrix = confusion_matrix(MLA_predict['Survived'], MLA_predict['AdaBoostClassifier'], labels=[1,0])

np.set_printoptions(precision=2)



plt.figure()

plot_confusion_matrix(cnf_matrix, classes=['survived', 'died'], normalize=False, title='AdaBoost Model- Confusion Matrix')
# Evaluate the decision tree model- accuracy score

metrics.accuracy_score(MLA_predict['Survived'], MLA_predict['DecisionTreeClassifier'])
# Evaluate the decision tree model- confusion matrix

cnf_matrix = confusion_matrix(MLA_predict['Survived'], MLA_predict['DecisionTreeClassifier'], labels=[1,0])

np.set_printoptions(precision=2)



plt.figure()

plot_confusion_matrix(cnf_matrix, classes=['survived', 'died'], normalize=False, title='Decision Tree Model- Confusion Matrix')
# Evaluate the random forest model- accuracy score

metrics.accuracy_score(MLA_predict['Survived'], MLA_predict['RandomForestClassifier'])
# Evaluate the random forest model- confusion matrix

cnf_matrix = confusion_matrix(MLA_predict['Survived'], MLA_predict['RandomForestClassifier'], labels=[1,0])

np.set_printoptions(precision=2)



plt.figure()

plot_confusion_matrix(cnf_matrix, classes=['survived', 'died'], normalize=False, title='Random Forest Model- Confusion Matrix')
validation.head(1)
# dataset for prediction

prediction = validation.drop(['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Title', 'AgeBin', 'FareBin', 'FamilySize'], axis=1)

prediction = pd.get_dummies(prediction)

prediction.head(3)
# model: bagging

bagging_predict = bagging.predict(prediction)
result_bagging = validation[['PassengerId']]

result_bagging['Survived'] = bagging_predict

result_bagging.head(3)
# model: dtree

dtree_predict = dtree.predict(prediction)
result_dtree = validation[['PassengerId']]

result_dtree['Survived'] = dtree_predict

result_dtree.head(3)