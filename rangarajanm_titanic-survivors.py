# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import os

os.getcwd()

print (os.listdir('/kaggle/input'))

#print (os.scandir('/kaggle/input'))

subdirs, filenames = os.walk("/kaggle/input/")

print(subdirs,filenames)
data_path = '/kaggle/input/titanic/'

train_df = pd.read_csv(data_path+'train.csv')

test_df = pd.read_csv(data_path+'test.csv')
combined_df = pd.concat([train_df,test_df])



train_df.name = 'Training Dataset'

test_df.name = 'Test Dataset'

combined_df.name = 'Combined Dataset'
train_df.info()

train_df.describe()
test_df.describe()

test_df.info()
def missing_columns(df):    

    for col in df.columns.tolist():          

        print('{} column missing values: {}'.format(col, df[col].isnull().sum()))

    print('\n')

    

for df in [train_df,test_df]:

    print('{}'.format(df.name))

    missing_columns(df)
train_df.head()
import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
#g1 = sns.catplot(x="Sex", y="Survived", hue="Pclass", kind="bar", data=train_df)



g = sns.catplot(x="Pclass", y="Survived", hue="Sex", data=train_df,

                height=6, kind="bar", palette="muted")

g.despine(left=True)

g.set_ylabels("Survival probability")
g = sns.catplot(x="Pclass", y='Survived', col='Embarked', data=train_df,

                height=6, kind="bar", palette="muted")

g.despine(left=True)

g.set_ylabels("Survival Rate")
plt.figure(figsize=(8,6))

g = sns.swarmplot(data=train_df,y='Fare',x='Survived', hue='Sex')
# Let's bin the age variable for some visual analysis



age_labels = ['Kids','9 to 17','18 to 35', '36 to 55', '55+']

age_bins = [0,8,17,35,55,100]

train_df['Age_group'] = pd.cut(train_df['Age'],bins=age_bins,labels=age_labels,include_lowest=True)

test_df['Age_group'] = pd.cut(test_df['Age'],bins=age_bins,labels=age_labels)
plt.figure(figsize=(6,2))

sns.countplot(y='Sex',data=train_df)

plt.ylabel("Gender")

plt.xlabel("Count")

#sns.catplot(x='Sex',y='Survived',data=train_df, kind='bar')
plt.figure(figsize=(15,6))

sns.catplot(x='Age_group',col='Sex',kind='count',color='g',data=train_df)
sns.catplot(x='Age_group',y='Survived',kind='bar',data=train_df,color='b')
g2 = sns.catplot(x='Age_group',y='Survived',hue='Sex',data=train_df, kind='bar',aspect=2, height=6)

g2.set_ylabels('Survival Rate')

g2.set_xlabels('Age Groups')

g2.despine(left=True)
## Fill NaN age values with the median within the respective class and gender for training data

mask_m1 = (train_df.Sex == "male") & (train_df.Pclass == 1)

mask_m2 = (train_df.Sex == "male") & (train_df.Pclass == 2)

mask_m3 = (train_df.Sex == "male") & (train_df.Pclass == 3)



mask_f1 = (train_df.Sex == "female") & (train_df.Pclass == 1)

mask_f2 = (train_df.Sex == "female") & (train_df.Pclass == 2)

mask_f3 = (train_df.Sex == "female") & (train_df.Pclass == 3)



# Median age of males in different passenger classes - we will be using the same in both train and test df

# m_age_class1_male = train_df.query('Sex == "male" and Pclass == 1').Age.dropna().median()

m_age_class1_male = combined_df.loc[(combined_df.Sex == "male") & (combined_df.Pclass == 1),'Age'].dropna().median()

m_age_class2_male = combined_df.loc[(combined_df.Sex == "male") & (combined_df.Pclass == 2),'Age'].dropna().median()

m_age_class3_male = combined_df.loc[(combined_df.Sex == "male") & (combined_df.Pclass == 3),'Age'].dropna().median()



# Median age of females in different passenger classes

m_age_class1_female = combined_df.loc[(combined_df.Sex == "female") & (combined_df.Pclass == 1),'Age'].dropna().median()

m_age_class2_female = combined_df.loc[(combined_df.Sex == "female") & (combined_df.Pclass == 2),'Age'].dropna().median()

m_age_class3_female = combined_df.loc[(combined_df.Sex == "female") & (combined_df.Pclass == 3),'Age'].dropna().median()





# fill NaN with the above values

# print(train_df.loc[(train_df.Sex == "male") & (train_df.Pclass == 1),'Age'])



train_df.loc[mask_m1,'Age'] = train_df.loc[mask_m1,'Age'].fillna(m_age_class1_male)

train_df.loc[mask_m2,'Age'] = train_df.loc[mask_m2,'Age'].fillna(m_age_class2_male)

train_df.loc[mask_m3,'Age'] = train_df.loc[mask_m3,'Age'].fillna(m_age_class3_male)



train_df.loc[mask_f1,'Age'] = train_df.loc[mask_f1,'Age'].fillna(m_age_class1_female)

train_df.loc[mask_f2,'Age'] = train_df.loc[mask_f2,'Age'].fillna(m_age_class2_female)

train_df.loc[mask_f3,'Age'] = train_df.loc[mask_f3,'Age'].fillna(m_age_class3_female)



# train_df['Age'].fillna(train_df['Age'].dropna().median(), inplace=True)

# test_df['Age'].fillna(test_df['Age'].dropna().median(), inplace=True)
## Fill NaN age values with the median within the respective class and gender for test datset

mask_m1 = (test_df.Sex == "male") & (test_df.Pclass == 1)

mask_m2 = (test_df.Sex == "male") & (test_df.Pclass == 2)

mask_m3 = (test_df.Sex == "male") & (test_df.Pclass == 3)



mask_f1 = (test_df.Sex == "female") & (test_df.Pclass == 1)

mask_f2 = (test_df.Sex == "female") & (test_df.Pclass == 2)

mask_f3 = (test_df.Sex == "female") & (test_df.Pclass == 3)



# fill NaN with the above values

# print(train_df.loc[(train_df.Sex == "male") & (train_df.Pclass == 1),'Age'])



test_df.loc[mask_m1,'Age'] = test_df.loc[mask_m1,'Age'].fillna(m_age_class1_male)

test_df.loc[mask_m2,'Age'] = test_df.loc[mask_m2,'Age'].fillna(m_age_class2_male)

test_df.loc[mask_m3,'Age'] = test_df.loc[mask_m3,'Age'].fillna(m_age_class3_male)



test_df.loc[mask_f1,'Age'] = test_df.loc[mask_f1,'Age'].fillna(m_age_class1_female)

test_df.loc[mask_f2,'Age'] = test_df.loc[mask_f2,'Age'].fillna(m_age_class2_female)

test_df.loc[mask_f3,'Age'] = test_df.loc[mask_f3,'Age'].fillna(m_age_class3_female)
train_df[train_df.Embarked.isna()]
sns.catplot(data=combined_df[combined_df.Pclass == 1], x='Embarked',kind='count',hue='Sex')
# Filling the missing values in Embarked with C

train_df['Embarked'] = train_df['Embarked'].fillna('C')
median_class3_fare = combined_df.loc[(combined_df.Pclass == 3),'Fare'].dropna().median()

test_df.Fare.fillna(median_class3_fare,inplace=True)



#print(median_class3_fare)

#print (sum(test_df.Fare.isna()))
train_df.head()
# Add new categorical variables for Siblings/Spouse and Parents/Children

# Some of these variables may not be used in our final model

train_df['SibSp_cat'] = (train_df['SibSp'] > 0) * 1

train_df['Parch_cat'] = (train_df['Parch'] > 0) * 1

train_df['Gender'] = (train_df['Sex'] == 'female') * 1 # Female = 1 and male = 0

train_df['Class 1'] = (train_df['Pclass'] == 1) * 1

train_df['Class 2'] = (train_df['Pclass'] == 2) * 1

train_df['Class 3'] = (train_df['Pclass'] == 3) * 1

train_df['IsAlone'] = ((train_df['SibSp'] == 0) & (train_df['Parch'] == 0)) * 1

train_df['High_Fare'] = (train_df['Fare'] >= 200.0) * 1

train_df['Family_Members'] = train_df.SibSp + train_df.Parch
# Add new categorical variables for Siblings/Spouse and Parents/Children in test dataset

test_df['SibSp_cat'] = (test_df['SibSp'] > 0) * 1

test_df['Parch_cat'] = (test_df['Parch'] > 0) * 1

test_df['Gender'] = (test_df['Sex'] == 'female') * 1

test_df['Class 1'] = (test_df['Pclass'] == 1) * 1

test_df['Class 2'] = (test_df['Pclass'] == 2) * 1

test_df['Class 3'] = (test_df['Pclass'] == 3) * 1

test_df['IsAlone'] = ((test_df['SibSp'] == 0) & (test_df['Parch'] == 0)) * 1

test_df['High_Fare'] = (test_df['Fare'] >= 200.0) * 1

test_df['Family_Members'] = test_df.SibSp + test_df.Parch
g4 = sns.catplot(x='Age_group',hue='IsAlone',data=train_df, kind='count',aspect=1.5)

g4.set_ylabels('Count')

g4.set_xlabels('Age Groups')

g4.despine(left=True)
g3 = sns.catplot(x='Age_group',y='Survived',hue='IsAlone',data=train_df, kind='bar',aspect=1.5)

g3.set_ylabels('Survival Rate')

g3.set_xlabels('Age Groups')

g3.despine(left=True)
sns.catplot(data=train_df,x='Family_Members',y='Survived',kind='bar')
sns.swarmplot(data=train_df,x='Family_Members',y='Fare',hue='Pclass')
#Target

y_train = train_df['Survived']



# Build attribute dataset

#features = ['Gender','Age','Class 1','Class 2','Class 3','SibSp','Parch','Embarked','Fare']

features = ['Gender','Age','Class 1','Class 2','Class 3','Family_Members','Embarked','Fare']



X_train = train_df[features]

X_test = test_df[features]
X_train_1hot = pd.get_dummies(X_train)

X_train_1hot
X_test_1hot = pd.get_dummies(X_test)

X_test_1hot
from sklearn.ensemble import RandomForestClassifier



base_model = RandomForestClassifier(random_state=42,n_estimators=100,max_depth=5)



# Fit the dataset using the RF model

fit = base_model.fit(X_train_1hot,y_train)
fit.score(X_train_1hot,y_train)
from sklearn.model_selection import RandomizedSearchCV

import pprint



# Number of trees in random forest

n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1000, num = 10)]



# Number of features to consider at every split

max_features = ['auto', 'sqrt']



# Maximum number of levels in tree

max_depth = [5,6,7,8,9,10]

max_depth.append(None)



# Minimum number of samples required to split a node

min_samples_split = [2, 5, 10]



# Minimum number of samples required at each leaf node

min_samples_leaf = [1, 2, 4]



# Method of selecting samples for training each tree

bootstrap = [True, False]



# Create the random grid

random_grid = {'n_estimators': n_estimators,

               'max_features': max_features,

               'max_depth': max_depth,

               'min_samples_split': min_samples_split,

               'min_samples_leaf': min_samples_leaf,

               'bootstrap': bootstrap}



pp = pprint.PrettyPrinter(indent=4)

pp.pprint(random_grid)


# Use the random grid to search for best hyperparameters

# First create the base model to tune

rf = RandomForestClassifier()

# Random search of parameters, using 5 fold cross validation, 

# search across 100 different combinations, and use all available cores

rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 200, cv = 4, verbose=2, random_state=42, n_jobs = -1)

# Fit the random search model

rf_random.fit(X_train_1hot, y_train)

rf_random.best_params_
rf_best = rf_random.best_estimator_



best_fit = rf_best.fit(X_train_1hot, y_train)

best_fit.score(X_train_1hot,y_train)
predictions = base_model.predict(X_test_1hot)



output = pd.DataFrame({'PassengerId': test_df.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")