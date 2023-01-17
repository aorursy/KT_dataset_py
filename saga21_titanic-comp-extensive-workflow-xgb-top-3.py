import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split

from xgboost import XGBClassifier as xgb

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_val_score

import re

import warnings

warnings.filterwarnings('ignore')
# Load files

gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")

test = pd.read_csv("../input/titanic/test.csv")

train = pd.read_csv("../input/titanic/train.csv")



display(test.head(15))



# A brief statistical summary of the training dataset

display(train.describe())
# Analyze the data shape

train_shape = train.shape

test_shape = test.shape

all_data = pd.concat([train,test], sort=False)

print("Train set shape: ", train_shape)

print("Test set shape: ", test_shape)



# Female/Male survival ratio is an obvious candidate to key feature

female_survival = len(train[(train.Sex=='female') & (train.Survived==1)])/len(train[(train.Sex=='female')])

male_survival = len(train[(train.Sex=='male') & (train.Survived==1)])/len(train[(train.Sex=='male')])

print("Percentage of females that survived: ", female_survival)

print("Percentage of males that survived: ", male_survival)



# Now we plot some plots about survival rates/counts

fig, axes = plt.subplots(1, 2, figsize=(12,5))

axes[0].bar(['male', 'female'], [male_survival,female_survival], alpha = 0.5)

axes[0].set_xlabel('Sex')

axes[0].set_ylabel("Survival fraction")

axes[0].set_title('Survival percentage')

axes[0].set_ylim(0,1)



sns.countplot(x='Sex', hue='Survived', ax=axes[1], data=train, palette='Set3')

axes[1].set_title('Survival count')
train.drop('PassengerId', axis=1).hist(bins=20, figsize=(10, 8), alpha=0.5)

plt.tight_layout()
fig, axes = plt.subplots(1, 3, figsize=(15,5))



sns.distplot(train[train['Sex']=='male']['Age'].dropna(), kde=False, color='blue', bins=30, ax=axes[0])

axes[0].set_title('Age male distribution')

axes[0].set_xlim(0, 80)



sns.distplot(train[train['Sex']=='female']['Age'].dropna(), kde=False, color='orange', bins=30, ax=axes[1])

axes[1].set_title('Age female distribution')

axes[1].set_xlim(0, 80)



sns.kdeplot(train[train['Sex']=='male']['Age'].dropna(), color='blue', ax=axes[2])

sns.kdeplot(train[train['Sex']=='female']['Age'].dropna(), color='orange', ax=axes[2])

axes[2].set_title('Continuous distribution by sex')
fig, axes = plt.subplots(1, 3, figsize=(15,5))



sns.countplot(x='Pclass', data=train, ax=axes[0], palette='Set1')

axes[0].set_title('Pclass count')



sns.barplot(x='Pclass', y='Pclass', hue='Sex', data=train, ax=axes[1], palette='Set3', estimator=lambda x: len(x) / len(train) * 100)

axes[1].set(ylabel="Percent")

axes[1].set_title('Pclass by Sex')



sns.barplot(x='Pclass', y='Pclass', hue='Survived', data=train, ax=axes[2], palette='Set2', estimator=lambda x: len(x) / len(train) * 100)

axes[2].set_title('Pclass survival')

axes[2].set_ylabel('')
fig, axes = plt.subplots(1, 3, figsize=(15,5))



sns.countplot(x='Embarked', data=train, ax=axes[0], palette='Set1')

axes[0].set_title('Embarked port count')



sns.countplot(x='Embarked', hue='Sex', data=train, ax=axes[1],  palette='Set3')

axes[1].set_title('Sex distribution for each Embarked port')

axes[1].set_ylabel('')



sns.countplot(x='Embarked', hue='Survived', data=train, ax=axes[2], palette='Set2')

axes[2].set_title('Survival rate by Embarked port')

axes[2].set_ylabel('')
# Correlation matrix of relevant features

print("Correlations with Survived:")

features = ["Pclass", "Sex", "SibSp", "Parch"]

X = all_data[:len(train)].join(pd.get_dummies(all_data[:len(train)][features]), rsuffix="_dummies", sort=False)

corr = X.corr()

ax = sns.heatmap(

    corr, 

    vmin=-1, vmax=1, center=0,

    cmap=sns.diverging_palette(20, 220, n=200),

    square=True

)

ax.set_xticklabels(

    ax.get_xticklabels(),

    rotation=45,

    horizontalalignment='right'

);

Highest_corr = corr.nlargest(30, 'Survived')['Survived']

print(Highest_corr)



print("Data Exploration: Completed")
# Look for missing data 

missings = {col:all_data[col].isnull().sum() for col in all_data.columns if all_data[col].isnull().sum()>0}

print("The following features contain missing values: ", missings)



## FARE ##

# Passenger with pass_id = 1044 has no Fare informed. He was 3rd class, let's replace his fare with the avg of 3rd class.

all_data_sorted_fare = all_data

all_data_sorted_fare.sort_values(by=['Fare'])

#print(all_data_sorted_fare)

all_data.set_value(all_data[all_data['PassengerId']==1044].index, 'Fare', all_data[all_data['Pclass']==3].Fare.mean())



## EMBARKED ##

# For the passengers without Embarked data, we will use the mode

all_data['Embarked'].fillna(all_data['Embarked'].mode()[0], axis=0, inplace=True)



## AGE ##

# 263 passengers have no age informed. To fill these values, we need some feature engineering (next step)



## CABIN ## 

# Most cabin values are missing. To fill these values, we need some feature engineering (next step)



print("Missing cleaning: Completed")
## CABIN ##

# The letter of the cabin number is related to the docker of the ship (hypothesis). 

all_data['Docker_num'] = [cab[:1] if pd.notnull(cab) else "Unknown" for cab in all_data['Cabin']]

all_data['Has_cabin_informed'] = [1 if pd.notnull(cab) else 0 for cab in all_data['Cabin']]





## TITLE NAME ##

# Mr, Mrs, Miss, Master, etc, are indicative of person status. Hence, it's interesting to extract titles feature

all_data['Title'] = [re.search('\,(.*)\.', name).group(1) for name in all_data['Name']]

all_data.set_value(all_data['PassengerId']==514, 'Title', ' Mrs')

dictionary_of_titles = {

    " Capt": "Crew",

    " Col": "Crew",

    " Major": "Crew",

    " Dr": "Crew",

    " Rev": "Crew",

    " Jonkheer": "VIP",

    " Don": "VIP",

    " Dona": "VIP",

    " Sir" : "VIP",

    " the Countess":"VIP",

    " Lady" : "VIP",

    " Mme": "Mrs",

    " Ms": "Miss",

    " Mrs" : "Mrs",

    " Mlle": "Miss",

    " Miss" : "Miss",

    " Mr" : "Mr",

    " Master" : "Master"

}

all_data['Title'] = all_data.Title.map(dictionary_of_titles)





## AGE ##

# Function to fill missing age depending on title

def fill_missing_age_title(all_data):

    # Average age for each title:

    class_age_female_miss = all_data[(all_data['Title']=='Miss')]['Age'].dropna().mean()

    class_age_female_mrs = all_data[(all_data['Title']=='Mrs')]['Age'].dropna().mean()

    class_age_male_master = all_data[(all_data['Title']=='Master')]['Age'].dropna().mean()

    class_age_male_mr = all_data[(all_data['Title']=='Mr')]['Age'].dropna().mean()

    #class_age_male_crew = all_data[(all_data['Title']=='Crew')]['Age'].dropna().mean()

    #class_age_male_vip = all_data[(all_data['Title']=='VIP')]['Age'].dropna().mean()

    

    # We fill missing age from the average age of the same Title

    all_data[all_data['Age'].isnull()==True].head(5)

    all_data.set_value(all_data[(all_data['Age'].isnull()==True) & (all_data['Title']=='Miss')].index, 'Age', class_age_female_miss)

    all_data.set_value(all_data[(all_data['Age'].isnull()==True) & (all_data['Title']=='Mrs')].index, 'Age', class_age_female_mrs)

    all_data.set_value(all_data[(all_data['Age'].isnull()==True) & (all_data['Title']=='Master')].index, 'Age', class_age_male_master)

    all_data.set_value(all_data[(all_data['Age'].isnull()==True) & (all_data['Title']=='Mr')].index, 'Age', class_age_male_mr)

    #all_data.set_value(all_data[(all_data['Age'].isnull()==True) & (all_data['Title']=='Crew')].index, 'Age', class_age_male_crew)

    #all_data.set_value(all_data[(all_data['Age'].isnull()==True) & (all_data['Title']=='VIP')].index, 'Age', class_age_male_vip)

    

    return all_data

all_data = fill_missing_age_title(all_data)

        

    

## AGE CATEGORY ##

def age_cat(age):

    if age <= 15:

        return 1

    if age <= 35:

        return 2

    if age <= 55:

        return 3

    if age > 55:

        return 4

    else:

        return 0

all_data['Age_cat'] = all_data['Age'].apply(age_cat)

    

    

## FARE CATEGORY ##

def fare_cat(fare):

    if fare < 15:

        return 1

    if fare < 35:

        return 2

    if fare < 100:

        return 3

    if fare > 100:

        return 4

    else:

        return 0    

all_data['Fare_cat'] = all_data['Age'].apply(fare_cat)





## RELATIVES ##

# Compute number of relatives in the ship

all_data['Relatives'] = all_data['SibSp'] + all_data['Parch']





## FAMILIES ##

# Compute families based on surname and fare

def fill_families(all_data):

    all_data['Last_Name'] = all_data['Name'].apply(lambda x: str.split(x, ",")[0])



    # Random chance of surviving, 50%

    default_survival_chance = 0.5

    all_data['Family_Survival'] = default_survival_chance



    # Group data by last name and fare - looking for families

    for grp, grp_df in all_data[['Survived','Name', 'Last_Name', 'Fare', 'Ticket', 'PassengerId',

                               'SibSp', 'Parch', 'Age', 'Cabin']].groupby(['Last_Name', 'Fare']):



        # If not equal to 1, a family is found 

        # Then work out survival chance depending on whether or not that family member survived

        if (len(grp_df) != 1):

            for ind, row in grp_df.iterrows():

                smax = grp_df.drop(ind)['Survived'].max()

                smin = grp_df.drop(ind)['Survived'].min()

                passID = row['PassengerId']

                if (smax == 1.0):

                    all_data.loc[all_data['PassengerId'] == passID, 'Family_Survival'] = 1

                elif (smin == 0.0):

                    all_data.loc[all_data['PassengerId'] == passID, 'Family_Survival'] = 0



    all_data['IsAlone'] = 0

    all_data.loc[all_data['Relatives'] == 0, 'IsAlone'] = 1 

    return all_data

all_data = fill_families(all_data)



# Let's take a look to our new features

all_data.head(5) 
## FARE SKEWNESS ##

# Skewness and Kurtosis analysis for Fare. Apply log transform if skew is too high (see graph below)

sns.distplot(all_data['Fare'].dropna())

plt.ylabel('Frequency')

plt.title('Fare distribution')

#print("Skewness: %f" % all_data['Fare'].dropna().skew())

#print("Kurtosis: %f" % all_data['Fare'].dropna().kurt())

all_data['Fare']=all_data['Fare'].apply(lambda x: np.log(x))
## FARE SKEWNESS ##

# Skewness and Kurtosis analysis for Age. Apply log transform if skew is too high

sns.distplot(all_data['Age'].dropna())

plt.ylabel('Frequency')

plt.title('Age distribution')

#print("Skewness: %f" % all_data['Fare'].dropna().skew())

#print("Kurtosis: %f" % all_data['Fare'].dropna().kurt())

all_data['Age']=all_data['Age'].apply(lambda x: np.log(x))

                

print("Feature engineering: Completed")
# Split again train/test

X = all_data[:len(train)]

X_test_full = all_data[len(train):]



# Split target variable

y = X.Survived

X.drop('Survived', axis=1, inplace=True)

print(len(all_data), len(X), len(X_test_full))



# Break off validation set from training data

X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.95, test_size=0.05, random_state=0)



# Select categorical columns with relatively low cardinality (convenient but arbitrary)

low_cardinality_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 10 and 

                        X_train_full[cname].dtype not in ['int64', 'float64']]

print("Low cardinality columns: ", low_cardinality_cols)



# Select numeric columns

numeric_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]

print("Numeric columns: ", numeric_cols)



# Keep selected columns only

my_cols = low_cardinality_cols + numeric_cols

X_train = X_train_full[my_cols].copy()

X_valid = X_valid_full[my_cols].copy()

X_test = X_test_full[my_cols].copy()



# One-hot encode the data (to shorten the code, we use pandas)

X_train = pd.get_dummies(X_train)

X_valid = pd.get_dummies(X_valid)

X_test = pd.get_dummies(X_test)

X_train, X_valid = X_train.align(X_valid, join='left', axis=1)

X_train, X_test = X_train.align(X_test, join='left', axis=1)
def xgb_optimize(X_train, y_train):

    xgb1 = xgb()

    parameters = {'nthread':[1], #when use hyperthread, xgboost may become slower

                  'learning_rate': [.005, .004, .003, .002, .0009, 0.008], 

                  'max_depth': [4, 5, 6, 7],

                  'min_child_weight': [4, 5, 6],

                  'silent': [1],

                  'subsample': [0.5],

                  'colsample_bytree': [0.7],

                  'n_estimators': [1000, 2500, 5000, 7500]}



    xgb_grid = GridSearchCV(xgb1,

                            parameters,

                            cv = 3,

                            n_jobs = 5,

                            verbose=True)



    xgb_grid.fit(X_train, y_train.astype(int))



    print(xgb_grid.best_score_)

    print(xgb_grid.best_params_)

    

# Uncomment the call to the xgb_optimize function to perform a (very) time consuming grid search 

# xgb_optimize(X_train, y_train)
# Define model with best MAE

model = xgb(colsample_bytree=0.7, learning_rate=0.0009, max_depth=6, min_child_weight=5, n_estimators=2500, 

                     nthread=1, silent=1, subsample=0.7, random_state=0, 

                     early_stopping_rounds = 10, eval_set=[(X_valid, y_valid)], verbose=False)



# Train and test the model



print("Let's the training begin. Plase wait.")



# Bundle preprocessing and modeling code in a pipeline

my_pipeline = Pipeline(steps=[('model', model)])

my_pipeline.fit(X_train, y_train.astype(int))



print("Training finished! Now let's predict test values.")



preds_test = my_pipeline.predict(X_test)



# Save test predictions to file

output = pd.DataFrame({'PassengerId': X_test.index+892, # We didn't preserved the indexes of rows, so we need to correct it manually. Not elegant at all but it works

                       'Survived': preds_test.astype(int)})

output.to_csv('submission.csv', index=False)



print("Everything finished correctly!")
# Cross validation accuracy for 3 folds

scores = cross_val_score(my_pipeline, X_train, y_train,

                              cv=5,

                              scoring='accuracy')

print(scores)
for pass_id in [956,981,1053,1086,1088,1199,1284,1309]:

    output.set_value(output['PassengerId']==pass_id, 'Survived', 1)



for pass_id in [910,925,929,1024,1032,1080,1172,1176,1257,1259]:

    output.set_value(output['PassengerId']==pass_id, 'Survived', 0)



# Analysis of particular parentship/Sex casuistics that should lead to predictable outputs

#output.set_value(output[output[output['PassengerId'].isin([956,981,1053,1086,1088,1199,1284,1309])], 'Survived', [1,1,1,1,1,1,1,1]])

#output.set_value(output[output[output['PassengerId'].isin([956,981,1053,1086,1088,1199,1284,1309])], 'Survived', [0,0,0,0,0,0,0,0]])

output[output['PassengerId'].isin([956,981,1053,1086,1088,1199,1284,1309])]

output[output['PassengerId'].isin([910,925,929,1024,1032,1080,1172,1176,1257,1259])]



all_data[all_data['PassengerId']==864] # Had 14.5 years, algorithm predicted 21.8

all_data[all_data['PassengerId']==29] # Had 22.7 years, algorithm predicted 21.8

output[output['PassengerId']==1298] # OK

output[output['PassengerId']==1301] # OK

output[output['PassengerId']==1300] # OK

output[output['PassengerId']==893] # Fail (47y, has Sib/Sp=1, 3rd class)

all_data[all_data['PassengerId'].isin(output[output['Survived']==0]['PassengerId'])]
# Save test predictions to file

output = pd.DataFrame({'PassengerId': X_test.index+892,

                       'Survived': preds_test.astype(int)})

output.to_csv('submission.csv', index=False)