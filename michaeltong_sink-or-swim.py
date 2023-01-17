import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from IPython.display import display

# magic function below allows us to display plots
%matplotlib inline 
warnings.filterwarnings('ignore') # ignores common warnings
#np.random.seed(12345678) # remove this if you'd like to randomize results
df_train = pd.read_csv("../input/train.csv")
print("Data Shape:",        df_train.shape, '----', sep='\n')
print("Example Data:")      
display(df_train.head(2))
print('----')
print("Datatypes:",         df_train.dtypes, '----', sep='\n')
print("Basic Information:", df_train.describe(), '----', sep='\n')
print("Unique Datapoints:", df_train.nunique(), '----', sep='\n')
df_train.drop('PassengerId', axis=1, inplace=True) # removing the PassengerId feature since it is essentialy the index.

df_train[(df_train['Age']   == df_train['Age'].min()) |\
         (df_train['Age']   == df_train['Age'].max()) |\
         (df_train['SibSp'] == df_train['SibSp'].max()) |\
         (df_train['Parch'] == df_train['Parch'].max())]
df_train.isna().sum()
# Cabin and Ticket feature don't appear to provide any useful information for our model
df_train.drop(['Cabin'], axis=1, inplace=True)
df_train.drop(['Ticket'], axis=1, inplace=True)

sns.set_context(context='poster')
sns.set_style('whitegrid')
title_size = 20 # fontsize variables
axis_size = 16

def add_labels_bar(ax, l_labels):
    """Accepts an axes object and a list of labels, will place a visual label above each bar"""
    p = ax.patches
    
    for i, label in zip(p, l_labels):
        height = i.get_height()
        ax.text(i.get_x() + i.get_width() / 2, height + 5, label,
                ha='center', va='bottom')
fig, ax = plt.subplots(ncols=2, figsize=(12,8))

ax[0].set_title('Age vs Survival', fontsize=title_size)
ax[0].set_xlabel('Age', fontsize=axis_size)
ax[0].set_ylabel('Estimated Density', fontsize=axis_size)
ax[0].set_ylim(0, 0.038) # top of plot was cut off
sns.distplot(a=df_train['Age'][df_train['Survived']==1].dropna(), bins=10, color='g', ax=ax[0])
sns.distplot(a=df_train['Age'][df_train['Survived']==0].dropna(), bins=10, color='r', ax=ax[0])

ax[1].set_title('Survival vs Age', fontsize=title_size)
ax[1].set_xlabel('Survived', fontsize=axis_size)
ax[1].set_ylabel('Age', fontsize=axis_size)
sns.violinplot(x='Survived', y='Age', data=df_train, ax=ax[1])

fig.tight_layout() # prevent overlap
fig, ax = plt.subplots(ncols=3, figsize=(16,8))
ax[0].set_title('Death Rate by Age', fontsize=title_size)
ax[1].set_title('Survival Rate by Age', fontsize=title_size)
ax[2].set_title('Under 16 Age Count Comparison')

ax[0].tick_params(axis='x', rotation=-90, length=5, width=1)
ax[1].tick_params(axis='x', rotation=-90, length=5, width=1, labelsize=15)
ax[2].tick_params(axis='x', rotation=-90, length=5, width=1, labelsize=14)

ax[0] = sns.countplot(x='Age', data=df_train[(df_train['Age'] < 16) & (df_train['Survived'] == 0)], color='r', ax=ax[0])
ax[1] = sns.countplot(x='Age', data=df_train[(df_train['Age'] < 16) & (df_train['Survived'] == 1)], color='g', ax=ax[1])
ax[2] = sns.countplot(x='Age', hue='Survived', data=df_train[(df_train['Age'] < 16)], ax=ax[2])
ax[2].legend(labels=['Perished','Survived'] ,loc=1)
children = df_train[df_train['Age'] < 16]
children_survived = df_train[(df_train['Age'] < 16) & (df_train['Survived'] == 1)]
adult = df_train[df_train['Age'] >= 16]
adult_survived = df_train[(df_train['Age'] >= 16) & (df_train['Survived'] == 1)]

perc_child_survival = children_survived.Age.count() / children.Age.count()
perc_adult_survival = adult_survived.Age.count() / adult.Age.count()

SE = df_train[df_train['Survived'] == 1].Age.std()/(children.Age.count()**0.5)
z_score = (children.Age.mean()-df_train[df_train['Survived'] == 1].Age.mean()) / SE    

print("Chance for a child to survive:", perc_child_survival)
print("Chance for an adult to survive:", perc_adult_survival)
print("Z-score for child survival:", z_score)
def find_suffix(name):
    name = name.strip()
    last_first_name = name.split(',')
    first_name = last_first_name[1].strip()
    suffix = first_name.split('.')[0]
    return(suffix)

def get_suffixes(df):
    return([find_suffix(name) for name in df.loc[:, 'Name']])

suffixes = get_suffixes(df_train)
count = dict(zip(set(suffixes), [0]*len(set(suffixes))))
for suffix in suffixes:
    count[suffix] += 1
    
print("The unique suffixes are:", *[str(i)+',' for i in count.keys()])
print("Their occurances are:", count)
odd_suffixes = ['Rev', 'Mlle', 'Mme', 'Jonkheer']

for index, value in df_train.iterrows():
    if find_suffix(value['Name']) in odd_suffixes:
        print(df_train.loc[index, ['Name', 'Sex', 'Age']])
        print('---')
df_train['suffix'] = suffixes
print("Mean: ", df_train[df_train['suffix'] == 'Miss'].Age.mean())
print("Mendian: ", df_train[df_train['suffix'] == 'Miss'].Age.median())
print("Percentage of female children with suffix:",\
      df_train[(df_train['suffix'] == 'Miss') & (df_train['Age'] < 16)].Age.count() / df_train[df_train['suffix'] == 'Miss'].Age.count())

fig = plt.figure(figsize=(12,6))
plt.title("Age of 'Miss' Suffix Occurences")
plt.tick_params(rotation=-90, labelsize=14, length=5, width=1)
sns.countplot(df_train[df_train['suffix'] == 'Miss'].Age)


avg_adult = adult.Age.mean()
avg_male_adult = adult[adult['Sex'] == 'male'].Age.mean()
avg_female_adult = adult[adult['Sex'] == 'female'].Age.mean()

avg_child = children.Age.mean()
avg_male_child = children[children['Sex'] == 'male'].Age.mean()
avg_female_child = children[children['Sex'] == 'female'].Age.mean()

print("Average adult age:", avg_adult)
print("Average male adult age:", avg_male_adult)
print("Average female adult age:", avg_female_adult)
print("Average child age:", avg_child)
print("Average male child age:", avg_male_child)
print("Average female child age:", avg_female_child)
# age_fitted
from scipy import stats

def chance_female_child(df): 
    """Returns the percentage of female children based on the Miss Suffix"""
    return(df[(df['suffix'] == 'Miss') & (df['Age'] < 16)].Age.count() /\
             df[df['suffix'] == 'Miss'].Age.count())
    

def random_age(upper, lower, mean, std):
    """This function bounds the Gaussian distribution to a high and low value"""
    x = stats.truncnorm((lower-mean)/std, (upper-mean)/std, loc=mean, scale=std)
    return(x.rvs(1)[0])

def fit_age(df):
    """Fits NaN ages with a normal distribution"""
    df['age_fitted'] = np.nan
    for index, value in df['suffix'].iteritems():
        if np.isnan(df.loc[index, 'Age']):
            if value == 'Master':
                df.loc[index, 'age_fitted'] = round(random_age(15, df.Age.min(),\
                                                          df[df['Age'] < 16].Age.mean(),\
                                                          df[df['Age'] < 16].Age.std()), 2)
            elif value == 'Miss':
                if np.random.randint(0, 101)/100 <= chance_female_child(df):
                    df.loc[index, 'age_fitted'] = round(random_age(17, df.Age.min(),\
                                                                  df[df['Age'] < 16].Age.mean(),\
                                                                  df[df['Age'] < 16].Age.std()), 2)
                else:
                    df.loc[index, 'age_fitted'] = round(random_age(df.Age.max(), 16,\
                                                                  df[df['Age'] >= 16].Age.mean(),\
                                                                  df[df['Age'] >= 16].Age.std()), 2)
            else:
                df.loc[index, 'age_fitted'] = round(random_age(df.Age.max(), 16,\
                                                          df[df['Age'] >= 16].Age.mean(),\
                                                          df[df['Age'] >= 16].Age.std()), 2)
        else:
            df.loc[index, 'age_fitted'] = df.loc[index,'Age']    
    return(df)

df_train = fit_age(df_train)

fit_adult_avg_age = df_train[df_train['age_fitted'] >= 16].age_fitted.mean()
fit_child_avg_age = df_train[df_train['age_fitted'] < 16].age_fitted.mean()


print("Number of 'fitted' children:", df_train[df_train['age_fitted'] < 16].shape[0] - df_train[df_train['Age'] < 16].shape[0])
print("Previous average adult age:", avg_adult)
print("Previous average child age:", avg_child)
print("Fitted average adult age:", fit_adult_avg_age)
print("Fitted average child age:", fit_child_avg_age)

def age_adult(df):
    """Creates the age_adult feature, must be run after fit_age"""
    df['age_adult'] = 1

    for index, value in df['age_fitted'].iteritems():
        if value < 16:
            df.loc[index, 'age_adult'] = 0
    return(df)

df_train = age_adult(df_train)
print('The number of adults is:', df_train['age_adult'].sum())
print('The number of children is:', df_train[df_train['age_adult'] == 0].age_adult.count())
def age_nan(df):
    df['age_nan'] = 1
    for index, value in df['Age'].iteritems():
        if not np.isnan(value):
            df.loc[index, 'age_nan'] = 0
    return(df)
df_train = age_nan(df_train)

num_survivors = df_train[(df_train['Survived'] == 1) & (df_train['age_nan'] == 0)].shape[0] 
num_survivors_nan = df_train[(df_train['Survived'] == 1) & (df_train['age_nan'] == 1)].shape[0]
num_deceased = df_train[(df_train['Survived'] == 0) & (df_train['age_nan'] == 0)].shape[0]
num_deceased_nan = df_train[(df_train['Survived'] == 0) & (df_train['age_nan'] == 1)].shape[0]

print("Number of survivors with known age:", num_survivors)
print("Number of survivors without known age:", num_survivors_nan)
print("Number of deceased with known age:", num_deceased)
print("Number of deceased without known age:", num_deceased_nan)
print("Percentage of survivors without known age:", num_survivors_nan/num_survivors)
print("Percentage of deceased without known age:", num_deceased_nan/num_deceased)
fig, ax = plt.subplots(figsize=(12,6))
ax.set_title('Male vs Female Survival Rates', fontsize=title_size)
sns.countplot(x='Survived',hue='Sex', data=df_train, ax=ax)
add_labels_bar(ax, [df_train[(df_train['Survived']==0) & (df_train['Sex']=='male')].Sex.count(),\
                    df_train[(df_train['Survived']==0) & (df_train['Sex']=='female')].Sex.count(),\
                    df_train[(df_train['Survived']==1) & (df_train['Sex']=='male')].Sex.count(),\
                    df_train[(df_train['Survived']==1) & (df_train['Sex']=='female')].Sex.count()])

ax.set_xticklabels = (['Deceased', 'Survived'])
fig.tight_layout() # prevent overlap
df_train['is_male'] = pd.get_dummies(data=df_train['Sex'], drop_first=True)
df_train.head()
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,6))

sns.barplot(x='Pclass', y='Survived', data=df_train, ax=ax[0], ci=None)
ax[0].set_title('Class vs Survival Chances', size=20)
ax[0].set_xlabel('Class', size=15)
ax[0].set_ylabel('Survival Probability', size=15)

sns.countplot(x='Pclass', data=df_train, ax=ax[1])
ax[1].set_title('Number of People per Class', size=20)
ax[1].set_xlabel('Class', size=15)
ax[1].set_ylabel('Number of People', size=15)

fig.tight_layout()
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,6))

sns.barplot(x='Pclass', y='Survived', data=df_train[df_train['Age'] < 16], ax=ax[0], ci=None)
ax[0].set_title('Class vs Survival Chances for Children', size=18)
ax[0].set_xlabel('Class', size=15)
ax[0].set_ylabel('Survival Probability', size=15)

sns.countplot(x='Pclass', data=df_train[df_train['Age'] < 16], ax=ax[1])
ax[1].set_title('Number of Children per Class', size=20)
ax[1].set_xlabel('Class', size=15)
ax[1].set_ylabel('Number of People', size=15)

fig.tight_layout()
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,6))

sns.barplot(x='Pclass', y='Survived', data=df_train[df_train['Age'] >= 16], ax=ax[0], ci=None)
ax[0].set_title('Class vs Survival Chances for Adults', size=18)
ax[0].set_xlabel('Class', size=15)
ax[0].set_ylabel('Survival Probability', size=15)

sns.countplot(x='Pclass', data=df_train[df_train['Age'] >= 16], ax=ax[1])
ax[1].set_title('Number of Adults per Class', size=20)
ax[1].set_xlabel('Class', size=15)
ax[1].set_ylabel('Number of People', size=15)

fig.tight_layout()
def fam_size(df):
    df['fam_size'] = df['SibSp'] + df['Parch'] + 1
    return(df)

def prob_survival(df, feature):
    """Returns the probability of survival for each feature as a dictionary (survived / total count)"""
    unique_values = df[feature].unique()
    count_total = dict(zip(unique_values, [0]*len(unique_values)))
    count_survived = count_total.copy()
    prob_survival = count_total.copy()
    
    for index,value in df.iterrows():
        count_total[df.loc[index, feature]] += 1
        if df.loc[index, 'Survived'] == 1:
            count_survived[df.loc[index, feature]] += 1
    
    for i in prob_survival.keys():
        prob_survival[i] = count_survived[i]/count_total[i]
    
    return(prob_survival)
df_train = fam_size(df_train)

Parch_survival = prob_survival(df_train, 'Parch')
SibSp_survival = prob_survival(df_train, 'SibSp')
fam_survival = prob_survival(df_train, 'fam_size')

fig, ax = plt.subplots(ncols=3, figsize=(14,6))
fig.tight_layout()

sns.pointplot(x=list(Parch_survival.keys()), y=list(Parch_survival.values()), ax=ax[0])
ax[0].set_title('Survival Rate of Parch')
ax[0].set_xlabel('Size')
ax[0].set_ylabel('Probability of Survival')

sns.pointplot(x=list(SibSp_survival.keys()), y=list(SibSp_survival.values()), ax=ax[1])
ax[1].set_title('Survival Rate of SibSp')
ax[1].set_xlabel('Size')
ax[1].set_ylabel('Probability of Survival')

sns.pointplot(x=list(fam_survival.keys()), y=list(fam_survival.values()), ax=ax[2])
ax[2].set_title('Survival Rate of Family Size')
ax[2].set_xlabel('Size')
ax[2].set_ylabel('Probability of Survival')


df_train.head()
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score

df_test = pd.read_csv('../input/test.csv')
df_test['suffix'] = get_suffixes(df_test)
df_test = fit_age(df_test)
df_test = age_adult(df_test)
df_test = age_nan(df_test)
df_test = fam_size(df_test)
df_test['is_male'] = pd.get_dummies(data=df_test['Sex'], drop_first=True)

training_features_fitted = df_train.loc[:, ['Pclass', 'age_fitted', 'fam_size', 'is_male']]
training_features_adult = df_train.loc[:, ['Pclass', 'age_adult', 'fam_size', 'is_male']]
training_features_nan = df_train.loc[:, ['Pclass', 'age_nan', 'fam_size', 'is_male']]

training_labels = df_train['Survived'].values
# AdaBoost 
classifier = AdaBoostClassifier(n_estimators=100, learning_rate=0.381, random_state=1000)
train_age_fitted_ada = classifier
train_age_adult_ada = classifier
train_age_nan_ada = classifier

train_age_fitted_ada.fit(training_features_fitted, training_labels)
train_age_adult_ada.fit(training_features_adult, training_labels)
train_age_nan_ada.fit(training_features_nan, training_labels)

predict_age_fitted = train_age_fitted_ada.predict(training_features_fitted)
predict_age_adult = train_age_adult_ada.predict(training_features_adult)
predict_age_nan = train_age_nan_ada.predict(training_features_nan)

accuracy_age_fitted = cross_val_score(train_age_fitted_ada, training_features_fitted, training_labels)
accuracy_age_adult = cross_val_score(train_age_adult_ada, training_features_adult, training_labels)
accuracy_age_nan = cross_val_score(train_age_nan_ada, training_features_nan, training_labels)

print("~Adaptive Boost results~")
print("age_fitted accuracy:", accuracy_age_fitted.mean())
print("age_adult accuracy:", accuracy_age_adult.mean())
print("age_nan accuracy:", accuracy_age_nan.mean())
print("Feature importance", dict(zip(training_features_fitted.columns, np.float16(train_age_fitted_ada.feature_importances_)))) # float16 was a cheap way to round since around was not working
# RandomForest
#classifier = RandomForestClassifier(n_estimators=100, min_samples_leaf=3, max_depth=3)
classifier = RandomForestClassifier()
train_age_fitted_rf = classifier
train_age_adult_rf = classifier
train_age_nan_rf = classifier

train_age_fitted_rf.fit(training_features_fitted, training_labels)
train_age_adult_rf.fit(training_features_adult, training_labels)
train_age_nan_rf.fit(training_features_nan, training_labels)

predict_age_fitted = train_age_fitted_rf.predict(training_features_fitted)
predict_age_adult = train_age_adult_rf.predict(training_features_adult)
predict_age_nan = list(map(float, train_age_nan_rf.predict(training_features_nan)))

accuracy_age_fitted = cross_val_score(train_age_fitted_rf, training_features_fitted, training_labels)
accuracy_age_adult = cross_val_score(train_age_adult_rf, training_features_adult, training_labels)
accuracy_age_nan = cross_val_score(train_age_nan_rf, training_features_nan, training_labels)

print("~Random Forest results~")
print("age_fitted accuracy:", accuracy_age_fitted.mean())
print("age_adult accuracy:", accuracy_age_adult.mean())
print("age_nan accuracy:", accuracy_age_nan.mean())
print("Feature importance", dict(zip(training_features_fitted.columns, np.float16(train_age_fitted_rf.feature_importances_))))
# XGBoost
train_age_fitted_xgb = XGBClassifier()
train_age_adult_xgb = XGBClassifier()
train_age_nan_xgb = XGBClassifier()

train_age_fitted_xgb.fit(training_features_fitted, training_labels)
train_age_adult_xgb.fit(training_features_adult, training_labels)
train_age_nan_xgb.fit(training_features_nan, training_labels)

predict_age_fitted = train_age_fitted_xgb.predict(training_features_fitted)
predict_age_adult = train_age_adult_xgb.predict(training_features_adult)
predict_age_nan = train_age_nan_xgb.predict(training_features_nan)

accuracy_age_fitted = cross_val_score(train_age_fitted_xgb, training_features_fitted, training_labels)
accuracy_age_adult = cross_val_score(train_age_adult_xgb, training_features_adult, training_labels)
accuracy_age_nan = cross_val_score(train_age_nan_xgb, training_features_nan, training_labels)

print("~XGBoost results~")
print("age_fitted accuracy:", accuracy_age_fitted.mean())
print("age_adult accuracy:", accuracy_age_adult.mean())
print("age_nan accuracy:", accuracy_age_nan.mean())
print("Feature importance", dict(zip(training_features_fitted.columns, np.float16(train_age_fitted_xgb.feature_importances_))))
test_ada = train_age_fitted_ada.predict(df_test.loc[:, ['Pclass', 'age_fitted', 'fam_size', 'is_male']])
test_rf = train_age_fitted_rf.predict(df_test.loc[:, ['Pclass', 'age_fitted', 'fam_size', 'is_male']])
test_xgb = train_age_fitted_xgb.predict(df_test.loc[:, ['Pclass', 'age_fitted', 'fam_size', 'is_male']])

submission_ada = pd.DataFrame(data=test_ada, index=df_test['PassengerId'], columns=['Survived'])
submission_ada.index.name = 'PassengerId'
#submission_ada.to_csv('predictionAda.csv')

submission_rf = pd.DataFrame(data=test_rf, index=df_test['PassengerId'], columns=['Survived'])
submission_rf.index.name = 'PassengerId'
#submission_rf.to_csv('predictionRF.csv')

submission_xgb = pd.DataFrame(data=test_xgb, index=df_test['PassengerId'], columns=['Survived'])
submission_xgb.index.name = 'PassengerId'
#submission_xgb.to_csv('predictionXGB.csv')

peak = pd.concat([submission_ada, submission_rf, submission_xgb], axis=1)
peak.columns = ['Ada_survived', 'RF_survived', 'XGB_survived']
peak.sample(10)