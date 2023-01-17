import pandas as pd
import numpy as np

mytrain = pd.read_csv('../input/machine-learning-on-titanic-data-set/train.csv', index_col=0)     # index: PassengerId
print(mytrain.shape)
mytrain.info()
mytest  = pd.read_csv('../input/machine-learning-on-titanic-data-set/test.csv', index_col=0)     # index: PassengerId

print(mytest.shape)
mytest.info()
combined = pd.concat([mytrain, mytest], axis=0)
combined.info()
print(combined.index)
print(combined.columns)
print(combined.columns[0])
combined.describe(include='all').T
combined.describe(include=['O']).T         # duplicates exist in String columns
print(combined.head(5))
print(combined.tail(5))
combined.isna().sum()
combined.isnull().sum()
combined[combined['Age'].isna() == True]
combined[combined['Age'].isna() == True].head(5)
combined.iloc[5]
combined[combined['Cabin'].isna() == True]
combined[combined['Embarked'].isna() == True]
combined[combined['Fare'].isna() == True]
combined[(combined['Cabin'].isna() == True) &
        (combined['Age'].isna() == True)]
import seaborn as sns            # Why sns?  It's a reference to The West Wing
import matplotlib.pyplot as plt  # seaborn is based on matplotlib
sns.set(color_codes=True)        # adds a nice background to the graphs
%matplotlib inline               
# tells python to actually display the graphs
pd.value_counts(combined['Survived'])   # 549 of 891 is 61.6% in Train did not survive
sns.catplot(y='Survived', kind='count', aspect=2, data=combined);
combined['Pclass'].nunique()
combined['Pclass'].unique()
pd.value_counts(combined['Pclass'])
sns.distplot(combined['Pclass']);
# we can turn the kde off and put a tic mark along the x-axis for every data point with rug
sns.distplot(combined['Pclass'], kde=False, rug=True);
pd.value_counts(combined['Sex'])     # 843 of 1309 is 64.4% are male in (Train+Test)
ax=sns.countplot('Sex', data=combined)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha='right');
pd.value_counts(combined['Age'])
sns.boxplot(combined['Age']);
#combined['Age'].boxplot(vert=0)
fig_dims = (10,5)
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=fig_dims)
sns.distplot(combined['Age'], kde=True, ax=axs[0], rug=True)
sns.boxplot(combined['Age'], ax=axs[1])
plt.show();
def univariate_float(float_col):
    print(combined[float_col].mean())
    sns.boxplot(combined[float_col], palette = 'pastel')
    plt.show();
    
combined_float = combined.select_dtypes(include = ['float64'])
listcols = list(combined_float.columns.values)
for x in listcols:
    univariate_float(x)
pd.value_counts(combined['SibSp'])
pd.value_counts(combined['Parch'])
combined[combined['Fare'] == combined['Fare'].max()]
combined[['Ticket', 'Name']][combined['Fare'] == combined['Fare'].max()]
combined[combined['Fare'] == combined['Fare'].max()][['Ticket', 'Name']]
pd.value_counts(combined['Fare'])
sns.boxplot(combined['Fare']);
pd.value_counts(combined['Embarked'])
combined['NameLength'] = 0

for i in range(1, len(combined)+1):
    combined['NameLength'][i] = len(combined.iloc[i-1,2])
pd.crosstab(combined['Survived'], combined['Sex'])      # most females Survived
combined.pivot_table(index=['Survived'], values=['Fare'], aggfunc='mean').round(0)
combined[['Survived', 'Fare']].groupby(combined['Survived']).mean()
combined[['Survived', 'Fare']].groupby('Survived').mean()
sns.stripplot(combined['Survived'], combined['Fare'], jitter=True);     # more Survivors among higher Fare
sns.swarmplot(combined['Survived'], combined['Fare']);
pd.crosstab(combined['Survived'], combined['Pclass'])      # more Survivors from Class = 1 (or those who were rescued)
sns.stripplot(combined['Pclass'], combined['Fare'], jitter=True);     # highest Fares in Class 1
sns.boxplot(combined['Pclass'], combined['Fare'], hue=combined['Survived']);
sns.catplot(x="Survived",
               y = "Fare",
               hue="Pclass", 
               col="Sex", 
               data=combined, 
               kind="swarm");
combined.pivot_table(index=['Survived'], values=['Age'], aggfunc='mean')
sns.stripplot(combined['Survived'], combined['Age'], jitter=True);       # Age not visibly correlated with Survival
sns.stripplot(combined['Sex'], combined['Fare'], jitter=True);           # no correlation between Fare and Sex
combined.pivot_table(index=['Pclass'], values=['Fare'], aggfunc='mean' )
sns.stripplot(combined['Pclass'], combined['Fare'], jitter=True);     # higher Fare in Class 1, as expected
combined.pivot_table(index=['Embarked'], values=['Fare'], aggfunc='mean' )
sns.stripplot(combined['Embarked'], combined['Fare'], jitter=True);    # wealthy passengers boarded from 'C'
combined.pivot_table(index=['Pclass'], values=['SibSp'], aggfunc='mean' )     # mean(SibSp) = 0.5
combined.pivot_table(index=['Pclass'], values=['Parch'], aggfunc='mean' )     # mean(Parch) = 0.39
combined.pivot_table(index=['Sex'], values=['SibSp'], aggfunc='mean' )     # mean(SibSp) = 0.5
combined.pivot_table(index=['Sex'], values=['Parch'], aggfunc='mean' )     # mean(Parch) = 0.39
combined.pivot_table(index=['Pclass', 'Sex'], values=['Age'], aggfunc='mean' )     # mean(Age) = 29.9
pd.crosstab(combined['Survived'], combined['SibSp'])
pd.crosstab(combined['Survived'], combined['Parch'])
sns.jointplot(combined['Age'], combined['Fare']);         # doesn't appear that an increase in Age goes together with increase in Fare
sns.scatterplot(x='Age', y='Fare', data=combined);
sns.jointplot(combined['Age'], combined['Fare'], kind="hex");      # shows record concentration as well
sns.jointplot(combined['Age'], combined['Fare'], kind="kde");       # Kernel Density Estimation is alternative to Hex Bin plot above
sns.pairplot(combined[['Age', 'Fare']], corner=True);     # Scatterplot + Histogram
sns.stripplot(combined['Survived'], combined['NameLength'], jitter=True);
#combined['Survived'] = combined['Survived'].astype('category')
#combined['Pclass']   = combined['Pclass'].astype('category')
combined.info()
combined._get_numeric_data()
from scipy import stats
stats.normaltest(combined['Age'], nan_policy = 'omit')       # p-value < 0.05; reject H0; not Gaussian
stats.normaltest(combined['Fare'], nan_policy = 'omit')
stats.boxcox(combined['Age'])
from scipy.stats import boxcox
from scipy.special import inv_boxcox

Age_n,fitted_lambda = boxcox(combined['Age'],lmbda=None)
fitted_lambda
#Age_n = inv_boxcox(y,fitted_lambda)
stats.normaltest(Age_n, nan_policy = 'omit') 
# Fare_n,fitted_lambda = boxcox(combined['Fare'],lmbda=None)
# stats.normaltest(Fare_n, nan_policy = 'omit') 
from sklearn import preprocessing

centered_scaled_Age = preprocessing.scale(combined['Age'])
centered_scaled_Age
pd.DataFrame(centered_scaled_Age).mean(skipna=True)
#Age_in = np.array(mytrain['Age']).reshape(-1,1)
#Age_n = preprocessing.normalize(Age_in)
#stats.normaltest(Age_n, nan_policy = 'omit') 
from scipy.stats import skew
skew(combined['Fare'])           # +ve, meaning tilted towards left.
                                 # this means more records at lower fares. also seen above.
# Chi-square test among categorical variables
from scipy.stats import chisquare
Survived_Pclass = pd.crosstab(combined['Survived'], combined['Pclass'])

chisquare(Survived_Pclass, axis = None)   # p-value < 0.05. reject H0. not independent.

# import statsmodels.api as sm
# from scipy.stats import chi2_contingency

#table = sm.stats.Table.from_data(mytrain[['Survived', 'Pclass']])
#stat, p, dof, expected = scipy.stats.chi2_contingency(table)
Survived_Sex = pd.crosstab(combined['Survived'], combined['Sex'])

chisquare(Survived_Sex, axis = None)      # p-value < 0.05. reject H0. not independent.
Survived_Embarked = pd.crosstab(combined['Survived'], combined['Embarked'])

chisquare(Survived_Embarked, axis = None) # p-value < 0.05. reject H0. not independent.
Survived_SibSp = pd.crosstab(combined['Survived'], combined['SibSp'])

chisquare(Survived_SibSp, axis = None) # p-value < 0.05. reject H0. not independent.
Survived_SibSp
Survived_Parch = pd.crosstab(combined['Survived'], combined['Parch'])

chisquare(Survived_Parch, axis = None) # p-value < 0.05. reject H0. not independent.
Survived_Parch
cor = combined[['Age', 'Fare']].corr()
plt.figure(figsize=(5,5))
sns.set(font_scale=1.5)
sns.heatmap(cor, annot=True, cmap='plasma', vmin=-1, vmax=1)
plt.show()
plt.figure(figsize=(5,5))
sns.set(font_scale=1.5)
sns.heatmap(round(cor,1), annot=True, mask=(np.triu(cor)), vmin=-1, vmax=1)
plt.show();
plt.figure(figsize=(5,5))
sns.set(font_scale=1.5)
sns.heatmap(round(cor,1), annot=True, mask=(np.triu(cor,+1)), vmin=-1, vmax=1)
plt.show();
combined['SibSp_new'] = 0

for i in range(1, len(combined)+1):
    if (combined['SibSp'][i] == 0):
        combined['SibSp_new'][i] = 0
    elif (combined['SibSp'][i] == 1):
        combined['SibSp_new'][i] = 0
    elif (combined['SibSp'][i] == 2):
        combined['SibSp_new'][i] = 0
    else:
        combined['SibSp_new'][i] = 1

pd.value_counts(combined['SibSp_new'])
#combined['Parch_new'] = 0

#for i in range(1, len(combined)+1):
#    if (combined['Parch'][i] == 0):
#        combined['Parch_new'][i] = 0
#    elif (combined['Parch'][i] == 1):
#        combined['Parch_new'][i] = 1
#    elif (combined['Parch'][i] == 2):
#        combined['Parch_new'][i] = 1
#    elif (combined['Parch'][i] == 3):
#        combined['Parch_new'][i] = 1
#    else:
#        combined['Parch_new'][i] = 2
#
#pd.value_counts(combined['Parch_new'])
#pd.crosstab(combined['Survived'], combined['Parch_new'])
Survived_SibSp_new = pd.crosstab(combined['Survived'], combined['SibSp_new'])

chisquare(Survived_SibSp_new, axis = None) # p-value < 0.05. reject H0. not independent.
pd.crosstab(combined['Survived'], combined['SibSp_new'])
#Survived_Parch_new = pd.crosstab(combined['Survived'], combined['Parch_new'])

#chisquare(Survived_Parch_new, axis = None) # p-value < 0.05. reject H0. not independent.
#pd.crosstab(combined['Survived'], combined['Parch_new'])
combined['Ticket_Len'] = combined['Ticket'].apply(lambda x: len(x))
print(combined['Ticket_Len'].value_counts())
pd.crosstab(combined['Survived'], combined['Ticket_Len'])
# skip column not important for classification
#combined['Ticket_Lett'] = combined['Ticket'].apply(lambda x: str(x)[0])
#print(combined['Ticket_Lett'].value_counts())
#pd.crosstab(combined['Survived'], combined['Ticket_Lett'])
combined['Embarked'].fillna(method='backfill', inplace=True)
pd.value_counts(combined['Embarked'])                        # filled with 1S, 1C. next available valid value.
embarked_mapping = {'S': 0, 'C': 1, 'Q': 2}
combined.replace({'Embarked': embarked_mapping}, inplace=True)
print(pd.value_counts(combined['Embarked']))
combined['Fare'].fillna(method='backfill', inplace=True)
pd.value_counts(combined['Fare'])                        
combined['Fare'].mean()
combined['Fare'].fillna(combined['Fare'].mean(), inplace=True)
#combined['FareBand'] = pd.cut(combined['Fare'], 10)
#pd.crosstab(combined['Survived'], combined['FareBand'])
def remove_outlier(col):
    sorted(col)
    Q1, Q3 = np.percentile(col, [25,75])
    IQR=Q3-Q1
    lower_range = Q1 - (1.5*IQR)
    upper_range = Q3 + (1.5*IQR)
    return lower_range, upper_range

lr, ur = remove_outlier(combined['Fare'])
combined['Fare'] = np.where(combined['Fare'] > ur, ur, combined['Fare'])
combined['Fare'] = np.where(combined['Fare'] < lr, lr, combined['Fare'])

sns.boxplot(combined['Fare']);
combined['Title'] = combined['Name'].str.extract('([A-Za-z]+)\.', expand=True)
print(pd.value_counts(combined['Title']))

# combine Titles
mapping = {'Mlle': 'Miss', 'Major': 'Mr', 'Col': 'Mr', 'Sir': 'Mr',
           'Don': 'Mr', 'Mme': 'Mrs', 'Jonkheer': 'Mr', 'Lady': 'Mrs',
           'Capt': 'Mr', 'Countess': 'Mrs', 'Ms': 'Miss', 'Dona': 'Mrs'}
combined.replace({'Title': mapping}, inplace=True)
print(pd.value_counts(combined['Title']))
Title_ages = dict(combined.groupby('Title')['Age'].median())
print(Title_ages)

combined['age_med'] = combined['Title'].apply(lambda x: Title_ages[x])
print(combined['age_med'].isna().sum())

combined['Age'].fillna(combined['age_med'], inplace=True)
print(combined['Age'].describe())                          # mean still around same (29.9 earlier)

del combined['age_med']
#combined['AgeBand'] = pd.cut(combined['Age'], 8)
#pd.crosstab(combined['Survived'], combined['AgeBand'])
#combined['NameLengthBand'] = pd.cut(combined['NameLength'], 3)
#pd.crosstab(combined['Survived'], combined['NameLengthBand'])
combined['Family_Size'] = combined['SibSp'] + combined['Parch']
pd.crosstab(combined['Survived'], combined['Family_Size'])
#combined['Is_Alone'] = 0

#for i in range(1, len(combined)+1):
#    if (combined['Family_Size'][i] == 1):
#        combined['Is_Alone'][i] = 1
combined['Age*Class'] = combined['Age'] * combined['Pclass']
pd.crosstab(combined['Survived'], combined['Age*Class'])
# combined['FareBand'] = pd.qcut(combined['Fare'], 4)
# combined[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)
#combined.loc[ combined['Fare'] <= 7.9, 'Fare'] = 0
#combined.loc[(combined['Fare'] > 7.9)    & (combined['Fare'] <= 14.454), 'Fare'] = 1
#combined.loc[(combined['Fare'] > 14.454) & (combined['Fare'] <= 31), 'Fare']   = 2
#combined.loc[ combined['Fare'] > 31, 'Fare'] = 3

#combined['Fare'] = combined['Fare'].astype(int)
#combined['Fare'].describe()

#combined = combined.drop(['FareBand'], axis=1)
combined['Sex_num'] = 0

for i in range(1, len(combined)+1):
    if (combined['Sex'][i] == 'male'):
        combined['Sex_num'][i] = 1

pd.value_counts(combined['Sex_num'])
del combined['Sex']
combined.head(2).T
#cols_1hot = ['Ticket_Lett', 'FareBand', 'Title', 'AgeBand', 'NameLengthBand']
cols_1hot = ['Title']

for i in cols_1hot:
    df_1hot = pd.get_dummies(combined[i], prefix=i, prefix_sep='_')
    combined = pd.concat([combined, df_1hot], axis=1)

combined.info()
combined.drop(columns=cols_1hot, inplace=True)
combined.info()
train_new = combined[combined['Survived'].isna() == False]
train_new.drop(columns=['Name', 'Ticket', 'Cabin'], inplace=True)
train_new.info()
test_new = combined[combined['Survived'].isna() == True]
test_new.drop(columns=['Name', 'Ticket', 'Cabin', 'Survived'], inplace=True)
test_new.info()
test_new.index
pd.value_counts(train_new['Survived'])
from sklearn.utils import resample

# Separate majority and minority classes
df_majority = train_new[train_new.Survived==0]
df_minority = train_new[train_new.Survived==1]
 
# Upsample minority class
df_minority_upsampled = resample(df_minority, 
                                 replace=True,     # sample with replacement
                                 n_samples=549,    # to match majority class
                                 random_state=123) # reproducible results
 
# Combine majority class with upsampled minority class
df_upsampled = pd.concat([df_majority, df_minority_upsampled])
 
# Display new class counts
df_upsampled.Survived.value_counts()
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(max_features='auto', oob_score=True, random_state=123, n_jobs=-1)

param_grid = { "criterion" : ["gini", "entropy"], "min_samples_leaf" : [2, 3, 4], "min_samples_split" : [3, 5, 7, 9], "n_estimators": [100]}

gs = GridSearchCV(estimator=rf, param_grid=param_grid, scoring='accuracy', cv=10, n_jobs=-1, verbose=2)

gs = gs.fit(df_upsampled.iloc[:, 1:], df_upsampled.iloc[:, 0])
print(gs.best_score_)
print(gs.best_params_)
#print(gs.cv_results_)
#help(RandomForestClassifier)
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(criterion='entropy', 
                             n_estimators=101,                  # number of trees
                             min_samples_split=3,              # minimum number of samples required to split an internal node
                             min_samples_leaf=2,                # minimum number of samples required in a leaf node
                             max_features='auto',
                             oob_score=True,
                             max_depth = 20,
                             random_state=123,
                             n_jobs=-1)                         # all processors in parallel
rf.fit(df_upsampled.iloc[:, 1:], df_upsampled.iloc[:, 0])
print("%.4f" % rf.oob_score_)
pd.concat((pd.DataFrame(df_upsampled.iloc[:, 1:].columns, columns = ['variable']), 
           pd.DataFrame(rf.feature_importances_, columns = ['importance'])), 
          axis = 1).sort_values(by='importance', ascending = False)[:20]
predictions = rf.predict(test_new)
predictions = pd.DataFrame(predictions, columns=['Survived'])

predictions = pd.concat((pd.DataFrame(test_new.index), predictions.astype('int64')), axis = 1)
predictions

predictions.to_csv('Titanic_predict.csv', sep=",", index = False)