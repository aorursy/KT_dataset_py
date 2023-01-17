import pandas as pd
train = pd.read_csv('../input/train.csv')
train.head()
test = pd.read_csv('../input/test.csv')
test.head()
train.info()
test.info()
train.isnull().sum()
test.isnull().sum()
print('Age : %d  percent missing values in the training set' % (train['Age'].isna().sum()*100/len(train)))
print('Cabin : %d percent missing values in the training set' % (train['Cabin'].isna().sum()*100/len(train)))
train.Ticket.value_counts()[:10]
train.Cabin.value_counts()[:10]
train.Embarked.value_counts()[:10]
train.Pclass.value_counts()
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
survivors = train[train['Survived']==1]
not_survivors = train[train['Survived']==0]
survived_class = survivors['Pclass'].value_counts().sort_index()
died_class = not_survivors['Pclass'].value_counts().sort_index()
classes = survived_class.index

fig,ax = plt.subplots(figsize=(12,8))
bar_width = 0.35
bar0 = ax.bar(classes,died_class,bar_width,label='Died')
bar1 = ax.bar(classes+bar_width,survived_class,bar_width,label='Survived')
ax.set_title('Survivals by passenger class')
ax.set_xlabel('Passenger Class')
ax.set_xticklabels(classes)
ax.set_xticks(classes + bar_width / 2)
ax.set_ylabel('Count')
ax.legend()
plt.show()
survived = survivors['Sex'].value_counts().sort_index()
died = not_survivors['Sex'].value_counts().sort_index()

fig,ax = plt.subplots(figsize=(12,8))
bar_width = 0.35
index = pd.Index([1,2])
bar0 = ax.bar(index,died,bar_width,label='Died')
bar1 = ax.bar(index+bar_width,survived,bar_width,label='Survived')
ax.set_title('Survival by sex')
ax.set_xlabel('Sex')
ax.set_xticklabels(survived.index)
ax.set_xticks(index + bar_width / 2)
ax.set_ylabel('Count')
ax.legend()
import seaborn as sns
ax = plt.figure(figsize=(12,8))
ax = sns.swarmplot(x='Survived',y='Age',data=train)
ax.set_title('Suvival by Age')
ax.set_xticklabels(['Died','Survived'])
ax.set_xlabel('')
survived_class = survivors['Embarked'].value_counts().sort_index()
died_class = not_survivors['Embarked'].value_counts().sort_index()
index = pd.Index([1,2,3])

fig,ax = plt.subplots(figsize=(12,8))
bar_width = 0.35
bar0 = ax.bar(index,died_class,bar_width,label='Died')
bar1 = ax.bar(index+bar_width,survived_class,bar_width,label='Survived')
ax.set_title('Survivals by port of embarkation')
ax.set_xlabel('Port of embarkation')
ax.set_xticklabels(survived_class.index)
ax.set_xticks(index + bar_width / 2)
ax.set_ylabel('Count')
ax.legend()
plt.show()
s_sib = survivors['SibSp'].value_counts().sort_index()
d_sib = not_survivors['SibSp'].value_counts().sort_index()
s_sib = s_sib.reindex(d_sib.index).fillna(0.0)
fig,ax = plt.subplots(figsize=(12,8))
index = s_sib.index
bar_width = 0.25
bar0 = ax.bar(index,d_sib,bar_width,label='Died')
bar1 = ax.bar(index+bar_width,s_sib,bar_width,label='Survived')
ax.set_title('Survival by number of sibling/spouse on board')
ax.set_xlabel('Number of sibling/spouse')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(s_sib.index)
ax.set_ylabel('Count')
ax.legend()
sibsp = train['SibSp'].value_counts()
sibsp_survival = survivors['SibSp'].value_counts()
survival_pct_by_sibling = sibsp_survival / sibsp * 100

plt.bar(survival_pct_by_sibling.index,survival_pct_by_sibling)
plt.title('Survival % by number of sibling/spouse on board')
plt.xlabel('Number of sibling/spouse on board')
plt.ylabel('Survival %')
s_parch = survivors['Parch'].value_counts().sort_index()
d_parch = not_survivors['Parch'].value_counts().sort_index()
s_parch = s_parch.reindex(d_parch.index).fillna(0.0)

fig,ax = plt.subplots(figsize=(12,8))
index = s_parch.index
bar_width = 0.25
bar0 = ax.bar(index,d_parch,bar_width,label='Died')
bar1 = ax.bar(index+bar_width,s_parch,bar_width,label='Survived')
ax.set_title('Survival by number of parents/children on board')
ax.set_xlabel('Number of parents/children')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(s_parch.index)
ax.set_ylabel('Count')
ax.legend()
train['par_ch'] = np.where(train['Parch']==0,0,1)

survivors = train.loc[train['Survived']==1]
not_survivors = train.loc[train['Survived']==0]
d_parch = not_survivors['par_ch'].value_counts().sort_index()
s_parch = survivors['par_ch'].value_counts().sort_index()

fig, ax = plt.subplots(figsize=(12,8))
index = pd.Index([1,2])
bar_width = 0.4

bar0 = ax.bar(index,d_parch,bar_width,label='Died')
bar1 = ax.bar(index+bar_width,s_parch,bar_width,label='Survived')
ax.set_title('Survival by having parents/children onboard')
ax.set_xlabel('')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(['No parents/children','One or more'])
ax.legend()
ax = plt.figure(figsize=(12,8))
ax = sns.boxplot(x = 'Survived', y = 'Fare', data = train)
ax.set_title('Survival by fare')
ax.set_xticklabels(['Died','Survived'])
ax.set_xlabel('')
plt.figure(figsize=(12,8))
ax = sns.boxplot(x = 'Pclass', y = 'Fare', data = train)
ax.set_title('Fare paid for each Pclass')
cols = ['Survived','Pclass','Sex','Age','SibSp','Parch','Fare']
corr = train[cols].corr()
corr
plt.figure(figsize=(12,8))
plt.title('Age of passengers')
plt.xlabel('Age')
plt.ylabel('Count')
train['Age'].hist(bins=30)
age_mean = train['Age'].mean()
age_median = train['Age'].median()
print('Mean age : ', age_mean, ' Median age : ',age_median)
mean_imp_age = train['Age'].fillna(age_mean)
median_imp_age = train['Age'].fillna(age_median)

plt.figure(figsize=(12,8))
ax1 = plt.subplot(1,2,1)
ax1 = mean_imp_age.hist(bins=30)
plt.title('Mean imputation')
plt.xlabel('Age')

ax2 = plt.subplot(1,2,2)
plt.title('Median imputation')
plt.xlabel('Age')
ax2 = median_imp_age.hist(bins=30)
male_children = train.loc[train['Name'].str.contains('Master')]
male_children['Age'].describe()
train.loc[train['Sex']=='male'].drop(male_children.index).sort_values(by='Age')[:5]
male_children[male_children['Age'].isnull()]
mrs = train['Name'].str.contains('Mrs.')
miss = train['Name'].str.contains('Miss.')
parch = train['Parch']>0
no_parch = train['Parch']==0

fig,axes = plt.subplots(figsize=(12,7))

ax = sns.distplot(train[mrs]['Age'].dropna(),axlabel='Age',label='"Mrs"',kde=False,bins=20)
ax.set_title('Comparison of ages with the titles "Mrs" and "Miss"')
ax = sns.distplot(train[miss]['Age'].dropna(),axlabel='',label="Miss",kde=False,bins=20)
ax.set_yticks([0,5,10,15,20])
plt.legend()

fig,axes = plt.subplots(figsize=(12,7))

ax1 = sns.distplot(train[miss & parch]['Age'].dropna(),axlabel='Age',label='"Miss with parch"',kde=False,bins=20)
ax1 = sns.distplot(train[miss & no_parch]['Age'].dropna(),axlabel='',label='"Miss without parch"',kde=False,bins=20)
ax1.set_title('Comparison of ages of "Miss" with and without "parch onboard')
plt.legend()
from sklearn.base import TransformerMixin

class Age_Imputer(TransformerMixin):
    def __init__(self):
        """
        Imputes ages of passengers in the Titanic, values to be imputed will be dependant 
        on passenger titles and the presence of parents or children on board
        """
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        def value_imp(passengers):
            """
            Imputes an age, based on a weighted random choice derived from the non
            null entries in the subsets of the dataset.
            """
            passengers=passengers.copy()
            # Create 3 year age bins
            bins = np.arange(0,passengers['Age'].max()+3,step=3)
            # Assign each passenger an age bin
            passengers['age_bins'] = pd.cut(passengers['Age'],bins=bins,labels=bins[:-1]+1.5)
            # Count totals of age bins
            count = passengers.groupby('age_bins')['age_bins'].count()
            # Assign each age bin a weight
            weights = count/len(passengers['Age'].dropna())
            null = passengers['Age'].isna()
            # For each missing value, give the passenger an age from the age bins available
            passengers.loc[passengers['Age'].isna(),'Age']=np.random.RandomState(seed=42).choice(weights.index,
                           p=weights.values,size=len(passengers[null]))
            return passengers
        master = X.loc[X['Name'].str.contains('Master')]
        mrs = X.loc[X['Name'].str.contains('Mrs')]
        miss = X.loc[X['Name'].str.contains('Miss')]
        no_parch = X.loc[X['Parch']==0]
        parch = X.loc[X['Parch']!=0]
        miss_no_parch = miss.drop([x for x in miss.index if x in parch.index])
        miss_parch = miss.drop([x for x in miss.index if x in no_parch.index])
        remaining_mr = X.loc[X['Name'].str.contains('Mr. ')]
        # Imputing 'Mrs' first, as in cases where passengers have the titles
        # 'Miss' and 'Mrs', they are married so will be in the older category
        name_cats = [master,mrs,miss_no_parch,miss_parch,remaining_mr]
        for name in name_cats:
            X.loc[name.index] = value_imp(name)
        return X
def value_imp(passengers):
            """
            Imputes an age, based on a weighted random choice derived from the non
            null entries in the subsets of the dataset.
            """
            passengers = passengers.copy()
            bins = np.arange(0,passengers['Age'].max()+3,step=3)
            passengers['age_bins'] = pd.cut(passengers['Age'],bins=bins,labels=bins[:-1]+1.5)
            count = passengers.groupby('age_bins')['age_bins'].count()
            weights = count/len(passengers['Age'].dropna())
            null = passengers['Age'].isna()
            passengers.loc[passengers['Age'].isna(),'Age']=np.random.RandomState(seed=42).choice(weights.index,
                           p=weights.values,size=len(passengers[null]))
            return passengers
train2 = train.copy()
mrs = train2.loc[train2['Name'].str.contains('Mrs.')]
train2.loc[mrs.index] = value_imp(mrs)
fig,axes = plt.subplots(figsize = (12,7))

ax = sns.distplot(train2.loc[mrs.index]['Age'],label='Custom Imputation',kde=False,bins=20)
ax = sns.distplot(train.loc[mrs.index]['Age'].fillna(value=mrs['Age'].mean()),
                  label='Mean Imputation',kde=False,bins=20
                 )
ax.set_title('Comparison of custom and mean imputation')
plt.legend()
train3 = train.copy()
imp = Age_Imputer()
imp.fit_transform(train3)
fig,axes = plt.subplots(figsize = (12,7))

ax = sns.distplot(train.loc[mrs.index]['Age'].dropna(),label='No imputation',kde=False,bins=20)
ax = sns.distplot(train3.loc[mrs.index]['Age'],label='Full custom imputation',kde=False,bins=20)
ax.set_title('Comparison of full custom imputation and no imputation')
plt.legend()
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder

class MultiColumnLabelEncoder(TransformerMixin):
    def __init__(self,columns = None):
        self.columns = columns 

    def fit(self,X,y=None):
        return self

    def transform(self,X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname,col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output
    
class DataFrameSelector(TransformerMixin):
    def __init__(self,attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        return X[self.attribute_names].values

class ValueImputer(TransformerMixin):
    """
    Imputes a fixed value
    """
    def __init__(self,attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        X[self.attribute_names] = X[self.attribute_names].fillna('S')
        return X[self.attribute_names]

numerical_atts = ['Age','SibSp','Parch','Fare']
cat_atts = ['Sex','Pclass','Embarked']

num_pipeline = Pipeline([
    ('imputer', Age_Imputer()),
    ('selector', DataFrameSelector(numerical_atts)),
    ('imp', Imputer(strategy='mean')),
    ('scaler', StandardScaler()),
])

cat_pipeline = Pipeline([
    ('cat_imputer', ValueImputer(cat_atts)),
    ('encoder', MultiColumnLabelEncoder(columns=cat_atts)),
    ('selector', DataFrameSelector(cat_atts)),
])


full_pipeline = FeatureUnion(transformer_list=[
    ('num_pipeline', num_pipeline),
    ('cat_pipeline', cat_pipeline),
])

train_data_prepared = full_pipeline.fit_transform(train)
train_labels = train['Survived']

feature_list = numerical_atts + cat_atts
train_data_prepared.shape
feature_list
from sklearn.linear_model import RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.cross_validation import cross_val_score

classifiers = [RidgeClassifier(),KNeighborsClassifier(),
              SGDClassifier(
                  max_iter=1000),DecisionTreeClassifier(),RandomForestClassifier(),
              MLPClassifier(max_iter=1000)]
names = ['Ridge','KNN','SGD','Decision Tree','Random Forest','MLP']

for name, classifier in zip(names,classifiers):
    classifier.fit(train_data_prepared,train_labels)
    print('Scores for ',name,' : ',cross_val_score(classifier,train_data_prepared,train_labels,cv=3).mean())
    

from sklearn.model_selection import RandomizedSearchCV

mlp = RandomizedSearchCV(MLPClassifier(),cv=3,n_iter=20,param_distributions=(
    {'hidden_layer_sizes':[(100,),(200,),(500,),(1000,)],
    'activation' : ['identity', 'logistic', 'tanh', 'relu'],
    'solver' : ['lbfgs','sgd','adam'],
    'alpha' : np.linspace(0,0.001),
    'max_iter' : [200,500,1000,2000],
    }))
mlp.fit(train_data_prepared,train_labels)

print('Best score : {}'.format(mlp.best_score_))
mlp.best_params_

test_prepared = full_pipeline.fit_transform(test)
best_mlp = mlp.best_estimator_
predictions = best_mlp.predict(test_prepared)
predictions[:20]
submission = pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':predictions})
submission.to_csv('submission.csv',index=False)
submission.head()
forest = RandomForestClassifier()
forest.fit(train_data_prepared,train_labels)
sorted(zip(forest.feature_importances_,feature_list),reverse=True)