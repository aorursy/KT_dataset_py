import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

import statistics



titanic = pd.read_csv("/kaggle/input/titanic/train.csv", sep=",")

titanic_sub = pd.read_csv("/kaggle/input/titanic/test.csv", sep=",")
titanic.info()
titanic_sub.info()
from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(titanic, titanic['Sex']):

    train_set = titanic.loc[train_index]

    test_set = titanic.loc[test_index]



train_set = train_set.reset_index(drop=True)
train_set.info()
train_set['Survived'].value_counts().plot(kind='pie', labels=None, explode=[0,0.1],autopct='%1.1f%%')

plt.legend(['No','Yes'], title="Survived", loc='upper right', bbox_to_anchor=(1.1,1))

plt.axes().set_ylabel('')

plt.title("Titanic survivors")

plt.show()
sns.countplot('Pclass',hue='Survived',data=train_set)

plt.legend(['No','Yes'], title="Survived")

plt.title("Pclass: Survived vs Dead")

plt.show()
Title = list()

for i in range(0, len(train_set['Name'])):

    Title.append(((train_set['Name'][i].rsplit(',', 1)[1]).rsplit('.', 1)[0]).strip())



pd.crosstab([Title], train_set['Survived'], rownames=['Title'])
Title = ['Other' if x!='Mr' and x!='Mrs' and x!='Master' and x!='Miss' else x for x in Title]

pd.crosstab([Title], train_set['Survived'], rownames=['Title'])
Title = pd.Series(Title)

sns.countplot(Title,hue='Survived',data=train_set)

plt.legend(['No','Yes'], title="Survived")

plt.title("Title: Survived vs Dead")

plt.show()
np.nanmean(train_set['Age'].loc[Title[Title == 'Miss'].index])
np.nanmean(train_set['Age'].loc[Title[Title == 'Mrs'].index])
sns.countplot('Sex',hue='Survived',data=train_set)

plt.legend(['No','Yes'], title="Survived")

plt.title("Sex: Survived vs Dead")

plt.show()
plot = sns.catplot(x="Sex", y="Survived", col="Pclass", data=train_set, kind="bar", ci=None, aspect=.6)

plot.set_axis_labels("", "Survival Rate")
sns.countplot('SibSp',hue='Survived',data=train_set)

plt.legend(['No','Yes'], title="Survived", loc="upper right")

plt.title("SibSp: Survived vs Dead")

plt.show()
pd.crosstab(train_set['Survived'], train_set['SibSp'])
sns.countplot('Parch',hue='Survived',data=train_set)

plt.legend(['No','Yes'], title="Survived", loc="upper right")

plt.title("Parch: Survived vs Dead")

plt.show()
pd.crosstab(train_set['Survived'], train_set['Parch'])
Family_members = train_set['SibSp'] + train_set['Parch']



sns.countplot(Family_members,hue='Survived',data=train_set)

plt.legend(['No','Yes'], title="Survived", loc="upper right")

plt.title("Family_members: Survived vs Dead")

plt.show()
pd.crosstab(train_set['Survived'], Family_members, colnames=['Family_members'])
pd.crosstab(train_set['Survived'], [Family_members, train_set['Pclass']], colnames=['Family_members', 'Pclass'])
ticket_base = []

ticket_base.extend(titanic['Ticket'].tolist())

ticket_base.extend(titanic_sub['Ticket'].tolist())



People_on_ticket = list()

tickets = list(train_set['Ticket'])

for i in tickets:

    People_on_ticket.append(ticket_base.count(i))

People_on_ticket = np.int64(People_on_ticket)



sns.countplot(People_on_ticket,hue='Survived',data=train_set)

plt.legend(['No','Yes'], title="Survived", loc="upper right")

plt.title("People_on_ticket: Survived vs Dead")

plt.show()
pd.crosstab(train_set['Survived'], People_on_ticket, colnames=['Ticket_counts'])
np.corrcoef(Family_members, People_on_ticket)
pd.crosstab(Family_members, People_on_ticket, colnames=['People_on_ticket'], rownames=['Family_members'])
sum(People_on_ticket<Family_members+1)
sum(People_on_ticket>Family_members+1)
Is_alone = list()

for i in range(0, len(People_on_ticket)):

    if(People_on_ticket[i] == 1 and Family_members[i] == 0):

        Is_alone.append(1)

    else:

        Is_alone.append(0)

Is_alone = np.int64(Is_alone)



sns.countplot(Is_alone,hue='Survived',data=train_set)

plt.legend(['No','Yes'], title="Survived", loc="upper right")

plt.title("Is_alone: Survived vs Dead")

plt.show()
pd.crosstab(train_set['Survived'], Is_alone, colnames=['Is_alone'])
family_variables = pd.DataFrame({'SibSp': train_set['SibSp'], 'Parch': train_set['Parch'],

                                 'Family_members': Family_members, 'People_on_ticket': People_on_ticket,

                                 'Is_alone': Is_alone})



corr = family_variables.corr()

fig, (ax) = plt.subplots(1, 1, figsize=(10,6))

hm = sns.heatmap(corr, ax=ax, cmap="coolwarm", annot=True, fmt='.2f', linewidths=.05)



fig.subplots_adjust(top=0.93)

fig.suptitle('Travel Companions Correlation Heatmap', fontsize=14, fontweight='bold')
train_set_lone = train_set.loc[np.where(Is_alone == 1)]



pd.crosstab([train_set_lone['Sex'], train_set_lone['Pclass']], train_set_lone['Embarked'])
print("Average age of lone passenger: ", round(np.nanmean(train_set_lone['Age']),0), sep="")

print("Number of missing Age values: ", sum(np.isnan(train_set_lone['Age'])), " - ",

     round((sum(np.isnan(train_set_lone['Age']))*100/len(train_set_lone['Age'])),2), "% of train_set_lone.", sep="")

print("The number of missing values is ", 

      round(sum(np.isnan(train_set_lone['Age']))*100/sum(np.isnan(train_set['Age'])),2),

     "% of entire missing values in Age variable in train_set.", sep="")
train_set.pivot(columns="Pclass", values="Fare").plot.hist(bins=50, histtype='barstacked', alpha=0.5)

plt.title("Fare by Pclass")

plt.show()
outlier_ind = train_set.loc[train_set['Fare']==max(train_set['Fare'])].index

train_set = train_set.drop(outlier_ind)

train_set = train_set.reset_index(drop=True)
train_set.pivot(columns="Pclass", values="Fare").plot.hist(bins=50, histtype='barstacked', alpha=0.5)

plt.title("Fare by Pclass")

plt.show()
print("Minimum price for ticket in first class: ", min(train_set.loc[train_set['Pclass']==1]['Fare']), sep="")
print("Maximum price for ticket in first class: ", max(train_set.loc[train_set['Pclass']==1]['Fare']), sep="")
first_class = train_set.loc[train_set['Pclass']==1]

second_class = train_set.loc[train_set['Pclass']==2]

third_class = train_set.loc[train_set['Pclass']==3]

p_class = list(train_set['Pclass'])

fare = list(train_set['Fare'])

Fare_class = list()



for i in range(0,len(p_class)):

    if (p_class[i] == 1):

        if (fare[i] < statistics.mean(second_class['Fare'])):

            Fare_class.append("cheap")

        else:

            Fare_class.append("normal")

    elif (p_class[i] == 2):

        if (fare[i] < statistics.mean(third_class['Fare'])):

            Fare_class.append("cheap")

        elif (fare[i] > statistics.mean(first_class['Fare'])):

            Fare_class.append("expensive")

        else:

            Fare_class.append("normal")

    else:

        if (fare[i] > statistics.mean(second_class['Fare'])):

            Fare_class.append("expensive")

        else:

            Fare_class.append("normal")



Fare_class = pd.Series(Fare_class)

sns.countplot(Fare_class, hue='Survived', data=train_set)

plt.legend(['No','Yes'], title="Survived", loc="upper right")

plt.title("Fare_class: Survived vs Dead")

plt.show()
pd.crosstab([Fare_class, train_set['Survived']], train_set['Pclass'], rownames=['Fare_class', 'Survived'])
train_set_no_cabins = train_set.loc[np.where(pd.isnull(train_set['Cabin']))]

train_set_cabins = train_set.loc[~train_set.index.isin(train_set_no_cabins.index)]



train_set_cabins['Pclass'].value_counts()
Is_Cabin = list()

cabins = list(train_set['Cabin'])

for i in cabins:

        if (pd.isnull(i)):

            Is_Cabin.append(0)

        else:

            Is_Cabin.append(1)

Is_Cabin = pd.Series(Is_Cabin)



sns.countplot(Is_Cabin,hue='Survived',data=train_set)

plt.legend(['No','Yes'], title="Survived", loc="upper right")

plt.title("Is_Cabin: Survived vs Dead")

plt.show()
train_set2 = train_set

train_set2['Is_Cabin'] = Is_Cabin

plot = sns.catplot(x="Is_Cabin", y="Survived", col="Pclass", data=train_set2, kind="bar", ci=None, aspect=.6)

plot.set_axis_labels("", "Survival Rate")
pd.crosstab([train_set['Pclass'], Is_Cabin], train_set['Survived'], rownames=['Pclass','Is_Cabin'])
sns.countplot('Embarked',hue='Survived',data=train_set)

plt.legend(['No','Yes'], title="Survived", loc="upper right")

plt.title("Embarked: Survived vs Dead")

plt.show()
pd.crosstab(train_set['Survived'], train_set['Embarked'])
f,ax = plt.subplots(2,2,figsize=(20,15))



sns.countplot('Embarked',hue='Sex',data=train_set,ax=ax[0,0])

plt.legend(title="Sex", loc="upper right")

ax[0,0].set_title('Embarked vs Sex')



sns.countplot('Embarked',hue='Pclass',data=train_set,ax=ax[0,1])

plt.legend(title="Pclass", loc="upper right")

ax[0,1].set_title('Embarked vs Pclass')



sns.countplot('Embarked',hue=Title,data=train_set,ax=ax[1,0])

plt.legend(title="Title", loc="upper right")

ax[1,0].set_title('Embarked vs Title')



sns.countplot('Embarked',hue=Family_members,data=train_set,ax=ax[1,1])

plt.legend(title="Family_members", loc="upper right")

ax[1,1].set_title('Embarked vs Family_members')



plt.show()
print("Average age of Southampton passenger:", round(np.nanmean(train_set[train_set['Embarked'] == 'S']['Age'])))

print("Average age of Queenstown passenger:", round(np.nanmean(train_set[train_set['Embarked'] == 'Q']['Age'])))

print("Average age of Cherbourg passenger:", round(np.nanmean(train_set[train_set['Embarked'] == 'C']['Age'])))
train_set_age = train_set

train_set_age['Is_alone'] = np.delete(Is_alone, outlier_ind)

train_set_unknown_age = train_set_age[np.isnan(train_set_age['Age'])]



pd.crosstab([train_set_unknown_age['Pclass'], train_set_unknown_age['Sex']],

            [train_set_unknown_age['Embarked'], train_set_unknown_age['Is_alone']])
train_set_known_age = train_set_age[~np.isnan(train_set_age['Age'])]

train_set_known_age = train_set_known_age[((train_set_known_age['Pclass']==3) & 

                                           (train_set_known_age['Is_alone']==1))|(train_set_known_age['Pclass']==3)]



print("Percentage of single and/or third class passengers with a known age: ", 

      round(len(train_set_known_age.index)*100/len(train_set.index), 2), "%", sep="")
print("Survival ratio for people of an unknown age: ", 

      round((np.sum(train_set_unknown_age['Survived']==1)*100/len(train_set_unknown_age.index)),2), "%", sep="")

print("Survival ratio for people of a known age: ", 

      round((np.sum(train_set_known_age['Survived']==1)*100/len(train_set_known_age.index)),2), "%", sep="")
train_set.pivot(columns="Survived", values="Age").plot.hist(bins=80, histtype='barstacked', alpha=0.5)

plt.legend(['No','Yes'], title="Survived", loc="upper right")

plt.title("Age: Survived vs Dead")

plt.show()
Age_group = list()

age = list(train_set['Age'])



for i in age:

    if (i <= 15):

        Age_group.append('young') 

    elif (i < 50):

        Age_group.append('mid')

    elif (i >= 50):

        Age_group.append('old')

    else:

        Age_group.append('unknown')

Age_group = pd.Series(Age_group)



train_set_age_groups = train_set

train_set_age_groups['Age_group'] = Age_group

plot = sns.catplot(x="Age_group", y="Survived", data=train_set_age_groups, kind="bar", ci=None)

plot.set_axis_labels("", "Survival Rate")

plt.title("Survivability by Age_group")

plt.gca().set_ylim([0,1])
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.impute import SimpleImputer

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import OneHotEncoder

from sklearn.pipeline import FeatureUnion

from sklearn.preprocessing import StandardScaler
class TitleSelector(BaseEstimator, TransformerMixin):

    def __init__(self, attribute_names):

        self._attribute_names = attribute_names

    def fit(self, X, y=None):

        return self

    def get_title(self, obj):

        title = ((obj.rsplit(',', 1)[1]).rsplit('.', 1)[0]).strip()

        return title

    def transform(self, X):

        X.loc[:, 'Title'] = X[self._attribute_names].apply(self.get_title)

        X = X.drop(self._attribute_names, 1)

        return X.values



class TitleCoder(BaseEstimator, TransformerMixin):

    def __init__(self, attribute_names):

        self._attribute_names = attribute_names

    def fit(self, X, y=None):

        return self

    def code_title(self, obj):

        if (obj == 'Mr'):

            title_c = 'Mr'

        elif (obj == 'Mrs'):

            title_c = 'Mrs'

        elif (obj == 'Miss'):

            title_c = 'Miss'

        elif (obj == 'Master'):

            title_c = 'Master'

        else:

            title_c = 'Other'

        return title_c

    def transform(self, X):

        X.loc[:, 'Title_c'] = X[self._attribute_names].apply(self.code_title)

        X = X.drop(self._attribute_names, 1)

        return X.values



class AgeCoder(BaseEstimator, TransformerMixin):

    def __init__(self, attribute_names):

        self._attribute_names = attribute_names

    def fit(self, X, y=None):

        return self

    def code_age(self, obj):

        if (obj <= 15):

            age_c = 'young'

        elif (obj < 50):

            age_c = 'mid'

        elif (obj >= 50):

            age_c = 'old'

        else:

            age_c = 'unknown'

        return age_c

    def transform(self, X):

        X.loc[:, 'Age_c'] = X[self._attribute_names].apply(self.code_age)

        X = X.drop(self._attribute_names, 1)

        return X.values



class SibSpCoder(BaseEstimator, TransformerMixin):

    def __init__(self, attribute_names):

        self._attribute_names = attribute_names

    def fit(self, X, y=None):

        return self

    def code_SibSp(self, obj):

        if (obj == 0):

            SibSp_c = 'zero'

        elif (obj == 1):

            SibSp_c = 'one'

        else:

            SibSp_c = 'more'

        return SibSp_c

    def transform(self, X):

        X.loc[:, 'SibSp_c'] = X[self._attribute_names].apply(self.code_SibSp)

        X = X.drop(self._attribute_names, 1)

        return X.values



class ParchCoder(BaseEstimator, TransformerMixin):

    def __init__(self, attribute_names):

        self._attribute_names = attribute_names

    def fit(self, X, y=None):

        return self

    def code_Parch(self, obj):

        if (obj == 0):

            Parch_c = 'zero'

        elif (obj == 1 or obj == 2):

            Parch_c = 'one/two'

        else:

            Parch_c = 'more'

        return Parch_c

    def transform(self, X):

        X.loc[:, 'Parch_c'] = X[self._attribute_names].apply(self.code_Parch)

        X = X.drop(self._attribute_names, 1)

        return X.values



class Family_members(BaseEstimator, TransformerMixin):

    def __init__(self, attribute_names):

        self._attribute_names = attribute_names

    def fit(self, X, y=None):

        return self

    def transform(self, X):

        X.loc[:, 'Family_members'] = X[self._attribute_names] + X['Parch']

        return X.values



class Family_membersCoder(BaseEstimator, TransformerMixin):

    def __init__(self, attribute_names):

        self._attribute_names = attribute_names

    def fit(self, X, y=None):

        return self

    def code_Family_members(self, obj):

        if (obj == 0):

            Family_members_c = 'zero'

        elif (obj == 1 or obj == 2):

            Family_members_c = 'one/two'

        elif (obj == 3):

            Family_members_c = 'three'

        else:

            Family_members_c = 'more'

        return Family_members_c

    def transform(self, X):

        X.loc[:, 'Family_members_c'] = X[self._attribute_names].apply(self.code_Family_members)

        X = X.drop(self._attribute_names, 1)

        return X.values



class People_on_ticket(BaseEstimator, TransformerMixin):

    def __init__(self, attribute_names):

        self._attribute_names = attribute_names

    def fit(self, X, y=None):

        return self

    def transform(self, X):

        counts = list()

        tickets = list(X[self._attribute_names])

        for i in tickets:

            counts.append(ticket_base.count(i))

        #is_ticket_uni = [x if x == 1 else 0 for x in counts]

        X.loc[:, 'People_on_ticket'] = counts

        return X.values

    

class People_on_ticketCoder(BaseEstimator, TransformerMixin):

    def __init__(self, attribute_names):

        self._attribute_names = attribute_names

    def fit(self, X, y=None):

        return self

    def code_People_on_ticket(self, obj):

        if (obj == 1):

            People_on_ticket_c = 'one'

        elif (obj == 2):

            People_on_ticket_c = 'two'

        elif (obj == 3):

            People_on_ticket_c = 'three'

        elif (obj == 4):

            People_on_ticket_c = 'four' 

        else:

            People_on_ticket_c = 'more'

        return People_on_ticket_c

    def transform(self, X):

        X.loc[:, 'People_on_ticket_c'] = X[self._attribute_names].apply(self.code_People_on_ticket)

        X = X.drop(self._attribute_names, 1)

        return X.values



class Is_Alone(BaseEstimator, TransformerMixin):

    def __init__(self, attribute_names):

        self._attribute_names = attribute_names

    def fit(self, X, y=None):

        return self

    def transform(self, X):

        family_members = list(X['Family_members'])

        ppl_on_ticket = list(X[self._attribute_names])

        is_alone = list()

        for i in range(0,len(ppl_on_ticket)):

            if (ppl_on_ticket[i] == 1 and family_members[i] == 0):

                is_alone.append(1)

            else:

                is_alone.append(0)

        X.loc[:, 'Is_alone'] = is_alone

        return X.values

    

class Is_Cabin(BaseEstimator, TransformerMixin):

    def __init__(self, attribute_names):

        self._attribute_names = attribute_names

    def fit(self, X, y=None):

        return self

    def code_Cabin(self, obj):

        if (pd.isnull(obj)):

            is_cabin = 0

        else:

            is_cabin = 1

        return is_cabin

    def transform(self, X):

        X.loc[:, 'Is_cabin'] = X[self._attribute_names].apply(self.code_Cabin)

        X = X.drop(self._attribute_names, 1)

        return X.values



class FareClass(BaseEstimator, TransformerMixin):

    def __init__(self, attribute_names):

        self._attribute_names = attribute_names

    def fit(self, X, y=None):

        return self

    def transform(self, X):

        first_class = X.loc[X['Pclass']==1]

        second_class = X.loc[X['Pclass']==2]

        third_class = X.loc[X['Pclass']==3]

        p_class = list(X['Pclass'])

        fare = list(X[self._attribute_names])

        fare_class = list()

        for i in range(0,len(p_class)):

            if (p_class[i] == 1):

                if (fare[i] < statistics.mean(second_class['Fare'])):

                    fare_class.append("cheap")

                else:

                    fare_class.append("normal")

            elif (p_class[i] == 2):

                if (fare[i] < statistics.mean(third_class['Fare'])):

                    fare_class.append("cheap")

                elif (fare[i] > statistics.mean(first_class['Fare'])):

                    fare_class.append("expensive")

                else:

                    fare_class.append("normal")

            else:

                if (fare[i] > statistics.mean(second_class['Fare'])):

                    fare_class.append("expensive")

                else:

                    fare_class.append("normal")

        X.loc[:, 'Fare_class'] = fare_class

        return X.values
#Pipelines for creating/transforming variables.

name = 'Name'

name_pipeline = Pipeline(steps=[

    ('get_title', TitleSelector(name))

])

    

title = 'Title'  

title_pipeline = Pipeline(steps=[

    ('code_title', TitleCoder(title))

]) 



age = 'Age'

age_pipeline = Pipeline(steps=[

    ('code_age', AgeCoder(age))

]) 



sibsp = 'SibSp'

sibsp_pipeline = Pipeline(steps=[

    ('code_sibsp', SibSpCoder(sibsp))#,

])



sibsp_pipeline2 = Pipeline(steps=[

    ('create_family_members', Family_members(sibsp))

])

    

parch = 'Parch'

parch_pipeline = Pipeline(steps=[

    ('code_parch', ParchCoder(parch))

])



family_members = 'Family_members'

family_members_pipeline = Pipeline(steps=[

    ('code_family_members', Family_membersCoder(family_members))

])



ticket = 'Ticket'

ticket_pipeline = Pipeline(steps=[

    ('create_People_on_ticket', People_on_ticket(ticket)),

])



people_on_ticket = 'People_on_ticket'

ticket_pipeline2 = Pipeline(steps=[

    ('code_People_on_ticket', People_on_ticketCoder(people_on_ticket)),

])



is_alone_pipeline = Pipeline(steps=[

    ('create_is_alone', Is_Alone(people_on_ticket)),

])



cabin = 'Cabin'

cabin_pipeline = Pipeline(steps=[

    ('code_cabin', Is_Cabin(cabin)),

])

    

fare = 'Fare'

fare_pipeline = Pipeline(steps=[

    ('fare_class', FareClass(fare)),

])



#Categorical variables - they need One-Hot Encoding. Missing values are inputed with most frequent values

#(a maximum of a few observations may be missing for these variables). All variables are standardized.

attribs1 = ['Pclass', 'Title_c', 'Sex', 'Age_c',# 'SibSp_c', 'Parch_c',

            #'People_on_ticket_c',

            'Family_members_c', 'Embarked', #'Fare_class',

            ]

attribs1_pipeline = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy="most_frequent")),

    ('encoder', OneHotEncoder(sparse=False)),

    ('standarizer', StandardScaler())

])



#Numerical variables - no need for One-Hot Encoding. Missing values are inputed with median

#(a maximum of a few observations may be missing for these variables). All variables are standardized.

attribs2 = ['SibSp', 'Parch', #'Family_members', 'People_on_ticket', 'Is_alone',

            'Fare', 'Is_cabin']

attribs2_pipeline = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy="median")),

    ('standarizer', StandardScaler())

])



#Pipeline dropping unused variables and joining both pipelines from above: one for categorical variables

#and one for numerical variables.

preprocessor = ColumnTransformer(

    remainder = 'drop',

    transformers=[

        ('first', attribs1_pipeline, attribs1),

        ('second', attribs2_pipeline, attribs2)

])



#Joining all created pipelines into one.

full_pipeline = FeatureUnion([

    ('name_pipeline', name_pipeline),

    ('title_pipeline', title_pipeline),

    ('age_pipeline', age_pipeline),

    #('sibsp_pipeline', sibsp_pipeline),

    ('sibsp_pipeline2', sibsp_pipeline2),

    #('parch_pipeline', parch_pipeline),

    ('family_members_pipeline', family_members_pipeline),

    #('ticket_pipeline', ticket_pipeline),

    #('is_alone_pipeline', is_alone_pipeline),

    #('ticket_pipeline2', ticket_pipeline2),

    ('cabin_pipeline', cabin_pipeline),

    #('fare_pipeline', fare_pipeline),

    ('preprocessor', preprocessor)

])



#Pipeline which will definitely drop unused columns after usage of full_pipeline 

#(one usage of preprocessor is not enough).

#Variables won't be transformed again in this pipeline, I checked that.

full_pipeline2 = FeatureUnion([

    ('preprocessor', preprocessor)

])
train_set_prepared = full_pipeline.fit_transform(train_set)

train_set_prepared = full_pipeline2.fit_transform(train_set)

X_train_prepared = train_set_prepared

y_train_prepared = train_set['Survived']



test_set_prepared = full_pipeline.fit_transform(test_set)

test_set_prepared = full_pipeline2.fit_transform(test_set)

X_test_prepared = test_set_prepared

y_test_prepared = test_set['Survived']
from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score



rf_model = RandomForestClassifier(random_state=42)



params_grid = [

        {'n_estimators': [100, 200, 300, 400, 500],

         'criterion': ['gini', 'entropy'],

         'min_samples_split': [2, 3, 4, 5],

         'max_features': ['auto', 'log2', None],

         'bootstrap': ['True', 'False']}

]



grid_search = GridSearchCV(rf_model, params_grid, cv=5, scoring="accuracy", n_jobs=1)

grid_search.fit(X_train_prepared, y_train_prepared)

grid_search.best_params_
params_grid2 = [

        {'n_estimators': [120, 140, 160, 180, 200, 220, 240, 260, 280],

         'criterion': ['gini', 'entropy'],

         'min_samples_split': [4, 5],

         'max_features': ['auto', None],

         'bootstrap': ['True']}

]



grid_search2 = GridSearchCV(rf_model, params_grid2, cv=5, scoring="accuracy", n_jobs=1)

grid_search2.fit(X_train_prepared, y_train_prepared)



#Function for getting the importance of top k features 

def indices_of_top_k(arr, k):

    return np.sort(np.argpartition(np.array(arr), -k)[-k:])



#Function calculating scores for each restricted data set

def fs_calculate_results():

    train_prediction = list()

    test_prediction = list()

    knn_model = KNeighborsClassifier()

    feature_importances = grid_search2.best_estimator_.feature_importances_

    for i in range(1,26):

       indices_of_top = indices_of_top_k(feature_importances, i)

       X_train_restricted = X_train_prepared[:, indices_of_top]

       X_test_restricted = X_test_prepared[:, indices_of_top]

    #The square of the number of observations in train_set is close to 27, so the number

    #of neighbours will oscillate around this number

       params_grid_fs = [

        {'n_neighbors': [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35],

         'weights': ['uniform', 'distance'],

         'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],

         'metric': ['minkowski', 'euclidean', 'manhattan']}

       ]

       #print(i)

       grid_search_fs = GridSearchCV(knn_model, params_grid_fs, cv=5, scoring="accuracy", n_jobs=1)

       grid_search_fs.fit(X_train_restricted, y_train_prepared)

       train_prediction.append(grid_search_fs.best_score_)

       #print("Train_set: ", grid_search_fs.best_score_)

       knn_final_model = grid_search_fs.best_estimator_

       knn_predictions = knn_final_model.predict(X_test_restricted)

       test_prediction.append(accuracy_score(y_test_prepared, knn_predictions))

       #print("Test_set: ", accuracy_score(y_test_prepared, knn_predictions))

    return train_prediction, test_prediction



train_pred, test_pred = fs_calculate_results()

fs_results = pd.DataFrame({'train_set': train_pred, 'test_set': test_pred})

fs_results['train-test_difference'] = fs_results['train_set'] - fs_results['test_set']

fs_results.index += 1

fs_results = fs_results.sort_values('train-test_difference')

fs_results
feature_importances = grid_search2.best_estimator_.feature_importances_

indices_of_top = indices_of_top_k(feature_importances, 25)

X_train_prepared = X_train_prepared[:, indices_of_top]

X_test_prepared = X_test_prepared[:, indices_of_top]



knn_model = KNeighborsClassifier()

params_grid = [

        {'n_neighbors': [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35],

         'weights': ['uniform', 'distance'],

         'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],

         'metric': ['minkowski', 'euclidean', 'manhattan']}

]



grid_search = GridSearchCV(knn_model, params_grid, cv=5, scoring="accuracy", n_jobs=1)

grid_search.fit(X_train_prepared, y_train_prepared)



grid_search.best_params_
print("Training dataset accuracy: ", grid_search.best_score_, sep="")

knn_model = grid_search.best_estimator_

knn_predictions = knn_model.predict(X_test_prepared)

print("Test dataset accuracy: ", accuracy_score(y_test_prepared, knn_predictions), sep="")
KNN_models = pd.DataFrame({'params': list(grid_search.cv_results_["params"]),

                    'results': list(grid_search.cv_results_["mean_test_score"])})

KNN_models = KNN_models.sort_values('results', ascending=False)

KNN_models = KNN_models.head(20)

KNN_models.style.set_properties(subset=['params'], **{'width-min': '300px'})
knn_final_model = KNeighborsClassifier(algorithm='auto', metric='manhattan', n_neighbors=25, weights='uniform')

knn_final_model.fit(X_train_prepared, y_train_prepared)
titanic_sub_prepared = full_pipeline.fit_transform(titanic_sub)

titanic_sub_prepared = full_pipeline2.fit_transform(titanic_sub)



titanic_sub_prepared = titanic_sub_prepared[:, indices_of_top]



titanic_sub_predictions = knn_final_model.predict(titanic_sub_prepared)

submission = pd.DataFrame({'PassengerId':titanic_sub['PassengerId'], 'Survived':titanic_sub_predictions})

submission.to_csv('submission.csv', sep=',', index=False)