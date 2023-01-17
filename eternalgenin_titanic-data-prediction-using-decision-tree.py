pwd
# Running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt



%matplotlib inline

sns.set_style('whitegrid')
train = pd.read_csv('/kaggle/input/titanic/train.csv')
train.head()
## Creating the first letter of ticket as a variable



train['Ticket_First'] = train['Ticket'].apply(lambda x:x.split()[0][:1])



train['Ticket_First'].unique()
## Extracting the salutation from name as a variable



train['Salute'] = train['Name'].apply(lambda x:x.split()[1])
pd.value_counts(train['Salute']).head()
## Grouping the minor salutations as others



def Salute_group(col):

    

    if col[0] in ['Mr.', 'Miss.', 'Mrs.', 'Master.']:

        return col[0]

    else:

        return 'Others'
train['Salute_Grp'] = train[['Salute']].apply(Salute_group, axis =1)
sns.set_style('whitegrid')

sns.countplot(x='Salute_Grp', data = train, hue = 'Survived')

train.info()
sns.countplot(x='Survived', data = train, hue = 'Pclass')



##More number of people died than survived

##More females survived than males. Also, the probability of a male surviving was low.

##Probability of surviving if they were third class was low



sns.countplot(x='Parch', data = train)
sns.countplot(x='SibSp', data = train )
##Missing Values



sns.heatmap(train.isnull())



##Age and Cabin have majorly got issues
# Treat Age



train['Age'].median()

train['Age'].mean()



## Options 

    # 1 Can just add median age ie 28 here.

    # 2 Can just add mean age ie 29.69 ~= 30 here.

    # 3 Use some other feature variable to adjudge the age. ex. Pclass - Old people should be first class. 

    # 4 Use combination of other feature variables to adjudge the age. Pclass X Sex



    

# Option1

train['Age_Med'] = train['Age'].fillna(train['Age'].median())



sns.distplot(train['Age_Med'])
# Option2

train['Age_Mean'] = train['Age'].fillna(round(train['Age'].mean()))



sns.distplot(train['Age_Mean'])
# Defining a function to impute median (using median since the data is skewed) for each Pclass.



def age_Pclass(cols):

    age = cols[0]

    Pclass = cols[1]

    

    if pd.isnull(age) == True:

        

        if Pclass == 1:

            return 37

        elif Pclass == 2:

            return 29

        else:

            return 24

    else:

        return age
# Since, the above functions operates for a vector cols, and not on a dataframe. We'll use 'apply' function to apply 

# the function to the entire rows. Use axis = 1 to apply across rows.



train['Age_Pclass'] = train[['Age', 'Pclass']].apply(age_Pclass, axis = 1)



sns.distplot(train['Age_Pclass'])
# Option4

#train[pd.isnull(train['Age'])]



sns.boxplot (x='Sex', y='Age', data = train, hue = 'Pclass')



# Age is not a huge differentiator here but going ahead to see difference in the model results



# Calculating medians



PclassXSex_med = train[['Sex','Age','Pclass']].groupby(['Sex','Pclass']).median()

PclassXSex_med
# Defining a function to impute median (using median since the data is skewed) for each PclassXSex.



## MUCH MORE EFFICIENT WAY TO WRITING FUNCTION THAN BEFORE ##



def age_PclassSex(cols):

    age = cols[0]

    Pclass = cols[1]

    Sex = cols[2]

    

    if pd.isnull(age) == True:

        return PclassXSex_med.loc[Sex].loc[Pclass][0]

    else:

        return age
train['Age_PclXSex'] = train[['Age', 'Pclass', 'Sex']].apply(age_PclassSex, axis = 1)



fig, ax =plt.subplots(1,2, figsize=(10,5))



sns.distplot(train['Age_PclXSex'], ax= ax[0])

sns.distplot(train['Age_Pclass'], ax = ax[1])
# Removing the unneeded and NA-dominated columns



train.drop(['Age', 'Cabin'], axis =1 , inplace = True)

# Drop the na rows



train.dropna(inplace = True)
train.info()
## Now creating dummy variables for Sex and Embarked





Sex_Dumm = pd.get_dummies(train['Sex'], drop_first = True)

Embarked_Dumm = pd.get_dummies(train['Embarked'], drop_first = True)

Ticket_First = pd.get_dummies(train['Ticket_First'], drop_first = True, prefix = 'Ticket')

Salute_Group = pd.get_dummies(train['Salute_Grp'], drop_first = True)

train = pd.concat([train, Sex_Dumm, Embarked_Dumm, Ticket_First, Salute_Group], axis = 1)

#train.drop(['Q','S', 'male','A', 'P', '3', '2', 'C', '7', 'W', '4', 'F', 'L', '9',

#       '6', '5', '8'], inplace = True, axis = 1)

train.head()

train.info()
### PREPARING TRAIN AND TEST DATASETS NOW



### The test dataset provided separately does not have the real result with it to evaluate the model performance. So, segrgating

### the train data into test and train itself





y = train['Survived']



##X1 has the missing Ages substituted by median age



X1 = train[['Pclass', 'SibSp', 'Parch', 'Fare',

       'Age_Med', 'male', 'Q', 'S', 'Ticket_2', 'Ticket_3', 'Ticket_4',

       'Ticket_5', 'Ticket_6', 'Ticket_7', 'Ticket_8', 'Ticket_9', 'Ticket_A',

       'Ticket_C', 'Ticket_F', 'Ticket_L', 'Ticket_P', 'Ticket_S', 'Ticket_W', 

       'Miss.', 'Mr.', 'Mrs.', 'Others']]



##X2 has the missing Ages substituted by mean age



X2 = train[['Pclass', 'SibSp', 'Parch', 'Fare',

       'Age_Mean', 'male', 'Q', 'S', 'Ticket_2', 'Ticket_3', 'Ticket_4',

       'Ticket_5', 'Ticket_6', 'Ticket_7', 'Ticket_8', 'Ticket_9', 'Ticket_A',

       'Ticket_C', 'Ticket_F', 'Ticket_L', 'Ticket_P', 'Ticket_S', 'Ticket_W',

        'Miss.', 'Mr.', 'Mrs.', 'Others']]





##X3 has the missing Ages substituted by median age based on the corresponding Pclass 



X3 = train[['Pclass', 'SibSp', 'Parch', 'Fare',

       'Age_Pclass', 'male', 'Q', 'S', 'Ticket_2', 'Ticket_3', 'Ticket_4',

       'Ticket_5', 'Ticket_6', 'Ticket_7', 'Ticket_8', 'Ticket_9', 'Ticket_A',

       'Ticket_C', 'Ticket_F', 'Ticket_L', 'Ticket_P', 'Ticket_S', 'Ticket_W',

        'Miss.', 'Mr.', 'Mrs.', 'Others']]



##X4 has the missing Ages substituted by median age based on the corresponding Pclass and Sex



X4 = train[['Pclass', 'SibSp', 'Parch', 'Fare',

       'Age_PclXSex', 'male', 'Q', 'S', 'Ticket_2', 'Ticket_3', 'Ticket_4',

       'Ticket_5', 'Ticket_6', 'Ticket_7', 'Ticket_8', 'Ticket_9', 'Ticket_A',

       'Ticket_C', 'Ticket_F', 'Ticket_L', 'Ticket_P', 'Ticket_S', 'Ticket_W',

        'Miss.', 'Mr.', 'Mrs.', 'Others']]



## Creating X41 what happens if the PClass is changed from labelencoded to one hot encoded



PClass = pd.get_dummies(train['Pclass'], drop_first = True)



X41 = pd.concat([X4[['SibSp', 'Parch', 'Fare',

                                  'Age_PclXSex', 'male', 'Q', 'S', 'Ticket_2', 'Ticket_3', 'Ticket_4',

       'Ticket_5', 'Ticket_6', 'Ticket_7', 'Ticket_8', 'Ticket_9', 'Ticket_A',

       'Ticket_C', 'Ticket_F', 'Ticket_L', 'Ticket_P', 'Ticket_S', 'Ticket_W', 'Miss.', 'Mr.', 'Mrs.', 'Others']], PClass], axis =1)



from sklearn.model_selection import train_test_split



rand= 150



X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y, test_size =0.3, random_state=rand)



X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y, test_size =0.3, random_state=rand)



X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y, test_size =0.3, random_state=rand)



X4_train, X4_test, y4_train, y4_test = train_test_split(X4, y, test_size =0.3, random_state=rand)



X41_train, X41_test, y41_train, y41_test = train_test_split(X41, y, test_size =0.3, random_state=rand)

### Since, I have already tested the relative performance of different ways of substituting age earlier, I am

### moving straight ahead using X4 option
### Fitting Classification decision tree without any paramter change



from sklearn.tree import DecisionTreeClassifier



dtree = DecisionTreeClassifier()

dtree.fit(X4_train, y4_train)

pred = dtree.predict(X4_test)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score



print(classification_report(y4_test, pred))

print(confusion_matrix(y4_test, pred))

print(accuracy_score(y4_test, pred))
## Pruning the tree



## Vary two parameters namely, criterion (gini or entropy) and max depth to see the combination of parameters at which

## we have least test error







Gini_Accuracy=[]

for i in range(1,41):

    dtree = DecisionTreeClassifier(criterion = 'gini', max_depth = i)

    dtree.fit(X4_train, y4_train)

    pred = dtree.predict(X4_test)

    Gini_Accuracy.append(accuracy_score(y4_test, pred))



Entropy_Accuracy=[]

for i in range(1,41):

    dtree = DecisionTreeClassifier(criterion = 'entropy', max_depth = i)

    dtree.fit(X4_train, y4_train)

    pred = dtree.predict(X4_test)

    Entropy_Accuracy.append(accuracy_score(y4_test, pred))
sns.lineplot(range(1,41),Gini_Accuracy, estimator = None, label = 'Gini')

sns.lineplot(range(1,41),Entropy_Accuracy, estimator = None, label = 'Entropy')



## Seems like Gini coefficient performs better overall, performing best at max depth 5.
##Prepare test data first



test = pd.read_csv('/kaggle/input/titanic/test.csv')
test[test['Pclass']==1]['Ticket'].unique()



test['Ticket_First'] = test['Ticket'].apply(lambda x:x.split()[0][:1])



test['Ticket_First'].unique()
test['Salute'] = test['Name'].apply(lambda x:x.split()[1])
def Salute_group(col):

    

    if col[0] in ['Mr.', 'Miss.', 'Mrs.', 'Master.']:

        return col[0]

    else:

        return 'Others'
test['Salute_Grp'] = test[['Salute']].apply(Salute_group, axis =1)
test['Age_Med'] = test['Age'].fillna(test['Age'].median())

test['Age_Mean'] = test['Age'].fillna(round(test['Age'].mean()))

# Calculating medians



Pclass_med = test[['Age','Pclass']].groupby(['Pclass']).median()

Pclass_med
# Defining a function to impute median (using median since the data is skewed) for each Pclass.







def age_Pclass(cols):

    age = cols[0]

    Pclass = cols[1]

    

    if pd.isnull(age) == True:

        return Pclass_med.loc[Pclass][0]

    else:

        return age
test['Age_Pclass'] = test[['Age', 'Pclass']].apply(age_Pclass, axis = 1)



sns.distplot(test['Age_Med'])
# Calculating medians



PclassXSex_med = test[['Sex','Age','Pclass']].groupby(['Sex','Pclass']).median()

PclassXSex_med
test['Age_PclXSex'] = test[['Age', 'Pclass', 'Sex']].apply(age_PclassSex, axis = 1)



fig, ax =plt.subplots(1,2, figsize=(10,5))



sns.distplot(test['Age_PclXSex'], ax= ax[0])

sns.distplot(test['Age_Pclass'], ax = ax[1])
# Removing the unneeded and NA-dominated columns



test.drop(['Cabin', 'Age'], axis =1 , inplace = True)

test.info()
# Substituting the missing value of fare using mean fare of that "Passenger_class X Sex X Embarked" group



test[pd.isnull(test['Fare'])]
Fare_med = test[['Pclass','Fare','Sex', 'Embarked']].groupby(['Pclass','Sex', 'Embarked']).agg(['count', 'mean'])



Fare_med
test['Fare'].fillna(12.718, inplace = True)
## Now creating dummy variables for Sex and Embarked





Sex_Dumm = pd.get_dummies(test['Sex'], drop_first = True)

Embarked_Dumm = pd.get_dummies(test['Embarked'], drop_first = True)

Ticket_First = pd.get_dummies(test['Ticket_First'], drop_first = True, prefix = 'Ticket')

Salute_Group = pd.get_dummies(test['Salute_Grp'], drop_first = True)

test = pd.concat([test, Sex_Dumm, Embarked_Dumm, Ticket_First, Salute_Group], axis = 1)

test.head()
test.columns
test['Ticket_5']=0

test['Ticket_8']=0
# Now using all the train dataset to fit the model and then predicting the test data



X = train[['Pclass' ,'SibSp', 'Parch', 'Fare',

                                  'Age_PclXSex', 'male', 'Q', 'S', 'Ticket_2', 'Ticket_3', 'Ticket_4',

       'Ticket_5', 'Ticket_6', 'Ticket_7', 'Ticket_8', 'Ticket_9', 'Ticket_A',

       'Ticket_C', 'Ticket_F', 'Ticket_L', 'Ticket_P', 'Ticket_S', 'Ticket_W', 'Miss.', 'Mr.', 'Mrs.', 'Others']]

y = train['Survived']
tree_fin= DecisionTreeClassifier(criterion='gini', max_depth=5)



tree_fin.fit(X,y)

X4_train.columns
X.columns

## VISUALISING THE DECISION TREE



# Export as dot file

from sklearn.tree import export_graphviz



#Can customise the tree visualisation below

export_graphviz(tree_fin, out_file='tree.dot', feature_names = X.columns)





# Convert dot file to pdf and open in the pdf reader

# from graphviz import Source

# path = '/kaggle/working/tree.dot'

# s = Source.from_file(path)

# s.view()



# Convert to png

from subprocess import call

call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])



# Display in python

import matplotlib.pyplot as plt

plt.figure(figsize = (14, 18))

plt.imshow(plt.imread('tree.png'))

plt.axis('off');

plt.show();
test.set_index('PassengerId', inplace = True)
X_test =test[['Pclass', 'SibSp', 'Parch', 'Fare',

                                  'Age_PclXSex', 'male', 'Q', 'S', 'Ticket_2', 'Ticket_3', 'Ticket_4',

       'Ticket_6', 'Ticket_7', 'Ticket_9', 'Ticket_A', 'Ticket_5', 'Ticket_8',

       'Ticket_C', 'Ticket_F', 'Ticket_L', 'Ticket_P', 'Ticket_S', 'Ticket_W', 'Miss.', 'Mr.', 'Mrs.', 'Others']]



X_test
pred_fin = tree_fin.predict(X_test)



pred_df = pd.DataFrame(pred_fin, columns = ['Predicted Value'],index = X_test.index)

pred_df
df_fin = pd.concat([X_test, pred_df], axis =1)



df_fin.info()
## Renaming the Predicted Value' to 'Survived' for submission



df_fin.rename(columns={'Predicted Value':'Survived'}, inplace=True)



df_fin.head()
# Output Result

df_fin['Survived'].to_csv('My_Titanic_Predictions3.csv', index = True, header = True)