import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt



%matplotlib inline

sns.set_style('whitegrid')
# Add the titanic dataset from the right ( + Add data) button



# Running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
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
sns.set_style('whitegrid')

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
# Option3

sns.boxplot (x='Pclass', y='Age', data = train)



# There does seem to be a decrease in age as the class lowers. So, seems a good way to estimate the missing Ages

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
# Check if all the null values are gone



sns.heatmap(pd.isnull(train), cmap = 'viridis')
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



X1 = train[['Pclass', 'SibSp', 'Parch', 'Fare',

       'Age_Med', 'male', 'Q', 'S', 'Ticket_2', 'Ticket_3', 'Ticket_4',

       'Ticket_5', 'Ticket_6', 'Ticket_7', 'Ticket_8', 'Ticket_9', 'Ticket_A',

       'Ticket_C', 'Ticket_F', 'Ticket_L', 'Ticket_P', 'Ticket_S', 'Ticket_W', 

       'Miss.', 'Mr.', 'Mrs.', 'Others']]



X2 = train[['Pclass', 'SibSp', 'Parch', 'Fare',

       'Age_Mean', 'male', 'Q', 'S', 'Ticket_2', 'Ticket_3', 'Ticket_4',

       'Ticket_5', 'Ticket_6', 'Ticket_7', 'Ticket_8', 'Ticket_9', 'Ticket_A',

       'Ticket_C', 'Ticket_F', 'Ticket_L', 'Ticket_P', 'Ticket_S', 'Ticket_W',

        'Miss.', 'Mr.', 'Mrs.', 'Others']]



X3 = train[['Pclass', 'SibSp', 'Parch', 'Fare',

       'Age_Pclass', 'male', 'Q', 'S', 'Ticket_2', 'Ticket_3', 'Ticket_4',

       'Ticket_5', 'Ticket_6', 'Ticket_7', 'Ticket_8', 'Ticket_9', 'Ticket_A',

       'Ticket_C', 'Ticket_F', 'Ticket_L', 'Ticket_P', 'Ticket_S', 'Ticket_W',

        'Miss.', 'Mr.', 'Mrs.', 'Others']]





X4 = train[['Pclass', 'SibSp', 'Parch', 'Fare',

       'Age_PclXSex', 'male', 'Q', 'S', 'Ticket_2', 'Ticket_3', 'Ticket_4',

       'Ticket_5', 'Ticket_6', 'Ticket_7', 'Ticket_8', 'Ticket_9', 'Ticket_A',

       'Ticket_C', 'Ticket_F', 'Ticket_L', 'Ticket_P', 'Ticket_S', 'Ticket_W',

        'Miss.', 'Mr.', 'Mrs.', 'Others']]





PClass = pd.get_dummies(train['Pclass'], drop_first = True)



X41 = pd.concat([X4[['SibSp', 'Parch', 'Fare',

                                  'Age_PclXSex', 'male', 'Q', 'S', 'Ticket_2', 'Ticket_3', 'Ticket_4',

       'Ticket_5', 'Ticket_6', 'Ticket_7', 'Ticket_8', 'Ticket_9', 'Ticket_A',

       'Ticket_C', 'Ticket_F', 'Ticket_L', 'Ticket_P', 'Ticket_S', 'Ticket_W', 'Miss.', 'Mr.', 'Mrs.', 'Others']], PClass], axis =1)



from sklearn.model_selection import train_test_split



## Looping for 50 random train-test splits to see which option performs best in accurate test survival prediction.



X1_Accuracy=[]

X2_Accuracy=[]

X3_Accuracy=[]

X4_Accuracy=[]

X41_Accuracy=[]



from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score



for i in range(100,150):

    

    rand= i

    

    ## Split the data



    X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y, test_size =0.3, random_state=rand)



    X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y, test_size =0.3, random_state=rand)



    X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y, test_size =0.3, random_state=rand)



    X4_train, X4_test, y4_train, y4_test = train_test_split(X4, y, test_size =0.3, random_state=rand)



    X41_train, X41_test, y41_train, y41_test = train_test_split(X41, y, test_size =0.3, random_state=rand)

    

    ## Creating model and predicting values

    

    logistic1= LogisticRegression(max_iter= 4000)

    logistic1.fit(X1_train,y1_train)

    Y1_pred = logistic1.predict(X1_test)

    X1_Accuracy.append(accuracy_score(y1_test, Y1_pred))

    

    logistic2= LogisticRegression(max_iter= 4000)

    logistic2.fit(X2_train,y2_train)

    Y2_pred = logistic2.predict(X2_test)

    X2_Accuracy.append(accuracy_score(y2_test, Y2_pred))   

    

    logistic3= LogisticRegression(max_iter= 4000)

    logistic3.fit(X3_train,y3_train)

    Y3_pred = logistic3.predict(X3_test)

    X3_Accuracy.append(accuracy_score(y3_test, Y3_pred))  

    

    logistic4= LogisticRegression(max_iter= 4000)

    logistic4.fit(X4_train,y4_train)

    Y4_pred = logistic4.predict(X4_test)

    X4_Accuracy.append(accuracy_score(y4_test, Y4_pred))    

    

    logistic41= LogisticRegression(max_iter= 4000)

    logistic41.fit(X41_train,y41_train)

    Y41_pred = logistic41.predict(X41_test)

    X41_Accuracy.append(accuracy_score(y41_test, Y41_pred))
## Compare the performance



sns.lineplot(range(100,150), X1_Accuracy, label= 'X1_Accuracy')

sns.lineplot(range(100,150), X2_Accuracy, label= 'X2_Accuracy')

sns.lineplot(range(100,150), X3_Accuracy, label= 'X3_Accuracy')

sns.lineplot(range(100,150), X4_Accuracy, label= 'X4_Accuracy')

sns.lineplot(range(100,150), X41_Accuracy, label= 'X41_Accuracy')





## while 1 and 2 are certainly underperforming, rest are not much different. Also, overall, there is not a huge

## improvement in using one approach over another. So, going with X4 option. Not using One hot encoder for

## makes sense because it is not nominal category ie category 1 is better than 2 and so on.
## Prepare the test dataset in the same way



test = pd.read_csv('/kaggle/input/titanic/test.csv')
test.info()
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
test.info()
## Now creating dummy variables for Sex and Embarked





Sex_Dumm = pd.get_dummies(test['Sex'], drop_first = True)

Embarked_Dumm = pd.get_dummies(test['Embarked'], drop_first = True)

Ticket_First = pd.get_dummies(test['Ticket_First'], drop_first = True, prefix = 'Ticket')

Salute_Group = pd.get_dummies(test['Salute_Grp'], drop_first = True)

test = pd.concat([test, Sex_Dumm, Embarked_Dumm, Ticket_First, Salute_Group], axis = 1)

test.head()
test.columns
## Adding these two variables as these were not present in test data 



test['Ticket_5']=0

test['Ticket_8']=0
test.columns
train.columns




# Now using all the train dataset to fit the model and then predicting the test data



X = train[['Pclass', 'SibSp', 'Parch', 'Fare',

                                  'Age_PclXSex', 'male', 'Q', 'S', 'Ticket_2', 'Ticket_3', 'Ticket_4',

       'Ticket_5', 'Ticket_6', 'Ticket_7', 'Ticket_8', 'Ticket_9', 'Ticket_A',

       'Ticket_C', 'Ticket_F', 'Ticket_L', 'Ticket_P', 'Ticket_S', 'Ticket_W', 'Miss.', 'Mr.', 'Mrs.', 'Others']]

y = train['Survived']
### Fit the model now for all five options



from sklearn.linear_model import LogisticRegression



log= LogisticRegression(max_iter= 4000)



log.fit(X,y)

test.set_index('PassengerId', inplace = True)
test.info()
X_test =test[['Pclass', 'SibSp', 'Parch', 'Fare',

                                  'Age_PclXSex', 'male', 'Q', 'S', 'Ticket_2', 'Ticket_3', 'Ticket_4',

       'Ticket_6', 'Ticket_7', 'Ticket_9', 'Ticket_A', 'Ticket_5', 'Ticket_8',

       'Ticket_C', 'Ticket_F', 'Ticket_L', 'Ticket_P', 'Ticket_S', 'Ticket_W', 'Miss.', 'Mr.', 'Mrs.', 'Others']]



X_test
pred_fin = log.predict(X_test)



pred_df = pd.DataFrame(pred_fin, columns = ['Predicted Value'],index = X_test.index)

pred_df
df_fin = pd.concat([X_test, pred_df], axis =1)

df_fin.info()
## Renaming the Predicted Value' to 'Survived' for submission



df_fin.rename(columns={'Predicted Value':'Survived'}, inplace=True)



df_fin.head()


# Output Result

df_fin['Survived'].to_csv('My_Titanic_Predictions2.csv', index = True, header = True)