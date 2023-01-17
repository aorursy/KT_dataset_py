#Import all required libraries for reading data, analysing and visualizing data

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from sklearn.preprocessing import LabelEncoder

#machine learning

from sklearn import linear_model

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier
#Read the training & test data

titanic_train_df = pd.read_csv('../input/train.csv')

titanic_test_df = pd.read_csv('../input/test.csv')
#preview the data

titanic_train_df.head()
# Get the overall info of the dataframe

titanic_train_df.info()
#Which features are available in the dataset?

titanic_train_df.columns.values
#Lets do some high level analysis of the data. This data consists of the lists of passengers in the Titanic - 

#their sex, Passengerclass, Age, family status(how many siblings, children etc), cabin details, 

#where the passenger was embarked.

#We need to figure out what factors played major role in the survival of the sinking of Titanic.
#shape command will give number of rows/samples/examples and number of columns/features/predictors in dataset

#(rows,columns)

titanic_train_df.shape
titanic_test_df.shape
#Describe gives statistical information about NUMERICAL columns in the dataset

titanic_train_df.describe()

#We can see that there are missing values for Age as only 714 entries have valid age values.
titanic_train_df.describe(include=['O'])
#Find other columns that have missing values

titanic_train_df.isnull().sum()
#DATA ANALYSIS - Just random analysis of the data

#Before imputing missing values, lets just do high level analysis of the data to figure out the impact of different features in survival

titanic_train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
titanic_train_df[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
titanic_train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)
titanic_train_df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
titanic_train_df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
#Overall the highlevel data analysis indicate that Pclass, gender impacts the survival more. Survival of pclass=1 and gender male is less

#More number of kids => large family survival is less. Age might have impacted the survival as well.

#Since the ages vary, lets try to group the age to different category.

#Before that lets play around with data visualization for easy representation of what we figured out in data analysis

#Finding the different age groups corresponding to the sex and their survival

g = sns.FacetGrid(titanic_train_df, col="Sex", row="Survived", margin_titles=True)

g.map(plt.hist, "Age",color="red");

plt.subplots_adjust(top=0.9)

g.fig.suptitle('Survival by Sex and Age');
#Note that age is missing for some rows in the training data. The above histogram indicates that the survival is less for males

#Lets see how pclass impacts the survival

g = sns.FacetGrid(titanic_train_df, col="Pclass", row="Survived", margin_titles=True)

g.map(plt.hist, "Age",color="purple");

plt.subplots_adjust(top=0.9)

g.fig.suptitle('Survival by Pclass and Age');
g = sns.FacetGrid(titanic_train_df, hue="Survived", col="Pclass", margin_titles=True,

                  palette={1:"grey", 0:"red"},hue_kws=dict(marker=["v", "^"]))

g=g.map(plt.scatter, "Fare", "Age",edgecolor="w").add_legend();

plt.subplots_adjust(top=0.8)

g.fig.suptitle('Survival by Pclass, Age & Fare');
#Pclass1 survival seems to be more. Pclass2 seems not to make any difference. Pclass3 survival rate is low.

g = sns.FacetGrid(titanic_train_df, hue="Survived", col="Sex", margin_titles=True,

                palette={1:"grey", 0:"red"},hue_kws=dict(marker=["v", "^"]))

g.map(plt.scatter, "Fare", "Age",edgecolor="w").add_legend()

plt.subplots_adjust(top=0.8)

g.fig.suptitle('Survival by Gender , Age and Fare');
sns.barplot(x = 'Embarked',y="Survived", data = titanic_train_df, color="r");

sns.factorplot('Embarked',hue='Sex',data=titanic_train_df,kind='count')
# Let's check the impact of gender for Survival

sns.barplot(x='Sex',y='Survived',data=titanic_train_df)

sns.factorplot('Survived',hue='Sex',data=titanic_train_df,kind='count')
# We can see that the survival rate of pclass3 and males are low. Now let's seperate the genders by classes

#sns.factorplot('Pclass',data=titanic_train_df,hue='Sex',kind='bar')

sns.factorplot(x="Sex", y="Survived", col="Pclass",

                    data=titanic_train_df, saturation=.5,

                    kind="bar", ci=None, aspect=.6)
#The above grid indicates that the survival rate of pclass3 is less compared to other classes 1 &2 irrespective of the Sex.

#Looks like pclass plays a major role in Titanic survival rate
titanic_train_df.corr()['Survived']
#Lets try to see how many kids are there and find whether the age plays any role in the survival.

# First let's make a function to sort through the sex 

def age_cat(passenger):

    # Take the Age and Sex

    age,sex = passenger

    # Compare the age, otherwise leave the sex

    if age <=4:

        return 'infant'

    elif age > 4 and age <=12:

        return 'child'

    elif age > 12 and age <=25:

        return 'young'  

    elif age > 25 and age <=75:

        return 'adult' 

    elif age >75:

        return 'Senior'



# We'll define a new column called 'Agecat', remember to specify axis=1 for columns and not index

titanic_train_df['Agecat'] = titanic_train_df[['Age','Sex']].apply(age_cat,axis=1)

titanic_test_df['Agecat'] = titanic_test_df[['Age','Sex']].apply(age_cat,axis=1)
titanic_train_df[["Survived","Pclass","Agecat"]].groupby(['Agecat',"Pclass"],as_index=False).mean()
sns.factorplot(x="Agecat", y="Survived", col="Pclass",

                    data=titanic_train_df, saturation=.5,

                    kind="bar", ci=None, aspect=.6)

sns.plt.subplots_adjust(top=0.8)

sns.plt.suptitle("Age categories against Pclass for survival")
titanic_train_df[["Agecat", "Survived"]].groupby(['Agecat'],as_index=False).mean()
#Lets see how the parch and subsp impacts the survival

#parch indicates  parents/children and sibsp indicates the siblings/spouses that boared the Titanic

#Lets create a new feature "Family" that has the total number of people that boarded in a family.

titanic_train_df["Familysize"] = titanic_train_df["SibSp"] + titanic_train_df["Parch"]

titanic_train_df.head()
titanic_test_df["Familysize"] = titanic_test_df["SibSp"] + titanic_test_df["Parch"]
#Lets try to differentiate small and large families and see whether there is any significant difference

def famsz(passenger):

    # Take the Age and Sex

    familysize = passenger

    # Compare the age, otherwise leave the sex

    if familysize == 0:

        return 'Alone'

    elif familysize < 4:

        return 'Small'

    else:

        return 'Large'



# We'll define a new column called 'Family', remember to specify axis=1 for columns and not index

titanic_train_df['Family'] = titanic_train_df['Familysize'].apply(famsz)

titanic_test_df['Family'] = titanic_test_df['Familysize'].apply(famsz)
titanic_train_df.head()
sns.factorplot(x="Agecat", y="Survived", col="Family",

                    data=titanic_train_df, saturation=.5,

                    kind="bar", ci=None, aspect=.6)

sns.plt.subplots_adjust(top=0.8)

sns.plt.suptitle("Age groups against familysize for survival")
titanic_train_df[["Family", "Survived"]].groupby(['Family'],as_index=False).mean()
sns.factorplot(x="Agecat", y="Survived", col="Pclass",

                    data=titanic_train_df, saturation=.5,

                    kind="bar", ci=None, aspect=.6)

sns.plt.subplots_adjust(top=0.8)

sns.plt.suptitle("Age groups against passengerclassfor survival")

#NOTE AGE value is still missing for 177 passengers
#small sized families in pclass 1&2 - survival rate is more

#Large sized families in pclass 1&2 - survival rate is more

#Lonely boarders in pclass3 - surival rate is less

# Overall lonely boarders survival rate is low compared to those traveled with family.
#MISSING VALUE IMPUTATION

titanic_train_df.isnull().sum()
titanic_test_df.isnull().sum()
#Age, Fare has missing values in training data. Note that cabin dont seem to make an impact to the data. So, lets ignore this as of now.
#Lets check which rows have null Embarked column

titanic_train_df[titanic_train_df['Embarked'].isnull()]
sns.barplot(x="Embarked", y="Fare", hue="Pclass", data=titanic_train_df);
titanic_train_df["Embarked"] = titanic_train_df["Embarked"].fillna('C')
titanic_test_df[titanic_test_df['Fare'].isnull()]
titanic_test_df[(titanic_test_df['Pclass'] == 3) & (titanic_test_df['Embarked'] == 'S')]['Fare'].mean()
#we can replace missing value in fare by taking median of all fares of those passengers 

#who share 3rd Passenger class and Embarked from 'S' 

titanic_test_df["Fare"] = titanic_test_df.Fare.fillna(titanic_test_df[(titanic_test_df['Pclass'] == 3) & (titanic_test_df['Embarked'] == 'S')]['Fare'].mean())
#titanic_test_df[titanic_test_df['PassengerId'] == 1044]
#Lets move onto looking at the different titles

titanic_train_df['Title'] = titanic_train_df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
pd.crosstab(titanic_train_df['Title'], titanic_train_df['Sex'])
titanic_train_df['Age'].isnull().sum()
titanic_train_df['Title'] = titanic_train_df['Title'].replace('Mlle', 'Miss')

titanic_train_df['Title'] = titanic_train_df['Title'].replace('Ms', 'Miss')

titanic_train_df['Title'] = titanic_train_df['Title'].replace('Mme', 'Mrs')
pd.crosstab(titanic_train_df['Title'], titanic_train_df['Sex'])
titles = ['Jonkheer','Col','Capt','Countess','Don','Dona','Dr','Lady','Major','Rev','Sir']

titanic_train_df[(titanic_train_df['Title'].isin(titles))& (titanic_train_df['Age'].isnull())]
#NOTE: The titles 'Jonkheer','Col','Capt','Countess','Don','Dona',Lady','Major','Rev','Sir', Dr all have valid age values except Dr. Lets try to find the mean and fill the age.
titanic_train_df.loc[titanic_train_df['Title'] == 'Dr']
titanic_train_df.Age.loc[titanic_train_df['Title'] == 'Dr'] = titanic_train_df.Age.fillna(titanic_train_df[(titanic_train_df['Title'] == 'Dr')]['Age'].mean())
#Lets replace all these titles to "Rare"

titanic_train_df['Title'] = titanic_train_df['Title'].replace('Jonkheer', 'Rare')

titanic_train_df['Title'] = titanic_train_df['Title'].replace('Col', 'Rare')

titanic_train_df['Title'] = titanic_train_df['Title'].replace('Capt', 'Rare')

titanic_train_df['Title'] = titanic_train_df['Title'].replace('Countess', 'Rare')

titanic_train_df['Title'] = titanic_train_df['Title'].replace('Don', 'Rare')

titanic_train_df['Title'] = titanic_train_df['Title'].replace('Dona', 'Rare')

titanic_train_df['Title'] = titanic_train_df['Title'].replace('Dr', 'Rare')

titanic_train_df['Title'] = titanic_train_df['Title'].replace('Lady', 'Rare')

titanic_train_df['Title'] = titanic_train_df['Title'].replace('Major', 'Rare')

titanic_train_df['Title'] = titanic_train_df['Title'].replace('Rev', 'Rare')

titanic_train_df['Title'] = titanic_train_df['Title'].replace('Sir', 'Rare')
pd.crosstab(titanic_train_df['Title'], titanic_train_df['Sex'])
titanic_train_df['Age'].isnull().sum()
titles = ['Master','Miss','Mr','Mrs','Rare']

titanic_train_df[(titanic_train_df['Title'].isin(titles))& (titanic_train_df['Age'].isnull())]
titanic_train_df[(titanic_train_df['Title'] == 'Master')]['Age'].mean()
titanic_train_df.Age.loc[titanic_train_df['Title'] == 'Master'] = titanic_train_df.Age.fillna(titanic_train_df[(titanic_train_df['Title'] == 'Master')]['Age'].mean())
titanic_train_df[(titanic_train_df['Title'] == 'Miss')]['Age'].mean()
titanic_train_df.Age.loc[titanic_train_df['Title'] == 'Miss'] = titanic_train_df.Age.fillna(titanic_train_df[(titanic_train_df['Title'] == 'Miss')]['Age'].mean())
titanic_train_df[(titanic_train_df['Title'] == 'Mrs')]['Age'].mean()
titanic_train_df.Age.loc[titanic_train_df['Title'] == 'Mrs'] = titanic_train_df.Age.fillna(titanic_train_df[(titanic_train_df['Title'] == 'Mrs')]['Age'].mean())
titanic_train_df[(titanic_train_df['Title'] == 'Mr')]['Age'].mean()
titanic_train_df.Age.loc[titanic_train_df['Title'] == 'Mr'] = titanic_train_df.Age.fillna(titanic_train_df[(titanic_train_df['Title'] == 'Mr')]['Age'].mean())
titanic_train_df['Age'].isnull().sum()
pd.crosstab(titanic_train_df['Title'], titanic_train_df['Sex'])
#Lets do the age cat again for test data set

#Lets try to see how many kids are there and find whether the age plays any role in the survival.

# First let's make a function to sort through the sex 

def age_cat(passenger):

    # Take the Age and Sex

    age,sex = passenger

    # Compare the age, otherwise leave the sex

    if age <=4:

        return 'infant'

    elif age > 4 and age <=12:

        return 'child'

    elif age > 12 and age <=25:

        return 'young'  

    elif age > 25 and age <=75:

        return 'adult' 

    elif age >75:

        return 'Senior'



# We'll define a new column called 'Agecat', remember to specify axis=1 for columns and not index

titanic_train_df['Agecat'] = titanic_train_df[['Age','Sex']].apply(age_cat,axis=1)
titanic_train_df.isnull().sum()
#Lets do the process of filling in missing values for test set as well.

titanic_test_df['Title'] = titanic_test_df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
pd.crosstab(titanic_test_df['Title'], titanic_test_df['Sex'])
titanic_test_df['Age'].isnull().sum()
titanic_test_df['Title'] = titanic_test_df['Title'].replace('Mlle', 'Miss')

titanic_test_df['Title'] = titanic_test_df['Title'].replace('Ms', 'Miss')

titanic_test_df['Title'] = titanic_test_df['Title'].replace('Mme', 'Mrs')
pd.crosstab(titanic_test_df['Title'], titanic_test_df['Sex'])
titles = ['Jonkheer','Col','Capt','Countess','Don','Dona','Dr','Lady','Major','Rev','Sir']

titanic_test_df[(titanic_test_df['Title'].isin(titles))& (titanic_test_df['Age'].isnull())]
#No null age values for the rare titles.

#Lets replace all these titles to "Rare"

titanic_test_df['Title'] = titanic_test_df['Title'].replace('Jonkheer', 'Rare')

titanic_test_df['Title'] = titanic_test_df['Title'].replace('Col', 'Rare')

titanic_test_df['Title'] = titanic_test_df['Title'].replace('Capt', 'Rare')

titanic_test_df['Title'] = titanic_test_df['Title'].replace('Countess', 'Rare')

titanic_test_df['Title'] = titanic_test_df['Title'].replace('Don', 'Rare')

titanic_test_df['Title'] = titanic_test_df['Title'].replace('Dona', 'Rare')

titanic_test_df['Title'] = titanic_test_df['Title'].replace('Dr', 'Rare')

titanic_test_df['Title'] = titanic_test_df['Title'].replace('Lady', 'Rare')

titanic_test_df['Title'] = titanic_test_df['Title'].replace('Major', 'Rare')

titanic_test_df['Title'] = titanic_test_df['Title'].replace('Rev', 'Rare')

titanic_test_df['Title'] = titanic_test_df['Title'].replace('Sir', 'Rare')

pd.crosstab(titanic_test_df['Title'], titanic_test_df['Sex'])
titles = ['Master','Miss','Mr','Mrs','Rare']

titanic_test_df[(titanic_test_df['Title'].isin(titles))& (titanic_test_df['Age'].isnull())]
titanic_test_df.Age.loc[titanic_test_df['Title'] == 'Master'] = titanic_test_df.Age.fillna(titanic_test_df[(titanic_test_df['Title'] == 'Master')]['Age'].mean())
titanic_test_df.Age.loc[titanic_test_df['Title'] == 'Mr'] = titanic_test_df.Age.fillna(titanic_test_df[(titanic_test_df['Title'] == 'Mr')]['Age'].mean())
titanic_test_df.Age.loc[titanic_test_df['Title'] == 'Miss'] = titanic_test_df.Age.fillna(titanic_test_df[(titanic_test_df['Title'] == 'Miss')]['Age'].mean())
titanic_test_df.Age.loc[titanic_test_df['Title'] == 'Mrs'] = titanic_test_df.Age.fillna(titanic_test_df[(titanic_test_df['Title'] == 'Mrs')]['Age'].mean())
titanic_test_df.isnull().sum()
#Lets do the age cat again for test data set

#Lets try to see how many kids are there and find whether the age plays any role in the survival.

# First let's make a function to sort through the sex 

def age_cat(passenger):

    # Take the Age and Sex

    age,sex = passenger

    # Compare the age, otherwise leave the sex

    if age <=4:

        return 'infant'

    elif age > 4 and age <=12:

        return 'child'

    elif age > 12 and age <=25:

        return 'young'  

    elif age > 25 and age <=75:

        return 'adult' 

    elif age >75:

        return 'Senior'



# We'll define a new column called 'Agecat', remember to specify axis=1 for columns and not index

titanic_test_df['Agecat'] = titanic_test_df[['Age','Sex']].apply(age_cat,axis=1)
titanic_test_df.isnull().sum()
#Lets try to see how many kids are there and find whether the age plays any role in the survival.

# First let's make a function to sort through the sex 

def child(passenger):

    # Take the Age and Sex

    agecat = passenger

    # Compare the age, otherwise leave the sex

    if agecat == 'infant' or agecat == 'child':

        return 0

    else:

        return 1



# We'll define a new column called 'Agecat', remember to specify axis=1 for columns and not index

titanic_train_df['Child'] = titanic_train_df['Agecat'].apply(child)

titanic_test_df['Child'] = titanic_test_df['Agecat'].apply(child)
#Cabin dont seem to play a significant role in the survival. Hence, I'm going to drop cabin & Ticket columns. Also parch & Sibsp as I already have familysize and family columns

titanic_train_df = titanic_train_df.drop(['Parch', 'SibSp', 'Ticket', 'Cabin', 'Name'], axis=1)

titanic_test_df = titanic_test_df.drop(['Parch', 'SibSp', 'Ticket', 'Cabin', 'Name'], axis=1)
#Building a Predictive Model in Python

#sklearn requires all inputs to be numeric, hence we should convert all our categorical variables into numeric 

#by encoding the categories. 



from sklearn.preprocessing import LabelEncoder

#Library "Scikit Learn" only works with numeric array. Hence, we need to label all the character variables into a numeric array.

le = LabelEncoder()

#var_mod = ['Sex','Family','Embarked','Agecat','Title']

titanic_train_df['Sex'] = le.fit_transform(titanic_train_df['Sex'].map( {'female': 0, 'male': 1} ).astype(str))

titanic_train_df['Family'] = le.fit_transform(titanic_train_df['Family'].map( {'Alone': 0, 'Small': 1, 'Large': 2}).astype(str))

titanic_train_df['Agecat'] = le.fit_transform(titanic_train_df['Agecat'].map( {'is': 0, 'child': 1, 'young': 2, 'adult': 3, 'Senior': 4}).astype(str))

titanic_train_df['Embarked'] = le.fit_transform(titanic_train_df['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2}).astype(str))

titanic_train_df['Title'] = le.fit_transform(titanic_train_df['Title'].map({'Master': 0, 'Miss': 1, 'Mr': 2, 'Mrs': 3, 'Rare': 4}).astype(str))
titanic_train_df.head()
titanic_test_df.head()
#Building a Predictive Model in Python

#sklearn requires all inputs to be numeric, hence we should convert all our categorical variables into numeric 

#by encoding the categories. 



from sklearn.preprocessing import LabelEncoder

#Library "Scikit Learn" only works with numeric array. Hence, we need to label all the character variables into a numeric array.

# Perform label encoding for variable 'Married':  Married No = 0, Married Yes = 1

le = LabelEncoder()

#var_mod = ['Sex','Family','Embarked','Agecat','Title']

titanic_test_df['Sex'] = le.fit_transform(titanic_test_df['Sex'].map( {'female': 0, 'male': 1} ).astype(str))

titanic_test_df['Family'] = le.fit_transform(titanic_test_df['Family'].map( {'Alone': 0, 'Small': 1, 'Large': 2}).astype(str))

titanic_test_df['Agecat'] = le.fit_transform(titanic_test_df['Agecat'].map( {'infant': 0, 'child': 1, 'young': 2, 'adult': 3, 'Senior': 4}).astype(str))

titanic_test_df['Embarked'] = le.fit_transform(titanic_test_df['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2}).astype(str))

titanic_test_df['Title'] = le.fit_transform(titanic_test_df['Title'].map({'Master': 0, 'Miss': 1, 'Mr': 2, 'Mrs': 3, 'Rare': 4}).astype(str))
corr=titanic_train_df.corr()#["Survived"]

plt.figure(figsize=(10, 10))



sns.heatmap(corr, vmax=.8, linewidths=0.01,

            square=True,annot=True,cmap='YlGnBu',linecolor="white")

plt.title('Correlation between features');
titanic_train_df.corr()['Survived']
titanic_train_df.shape
titanic_test_df.shape
#Model, Predict & Solve

#This problem is a Classification & Regression problem. We want to identify relationship between output (Survived or not) with other variables or features (Gender, Age, Port...). We are also perfoming a category of machine learning which is called supervised learning as we are training our model with a given dataset. 

#With these two criteria - Supervised Learning plus Classification and Regression, we can narrow down our choice of models 

#to a few. These include:

#Logistic Regression

#KNN or k-Nearest Neighbors

#Support Vector Machines

#Naive Bayes classifier

#Decision Tree

#Random Forrest

#Perceptron

#Artificial neural network

#RVM or Relevance Vector Machine
#Selecting  predictors

predictors =['Pclass','Sex','Child', 'Family']



#Converting predictors and outcome to numpy array

X_train = titanic_train_df[predictors].values

Y_train = titanic_train_df['Survived'].values



# Converting predictors and outcome to numpy array

X_test = titanic_test_df[predictors].values
#Y_train = titanic_train_df['Survived'].values

#X_train =  titanic_train_df.drop(['Survived'], axis=1)

#X_test  = titanic_test_df.values

#X_train.shape, Y_train.shape, X_test.shape
# Linear Regression

linear = linear_model.LinearRegression()

# Train the model using the training sets and check score

linear.fit(X_train, Y_train)

linear_score = round(linear.score(X_train, Y_train) * 100, 2)

#Equation coefficient and Intercept

print('Linear Regression Score: \n', linear_score)

print('Coefficient: \n', linear.coef_)

print('Intercept: \n', linear.intercept_)

#Predict Output

predicted= linear.predict(X_test)
coeff_df = pd.DataFrame(titanic_train_df.columns.delete(0))

coeff_df.columns = ['Feature']

coeff_df["Correlation"] = pd.Series(linear.coef_[0])



coeff_df.sort_values(by='Correlation', ascending=False)
#Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset

# Create logistic regression object

logreg = LogisticRegression()

# Train the model using the training sets and check score

logreg.fit(X_train, Y_train)

logreg_score = round(logreg.score(X_train, Y_train) * 100, 2)

#Equation coefficient and Intercept

print('Logistic Regression Score: \n', logreg_score)

print('Coefficient: \n', logreg.coef_)

print('Intercept: \n', logreg.intercept_)



#Predict Output

predicted= logreg.predict(X_test)
coeff_df = pd.DataFrame(titanic_train_df.columns.delete(0))

coeff_df.columns = ['Feature']

coeff_df["Correlation"] = pd.Series(logreg.coef_[0])



coeff_df.sort_values(by='Correlation', ascending=False)
#Positive coefficients increase the log-odds of the response (and thus increase the probability), 

#and negative coefficients decrease the log-odds of the response (and thus decrease the probability).

#Family is highest positivie coefficient

#Pclass is highest negative coefficient

#When Pclass increases, survival decreases

#When Family increases, survival increases.
#Next we model using Support Vector Machines which are supervised learning models with associated learning algorithms 

#that analyze data used for classification and regression analysis. 

#Given a set of training samples, each marked as belonging to one or the other of two categories, 

#an SVM training algorithm builds a model that assigns new test samples to one category or the other, 

#making it a non-probabilistic binary linear classifier. Reference Wikipedia.

#Note that the model generates a confidence score which is higher than Logistics Regression model.



# Support Vector Machines



svc = SVC()

svc.fit(X_train, Y_train)

svc_score = round(svc.score(X_train, Y_train) * 100, 2)

print('SVM Score: \n', svc_score)



#Predict Output

predicted= svc.predict(X_test)
#In pattern recognition, the k-Nearest Neighbors algorithm (or k-NN for short) is a non-parametric method 

#used for classification and regression. A sample is classified by a majority vote of its neighbors, 

#with the sample being assigned to the class most common among its k nearest neighbors (k is a positive integer, 

#typically small). If k = 1, then the object is simply assigned to the class of that single nearest neighbor. 

#KNN confidence score is better than Logistics Regression but worse than SVM.



knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train, Y_train)

knn_score = round(knn.score(X_train, Y_train) * 100, 2)

print('KNN Score: \n', knn_score)



#Predict Output

predicted =  knn.predict(X_test)
#In machine learning, naive Bayes classifiers are a family of simple probabilistic classifiers 

#based on applying Bayes' theorem with strong (naive) independence assumptions between the features. 

#Naive Bayes classifiers are highly scalable, requiring a number of parameters linear in the 

#number of variables (features) in a learning problem.

#The model generated confidence score is the lowest among the models evaluated so far.



# Gaussian Naive Bayes



gaussian = GaussianNB()

gaussian.fit(X_train, Y_train)

gauss_score = round(gaussian.score(X_train, Y_train) * 100, 2)

print('Gaussian Score: \n', gauss_score)

#Predict Output

predicted = gaussian.predict(X_test)
#The perceptron is an algorithm for supervised learning of binary classifiers 

#(functions that can decide whether an input, represented by a vector of numbers, belongs to some specific class or not). 

#It is a type of linear classifier, i.e. a classification algorithm that makes its predictions 

#based on a linear predictor function combining a set of weights with the feature vector. 

#The algorithm allows for online learning, in that it processes elements in the training set one at a time. 



# Perceptron



perceptron = Perceptron()

perceptron.fit(X_train, Y_train)

perceptron_score = round(perceptron.score(X_train, Y_train) * 100, 2)

print('Perceptron Score: \n', perceptron_score)



#Predict Output

predicted = perceptron.predict(X_test)
# Linear SVC

linear_svc = LinearSVC()

linear_svc.fit(X_train, Y_train)

linear_svc_score = round(linear_svc.score(X_train, Y_train) * 100, 2)

print('Linear SVC Score: \n', linear_svc_score)



#Predict Output

predicted = linear_svc.predict(X_test)
# Stochastic Gradient Descent

sgd = SGDClassifier()

sgd.fit(X_train, Y_train)

sgd_score = round(sgd.score(X_train, Y_train)* 100, 2)

print('SGD Score: \n', sgd_score)



#Predict Output

predicted = sgd.predict(X_test)
#This model uses a decision tree as a predictive model which maps features (tree branches) to conclusions 

#about the target value (tree leaves). Tree models where the target variable can take a finite set of values 

#are called classification trees; in these tree structures, leaves represent class labels and branches 

#represent conjunctions of features that lead to those class labels. Decision trees where the target variable 

#can take continuous values (typically real numbers) are called regression trees. 

#The model confidence score is the highest among models evaluated so far.



# Decision Tree



decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_train, Y_train)

decision_tree_score = round(decision_tree.score(X_train, Y_train)* 100, 2)

print('Decision Tree Score: \n', decision_tree_score)



#Predict Output

predicted = decision_tree.predict(X_test)
from sklearn import tree

# Decision Tree Classification

dectreeCls = tree.DecisionTreeClassifier(criterion='gini') 

# for classification, here you can change the algorithm as gini or entropy (information gain) by default it is gini  

# model = tree.DecisionTreeRegressor() for regression

# Train the model using the training sets and check score

dectreeCls.fit(X_train, Y_train)

dectreeCls_score = round(dectreeCls.score(X_train, Y_train) * 100, 2)



print(' Decision Tree Classification Score: \n', dectreeCls_score)

#Predict Output

predicted= dectreeCls.predict(X_test)
from sklearn import tree

# Decision Tree for regression

dectreeR = tree.DecisionTreeRegressor()

# for classification, here you can change the algorithm as gini or entropy (information gain) by default it is gini  

# model = tree.DecisionTreeRegressor() for regression

# Train the model using the training sets and check score

dectreeR.fit(X_train, Y_train)

dectreeR_score = round(dectreeR.score(X_train, Y_train) * 100, 2)

print('Decision Tree Regression Score: \n', dectreeR_score)



#Predict Output

predicted= dectreeR.predict(X_test)
#Import Library

from sklearn.ensemble import GradientBoostingClassifier

#Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset

# Create Gradient Boosting Classifier object

gbclass = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)

# Train the model using the training sets and check score

gbclass.fit(X_train, Y_train)

gbclass_score = round(gbclass.score(X_train, Y_train) * 100, 2)

print('Score: \n', gbclass_score)



#Predict Output

predicted= gbclass.predict(X_test)
#The next model Random Forests is one of the most popular. Random forests or random decision forests are an 

#ensemble learning method for classification, regression and other tasks, that operate by constructing a 

#multitude of decision trees (n_estimators=100) at training time and outputting the class that is the 

#mode of the classes (classification) or mean prediction (regression) of the individual trees. 

#The model confidence score is the highest among models evaluated so far. 

#We decide to use this model's output (Y_pred) for creating our competition submission of results.



# Random Forest



random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)

random_forest_score = round(random_forest.score(X_train, Y_train) * 100, 2)

print('Random Forest Score: \n', random_forest_score)



#Predict Output

predicted = random_forest.predict(X_test)
###Model evaluation

#We can now rank our evaluation of all the models to choose the best one for our problem. 

#While both Decision Tree and Random Forest score the same, we choose to use Random Forest as they correct for decision trees' 

#habit of overfitting to their training set.



models = pd.DataFrame({

    'Model': ['Linear Regression', 'Logistic Regression', 'Support Vector Machines', 'KNeighborsClassifier', 'Gaussian Naive Bayes',

              'Linear SVC', 'Stochastic Gradient Descent', 'Decision Tree', 'Decision Tree Classification', 'Decision Tree Regression', 

              'Random Forest', 'Perceptron', 'Gradient Boosting'

              ],

    'Score': [linear_score, logreg_score, svc_score, knn_score, gauss_score, linear_svc_score, sgd_score, decision_tree_score,

              dectreeCls_score, dectreeR_score, random_forest_score, perceptron_score, gbclass_score]})

models.sort_values(by='Score', ascending=False)
submission = pd.DataFrame({

        "PassengerId": titanic_test_df["PassengerId"],

        "Survived": predicted

    })

submission.to_csv('titanic_survival_op1.csv', index=False)