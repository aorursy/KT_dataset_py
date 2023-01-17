# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import RFECV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import LabelEncoder

%matplotlib inline

import warnings
warnings.filterwarnings('ignore')
train_df = pd.read_csv('../input/titanic-machine-learning-from-disaster/train.csv')
test_df = pd.read_csv('../input/titanic-machine-learning-from-disaster/test.csv')
train_df.info() # we only have 204 cabin info in the training set
test_df.info() #we only have 91 cabine info in the test set
train_df.describe()
test_df.describe()
# Full dataset is needed for imputing missing values & also for pruning outliers

#whole_df = pd.concat([train_df, test_df], axis=0, ignore_index=True, sort=True) #If True, do not use the index values along the concatenation axis.

whole_df = train_df.append(test_df,sort=False)
whole_df.info()
#checking number of columns of each data type for general EDA
whole_df.dtypes.value_counts()
print(whole_df.select_dtypes(['int64','float64']).columns)
whole_df['Age'].hist(bins=70)
plt.show()
fig = sns.FacetGrid(whole_df, hue = 'Sex', aspect = 4)
fig.map(sns.kdeplot, 'Age', shade = True)

oldest = train_df['Age'].max()
fig.set(xlim = (0, oldest))
fig.add_legend()
plt.show()
whole_df['Age'].mean()  #get the mean age of all passengers, around 30yr
fig = sns.FacetGrid(whole_df, hue = 'Pclass', aspect = 4)
fig.map(sns.kdeplot, 'Age', shade = True)

oldest = train_df['Age'].max()
fig.set(xlim = (0, oldest))
fig.add_legend()
plt.show()
# We look at Age column and set Intevals on the ages and the map them to their categories as
# (Children, Teen, Adult, Old)
interval = (0,2,4,10,19,35,60,100)
categories = ['Infant','Toddler','Kid','Teen','Young Adult','Adult','Senior']
whole_df['Age_cats'] = pd.cut(whole_df.Age, interval, labels = categories)

ax = sns.countplot(x = 'Age_cats',  data = whole_df, hue = 'Survived', palette = 'Set1')

ax.set(xlabel='Age Categorical', ylabel='Total',
       title="Age Categorical Survival Distribution")

plt.show()
sns.distplot(whole_df.Fare)
plt.show()
whole_df.Fare.describe()
whole_df[whole_df.Fare == 0]
# Use a pandas plotting method to plot the column 'Fare' for each value of 'Survived' on the same plot.
train_df.groupby('Survived').Fare.hist(alpha=0.6)
plt.show()
sns.swarmplot(x='Survived', y='Fare', data=train_df)
plt.show()
# Use the DataFrame method .describe() to check out summary statistics of 'Fare' as a function of survival
train_df.groupby('Survived').Fare.describe()
sns.countplot(x='Pclass', hue = 'Sex',data=whole_df, palette="Set2") #Most of the males were in the 3rd class
plt.show()
# fig, ax = plt.subplots(1,1, figsize = (12,10))
ax = sns.countplot(x = 'Pclass', data=train_df,hue = 'Survived', palette = 'Set1')
ax.set(title = 'Passenger status (Survived/Died) against Passenger Class', 
       xlabel = 'Passenger Class', ylabel = 'Total')
plt.show()
whole_df.SibSp.value_counts()
plt.figure(figsize=(12,5))
sns.boxplot(y = whole_df.SibSp, x = whole_df.Age_cats)
plt.xticks(rotation=90)
plt.tight_layout()
plt.figure(figsize=(12,5))
sns.violinplot(y = whole_df.SibSp, x = whole_df.Age_cats,hue='Sex',
                    data=whole_df, palette="Set3",split=True)
plt.show()
plt.figure(figsize=(12,5))
sns.boxplot(y = whole_df.Parch, x = whole_df.Age_cats)
plt.xticks(rotation=90)
plt.tight_layout()
plt.figure(figsize=(12,5))
sns.violinplot(y = whole_df.Parch, x = whole_df.Age_cats,hue='Sex',
                    data=whole_df, palette="Set3",split=True)
plt.show()
print(whole_df.select_dtypes(['object']).columns)
deck = whole_df['Cabin'].dropna()
deck.head()
cabin_df = DataFrame(deck)
cabin_df
cabin_df['Cabin'] = cabin_df['Cabin'].astype(str).str[0] 
#change the datatype to str and get the first letter
cabin_df['Cabin'].unique()  #get all the unique values of column 'Cabin'
cabin_df['Cabin'].value_counts()
whole_df[whole_df['Cabin'].str.contains('T') == True]
cabin_df = cabin_df[cabin_df.Cabin != 'T']
sns.countplot('Cabin',data = cabin_df,palette = 'summer',order=['A','B','C','D','E','F','G'])
plt.show()
train_df['Deck']= train_df['Cabin'].dropna().astype(str).str[0] 
train_df
sns.catplot(x="Deck", hue="Survived", col="Sex",order=['A','B','C','D','E','F','G'],
                data=train_df, kind="count",
                height=5, aspect=1.2)
plt.show()
#Sex
sns.set(style="darkgrid")
sns.countplot(x='Sex', data=whole_df, palette="Set2")
plt.tight_layout()
plt.show()
ax = sns.countplot(x = 'Sex', data=train_df,hue = 'Survived', palette = 'Set1')
ax.set(title = 'Passenger status (Survived/Died) against Passenger Class', 
       xlabel = 'Passenger Sex', ylabel = 'Total')
plt.show()
#Explore the feature of Embarked
sns.countplot('Embarked',data = whole_df, hue = 'Pclass', order=['C','Q','S'], palette = 'husl')
plt.show()
sns.catplot(x="Embarked", hue="Survived", col="Pclass",
                data=train_df, kind="count",
                height=5, aspect=1.2)
plt.show()
whole_df['Fare'].describe()
whole_df.isna().sum() 
# I am going to drop Cabin and Cabin_cats since they have to many missing data
# drop Survivor, Child, Age_cats since we don't need them for further analysis
#Fill in missing data for Age, Embarked and Fare
whole_df.Name.head(20)
whole_df['Title'] = whole_df.Name.str.extract(r'([A-Za-z]+)\.', expand = False)
whole_df.Title.value_counts()
#[a-zA-Z]+: a word consisting of only Latin characters with a length at least one
#+: something repeating once or more
whole_df[whole_df['Name'].astype(str).str.contains('Col\.') == True]
whole_df[whole_df['Name'].astype(str).str.contains('Don\.') == True]
whole_df[whole_df['Name'].astype(str).str.contains('Master') == True]
Common_Title = ['Mr','Miss','Mrs','Master']
whole_df['Title'].replace(['Ms','Mme','Mlle','Dona'],'Miss', inplace = True)
whole_df['Title'].replace(['Lady'],'Mrs', inplace = True)
whole_df['Title'].replace(['Sir','Rev','Capt','Col','Don','Major'],'Mr', inplace = True)
whole_df['Title'][~whole_df.Title.isin(Common_Title)] = 'Others'
whole_df[whole_df['Title'] == 'Others']
whole_df.loc[796,'Title'] ='Mrs'
whole_df[(whole_df['Name'].str.contains('Dr\.') == True) & (whole_df['Title'] == 'Others') ]
whole_df[(whole_df['Name'].str.contains('Dr\.') == True) & (whole_df['Title'] == 'Others') ].index
whole_df.loc[[245, 317, 398, 632, 660, 766, 293],'Title'] ='Mr'
whole_df[whole_df['Title'] == 'Others']
whole_df.loc[759,'Title'] ='Mrs'
whole_df.loc[822,'Title'] ='Mr'
whole_df['Title'].value_counts()
#train_df = whole_df[:len(train_df)]
#test_df = whole_df[len(train_df):]
#train_df
# compute mean per group and find index after sorting
sorted_index = whole_df.groupby('Title')['Age'].mean().sort_values().index
sorted_index
sns.boxplot(x='Title', y = 'Age', data = whole_df, order=sorted_index)
plt.show()
AgeMedian_by_titles = whole_df.groupby('Title')['Age'].median()
AgeMedian_by_titles
#Impute the missing Age values according to the titles.
for title in AgeMedian_by_titles.index:
    whole_df['Age'][(whole_df.Age.isnull()) & (whole_df.Title == title)] = AgeMedian_by_titles[title]
whole_df.info()
whole_df[whole_df.Fare.isnull() == True]
plt.figure(figsize=(15,8))
sns.violinplot(x='Pclass', y = 'Fare', data = whole_df, hue='Sex',
              palette="Set3",split=True)
plt.show()
med_fare = whole_df.groupby(['Pclass', 'Sex']).Fare.median()
# Filling the missing value in Fare with the median Fare of 3rd class male passenger
whole_df['Fare'] = whole_df['Fare'].fillna(med_fare[3][1])
whole_df[whole_df.Fare == 0].sort_values('Ticket')
med_fare = whole_df.groupby(['Pclass', 'Sex']).Fare.median()
med_fare
whole_df[(whole_df.Fare == 0) & (whole_df.Pclass == 1)].index
whole_df.loc[[263, 633, 806, 815, 822, 266, 372],'Fare'] = med_fare[1][1]
whole_df[(whole_df.Fare == 0) & (whole_df.Pclass == 2)].index
whole_df.loc[[277, 413, 466, 481, 674, 732],'Fare'] = med_fare[2][1]
whole_df[(whole_df.Fare == 0) & (whole_df.Pclass == 3)].index
whole_df.loc[[179, 271, 302, 597],'Fare'] = med_fare[3][1]
whole_df.Fare.describe()
#Embarked
#For the dataset, there are only 2 missing values in the training dataset 
whole_df[whole_df['Embarked'].isnull()== True]
# Filling the missing values in Embarked with S
whole_df['Embarked'] = whole_df['Embarked'].fillna('S')
# Creating Deck column from the first letter of the Cabin column (M stands for Missing)
whole_df['Deck'] = whole_df['Cabin'].apply(lambda s: s[0] if pd.notnull(s) else 'M')

df_all_decks = whole_df.groupby(['Deck', 'Pclass']).count().drop(columns=['Survived', 'Sex', 'Age', 'SibSp', 'Parch', 
                                                                        'Fare', 'Embarked', 'Cabin', 'PassengerId', 'Ticket']).rename(columns={'Name': 'Count'}).transpose()

def get_pclass_dist(df):
    
    # Creating a dictionary for every passenger class count in every deck
    deck_counts = {'A': {}, 'B': {}, 'C': {}, 'D': {}, 'E': {}, 'F': {}, 'G': {}, 'M': {}, 'T': {}}
    decks = df.columns.levels[0]   
    for deck in decks:
        for pclass in range(1, 4):
            try:
                count = df[deck][pclass][0]
                deck_counts[deck][pclass] = count 
            except KeyError:
                deck_counts[deck][pclass] = 0
                
    df_decks = pd.DataFrame(deck_counts)    
    deck_percentages = {}

    # Creating a dictionary for every passenger class percentage in every deck
    for col in df_decks.columns:
        deck_percentages[col] = [(count / df_decks[col].sum()) * 100 for count in df_decks[col]]
        
    return deck_counts, deck_percentages

def display_pclass_dist(percentages):
    
    df_percentages = pd.DataFrame(percentages).transpose()
    deck_names = ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'M', 'T')
    bar_count = np.arange(len(deck_names))  
    bar_width = 0.85
    pclass1 = df_percentages[0]
    pclass2 = df_percentages[1]
    pclass3 = df_percentages[2]
    
    plt.figure(figsize=(20, 10))
    plt.bar(bar_count, pclass1, color='#b5ffb9', edgecolor='white', width=bar_width, label='Passenger Class 1')
    plt.bar(bar_count, pclass2, bottom=pclass1, color='#f9bc86', edgecolor='white', width=bar_width, label='Passenger Class 2')
    plt.bar(bar_count, pclass3, bottom=pclass1 + pclass2, color='#a3acff', edgecolor='white', width=bar_width, label='Passenger Class 3')

    plt.xlabel('Deck', size=15, labelpad=20)
    plt.ylabel('Passenger Class Percentage', size=15, labelpad=20)
    plt.xticks(bar_count, deck_names)    
    plt.tick_params(axis='x', labelsize=15)
    plt.tick_params(axis='y', labelsize=15)
    
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), prop={'size': 15})
    plt.title('Passenger Class Distribution in Decks', size=18, y=1.05)   
    
    plt.show()    

all_deck_count, all_deck_per = get_pclass_dist(df_all_decks)
display_pclass_dist(all_deck_per)
# Passenger in the T deck is changed to A
idx = whole_df[whole_df['Deck'] == 'T'].index
whole_df.loc[idx, 'Deck'] = 'A'
whole_df['Deck'] = whole_df['Deck'].replace(['A', 'B', 'C'], 'ABC')
whole_df['Deck'] = whole_df['Deck'].replace(['D', 'E'], 'DE')
whole_df['Deck'] = whole_df['Deck'].replace(['F', 'G'], 'FG')

whole_df['Deck'].value_counts()
whole_df.columns
#drop Cabin and Cabin_cats
whole_df = whole_df.drop(['Cabin','Age_cats'],axis=1)
whole_df['FamilySize'] = whole_df.SibSp + whole_df.Parch + 1
sns.countplot(whole_df.FamilySize)
plt.show()
facet = sns.FacetGrid(whole_df, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'FamilySize',shade= True)
facet.set(xlim=(0, whole_df['FamilySize'].max()))
facet.add_legend()
plt.xlim(0)
plt.show()
whole_df['Alone'] = whole_df.FamilySize.map(lambda x: 1 if x == 1 else 0)
sns.countplot(whole_df.Alone)
plt.show()
sns.barplot(x='Alone', y='Survived', data=whole_df)
plt.show()
sns.catplot(x='Alone', hue='Sex', col= 'Survived',
                data=whole_df, kind="count",
                height=5, aspect=1.2)
plt.show()
sns.catplot(x='Title', hue='Alone', col= 'Survived',
                data=whole_df, kind="count",
                height=5, aspect=1.2)
plt.show()
sns.barplot(x='Title', y='Survived', data=whole_df)
plt.show() #It is obviously that Title Mr. is much less likely to survive compared to others .
whole_df[['Name', 'Ticket']].sort_values('Name').head(20)
#It appears that passengers with same surnames have the same Ticket names.
whole_df[whole_df['Ticket']== 'LINE']
whole_df['Ticket'] = whole_df['Ticket'].str.replace('LINE', '370160',case = True)
whole_df['Surname'] = whole_df.Name.str.extract(r'([A-Za-z]+),', expand=False)
whole_df['Surname']
whole_df['TicNum'] = whole_df.Ticket.str.extract(r'([0-9]*$)', expand=False)
whole_df['TicNum']
## *: zero or more (0+), e.g., [0-9]* matches zero or more digits. 
## . (dot): ANY ONE character except newline. Same as [^\n]
## \d, \D: ANY ONE digit/non-digit character. Digits are [0-9]
whole_df['SurTix'] = whole_df['Surname'] + whole_df['TicNum']
whole_df['IsFamily'] = whole_df.SurTix.duplicated(keep=False)*1
sns.countplot(whole_df.IsFamily)
plt.show()
sns.catplot(col='IsFamily', x= 'Survived',
                data=whole_df, kind="count",
                height=5, aspect=1.2)
plt.show()
whole_df.sort_values('SurTix')
#Split the whole_df to training and test dataset
train_df = whole_df[:len(train_df)] #train_df
train_df.head()
test_df = whole_df[len(train_df):]
test_df.head()
train_df.columns
correlation = train_df.select_dtypes(include=[np.number]).corr()
print(correlation['Survived'].sort_values(ascending=False))
# Heatmap of correlation of numeric features
plt.figure(figsize=(25,14))
plt.title('Correlation Between Numeric Features', size=15)

sns.heatmap(correlation, square=True, vmax=0.8, cmap='coolwarm', linewidths=0.01,annot= True, annot_kws={"size": 8})

plt.show()
train_df.drop(['SibSp', 'Parch','FamilySize'], axis=1, inplace = True)
test_df.drop(['SibSp', 'Parch','FamilySize'], axis=1, inplace = True)
correlation = train_df.select_dtypes(include=[np.number]).corr()
plt.figure(figsize=(25,14))
plt.title('Correlation Between Numeric Features', size=15)

sns.heatmap(correlation, square=True, vmax=0.8, cmap='coolwarm', linewidths=0.01,annot= True, annot_kws={"size": 8})

plt.show()
whole_df.dtypes.value_counts() #there are 9 categorical variables 
print(whole_df.select_dtypes(['object']).columns)
#Encode string to numbers for modelling.
#Sex
train_df['Sex_Code'] = train_df['Sex'].map({'female':1, 'male':0}).astype('int')
test_df['Sex_Code'] = test_df['Sex'].map({'female':1, 'male':0}).astype('int')
#Embarked
train_df['Embarked_Code'] = train_df['Embarked'].map({'S':0, 'C':1, 'Q':2}).astype('int')
test_df['Embarked_Code'] = test_df['Embarked'].map({'S':0, 'C':1, 'Q':2}).astype('int')
# Title
train_df['Title_Code'] = train_df.Title.map({'Mr':0,'Others':1, 'Master':2,'Miss':3, 'Mrs':4}).astype('int')
test_df['Title_Code'] = test_df.Title.map({'Mr':0,'Others':1, 'Master':2,'Miss':3, 'Mrs':4}).astype('int')
#Deck
train_df['Deck_Code'] = train_df['Deck'].map({'M':0, 'ABC':1, 'DE':2,'FG':3}).astype('int')
test_df['Deck_Code'] = test_df['Deck'].map({'M':0, 'ABC':1, 'DE':2,'FG':3}).astype('int')
#Age
interval = (0,2,4,10,19,35,60,100)
categories = ['Infant','Toddler','Kid','Teen','Young Adult','Adult','Senior']
train_df['Age_category'] = pd.cut(train_df.Age, interval, labels = categories)
test_df['Age_category'] = pd.cut(test_df.Age, interval, labels = categories)
train_df['Age_category'] = train_df['Age_category'].map({'Infant':0,'Toddler':1,'Kid':2,
                                                         'Teen':3,'Young Adult':4,'Adult':5,'Senior':6}).astype('int')
test_df['Age_category'] = test_df['Age_category'].map({'Infant':0,'Toddler':1,'Kid':2,
                                                         'Teen':3,'Young Adult':4,'Adult':5,'Senior':6}).astype('int')
# Defining the map function
#def dummies(x,df):
#    temp = pd.get_dummies(df[x], drop_first = True)
#    df = pd.concat([df, temp], axis = 1)
#    df.drop([x], axis = 1, inplace = True)
#    return df

train_df.columns
#drop unused columns
X_train = train_df.drop(['PassengerId', 'Name', 'Sex', 'Age','Ticket','Embarked',
       'Title', 'Deck','Surname', 'TicNum','SurTix','Survived'], axis=1)
y_train = train_df['Survived']
X_train
X_test = test_df.drop(['PassengerId', 'Name', 'Sex', 'Age','Ticket','Embarked',
       'Title', 'Deck','Surname', 'TicNum','SurTix','Survived'], axis=1)
X_test
model = RandomForestClassifier(n_estimators=400, random_state=2)
#feature importance
model.fit(X_train,y_train)
importance = pd.DataFrame({'feature':X_train.columns, 'importance': np.round(model.feature_importances_,3)})
importance = importance.sort_values('importance', ascending=False).set_index('feature')
importance.plot(kind='bar', rot=90)
plt.show()
final = ['Fare', 'Title_Code','Sex_Code','Pclass','Age_category','Deck_Code']
#Tune Random Forest model parameters
grid_param = {
 'n_estimators': [10, 15, 20, 30,50,100,200,300,400,800],
 'criterion':['gini', 'entropy'],
 'min_samples_split': [2, 4, 10, 20],
 'min_samples_leaf': [1,2,5],
 'max_features':["sqrt", "auto", "log2"],
 'bootstrap': [True, False],
}
gd_sr = GridSearchCV(estimator=model,
 param_grid=grid_param,
 scoring='accuracy',
 cv=5,
 n_jobs=-1)
gd_sr.fit(X_train[final], y_train)
best_parameters = gd_sr.best_params_
print(best_parameters)
#Set the model paramters after tunning.
model = RandomForestClassifier(bootstrap=False,criterion= 'gini',  
                               min_samples_leaf=5, min_samples_split=20,
                               max_features='sqrt' , n_estimators=800, 
                               random_state=5)
#Calculate the accuracy of prediction using 5-fold cross-validation.
all_accuracies = cross_val_score(estimator=model, X=X_train[final], y=y_train, cv=10)
all_accuracies

print('Accuracy: %.3f stdev: %.2f' % (np.mean(np.abs(all_accuracies)), np.std(all_accuracies)))
X_test = test_df[final]

model.fit(X_train[final],y_train)
prediction = model.predict(X_test)
output = pd.DataFrame({'PassengerId': test_df.PassengerId, 'Survived': prediction.astype(int)})
output.to_csv('my_submission.csv', index=False)
# Instantiate XGB classifier - its hyperparameters are tuned through SkLearn Grid Search below

XGBmodel = XGBClassifier(n_estimators=400, random_state=5)
scores = cross_val_score(XGBmodel, X_train[final], y_train, cv=10, n_jobs=1, scoring='accuracy')
XGBmodel.fit(X_train[final],y_train)
print(scores)
print('Accuracy: %.3f stdev: %.2f' % (np.mean(np.abs(scores)), np.std(scores)))
#Tune XGB classification model parameters
xgbcParams = {
    'max_depth': range (3, 10, 1),
    'n_estimators': [100,200,300,400,800],
    'learning_rate': [0.002, 0.006, 0.1, 0.01, 0.05],
    'reg_lambda':[0,0.10, 0.50, 1],
    'subsample': [0.3, 0.9],
    'colsample_bytree': (0.5, 0.9),
    'min_child_weight': [1, 2, 3, 4],
}
grid_search = GridSearchCV(estimator=XGBmodel,
    param_grid=xgbcParams,
    scoring = 'accuracy',
    n_jobs = 4,
    cv = 5,
    verbose=True
)
grid_search.fit(X_train[final], y_train)
print(grid_search.best_params_)
print(grid_search.best_score_)
#Set the model paramters after tunning.
XGBmodel = XGBClassifier(max_depth = 4,
                       n_estimators=400, 
                       learning_rate=0.1, 
                       reg_lamda= 1, 
                       subsample =0.3 ,
                       colsample_bytree =0.9 ,
                       min_child_weight =3 ,
                       random_state=5)

#Calculate the accuracy of prediction using 5-fold cross-validation.
all_accuracies = cross_val_score(estimator=XGBmodel, X=X_train[final], y=y_train, cv=10)
all_accuracies
print('Accuracy: %.3f stdev: %.2f' % (np.mean(np.abs(all_accuracies)), np.std(all_accuracies)))
X_test = test_df[final]
XGBmodel.fit(X_train[final],y_train)
prediction = XGBmodel.predict(X_test)
output = pd.DataFrame({'PassengerId': test_df.PassengerId, 'Survived': prediction.astype(int)})
output.to_csv('my_submission2.csv', index=False)
#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train2 = sc.fit_transform(X_train[final])
X_test2 = sc.transform(X_test)
# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 6))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
from keras.wrappers.scikit_learn import KerasClassifier
classifier = KerasClassifier(build_fn = build_classifier, epochs = 500, batch_size = 36,optimizer = 'adam')
accuracies = cross_val_score(estimator=classifier, X=X_train2, y=y_train, cv=10) 
mean = accuracies.mean()
variance = accuracies.std()
print(mean)
print(variance)
classifier.fit(X_train2, y_train)
y_pred = classifier.predict(X_test2)
y_pred = (y_pred > 0.5)
y_pred = y_pred.astype(int)
submission = pd.DataFrame({ 'PassengerId': test_df['PassengerId'],
                          'Survived': y_pred[:,-1]}) 
submission = submission.to_csv("submission3.csv", index=False)
submission = pd.read_csv('submission3.csv')
print(submission)
