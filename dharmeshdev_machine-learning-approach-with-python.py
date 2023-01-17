# linear algebra
import numpy as np 

# data processing
import pandas as pd 

# data visualization
import seaborn as sns
sns.set(font_scale=1) # default settings for cleaner graphs

# another data visualization
%matplotlib inline
from matplotlib import pyplot as plt
from matplotlib import style
# This creates a pandas dataframe and assigns it to the train variable.
train_df = pd.read_csv("../input/train.csv")
# This creates a pandas dataframe and assigns it to the test variable.
test_df = pd.read_csv("../input/test.csv")
# Print the first 5 rows of the train dataframe.
train_df.head()
# Print the first 5 rows of the test dataframe.
# note their is no Survived column here which is our target varible we are trying to predict
test_df.head()
# lets print data info
train_df.info()
train_df.describe()
sns.barplot(x="Embarked", y="Survived", hue="Sex", data=train_df);
# From below we can see that Male/Female that Emarked from C have higher chances of survival 
# compared to other Embarked points.
sns.pointplot(x="Pclass", y="Survived", hue="Sex", data=train_df,
              palette={"male": "blue", "female": "pink"},
              markers=["*", "o"], linestyles=["-", "--"]);
g = sns.FacetGrid(train_df, col="Sex", row="Survived", margin_titles=True)
g.map(plt.hist, "Age",color="purple");
corr=train_df.corr()#["Survived"]
plt.figure(figsize=(10, 10))

sns.heatmap(corr, vmax=.8, linewidths=0.01,
            square=True,annot=True,cmap='YlGnBu',linecolor="white")
plt.title('Correlation between features');
#lets see which are the columns with missing values in train dataset
train_df.isnull().sum()
labels = []
values = []
null_columns = train_df.columns[train_df.isnull().any()]
for col in null_columns:
    labels.append(col)
    values.append(train_df[col].isnull().sum())

ind = np.arange(len(labels))
width=0.6
fig, ax = plt.subplots(figsize=(6,5))
rects = ax.barh(ind, np.array(values), color='purple')
ax.set_yticks(ind+((width)/2.))
ax.set_yticklabels(labels, rotation='horizontal')
ax.set_xlabel("Count of missing values")
ax.set_ylabel("Column Names")
ax.set_title("Columns with missing values in train dataset");
#lets see which are the columns with missing values in test dataset
test_df.isnull().sum()
labels = []
values = []
null_columns = test_df.columns[test_df.isnull().any()]
for col in null_columns:
    labels.append(col)
    values.append(test_df[col].isnull().sum())

ind = np.arange(len(labels))
width=0.6
fig, ax = plt.subplots(figsize=(6,5))
rects = ax.barh(ind, np.array(values), color='purple')
ax.set_yticks(ind+((width)/2.))
ax.set_yticklabels(labels, rotation='horizontal')
ax.set_xlabel("Count of missing values")
ax.set_ylabel("Column Names")
ax.set_title("Columns with missing values in test dataset");
#Lets check which rows have null Embarked column
train_df[train_df['Embarked'].isnull()]
from numpy import median
sns.barplot(x="Embarked", y="Fare", hue="Pclass", data=train_df, estimator=median)
train_df["Embarked"] = train_df["Embarked"].fillna('C')
#Lets check which rows have null Fare column in test dataset
test_df[test_df['Fare'].isnull()]
# we can replace missing value in fare by taking median of all fares of those passengers
# who share 3rd Passenger class and Embarked from 'S' , so lets find out those rows
test_df[(test_df['Pclass'] == 3) & (test_df['Embarked'] == 'S')].head()
# now lets find the median of fare for those passengers.

def fill_missing_fare(df):
    median_fare = df[(df['Pclass'] == 3) & (df['Embarked'] == 'S')]['Fare'].median()
    df["Fare"] = df["Fare"].fillna(median_fare)
    return df

test_df = fill_missing_fare(test_df)
# age distribution in train dataset
with sns.plotting_context("notebook",font_scale=1.5):
    sns.set_style("whitegrid")
    sns.distplot(train_df["Age"].dropna(),
                 bins=80,
                 kde=False,
                 color="red")
    plt.title("Age Distribution")
    plt.ylabel("Count")
# age distribution in test dataset
with sns.plotting_context("notebook",font_scale=1.5):
    sns.set_style("whitegrid")
    sns.distplot(test_df["Age"].dropna(),
                 bins=80,
                 kde=False,
                 color="red")
    plt.title("Age Distribution")
    plt.ylabel("Count")
# predicting missing values in age using Random Forest
# import the RandomForestRegressor Object
from sklearn.ensemble import RandomForestRegressor

def fill_missing_age(df):
    
    #Feature set
    age_df = df[['Age','Pclass','SibSp','Parch','Fare']]
    
    # Split sets into train and test
    train  = age_df.loc[ (df.Age.notnull()) ]# known Age values
    test = age_df.loc[ (df.Age.isnull()) ]# null Ages
    
    # All age values are stored in a target array
    y = train.values[:, 0]
    
    # All the other values are stored in the feature array
    X = train.values[:, 1::]
    
    # Create and fit a model
    rtr = RandomForestRegressor(n_estimators=2000, n_jobs=-1)
    rtr.fit(X, y)
    
    # Use the fitted model to predict the missing values
    predictedAges = rtr.predict(test.values[:, 1::])
    
    # Assign those predictions to the full data set
    df.loc[ (df.Age.isnull()), 'Age' ] = predictedAges 
    
    return df
train_df = fill_missing_age(train_df)
test_df = fill_missing_age(test_df)
train_df["Deck"] = train_df.Cabin.str[0] # the first character denotes which deck the passenger is allocated
test_df["Deck"] = test_df.Cabin.str[0]
train_df["Deck"].unique() # 0 is for null values
sns.set(font_scale=1)
g = sns.factorplot("Survived", col="Deck", col_wrap=4,
                    data=train_df[train_df.Deck.notnull()],
                    kind="count", size=2.5, aspect=.8);
train_df.Deck.fillna('Z', inplace=True)
test_df.Deck.fillna('Z', inplace=True)

sorted_deck_values = train_df["Deck"].unique() # Z is for null values
sorted_deck_values.sort()
sorted_deck_values
# Drop Cabin Feature from final dataset for test and train
train_df = train_df.drop(['Cabin'], axis=1)
test_df = test_df.drop(['Cabin'], axis=1)
data = [train_df, test_df]

for dataset in data:
    dataset['relatives'] = dataset['SibSp'] + dataset['Parch']
    dataset.loc[dataset['relatives'] > 0, 'not_alone'] = 0
    dataset.loc[dataset['relatives'] == 0, 'not_alone'] = 1
    dataset['not_alone'] = dataset['not_alone'].astype(int)

train_df['not_alone'].value_counts()
sns.set(font_scale=1)
sns.factorplot("Survived", col="relatives", col_wrap=4,
                    data=train_df,
                    kind="count", size=2.5, aspect=.8);
sns.catplot('relatives','Survived',kind='point', 
                      data=train_df, aspect = 2.5, )
data = [train_df, test_df]

for dataset in data:
    # extract titles
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    # replace titles with a more common title or as Rare
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr',\
                                            'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
# Drop Name Feature from final dataset for test and train
train_df = train_df.drop(['Name'], axis=1)
test_df = test_df.drop(['Name'], axis=1)
train_df.head()
data = [train_df, test_df]
titles = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

for dataset in data:
    # convert titles into numbers
    dataset['Title'] = dataset['Title'].map(titles)
    
    # filling NaN with 0, to be safe
    dataset['Title'] = dataset['Title'].fillna(0)
genders = {"male": 0, "female": 1}
data = [train_df, test_df]

for dataset in data:
    dataset['Sex'] = dataset['Sex'].map(genders)
deck = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "T": 8, "Z": 9}
data = [train_df, test_df]

for dataset in data:
    dataset['Deck'] = dataset['Deck'].map(deck)
ports = {"S": 0, "C": 1, "Q": 2}
data = [train_df, test_df]

for dataset in data:
    dataset['Embarked'] = dataset['Embarked'].map(ports)
train_df['Ticket'].describe()
train_df = train_df.drop(['Ticket'], axis=1)
test_df = test_df.drop(['Ticket'], axis=1)
from sklearn.preprocessing import StandardScaler

data = [train_df, test_df]

for dataset in data:
    std_scale = StandardScaler().fit(dataset[['Age', 'Fare']])
    dataset[['Age', 'Fare']] = std_scale.transform(dataset[['Age', 'Fare']])
train_df.head()
test_df.head()
X_train = train_df.drop(['Survived', 'PassengerId'], axis=1)
y_train = train_df["Survived"]
X_test  = test_df.drop("PassengerId", axis=1)
# Now that we have our dataset, we can start building algorithms! 
# We'll need to import each algorithm we plan on using

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB # (Naive Bayes)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# define scoring method
scoring = 'accuracy'

# Define models to train

names = ["Random Forest", "AdaBoost","Nearest Neighbors", 
         "Naive Bayes","Decision Tree","Logistic Regression","Gaussian Process",
         "SVM RBF", "SVM Linear", "SVM Sigmoid"]

classifiers = [
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    AdaBoostClassifier(),
    KNeighborsClassifier(n_neighbors = 3),
    GaussianNB(),
    DecisionTreeClassifier(max_depth=5),
    LogisticRegression(random_state = 0),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    SVC(kernel = 'rbf', random_state=0),
    SVC(kernel = 'linear', random_state=0),
    SVC(kernel = 'sigmoid', random_state=0)
]

models = dict(zip(names, classifiers))
# Lets determine the accuracy score of each classifier

models_accuracy_score = []

for name in models:
    model = models[name]
    model.fit(X_train, y_train)
    predictions = model.predict(X_train)
    models_accuracy_score.append((name, round(accuracy_score(y_train, predictions)* 100, 2)))
results = pd.DataFrame({
    'Model': names,
    'Score': [curr_model_score[1] for curr_model_score in models_accuracy_score]})
result_df = results.sort_values(by='Score', ascending=False)
result_df = result_df.set_index('Score')
result_df
# import K-fold class
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

# define seed for reproducibility
seed = 1

kfold = KFold(n_splits=10, random_state = seed)
model = KNeighborsClassifier(n_neighbors = 3)
cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)

print("Model Name: KNN")
print("Scores:", cv_results)
print("Mean:", cv_results.mean())
print("Standard Deviation:", cv_results.std())
# Applying Grid Search to find the best parameters
from sklearn.model_selection import GridSearchCV

classifier = KNeighborsClassifier()

parameters = [{'n_neighbors': np.arange(1,30), 'algorithm': ['ball_tree', 'kd_tree', 'brute'], 
               'leaf_size': np.arange(1,30), 'p': [1,2]}
             ]

grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)

grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_


print("Accuracy: ", round(best_accuracy * 100, 2))
print("Best Params: ", best_parameters)
# perform classification again using optimal parameter from grid search cv
classifier = grid_search.best_estimator_
classifier.fit(X_train, y_train)
y_train_predictions = classifier.predict(X_train)
y_test_predictions = classifier.predict(X_test)

print("Model accuracy score: ", round(accuracy_score(y_train, y_train_predictions) * 100, 2))
# confusion matrix
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix

confusion_matrix(y_train, y_train_predictions)
submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": y_test_predictions
    })
submission.to_csv('submission.csv', index=False)
output = pd.DataFrame({ 'PassengerId' : test_df["PassengerId"], 'Survived': y_test_predictions })
output.head()