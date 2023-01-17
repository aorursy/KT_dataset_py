# pandas to open data files & processing it.

import pandas as pd



# numpy for numeric data processing

import numpy as np



# sklearn to do preprocessing & ML models

from sklearn import preprocessing

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.metrics import accuracy_score

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier



# Matplotlob & seaborn to plot graphs & visulisation

import matplotlib.pyplot as plt 

import seaborn as sns



# to fix random seeds

import random, os



# ignore warnings

import warnings

warnings.simplefilter(action='ignore')
train_df = pd.read_csv('../input/titanic/train.csv')

predict_df = pd.read_csv('../input/titanic/test.csv')
# Merge both datasets to work with all data at once

df=pd.concat([train_df,predict_df], ignore_index=True, sort =False)

df.head()
df.describe().round(2)
#NaN values with %, we have some NaN, later we will deal with that...

total = df.isnull().sum().sort_values(ascending=False)

percent_1 = df.isnull().sum()/df.isnull().count()*100

percent_2 = (round(percent_1, 1)).sort_values(ascending=False)

missing_data = pd.concat([total, percent_2], axis=1, keys=['Total', '%'])

missing_data.head(5)
grid = sns.FacetGrid(df, row='Pclass', size=2.2, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=.5, bins=20)

grid.add_legend();
bp = df.boxplot(column='Age', by='Pclass', grid=False)



for i in [1,2,3]:

    y = df.Age[df.Pclass==i].dropna()

    # Add some random "jitter" to the x-axis

    x = np.random.normal(i, 0.04, size=len(y))

    plt.plot(x, y, 'r.', alpha=0.2)
a = sns.countplot(x='Embarked',data=df,hue='Pclass')
df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
a = sns.countplot(x='Pclass',data=df,hue='Survived')
df[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
#sex to categorical value

df['Sex'] = df['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
sns.barplot(x="Sex", y="Survived", data=df)

plt.legend(title='Survived', loc='upper left', labels=['0 = MALE', '1= FEMALE'])

sns.catplot(x="Sex", y="Survived", data=df, col="Pclass", kind="bar",height=4, aspect=.7);
corr = df.corr()

corr.style.background_gradient(cmap='coolwarm')
g = sns.FacetGrid(df, col='Survived',height=3, aspect=2)

g.map(plt.hist, 'Fare', bins=20)

g = sns.FacetGrid(df, col='Survived',height=3, aspect=2)

g.map(plt.hist, 'Age', bins=20)
df['family_members'] = df['Parch'] + df['SibSp'] + 1



# if total family size is 1, person is alone.

df['alone'] = df['family_members'].apply(lambda x: 0 if x > 1 else 1)



a =sns.barplot(df['family_members'], df['Survived'])
a = sns.barplot(df['alone'], df['Survived'])
#get the word before the dot and put it into new column "name_title"

df['name_title'] = df['Name'].str.extract('([A-Za-z]+)\.', expand=False)
df['name_title'].value_counts()
def clean_name_title(val):

    if val in ['Rev', 'Col', 'Mlle', 'Mme', 'Ms','Major', 'Jonkheer', 'Countess', 'Capt','Lady','Dona','Dr']:

        return 'RARE'   

    elif val in ['Sir', 'Don']:

        return 'Mr'

    elif val in ['Lady', 'Dona']:

        return 'Mrs'

    else:

        return val



df['name_title'] = df['name_title'].apply(clean_name_title)

df['name_title'].value_counts()
a = sns.barplot(df['name_title'], df['Survived'])
a = df.groupby(['Sex','Pclass','name_title']).mean().round(2)

a = a['Age']
c = a.unstack(level=0).plot(kind='bar', subplots=False)
b = df.iloc[:891].groupby(['Sex','Pclass','name_title'])

b_median = b.median()

b = b_median.reset_index()[['Sex', 'Pclass', 'name_title', 'Age']]



b.head()
for key, value in df['Age'].iteritems(): 

    if pd.isna(df['Age'][key]):

        for key2, value2 in b['Age'].iteritems():

            if df['Sex'][key] == b['Sex'][key2] and df['name_title'][key] == b['name_title'][key2] and df['Pclass'][key] == b['Pclass'][key2]:

                df['Age'][key] = b['Age'][key2]

    else:

        df['Age'][key] = df['Age'][key]
# Explore Fare distribution 

g = sns.distplot(df["Fare"], color="m", label="Skewness : %.2f"%(df["Fare"].skew()))

g = g.legend(loc="best")
# Apply log to Fare to reduce skewness distribution

df["Fare_log"] = df["Fare"].map(lambda i: np.log(i) if i > 0 else 0)



g = sns.distplot(df["Fare_log"], color="b", label="Skewness : %.2f"%(df["Fare_log"].skew()))

g = g.legend(loc="best")
# Let's do another column with fare limited to 100, because it can be a problem with mean NaN values.

def group_fare(fare):

    if fare < 100:

        return fare

    else:

        return 100



df['Fare_limited'] = df['Fare'].apply(group_fare)

df["Fare_log2"] = df["Fare_limited"].map(lambda i: np.log(i) if i > 0 else 0)



g = sns.distplot(df["Fare_log2"], color="b", label="Skewness : %.2f"%(df["Fare_log2"].skew()))

g = g.legend(loc="best")
df['has_cabin'] = 0



for Key, value in df['Cabin'].iteritems(): 

    if pd.isna(df['Cabin'][Key]):

        df['has_cabin'][Key] = int(0)

    else:

        df['has_cabin'][Key] = int(1)
df['Embarked'].fillna("S", inplace=True)

df['Fare'].fillna(df['Fare_limited'].mean(), inplace=True)

df['Age'].fillna(df['Age'].mean(), inplace=True)
le = preprocessing.LabelEncoder()

df['Embarked'] = le.fit_transform(df['Embarked'])

df['Sex'] = le.fit_transform(df['Sex'])

df['name_title'] = le.fit_transform(df['name_title'])
#cut database in two

train_df = df[0:891]

pred_df = df[-418::]
train_df['Survived'] = train_df['Survived'].astype(int)
# let's see again the correlations to select features

corr = train_df.corr().round(2)

corr['Survived'].sort_values()
#select columns for the model

features_X = train_df[['Sex','Age','Pclass','Fare_limited','alone','family_members','name_title']]

target_Y = train_df['Survived']

predict_X = pred_df[['Sex','Age','Pclass','Fare_limited','alone','family_members','name_title']]
train_data, val_data, train_target, val_target = train_test_split(features_X, target_Y, test_size=0.4, random_state=42)

train_data.shape, val_data.shape, len(train_target), len(val_target)
# Train Random Forest

model = RandomForestClassifier(n_estimators=1000,min_samples_split=8, min_samples_leaf=4)

model.fit(train_data, train_target)



# Predict labels on Validation data which model have never seen before.

val_predictions = model.predict(val_data)

accuracy_score(val_target, val_predictions)
from sklearn.metrics import plot_roc_curve, mean_absolute_error, r2_score, mean_squared_error



rmse = np.sqrt(mean_squared_error(val_target, val_predictions))

print("RMSE: %f" % (rmse))

print("MAE: " + str(mean_absolute_error(val_predictions, val_target)))

print("R2:" + str(r2_score(val_target, val_predictions)))
svc_disp = plot_roc_curve(model, train_data, train_target)

plt.show()
test_df = pd.read_csv('../input/titanic/test.csv')

predict_Y = model.predict(predict_X)



submissionRF = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": predict_Y

    })

submissionRF.to_csv('gender_submission.csv', index=False)
def algorithm_pipeline(X_train_data, X_test_data, y_train_data, y_test_data, 

                       model, param_grid, cv=10, scoring_fit='neg_mean_squared_error',

                       do_probabilities = False):

    gs = GridSearchCV(

        estimator=model,

        param_grid=param_grid, 

        cv=cv, 

        n_jobs=-1, 

        scoring=scoring_fit,

        verbose=2

    )

    fitted_model = gs.fit(X_train_data, y_train_data)

    

    if do_probabilities:

      pred = fitted_model.predict_proba(X_test_data)

    else:

      pred = fitted_model.predict(X_test_data)

    

    return fitted_model, pred
model = RandomForestClassifier()



# Various hyper-parameters to tune

parameters = {

              'min_samples_split': [4,8],

              'min_samples_leaf': [2,4,8],

              'criterion': ['gini', 'entropy'],

              'n_estimators': [100]}



model, pred = algorithm_pipeline(train_data, val_data, train_target, val_target, model, 

                                 parameters, cv=5)



# Root Mean Squared Error

print(model.best_params_)

# evaluate predictions

rmse = np.sqrt(-model.best_score_)

print("RMSE: %f" % (rmse))
import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.metrics import accuracy_score, log_loss

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

from sklearn.linear_model import LogisticRegression



classifiers = [

    KNeighborsClassifier(),

    SVC(probability=True),

    DecisionTreeClassifier(),

    RandomForestClassifier(),

    AdaBoostClassifier(),

    GradientBoostingClassifier(),

    GaussianNB(),

    LinearDiscriminantAnalysis(),

    QuadraticDiscriminantAnalysis(),

    LogisticRegression()]



log_cols = ["Classifier", "Accuracy"]

log 	 = pd.DataFrame(columns=log_cols)



sss = StratifiedShuffleSplit(n_splits=8, test_size=0.5, random_state=1)



X = features_X.values

y = target_Y.values



acc_dict = {}



for train_index, test_index in sss.split(X, y):

    X_train, X_test = X[train_index], X[test_index]

    y_train, y_test = y[train_index], y[test_index]

    

    for clf in classifiers:

        name = clf.__class__.__name__

        clf.fit(X_train, y_train)

        train_predictions = clf.predict(X_test)

        acc = accuracy_score(y_test, train_predictions)

        if name in acc_dict:

            acc_dict[name] += acc

        else:

            acc_dict[name] = acc



for clf in acc_dict:

    acc_dict[clf] = acc_dict[clf] / 10.0

    log_entry = pd.DataFrame([[clf, acc_dict[clf]]], columns=log_cols)

    log = log.append(log_entry)



log.sort_values('Accuracy', ascending=False)
# Train Random Forest

model = GradientBoostingClassifier(learning_rate=0.005,n_estimators=1000, min_samples_split=2, min_samples_leaf=2)

model.fit(train_data, train_target)



# Predict labels on Validation data which model have never seen before.

val_predictions = model.predict(val_data)

accuracy_score(val_target, val_predictions)
from sklearn.metrics import plot_roc_curve, mean_absolute_error, r2_score, mean_squared_error



rmse = np.sqrt(mean_squared_error(val_target, val_predictions))

print("RMSE: %f" % (rmse))

print("MAE: " + str(mean_absolute_error(val_predictions, val_target)))

print("R2:" + str(r2_score(val_target, val_predictions)))
'''test_df = pd.read_csv('../input/titanic/test.csv')

predict_Y = model.predict(predict_X)



submissionRF = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": predict_Y

    })

submissionRF.to_csv('gender_submission2.csv', index=False)'''
import keras 

from keras.models import Sequential # intitialize the ANN

from keras.layers import Dense      # create layers



# Initialising the NN

model = Sequential()



# layers

model.add(Dense(units = 7, kernel_initializer = 'uniform', activation = 'relu', input_dim = 7))

model.add(Dense(units = 9, kernel_initializer = 'uniform', activation = 'relu'))

model.add(Dense(units = 5, kernel_initializer = 'uniform', activation = 'relu'))

model.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))



# Compiling the ANN

keras.optimizers.Adam(learning_rate=0.0009)

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])



# Train the ANN

model.fit(features_X, target_Y, batch_size = 32, epochs = 200)

#model.fit(train_data, train_target, batch_size = 32, epochs = 500)
y_pred = model.predict(features_X)

y_pred = (y_pred > 0.5).astype(int).reshape(features_X.shape[0])

accuracy_score(target_Y, y_pred)
y_pred = model.predict(predict_X)

y_final = (y_pred > 0.5).astype(int).reshape(predict_X.shape[0])
'''temp = pd.DataFrame(pd.read_csv("../input/titanic/test.csv")['PassengerId'])

temp['Survived'] = y_final

temp.to_csv("submission_neural.csv", index = False)

submissionNN = temp'''