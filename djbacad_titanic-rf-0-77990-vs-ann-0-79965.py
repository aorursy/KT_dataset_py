import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

import os

warnings.filterwarnings("ignore")
train = pd.read_csv('../input/titanic/train.csv',sep = ",", header = 0)

test = pd.read_csv('../input/titanic/test.csv',sep = ",", header = 0)
train.head()
test.head()
train.info()
test.info()
#Inject SalePrice in test

test['Survived'] = 0

print("Train Shape: " + str(train.shape))

print("Test Shape: " + str(test.shape))
full = pd.concat([train,test])

full.shape
full.head()
full['Sex'] = full['Sex'].map({'male':0, 'female':1})
full.info()
full.drop(['Ticket','Cabin'], axis = 1, inplace = True)
sns.set_palette("GnBu_d")

plt.title("Missingess Map")

plt.rcParams['figure.figsize'] = (8.0, 5.0) #Adjust values as necessary

sns.heatmap(full.isnull(), cbar=False)
full['Age'].replace('',np.nan,inplace=True)

full['Age'].fillna(value=full['Age'].mean(),inplace = True)

full['Fare'].replace('',np.nan,inplace=True)

full['Fare'].fillna(value=full['Fare'].mean(),inplace = True)

full[['Age','Fare']].info()
full['Embarked'].fillna(value=full['Embarked'].value_counts().idxmax(),inplace = True)

full[['Embarked']].info()
full.info()
full['Survived'] = full['Survived'].astype('category')

full['Sex'] = full['Sex'].astype('category')

full['Embarked'] = full['Embarked'].astype('category')
full.info()
train = full[0:891]

test = full[891:1309]
fig, axs = plt.subplots(ncols=4,nrows=1, figsize = (20,5),squeeze = False)

sns.countplot(x = "Survived", data = train, ax = axs[0][0])

sns.countplot(x = "Pclass", data = train, ax = axs[0][1])

sns.countplot(x = "Embarked", data = train, ax = axs[0][2])

sns.countplot(x = "Sex", data = train, ax = axs[0][3])
colors = ["#B83636", "#2FB756"]

sns.set_palette(sns.color_palette(colors))

fig, axs = plt.subplots(ncols=3,nrows=1, figsize = (20,5), squeeze = False)

sns.countplot(x = 'Pclass', data = train, hue = 'Survived', ax=axs[0][0])

sns.countplot(x = 'Embarked', data = train, hue = 'Survived', ax=axs[0][1])

sns.countplot(x = 'Sex', data = train, hue = 'Survived', ax=axs[0][2])
fig, axs = plt.subplots(ncols=4,nrows=2, figsize = (20,10))

sns.boxplot(x="Survived", y="Age", data = train, ax=axs[0][0])

sns.stripplot(x='Survived',y='Age', data=train, jitter=True, ax=axs[0][0])



sns.boxplot(x="Survived", y="Fare", data = train, ax=axs[0][1])

sns.stripplot(x='Survived',y='Fare', data=train, jitter=True, ax=axs[0][1])



sns.boxplot(x="Survived", y="SibSp", data = train, ax=axs[0][2])

sns.stripplot(x='Survived',y='SibSp', data=train, jitter=True, ax=axs[0][2])



sns.boxplot(x="Survived", y="Parch", data = train, ax=axs[0][3])

sns.stripplot(x='Survived',y='Parch',  data=train, jitter=True, ax=axs[0][3])



sns.violinplot(x="Survived", y="Age", data = train, ax=axs[1][0])

sns.violinplot(x="Survived", y="Fare", data = train, ax=axs[1][1])

sns.violinplot(x="Survived", y="SibSp", data = train, ax=axs[1][2])

sns.violinplot(x="Survived", y="Parch", data = train, ax=axs[1][3])
fig, axs = plt.subplots(ncols=3,nrows=2, figsize = (20,12), squeeze = False)

sns.distplot(train[(train['Survived'] == 0)]['Age'], kde=False, color = "#2FB756", ax=axs[0][0])

sns.distplot(train[(train['Survived'] == 1)]['Age'], kde=False,color = "#B83636", ax=axs[0][1])

sns.distplot(train[(train['Survived'] == 0)]['Age'], kde=True, color = "#2FB756", ax=axs[0][2])

sns.distplot(train[(train['Survived'] == 1)]['Age'], kde=True,color = "#B83636", ax=axs[0][2])

sns.distplot(train[(train['Survived'] == 0)]['Fare'], kde=False, color = "#2FB756", ax=axs[1][0])

sns.distplot(train[(train['Survived'] == 1)]['Fare'], kde=False, color = "#B83636", ax=axs[1][1])

sns.distplot(train[(train['Survived'] == 0)]['Fare'], kde=True, color = "#2FB756", ax=axs[1][2])

sns.distplot(train[(train['Survived'] == 1)]['Fare'], kde=True, color = "#B83636", ax=axs[1][2])
train['0to18'] = train['Age'].apply(lambda x: 1 if x <= 18 else 0)

train['18to50'] = train['Age'].apply(lambda x: 1 if (x > 18 and x <=50) else 0)

train['50above'] = train['Age'].apply(lambda x: 1 if x > 50 else 0)

test['0to18'] = test['Age'].apply(lambda x: 1 if x <= 18 else 0)

test['18to50'] = test['Age'].apply(lambda x: 1 if (x > 18 and x <=50) else 0)

test['50above'] = test['Age'].apply(lambda x: 1 if x > 50 else 0)
train.info()
test.info()
fig, axs = plt.subplots(ncols=3,nrows=1, figsize = (20,5), squeeze = False)

sns.countplot(x = '0to18', data = train, hue = 'Survived', ax=axs[0][0])

sns.countplot(x = '18to50', data = train, hue = 'Survived', ax=axs[0][1])

sns.countplot(x = '50above', data = train, hue = 'Survived', ax=axs[0][2])
def extract_titles(name):

    if '.' in name:

        return name.split(',')[1].split('.')[0].strip()

    else:

        return 'Unknown'

    

def replace_titles(x):

    title = x['Title']

    if title in ['Capt', 'Col', 'Don', 'Jonkheer', 'Major', 'Rev', 'Sir']:

        return 'Mr'

    elif title in ['the Countess', 'Mme', 'Lady']:

        return 'Mrs'

    elif title in ['Mlle', 'Ms']:

        return 'Miss'

    elif title =='Dr':

        if x['Sex']=='male':

            return 'Mr'

        else:

            return 'Mrs'

    else:

        return title



train['Title'] = train['Name'].map(lambda x: extract_titles(x))

train['Title'] = train.apply(replace_titles, axis=1)

test['Title'] = test['Name'].map(lambda x: extract_titles(x))

test['Title'] = test.apply(replace_titles, axis=1)
sns.countplot(x = 'Title', data = train, hue = 'Survived')
train['FamilySize'] = train['SibSp'] + train['Parch'] + 1

test['FamilySize'] = test['SibSp'] + test['Parch'] + 1
sns.countplot(x = 'Title', data = train, hue = 'Survived')
print(train.info())

print(test.info())
todrop = ['Age','SibSp','Parch','Name','PassengerId']

todrop2 = ['Age','SibSp','Parch','Name','Survived']

train.drop(todrop, axis = 1, inplace = True)

test.drop(todrop2, axis = 1, inplace = True)

train['0to18'] = train['0to18'].astype('category')

test['0to18'] = test['0to18'].astype('category')

train['18to50'] = train['18to50'].astype('category')

test['18to50'] = test['18to50'].astype('category')

train['50above'] = train['50above'].astype('category')

test['50above'] = test['50above'].astype('category')

print(train.info())

print(test.info())
from sklearn.preprocessing import OneHotEncoder

todummify = list(train.select_dtypes(include=['object','category']).columns)

binaord = {'0to18','18to50','50above','Survived','Sex'}

todummify = [var for var in todummify if var not in binaord]

enc = OneHotEncoder(handle_unknown='ignore')

enc_train = pd.DataFrame(enc.fit_transform(train[todummify]).toarray(),

                      columns=enc.get_feature_names(todummify))

train = train.join(enc_train,how='inner')

train.drop(todummify, axis = 1, inplace = True )



tocategorify = [col for col in train.columns if '_' in col]

train[tocategorify] = train[tocategorify].astype('category')





enc_test = pd.DataFrame(enc.transform(test[todummify]).toarray(),

                      columns=enc.get_feature_names(todummify))

test = test.join(enc_test,how='inner')

test.drop(todummify, axis = 1, inplace = True )



tocategorify = [col for col in test.columns if '_' in col]

test[tocategorify] = test[tocategorify].astype('category')





print(train.info(verbose=True))

print(test.info(verbose=True))
X = train.drop(['Survived'], axis = 1)

y = train['Survived']

print("Dependent Variables")

display(X.head())

print("Independent Variable")

display(y.to_frame().head())
from statsmodels.stats.outliers_influence import variance_inflation_factor

from statsmodels.tools.tools import add_constant

X_numeric = X._get_numeric_data() #drop non-numeric cols

X_numeric = add_constant(X_numeric)

VIF_frame = pd.Series([variance_inflation_factor(X_numeric.values, i) 

               for i in range(X_numeric.shape[1])], 

              index=X_numeric.columns).to_frame()



VIF_frame.drop('const', axis = 0, inplace = True) 

VIF_frame.rename(columns={VIF_frame.columns[0]: 'VIF'},inplace = True)

VIF_frame[~VIF_frame.isin([np.nan, np.inf, -np.inf]).any(1)]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, 

                                                    test_size = 0.25, 

                                                    random_state = 823)

print("X_train")

print(X_train.head())

print(" ")

print("X_test")

print(X_test.head())
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(random_state = 823)

rfc.fit(X_train,y_train)

features = X_train.columns.tolist()

feature_value = rfc.feature_importances_

d = {'Features' : features, 'Values' : feature_value}

fi = pd.DataFrame(d).sort_values('Values', ascending = False).reset_index()

fi

plt.rcParams['figure.figsize'] = (25, 5.0)

ax = sns.barplot(x=fi['Features'], y = fi['Values'], data = fi, palette="Blues_d")
from sklearn.model_selection import GridSearchCV



parameters = [{'n_estimators' : [10000], #number of trees in the forest

               'criterion': ['gini'], #gini or entropy

               'max_depth': [4], #The maximum depth of the tree. 

               'min_samples_split': [2], #The minimum number of samples required to split an internal node.

               'min_samples_leaf': [2], #The minimum number of samples required to be at a leaf node.

               'max_features': [5], #The number of features to consider when looking for the best split:

               'max_leaf_nodes': [7]}] #Grow a tree with max_leaf_nodes in best-first fashion. Best nodes are defined as relative reduction in impurity. If None then unlimited number of leaf nodes.

               #'ccp_alpha': np.arange(0.0,,0.01)}]

rf_clf = GridSearchCV(rfc, parameters,scoring = 'accuracy', cv = 10)

rf_clf.fit(X_train,y_train)

print("Best Parameter Values: ")

pd.DataFrame.from_dict(rf_clf.best_params_,orient='index',columns=['Values'])
best_rfc_model = rf_clf.best_estimator_

best_rfc_model.fit(X_train,y_train)

predictions = best_rfc_model.predict(X_test)

predictions
from sklearn.metrics import classification_report,confusion_matrix

data = confusion_matrix(y_test, predictions)

df_cm = pd.DataFrame(data, columns=np.unique(y_test), index = np.unique(y_test))

df_cm.index.name = 'Predicted'

df_cm.columns.name = 'Actual'

plt.figure(figsize = (10,7))

sns.set(font_scale=1.5)

ax = sns.heatmap(df_cm,cmap = 'Greens', annot=True,annot_kws={"size": 16}, fmt='g')

ax.set_title('Confusion Matrix')

print("Classification Report: ")

print(classification_report(y_test,predictions))
import scikitplot as skplt

import matplotlib.pyplot as plt

y_true = y_test

y_probas = best_rfc_model.predict_proba(X_test)

skplt.metrics.plot_roc(y_true, y_probas, 

                             title = 'ROC Curve',

                             figsize = (12,8))

plt.grid(b = 'Whitegrid')
from sklearn.metrics import accuracy_score

print("Test Accuracy: " + str("{:.2f}".format(accuracy_score(y_test, predictions))))

predictions2 = best_rfc_model.predict(X_train)

print("Train Accuracy: " + str("{:.2f}".format(accuracy_score(y_train, predictions2))))
Survived = pd.Series(best_rfc_model.predict(test.drop('PassengerId',axis=1)),name='Survived')

my_solution = pd.concat([test['PassengerId'],Survived], axis=1)

my_solution.to_csv('my_output_rfc.csv',index=False)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_numeric = X_train._get_numeric_data()

X_test_numeric = X_test._get_numeric_data()

test_copy = test.copy()

test_copy.drop('PassengerId', axis = 1, inplace = True)

test_numeric = test_copy._get_numeric_data()

X_train_numeric_scaled = pd.DataFrame(scaler.fit_transform(X_train_numeric), 

                                      index=X_train.index,

                                      columns=X_train_numeric.columns)

X_test_numeric_scaled = pd.DataFrame(scaler.transform(X_test_numeric), 

                                     index = X_test.index, 

                                     columns=X_test_numeric.columns)

test_numeric_scaled = pd.DataFrame(scaler.transform(test_numeric), 

                                     index = test.index, 

                                     columns = test_numeric.columns)

X_train.update(X_train_numeric_scaled)

X_test.update(X_test_numeric_scaled)

test.update(test_numeric_scaled)

display(X_train.head())

display(X_test.head())

display(test.head())
import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Activation,Dropout

from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.layers import Dropout



model = Sequential()



##### Some References:

# https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw

# https://stats.stackexchange.com/questions/164876/tradeoff-batch-size-vs-number-of-iterations-to-train-a-neural-network

# https://datascience.stackexchange.com/questions/18414/are-there-any-rules-for-choosing-the-size-of-a-mini-batch



#Initializer

model = Sequential()



#Input Layer

model.add(Dense(units = 14,activation='relu'))

model.add(Dropout(0.5))



#Hidden Layer

model.add(Dense(units = 9,activation='relu'))

model.add(Dropout(0.5))



#Output Layer

model.add(Dense(units = 1,activation='sigmoid'))



#For binary classification problem, loss function is 'binary_crossentropy'

model.compile(loss='binary_crossentropy', optimizer='adam')



#EarlyStopping

early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience = 25)



model.fit(x=X_train, 

          y=y_train, 

          epochs=5000,

          validation_data=(X_test, y_test), verbose=1,

          callbacks=[early_stop])
model.summary()
model_loss = pd.DataFrame(model.history.history)

model_loss.plot()
predictions = (model.predict(X_test) > 0.5).astype("int32")

predictions[0:5]
from sklearn.metrics import classification_report,confusion_matrix

data = confusion_matrix(y_test, predictions)

df_cm = pd.DataFrame(data, columns=np.unique(y_test), index = np.unique(y_test))

df_cm.index.name = 'Predicted'

df_cm.columns.name = 'Actual'

plt.figure(figsize = (10,7))

sns.set(font_scale=1.5)

ax = sns.heatmap(df_cm,cmap = 'Greens', annot=True,annot_kws={"size": 16}, fmt = 'g')# font size

ax.set_title('Confusion Matrix')

print("Classification Report: ")

print(classification_report(y_test,predictions))
Survived = (model.predict(test.drop('PassengerId',axis=1)) > 0.5).astype("int32")

Survived = Survived.flatten()

Survived = pd.Series(Survived,name='Survived')

my_solution = pd.concat([test['PassengerId'],Survived], axis=1)

my_solution.to_csv('my_output_ann.csv',index=False)