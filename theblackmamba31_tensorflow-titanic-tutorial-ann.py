import pandas as pd

import numpy as np

import seaborn as sns 

import matplotlib.pyplot as plt

%matplotlib inline
 

train_data = pd.read_csv('../input/titanic/train.csv')

test_data = pd.read_csv('../input/titanic/test.csv')
# to print first five rows of train data

train_data.head()
# to print first five rows of test data

test_data.head()
print("Total number of rows in training data ", train_data.shape[0])

print("Total number of columns in training data ", train_data.shape[1])

print("Total number of rows in test data ", test_data.shape[0])

print("Total number of columns in test data ", test_data.shape[1])

plt.figure(figsize = (13,5))

plt.bar(train_data.columns, train_data.isna().sum())

plt.xlabel("Columns name")

plt.ylabel("Number of missing values in training data")

plt.show()

# from the bar plot of missing value we can conclude that Cabin, Embarked and Cabin column has null value so, we 

# can either drop the entire row or can fill the nan value with some values like mean, meadian. 
plt.figure(figsize = (13,5))

plt.bar(test_data.columns, test_data.isnull().sum().values, color = 'red')

plt.xlabel("Columns name")

plt.ylabel("Number of missing values in test data")

plt.show()

# similarly we can conclude that Age Cabin and Fare column has nan values . 
#Visualizing the Number of survived passenger

sns.countplot('Survived', data = train_data)

plt.show()

# here we plot only for train_data as we donot have Survived column for test data,

# This plot show that around 600 people died while around 300 survived
# visualizing the number of passenger from different embarked column in train_data

sns.countplot('Embarked', data = train_data)

plt.show()
#visualizing whether gender affect the survival rate or not

sns.countplot('Survived', hue = 'Sex', data = train_data)

plt.plot()

# the graph clearly show that death rate for male passenger is way more than that for female

# visualizing whether pclass affect the survial rate or not

sns.countplot("Survived", hue = 'Pclass', data = train_data)

plt.show()

# this graph clearly show that people in third class are more likely to die 
# visualizing whether embarked place affects the survival rate or not

sns.countplot('Survived', hue = 'Embarked', data = train_data)

plt.show()
sns.boxplot('Fare', data = train_data)

plt.show()

# this shows that there were very few people who payed more than 100
sns.boxplot('Age', data = train_data)

plt.show()

# this shows that there were very few people more than 65 years old in training data
# ploting histogram

# choosing value for bin 

interval = 10

value_for_bin = np.ceil((train_data.Age.max() - train_data.Age.min()) / interval).astype(int)



plt.hist(train_data.Age, bins = value_for_bin)

plt.xlabel("Age")

plt.ylabel("Number")

plt.show()

# this shows that lots of passenger we from age between 20 to 40
plt.figure(figsize = (10,4))

plt.hist(train_data.Fare, bins = 10, color = 'lime')

plt.xlabel("Fare")

plt.ylabel("Number")

plt.show()

# this shows that around 700 people pay in between 0 and 50
grid = sns.FacetGrid(train_data, col='Survived', row='Pclass', size=2.2, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=.5, bins=20)

grid.add_legend()

plt.show()
grid = sns.FacetGrid(train_data, row='Embarked', col='Survived', size=2.2, aspect=1.6)

grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)

grid.add_legend()

plt.show()
corr_train = train_data.corr()

sns.heatmap(corr_train)

plt.show()

# this shows that SibSp and Parch columns are releted , so we can combine this two column to reduce the dimension

# of our data.. this plot only works for columns with numercal data 
((train_data.groupby(['Sex','Survived']).Survived.count() * 100) / train_data.groupby('Sex').Survived.count())

# this shows that female have around 74% chance of survival while male have around 81% chance of death
(train_data.groupby(['Pclass','Survived']).Survived.count() * 100) / train_data.groupby('Pclass').Survived.count()

# this shows that people belonging to third class are likely to die while people in class one are likely to survive
(train_data.groupby(['Embarked','Survived']).Survived.count() * 100) / train_data.groupby('Embarked').Survived.count()

# this shows that people who embarked from Southampton are likely to die
train_data.groupby(by=['Survived']).mean()["Age"]

# this show that average age of people who survived was around 28 years old
# before filling the missing values, let's drop Cabin column from both data.

train_data.drop('Cabin', axis = 1, inplace = True)

test_data.drop('Cabin', axis = 1, inplace = True)
combined_data = [train_data, test_data]

for data in combined_data:

    print(data.isnull().sum())

    print('*' * 20)

      
# filling the nan values fo Age and fare column with the mean while Embarked column with most_frequent value

for data in combined_data:

    data.Age.fillna(data.Age.mean(), inplace = True)

    data.Fare.fillna(data.Fare.mean(), inplace = True)

    

# from visualization we know that Southamptom is most frequent Embarked place so, filling the missing value 

# with 'S'

train_data.Embarked.fillna('S', inplace = True)



# we simply can use SimpleImputer class form the sklearn to deal with the missing value

# from sklearn.impute import SimpleImputer

# impute = SimpleImputer(missing_values = np.nan, strategy = 'mean')

# train_data[['Age']] = impute.fit_transform(train_data[['Age']])

    


def change_gender(x):

    if x == 'male':

        return 0

    elif x == 'female':

        return 1

train_data.Sex = train_data.Sex.apply(change_gender)

test_data.Sex = test_data.Sex.apply(change_gender)

# we simply can use mapfunction to change the gender

# train_data.Sex = train_data.Sex.map({'female':1, 'male':0})



change = {'S':1,'C':2,'Q':0}

train_data.Embarked = train_data.Embarked.map(change)

test_data.Embarked = test_data.Embarked.map(change)


train_data['Alone'] = train_data.SibSp + train_data.Parch

test_data['Alone'] = test_data.SibSp + test_data.Parch



train_data.Alone = train_data.Alone.apply(lambda x: 1 if x == 0 else 0)

test_data.Alone = test_data.Alone.apply(lambda x: 1 if x == 0 else 0)
# now lets drop SibSp and Parch column for both training and testing data

train_data.drop(['SibSp','Parch'], axis = 1, inplace = True)

test_data.drop(['SibSp','Parch'], axis = 1, inplace = True )
train_data.Name.str.extract(' ([A-Za-z]+)\.', expand=False).unique().size

# there are total 17 unique title
# lets create the Title feature which contain the title of the passenger and drop Name column

for data in combined_data:

    data['Title'] = data.Name.str.extract('([A-Za-z]+)\.', expand = False)

    data.drop('Name', axis = 1, inplace = True)

       
train_data.Title.value_counts()
test_data.Title.unique()
#lets replace least occuring title in the data with rare

least_occuring = [ 'Don', 'Rev', 'Dr', 'Mme', 'Ms',

       'Major', 'Lady', 'Sir', 'Mlle', 'Col', 'Capt', 'Countess','Dona',

       'Jonkheer']

for data in combined_data:

    data.Title = data.Title.replace(least_occuring, 'Rare')
# lets perform title mapping in order to change to ordinal

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

for data in combined_data:

    data['Title'] = data['Title'].map(title_mapping)





columns_to_drop = ['PassengerId','Ticket']

train_data.drop(columns_to_drop, axis = 1, inplace = True)

test_data.drop(columns_to_drop[1], axis = 1, inplace = True)
for dataset in combined_data:    

    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4

for data in combined_data:

    data.loc[data['Fare'] < 30, 'Fare'] = 1

    data.loc[(data['Fare'] >= 30) & (data['Fare'] < 50),'Fare'] = 2

    data.loc[(data['Fare'] >= 50) & (data['Fare'] < 100),'Fare'] = 3

    data.loc[(data['Fare'] >= 100),'Fare'] = 4
X_train = train_data.drop("Survived", axis=1)

Y_train = train_data["Survived"]

X_test = test_data.drop("PassengerId", axis = 1)

print("shape of X_train",X_train.shape)

print("Shape of Y_train",Y_train.shape)

print("Shape of x_test",X_test.shape)

train_data.head()
test_data.head()
from sklearn.metrics import classification_report, confusion_matrix

from sklearn import metrics 

from sklearn.model_selection import cross_val_score

import tensorflow as tf

from tensorflow import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout

model = Sequential([

    Dense(units = 128, activation = 'relu', input_dim = 7,),

    Dropout(0.1),

    Dense(units = 64, activation = 'relu', kernel_initializer = 'glorot_uniform'),

    Dense(units = 64, activation = 'relu'),

    Dropout(0.3),

    Dense(units = 16, activation = 'relu', kernel_initializer = 'he_normal'),

    Dropout(0.3),

    Dense(units = 8, activation = 'relu'),

    Dropout(0.1),

    Dense(units = 1, activation = 'sigmoid')

    

    

])
model.summary()
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['acc'])

history = model.fit(X_train, Y_train , epochs = 100,batch_size = 32,verbose = 1)
plt.plot(range(1,101), history.history['acc'], color = 'lime', label = 'Training accuracy')

plt.plot(range(1,101), history.history['loss'], color = 'lime', label = 'Training loss')

plt.show()
predict = model.predict(X_test)

#since we have use sigmoid activation function in output layer

predict = (predict > 0.5).astype(int).ravel()

submit = pd.DataFrame({"PassengerId":test_data.PassengerId, 'Survived':predict})

submit.to_csv("final_submission.csv",index = False)
Y_pred_rand = (model.predict(X_train) > 0.5).astype(int)

print('Precision : ', np.round(metrics.precision_score(Y_train, Y_pred_rand)*100,2))

print('Accuracy : ', np.round(metrics.accuracy_score(Y_train, Y_pred_rand)*100,2))

print('Recall : ', np.round(metrics.recall_score(Y_train, Y_pred_rand)*100,2))

print('F1 score : ', np.round(metrics.f1_score(Y_train, Y_pred_rand)*100,2))

print('AUC : ', np.round(metrics.roc_auc_score(Y_train, Y_pred_rand)*100,2))
# plotting the confusion matrix in heatmap

matrix = confusion_matrix(Y_train, Y_pred_rand)

sns.heatmap(matrix, annot = True)

plt.show()