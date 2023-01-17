import numpy as np

import pandas as pd 



import matplotlib.pyplot as plt

import matplotlib.pylab as pylab



%matplotlib inline

pylab.rcParams[ 'figure.figsize' ] = 10 , 8



from sklearn import datasets

from sklearn.neighbors import KNeighborsClassifier 

from sklearn.model_selection import train_test_split



import seaborn as sns



plt.style.use("ggplot")
titanic_train_df = pd.read_csv('../input/train.csv')

titanic_test_df = pd.read_csv('../input/test.csv')



titanic_train_df.head()
titanic_train_df = titanic_train_df[pd.notnull(titanic_train_df['Embarked'])]

embarked_vals = titanic_train_df.Embarked.unique()



s_embarked_vals = []

#for val in embarked_vals:

#    s_embarked_vals.append(val + '-S')

#    s_embarked_vals.append(val + '-D')

    

# y_values = list of values of how many occurences of survived and not survived for each cat.

for val in embarked_vals:

    val_rows = titanic_train_df.loc[titanic_train_df['Embarked'] == val]

    val_rows_survived = val_rows.loc[val_rows['Survived'] == 1]

    s_embarked_vals.append(val_rows_survived.shape[0])

    

objects = ('Southhampton', 'Cherbourg', 'Queenstown')

y_pos = np.arange(len(objects))

performance = s_embarked_vals

    

plt.bar(y_pos, performance, align='center', alpha=0.5)

plt.xticks(y_pos, objects)

plt.ylabel('Usage')

plt.title('Embarked City vs. Survived')



plt.show()



def getEmbarkedValue(row):

    if row == 'S':

        return 3

    elif row == 'C':

        return 2

    else:

        return 1



titanic_train_df['Embarked'] = titanic_train_df['Embarked'].apply(lambda x: getEmbarkedValue(x))

titanic_test_df['Embarked'] = titanic_test_df['Embarked'].apply(lambda x: getEmbarkedValue(x))
def titanic_corr(data):

    correlation = data.corr()

    sns.heatmap(correlation, annot=True, cbar=True, cmap="RdYlGn")

    

titanic_corr(titanic_train_df)
def findSex(row):

    if str(row) == 'male':

        return 0

    elif str(row) == 'female':

        return 1



# Create arrays for the features and the response variable

y = titanic_train_df['Survived'].values   # => Target

X = titanic_train_df.drop('Survived', axis=1)# => Feature

X = X.drop('Embarked', axis=1)

X = X.drop('PassengerId', axis=1)

X = X.drop('Cabin', axis=1)

X = X.drop('Ticket', axis=1)

X = X.drop('Name', axis=1)

X = X.drop('SibSp', axis=1)

X = X.drop('Age', axis=1)



X['Sex'] = X['Sex'].apply(lambda x: findSex(x))



X = X.fillna(value=30)



test_x = titanic_test_df.drop('Embarked', axis=1)# => Feature

test_x = test_x.drop('PassengerId', axis=1)

test_x = test_x.drop('Cabin', axis=1)

test_x = test_x.drop('Ticket', axis=1)

test_x = test_x.drop('Name', axis=1)

test_x = test_x.drop('SibSp', axis=1)

test_x = test_x.drop('Age', axis=1)



test_x['Sex'] = test_x['Sex'].apply(lambda x: findSex(x))



test_x = test_x.fillna(value=30)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42, stratify=y)
# Setup arrays to store train and test accuracies

neighbors = np.arange(1, 9)

train_accuracy = np.empty(len(neighbors))

test_accuracy = np.empty(len(neighbors))



# Loop over different values of k

for i, k in enumerate(neighbors):

    # Setup a k-NN Classifier with k neighbors: knn

    knn = KNeighborsClassifier(n_neighbors=k)



    # Fit the classifier to the training data

    knn.fit(X_train, y_train)

    

    #Compute accuracy on the training set

    train_accuracy[i] = knn.score(X_train, y_train)



    #Compute accuracy on the testing set

    test_accuracy[i] = knn.score(X_test, y_test)



# Generate plot

plt.title('k-NN: Varying Number of Neighbors')

plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')

plt.plot(neighbors, train_accuracy, label = 'Training Accuracy')

plt.legend()

plt.xlabel('Number of Neighbors')

plt.ylabel('Accuracy')

plt.show()
knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train, y_train)



print(knn.score(X_test, y_test))



prediction = knn.predict(test_x)



new_df = pd.DataFrame(columns=['PassengerId', 'Survived'])

new_df['Survived'] = prediction

new_df['PassengerId'] = titanic_test_df.PassengerId



new_df