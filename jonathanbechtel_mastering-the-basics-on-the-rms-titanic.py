import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

plt.style.use('ggplot')

from sklearn.preprocessing import LabelEncoder

from sklearn.cross_validation import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import precision_recall_fscore_support

from sklearn import svm



#Read files into the program

test = pd.read_csv("../input/test.csv", index_col='PassengerId')

train = pd.read_csv("../input/train.csv", index_col='PassengerId')
train.head(5)
y = train['Survived']

del train['Survived']



train = pd.concat([train, test])



#Drop unnecessary columns

train = train.drop(train.columns[[6,9]], axis=1)  
train.head(5)
train['Sex'] = LabelEncoder().fit_transform(train.Sex)

train['Pclass'] = LabelEncoder().fit_transform(train.Pclass)
train['Cabin'] = train.Cabin.apply(lambda x: x[0] if pd.notnull(x) else 'X')

train['Cabin'] = LabelEncoder().fit_transform(train.Cabin)
train[['Pclass', 'Sex', 'Cabin']][0:3]
train.info()
train.groupby(['Pclass', 'Sex'])['Age'].median()
train['Age'] = train.groupby(['Pclass', 'Sex'])['Age'].transform(lambda x: x.replace(np.nan, x.median()))

train.iloc[1043, 6] = 7.90
train['Family_Size'] = train.SibSp + train.Parch
train['Name'].iloc[0]
#Used to create new pd Series from Name data that extracts the greeting used for their name to be used as a separate variable

def greeting_search(words):

    for word in words.split():

        if word[0].isupper() and word.endswith('.'):

            return word
train['Greeting'] = train.Name.apply(greeting_search)
train['Greeting'].value_counts()
train['Greeting'] = train.groupby('Greeting')['Greeting'].transform(lambda x: 'Rare' if x.count() < 9 else x)

del train['Name']
train['Greeting'] = LabelEncoder().fit_transform(train.Greeting)
#Categorical coding for data with more than two labels

Pclass = pd.get_dummies(train['Pclass'], prefix='Passenger Class', drop_first=True)
Pclass.head(5)
Greetings = pd.get_dummies(train['Greeting'], prefix='Greeting', drop_first=True)

Cabins = pd.get_dummies(train['Cabin'], prefix='Cabin', drop_first=True)
#Scale Continuous Data

train['SibSp_scaled'] = (train.SibSp - train.SibSp.mean())/train.SibSp.std()

train['Parch_scaled'] = (train.Parch - train.Parch.mean())/train.Parch.std()

train['Family_scaled'] = (train.Family_Size - train.Family_Size.mean())/train.Family_Size.std()

train['Age_scaled'] = (train.Age - train.Age.mean())/train.Age.std()

train['Fare_scaled'] = (train.Fare - train.Fare.mean())/train.Fare.std()
#Drop unmodified data since it's no longer needed

train = train.drop(train.columns[[0,2,3,4,5,6,7,8]], axis=1)



#Concat modified data to be used for analysis, set to X and y values

data = pd.concat([train, Greetings, Pclass, Cabins], axis=1)



#Split the data back into its original training and test sets

test = data.iloc[891:]

X = data[:891]
#Create cross - validation set 

X_train, X_val, y_train, y_val = train_test_split(X, y, train_size = 0.6)
clf = LogisticRegression()
def find_C(X, y):

    Cs = np.logspace(-4, 4, 10)

    score = []  

    for C in Cs:

        clf.C = C

        clf.fit(X_train, y_train)

        score.append(clf.score(X, y))

  

    plt.figure()

    plt.semilogx(Cs, score, marker='x')

    plt.xlabel('Value of C')

    plt.ylabel('Accuracy on Cross Validation Set')

    plt.title('What\'s the Best Value of C?')

    plt.show()

    clf.C = Cs[score.index(max(score))]

    print("Ideal value of C is %g" % (Cs[score.index(max(score))]))

    print('Accuracy: %g' % (max(score)))
find_C(X_val, y_val)
answer = pd.DataFrame(clf.predict(test), index=test.index, columns=['Survived'])

answer.to_csv('answer.csv')
coef = pd.DataFrame({'Variable': data.columns, 'Coefficient': clf.coef_[0]})

coef
results = y_val.tolist()

predict = clf.predict(X_val)



def precision_recall(predictions, results):

    

    tp, fp, fn, tn, i = 0.0, 0.0, 0.0, 0.0, 0

    

    while i < len(results):

        

            if predictions[i] == 1 and results[i] == 1:

                tp = tp + 1

            elif predictions[i] == 1 and results[i] == 0:

                fp = fp + 1

            elif predictions[i] == 0 and results[i] == 0:

                tn = tn + 1

            else: 

                fn = fn + 1

            i = i + 1

    

    precision = tp / (tp + fp)

    recall = tn / (tn + fn)

    f1 = 2*precision*recall / (precision + recall)

    print("Precision: %g, Recall: %g, f1: %g" % (precision, recall, f1))
precision_recall(predict, results)