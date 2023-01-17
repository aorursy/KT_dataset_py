%matplotlib inline

import matplotlib.pyplot as plt

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# Load and view the dataset

train = pd.read_csv('../input/train.csv')

train.head()
train.info()
embarked_mode = train['Embarked'].mode()

train['Embarked'].fillna(embarked_mode[0], inplace=True)

cabin_mode = train['Cabin'].mode()

train['Cabin'].fillna(cabin_mode[0], inplace=True)

age_median = train['Age'].median()

train['Age'].fillna(age_median, inplace=True)

train.info()
test = pd.read_csv('../input/test.csv')

test.head()
test['Age'].fillna(age_median, inplace=True)

test.info()
print('Number of training examples: ', len(train))

print('Number of testing  examples: ', len(test))

print('Number of attributes: ', len(test.columns))

print('Available attributes: ', test.columns)
train.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

test.drop(['Name', 'Ticket', 'Cabin'],  axis=1, inplace=True)

train.info()

test.info()
# Plot attributes

import seaborn as sns



train['Age'].hist(bins=70)



# peaks for survived/not survived passengers by their age

facet = sns.FacetGrid(train, hue="Survived",aspect=4)

facet.map(sns.kdeplot,'Age',shade= True)

facet.set(xlim=(0, train['Age'].max()))

facet.add_legend()



# average survived passengers by age

fig, axis1 = plt.subplots(1,1,figsize=(38,4))

average_age = train[['Age', 'Survived']].groupby(['Age'],as_index=False).mean()

sns.barplot(x='Age', y='Survived', data=average_age)



min_age = min(train['Age'])

max_age = max(train['Age'])

mean_age = train['Age'].mean()

median_age = train['Age'].median()

mode_age = train['Age'].mode()

std_age = train['Age'].std()

range_age = max_age - min_age

print('Age - min: ', min_age, ' max: ', max_age, ' mean: ', mean_age, ' median: ',

      median_age, ' mode: ', mode_age.values[0], ' std: ', std_age, ' range: ', range_age)



# Show the survival proability as a function of age and sex

# sns.lmplot(x="Age", y="Survived",  data=train, y_jitter=.02, logistic=True)
# Make a custom palette with gendered colors

pal = dict(male='#6495ED', female='#F08080')

# g = sns.lmplot(x='Age', y='Survived', data=train, palette=pal, y_jitter=.02, logistic=True, hue='Sex', col='Sex')

# g.set(xlim=(0, 80), ylim=(-.05, 1.05))
sns.factorplot('Embarked', 'Survived', data=train, size=4, aspect=3)







fig, (axis1,axis2,axis3) = plt.subplots(1,3,figsize=(15,5))



sns.countplot(x='Embarked', data=train, ax=axis1)

sns.countplot(x='Survived', hue="Embarked", data=train, order=[1,0], ax=axis2)



# group by embarked, and get the mean for survived passengers for each value in Embarked

embark_perc = train[["Embarked", "Survived"]].groupby(['Embarked'],as_index=False).mean()

sns.barplot(x='Embarked', y='Survived', data=embark_perc,order=['S','C','Q'],ax=axis3)



# Either to consider Embarked column in predictions,

# and remove "S" dummy variable, 

# and leave "C" & "Q", since they seem to have a good rate for Survival.



# OR, don't create dummy variables for Embarked column, just drop it, 

# because logically, Embarked doesn't seem to be useful in prediction.



embark_dummies_titanic  = pd.get_dummies(train['Embarked'])

embark_dummies_titanic.drop(['S'], axis=1, inplace=True)



embark_dummies_test  = pd.get_dummies(test['Embarked'])

embark_dummies_test.drop(['S'], axis=1, inplace=True)



train = train.join(embark_dummies_titanic)

test    = test.join(embark_dummies_test)



train.drop(['Embarked'], axis=1,inplace=True)

test.drop(['Embarked'], axis=1,inplace=True)

train.info()

test.info()
train['Family'] =  train["Parch"] + train["SibSp"]

train['Family'].loc[train['Family'] > 0] = 1

train['Family'].loc[train['Family'] == 0] = 0



test['Family'] =  test["Parch"] + test["SibSp"]

test['Family'].loc[test['Family'] > 0] = 1

test['Family'].loc[test['Family'] == 0] = 0



# drop Parch & SibSp

train = train.drop(['SibSp','Parch'], axis=1)

test  = test.drop(['SibSp','Parch'], axis=1)



# plot

fig, (axis1,axis2) = plt.subplots(1,2,sharex=True,figsize=(10,5))



sns.countplot(x='Family', data=train, order=[1,0], ax=axis1)



# average of survived for those who had/didn't have any family member

family_perc = train[["Family", "Survived"]].groupby(['Family'],as_index=False).mean()

sns.barplot(x='Family', y='Survived', data=family_perc, order=[1,0], ax=axis2)



axis1.set_xticklabels(["With Family","Alone"], rotation=0)
def get_person(passenger):

    age,sex = passenger

    return 'child' if age < 16 else sex

    

train['Person'] = train[['Age','Sex']].apply(get_person,axis=1)

test['Person']  = test[['Age','Sex']].apply(get_person,axis=1)



# No need to use Sex column since we created Person column

train.drop(['Sex'],axis=1,inplace=True)

test.drop(['Sex'],axis=1,inplace=True)



# create dummy variables for Person column, & drop Male as it has the lowest average of survived passengers

person_dummies_titanic  = pd.get_dummies(train['Person'])

person_dummies_titanic.columns = ['Child','Female','Male']

person_dummies_titanic.drop(['Male'], axis=1, inplace=True)



person_dummies_test  = pd.get_dummies(test['Person'])

person_dummies_test.columns = ['Child','Female','Male']

person_dummies_test.drop(['Male'], axis=1, inplace=True)



train = train.join(person_dummies_titanic)

test    = test.join(person_dummies_test)



fig, (axis1,axis2) = plt.subplots(1,2,figsize=(10,5))



sns.countplot(x='Person', data=train, ax=axis1)



# average of survived for each Person(male, female, or child)

person_perc = train[["Person", "Survived"]].groupby(['Person'],as_index=False).mean()

sns.barplot(x='Person', y='Survived', data=person_perc, ax=axis2, order=['male','female','child'])



train.drop(['Person'],axis=1,inplace=True)

test.drop(['Person'],axis=1,inplace=True)
sns.factorplot('Pclass','Survived',order=[1,2,3], data=train,size=5)



# create dummy variables for Pclass column, & drop 3rd class as it has the lowest average of survived passengers

pclass_dummies_titanic  = pd.get_dummies(train['Pclass'])

pclass_dummies_titanic.columns = ['Class_1','Class_2','Class_3']

pclass_dummies_titanic.drop(['Class_3'], axis=1, inplace=True)



pclass_dummies_test  = pd.get_dummies(test['Pclass'])

pclass_dummies_test.columns = ['Class_1','Class_2','Class_3']

pclass_dummies_test.drop(['Class_3'], axis=1, inplace=True)



train.drop(['Pclass'],axis=1,inplace=True)

test.drop(['Pclass'],axis=1,inplace=True)



train = train.join(pclass_dummies_titanic)

test    = test.join(pclass_dummies_test)
test["Fare"].fillna(test["Fare"].median(), inplace=True)



# convert from float to int

train['Fare'] = train['Fare'].astype(int)

test['Fare']    = test['Fare'].astype(int)



# get fare for survived & didn't survive passengers 

fare_not_survived = train["Fare"][train["Survived"] == 0]

fare_survived     = train["Fare"][train["Survived"] == 1]



# get average and std for fare of survived/not survived passengers

avgerage_fare = pd.DataFrame([fare_not_survived.mean(), fare_survived.mean()])

std_fare      = pd.DataFrame([fare_not_survived.std(), fare_survived.std()])



# plot

train['Fare'].plot(kind='hist', figsize=(15,3),bins=100, xlim=(0,50))



avgerage_fare.index.names = std_fare.index.names = ["Survived"]

avgerage_fare.plot(yerr=std_fare,kind='bar',legend=False)
train.info()

test.info()
x_train = train.drop(['PassengerId', 'Survived'],axis=1).values.astype(np.float32)

n_examples = len(x_train)

y_train = train["Survived"].values.reshape(n_examples,-1).astype(np.int32)

x_test  = test.drop('PassengerId',axis=1).values.astype(np.float32)
from chainer.dataset import iterator

from chainer.iterators import SerialIterator

from chainer import Chain

from chainer.training import Trainer

from chainer.training import StandardUpdater

import numpy as np

import chainer.links as L

from chainer.optimizers import AdaGrad, SGD, MomentumSGD

from chainer.training.extensions import ProgressBar

from chainer.training.extensions import Evaluator

from chainer.training.extensions import PrintReport

from chainer.training.extensions import LogReport

import chainer.functions as F

from chainer import Variable

from chainer.optimizer import WeightDecay

from chainer.functions.loss import sigmoid_cross_entropy



class TitanicModel(Chain):

    def __init__(self):

        super(TitanicModel, self).__init__(lin=L.Linear(9, 1))

    

    def __call__(self, x):

        output = self.lin(x)

        return output





train_data = [(x_train[i,:], y_train[i]) for i in range(n_examples)]

train_iter = SerialIterator(train_data, batch_size = n_examples, repeat=True, shuffle=True)

valid_iter = SerialIterator(train_data, batch_size = 1, repeat=False, shuffle=False)

titanic_model = TitanicModel()

model = L.Classifier(titanic_model, lossfun=sigmoid_cross_entropy.sigmoid_cross_entropy)

model.compute_accuracy = False

# opt = AdaGrad()

opt = MomentumSGD(lr=0.001)

opt.use_cleargrads()

opt.setup(model)

opt.add_hook(WeightDecay(0.0))

updater = StandardUpdater(train_iter, opt, device=-1)

trainer = Trainer(updater, (12000, 'epoch'))

#trainer.extend(ProgressBar())

evaluator = Evaluator(valid_iter, model)

# trainer.extend(evaluator)

# trainer.extend(LogReport())

# trainer.extend(PrintReport(['epoch', 'main/accuracy', 'main/loss', 'validation/main/accuracy',

#'validation/main/loss']))

trainer.run()
# F.sigmoid(titanic_model(x_test[0,:].reshape(9,-1)))

test_pred = F.sigmoid(titanic_model(x_test)).data

test_pred = (test_pred.reshape(len(x_test),) > 0.5).astype(np.int32)

# print test_pred

submission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": test_pred

    })

submission.to_csv('titanic.csv', index=False)