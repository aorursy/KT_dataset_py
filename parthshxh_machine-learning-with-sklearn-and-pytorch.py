# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import torch

from torch import nn

import torch.nn.functional as F

from torch import optim



import sklearn

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.pylab as pylab

import seaborn as sns



%matplotlib inline

mpl.style.use('ggplot')

sns.set_style('white')

pylab.rcParams['figure.figsize'] =12,8
path =  "../input/"



df_train = pd.read_csv(f'{path}train.csv', index_col = 'PassengerId')

df_test = pd.read_csv(f'{path}test.csv', index_col = 'PassengerId')



target = df_train['Survived']

target.columns = ['Survived']

df_train = df_train.drop(labels = 'Survived', axis = 1)
df_train['Training_set']= True

df_test['Training_set'] = False



df_full = pd.concat([df_train,df_test])
df_full.info()
df_full.isnull().sum()[df_full.isnull().sum()>0]
df_full = df_full.drop(labels = ['Ticket','Name', 'Cabin'], axis = 1)

df_full
df_full.Age = df_full.Age.fillna(df_full.Age.mean())

df_full.Fare = df_full.Fare.fillna(df_full.Fare.mean())

df_full.Embarked = df_full.fillna(df_full.Embarked.mode()[0])



df_full = df_full.interpolate()

df_full = pd.get_dummies(df_full)

df_full
df_train = df_full[df_full['Training_set']==True]

df_test = df_full[df_full['Training_set']==False]
df_train.drop(labels = 'Training_set', inplace = True, axis = 1)

df_test.drop(labels ='Training_set', inplace = True, axis = 1)
torch.manual_seed(2) #setting a seed so that the results are reproducible

msk = np.random.randn(len(df_train)) < 0.8



training_features = torch.tensor(df_train[msk].values.astype('float32'))

dev_features = torch.tensor(df_train[~msk].values.astype('float32'))



training_labels = torch.tensor(target[msk].values)

dev_labels = torch.tensor(target[~msk].values)



test_features = torch.tensor(df_test.values.astype('float32'))
model = nn.Sequential(nn.Linear(10,5),nn.ReLU(),nn.Dropout(p=0.2),nn.Linear(5,5),nn.ReLU(), nn.Dropout(p=0.1),nn.Linear(5,2),nn.LogSoftmax(dim =1))



criterion = nn.NLLLoss()

optimizer = optim.Adam(model.parameters(), lr = 0.003)
def evaluate(model,test_features,test_labels=None,print_acc_and_cost = False, testing = False):

    with torch.no_grad():

        test_loss =0

        log_preds = model(test_features)

        preds=torch.exp(log_preds)

        preds, survived = torch.max(preds, 1)

        if(testing):

            loss = criterion(log_preds,test_labels)

            equals = survived==test_labels.view(*survived.shape)

            accuracy = torch.mean(equals.type(torch.FloatTensor))

            test_loss += loss.item()

            if(print_acc_and_cost):

                print(f'Testing accuracy : {accuracy*100:.3f}%')

                print(f'Testing Loss : {test_loss:.3f}')

            return test_loss

        else:

            return survived

        #print(survived)
epochs = 1000

steps = 0

training_losses,testing_losses = [],[]

for e in range(epochs):

    running_loss = 0

    optimizer.zero_grad()

    

    log_outputs = model(training_features)

    loss = criterion(log_outputs,training_labels)

    

    loss.backward()

    outputs = torch.exp(log_outputs)

    

    optimizer.step()

    steps += 1

    running_loss += loss.item()

    training_losses.append(running_loss)

    if(steps%100==0):

        with torch.no_grad():

            model.eval()

            print(f'Epoch : {e+1}/{epochs}...')

            testing_losses.append(evaluate(model,dev_features, test_labels = dev_labels, print_acc_and_cost=True, testing = True))

        print(f'Training loss {running_loss:.3f}\n')

    else:

        with torch.no_grad():

            model.eval()

            testing_losses.append(evaluate(model,dev_features, test_labels = dev_labels, print_acc_and_cost=False, testing = True))

    model.train()



plt.plot(training_losses,label='Training loss')

plt.plot(testing_losses, label='Testing loss')

plt.xlabel("Epochs")

plt.ylabel("Loss")

plt.title("3 Layer Neural Network\n")

plt.legend(frameon = False)
survived = evaluate(model,test_features)

my_submission = pd.DataFrame({'PassengerId':df_test.index,'Survived':survived})

my_submission.to_csv('./submission.csv', index = False)

!ls
np.random.seed(0)

from sklearn import model_selection

X_train, X_val, y_train, y_val = model_selection.train_test_split(df_train,target, test_size = 0.2, train_size = 0.8, random_state = 0 )

#print(X_train.shape,y_train.shape,X_val.shape, y_val.shape)
from sklearn import tree

from sklearn import ensemble

from sklearn import linear_model

from sklearn import gaussian_process

from sklearn import naive_bayes

from sklearn import neighbors

from sklearn import svm

from sklearn import discriminant_analysis

from xgboost.sklearn import XGBClassifier
MLA = [

    ensemble.AdaBoostClassifier(),

    ensemble.BaggingClassifier(),

    ensemble.GradientBoostingClassifier(),

    ensemble.ExtraTreesClassifier(),

    ensemble.RandomForestClassifier(n_estimators = 100, random_state = 0),

    

    gaussian_process.GaussianProcessClassifier(),

    

    naive_bayes.BernoulliNB(),

    naive_bayes.GaussianNB(),

    

    neighbors.KNeighborsClassifier(),

    

    svm.SVC(probability=True),

    svm.NuSVC(probability = True),

    svm.LinearSVC(),

    

    tree.DecisionTreeClassifier(),

    tree.ExtraTreeClassifier(),

    

    discriminant_analysis.LinearDiscriminantAnalysis(),

    discriminant_analysis.QuadraticDiscriminantAnalysis(),

    

    XGBClassifier(), 

    

    linear_model.LogisticRegressionCV(),

    linear_model.PassiveAggressiveClassifier(),

    linear_model.RidgeClassifierCV(),

    linear_model.SGDClassifier(),

    linear_model.Perceptron()

]
MLA_columns = ['MLA_names', 'MLA_parameters', 'MLA_Train_Accuracy_Mean'

               ,'MLA_Test_Accuracy_Mean', 'MLA_Test_Accuracy_3*STD', 

               'MLA_Time']



MLA_compare = pd.DataFrame(columns = MLA_columns)
import warnings

warnings.filterwarnings('ignore')
MLA_Predict = y_val

row_index = 0

for alg in MLA:

    MLA_name = alg.__class__.__name__

    MLA_compare.loc[row_index, 'MLA_names'] = MLA_name

    MLA_compare.loc[row_index, 'MLA_parameters'] = str(alg.get_params())

    cv_results = model_selection.cross_validate(alg, X_train, y_train, cv=3, return_train_score = True)

    MLA_compare.loc[row_index, 'MLA_Time'] = cv_results['fit_time'].mean()

    MLA_compare.loc[row_index, 'MLA_Train_Accuracy_Mean'] = cv_results["train_score"].mean()

    MLA_compare.loc[row_index, 'MLA_Test_Accuracy_Mean'] = cv_results['test_score'].mean()

    MLA_compare.loc[row_index, 'MLA_Test_Accuracy_3*STD'] = cv_results['test_score'].std()*3

    

    alg.fit(X_train, y_train)

    MLA_Predict[MLA_name] = alg.predict(X_val)

    row_index += 1

    print(".", end="")

MLA_compare.sort_values(by = 'MLA_Test_Accuracy_Mean', ascending = False, inplace = True)

MLA_compare
sns.barplot(x = 'MLA_Test_Accuracy_Mean', y = 'MLA_names', data = MLA_compare, color = 'm')



plt.title('Machine Learning Algorithms Accuracy Score \n')

plt.ylabel('Algortithm')

plt.xlabel('Accuracy Test Score (%)')
np.random.seed(0)

from sklearn import model_selection

X_train, X_val, y_train, y_val = model_selection.train_test_split(df_train,target, test_size = 0.2, train_size = 0.8, random_state = 0 )

#print(X_train.shape,y_train.shape,X_val.shape, y_val.shape)
from sklearn import ensemble

alg = ensemble.GradientBoostingClassifier(n_estimators= 100, random_state = 0)
train_score = np.mean(model_selection.cross_val_score(alg, X_train, y_train, cv=5))



alg.fit(X_train, y_train)

survived = alg.predict(X_val)



print(train_score)

#print(test_score)

#print(survived.shape,y_val.shape,X_val.shape)

equals = survived == y_val.values.reshape(*survived.shape)

print(f"Accuracy = {np.mean(equals.astype(int))*100:.3f}%")
survived_test = alg.predict(df_test)

submission = pd.DataFrame({'PassengerId':df_test.index,'Survived':survived_test})

#submission.to_csv('./submission.csv',index=False)

!ls