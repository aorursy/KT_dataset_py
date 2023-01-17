import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pandas.core import datetools

import statsmodels.api as sm

from patsy import dmatrices

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
input_ds = pd.read_csv('../input/train.csv', index_col = 0)

input_ds.head(3)
# clean datasets

def clean(ds):

    # 'Name','Ticket','Cabin' - some of the records are NaN, another seems to be irrelevant

    # 'Fare' - we already have class and port of embarkment. Fare might not be required

    ds = ds.drop(['Name','Ticket','Cabin','Fare'], axis=1) 

    # Remove NaN values

    return ds.dropna() 



input_ds = clean(input_ds)

input_ds.head(3)
def convertDiscreteValues(ds):

    #convert values of Sex column to 1 - male, 0 - female

    ds['Sex'] = ds['Sex'].apply(lambda x: 1 if x=='male' else 0)



    #convert values of Embarked column to 1 - C = Cherbourg, 2 - Q = Queenstown, 3 - S = Southampton

    ds['Embarked'] = ds['Embarked'].apply(lambda x: 1 if x == 'C' else 2 if x == 'Q' else 3 if x == 'S' else 0)



def compareDiscreteValues(ds, col, figsize=(6,4), kind='barh'):

    plt.figure(figsize=figsize)

    fig, ax = plt.subplots()

    

    summon = ds.groupby(by=col).size() 

    summon.plot(kind=kind, label='Amount', color="blue", alpha=.65)

    group = ds[(ds.Survived == 1)].groupby(by=col).size()

    group.plot(kind=kind, label='Survived', color="orange", alpha=.65)

    

    if kind=='barh':

        ax.set_ylim(-1, len(summon))

    plt.title('Survival Breakdown according to ' + col)

    plt.legend(loc='best')
compareDiscreteValues(input_ds, 'Pclass')

#1st clase has more changes to survive than 2st 

# which has a much highest chanes to survive in comparison to 3st

#less raw value means more changes to survive



compareDiscreteValues(input_ds, 'Sex')

#female had more changes



compareDiscreteValues(input_ds, 'SibSp')

compareDiscreteValues(input_ds, 'Parch')

#seems indifferent



#port of embarkation

#C = Cherbourg, Q = Queenstown, S = Southampton

compareDiscreteValues(input_ds, 'Embarked')

#evidently part of passangers embarked in Cherbourg had more chances to survive



compareDiscreteValues(input_ds, 'Age', (15,3), 'area')

# children younger 5 and people oldest than 65 survived 
def accuracy(ds, predicted):

    amount = np.where(ds['Survived'] != predicted['Survived'])[0].size

    return (ds.size - amount)/ds.size



def getLimit(ds, pred):

    survived = ds.Survived.sum()

    limit = pred.sort_values(ascending=False).iloc[survived]

    return limit



def summarizePrediction(ds, pred, limit):

    output = pd.DataFrame(index=ds.index)

    output['Survived'] = [1 if v >= limit else 0 for v in pred]

    return output



def compileAndPlotLogitModel(formula, ds):

    y,x = dmatrices(formula, data=ds, return_type='dataframe')

    model = sm.Logit(y,x)

    return model.fit()



def predict(model, formula, ds):

    _,x = dmatrices(formula, data=ds, return_type='dataframe')

    return model.predict(x)

#patsy formula

formula = 'Survived ~ C(Pclass) + C(Sex) + Age + SibSp + Parch + C(Embarked)'



#split randomly and proportioanlly train and validation dataset

train_ds, val_ds = train_test_split(input_ds, test_size=0.05)
model = compileAndPlotLogitModel(formula, train_ds)



pred = predict(model, formula, train_ds)

limit = getLimit(train_ds, pred)

train_pred = summarizePrediction(train_ds,pred, limit)



print("Accuracy of training set prediction: " + str(accuracy(train_ds, train_pred)))
pred = predict(model, formula, val_ds)

val_pred = summarizePrediction(val_ds, pred, limit)



print("Accuracy of validation set prediction: " + str(accuracy(val_ds, val_pred)))
# train dataset

test_ds = pd.read_csv('../input/test.csv', index_col = 0)



# evaluate and submit to Kaggle

ds = clean(test_ds)

ds['Survived'] = 1.23

pred = predict(model, formula, ds)

pred_test = summarizePrediction(ds, pred, limit)
#save output

pred_test.to_csv('output.csv', index=False)