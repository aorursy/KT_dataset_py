import pandas as pd
# get titanic training file as a DataFrame
titanic = pd.read_csv("../input/train.csv")
titanic.shape
# preview the data
titanic.head()
titanic.describe()
titanic.info()
ports = pd.get_dummies(titanic.Embarked , prefix='Embarked')
ports.head()
titanic = titanic.join(ports)
titanic.drop(['Embarked'], axis=1, inplace=True) # then drop the original column
titanic.Sex = titanic.Sex.map({'male':0, 'female':1})
y = titanic.Survived.copy() # copy “y” column values out
X = titanic.drop(['Survived'], axis=1) # then, drop y column
X.drop(['Cabin'], axis=1, inplace=True) 
X.drop(['Ticket'], axis=1, inplace=True) 
X.drop(['Name'], axis=1, inplace=True) 
X.drop(['PassengerId'], axis=1, inplace=True)
X.info()
X.isnull().values.any()
#X[pd.isnull(X).any(axis=1)]  # check which rows have NaNs
X.Age.fillna(X.Age.mean(), inplace=True)  # replace NaN with average age
X.isnull().values.any()
from sklearn.model_selection import train_test_split
  # 80 % go into the training test, 20% in the validation test
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=7)
def simple_heuristic(titanicDF):
    '''
    predict whether or not the passngers survived or perished.
    Here's the algorithm, predict the passenger survived:
    1) If the passenger is female or
    2) if his socioeconomic status is high AND if the passenger is under 18
    '''

    predictions = [] # a list
    
    for passenger_index, passenger in titanicDF.iterrows():
          
        if passenger['Sex'] == 1:
                    # female
            predictions.append(1)  # survived
        elif passenger['Age'] < 18 and passenger['Pclass'] == 1:
                    # male but minor and rich
            predictions.append(1)  # survived
        else:
            predictions.append(0) # everyone else perished

    return predictions
simplePredictions = simple_heuristic(X_valid)
correct = sum(simplePredictions == y_valid)
print ("Baseline: ", correct/len(y_valid))
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
model.score(X_train, y_train)
model.score(X_valid, y_valid)
model.intercept_ # the fitted intercept
model.coef_  # the fitted coefficients
titanic.corr()
