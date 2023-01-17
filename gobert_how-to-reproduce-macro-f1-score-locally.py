import pandas as pd
# This will be an helper method to re-use the code between train and test set

class Household():
    def __init__(self, individuals): 
        self.individuals = individuals
        self.grouped_individuals = individuals.groupby('idhogar')
        self.household = pd.DataFrame()

    def add_SQBedjefe(self):
        self.household['SQBedjefe'] = self.grouped_individuals.SQBedjefe.mean()
        
    def add_SQBdependency(self):
        self.household['SQBdependency'] = self.grouped_individuals.SQBdependency.mean()
        
    def add_overcrowding(self):
        self.household['overcrowding'] = self.grouped_individuals.overcrowding.mean()
        
    def add_qmobilephone(self):
        self.household['qmobilephone'] = self.grouped_individuals.qmobilephone.mean()
        
    def add_rooms(self):
        self.household['rooms'] = self.grouped_individuals.rooms.mean()
        
    def add_SQBhogar_nin(self):
        self.household['SQBhogar_nin'] = self.grouped_individuals.SQBhogar_nin.mean()
        
    def add_Target(self):
        self.household['Target'] = self.grouped_individuals.Target.mean().round().astype(int)
# Build dataset on Household level.
individuals = pd.read_csv('../input/train.csv')
train = Household(individuals)

train.add_SQBedjefe()
train.add_SQBdependency()
train.add_overcrowding()
train.add_qmobilephone()
train.add_rooms()
train.add_SQBhogar_nin()
train.add_Target()
X = train.household.loc[:, train.household.columns != 'Target']
y = train.household.Target

from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=112, test_size=0.2)
from sklearn.ensemble import RandomForestClassifier

clf= RandomForestClassifier()
clf.fit(X_train, y_train)
y_predict = clf.predict(X_valid)
from sklearn.metrics import f1_score

f1_score(y_valid, y_predict, average='macro')
# Build test dataset on Household level
df_test = pd.read_csv('../input/test.csv')
test = Household(df_test)

test.add_SQBedjefe()
test.add_SQBdependency()
test.add_overcrowding()
test.add_qmobilephone()
test.add_rooms()
test.add_SQBhogar_nin()
X_test = test.household
X_test['Target'] = clf.predict(X_test)
df_test['Target'] = None

def target(idhogar):
    return X_test.Target[idhogar]

df_test['Target'] = df_test.idhogar.map(target)
df_test[['Id', 'Target']].to_csv('sample_submission.csv', index=False)