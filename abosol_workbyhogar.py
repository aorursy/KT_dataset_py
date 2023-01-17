import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import train_test_split

import seaborn as sns
sns.set()

%matplotlib inline
train_set = pd.read_csv('../input/train.csv', index_col='Id')
test_set = pd.read_csv('../input/test.csv', index_col='Id')

print("train shape", train_set.shape)
print("test shape", test_set.shape)

print(train_set['Target'].describe())
class AggregatePerHogar(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        pass
        
    def fit(self, X, y = None):
        print("AggregatePerHogar")
        return self
    
    def transform(self, X, y = None):
        ser = X.groupby('idhogar').apply(self.agg_func)
        return pd.DataFrame(ser.tolist(), index=ser.index)

    def lugar_of(self, row):
        lugar = 0
        for l in range(1,6):
            if row['lugar{}'.format(l)] == 1:
                lugar = l
        return lugar
    
    def agg_func(self, df):
        
        first_rec = df.iloc[0]
        
        return {
            'count': df.shape[0],
            'members_ids': df.index.tolist(),
            'sum_escolari': sum(df['escolari']),
            'num_instlevel7': sum(df['instlevel7'] == 1),
            'area1': first_rec['area1'],
            'lugar': self.lugar_of(first_rec),
            'min_age': df['age'].min(),
            'max_age': df['age'].max(),
            'mode_age': np.mean(df['age'].mode()),
            'Target': first_rec['Target'] if ('Target' in first_rec) else -1,
        }

aph = AggregatePerHogar()
agg_res = aph.fit_transform(train_set)

# I suggest to train and predict on this df

#print(type(agg_res))
#print(agg_res.shape)
#print(agg_res.head())
print(agg_res.info())
#print(agg_res.loc['003123ec2', 'members_ids'])

# The submission should then be given as follows:

#m = pd.merge(train_set, agg_res[['sum_escolari']], left_on='idhogar', right_index=True)

#print(m[['idhogar', 'sum_escolari']].head())
X = agg_res.drop(['Target', 'members_ids'], axis=1)
y = agg_res['Target']

print(pd.Series(y).value_counts(normalize=True))
np.random.seed(200)

X_train, X_validation, y_train, y_validation = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=200)
def evaluate(y, predicted_y):
    f1 = f1_score(y, predicted_y, average='macro')
    print(f1)
    print()
    print(classification_report(y, predicted_y))
    return f1
train_res = []
validation_res = []

for d in range(1, 31):
    print()
    print("-->", d)
    print()
    dt = DecisionTreeClassifier(max_depth=d, class_weight='balanced', random_state=300)
    dt.fit(X_train, y_train)
    predictions = dt.predict(X_validation)

    print("train")
    print()
    f1_train = evaluate(y_train, dt.predict(X_train))
    print("test")
    print()
    f1_validation = evaluate(y_validation, predictions)

    train_res.append(f1_train)
    validation_res.append(f1_validation)
print(np.argmax(train_res) + 1, np.amax(train_res))
print(np.argmax(validation_res) + 1, np.amax(validation_res))

plt.plot(range(1, 31), train_res, label="train")
plt.plot(range(1, 31), validation_res, label="validation")
plt.legend()
dt = DecisionTreeClassifier(max_depth=13, class_weight='balanced', random_state=300)

dt.fit(X, y) # all train data

# TODO: work with pipe

test = aph.transform(test_set)

predictions = dt.predict(test.drop(['Target', 'members_ids'], axis=1))

predictions = pd.Series(predictions, index=test.index)

print(predictions.value_counts(normalize=True))
m = pd.merge(test_set, pd.DataFrame({'Target': predictions}), left_on='idhogar', right_index=True)

print(m.head())

m[['Target']].to_csv('submission.csv')