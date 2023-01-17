import numpy as np
import pandas as pd
df = pd.read_csv('../input/train.csv')
df.info()
df.head()
# drop irrelevant features
df = df.drop(['Name', 'Ticket', 'Cabin'], axis='columns')
from collections import Counter
from sklearn import preprocessing

# compute most frequent/mean values
'''
ctr = Counter(df['Cabin'])
print("Cabin feature most common 2 data points:", ctr.most_common(2))
'''

ctr = Counter(df['Embarked'])
print("Embarked feature most common 2 data points:", ctr.most_common(2))

print("Age feature mean value:", np.mean(df['Age'].dropna()))
# impute the feature columns
#df['Cabin'].fillna('G6', inplace=True)

df['Embarked'].fillna('S', inplace=True)

df['Age'].fillna(30, inplace=True) # 29.69... does not specify a valid age, round it
import copy

# encode the categorical features into numerical values
encoder = preprocessing.LabelEncoder()

embarkedEncoder = copy.copy(encoder.fit(df['Embarked']))
df['Embarked'] = embarkedEncoder.transform(df['Embarked'])
#df['Cabin'] = encoder.fit_transform(df['Cabin'])

sexEncoder = copy.copy(encoder.fit(df['Sex']))
df['Sex'] = sexEncoder.transform(df['Sex'])
df.describe()
df.info()
from sklearn.model_selection import train_test_split

X = df.drop('Survived', axis='columns')
Y = df['Survived']
trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.33)

print(len(trainX), 'training records and', len(testX), 'testing records')

def trainAndPredict(model):
    model.fit(trainX, trainY)
    predictions = model.predict(testX)
    mismatch = 0
    for estimate, real in zip(predictions, testY):
        if estimate != real:
            mismatch += 1
    return mismatch
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

gnb = GaussianNB()
mnb = MultinomialNB()
bnb = BernoulliNB()

modelNames = ["Gaussian Naive Bayes", "Multinomial Naive Bayes", "Bernoulli Naive Bayes"]
predictionErrors = [trainAndPredict(gnb), trainAndPredict(mnb), trainAndPredict(bnb)]

for i in range(3):
    print(f"Out of {len(testX)} records, the {modelNames[i]} classifier has {predictionErrors[i]} incorrect predictions")
from sklearn import svm

svc = svm.SVC()
print(f"Out of {len(testX)} records, the SVM classifier has {trainAndPredict(svc)} incorrect predictions")
testDF = pd.read_csv('../input/test.csv')
testDF.head()
testDF.info()
testDF.drop(['Name', 'Ticket'], axis='columns', inplace=True)
ctr = Counter(testDF['Cabin'])
print(f'Cabin feature most common values:', ctr.most_common(4))

meanAge = np.mean(testDF['Age'])
print(f'Mean age for the age feature:', meanAge)
# drop the Cabin feature and fill perform mean substitution for missing records in the Age feature
testDF.drop('Cabin', axis='columns', inplace=True)
testDF['Age'].fillna(30, inplace=True)
# encode the Embarked and Sex features
testDF['Embarked'] = embarkedEncoder.transform(testDF['Embarked'])
testDF['Sex'] = sexEncoder.transform(testDF['Sex'])
testDF.info()
testDF['Fare'].fillna(np.mean(testDF['Fare']), inplace=True)
testDF.info()
predictions = gnb.predict(testDF)
def writeCSV(predictions):
    outputDF = pd.DataFrame(np.column_stack([testDF['PassengerId'], predictions]), columns=['PassengerId', 'Survived'])
    outputDF.to_csv('./predictions.csv', index=False)
writeCSV(predictions)
predDF = pd.read_csv('./predictions.csv')
predDF.head()
