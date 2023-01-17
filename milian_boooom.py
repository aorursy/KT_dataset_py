import numpy as np

import pandas as pd



from sklearn.linear_model import LogisticRegression

from sklearn.multiclass import OneVsOneClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

from sklearn.neural_network import MLPClassifier



import seaborn as sns
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
sns.set()

sns.pairplot(train, hue="type")
types = train["type"]

indexes_test = test["id"]

colors_train = train.color

colors_test = test.color

ids = test.id



train.drop(["type","id", 'color'], axis=1, inplace=True)

test.drop(["id", 'color'], axis=1, inplace=True)
train_eng = train.copy(deep=True)



train_eng['hairXsoul'] = train_eng.hair_length * train_eng.has_soul

train_eng['hairXbone'] = train_eng.hair_length * train_eng.bone_length

train_eng['boneXsoul'] = train_eng.bone_length * train_eng.has_soul



test_eng = test.copy(deep=True)



test_eng['hairXsoul'] = test_eng.hair_length * test_eng.has_soul

test_eng['hairXbone'] = test_eng.hair_length * test_eng.bone_length

test_eng['boneXsoul'] = test_eng.bone_length * test_eng.has_soul



# train_eng = train_eng[['alpha', 'r', 'g', 'b', 'hairXsoul', 'hairXbone', 'boneXsoul']]
train_eng['type'] = types

sns.set()

sns.pairplot(train_eng, hue="type")



train_eng.drop('type', axis=1, inplace=True)
dummies = pd.get_dummies(types)

y = np.argmax(dummies.as_matrix(), 1)

labels = pd.DataFrame(y, columns=['monster'])
train_eng[colors_train=='white'].shape, labels[colors_train=='white'].shape
models = []

scores = []

predictions = []



estimators = [LogisticRegression(penalty='l2',

                       tol=0.01, 

                       C=1000.0, 

                       solver='liblinear', 

                       max_iter=10000, 

                       multi_class='ovr'),

              LogisticRegression(),

              RandomForestClassifier(n_estimators=500),

              MLPClassifier(max_iter=1000, 

                            solver='lbfgs',

                            hidden_layer_sizes=[16])

             ]



for estimator in estimators:

    print(estimator)

    for color in colors_train.unique():

        X = train_eng[colors_train!=color].values

        y = labels[colors_train!=color]

        X_train, X_test, y_train, y_test = train_test_split(X, y,

                                                            test_size=0.3)

        clf = OneVsOneClassifier(estimator)

       

        clf.fit(X_train, y_train.values.ravel())

        score = accuracy_score(y_test,

                               clf.predict(X_test))

        scores.append(score)

        print(score)

        predictions.append(clf.predict(test_eng))

    print('\n')   

print("{m} +/- {sd}".format(m=np.mean(scores), sd=np.std(scores)*2))
top_n = 7

scores = np.array(scores)

scores[scores.argsort()[-top_n:][::-1]]


predictions_top = np.array(predictions)[scores.argsort()[-top_n:][::-1]]
for x in range(0, len(ids)):

    a = predictions_top[:,x]

    v, c = np.unique(a,return_counts=True)

    if max(c)>=(top_n/2):

        continue

    print(a, v, c)
voted = []

for x in range(0, len(ids)):

    a = np.array(predictions)[:,x]

    (values, counts) = np.unique(a,return_counts=True) 

    if max(c)>=(top_n/2):

        pred = values[counts==max(counts)][0]        

    else:

        pred = int(a[np.argmax(scores)])

    

    voted.append(pred)

    
Y = pd.DataFrame()

Y["id"] = ids

Y["type"] = dummies.columns[voted]

Y.to_csv("submission_voter.csv",index=False)