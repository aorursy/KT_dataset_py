import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier

from sklearn.metrics import roc_curve
train = pd.read_csv('../input/train.csv')

test_features = pd.read_csv('../input/test.csv')
feature_columns = [col for col in train.columns if col != 'Survived'] 

train_features = train[feature_columns]

train_classifier = train['Survived']
feature_columns
age_not_NaN = pd.notnull(train_features['Age'])

training_cols = ['Pclass','Age']
train_features[training_cols][age_not_NaN]
# split the given training set up into a test part (20%) and a training part (80%)

features = train[training_cols][age_not_NaN]

classifier = train['Survived'][age_not_NaN]

feature_train, feature_test, class_train, class_test = train_test_split(features, classifier,

                                                                        test_size=0.2, random_state=1)
# train the model. Called forest but it's actually gradient boosting classifier not random forest...

forest = GradientBoostingClassifier(n_estimators=1000, learning_rate=0.5, loss='deviance')

forest = forest.fit(feature_train, class_train)
output = forest.predict(feature_test)

proba_output = forest.predict_proba(feature_test)
fpr, tpr, thresholds = roc_curve(class_test, proba_output[:,1])
output
# proba output has (p) and (1-p) columns

proba_output[:,1]
plt.plot(fpr,tpr,label="True positive rate")

plt.plot(fpr,thresholds,"--g",label="Probability threshold for surviving")

plt.xlim([-0.05, 1.05])

plt.ylim([-0.05, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate or Probability Threshold')

plt.title('ROC Curve')

plt.legend(loc="right")

plt.savefig('ROC.png',dpi=300)