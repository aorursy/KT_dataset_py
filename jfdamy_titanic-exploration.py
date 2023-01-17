import pandas as pd

df = pd.read_csv('../input/train.csv')
dfTestTemp = pd.read_csv('../input/test.csv')
del df['PassengerId']

del df['Name']

del df['Ticket']

del df['Cabin']

df['Family'] =  df["Parch"] + df["SibSp"]

del df["Parch"]

del df["SibSp"]



dfTest = dfTestTemp.drop("PassengerId",axis=1).copy()

del dfTest['Name']

del dfTest['Ticket']

del dfTest['Cabin']

dfTest['Family'] =  dfTest["Parch"] + dfTest["SibSp"]

del dfTest["Parch"]

del dfTest["SibSp"]
df = pd.get_dummies(df)

dfTest = pd.get_dummies(dfTest)

df.tail()
from sklearn.preprocessing import Imputer

imr = Imputer(missing_values='NaN', strategy='mean', axis=0)

imr = imr.fit(df.values[:, 1:])

imputed_data = imr.transform(df.values[:, 1:])

predic_data = imr.transform(dfTest.values)
from sklearn.cross_validation import train_test_split

X, y = imputed_data, df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
def precision(X_test_std, classifier):

    y_pred = classifier.predict(X_test_std)

    print('Misclassified samples: %d' % (y_test != y_pred).sum())

    from sklearn.metrics import accuracy_score

    print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
import numpy as np

from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt



feat_labels = df.columns[1:]



forest = RandomForestClassifier(n_estimators=10,

                                random_state=0,

                                n_jobs=-1)



forest.fit(X_train, y_train)

importances = forest.feature_importances_



indices = np.argsort(importances)[::-1]



for f in range(X_train.shape[1]):

    print("%2d) %-*s %f" % (f + 1, 30, 

                            feat_labels[indices[f]], 

                            importances[indices[f]]))



plt.title('Feature Importances')

plt.bar(range(X_train.shape[1]), 

        importances[indices],

        color='lightblue', 

        align='center')



plt.xticks(range(X_train.shape[1]), 

           feat_labels[indices], rotation=90)

plt.xlim([-1, X_train.shape[1]])

plt.tight_layout()

plt.show()
precision(X_test, forest)
forest.score(X_train, y_train)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

sc.fit(X_train)

X_train_std = sc.transform(X_train)

X_test_std = sc.transform(X_test)
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()



logreg.fit(X_train_std, y_train)



precision(X_test_std, logreg)
logreg.score(X_train_std, y_train)
precision(X_test, logreg)
logreg.predict([[3, 30, 263, 0, 0, 1, 0, 0, 1]]) #man 30 years old => die
logreg.predict([[3, 30, 263, 0, 1, 0, 0, 0, 1]]) #woman 30 years old => surviv
from pandas import DataFrame



# get Correlation Coefficient for each feature using Logistic Regression

coeff_df = DataFrame(df.columns.delete(0))

coeff_df.columns = ['Features']

coeff_df["Coefficient Estimate"] = pd.Series(logreg.coef_[0])



# preview

coeff_df
Y_predic = forest.predict(predic_data)

submission = pd.DataFrame({

        "PassengerId": dfTestTemp["PassengerId"],

        "Survived": Y_predic

    })

submission.to_csv('titanic_predic.csv', index=False)