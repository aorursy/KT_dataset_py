#import these libraries

import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

df = pd.read_excel('../input/lr-labels/LR_label.xlsx')

X = df.iloc[: , :-1]

Y = df.iloc[: , -1]

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

y = le.fit_transform(df['Q14'])

X = pd.get_dummies(df.drop('Q14', axis=1))

import pandas as pd

import numpy as np

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2

bestfeatures = SelectKBest(score_func=chi2, k=10)

fit = bestfeatures.fit(X,Y)

dfscores = pd.DataFrame(fit.scores_)

dfcolumns = pd.DataFrame(X.columns)

#concat two dataframes for better visualization 

featureScores = pd.concat([dfcolumns,dfscores],axis=1)

featureScores.columns = ['Specs','Score']  #naming the dataframe columns

print(featureScores.nlargest(10,'Score'))  #print 10 best features



# Import your necessary dependencies

from sklearn.feature_selection import RFE

from sklearn.linear_model import LogisticRegression

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20)



logreg = LogisticRegression(multi_class = 'multinomial', solver = 'newton-cg')

logreg = logreg.fit(X_train, Y_train)

output2 = logreg.predict(X_test)



logreg.intercept_

logreg.coef_

logreg.classes_

rfe = RFE(logreg, 5)

fit = rfe.fit(X, Y)

print("Num Features: %s" % (fit.n_features_))

print("Selected Features: %s" % (fit.support_))

print("Feature Ranking: %s" % (fit.ranking_))



from sklearn.linear_model import Ridge

ridge = Ridge(alpha=1.0)

ridge.fit(X,Y)

Ridge(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=None,normalize=False, random_state=None, solver='auto', tol=0.001)



# A helper method for pretty-printing the coefficients

def pretty_print_coefs(coefs, names = None, sort = False):

    if names == None:

        names = ["X%s" % x for x in range(len(coefs))]

    lst = zip(coefs, names)

    if sort:

        lst = sorted(lst,  key = lambda x:-np.abs(x[0]))

    return " + ".join("%s * %s" % (round(coef, 3), name)for coef, name in lst)



print ("Ridge model:", pretty_print_coefs(ridge.coef_))