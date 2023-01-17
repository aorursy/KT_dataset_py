import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier  
from sklearn import ensemble

from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.cross_validation import train_test_split

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

all_X = train.iloc[:, 2:-1].values
all_y = train.iloc[:, 21].values
holdout = test
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(all_X)
all_X = imputer.transform(all_X)


train_X, test_X, train_y, test_y = train_test_split(all_X, all_y, test_size=0.2, random_state=0)

sc = StandardScaler()
train_X = sc.fit_transform(train_X)
test_X = sc.transform(test_X)

lr = LogisticRegression()
lr.fit(all_X, all_y)

columns = [ 'GP', 'MIN', 'PTS', 'FGM',
       'FGA','FG%',
       '3P Made', '3PA',
       '3P%', 'FTM',
       'FTA', 'FT%' , 'OREB' , 'DREB' , 'REB' , 'AST' , 'STL'
        ,'BLK' , 'TOV']

holdout_predictions = lr.predict(holdout[columns])


holdout_ids = holdout["PlayerID"]
#submission_df = {"PlayerID": holdout_ids,
                # "TARGET_5Yrs": holdout_predictions}
#submission = pd.DataFrame(submission_df)

#submission.to_csv("submissionlgrg.csv", index=False)
#print(submission)
GBC = ensemble.GradientBoostingClassifier(n_estimators = 2000, learning_rate = 0.001)
GBC.fit(all_X, all_y)
GBCpredictions = GBC.predict(test_X)
f1_score(test_y, GBCpredictions, average='binary')

GBC_predictions = GBC.predict(holdout[columns])
holdout_ids = holdout["PlayerID"]
submission_df = {"PlayerID": holdout_ids,
                 "TARGET_5Yrs": GBC_predictions}
submission = pd.DataFrame(submission_df)
submission.to_csv("submission2GBC.csv", index=False)
print(submission)






GBC = ensemble.GradientBoostingClassifier(n_estimators = 2000, learning_rate = 0.001)
GBC.fit(all_X, all_y)
GBCpredictions = GBC.predict(test_X)
f1_score(test_y, GBCpredictions, average='binary')  

#RandomForest
#rf = RandomForestClassifier(n_estimators = 120, criterion = 'entropy')
#rf.fit(all_X, all_y)
#rfpredictions = rf.predict(test_X)
#f1_score(test_y, rfpredictions, average='binary') 
 
#svc = SVC(kernel = 'rbf')
#svc.fit(all_X, all_y)

#NaiveBayes
#nb = GaussianNB()
#nb.fit(all_X, all_y)

#DecisionTree
#classifier = DecisionTreeClassifier(criterion = 'entropy')
#classifier.fit(all_X, all_y)




