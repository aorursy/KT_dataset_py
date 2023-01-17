import pandas as pd

from xgboost import XGBClassifier

from sklearn.ensemble import BaggingClassifier

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.ensemble import VotingClassifier
test = pd.read_csv("../input/test.csv")

train = pd.read_csv("../input/train.csv")
X = train.drop(["label"],axis=1)

y = train["label"]
scale = StandardScaler()

X = scale.fit_transform(X)

test = scale.transform(test)
X_train,X_eval,y_train,y_eval = train_test_split(X,y,test_size=0.2,stratify=y)
xgb_model = XGBClassifier()
bag_model = BaggingClassifier()

xt_model = ExtraTreesClassifier()
voting = VotingClassifier(estimators=[('xgb',xgb_model),('bag',bag_model), ('xt',xt_model)])
voting.fit(X_train,y_train)
voting.score(X_eval,y_eval)
predict = voting.predict(test)
my_submission = pd.DataFrame({'ImageId': list(range(1,len(predict)+1)), 'Label': predict})

my_submission.to_csv('submission_xgb2.csv', index=False)