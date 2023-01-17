#individual classification learners

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB





#voting classifier

from sklearn.ensemble import VotingClassifier



#voting classifier

from sklearn.ensemble import VotingClassifier



log_clf = LogisticRegression(C=1,solver="liblinear")

svc_clf = SVC(C=1,kernel='linear',gamma='auto')

naive_clf = GaussianNB()

voting_classifier = VotingClassifier(estimators=[("lr",log_clf),

                                                 ("svc",svc_clf),

                                                 ("naive",naive_clf)],

                                     voting="hard"

                                    )
#individual regression learners

from sklearn.linear_model import LinearRegression

from sklearn.svm import SVR

from sklearn.ensemble import RandomForestRegressor



#voting classifier

from sklearn.ensemble import VotingRegressor



lin_reg = LinearRegression()

svr_reg = SVR()

rf_reg = RandomForestRegressor(n_estimators=10, random_state=1)
voting_regressor = VotingRegressor(estimators=[("lin_reg",lin_reg),

                                                 ("svr_reg",svr_reg),

                                                 ("rf_reg",rf_reg)])
from sklearn.ensemble import BaggingRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import BaggingClassifier

from sklearn.tree import DecisionTreeClassifier
bag_reg_pasting = BaggingRegressor(base_estimator = DecisionTreeRegressor(),

                           n_estimators=500,

                           bootstrap=False, # using pasting

                           max_samples=1,   # use entire training data for each predictor 

                           n_jobs=-1)



bag_reg_bagging = BaggingRegressor(base_estimator = DecisionTreeRegressor(),

                           n_estimators=500,

                           bootstrap=True,    # using bagging

                           max_samples=0.8,   # use 80% training data for each predictor

                           oob_score=True,     # out-of-bag evaluation

                           n_jobs=-1)



bag_clf_pasting = BaggingClassifier(base_estimator = DecisionTreeClassifier(),

                           n_estimators=500,

                           bootstrap=False, # using pasting

                           max_samples=1,   # use entire training data for each predictor 

                           n_jobs=-1)



bag_clf_bagging = BaggingClassifier(base_estimator = DecisionTreeClassifier(),

                           n_estimators=500,

                           bootstrap=True,    # using bagging

                           max_samples=0.8,   # use 80% training data for each predictor

                           oob_score=True,     # out-of-bag evaluation

                           n_jobs=-1)
from sklearn.ensemble import AdaBoostRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import AdaBoostClassifier

from sklearn.tree import DecisionTreeClassifier
ada_reg = AdaBoostRegressor(base_estimator=DecisionTreeRegressor(),

                            n_estimators=100,

                            learning_rate=1.0)



ada_clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(),

                            n_estimators=100,

                            algorithm = "SAMME",

                            learning_rate=1.0)
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.ensemble import GradientBoostingClassifier
gbr = GradientBoostingRegressor(max_depth=3,n_estimators=3,learning_rate=1)

gbr = GradientBoostingClassifier(max_depth=3,n_estimators=3,learning_rate=1)
from sklearn.ensemble import StackingRegressor

from sklearn.ensemble import StackingClassifier



#first layer estimators

from sklearn.linear_model import RidgeCV

from sklearn.svm import LinearSVR



from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import LinearSVC



#second layer estimator

from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import LogisticRegression
reg_estimators_l1 = [ ('lr', RidgeCV()),('svr', LinearSVR(random_state=42))]

stack_reg = StackingRegressor(estimators=reg_estimators_l1,

                             final_estimator=RandomForestRegressor(n_estimators=10,random_state=42))

clf_estimators_l1 = [ ('rf', RandomForestClassifier()),('svc', LinearSVC(random_state=42))]

stack_reg = StackingRegressor(estimators=clf_estimators_l1,

                             final_estimator=LogisticRegression())
