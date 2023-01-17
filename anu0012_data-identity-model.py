import numpy as np
import os
import pandas as pd
import sys
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from sklearn.svm import LinearSVC,SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, ExtraTreesClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, roc_auc_score, recall_score, precision_score,make_scorer
import seaborn as sns
sns.set(color_codes=True)
from scipy import stats
from scipy.stats import norm, skew #for some statistics
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
np.random.seed(25)
import os
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15
plt.rcParams['font.size'] = 15
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
train.head()
train.nunique()
# train['duration_per_program'] = train['program_duration'] / train['total_programs_enrolled']
# test['duration_per_program'] = test['program_duration'] / test['total_programs_enrolled']
train.isnull().sum(axis=0)
train.fillna({"age": train["age"].median(), "trainee_engagement_rating": train["trainee_engagement_rating"].mode()[0]},inplace=True)
test.fillna({"age": test["age"].median(), "trainee_engagement_rating": test["trainee_engagement_rating"].mode()[0]},inplace=True)
sns.countplot(train['is_pass'])
sns.distplot(train['program_duration'])
sns.distplot(train['test_id'])
sns.distplot(train['age'])
sns.distplot(train['city_tier'])
sns.distplot(train['total_programs_enrolled'])
sns.distplot(train['trainee_engagement_rating'])
sns.countplot(x="is_handicapped", data=train, palette="Greens_d");
sns.countplot(y="education", data=train, palette="Greens_d");
sns.countplot(x="gender", data=train, palette="Greens_d");
sns.countplot(x="difficulty_level", data=train, palette="Greens_d");
sns.countplot(x="test_type", data=train, palette="Greens_d");
sns.countplot(x="program_type", data=train, palette="Greens_d");
sns.pointplot(y="is_pass", x="test_type", hue="gender", data=train);
sns.pointplot(y="is_pass", x="difficulty_level", hue="gender", data=train);
sns.pointplot(y="is_pass", x="difficulty_level", hue="test_type", data=train);
sns.pointplot(y="is_pass", x="gender",  data=train);
plt.figure(figsize=(12,6))
sns.pointplot(y="is_pass", x="education", data=train);
sns.pointplot(y="is_pass", x="is_handicapped", data=train);
train.dtypes
# Label encoding
from sklearn.preprocessing import LabelEncoder
lb_make = LabelEncoder()

train['program_id'] = lb_make.fit_transform(train['program_id'])
train['program_type'] = lb_make.fit_transform(train['program_type'])
train['test_type'] = lb_make.fit_transform(train['test_type'])
train['difficulty_level'] = lb_make.fit_transform(train['difficulty_level'])
train['gender'] = lb_make.fit_transform(train['gender'])
train['education'] = lb_make.fit_transform(train['education'])
train['is_handicapped'] = lb_make.fit_transform(train['is_handicapped'])

test['program_id'] = lb_make.fit_transform(test['program_id'])
test['program_type'] = lb_make.fit_transform(test['program_type'])
test['test_type'] = lb_make.fit_transform(test['test_type'])
test['difficulty_level'] = lb_make.fit_transform(test['difficulty_level'])
test['gender'] = lb_make.fit_transform(test['gender'])
test['education'] = lb_make.fit_transform(test['education'])
test['is_handicapped'] = lb_make.fit_transform(test['is_handicapped'])

# train=pd.get_dummies(train)
# test=pd.get_dummies(test)
#correlation matrix
corrmat = train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, square=True);
feature_names = [x for x in train.columns if x not in ['id', 'is_pass']]
target = train['is_pass']
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
np.random.seed(42)
def create_baseline():
    # create model
    model = Sequential()
    model.add(Dense(100, input_dim=14, kernel_initializer='normal', activation='relu'))
    model.add(Dense(70, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
# model = KerasClassifier(build_fn=create_baseline, epochs=25, batch_size=32, verbose=1)
# # evaluate model with standardized dataset
# estimators = []
# estimators.append(('standardize', StandardScaler()))
# estimators.append(('mlp', model))
# pipeline = Pipeline(estimators)
# results = cross_val_score(pipeline, train[feature_names].values, target.values, cv=2, scoring = make_scorer(f1_score))
# print("Results: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
# # Splitting the dataset into the Training set and Test set
# from sklearn.cross_validation import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(train[feature_names], target, test_size = 0.25, random_state = 42)

# model = DecisionTreeClassifier()#CatBoostClassifier(depth=10, iterations=500, learning_rate=0.01, eval_metric='AUC', random_seed=42,verbose=False)

# ## model training and prediction
# model.fit(X_train, y_train)
# pred = model.predict_proba(X_test)

# ## model performance evaluation using different matrices
# print("Voting Model :\n")
# print("confusion_matrix: " + str(confusion_matrix(pred, y_test.values)))
# print("recall_score: " + str(recall_score(pred, y_test.values)))
# print("precision_score: " + str(precision_score(pred, y_test.values)))
# print("f1_score: " + str(f1_score(pred, y_test.values)))
# print("accuracy_score: " + str(accuracy_score(pred, y_test.values)))
# print("roc_auc_score: " + str(roc_auc_score(pred, y_test.values)))
vote_est = [
    ('gbc',GradientBoostingClassifier())
    #('cat',CatBoostClassifier(depth=8, iterations=5000, learning_rate=0.1, eval_metric='AUC', random_seed=42,verbose=False))
]

model = VotingClassifier(estimators = vote_est , voting = 'soft')
#model = CatBoostClassifier(depth=13, iterations=1500, learning_rate=0.01, eval_metric='AUC', random_seed=42,verbose=False)

## model training and prediction
model.fit(train[feature_names], target)
pred = model.predict_proba(test[feature_names])
## make submission
sub = pd.DataFrame()
sub['id'] = test['id']
sub['is_pass'] = [i[1] for i in pred]
sub.to_csv('result.csv', index=False)
sub.head(11)
