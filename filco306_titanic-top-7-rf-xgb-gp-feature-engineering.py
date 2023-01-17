def create_submission(test, preds_test, file_name):

    predictions = []

    for pred in preds_test:

        if pred == 1:

            predictions.append(1)

        else:

            predictions.append(0)

    submission = pd.concat([test_orig['PassengerId'], pd.Series(predictions).astype("int")], axis=1)

    submission.columns = ['PassengerId', 'Survived']

    submission['Survived'] = submission['Survived'].astype("int")

    # Not here since we do not submit it.

    submission.to_csv(file_name, index = False)



# Returns a fitted and tuned model. Will also create predictions. 

def run_model(train, test, cv_grid_params,file_name = "submission.csv", regression = False, gauss_proc = False):

    from sklearn.model_selection import cross_val_score, GridSearchCV

    from sklearn.metrics import confusion_matrix

    

    X_train = train.drop(['Survived'], axis = 1)

    y_train = train['Survived']

    X_test = test.drop(['Survived'], axis = 1)

    y_test = test['Survived']



    grid_search = GridSearchCV(**cv_grid_params)

    grid_search.fit(X_train,y_train)

    

    model = cv_grid_params['estimator']

    

    model.set_params(**grid_search.best_params_)

    model.fit(X_train,y_train)

    cv_scores = cross_val_score(model, X_train, y_train, cv = 10)

    print("Cross validation scores:" + str(cv_scores))

    print("Mean score: " + str(cv_scores.mean()))

    test_preds = model.predict(X_test)

    if regression == False:

        print("Train confusion matrix:" )

        print(confusion_matrix(y_train.astype("int"), model.predict(X_train).astype("int")))

        print("Predicting and creating submission. ")

        create_submission(X_test, test_preds.astype("float"), file_name)

    else:

        train_preds = model.predict(X_train)

        

        preds_test = []

        preds_train = []

        for i in range(train_preds.shape[0]):

            if train_preds[i] < 0.5:

                preds_train.append(0)

            else:

                preds_train.append(1)

        print("Train confusion matrix:" )

        print(confusion_matrix(y_train, np.array(preds_train)))

        for i in range(test_preds.shape[0]):

            if test_preds[i] < 0.5:

                preds_test.append(0)

            else:

                preds_test.append(1)

        print("Predicting and creating ridge submission. ")

        create_submission(X_test, preds_test, file_name)

    

    return test_preds, model

        

    
import numpy as np

import pandas as pd



# Set seed to always be used

seed = 123



train_orig = pd.read_csv("../input/train.csv")

test_orig = pd.read_csv("../input/test.csv")

data = pd.concat([test_orig,train_orig], sort = False)

data = data.reset_index()



class_weights = {

    0: (train_orig['Survived'] == 0).sum()/train_orig.shape[0],

    1: (train_orig['Survived'] == 1).sum()/train_orig.shape[0]

}



data.describe()
data.isna().sum()
# Fill the missing one. 

data['Fare'] = data['Fare'].fillna(data['Fare'].dropna().median())
print(data['Cabin'].isnull().sum())

data['Cabin_dep'] = [cabin_no[0] for cabin_no in data['Cabin'].astype("str")]

data['Cabin_dep'] = data['Cabin_dep'].astype("category")
from matplotlib.pyplot import hist

import matplotlib.pyplot as plt



from scipy.stats import gaussian_kde

hist(data['Age'].dropna(), density = True)

from scipy.stats import poisson



dens0 = gaussian_kde(data.Age[data['Survived'] == 0].dropna())

dens1 = gaussian_kde(data.Age[data['Survived'] == 1].dropna())

densNA = gaussian_kde(data.Age[data['Survived'].isna()].dropna())



x0 = np.arange(0,data.Age[data['Survived'] == 0].dropna().max())

x1 = np.arange(0,data.Age[data['Survived'] == 1].dropna().max())

xNA = np.arange(0,data.Age[data['Survived'].isna()].dropna().max())



plt.plot(x0, dens0.evaluate(x0), 'r')

plt.plot(x1, dens1.evaluate(x1), 'g')

plt.plot(xNA, densNA.evaluate(xNA), 'b')

plt.legend(["Died", "Survived", "Test dist"])

plt.title("Density slightly different, but not very. Still worth to separate sampling though. ")



dist0 = dens0.evaluate(x0)

# Normalize

dist0 = np.divide(dist0,np.sum(dist0))

dist0 = dens0.evaluate(x0)

# Normalize

dist0 = np.divide(dist0,np.sum(dist0))



dist1 = dens1.evaluate(x1)

# Normalize

dist1 = np.divide(dist1,np.sum(dist1))

dist1 = dens1.evaluate(x1)

# Normalize

dist1 = np.divide(dist1,np.sum(dist1))



distNA = densNA.evaluate(xNA)

# Normalize

distNA = np.divide(distNA,np.sum(distNA))

distNA = dens1.evaluate(xNA)

# Normalize

distNA = np.divide(distNA,np.sum(distNA))





# We should sample out of this distribution to compensate. 

np.random.seed(seed)

nan_ages0 = np.random.choice(x0, p = dist0, size = data['Age'].isnull().sum())

nan_ages1 = np.random.choice(x1, p = dist1, size = data['Age'].isnull().sum())

nan_agesNA = np.random.choice(xNA, p = distNA, size = data['Age'].isnull().sum())



count = 0

for i in list(data.Age[data['Survived'] == 0].index[np.where(data.Age[data['Survived'] == 0].isna())]):

    data['Age'][i] = nan_ages0[count]

    count += 1

        

count = 0

for i in list(data.Age[data['Survived'] == 1].index[np.where(data.Age[data['Survived'] == 1].isna())]):

    data['Age'][i] = nan_ages1[count]

    count += 1



count = 0

for i in list(data.Age[data['Survived'].isna()].index[np.where(data.Age[data['Survived'].isna()].isna())]):

    data['Age'][i] = nan_agesNA[count]

    count += 1
from scipy.stats import zscore



indices_age = data.Age[data['Age'].isnull() == False].index

z_scores_age = np.abs(zscore(data.Age[data['Age'].isnull() == False]))





z_scores = pd.DataFrame(columns=['Age','SibSp','Parch','Fare'])

z_scores['SibSp'] = zscore(data.SibSp)

z_scores['Parch'] = zscore(data.Parch)

z_scores['Fare'] = zscore(data.Fare)

z_scores['Age'] = zscore(data.Age)

    

outlier_rows, outlier_cols = np.where(z_scores > 3)



outliers = {}



for i,col in enumerate(outlier_rows):

    if outlier_rows[i] not in outliers:

        outliers[outlier_rows[i]] = [outlier_cols[i]]

    else:

        outliers[outlier_rows[i]].append(outlier_cols[i])



        

# Remove all being outlier in 2 or more columns

to_del = []

for outlier in outliers:

    if len(outliers[outlier]) >= 2 and outlier >= test_orig.shape[0]:

        to_del.append(outlier)



print(len(to_del))

# Now delete all in list to_del

for i in to_del:

    data.drop(data.index[i], inplace=True)
data.isnull().sum()
data['Sex'].isnull().sum()

data['Sex'] = data['Sex'].astype('str')

data['Sex'] = data['Sex'].replace({'male':1, 'female':-1})

x=np.arange(1,4,2)

plt.bar(x, data.Sex[data['Survived'] == 1].value_counts().sort_index(), width = 0.5)

plt.title("Gender distributions of people dying and surviving")

plt.xticks(x - 0.25,["Female", "Male"])

plt.bar(x-0.5, data.Sex[data['Survived'] == 0].value_counts().sort_index(), width = 0.5, color = "r")

plt.legend(["Survived","Died"])
plt.plot(data.Fare[data['Survived'] == 1], 'ro')

plt.plot(data.Fare[data['Survived'] == 0], 'bo')

plt.legend(["Survived", "Died"])

print("Survived: Mean: "+str(np.mean(data.Fare[data['Survived'] == 1]))+ ", sd: " + str(np.std(data.Fare[data['Survived'] == 1])))

print("Died: Mean: "+str(np.mean(data.Fare[data['Survived'] == 0]))+ ", sd: " + str(np.std(data.Fare[data['Survived'] == 0])))
x = np.arange(0,14,2)

plt.bar(x - 0.25, data.Parch[data['Survived'] == 0].value_counts().sort_index(), width = 0.5)

plt.bar(2*data.Parch[data['Survived'] == 1].value_counts().sort_index().index + 0.25, data.Parch[data['Survived'] == 1].value_counts().sort_index(), color = 'r', width = 0.5)

plt.xticks(x,np.arange(0,7,1))

plt.legend(['Survived', 'Died'])

plt.show()
data.loc[data['Parch'] >= 3, 'Parch'] = 3

print(data.Parch.value_counts().sort_index())

x = np.arange(0,8,2)

plt.bar(x - 0.25, data.Parch[data['Survived'] == 0].value_counts().sort_index(), width = 0.5, color = "r")

plt.bar(2*data.Parch[data['Survived'] == 1].value_counts().sort_index().index + 0.25, data.Parch[data['Survived'] == 1].value_counts().sort_index(), color = 'b', width = 0.5)

plt.xticks(x,np.arange(0,7,1))

plt.legend(['Died', 'Survived'])
x = np.array([0,1,2,3,4,5,8])



plt.bar(np.array([0,2,4,6,8]) - 0.25,data.SibSp[data['Survived'] == 1].value_counts().sort_index(), color="b", width=0.5)

plt.bar(2*np.array([0,1,2,3,4,5,8]) + 0.25,data.SibSp[data['Survived'] == 0].value_counts().sort_index(), color = "r", width=0.5)

plt.legend(["Survived", "Died"])

plt.xticks(np.arange(0,18,2),np.arange(0,9))

plt.show()
data.loc[data['SibSp'] >= 4, 'SibSp'] = 4



x = np.arange(0,10,2)



plt.bar(x - 0.25,data.SibSp[data['Survived'] == 1].value_counts().sort_index(),width=0.5, color="b")

plt.bar(x + 0.25,data.SibSp[data['Survived'] == 0].value_counts().sort_index(),width=0.5, color="r")

plt.legend(['Survived', 'Died'])

plt.xticks(x,x/2)

plt.show()
data.Name.head()
#print(['yes' for x in [x.lower() for x in list(data['Name'].values)] if 'master' in x])

def create_title(name_col):

    import re

    for name in name_col:

        matches = re.findall("\w+\.",name)

        yield matches[0]



data['Title'] = list(create_title(data['Name']))

#print(data['Title'].value_counts())

data['Title'] = data['Title'].replace({'Mlle.':'Miss.','Ms.':'Miss.','Jonkheer.':"Mr.","Major.":"Mr.","Countess.":"Mrs.","Don.":"Mr.", "Dona.":"Mrs.","Sir.":"Mr.","Mme.":"Miss.", "Col.":"Military", "Major.":"Military","Capt.":"Military", "Lady.":"Mrs."})

print(data['Title'].value_counts())



x = np.arange(0,14,2)



plt.bar(x - 0.25, data.Title[data['Survived'] == 0].value_counts().sort_index(), color = "r", width = 0.5)

plt.bar(x[:6] + 0.25, data.Title[data['Survived'] == 1].value_counts().sort_index(), color = "b", width = 0.5)

plt.xticks(x,["Dr.", "Master.", "Military","Miss.","Mr.","Mrs.","Revenant"])

plt.legend(["Died","Survived"])

plt.show()

data.Ticket.head(20)

data.Ticket.loc[20]



ticket_type = []



import re



#print(data.Ticket)

for ticket in data.Ticket.values:

    if ticket.isdigit():

        ticket_type.append("Numbered")

    else:

        ticket_type.append([re.sub('[^a-zA-Z_]','', x)[0:1] for x in ticket.split(" ")][0])

        

data['Ticket_type'] = ticket_type

x = np.arange(0,16,2)

plt.bar(x - 0.25,data.Ticket_type[data['Survived'] == 0].value_counts().sort_index(), color = "r", width = 0.5)

plt.bar(x + 0.25, data.Ticket_type[data['Survived'] == 1].value_counts().sort_index(), color = "b", width = 0.5)

plt.legend(['Died', 'Survived'])

plt.title("Ticket type dying vs ")

plt.xticks(x,data.Ticket_type[data['Survived'] == 0].value_counts().sort_index().index)

plt.show()
data = pd.get_dummies(data.drop(['Cabin','Ticket','Name'], axis = 1))

data.head()
train, test = [x for _, x in data.groupby(data['Survived'].isnull())]

train = train.drop(['index', 'PassengerId'], axis = 1)

test = test.drop(['index', 'PassengerId'], axis = 1)

"""

# The upsampling, which later was chosen not to use. 

n_dying = train[train['Survived'] == 0].shape[0]

n_surviving = train[train['Survived'] == 1].shape[0]

n_to_sample = n_dying - n_surviving

print("Resampling "+str(n_to_sample)+" samples from training set. ")



resamples = train[train['Survived'] == 1].sample(n_to_sample, axis = 0)



train = pd.concat([train, resamples])"""



train.Survived.value_counts()
X_train = train.drop(['Survived'], axis = 1)

y_train = train['Survived']

X_test = test.drop(['Survived'], axis = 1)
from sklearn.kernel_ridge import KernelRidge



kern_ridge = KernelRidge()



params = {'alpha': [0.01,0.1, 1.0],

 'coef0': [0,0.1,1],

 'degree': [1,2,3],

 'gamma': [0.01,1,10],

 'kernel': ['rbf']

}



grid_params = {

    'estimator':kern_ridge,

    'param_grid':params,

    'n_jobs':5,

    'iid':False,

    'verbose':True,

    'scoring':'neg_mean_squared_error',

    'cv':10

}



y_preds_ridge, ridge_model = run_model(train, test, grid_params, file_name = "submission_ridge.csv", regression = True)
from sklearn.linear_model import LogisticRegression



log_reg_model = LogisticRegression()



log_reg_params = {'C': [0.1,1.0,10,100], 

                  'class_weight': [None], 

                  'dual': [None], 

                  'fit_intercept': [True,False], 

                  'max_iter': [10000], 

                  'multi_class': ['ovr'],

                  'class_weight':[class_weights],

                  'n_jobs': [1],

                  'penalty': ['l1','l2'],

                  'random_state': [123],  

                  'tol': [0.0001, 0.0005,0.001], 

                  'solver':['saga'],

                  'warm_start': [False]}

grid_params['estimator'] = log_reg_model

grid_params['param_grid'] = log_reg_params

grid_params['scoring'] = 'accuracy'



test_preds_log_reg, log_reg_model = run_model(train, test, grid_params, file_name = "submission_log_reg.csv")
from xgboost import XGBClassifier



xgb_model = XGBClassifier()



params_xgb = {

    'base_score':[0.3,0.5],

    'colsample_bytree':[0.4,0.7],

    'gamma':[0.01,0.5,0.9],

    'min_child_weight':[1,3],

    'learning_rate':[0.01,0.1,1],

    'max_depth':[3,4,5],

    'n_estimators':[500],

    'reg_alpha':[1e-5, 0.1],

    'reg_lambda':[1e-5, 0.1],

    'subsample':[0.8]

}



grid_params['estimator'] = xgb_model

grid_params['param_grid'] = params_xgb

grid_params['n_jobs'] = 5



test_preds_xgb, xgb_model = run_model(train, test, grid_params, file_name = "submission_xgb.csv")

from sklearn.preprocessing import scale

# Scale the data 

test_gp = test.copy()

train_gp = train.copy()

data_gp = pd.concat([test_gp,train_gp])

data_gp[['Age','Fare','SibSp','Parch']] = scale(data_gp[['Age','Fare','SibSp','Parch']])





print(data_gp.head())



data_gp['Survived'].replace({1:1, 0:-1})



from sklearn.preprocessing import scale

# Time to scale the numerical variables!

train_gp, test_gp = [x for _, x in data_gp.groupby(data_gp['Survived'].isnull())]



print(train_gp.shape)
from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.gaussian_process.kernels import RBF

params = {'copy_X_train': [True],

 'kernel': [RBF(1.0),RBF(0.01),RBF(0.001),RBF(10), RBF(5), RBF(3), RBF(2)],

 'max_iter_predict': [100],

 'multi_class': ['one_vs_rest'],

 'n_jobs': [3],

 'n_restarts_optimizer': [0],

 'optimizer': ['fmin_l_bfgs_b'],

 'random_state': [123],

 'warm_start': [False]

         }



grid_params['estimator'] = GaussianProcessClassifier()

grid_params['param_grid'] = params



test_preds_gauss, gp_model = run_model(train_gp, test_gp, grid_params, file_name = "submission_gauss_process.csv")
from sklearn.svm import SVC



params = {'C': [0.01,0.1,0.5,1.0,10,50],

 'class_weight': [class_weights],

 'coef0': [0.0,0.1,0.5,1],

 'decision_function_shape': ['ovr'],

 'degree': [2,3],

 'gamma': ['auto_deprecated'],

 'kernel': ['rbf','poly'],

 'max_iter': [-1],

 'probability': [False],

 'random_state': [123],

 'shrinking': [True],

 'tol': [0.001,0.003,0.005,0.01]}



grid_params['estimator'] = SVC()

grid_params['param_grid'] = params



test_preds_svm, svm_model = run_model(train_gp, test_gp, grid_params, file_name = "submission_gauss_process.csv")
from sklearn.ensemble import RandomForestClassifier



rf_model = RandomForestClassifier()



params_rf = {

     'bootstrap': [True,False],

     'class_weight': [None],

     'criterion': ['gini','entropy'],

     'max_depth': [None],

     'max_features': ['auto'],

     'max_leaf_nodes': [None],

     'min_impurity_decrease': [0.0,0.01,0.1,1],

     'min_impurity_split': [None],

     'min_samples_leaf': [1,5,10,15],

     'min_samples_split': [2,5,10],

     'min_weight_fraction_leaf': [0.0],

     'n_estimators': [5,10,20,100],

     'n_jobs': [None],

     'oob_score': [False],

     'random_state': [123]

            }



grid_params['estimator'] = rf_model

grid_params['param_grid'] = params_rf



test_preds_rf, rf_model = run_model(train, test, grid_params, file_name = "submission_rf.csv")
preds = pd.concat([pd.Series(test_preds_rf),pd.Series(test_preds_gauss),pd.Series(test_preds_xgb),pd.Series(test_preds_svm)], axis = 1)



preds.columns = ['Random Forest', 'Gaussian Process', 'XGBoost', 'SVM']

preds.corr()



import seaborn as sns



sns.heatmap(preds.corr(), xticklabels=preds.columns, yticklabels=preds.columns, annot=True)

from sklearn.ensemble import VotingClassifier

svm_model.set_params(probability = True)

vc_model = VotingClassifier([('rf',rf_model),('xgb',xgb_model),('gp',gp_model),('svm',svm_model)], voting = 'soft')



X_train = train_gp.drop(['Survived'], axis = 1)

y_train = train_gp['Survived']

X_test = test_gp.drop(['Survived'],axis = 1)



vc_model.fit(X_train, y_train)



preds_train = vc_model.predict(X_train)



from sklearn.metrics import confusion_matrix



print(confusion_matrix(y_train, preds_train))

y_preds_test = vc_model.predict(X_test)



create_submission(test_gp,y_preds_test, "submission_vc_soft.csv")