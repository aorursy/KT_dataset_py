%matplotlib inline

RANDOM_STATE = 0



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

pd.options.display.float_format = '{:.0f}'.format

from sklearn.model_selection import GridSearchCV

from sklearn import metrics

from sklearn.metrics import confusion_matrix, precision_recall_curve, auc, roc_auc_score, roc_curve, recall_score, classification_report

from pandas import Series

from sklearn.preprocessing import LabelEncoder, OneHotEncoder



from catboost import CatBoostClassifier

from sklearn.model_selection import ParameterGrid

from sklearn.model_selection import StratifiedKFold

from tqdm import tqdm



import warnings

warnings.filterwarnings("ignore")



import eli5 

from eli5.sklearn import PermutationImportance



import itertools
def summary(df):

    summary = pd.DataFrame(df.dtypes, columns=['dtypes'])

    summary = summary.reset_index()

    summary['Missing'] = df.isnull().sum().values    

    summary['Uniques'] = df.nunique().values

    return summary
def plot_cf_matrix_and_roc(model, 

                           X_train, 

                           y_train,

                           X_test, 

                           y_test,

                           y_pred, 

                           classes=[0,1],

                           normalize=False,

                           cmap=plt.cm.Blues):

    metrics_list = []

    

    # the main plot

    plt.figure(figsize=(15,5))



    # the confusion matrix

    plt.subplot(1,2,1)

    cm = confusion_matrix(y_test, y_pred)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=0)

    plt.yticks(tick_marks, classes)



    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        plt.title("Normalized confusion matrix")

    else:

        plt.title('Confusion matrix')



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        if normalize:

            plt.text(j, i, "{:0.2f}".format(cm[i, j]),

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")

        else:

            plt.text(j, i, format(cm[i, j]),

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    

    # the result metrix

    summary_df = pd.DataFrame([[str(np.unique( y_pred )),

                               str(round(metrics.precision_score(y_test, y_pred.round()),3)),

                               str(round(metrics.accuracy_score(y_test, y_pred.round()),3)),

                               str(round(metrics.recall_score(y_test, y_pred.round(), average='binary'),3)),

                               str(round(metrics.roc_auc_score(y_test, y_pred.round()),3)),

                               str(round(metrics.cohen_kappa_score(y_test, y_pred.round()),3)),

                               str(round(metrics.f1_score(y_test, y_pred.round(), average='binary'),3))]], 

                              columns=['Class', 'Precision', 'Accuracy', 'Recall', 'ROC-AUC', 'Kappa', 'F1-score'])

    # print the metrics

    print("\n");

    print(summary_df);

    print("\n");

    

    plt.show()
def cross_val(X, y, param, cat_features='', class_weights = '', n_splits=3):

    results = []

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    

    for tr_ind, val_ind in skf.split(X, y):

        X_train_i = X.iloc[tr_ind]

        y_train_i = y.iloc[tr_ind]

        

        X_valid_i = X.iloc[val_ind]

        y_valid_i = y.iloc[val_ind]

        

        if class_weights == '' :

            clf = CatBoostClassifier(iterations=param['iterations'],

                            loss_function = param['loss_function'],

                            depth=param['depth'],

                            l2_leaf_reg = param['l2_leaf_reg'],

                            eval_metric = param['eval_metric'],

                            leaf_estimation_iterations = 10,

                            use_best_model=True,

                            logging_level='Silent',

                            od_type="Iter",

                            early_stopping_rounds=param['early_stopping_rounds']

            )

        else:

            clf = CatBoostClassifier(iterations=param['iterations'],

                            loss_function = param['loss_function'],

                            depth=param['depth'],

                            l2_leaf_reg = param['l2_leaf_reg'],

                            class_weights = class_weights,

                            eval_metric = param['eval_metric'],

                            leaf_estimation_iterations = 10,

                            use_best_model=True,

                            logging_level='Silent',

                            od_type="Iter",

                            early_stopping_rounds=param['early_stopping_rounds']

            )

        

        

        if cat_features == '' :

            clf.fit(X_train_i, 

                    y_train_i,

                    eval_set=(X_valid_i, y_valid_i)

            )

        else:

            clf.fit(X_train_i, 

                    y_train_i,

                    cat_features=cat_features,

                    eval_set=(X_valid_i, y_valid_i)

            )

        

        # predict

        y_pred = clf.predict(X_valid_i)

        

        # select the right metric

        if(param['eval_metric'] == 'Recall'):

            metric = metrics.recall_score(y_valid_i, y_pred)

        elif(param['eval_metric'] == 'Accuracy'):

            metric = metrics.accuracy_score(y_valid_i, y_pred)

        elif(param['eval_metric'] == 'F1'):

            metric = metrics.f1_score(y_valid_i, y_pred)

        elif(param['eval_metric'] == 'AUC'):

            metric = metrics.roc_auc_score(y_valid_i, y_pred)

        elif(param['eval_metric'] == 'Kappa'):

            metric = metrics.cohen_kappa_score(y_valid_i, y_pred)

        else:

            metric = metrics.accuracy_score(y_valid_i, y_pred)

        

        #append the metric

        results.append(metric)

        

        print('Classes: '+str(np.unique( y_pred )))

        print('Precision: '+str(round(metrics.precision_score(y_valid_i, y_pred.round()),3)))

        print('Accuracy: '+str(round(metrics.accuracy_score(y_valid_i, y_pred.round()),3)))

        print('Recall: '+str(round(metrics.recall_score(y_valid_i, y_pred.round(), average='binary'),3)))

        print('Roc_Auc: '+str(round(metrics.roc_auc_score(y_valid_i, y_pred.round()),3)))

        print('F1 score: '+str(round(metrics.f1_score(y_valid_i, y_pred.round(), average='binary'),3)))

        print('Mean and standard deviation for '+param['eval_metric']+' oof prediction: ',np.mean(results),np.std(results))

        print("\n")

    return sum(results)/n_splits
def catboost_GridSearchCV(X, y, params, cat_features='', class_weights='', n_splits=5):

    ps = {'score':0,'param': []}

    for prms in tqdm(list(ParameterGrid(params)), ascii=True, desc='Params Tuning:'):

        score = cross_val(X, y, prms, cat_features, class_weights, n_splits)

        if score > ps['score']:

            ps['score'] = score

            ps['param'] = prms

    print('Score: '+str(ps['score']))

    print('Params: '+str(ps['param']))

    return ps['param']
# Load in the train and test datasets from the CSV files

train_df = pd.read_csv('../input/titanic/train.csv')

test_df = pd.read_csv('../input/titanic/test.csv')



# to cleanup and modify the data

data_cleaner = [train_df, test_df]
train_df.head()
train_df.describe()
train_df.info()
train_df.shape
train_df.isnull().sum()
train_df.info()
train_df['Initial']=0

for i in train_df:

    train_df['Initial']=train_df.Name.str.extract('([A-Za-z]+)\.') #lets extract the Salutations
test_df['Initial']=0

for i in test_df:

    test_df['Initial']=test_df.Name.str.extract('([A-Za-z]+)\.') #lets extract the Salutations
pd.crosstab(train_df.Initial,train_df.Sex).T.style.background_gradient(cmap='summer_r')
pd.crosstab(test_df.Initial,test_df.Sex).T.style.background_gradient(cmap='summer_r')
newtitles={

    "Capt":       "Officer",

    "Col":        "Officer",

    "Major":      "Officer",

    "Jonkheer":   "Royalty",

    "Don":        "Royalty",

    "Sir" :       "Royalty",

    "Dr":         "Officer",

    "Rev":        "Officer",

    "the Countess":"Royalty",

    "Dona":       "Royalty",

    "Mme":        "Mrs",

    "Mlle":       "Miss",

    "Ms":         "Mrs",

    "Mr" :        "Mr",

    "Mrs" :       "Mrs",

    "Miss" :      "Miss",

    "Master" :    "Master",

    "Lady" :      "Royalty"}
#train_df[train_df['Initial'].isnull()]
#test_df[test_df['Initial'].isnull()]
#train_df['Initial']=train_df.Initial.map(newtitles)

#test_df['Initial']=test_df.Initial.map(newtitles)
# Miss

train_df['Initial']=train_df['Initial'].replace(['Mlle', 'Mme','Ms'], 'Miss')

test_df['Initial']=test_df['Initial'].replace(['Mlle', 'Mme','Ms'], 'Miss')



# Noble passengers

train_df['Initial']=train_df['Initial'].replace(['Sir','Don','Dona','Jonkheer','Lady','Countess'], 'Noble')

test_df['Initial']=test_df['Initial'].replace(['Sir','Don','Dona','Jonkheer','Lady','Countess'], 'Noble')



# passengers with a higher social standing

train_df['Initial']=train_df['Initial'].replace(['Dr', 'Rev','Col','Major','Capt'], 'Others')

test_df['Initial']=test_df['Initial'].replace(['Dr', 'Rev','Col','Major','Capt'], 'Others')
pd.crosstab(train_df.Initial,train_df.Sex).T.style.background_gradient(cmap='summer_r')
pd.crosstab(test_df.Initial,test_df.Sex).T.style.background_gradient(cmap='summer_r')
train_df.head()
#Embarked

train_df['Embarked'].fillna('S', inplace = True)



#Fare

train_df['Fare'].fillna(train_df['Fare'].median(), inplace = True)
#Embarked

test_df['Embarked'].fillna('S', inplace = True)



#Fare

test_df['Fare'].fillna(test_df['Fare'].median(), inplace = True)
def newage (cols):

    Initial=cols[0]

    Sex=cols[1]

    Age=cols[2]

    if pd.isnull(Age):

        if Initial=='Master' and Sex=="male":

            return 4.57

        elif Initial=='Miss' and Sex=='female':

            return 21.8

        elif Initial=='Mr' and Sex=='male': 

            return 32.37

        elif Initial=='Mrs' and Sex=='female':

            return 35.72

        elif Initial=='Officer' and Sex=='female':

            return 49

        elif Initial=='Officer' and Sex=='male':

            return 46.56

        elif Initial=='Royalty' and Sex=='female':

            return 40.50

        else:

            return 42.33

    else:

        return Age
#train_df.Age=train_df[['Initial','Sex','Age']].apply(newage, axis=1)

#test_df.Age=test_df[['Initial','Sex','Age']].apply(newage, axis=1)
train_df.groupby('Initial')['Age'].mean()
#Age

train_df.loc[(train_df.Age.isnull())&(train_df.Initial=='Mr'),'Age']=32

train_df.loc[(train_df.Age.isnull())&(train_df.Initial=='Mrs'),'Age']=36

train_df.loc[(train_df.Age.isnull())&(train_df.Initial=='Master'),'Age']=5

train_df.loc[(train_df.Age.isnull())&(train_df.Initial=='Miss'),'Age']=22

train_df.loc[(train_df.Age.isnull())&(train_df.Initial=='Noble'),'Age']=42

train_df.loc[(train_df.Age.isnull())&(train_df.Initial=='YoungMiss'),'Age']=12

train_df.loc[(train_df.Age.isnull())&(train_df.Initial=='Other'),'Age']=47
train_df[train_df['Age'].isnull()]
train_df['Age'].fillna(47, inplace = True)
test_df.groupby('Initial')['Age'].mean()
#Age

test_df.loc[(test_df.Age.isnull())&(test_df.Initial=='Mr'),'Age']=32

test_df.loc[(test_df.Age.isnull())&(test_df.Initial=='Mrs'),'Age']=39

test_df.loc[(test_df.Age.isnull())&(test_df.Initial=='Master'),'Age']=7

test_df.loc[(test_df.Age.isnull())&(test_df.Initial=='Miss'),'Age']=22

test_df.loc[(test_df.Age.isnull())&(test_df.Initial=='Noble'),'Age']=39

test_df.loc[(test_df.Age.isnull())&(test_df.Initial=='YoungMiss'),'Age']=12

test_df.loc[(test_df.Age.isnull())&(test_df.Initial=='Other'),'Age']=45
test_df[test_df['Age'].isnull()]
for dataset in [train_df, test_df]:

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

    dataset["SmallFamily"]= np.where((dataset["FamilySize"] > 1) & (dataset["FamilySize"] < 4), 1, 0)

    dataset["MediumFamily"]= np.where((dataset["FamilySize"] >= 4) & (dataset["FamilySize"] < 7), 1, 0)

    dataset["LargeFamily"]= np.where(dataset["FamilySize"] >= 7, 1, 0)

    dataset['IsAlone'] = 0

    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

    dataset['Age*Class'] = dataset.Age * dataset.Pclass

    dataset["IsMother"]= np.where((dataset["Sex"]=="female") & (dataset["Parch"] > 0) & (dataset["Initial"] != "Miss"), 1, 0)

    #dataset['IsChild'] = np.where(dataset["Age"] < 16, 1, 0)

    #dataset["Is_Married"]= np.where(dataset["Initial"] == 'Mrs', 1, 0)

    #dataset['Embarked'] = dataset['Embarked'].map(embarkedMap)

    #dataset['Sex'] = dataset['Sex'].map(genderMap)

    #dataset['Ticket_Frequency'] = dataset.groupby('Ticket')['Ticket'].transform('count')

    #dataset['Ticket2']=dataset.Ticket.apply(lambda x : len(x))

    #dataset['Cabin2']=dataset.Cabin.apply(lambda x : len(x))

    #dataset['Name2']=dataset.Name.apply(lambda x: x.split(',')[0].strip())
# Store our passenger ID for the submission

PassengerId = test_df['PassengerId']

train_df = train_df.drop(columns=['PassengerId',  'Ticket', 'Name', 'Cabin', 'SibSp','Parch', 'FamilySize'])

test_df = test_df.drop(columns=['PassengerId',  'Ticket', 'Name', 'Cabin', 'SibSp','Parch','FamilySize'])
train_df.isnull().sum()
train_df.head()
sns.countplot(train_df['Survived'])
train_df.columns
#train_df = train_df.drop(columns=['SibSp','Parch','Ticket','Cabin'])

#test_df = test_df.drop(columns=['SibSp','Parch','Ticket','Cabin'])
sns.heatmap(train_df.corr(),annot=True,cmap='RdYlGn',linewidths=0.2)

fig=plt.gcf()

fig.set_size_inches(10,8)

plt.show()
train_df2 = train_df.copy().drop(columns=['Survived'])

train_df2.corrwith(train_df.Survived).plot.bar(figsize=(20,15), 

                                              title="Correlation with the response Variable", 

                                              fontsize=15, rot=90, grid=True)
train_df.columns
summary(train_df)
from sklearn.preprocessing import MinMaxScaler

col_names = train_df.columns.drop(['Sex', 'Embarked', "Initial", "Survived"])

features = train_df[col_names]

scaler = MinMaxScaler(feature_range = (-1,1)).fit(features.values)

features = scaler.transform(features.values)

train_df[col_names] = features



col_names = test_df.columns.drop(['Sex', 'Embarked', "Initial"])

features = test_df[col_names]

scaler = MinMaxScaler(feature_range = (-1,1)).fit(features.values)

features = scaler.transform(features.values)

test_df[col_names] = features
from sklearn.model_selection import train_test_split

X_train = train_df.drop('Survived', 1)

y_train = train_df['Survived']
from sklearn.utils import class_weight

cw = list(class_weight.compute_class_weight('balanced',

                                             np.unique(train_df['Survived']),

                                             train_df['Survived']))
cw
cat_features = ['Sex', 'Embarked', "Initial"]



params = {'depth':[1, 2, 3],

          'iterations':[3000],

          'early_stopping_rounds': [3000],

          'learning_rate':[0.01],

          'loss_function': ['Logloss'],

          'l2_leaf_reg':np.logspace(-20,-19, 3),

          'eval_metric':['Recall']

}



param = catboost_GridSearchCV(X_train, y_train, params, cat_features, cw)
# cross validate the best model

cross_val(X_train, y_train, param, cat_features, cw, 5)
# build the final model with the best parameters

clf = CatBoostClassifier(iterations=param['iterations'],

                        loss_function = param['loss_function'],

                        depth=param['depth'],

                        l2_leaf_reg = param['l2_leaf_reg'],

                        eval_metric = param['eval_metric'],

                        leaf_estimation_iterations = 10,

                        class_weights = cw,

                        use_best_model=True

)



X_train, X_test, y_train, y_test = train_test_split(X_train,

                                                        y_train, 

                                                        shuffle=True,

                                                        random_state=RANDOM_STATE,

                                                        test_size=0.2,

                                                        stratify=y_train

    )



clf.fit(X_train, 

        y_train,

        cat_features=cat_features,

        logging_level='Silent',

        eval_set=(X_test, y_test)

)
pred_y = clf.predict(X_test)
plot_cf_matrix_and_roc(clf, X_train, y_train, X_test, y_test, pred_y , classes=['No Survived','Survived'])
submission_predictions = clf.predict(test_df)
submission_predictions.shape
PassengerId.shape
submission = pd.DataFrame({

        "PassengerId": PassengerId,

        "Survived": submission_predictions

    })



submission.to_csv("submission.csv", index=False)

print(submission.shape)