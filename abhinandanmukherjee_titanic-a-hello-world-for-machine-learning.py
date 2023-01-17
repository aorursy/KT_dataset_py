# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Import related libraries



import matplotlib.pyplot as plt

import seaborn as sns

import os

import gc

import psutil



from sklearn.pipeline import Pipeline

from sklearn.preprocessing import LabelEncoder,StandardScaler

from sklearn.decomposition import PCA, KernelPCA



from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.metrics import (confusion_matrix,accuracy_score,precision_score,recall_score,f1_score,

                             make_scorer,classification_report,roc_auc_score,roc_curve,

                             average_precision_score,precision_recall_curve)



from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,VotingClassifier



from imblearn.pipeline import make_pipeline

from imblearn.under_sampling import OneSidedSelection

from imblearn.under_sampling import NearMiss

from imblearn.under_sampling import EditedNearestNeighbours

from imblearn.under_sampling import TomekLinks

from imblearn.under_sampling import RandomUnderSampler



pd.set_option('display.max_columns', None)



import warnings

warnings.filterwarnings("ignore")



RANDOM_SEED = 101



import collections

from mpl_toolkits import mplot3d
sub_file = pd.read_csv("/kaggle/input/titanic/gender_submission.csv")

sub_file.head()
train = pd.read_csv("/kaggle/input/titanic/train.csv")

train.head()
val = pd.read_csv("/kaggle/input/titanic/test.csv")

val.head()
train.columns
val.columns
train['Survived'].value_counts().plot.bar()
train.info()
train.isnull().mean()
train.shape
train.describe()
target = 'Survived'
"Braund, Mr. Owen Harris".split(',')[1].split()[0][:-1]
train["Name"].map(lambda x: x.split(',')[1].split()[0][:-1]).value_counts().plot.bar()
val["Name"].map(lambda x: x.split(',')[1].split()[0][:-1]).value_counts().plot.bar()
def get_salutation_map(df,var,rare):

    sal_dict = {}

    for sal, count in df[var].value_counts().to_dict().items():

        count = int(count)

        if count < 10:

            sal_dict[sal] = rare

        else:

            sal_dict[sal] = sal

    return sal_dict
train["Salutation"] = train["Name"].map(lambda x: x.split(',')[1].split()[0][:-1])
train.head()
get_salutation_map(train,"Salutation","Rare")
train["Salutation"] = train["Name"].map(lambda x: x.split(',')[1].split()[0][:-1])

train["Salutation"] = train["Salutation"].map(get_salutation_map(train,'Salutation','Rare'))

train.head(2)
train["Salutation"].value_counts().plot.bar()
sns.countplot(x="Salutation",data=train)
train.head()
sns.boxplot(y = 'Age',

            x = 'Salutation', 

            data = train)

plt.xlabel('Saluation')

plt.ylabel('Age')

plt.title('Distribution of Age with respect to Saluations', fontsize = 10)
sns.boxplot(y = 'Fare',

            x = 'Salutation', 

            data = train)

plt.xlabel('Saluation')

plt.ylabel('Fare')

plt.title('Distribution of Fare with respect to Saluations', fontsize = 10)
train['SibSp'].unique()
train['SibSp'].nunique()
train['SibSp'].value_counts().plot.bar()
train['Parch'].unique()
train['Parch'].value_counts().plot.bar()
train["Family_Size"] = train["SibSp"] + train["Parch"]

train["Family_Size"].unique()
(train["Family_Size"].value_counts(normalize=True)*100).plot.bar()
train["Family_Size"].value_counts(normalize=True)*100
def get_family_size_map(df,var):

    fam_dict = {}

    for size, pct in (df[var].value_counts(normalize=True)*100).to_dict().items():

        if size == 0:

            fam_dict[size] = "Alone"

        elif (size != 0) & (pct > 10.0):

            fam_dict[size] = "Small"

        else:

            fam_dict[size] = "Large"

    return fam_dict
train["Family_Size"] = train["Family_Size"].map(get_family_size_map(train,'Family_Size'))

train.head(2)
sns.boxplot(y = 'Age',

            x = 'Family_Size', 

            data = train)

plt.xlabel('Family_Size')

plt.ylabel('Age')

plt.title('Distribution of Age with respect to Family_Size', fontsize = 10)
sns.boxplot(y = 'Age',

            x = 'Family_Size', 

            hue = 'Pclass',

            data = train)

plt.xlabel('Family_Size')

plt.ylabel('Age')

plt.title('Distribution of Age with respect to Family_Size', fontsize = 10)
sns.boxplot(y = 'Fare',

            x = 'Family_Size', 

            data = train)

plt.xlabel('Family_Size')

plt.ylabel('Fare')

plt.title('Distribution of Fare with respect to Family_Size', fontsize = 10)
train.isnull().mean()
train['had_Cabin'] = np.where(train['Cabin'].isna(),0,1)
train.head()
train['Cabin'].dropna().map(lambda x:x[0]).value_counts()
train['Cabin'] = train['Cabin'].fillna("M")

train['Cabin'] = train['Cabin'].map(lambda x: x[0])
train.head()
sns.boxplot(y = 'Age',

            x = 'Cabin',

            data = train)

plt.xlabel('Cabin')

plt.ylabel('Age')

plt.title('Distribution of Age with respect to Cabin', fontsize = 10)
sns.boxplot(y = 'Cabin',

            x = 'Fare',

            data = train)

plt.xlabel('had_Cabin')

plt.ylabel('Fare')

plt.title('Distribution of Fare with respect to had_Cabin', fontsize = 10)
train.head()
train.groupby(['Salutation','had_Cabin'])
mean_dict = {}

for k, df in train.groupby(['Salutation','Family_Size','had_Cabin']):

    if df['Age'].isnull().sum() != 0:

        mean_dict[k] = df["Age"].mean()

mean_dict
for k,v in mean_dict.items():

    train.loc[(train["Salutation"] == k[0]) & (train["Family_Size"] == k[1]) & (train["had_Cabin"] == k[2]) & (train["Age"].isna()), "Age"] = v
train['Embarked'].value_counts()
train['Embarked'] = train['Embarked'].fillna(train['Embarked'].mode().values[0])
train.isnull().sum()
train.head()
num_cols = ['Age','Fare']

cat_cols = ['Pclass','Sex','Embarked','Cabin','had_Cabin','Salutation','Family_Size']
for col in num_cols:

    fig = plt.figure(figsize = (10,5))

    ax = fig.add_subplot(111)

    ax = sns.distplot(train[col], color="m", label="Skewness : %.2f"%(train[col].skew()))

    ax.set_xlabel(col)

    ax.set_ylabel("Frequency")

    ax.legend(loc='best')

    ax.set_title('Frequency Distribution of {}'.format(col), fontsize = 15)
fig = plt.figure(figsize = (50,15))

j = 1

for cat_col in cat_cols:

    ax = fig.add_subplot(1,len(cat_cols),j)

    sns.countplot(x = cat_col,

                  data = train,

                  ax = ax)

    ax.set_xlabel(cat_col)

    ax.set_ylabel("Frequency")

    ax.set_title('Frequency Distribution for individual classes in {}'.format(cat_col), fontsize = 10)

    j = j + 1
sns.pairplot(train[num_cols])
for col in num_cols:

    fig = plt.figure(figsize = (15,4))

    ax = fig.add_subplot(111)

    j = 0

    for key, df in train.groupby([target]):

        ax = sns.kdeplot(df[col], shade = True, label=key)

        ax.set_xlabel(col)

        ax.set_ylabel("Frequency")

        ax.legend(loc="best")

        fig.suptitle('Frequency Distribution of {}'.format(col), fontsize = 10)

        j = j + 1
for col in num_cols:

    fig = plt.figure(figsize = (15,4))

    j = 1

    for key, df in train.groupby([target]):

        ax = fig.add_subplot(1,train[target].nunique(),j)

        ax = sns.distplot(df[col], label="Skewness : %.2f"%(df[col].skew()))

        ax.set_xlabel(key)

        ax.set_ylabel("Frequency")

        ax.legend(loc="best")

        fig.suptitle('Frequency Distribution of {}'.format(col), fontsize = 10)

        j = j + 1
for num_col in num_cols:

    fig = plt.figure(figsize = (30,10))

    j = 1

    for cat_col in cat_cols:

        ax = fig.add_subplot(1,len(cat_cols),j)

        sns.boxplot(y = train[num_col],

                    x = train[cat_col], 

                    data = train, 

                    ax = ax)

        ax.set_xlabel(cat_col)

        ax.set_ylabel(num_col)

        ax.set_title('Distribution of {} with respect to {}'.format(num_col,cat_col), fontsize = 10)

        j = j + 1
for num_col in num_cols:

    fig = plt.figure(figsize = (30,10))

    j = 1

    for cat_col in cat_cols:

        ax = fig.add_subplot(1,len(cat_cols),j)

        sns.boxplot(y = train[num_col],

                    x = train[cat_col],

                    hue = target,

                    data = train, 

                    ax = ax)

        ax.set_xlabel(cat_col)

        ax.set_ylabel(num_col)

        ax.set_title('Distribution of {} with respect to {}'.format(num_col,cat_col), fontsize = 10)

        j = j + 1
train_data = pd.get_dummies(train,columns=cat_cols,drop_first=True)

train_data.head(2)
explore_data, validation_data = train_test_split(train_data, test_size = 0.2, random_state=RANDOM_SEED, stratify=train[target])
train_data, test_data = train_test_split(explore_data, test_size = 0.2, random_state=RANDOM_SEED)
def handle_outliers_per_target_class(df,var,target,tol):

    gdf = df[df[target] == 1]

    var_data = gdf[var].values

    q25, q75 = np.percentile(var_data, 25), np.percentile(var_data, 75)

    

    print('Outliers handling for {}'.format(var))

    print('Quartile 25: {} | Quartile 75: {}'.format(q25, q75))

    

    iqr = q75 - q25

    print('IQR {}'.format(iqr))

    

    cut_off = iqr * tol

    lower, upper = q25 - cut_off, q75 + cut_off

    

    print('Cut Off: {}'.format(cut_off))

    print('{} Lower: {}'.format(var,lower))

    print('{} Upper: {}'.format(var,upper))

    

    outliers = [x for x in var_data if x < lower or x > upper]



    print('Number of Outliers in feature {} in {}: {}'.format(var,key,len(outliers)))



    print('{} outliers:{}'.format(var,outliers))



    print('----' * 25)

    print('\n')

    print('\n')

        

    return list(df[(df[var] > upper) | (df[var] < lower)].index)
outliers_wrt_target = []

for num_col in num_cols:

    outliers_wrt_target.extend(handle_outliers_per_target_class(train_data,num_col,target,1.5))

outliers_wrt_target = list(set(outliers_wrt_target))



train_data = train_data.drop(outliers_wrt_target)
train_data["Fare"] = np.where(train_data["Fare"] != 0,np.log(train_data["Fare"]),np.log(0.00001))

test_data["Fare"] = np.where(test_data["Fare"] != 0,np.log(test_data["Fare"]),np.log(0.00001))

validation_data["Fare"] = np.where(validation_data["Fare"] != 0,np.log(validation_data["Fare"]),np.log(0.00001))
X_train = train_data.drop(['PassengerId', 'Survived', 'Name', 'SibSp', 'Parch', 'Ticket'],axis=1)

y_train = train_data[target]
X_test = test_data.drop(['PassengerId', 'Survived', 'Name', 'SibSp', 'Parch', 'Ticket'],axis=1)

y_test = test_data[target]
X_val = validation_data.drop(['PassengerId', 'Survived', 'Name', 'SibSp', 'Parch', 'Ticket'],axis=1)

y_val = validation_data[target]
y_enc = LabelEncoder()

y_train = y_enc.fit_transform(y_train)

y_test = y_enc.transform(y_test)

y_val = y_enc.transform(y_val)
X_train.head()
sc = StandardScaler()

X_train[num_cols] = sc.fit_transform(X_train[num_cols])

X_test[num_cols] = sc.transform(X_test[num_cols])

X_val[num_cols] = sc.transform(X_val[num_cols])
sc.mean_
sc.var_
X_train.shape
clf = LogisticRegression()
clf.fit(X_train,y_train)
clf.intercept_
y_pred = clf.predict(X_test)
y_pred
confusion_matrix(y_test,y_pred)
accuracy_score(y_test,y_pred)
classification_models = ['LogisticRegression',

                         'SVC',

                         'DecisionTreeClassifier',

                         'RandomForestClassifier',

                         'AdaBoostClassifier']
cm = []

acc = []

prec = []

rec = []

f1 = []

models = []

estimators = []
for classfication_model in classification_models:

    

    model = eval(classfication_model)()

    

    model.fit(X_train,y_train)

    y_pred = model.predict(X_test)

    

    models.append(type(model).__name__)

    estimators.append((type(model).__name__,model))

    cm.append(confusion_matrix(y_test,y_pred))

    acc.append(accuracy_score(y_test,y_pred))

    prec.append(precision_score(y_test,y_pred))

    rec.append(recall_score(y_test,y_pred))

    f1.append(f1_score(y_test,y_pred))
vc = VotingClassifier(estimators)

vc.fit(X_train,y_train)
y_pred = vc.predict(X_test)

    

models.append(type(vc).__name__)



cm.append(confusion_matrix(y_test,y_pred))

acc.append(accuracy_score(y_test,y_pred))

prec.append(precision_score(y_test,y_pred))

rec.append(recall_score(y_test,y_pred))

f1.append(f1_score(y_test,y_pred))
model_dict = {"Models":models,

             "CM":cm,

             "Accuracy":acc,

             "Precision":prec,

             "Recall":rec,

             "f1_score":f1}
model_df = pd.DataFrame(model_dict)

model_df
model_df.sort_values(by=['Accuracy','f1_score','Recall','Precision'],ascending=False,inplace=True)

model_df
model_param_grid = {}
model_param_grid['LogisticRegression'] = {'penalty' : ['l1', 'l2'],

                                          'C' : np.logspace(0, 4, 10)}
model_param_grid['SVC'] = [{'kernel': ['rbf'], 

                            'gamma': [1e-2, 1e-3, 1e-4, 1e-5],

                            'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]},

                           {'kernel': ['sigmoid'],

                            'gamma': [1e-2, 1e-3, 1e-4, 1e-5],

                            'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]},

                           {'kernel': ['linear'], 

                            'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]},

                           {'kernel': ['poly'], 

                            'degree' : [0, 1, 2, 3, 4, 5, 6]}

                          ]
model_param_grid['DecisionTreeClassifier'] = {'criterion' : ["gini","entropy"],

                                              'max_features': ['auto', 'sqrt', 'log2'],

                                              'min_samples_split': [10,11,12,13,14,15],

                                              'min_samples_leaf':[1,2,3,4,5,6,7]}
model_param_grid['RandomForestClassifier'] = {'n_estimators' : [50,100,150,200],

                                              'criterion' : ["gini","entropy"],

                                              'max_features': ['auto', 'sqrt', 'log2'],

                                              'class_weight' : ["balanced", "balanced_subsample"]}
model_param_grid['AdaBoostClassifier'] = {'n_estimators' : [25,50,75,100],

                                          'learning_rate' : [0.001,0.01,0.05,0.1,1,10],

                                          'algorithm' : ['SAMME', 'SAMME.R']}
from sklearn.model_selection import GridSearchCV

def tune_parameters(model_name,model,params,cv,scorer,X,y):

    best_model = GridSearchCV(estimator = model,

                              param_grid = params,

                              scoring = scorer,

                              cv = cv,

                              n_jobs = -1).fit(X, y)

    print("Tuning Results for ", model_name)

    print("Best Score Achieved: ",best_model.best_score_)

    print("Best Parameters Used: ",best_model.best_params_)

    return best_model.best_estimator_
from sklearn.metrics import make_scorer



# Define scorer

def roc_metric(y_test, y_pred):

    score = roc_auc_score(y_test, y_pred)

    return score
# Scorer function would try to maximize calculated metric

roc_scorer = make_scorer(roc_metric,greater_is_better=True)
best_estimators = []
for m_name, m_obj in estimators:

    best_estimators.append((m_name,tune_parameters(m_name,

                                                   m_obj,

                                                   model_param_grid[m_name],

                                                   10,

                                                   roc_scorer,

                                                   X_train,

                                                   y_train)))
best_estimators
tuned_vc = VotingClassifier(best_estimators)

tuned_vc.fit(X_train,y_train)
y_pred = tuned_vc.predict(X_test)
confusion_matrix(y_test,y_pred)
accuracy_score(y_test,y_pred)
precision_score(y_test,y_pred)
recall_score(y_test,y_pred)
f1_score(y_test,y_pred)
# Importing the Keras libraries and packages

import keras

from keras.utils import plot_model

from keras.models import Model,Sequential,load_model

from keras.layers import Input, Flatten, Dense, Dropout

from keras.layers.merge import concatenate

from keras import backend as K

from keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau
def nn_model(X,y,optimizer,kernels):

    input_shape = X.shape[1]

       

    if(len(np.unique(y)) == 2):

        op_neurons = 1

        op_activation = 'sigmoid'

        loss = 'binary_crossentropy'

    else:

        op_neurons = len(np.unique(y))

        op_activation = 'softmax'

        loss = 'categorical_crossentropy'

    

    classifier = Sequential()

    classifier.add(Dense(units = input_shape,

                         kernel_initializer = kernels,

                         activation = 'relu',

                         input_dim = input_shape))

    classifier.add(Dense(units = 8,

                         kernel_initializer = kernels,

                         activation = 'relu'))

    classifier.add(Dense(units = 4,

                         kernel_initializer = kernels,

                         activation = 'relu'))

    classifier.add(Dropout(rate = 0.25))

    classifier.add(Dense(units = op_neurons,

                         kernel_initializer = kernels,

                         activation = op_activation))

    

    classifier.compile(optimizer = optimizer,

                       loss = loss,

                       metrics = ['accuracy'])

    

    classifier.summary()

    return classifier
model = nn_model(X_train,y_train,'adam','he_uniform')

history = model.fit(X_train,

                    y_train,

                    batch_size = 64,

                    epochs = 1000,

                    validation_data=(X_test, y_test))
his_df = pd.DataFrame(history.history)

his_df.shape
plt.plot(his_df['loss'])

plt.plot(his_df['val_loss'])

plt.title("Loss Plot")

plt.legend(["train","test"])
plt.plot(his_df['accuracy'])

plt.plot(his_df['val_accuracy'])

plt.title("Accuracy Plot")

plt.legend(["train","test"])