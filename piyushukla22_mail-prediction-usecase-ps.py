import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.metrics import (confusion_matrix,accuracy_score,precision_score,recall_score,f1_score,make_scorer,classification_report,roc_auc_score,roc_curve,average_precision_score,precision_recall_curve)



from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import SGDClassifier

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,VotingClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier



RANDOM_SEED = 101

InputData = pd.read_csv("../input/task_InputData.csv")

InputData.head()
InputData.shape
InputData.info()
InputData.describe()
InputData.isnull().mean()
InputData['sports'].unique()
InputData['sports'] = InputData['sports'].fillna('others')

InputData['sports'].value_counts()
InputData['zip code'].nunique()
discardFeatures = ['name','zip code']

for col in discardFeatures:

    if col in InputData.columns:

        InputData = InputData.drop(columns=col, axis=1)

InputData.head(2)
target = 'label'

num_cols = ['age', 'earnings']

cat_cols = ['lifestyle', 'family status', 'car', 'sports', 'Living area']
InputData[target].value_counts().plot.bar()
for col in num_cols:

    fig = plt.figure(figsize = (10,5))

    ax = fig.add_subplot(111)

    ax = sns.distplot(InputData[col], color="m", label="Skewness : %.2f"%(InputData[col].skew()))

    ax.set_xlabel(col)

    ax.set_ylabel("Frequency")

    ax.legend(loc='best')

    ax.set_title('Frequency Distribution of {}'.format(col), fontsize = 15)
for col in num_cols:

    fig = plt.figure(figsize = (15,4))

    ax = fig.add_subplot(111)

    j = 0

    for key, df in InputData.groupby([target]):

        ax = sns.kdeplot(df[col], shade = True, label=key)

        ax.set_xlabel(col)

        ax.set_ylabel("Frequency")

        ax.legend(loc="best")

        fig.suptitle('Frequency Distribution of {}'.format(col), fontsize = 10)

        j = j + 1
fig = plt.figure(figsize = (50,15))

j = 1

for cat_col in cat_cols:

    ax = fig.add_subplot(1,len(cat_cols),j)

    sns.countplot(x = cat_col,

                  data = InputData,

                  ax = ax)

    ax.set_xlabel(cat_col)

    ax.set_ylabel("Frequency")

    ax.set_title('Frequency Distribution for individual classes in {}'.format(cat_col), fontsize = 10)

    j = j + 1
sns.boxplot(x = 'Living area',

           y = 'age',

           data = InputData)
sns.boxplot(y='age', 

            x='lifestyle',

           data = InputData)

plt.xlabel('Lifestyle')

plt.ylabel('Age')

plt.title('Distribution of Age with respect to Lifestyle', fontsize = 10)
sns.pairplot(InputData[num_cols])
for col in num_cols:

    fig = plt.figure(figsize = (15,4))

    j = 1

    for key, df in InputData.groupby([target]):

        ax = fig.add_subplot(1,InputData[target].nunique(),j)

        ax = sns.distplot(df[col], label="Skewness : %.2f"%(df[col].skew()))

        ax.set_xlabel(key)

        ax.set_ylabel("Frequency")

        ax.legend(loc="best")

        fig.suptitle('Frequency Distribution of {}'.format(col), fontsize = 10)

        j = j + 1
sns.boxplot(x=target, 

           y='earnings',

           data = InputData)
pd.crosstab(InputData['lifestyle'],InputData['label']).plot(kind='bar')
pd.crosstab(InputData['family status'],InputData['label']).plot(kind='bar')
j = 1

for cat_col in cat_cols:

    ax = fig.add_subplot(1,len(cat_cols),j)

    pd.crosstab(InputData[cat_col],InputData['label']).plot(kind='bar')

    j = j + 1
for num_col in num_cols:

    fig = plt.figure(figsize = (30,10))

    j = 1

    for cat_col in cat_cols:

        ax = fig.add_subplot(1,len(cat_cols),j)

        sns.boxplot(y = InputData[num_col],

                    x = InputData[cat_col],

                    hue = target,

                    data = InputData, 

                    ax = ax)

        ax.set_xlabel(cat_col)

        ax.set_ylabel(num_col)

        ax.set_title('Distribution of {} with respect to {}'.format(num_col,cat_col), fontsize = 10)

        j = j + 1
sns.boxplot(x='label',

           y='age',

           hue='Living area',

           data=InputData)
sns.boxplot(x='Living area',

           y='age',

           hue='label',

           data=InputData)
for col in num_cols:

    j=1

    fig = plt.figure(figsize=(40,10))

    ax = fig.add_subplot(1,len(num_cols),j)

    sns.boxplot(InputData['label'],InputData[col])
train_data = pd.get_dummies(InputData,columns=cat_cols,drop_first=True)

train_data.head(2)
explore_data, validation_data = train_test_split(train_data, test_size = 0.2, random_state=RANDOM_SEED)#, stratify=InputData[target])
explore_data.shape
validation_data.shape
train_data, test_data = train_test_split(explore_data, test_size = 0.2, random_state=RANDOM_SEED)
train_data.head()

train_data.shape
test_data.head()

test_data.label

test_data.shape
def handle_outliers_per_target_class(df,var,target,tol):

    var_data = df[var].values

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
train_data.columns
X_train = train_data.drop(['label'],axis=1)

y_train = train_data[target]
X_train.head()
X_test = test_data.drop(['label'],axis=1)

y_test = test_data[target]
X_test.head()
y_test.head()
X_val = validation_data.drop(['label'],axis=1)

y_val = validation_data[target]
y_val
X_train.head()
X_train = pd.get_dummies(X_train, prefix_sep='_', drop_first=True)

# X head

X_train.head()
X_train.columns
X_test = pd.get_dummies(X_test, prefix_sep='_', drop_first=True)

# X head

X_test.head()
X_test.columns
X_val = pd.get_dummies(X_val, prefix_sep='_', drop_first=True)

# X head

X_val.head()
lr = LogisticRegression()

lr
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)

y_pred
confusion_matrix(y_test,y_pred)
accuracy_score(y_test,y_pred)
classification_models = ['GaussianNB',

                         'SGDClassifier',

                         'KNeighborsClassifier',

                         'SVC',

                         'DecisionTreeClassifier',

                         'RandomForestClassifier',

                         'AdaBoostClassifier']
for classfication_model in classification_models:

    

    model = eval(classfication_model)()

    

    model.fit(X_train,y_train)

    y_pred = model.predict(X_test)

    

    print("{} accuracy score: {}".format(classfication_model,accuracy_score(y_test,y_pred)))

    #print("Confusion matrix: \n",confusion_matrix(y_test,y_pred))