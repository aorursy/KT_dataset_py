import numpy as np

import pandas as pd

from sklearn.preprocessing import LabelEncoder

from imblearn.pipeline import Pipeline

from sklearn.ensemble.weight_boosting import AdaBoostClassifier

from imblearn.over_sampling import RandomOverSampler



train = pd.read_csv("data/train.csv", sep=",", na_values='?')

test = pd.read_csv("data/test.csv", sep=",", na_values='?')



for col in test.columns:

    if(train[col].isnull().sum()>50000):

        train.drop(col,axis=1,inplace=True)

        test.drop(col,axis=1,inplace=True)

        

for col in ['Hispanic', 'COB SELF', 'COB MOTHER', 'COB FATHER']:

    train[col].fillna(value = train[col].mode()[0], inplace = True)

    test[col].fillna(value = test[col].mode()[0], inplace = True)

    

y = train['Class']

X = train.drop(['Class','ID'],axis=1)



test_ids = test['ID']

test.drop('ID', axis = 1, inplace = True)



cols = X.columns

num_cols = X._get_numeric_data().columns

cat_cols = list(set(cols) - set(num_cols))



le = LabelEncoder()

for ft in cat_cols:

    X[ft] = le.fit_transform(X[ft])

    test[ft] = le.fit_transform(test[ft])



col_rem = ['Vet_Benefits',

 'COB SELF',

 'Hispanic',

 'Citizen',

 'Cast',

 'Timely Income',

 'COB MOTHER',

 'COB FATHER',

 'WorkingPeriod',

 'Married_Life',

 'Summary',

 'Own/Self',

 'Tax Status',

 'Detailed']

    

X.drop(col_rem,axis=1,inplace=True)

test.drop(col_rem,axis=1,inplace=True)



model = Pipeline([

        ('sampling', RandomOverSampler()),

        ('classification', AdaBoostClassifier())

    ])



model.fit(X,y)

y_pred_tst = model.predict(test)

y_pred_tst = list(y_pred_tst)

test['Class'] = y_pred_tst

pd.concat([test_ids,test['Class']],axis = 1).to_csv(r'Submissions/'+'sub.csv',index = False)
train = pd.read_csv("data/train.csv", sep=",", na_values='?')

test = pd.read_csv("data/test.csv", sep=",", na_values='?')



for col in test.columns:

    if(train[col].isnull().sum()>50000):

        train.drop(col,axis=1,inplace=True)

        test.drop(col,axis=1,inplace=True)

        

for col in ['Hispanic', 'COB SELF', 'COB MOTHER', 'COB FATHER']:

    train[col].fillna(value = train[col].mode()[0], inplace = True)

    test[col].fillna(value = test[col].mode()[0], inplace = True)

    

y = train['Class']

X = train.drop(['Class','ID'],axis=1)



test_ids = test['ID']

test.drop('ID', axis = 1, inplace = True)



cols = X.columns

num_cols = X._get_numeric_data().columns

cat_cols = list(set(cols) - set(num_cols))



le = LabelEncoder()

for ft in cat_cols:

    X[ft] = le.fit_transform(X[ft])

    test[ft] = le.fit_transform(test[ft])



col_rem = ['Vet_Benefits',

 'COB SELF',

 'Hispanic',

 'Citizen',

 'Cast',

 'Timely Income',

 'COB MOTHER',

 'COB FATHER',

 'WorkingPeriod',

 'Married_Life',

 'Summary',

 'Own/Self',

 'Tax Status',

 'Detailed']

    

X.drop(col_rem,axis=1,inplace=True)

test.drop(col_rem,axis=1,inplace=True)



model = Pipeline([

        ('sampling', RandomOverSampler()),

        ('classification', AdaBoostClassifier())

    ])



model.fit(X,y)

y_pred_tst = model.predict(test)

y_pred_tst = list(y_pred_tst)

test['Class'] = y_pred_tst

pd.concat([test_ids,test['Class']],axis = 1).to_csv(r'Submissions/'+'sub.csv',index = False)
output_class.loc[17:17]['Columns Removed']
for ft in cat_cols:

    X[ft] = le.fit_transform(X[ft])
X.drop(['Vet_Benefits', 'COB SELF', 'Hispanic', 'Citizen'],axis=1,inplace=True)
X,X_tst = scalars(X,test,sca='std')
test.drop(['Class'],inplace=True, axis = 1)

mod = AdaBoostClassifier()

model = Pipeline([

        ('sampling', RandomOverSampler()),

        ('classification', mod)

    ])



model.fit(X_train,y_train)

y_pred_tst = model.predict(test)

y_pred_tst = list(y_pred_tst)

test['Class'] = y_pred_tst

pd.concat([test_ids,test['Class']],axis = 1).to_csv(r'Submissions/'+'sub05.csv',index = False)
names = [

#     'RandomForestClassifier',

#     'ExtraTreeClassifier',

#     'BaggingClassifier',

    'GradientBoostingClassifier',

    'AdaBoostClassifier A',

    'AdaBoostClassifier B',

#     'GaussianProcessClassifier',

#     'KNeighborsClassifier',

#     'MLPClassifier',

    'DecisionTreeClassifier A',

    'DecisionTreeClassifier B',

#     'ExtraTreeClassifier',

#     'SVC_RBF',

#     'LinearSVC',

#     'NuSVC',

#     'GaussianNB',

#     'QuadraticDiscriminantAnalysis'

]



classifiers = [

#     RandomForestClassifier(),

#     ExtraTreeClassifier(),

#     BaggingClassifier(),

    GradientBoostingClassifier(),

    AdaBoostClassifier(base_estimator=DecisionTreeClassifier(criterion='entropy',min_samples_split=4)),

    AdaBoostClassifier(base_estimator=DecisionTreeClassifier()),

#     GaussianProcessClassifier(),

#     KNeighborsClassifier(),

#     MLPClassifier(),

    DecisionTreeClassifier(criterion='entropy',min_samples_split=4),

    DecisionTreeClassifier(),

#     ExtraTreeClassifier(),

#     SVC(kernel = 'rbf'),

#     LinearSVC(),

# #     NuSVC(),

#     GaussianNB(),

#     QuadraticDiscriminantAnalysis()

]

displ = []

i = 0

#X = df_norm.drop(['Outcome'], inplace=False, axis=1)

listmodels =[]

for mod in classifiers:

    model = Pipeline([

        ('sampling', RandomOverSampler()),

        ('classification', mod)

    ])

    print(names[i])

    model.fit(X_train,y_train)

    listmodels.append(model)

    em = []

    em.append(names[i])

    y_pred = model.predict(X_cv)

    y_pred=list(y_pred)

    acc,rec,pre,f1,roc = performance_metrics(y_cv,y_pred)

    em.append(acc)

    em.append(rec)

    em.append(pre)

    em.append(f1)

    em.append(roc)

    displ.append(em)

    i = i + 1

    

output_class = pd.DataFrame(displ,columns=['Classifier Name','Accuracy','Recall','Precision','F1_Score','Area Under ROC'])

output_class.sort_values('Area Under ROC')
test_ids = test['ID']

test.drop('ID', axis = 1, inplace = True)
#DATE: 12-04-2019, Submission 5

test.drop(['Class'],inplace=True, axis = 1)

mod = AdaBoostClassifier()

model = Pipeline([

        ('sampling', RandomOverSampler()),

        ('classification', mod)

    ])



model.fit(X_train,y_train)

y_pred_tst = model.predict(test)

y_pred_tst = list(y_pred_tst)

test['Class'] = y_pred_tst

pd.concat([test_ids,test['Class']],axis = 1).to_csv(r'Submissions/'+'sub05.csv',index = False)
X_train.describe()
names = [

    'RandomForestClassifier',

    'ExtraTreeClassifier',

    'BaggingClassifier',

    'GradientBoostingClassifier',

    'AdaBoostClassifier',

#     'GaussianProcessClassifier',

    'KNeighborsClassifier',

    'MLPClassifier',

    'DecisionTreeClassifier',

    'ExtraTreeClassifier',

#     'SVC_RBF',

    'LinearSVC',

#     'NuSVC',

    'GaussianNB',

    'QuadraticDiscriminantAnalysis']



classifiers = [

    RandomForestClassifier(),

    ExtraTreeClassifier(),

    BaggingClassifier(),

    GradientBoostingClassifier(),

    AdaBoostClassifier(),

#     GaussianProcessClassifier(),

    KNeighborsClassifier(),

    MLPClassifier(),

    DecisionTreeClassifier(),

    ExtraTreeClassifier(),

#     SVC(kernel = 'rbf'),

    LinearSVC(),

#     NuSVC(),

    GaussianNB(),

    QuadraticDiscriminantAnalysis()

]

displ = []

i = 0

#X = df_norm.drop(['Outcome'], inplace=False, axis=1)

listmodels =[]

for mod in classifiers:

    model = Pipeline([

        ('sampling', RandomOverSampler()),

        ('classification', mod)

    ])

    print(names[i])

    model.fit(X_train_std,y_train)

    listmodels.append(model)

    em = []

    em.append(names[i])

    y_pred = model.predict(X_cv_std)

    y_pred=list(y_pred)

    acc,rec,pre,f1,roc = performance_metrics(y_cv,y_pred)

    em.append(acc)

    em.append(rec)

    em.append(pre)

    em.append(f1)

    em.append(roc)

    displ.append(em)

    i = i + 1

    

output_class = pd.DataFrame(displ,columns=['Classifier Name','Accuracy','Recall','Precision','F1_Score','Area Under ROC'])

output_class.sort_values('Area Under ROC')
names = [

    'RandomForestClassifier',

    'ExtraTreeClassifier',

    'BaggingClassifier',

    'GradientBoostingClassifier',

    'AdaBoostClassifier',

#     'GaussianProcessClassifier',

    'KNeighborsClassifier',

    'MLPClassifier',

    'DecisionTreeClassifier',

    'ExtraTreeClassifier',

#     'SVC_RBF',

    'LinearSVC',

#     'NuSVC',

    'GaussianNB',

    'QuadraticDiscriminantAnalysis']



classifiers = [

    RandomForestClassifier(),

    ExtraTreeClassifier(),

    BaggingClassifier(),

    GradientBoostingClassifier(),

    AdaBoostClassifier(),

#     GaussianProcessClassifier(),

    KNeighborsClassifier(),

    MLPClassifier(),

    DecisionTreeClassifier(),

    ExtraTreeClassifier(),

#     SVC(kernel = 'rbf'),

    LinearSVC(),

#     NuSVC(),

    GaussianNB(),

    QuadraticDiscriminantAnalysis()

]

displ = []

i = 0

#X = df_norm.drop(['Outcome'], inplace=False, axis=1)

listmodels =[]

for mod in classifiers:

    model = Pipeline([

        ('sampling', RandomOverSampler()),

        ('classification', mod)

    ])

    print(names[i])

    model.fit(X_train_mimx,y_train)

    listmodels.append(model)

    em = []

    em.append(names[i])

    y_pred = model.predict(X_cv_mimx)

    y_pred=list(y_pred)

    acc,rec,pre,f1,roc = performance_metrics(y_cv,y_pred)

    em.append(acc)

    em.append(rec)

    em.append(pre)

    em.append(f1)

    em.append(roc)

    displ.append(em)

    i = i + 1

    

output_class = pd.DataFrame(displ,columns=['Classifier Name','Accuracy','Recall','Precision','F1_Score','Area Under ROC'])

output_class.sort_values('Area Under ROC')
names = [

    'RandomForestClassifier',

    'ExtraTreeClassifier',

    'BaggingClassifier',

    'GradientBoostingClassifier',

    'AdaBoostClassifier',

#     'GaussianProcessClassifier',

    'KNeighborsClassifier',

    'MLPClassifier',

    'DecisionTreeClassifier',

    'ExtraTreeClassifier',

#     'SVC_RBF',

    'LinearSVC',

#     'NuSVC',

    'GaussianNB',

    'QuadraticDiscriminantAnalysis']



classifiers = [

    RandomForestClassifier(),

    ExtraTreeClassifier(),

    BaggingClassifier(),

    GradientBoostingClassifier(),

    AdaBoostClassifier(),

#     GaussianProcessClassifier(),

    KNeighborsClassifier(),

    MLPClassifier(),

    DecisionTreeClassifier(),

    ExtraTreeClassifier(),

#     SVC(kernel = 'rbf'),

    LinearSVC(),

#     NuSVC(),

    GaussianNB(),

    QuadraticDiscriminantAnalysis()

]

displ = []

i = 0

#X = df_norm.drop(['Outcome'], inplace=False, axis=1)

listmodels =[]

for mod in classifiers:

    model = Pipeline([

        ('sampling', RandomOverSampler()),

        ('classification', mod)

    ])

    print(names[i])

    model.fit(X_train_rob,y_train)

    listmodels.append(model)

    em = []

    em.append(names[i])

    y_pred = model.predict(X_cv_rob)

    y_pred=list(y_pred)

    acc,rec,pre,f1,roc = performance_metrics(y_cv,y_pred)

    em.append(acc)

    em.append(rec)

    em.append(pre)

    em.append(f1)

    em.append(roc)

    displ.append(em)

    i = i + 1

    

output_class = pd.DataFrame(displ,columns=['Classifier Name','Accuracy','Recall','Precision','F1_Score','Area Under ROC'])

output_class.sort_values('Area Under ROC')