# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import math



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

data = pd.read_csv('/kaggle/input/unimelb/unimelb_training.csv')

# data = pd.read_csv('/kaggle/input/unimelb/unimelb_test.csv')



# print(data['Role.1'].unique())

# data.drop(data[

# (data['RFCDSum'] == 0) &

# (data['SEOSum'] == 0) &

# (data['Grant.Status'] == 1)].index)

percent_missing = data.isnull().sum() * 100 / len(data)

missing_value_df = pd.DataFrame({'column_name': data.columns,

                                 'percent_missing': percent_missing})

missing_value_df.sort_values('percent_missing', inplace=True)

print(missing_value_df)

sponsorColunmName = 'Sponsor.Code'



# pt = pd.crosstab([data[sponsorColunmName]], data['Grant.Status'])

# print(pt)

# pt.plot(kind='bar', stacked=True)



data[sponsorColunmName] = data[sponsorColunmName].fillna('0')

# print(data[sponsorColunmName].isna().sum())

# data['SponsorSuccessGrants'] = data[sponsorColunmName].map(data.where(lambda row: row['Grant.Status'] == 1).groupby(sponsorColunmName).size() / data.groupby(sponsorColunmName).size())

data['SponsorSuccessGrants'] = data[sponsorColunmName].map(data.where(lambda row:

    row['Grant.Status'] == 1

).groupby(sponsorColunmName).size())



# print(data[sponsorColunmName])

data['Grant.Category.Code'] = data['Grant.Category.Code'].fillna('0')

data['SuccessfullCategoryGrant'] = data['Grant.Category.Code'].map(data.where(lambda row: row['Grant.Status'] == 1).groupby('Grant.Category.Code').size() / data.groupby('Grant.Category.Code').size())

# data['SuccessfullCategoryGrant'] = data['Grant.Category.Code'].map(data.where(lambda row: row['Grant.Status'] == 1).groupby('Grant.Category.Code').size())
#process grant cost

# del data['Contract.Value.Band...see.note.A']

data.loc[

data['Contract.Value.Band...see.note.A'].isnull(),

    'Contract.Value.Band...see.note.A'] = 'A'

data['GrantValueCategory'] = data['Contract.Value.Band...see.note.A'].apply(

        lambda x: ord(x[0]) - ord('A') + 1

    )

data['StartYear'] = pd.to_datetime(data['Start.date'],

    infer_datetime_format = True).map(

        lambda date: date.year

    )







for i in range(1, 16):

    birthColName = f"Year.of.Birth.{i}"

    data[f"Age.{i}"] = data.apply(

        lambda row: row['StartYear'] - (row['StartYear'] if np.isnan(row[birthColName]) else row[birthColName])

        , axis = 1

    )

data['GroupAge'] = data.apply(

    lambda row: row['Age.1'] + row['Age.2'] + row['Age.3'] + row['Age.4'] + row['Age.5']

              + row['Age.6'] + row['Age.7'] + row['Age.8'] + row['Age.9'] + row['Age.10']

              + row['Age.11'] + row['Age.12'] + row['Age.13'] + row['Age.14'] + row['Age.15']

    , axis = 1)

# minYear = data['StartYear'].min()

# data['StartYear'] -= minYear
for i in range(1, 6):

    rfcdCodeColName = f"RFCD.Code.{i}"

    rfcdPercColName = f"RFCD.Percentage.{i}"

    data[rfcdCodeColName] = data[rfcdCodeColName].fillna(0)

    

    seoCodeColName = f"SEO.Code.{i}"

    seoPercColName = f"SEO.Percentage.{i}"

    data[seoCodeColName] = data[seoCodeColName].fillna(0)



rfcdGrants = {}

rfcdSuccessfullGrants = {}

seoGrants = {}

seoSuccessfullGrants = {}



def codes_sums(row):

    rfcdCodes = []

    seoCodes = []

    for i in range(1, 6):

        rfcdCodeColName = f"RFCD.Code.{i}"

        rfcdCode = row[rfcdCodeColName]

        if (not np.isnan(rfcdCode)):

            rfcdCodes.append(rfcdCode)

        

        seoCodeColName = f"SEO.Code.{i}"

        seoCode = row[seoCodeColName]

        if (not np.isnan(seoCode)):

            seoCodes.append(seoCode)



    if (row['Grant.Status'] == 1):

        for code in rfcdCodes:

            if (rfcdSuccessfullGrants.get(code) == None):

                rfcdSuccessfullGrants[code] = 1

            else:

                rfcdSuccessfullGrants[code] += 1

        for code in seoCodes:

            if (seoSuccessfullGrants.get(code) == None):

                seoSuccessfullGrants[code] = 1

            else:

                seoSuccessfullGrants[code] += 1

                

    for code in rfcdCodes:

        if (rfcdGrants.get(code) == None):

            rfcdGrants[code] = 1

        else:

            rfcdGrants[code] += 1

    for code in seoCodes:

        if (seoGrants.get(code) == None):

            seoGrants[code] = 1

        else:

            seoGrants[code] += 1



print('summing...')



data.apply(codes_sums, axis=1)

    

print('mapping...')



for i in range(1, 6):

    rfcdCodeColName = f"RFCD.Code.{i}"

    rfcdPercColName = f"RFCD.Percentage.{i}"

    seoCodeColName = f"SEO.Code.{i}"

    seoPercColName = f"SEO.Percentage.{i}"

    data[f"RFCDSuccessGrants.{i}"] = data.apply(lambda row:

                                                0 if row[rfcdCodeColName] == 0

                                                else (rfcdSuccessfullGrants.get(row[rfcdCodeColName], 0) / rfcdGrants[row[rfcdCodeColName]]) *  (row[rfcdPercColName] / 100)

                                    , axis = 1)

#     data[f"SEOSuccessGrants.{i}"] = data[seoCodeColName].map(lambda x: 0 if x == 0 else seoSuccessfullGrants.get(x, 0) / seoGrants[x])

    data[f"SEOSuccessGrants.{i}"] = data.apply(lambda row:

                                                0 if row[seoCodeColName] == 0

                                                else (seoSuccessfullGrants.get(row[seoCodeColName], 0) / seoGrants[row[seoCodeColName]]) *  (row[seoPercColName] / 100)

                                    , axis = 1)

#     data[rfcdCodeColName] = data[rfcdCodeColName].map(data.where(lambda row: row['Grant.Status'] == 1).groupby(rfcdCodeColName).size() / data.groupby(rfcdCodeColName).size())

#     data[f"RFCDSuccessGrants.{i}"] = data[rfcdCodeColName].map(data.where(lambda row: row['Grant.Status'] == 1).groupby(rfcdCodeColName).size())

#     data[rfcdCodeColName] = data[rfcdCodeColName].map(lambda x: 0 if np.isnan(x) else x)

#     data.loc[data[rfcdCodeColName].isnull(), rfcdCodeColName] = 0

#     data.loc[data[rfcdCodeColName] != 0, rfcdCodeColName] = 1

    

#     data[seoCodeColName] = data[seoCodeColName].map(lambda x: 0 if np.isnan(x) else 1)

    

#     data[seoCodeColName] = data[seoCodeColName].map(data.where(lambda row: row['Grant.Status'] == 1).groupby(seoCodeColName).size() / data.groupby(seoCodeColName).size())

#     data[f"SEOSuccessGrants.{i}"] = data[seoCodeColName].map(data.where(lambda row: row['Grant.Status'] == 1).groupby(seoCodeColName).size())

#     data.loc[data[seoCodeColName].isnull(), seoCodeColName] = 0

#     data.loc[data[seoCodeColName] != 0, seoCodeColName] = 1



data['RFCDSum'] = data.apply(

    lambda row: row['RFCDSuccessGrants.1'] + row['RFCDSuccessGrants.2'] + row['RFCDSuccessGrants.3'] + row['RFCDSuccessGrants.4'] + row['RFCDSuccessGrants.5']

    , axis = 1)

data['SEOSum'] = data.apply(

    lambda row: row['SEOSuccessGrants.1'] + row['SEOSuccessGrants.2'] + row['SEOSuccessGrants.3'] + row['SEOSuccessGrants.4'] + row['SEOSuccessGrants.5']

    , axis = 1)
for i in range(1, 16):

    phdColName = f"With.PHD.{i}"

    data[phdColName] = data[phdColName].map(lambda isPhd: 1 if isPhd == 'Yes ' else 0)

    

    pIdColName = f"Person.ID.{i}"

    data[pIdColName] = data[pIdColName].map(lambda pid: 0 if math.isnan(pid) else 1)



    yearsInUnivColName = f"No..of.Years.in.Uni.at.Time.of.Grant.{i}"

    data[yearsInUnivColName] = data[yearsInUnivColName].map(lambda x: (4 if x == 'more than 15' else (3 if x == '>10 to 15' else (2 if x == '>5 to 10' else (1 if x == '>=0 to 5' else 0)))))

    
successfullFacs = {}

allFacs = {}

successfullDeps = {}

allDeps = {}



def univ_codes_sums(row):

    facsCodes = []

    deptsCodes = []

#     if (row['Grant.Application.ID'] % 100 == 0):

#         print(row['Grant.Application.ID'])

    for i in range(1, 4):

        facColName = f"Faculty.No..{i}"

        facCode = row[facColName]

        if (allFacs.get(facCode) == None):

            allFacs[facCode] = 1

        else:

            allFacs[facCode] += 1

        facsCodes.append(facCode)

        

        deptNoColName = f"Dept.No..{i}"

        deptCode = row[deptNoColName]

        newDepCode = facCode*deptCode

        if (allDeps.get(newDepCode) == None):

            allDeps[newDepCode] = 1

        else:

            allDeps[newDepCode] += 1

        deptsCodes.append(newDepCode)

        

    if (row['Grant.Status'] == 1):

        for facCode in facsCodes:

            if (successfullFacs.get(facCode) == None):

                successfullFacs[facCode] = 1

            else:

                successfullFacs[facCode] += 1

        for depCode in deptsCodes:

            if (successfullDeps.get(depCode) == None):

                successfullDeps[depCode] = 1

            else:

                successfullDeps[depCode] += 1



print('summing..')



data.apply(univ_codes_sums, axis=1)



print('mapping..')



for i in range(1, 4):

    facColName = f"Faculty.No..{i}"

    deptNoColName = f"Dept.No..{i}"

    

    data[facColName] = data[facColName].fillna(0)

    data[deptNoColName] = data[deptNoColName].fillna(0)

    

#     data[f"FacSuccess.{i}"] = data[facColName].map(lambda x: successfullFacs.get(x, 0))

    data[f"FacSuccess.{i}"] = data[facColName].map(lambda x: successfullFacs.get(x, 0) / allFacs.get(x, 1))

#     data[f"DeptSuccess.{i}"] = data.apply(lambda row: successfullDeps.get(row[facColName]*row[deptNoColName], 0), axis = 1)

    data[f"DeptSuccess.{i}"] = data.apply(lambda row: successfullDeps.get(row[facColName]*row[deptNoColName], 0) / allDeps.get(row[facColName]*row[deptNoColName], 1), axis = 1)



data['FacultiesSuccess'] = data.apply(

    lambda row: row['FacSuccess.1'] + row['FacSuccess.2'] + row['FacSuccess.3']

    , axis = 1)

data['DeptsSuccess'] = data.apply(

    lambda row: row['DeptSuccess.1'] + row['DeptSuccess.2'] + row['DeptSuccess.3']

    , axis = 1)
data = data.fillna(0)
data['Reseachers'] = data.apply(

    lambda row: row['Person.ID.1'] + row['Person.ID.2'] + row['Person.ID.3'] + row['Person.ID.4'] + row['Person.ID.5'] + 

        row['Person.ID.6'] + row['Person.ID.7'] + row['Person.ID.8'] + row['Person.ID.9'] + row['Person.ID.10'] + 

        row['Person.ID.11'] + row['Person.ID.12'] + row['Person.ID.13'] + row['Person.ID.14'] + row['Person.ID.15']

    , axis = 1)
data['Phds'] = data.apply(

    lambda row: row['With.PHD.1'] + row['With.PHD.2'] + row['With.PHD.3'] + row['With.PHD.4'] + row['With.PHD.5'] + 

        row['With.PHD.6'] + row['With.PHD.7'] + row['With.PHD.8'] + row['With.PHD.9'] + row['With.PHD.10'] + 

        row['With.PHD.11'] + row['With.PHD.12'] + row['With.PHD.13'] + row['With.PHD.14'] + row['With.PHD.15']

    , axis = 1)
data['PhdsPart'] = data.apply(lambda row: 0 if row['Phds'] == 0 else row['Phds'] / row['Reseachers'], axis = 1)
data["YearsInUniv"] = data.apply(

    lambda row: row['No..of.Years.in.Uni.at.Time.of.Grant.1'] + row['No..of.Years.in.Uni.at.Time.of.Grant.2'] + row['No..of.Years.in.Uni.at.Time.of.Grant.3'] + row['No..of.Years.in.Uni.at.Time.of.Grant.4'] + row['No..of.Years.in.Uni.at.Time.of.Grant.5'] + 

        row['No..of.Years.in.Uni.at.Time.of.Grant.6'] + row['No..of.Years.in.Uni.at.Time.of.Grant.7'] + row['No..of.Years.in.Uni.at.Time.of.Grant.8'] + row['No..of.Years.in.Uni.at.Time.of.Grant.9'] + row['No..of.Years.in.Uni.at.Time.of.Grant.10'] + 

        row['No..of.Years.in.Uni.at.Time.of.Grant.11'] + row['No..of.Years.in.Uni.at.Time.of.Grant.12'] + row['No..of.Years.in.Uni.at.Time.of.Grant.13'] + row['No..of.Years.in.Uni.at.Time.of.Grant.14'] + row['No..of.Years.in.Uni.at.Time.of.Grant.15']

    , axis = 1)
def sum_all_articles(row):

    res = 0

    for i in range(1, 16):

        res += row[f"A..{i}"]

        res += row[f"A.{i}"]

        res += row[f"B.{i}"]

        res += row[f"C.{i}"]

    return res

data['Articles'] = data.apply(sum_all_articles, axis = 1)

data['A..Sum'] = data.apply(

    lambda row: row['A..1'] + row['A..2'] + row['A..3'] + row['A..4'] + row['A..5']

    , axis = 1)

data['A.Sum'] = data.apply(

    lambda row: row['A.1'] + row['A.2'] + row['A.3'] + row['A.4'] + row['A.5']

    , axis = 1)

data['B.Sum'] = data.apply(

    lambda row: row['B.1'] + row['B.2'] + row['B.3'] + row['B.4'] + row['B.5']

    , axis = 1)

data['C.Sum'] = data.apply(

    lambda row: row['C.1'] + row['C.2'] + row['C.3'] + row['C.4'] + row['C.5']

    , axis = 1)
data['SuccessfullGrants'] = data.apply(

    lambda row: row['Number.of.Successful.Grant.1'] + row['Number.of.Successful.Grant.2'] + row['Number.of.Successful.Grant.3'] + row['Number.of.Successful.Grant.4'] + row['Number.of.Successful.Grant.5'] + 

        row['Number.of.Successful.Grant.6'] + row['Number.of.Successful.Grant.7'] + row['Number.of.Successful.Grant.8'] + row['Number.of.Successful.Grant.9'] + row['Number.of.Successful.Grant.10'] + 

        row['Number.of.Successful.Grant.11'] + row['Number.of.Successful.Grant.12'] + row['Number.of.Successful.Grant.13'] + row['Number.of.Successful.Grant.14'] + row['Number.of.Successful.Grant.15']

    , axis = 1)
data['UnsuccessfullGrants'] = data.apply(

    lambda row: row['Number.of.Unsuccessful.Grant.1'] + row['Number.of.Unsuccessful.Grant.2'] + row['Number.of.Unsuccessful.Grant.3'] + row['Number.of.Unsuccessful.Grant.4'] + row['Number.of.Unsuccessful.Grant.5'] + 

        row['Number.of.Unsuccessful.Grant.6'] + row['Number.of.Unsuccessful.Grant.7'] + row['Number.of.Unsuccessful.Grant.8'] + row['Number.of.Unsuccessful.Grant.9'] + row['Number.of.Unsuccessful.Grant.10'] + 

        row['Number.of.Unsuccessful.Grant.11'] + row['Number.of.Unsuccessful.Grant.12'] + row['Number.of.Unsuccessful.Grant.13'] + row['Number.of.Unsuccessful.Grant.14'] + row['Number.of.Unsuccessful.Grant.15']

    , axis = 1)
data = data.drop(data[(data['RFCDSum'] == 0) & (data['SEOSum'] == 0) & (data['Grant.Status'] == 1)].index)
data = data.drop(data[(data['Reseachers'] == 0) & (data['Grant.Status'] == 1)].index)
modeldata = data[[

    'SponsorSuccessGrants'

    , 'SuccessfullCategoryGrant'

    , 'GrantValueCategory'

    , 'StartYear'

    , 'GroupAge'

    , 'RFCDSum'

    , 'SEOSum'

    , 'Reseachers'

#     , 'Phds'

    , 'PhdsPart'

    , 'YearsInUniv'

    , 'FacultiesSuccess'

    , 'DeptsSuccess'

    , 'A..Sum'

    , 'A.Sum'

    , 'B.Sum'

    , 'C.Sum'

#     , 'Articles'

    , 'SuccessfullGrants'

    , 'UnsuccessfullGrants'

]].copy()





# for i in range(1, 6):

#     modeldata[f"RFCDSuccessGrants.{i}"] = data[f"RFCDSuccessGrants.{i}"]

#     modeldata[f"SEOSuccessGrants.{i}"] = data[f"SEOSuccessGrants.{i}"]



# for i in range(1, 16):

#     modeldata[f"Age.{i}"] = data[f"Age.{i}"]

#     modeldata[f"With.PHD.{i}"] = data[f"With.PHD.{i}"]

#     modeldata[f"Person.ID.{i}"] = data[f"Person.ID.{i}"]

#     modeldata[f"No..of.Years.in.Uni.at.Time.of.Grant.{i}"] = data[f"No..of.Years.in.Uni.at.Time.of.Grant.{i}"]



y = data['Grant.Status']
print(modeldata.shape)

print(modeldata)

# print(normilized_data)

# print(scaled_data)
from sklearn import preprocessing

# normilized_data = preprocessing.normalize(modeldata.loc[:,:])

scaled_data = preprocessing.scale(modeldata.loc[:,:])
from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn import svm

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import GradientBoostingClassifier



from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import RandomizedSearchCV



from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

from sklearn.datasets import load_iris



model = GradientBoostingClassifier()

random_forest_params  = {

    'bootstrap': [False],

    'max_depth': [70],

    'max_features': ['auto'],

    'min_samples_leaf': [1],

    'min_samples_split': [10],

    'n_estimators': [600, 650, 700]

}



gradient_boost_parameters = {

    'n_estimators':[200, 300, 400],

    'max_depth':[10],

    'learning_rate':[0.1],

    'loss' : ['linear', 'square', 'exponential']

}



model_params = gradient_boost_parameters



classifiers = {

    DecisionTreeClassifier() : {'max_leaf_nodes': [1,10,30,50,70,90,100], 'min_samples_split': [2, 3, 4]}

    , svm.SVC(): {'C':[100], 'tol': [0.005], 'kernel':['sigmoid']}

    , KNeighborsClassifier(): {'n_neighbors':[5], 'weights':['distance'],'leaf_size':[15]}

    , LogisticRegression(): {'C':[2000], 'tol': [0.0001]}

    , GradientBoostingClassifier(): {'learning_rate':[0.01],'n_estimators':[100], 'max_depth':[3], 'min_samples_split':[2],'min_samples_leaf': [2]}

    , AdaBoostClassifier(): {'learning_rate':[0.01], 'n_estimators':[150]}

}



# X_train, X_test, y_train, y_test = train_test_split(modeldata, y, test_size=0.20, random_state=47)

# import eli5

# from eli5.sklearn import PermutationImportance

# model = RandomForestClassifier()

# model.fit(X_train, y_train)

# perm = PermutationImportance(model, random_state=1).fit(X_test, y_test)

# eli5.show_weights(perm, feature_names = X_test.columns.tolist())



# print(classification_report( model.predict(X_test), y_test))



X_train, X_test, y_train, y_test = train_test_split(scaled_data, y, test_size=0.20, random_state=47)

# for model in classifiers:

#     print(f"{type(model).__name__}")

#     model.fit(X_train, y_train)

#     acc = accuracy_score(y_test, model.predict(X_test))

#     print(f"Accuracy: {acc:.4f}")

#     print()



#     print(classification_report(res, y_test))

# clf = GridSearchCV(model, model_params, cv=5, verbose=0, n_jobs=-1, scoring='accuracy')

# clf.fit(X_train, y_train)

# print("best params: " + str(clf.best_params_))

# print("best scores: " + str(clf.best_score_))

# print(classification_report( clf.predict(X_test), y_test))



best_params= {

    'bootstrap': False,

    'max_depth': 70,

    'max_features': 'auto',

    'min_samples_leaf': 1,

    'min_samples_split': 10,

    'n_estimators': 900

}



model = RandomForestClassifier(bootstrap = False, max_depth = 70, min_samples_leaf = 1, n_estimators = 900, max_features = 'auto')

model.fit(X_train, y_train)

res = model.predict(X_test)

print(classification_report(res, y_test))

print('end')

# data['class'] = y

# data.to_csv('data.csv',index=False)