import numpy as np
import pandas as pd 
import os
print(os.listdir("../input"))
['train.csv', 'gender_submission.csv', 'test.csv']
# Step 1 is to import both data sets
training_data = pd.read_csv("../input/train.csv")
testing_data = pd.read_csv("../input/test.csv")

# Step two is to create columns which I will add to the respective datasets, in order to know which row came from which dataset when I combine the datasets
training_column = pd.Series([1] * len(training_data))
testing_column = pd.Series([0] * len(testing_data))

# Now we append them by creating new columns in the original data. We use the same column name
training_data['is_training_data'] = training_column
testing_data['is_training_data'] = testing_column
# Now we can merge the datasets while retaining the key to split them later
combined_data = training_data.append(testing_data, ignore_index=True, sort=False)

# Encode gender (if == female, True)
combined_data['female'] = combined_data.Sex == 'female'

# Split out Title
title = []
for i in combined_data['Name']:
    period = i.find(".")
    comma = i.find(",")
    title_value = i[comma+2:period]
    title.append(title_value)
combined_data['title'] = title

# Replace the title values with an aliased dictionary
title_arr = pd.Series(title)
title_dict = {
    "Capt" : "Rare",
    "Col" : "Rare",
    "Major" : "Rare",
    "Jonkheer" : "Rare",
    "Don" : "Rare",
    "Sir" : "Rare",
    "Dr" : "Rare",
    "Rev" : "Rare",
    "the Countess" : "Rare",
    "Dona" : "Rare",
    "Mme" : "Mrs",
    "Mlle" : "Miss",
    "Ms" : "Mrs",
    "Mr" : "Mr",
    "Mrs" : "Mrs",
    "Miss" : "Miss",
    "Master" : "Master",
    "Lady" : "Rare"
}
cleaned_title = title_arr.map(title_dict)
combined_data['cleaned_title'] = cleaned_title

# Fill NaN of Age - first create groups to find better medians than just the overall median and fill NaN with the grouped medians
grouped = combined_data.groupby(['female','Pclass', 'cleaned_title']) 
combined_data['Age'] = grouped.Age.apply(lambda x: x.fillna(x.median()))

#add an age bin
age_bin_conditions = [
    combined_data['Age'] == 0,
    (combined_data['Age'] > 0) & (combined_data['Age'] <= 16),
    (combined_data['Age'] > 16) & (combined_data['Age'] <= 34),
    (combined_data['Age'] > 34) & (combined_data['Age'] <= 49),
    (combined_data['Age'] > 49) & (combined_data['Age'] <= 64),
    combined_data['Age'] > 64
]
age_bin_outputs = [0, 1, 2, 3, 4, 5]
combined_data['age_bin'] = np.select(age_bin_conditions, age_bin_outputs, 'Other').astype(int)

# Fill NaN of Embarked
combined_data['Embarked'] = combined_data['Embarked'].fillna("S") 

# Fill NaN of Fare, adding flag for boarded free, binning other fares
combined_data['Fare'] = combined_data['Fare'].fillna(combined_data['Fare'].mode()[0]) 
combined_data['boarded_free'] = combined_data['Fare'] == 0 
fare_bin_conditions = [
    combined_data['Fare'] == 0,
    (combined_data['Fare'] > 0) & (combined_data['Fare'] <= 7.9),
    (combined_data['Fare'] > 7.9) & (combined_data['Fare'] <= 14.4),
    (combined_data['Fare'] > 14.4) & (combined_data['Fare'] <= 31),
    combined_data['Fare'] > 31
]
fare_bin_outputs = [0, 1, 2, 3, 4]
combined_data['fare_bin'] = np.select(fare_bin_conditions, fare_bin_outputs, 'Other').astype(int)

# Fill NaN of Cabin with a U for unknown. Not sure cabin will help.
combined_data['Cabin'] = combined_data['Cabin'].fillna("U") 

# Counting how many people are riding on a ticket
from collections import Counter
tickets_count = pd.DataFrame([Counter(combined_data['Ticket']).keys(), Counter(combined_data['Ticket']).values()]).T
tickets_count.rename(columns={0:'Ticket', 1:'ticket_riders'}, inplace=True)
tickets_count['ticket_riders'] = tickets_count['ticket_riders'].astype(int)
combined_data = combined_data.merge(tickets_count, on='Ticket')

# Finding cabin group
cabin_group = []
for i in combined_data['Cabin']:
    cabin_group.append(i[0])
combined_data['cabin_group'] = cabin_group

# Adding a family_size feature as it may have an inverse relationship to either of its parts
combined_data['family_size'] = combined_data.Parch + combined_data.SibSp + 1

# Mapping ports to passenger pickup order
port = {
    'S' : 1,
    'C' : 2,
    'Q' : 3
}
combined_data['pickup_order'] = combined_data['Embarked'].map(port)

# Encode childhood
combined_data['child'] = combined_data.Age < 16

# One-Hot Encoding the titles
combined_data = pd.concat([combined_data, pd.get_dummies(combined_data['cleaned_title'], prefix="C_T")], axis = 1)

# One-Hot Encoding the Pclass
combined_data = pd.concat([combined_data, pd.get_dummies(combined_data['Pclass'], prefix="PClass")], axis = 1)

# One-Hot Encoding the  cabin group
combined_data = pd.concat([combined_data, pd.get_dummies(combined_data['cabin_group'], prefix="C_G")], axis = 1)

# One-Hot Encoding the ports
combined_data = pd.concat([combined_data, pd.get_dummies(combined_data['Embarked'], prefix="Embarked")], axis = 1)
new_train_data=combined_data.loc[combined_data['is_training_data']==1]
new_test_data=combined_data.loc[combined_data['is_training_data']==0]
# here is the expanded model set and metric tools
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier

k_fold = KFold(n_splits = 10, shuffle=True, random_state=0) 
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
# Here are the features
features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'female', 'child', 'Embarked_S', 'Embarked_C', 
            'Embarked_Q', 'pickup_order', 'C_T_Master', 'C_T_Miss', 'C_T_Mr', 'C_T_Mrs',
            'C_T_Rare', 'C_G_A', 'C_G_B', 'C_G_C', 'C_G_D', 'C_G_E', 'C_G_F', 'C_G_G', 
            'C_G_T', 'C_G_U', 'family_size', 'PClass_1', 'PClass_2', 'PClass_3', 'ticket_riders']
target = 'Survived'
cvs_train_data = new_train_data[features]
cvs_test_data = new_test_data[features]
cvs_target = new_train_data['Survived']
cvs_train_data.shape
# Define the classifiers I will use
classifiers = [
    RandomForestClassifier(n_estimators=10, random_state=0),
    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=10, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=2, min_samples_split=6,
            min_weight_fraction_leaf=0.0, n_estimators=35, n_jobs=None,
            oob_score=False, random_state=0, verbose=0, warm_start=False),
    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=9, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=3, min_samples_split=10,
            min_weight_fraction_leaf=0.0, n_estimators=40, n_jobs=1,
            oob_score=False, random_state=0, verbose=0, warm_start=False),
    DecisionTreeClassifier(random_state=0),
    LogisticRegression(solver='liblinear'),
    KNeighborsClassifier(n_neighbors=15),
    SVC(gamma='auto'),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    ExtraTreesClassifier(),
    XGBClassifier(),
    GaussianNB(),
    LinearSVC()]

# Fit and use cross_val_score and k_fold to score accuracy
clf_scores = [['Score', 'Classifier']]
i = 0
for clf in classifiers:
    name = clf.__class__.__name__
    acc_score = round(np.mean(cross_val_score(clf, cvs_train_data, cvs_target, cv=k_fold, n_jobs=1, scoring='accuracy'))*100,2)
    print(acc_score, "%", name);
    clf_scores.append([acc_score, name])
    clf.fit(cvs_train_data, cvs_target)
    prediction = clf.predict(cvs_test_data)
    submission = pd.DataFrame({
        "PassengerId" : new_test_data['PassengerId'],
        "Survived" : prediction.astype(int)
    })
    submission_name = 's_{}_{}.csv'.format(i, name)
    submission.to_csv(submission_name, index=False)
    i += 1
    
print(clf_scores)
clf_scores2 = clf_scores.copy()
df_scores = pd.DataFrame(clf_scores2, columns=clf_scores2.pop(0))
df_scores.sort_values('Score', ascending=False)
from sklearn.feature_selection import chi2, SelectKBest
fit_list = [SelectKBest(score_func=chi2, k=i) for i in range (0,32)]

fit_test = SelectKBest(score_func=chi2, k='all')
fit_test.fit(cvs_train_data, cvs_target)
np.set_printoptions(precision=3)

features3 = fit_test.transform(cvs_train_data)
fit_test_scores = pd.DataFrame(fit_test.scores_)
fit_test_scores['feature_name'] = pd.DataFrame(cvs_train_data.columns)
fit_test_scores.columns = ['chi_squared', 'feature_name']
fit_test_scores.sort_values('chi_squared', ascending=False)
# So we're just going to take the features with a chi_squared over 1. 
new_features = ['Fare','female','C_T_Mr','C_T_Mrs','C_T_Miss','PClass_1','PClass_3','C_G_B','C_G_U','Embarked_C','Age','C_G_D','C_G_E','child','C_G_C','Parch','C_T_Master','PClass_2','ticket_riders','Embarked_S','pickup_order','C_G_F','SibSp']
len(new_features)
# rf_ stands for reduced features
rf_cvs_train_data = new_train_data[new_features]
rf_cvs_test_data = new_test_data[new_features]
rf_cvs_target = new_train_data['Survived']
rf_cvs_train_data.shape
# Fit and use cross_val_score and k_fold to score accuracy
i = 0
for clf in classifiers:
    name = clf.__class__.__name__
    acc_score = round(np.mean(cross_val_score(clf, rf_cvs_train_data, rf_cvs_target, cv=k_fold, n_jobs=1, scoring='accuracy'))*100,2)
    print(acc_score, "%", name);
    clf_scores.append([acc_score, name])
    clf.fit(rf_cvs_train_data, rf_cvs_target)
    prediction = clf.predict(rf_cvs_test_data)
    submission = pd.DataFrame({
        "PassengerId" : new_test_data['PassengerId'],
        "Survived" : prediction.astype(int)
    })
    submission_name = 's_{}_{}_2.csv'.format(i, name)
    submission.to_csv(submission_name, index=False)
    i += 1

new_features_2 = ['age_bin', 'SibSp', 'Parch', 'fare_bin', 'female', 'child', 'Embarked_S', 'Embarked_C', 
            'Embarked_Q', 'pickup_order', 'C_T_Master', 'C_T_Miss', 'C_T_Mr', 'C_T_Mrs',
            'C_T_Rare', 'family_size', 'PClass_1', 'PClass_2', 'PClass_3', 'ticket_riders']



#nf stands for new features aka with bins
nf_cvs_train_data = new_train_data[new_features_2]
nf_cvs_test_data = new_test_data[new_features_2]
nf_cvs_target = new_train_data['Survived']
nf_cvs_train_data.shape, nf_cvs_test_data.shape
# Fit and use cross_val_score and k_fold to score accuracy
i = 0
for clf in classifiers:
    name = clf.__class__.__name__
    acc_score = round(np.mean(cross_val_score(clf, nf_cvs_train_data, nf_cvs_target, cv=k_fold, n_jobs=1, scoring='accuracy'))*100,2)
    print(acc_score, "%", name);
    clf_scores.append([acc_score, name])
    clf.fit(nf_cvs_train_data, nf_cvs_target)
    prediction = clf.predict(nf_cvs_test_data)
    submission = pd.DataFrame({
        "PassengerId" : new_test_data['PassengerId'],
        "Survived" : prediction.astype(int)
    })
    submission_name = 's_{}_{}_3.csv'.format(i, name)
    submission.to_csv(submission_name, index=False)
    i += 1
combined_data.dtypes.index
new_features_3 = ['PassengerId', 'Pclass', 'Age', 'SibSp',
       'Parch', 'Fare', 'female', 'age_bin', 'boarded_free',
       'fare_bin', 'ticket_riders', 'family_size',
       'pickup_order', 'child', 'C_T_Master', 'C_T_Miss', 'C_T_Mr', 'C_T_Mrs',
       'C_T_Rare', 'PClass_1', 'PClass_2', 'PClass_3', 'C_G_A', 'C_G_B',
       'C_G_C', 'C_G_D', 'C_G_E', 'C_G_F', 'C_G_G', 'C_G_T', 'C_G_U',
       'Embarked_C', 'Embarked_Q', 'Embarked_S']
#nf stands for new features aka with bins
nf3_cvs_train_data = new_train_data[new_features_3]
nf3_cvs_test_data = new_test_data[new_features_3]
nf3_cvs_target = new_train_data['Survived'].astype(int)
nf3_cvs_train_data.shape, nf3_cvs_test_data.shape, nf3_cvs_target.shape
fit_list = [SelectKBest(score_func=chi2, k=i) for i in range (0,32)]

fit_test = SelectKBest(score_func=chi2, k='all')
fit_test.fit(nf3_cvs_train_data, nf3_cvs_target)
np.set_printoptions(precision=3)

features3 = fit_test.transform(nf3_cvs_train_data)
fit_test_scores = pd.DataFrame(fit_test.scores_)
fit_test_scores['feature_name'] = nf3_cvs_train_data.columns
fit_test_scores.columns = ['chi_squared', 'feature_name']
fit_test_scores.sort_values('chi_squared', ascending=False)
from sklearn.feature_selection import RFE
classifiers2 = [
    RandomForestClassifier(n_estimators=10, random_state=0),
    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=10, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=2, min_samples_split=6,
            min_weight_fraction_leaf=0.0, n_estimators=35, n_jobs=None,
            oob_score=False, random_state=0, verbose=0, warm_start=False),
    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=9, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=3, min_samples_split=10,
            min_weight_fraction_leaf=0.0, n_estimators=40, n_jobs=1,
            oob_score=False, random_state=0, verbose=0, warm_start=False),
    DecisionTreeClassifier(random_state=0),
    LogisticRegression(solver='liblinear'),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    ExtraTreesClassifier(),
    XGBClassifier(),
    LinearSVC()]

# d = [['clf_name', 'feature_name', 'n_features', 'support', 'ranking']]
# for clf in classifiers2:
#     for i in range (1,3):
#         name = clf.__class__.__name__
#         print(name, i)
#         rfe = RFE(clf, i)
#         fitted_clf = rfe.fit(nf3_cvs_train_data, nf3_cvs_target)
#         n_feat = fitted_clf.n_features_
#         n_supp = fitted_clf.support_
#         rank = fitted_clf.ranking_
#         d_new = [name, new_features_3, n_feat, n_supp, rank]
#         d.append(d_new)
        
# d_f = pd.DataFrame(d)
# d_f
# from yellowbrick.features import RFECV

# import warnings
# warnings.filterwarnings("ignore")

# # Create RFECV visualizer with linear SVM classifier
# for clf in classifiers2:
#     viz = RFECV(clf)
#     viz.fit(nf3_cvs_train_data, nf3_cvs_target);
#     viz.poof();
for clf in classifiers2:
    for i in (1, 3, 8, 15, 20, 21, 24, 27, 28, 31, 33):
        name = clf.__class__.__name__
        print(name, i)
        rfe = RFE(clf, i)
        fitted_clf = rfe.fit(nf3_cvs_train_data, nf3_cvs_target)
        n_feat = fitted_clf.n_features_
        n_supp = fitted_clf.support_
        rank = fitted_clf.ranking_
        d_new = [name, new_features_3, n_feat, n_supp, rank]
        d.append(d_new)


d_f = pd.DataFrame(d)
d_f.to_csv('csv')

# Fit and use cross_val_score and k_fold to score accuracy
i = 0
for clf in classifiers:
    name = clf.__class__.__name__
    acc_score = round(np.mean(cross_val_score(clf, nf3_cvs_train_data, nf3_cvs_target, cv=k_fold, n_jobs=1, scoring='accuracy'))*100,2)
    print(acc_score, "%", name);
    clf_scores.append([acc_score, name])
    clf.fit(nf3_cvs_train_data, nf3_cvs_target)
    prediction = clf.predict(nf3_cvs_test_data)
    submission = pd.DataFrame({
        "PassengerId" : new_test_data['PassengerId'],
        "Survived" : prediction.astype(int)
    })
    submission_name = 's_{}_{}_4.csv'.format(i, name)
    submission.to_csv(submission_name, index=False)
    i += 1
