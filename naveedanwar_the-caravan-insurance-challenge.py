import pandas as pd
import numpy as np
from sklearn import metrics
pd.set_option("display.max_rows", 10)
import seaborn as sns
%matplotlib inline  
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectPercentile
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import RFE
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.linear_model import LogisticRegression,LinearRegression,Ridge
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix

url="../input/caravan-insurance-challenge -train.csv"
train_data=pd.read_table(url,sep=',',skiprows=1,names=['Customer Subtype', 'Number of houses', 'Avg size household', 'Avg age', 'Customer main type', 'Roman catholic',
       'Protestant', 'Other religion', 'No religion', 'Married', 'Living together', 'Other relation', 'Singles',
       'Household without children', 'Household with children', 'High level education', 'Medium level education', 'Lower level education', 'High status',
       'Entrepreneur', 'Farmer', 'Middle management', 'Skilled labourers', 'Unskilled labourers', 'Social class A',
       'Social class B1', 'Social class B2', 'Social class C', 'Social class D', 'Rented house', 'Home owners', '1 car', '2 cars',
       'No car', 'National Health Service', 'Private health insurance', 'Income < 30.000', 'Income 30-45.000', 'ncome 45-75.000',
       'Income 75-122.000', 'Income >123.000', 'Average income', 'Purchasing power class', 'Contribution private third party insurance', 'Contribution third party insurance',
       ' Contribution third party insurane', 'Contribution car policies', 'Contribution delivery van policies', 'Contribution motorcycle/scooter policies', 'Contribution lorry policies', 'Contribution trailer policies',
       'Contribution tractor policies', 'Contribution agricultural machines policies', 'Contribution moped policies', 'Contribution life insurances', 'Contribution private accident insurance policies', 'Contribution family accidents insurance policies',
       'Contribution disability insurance policies', 'Contribution fire policies', 'Contribution surfboard policies', 'Contribution boat policies', 'Contribution bicycle policies', 'Contribution property insurance policies',
       'Contribution social security insurance policies', 'Number of private third party insurance', ' Number of third party insurance (firms)', 'Number of third party insurane (agriculture)', 'Number of car policies', 'Number of delivery van policies',
       'Number of motorcycle/scooter policies', 'Number of lorry policies', 'Number of trailer policies', 'number of tractor policies', 'Number of agricultural machines policies', 'Number of moped policies',
       'Number of life insurances', 'number of private accident insurance policies', ' Number of family accidents insurance policies', ' Number of disability insurance policies', 'Number of fire policies', 'Number of surfboard policies',
       'Number of boat policies', 'Number of bicycle policies', 'Number of property insurance policies', 'Number of social security insurance policies', 'CARAVAN'])
train_with_target = train_data
train_target = train_data['CARAVAN']
train_data = train_data.drop('CARAVAN',axis=1)
train_data
url="../input/caravan-insurance-challenge-test.csv"
test_data=pd.read_table(url,sep=',',encoding="latin-1",names=['Customer Subtype', 'Number of houses', 'Avg size household', 'Avg age', 'Customer main type', 'Roman catholic',
       'Protestant', 'Other religion', 'No religion', 'Married', 'Living together', 'Other relation', 'Singles',
       'Household without children', 'Household with children', 'High level education', 'Medium level education', 'Lower level education', 'High status',
       'Entrepreneur', 'Farmer', 'Middle management', 'Skilled labourers', 'Unskilled labourers', 'Social class A',
       'Social class B1', 'Social class B2', 'Social class C', 'Social class D', 'Rented house', 'Home owners', '1 car', '2 cars',
       'No car', 'National Health Service', 'Private health insurance', 'Income < 30.000', 'Income 30-45.000', 'ncome 45-75.000',
       'Income 75-122.000', 'Income >123.000', 'Average income', 'Purchasing power class', 'Contribution private third party insurance', 'Contribution third party insurance',
       ' Contribution third party insurane', 'Contribution car policies', 'Contribution delivery van policies', 'Contribution motorcycle/scooter policies', 'Contribution lorry policies', 'Contribution trailer policies',
       'Contribution tractor policies', 'Contribution agricultural machines policies', 'Contribution moped policies', 'Contribution life insurances', 'Contribution private accident insurance policies', 'Contribution family accidents insurance policies',
       'Contribution disability insurance policies', 'Contribution fire policies', 'Contribution surfboard policies', 'Contribution boat policies', 'Contribution bicycle policies', 'Contribution property insurance policies',
       'Contribution social security insurance policies', 'Number of private third party insurance', ' Number of third party insurance (firms)', 'Number of third party insurane (agriculture)', 'Number of car policies', 'Number of delivery van policies',
       'Number of motorcycle/scooter policies', 'Number of lorry policies', 'Number of trailer policies', 'number of tractor policies', 'Number of agricultural machines policies', 'Number of moped policies',
       'Number of life insurances', 'number of private accident insurance policies', ' Number of family accidents insurance policies', ' Number of disability insurance policies', 'Number of fire policies', 'Number of surfboard policies',
       'Number of boat policies', 'Number of bicycle policies', 'Number of property insurance policies', 'Number of social security insurance policies','CARAVAN'])

test = test_data
test_data = test_data.loc[:,'Customer Subtype':'Number of social security insurance policies']
test_data
test_target = test.loc[:,'CARAVAN':]
test_target
train_with_target.describe()
train_with_target.hist(figsize=(40,40))
plt.show()
categorysubtype_caravan = pd.crosstab(train_data['Customer Subtype'], train_target)
categorysubtype_caravan_pct = categorysubtype_caravan.div(categorysubtype_caravan.sum(1).astype(float), axis=0)
categorysubtype_caravan_pct.plot(figsize= (15,5), kind='bar', stacked=True, color=['steelblue', 'springgreen'], title='category type vs Caravan', grid=True)
plt.xlabel('Category subtype')  # It has 41 different types
plt.ylabel('Caravan or not')
age_caravan = pd.crosstab(train_data['Avg age'], train_target)
age_caravan_pct = age_caravan.div(age_caravan.sum(1).astype(float),axis=0)
age_caravan_pct.plot(figsize=(15,5), kind='bar', stacked=True, color=['steelblue', 'springgreen'], title='dependency of caravan on age groups', grid=True)
plt.xlabel('age groups')
plt.ylabel('Caravan')
train_data['Customer main type'].value_counts().plot(kind='bar', color='steelblue', grid=True)
plt.xlabel('Customer Main Types')
plt.ylabel('count')
X_train = train_data
y_train = train_target
X_test = test_data
y_test = test_target

test_target['CARAVAN'].value_counts()
fig = plt.figure(figsize=(10,10))
plt.subplot(3,1,1)
test_target['CARAVAN'].value_counts().plot(kind='bar', title='Classifying CARAVAN', color='steelblue', grid=True)
# 94% are zeros
max(test_target['CARAVAN'].mean(), 1 - test_target['CARAVAN'].mean())
from sklearn.metrics import confusion_matrix
def draw_confusion_matrices(confusion_matricies,class_names):
    class_names = class_names.tolist()
    for cm in confusion_matrices:
        classifier, cm = cm[0], cm[1]
        print(cm)
        
        fig = plt.figure(figsize=(10,5))
        ax = fig.add_subplot(111)
        cax = ax.matshow(cm)
        plt.title('Confusion matrix for %s' % classifier)
        fig.colorbar(cax)
        ax.set_xticklabels([''] + class_names)
        ax.set_yticklabels([''] + class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(train_data,train_target)
predictions=knn.predict(test_data)
print("Accuracy on training set: {:.2f}".format(knn.score(train_data, train_target)))
print("Accuracy on test set: {:.2f}".format(knn.score(test_data, test_target)))
print("Mean square error " , metrics.mean_squared_error(test_target, predictions))

class_names = np.unique(np.array(y_test))
confusion_matrices = [
    ( "KNeighborsClassifier", confusion_matrix(test_target, predictions))
]

# calling below function
draw_confusion_matrices(confusion_matrices,class_names)

scores1 = cross_val_score(knn, X_train ,y_train)
print("Accuracy on training after cross validation: %0.2f (+/- %0.2f)" % (scores1.mean(), scores1.std() * 2))
scores2 = cross_val_score(knn, X_test , y_test)
print("Accuracy on testing after cross validation: %0.2f (+/- %0.2f)" % (scores2.mean(), scores2.std() * 2))
corr_matrix = train_with_target.corr().style.background_gradient()
corr_matrix
corr_matrix = train_with_target.corr()
corr_matrix['CARAVAN'].sort_values(ascending=False)
def correlation(dataset, threshold):
    col_corr = set() # Set of all the names of deleted columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if corr_matrix.iloc[i, j] > threshold:
                colname = corr_matrix.columns[i] # getting the name of column
                col_corr.add(colname)
                if colname in dataset.columns:
                    del dataset[colname] # deleting the column from the dataset

    return dataset
correlation(train_with_target, 0.4)
selected_features = correlation(train_with_target, 0.4) 
selected_features.shape
selected_features.loc[:,selected_features.columns != 'CARAVAN']
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(selected_features.loc[:,selected_features.columns != 'CARAVAN']  ,selected_features.loc[:,selected_features.columns == 'CARAVAN'])
predictions=knn.predict(test_data.loc[:,selected_features.columns!='CARAVAN'])
print("Accuracy on training set: {:.3f}".format(knn.score(selected_features.loc[:,selected_features.columns != 'CARAVAN'], selected_features.loc[:,selected_features.columns == 'CARAVAN'])))
print("Accuracy on test set: {:.3f}".format(knn.score(test_data.loc[:,selected_features.columns!='CARAVAN'], test_data.loc[:,selected_features.columns=='CARAVAN'])))

print("Mean square error " , metrics.mean_squared_error(test_target, predictions))
print(metrics.confusion_matrix(test_target, predictions))
scores1 = cross_val_score(knn, selected_features.loc[:,selected_features.columns != 'CARAVAN'] ,selected_features.loc[:,selected_features.columns == 'CARAVAN'])
print("Accuracy on training after cross validation: %0.2f (+/- %0.2f)" % (scores1.mean(), scores1.std() * 2))
scores2 = cross_val_score(knn, test_data.loc[:,selected_features.columns!='CARAVAN'] , test_data.loc[:,selected_features.columns=='CARAVAN'])
print("Accuracy on testing after cross validation: %0.2f (+/- %0.2f)" % (scores2.mean(), scores2.std() * 2))
select = SelectPercentile(percentile=50)
select.fit(X_train, y_train)
X_train_selected = select.transform(X_train)
print("X_train.shape: {}".format(X_train.shape))
print("X_train_selected.shape: {}".format(X_train_selected.shape))
X_test_selected = select.transform(X_test.loc[:,'Customer Subtype':'Number of social security insurance policies'])
X_test = X_test.loc[:,'Customer Subtype':'Number of social security insurance policies']
from sklearn.metrics import accuracy_score

print("Knn")
knn.fit(X_train, y_train)
print("Score with all features: {:.3f}".format(knn.score(X_test, y_test)))
knn.fit(X_train_selected, y_train)
print("Score with only selected features: {:.3f}".format(knn.score(X_test_selected, y_test)))
y_pred=knn.predict(X_test_selected)



lr = LogisticRegression()
lr.fit(X_train, y_train)
print("Logistic Regression:")
print("Score with all features: {:.3f}".format(lr.score(X_test, y_test)))
lr.fit(X_train_selected, y_train)
print("Score with only selected features: {:.3f}".format(lr.score(X_test_selected, y_test)))



dt = DecisionTreeClassifier(max_depth=10,random_state=0)
dt.fit(X_train, y_train)
print("Decision tree")
print("Accuracy on test set without select: {:.3f}".format(dt.score(X_test, y_test)))
dt.fit(X_train_selected, y_train)
print("Accuracy on test set with select: {:.3f}".format(dt.score(X_test_selected, y_test)))


#random forest
forest = RandomForestClassifier(n_estimators=25, random_state=0)
forest.fit(X_train, y_train)
print("Random forest")
print("Accuracy on test set without select: {:.3f}".format(forest.score(X_test, y_test)))
forest.fit(X_train_selected, y_train)
print("Accuracy on test set with select: {:.3f}".format(forest.score(X_test_selected, y_test)))

from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
select = SelectFromModel(
    RandomForestClassifier(n_estimators=100, random_state=42),
    threshold="median")
select.fit(X_train, y_train)
X_train_l1 = select.transform(X_train)
print("X_train.shape: {}".format(X_train.shape))
print("X_train_l1.shape: {}".format(X_train_l1.shape))
X_test_l1 = select.transform(X_test)


score = LogisticRegression().fit(X_train_l1, y_train).score(X_test_l1, y_test)
print("Logistic Regression Test score: {:.3f}".format(score))
knn = KNeighborsClassifier(n_neighbors=30).fit(X_train_l1, y_train).score(X_test_l1, y_test)
print("KNN Test score: {:.3f}".format(knn))
dt = DecisionTreeClassifier(max_depth=10, random_state=0).fit(X_train_l1, y_train).score(X_test_l1, y_test)
print("Decision Tree Test score: {:.3f}".format(knn))
forest = RandomForestClassifier(n_estimators=100, random_state=0).fit(X_train_l1, y_train).score(X_test_l1, y_test)
print("Random Forest Test score: {:.3f}".format(forest))
#univariate
rng = np.random.RandomState(42)
noise = rng.normal(size=(len(train_data), 0))
# add noise features to the data
# the first 30 features are from the dataset, the next 50 are noise
X_w_noise = np.hstack([train_data, noise])

X_train, X_test, y_train, y_test = train_test_split(
    X_w_noise, train_target, random_state=0, test_size=.5)


select = SelectPercentile(percentile=47)
select.fit(X_train, y_train)
# transform training set
X_train_selected = select.transform(X_train)

print("X_train.shape: {}".format(X_train.shape))
print("X_train_selected.shape: {}".format(X_train_selected.shape))
# transform test data
X_test_selected = select.transform(X_test)

#Logistic Regression
lr = LogisticRegression()
lr.fit(X_train, y_train)


print("Logistic Regression:")
print("Score with all features: {:.3f}".format(lr.score(X_test, y_test)))
lr.fit(X_train_selected, y_train)
print("Score with only selected features: {:.3f}".format(lr.score(X_test_selected, y_test)))

#decision tree
dt = DecisionTreeClassifier(max_depth=10, random_state=0)
dt.fit(X_train, y_train)


print("Decision tree")
print("Accuracy on test set without select: {:.3f}".format(dt.score(X_test, y_test)))
dt.fit(X_train_selected, y_train)
print("Accuracy on test set with select: {:.3f}".format(dt.score(X_test_selected, y_test)))

#knn
knn = KNeighborsClassifier(n_neighbors=20)
knn.fit(X_train, y_train)
print("K nearest neighbours")
print("Accuracy on test set without select: {:.3f}".format(knn.score(X_test, y_test)))
knn.fit(X_train_selected, y_train)
print("Accuracy on test set with select: {:.3f}".format(knn.score(X_test_selected, y_test)))

## random forest
forest = RandomForestClassifier(n_estimators=100, random_state=0)
forest.fit(X_train, y_train)
print("Random forest")
print("Accuracy on test set without select: {:.3f}".format(forest.score(X_test, y_test)))
forest.fit(X_train_selected, y_train)
print("Accuracy on test set with select: {:.3f}".format(forest.score(X_test_selected, y_test)))
from sklearn.feature_selection import RFE
select = RFE(RandomForestClassifier(n_estimators=5, random_state=42),
             n_features_to_select=25)

select.fit(X_train, y_train)
X_train_rfe = select.transform(X_train)
X_test_rfe = select.transform(X_test)
#Logistic Regression
score = LogisticRegression().fit(X_train_rfe, y_train).score(X_test_rfe, y_test)
print("Logistic Regression Test score without select: {:.3f}".format(score))
print("Logistic Regression Test score with selection: {:.3f}".format(select.score(X_test, y_test)))

print("\n")
#knn
score = KNeighborsClassifier(n_neighbors=30).fit(X_train_rfe, y_train).score(X_test_rfe, y_test)
print("Knn Test set accuracy without select: {:.2f}".format(score))
print("Knn Test set accuracy with selection: {:.2f}".format(select.score(X_test, y_test)))
print("\n")
#decision tree
score = DecisionTreeClassifier(max_depth=10, random_state=0).fit(X_train_rfe, y_train).score(X_test_rfe, y_test)
print("Decision Tree Accuracy on test set without select: {:.3f}".format(score))
print("Decision Tree Accuracy on test selection: {:.3f}".format(select.score(X_test, y_test)))
print("\n")
#random forest
forest = RandomForestClassifier(n_estimators=100, random_state=0).fit(X_train_rfe, y_train).score(X_test_rfe, y_test)
print("Random Forest Accuracy on test set without select: {:.3f}".format(score))
print("Random Forest Accuracy on test selection: {:.3f}".format(select.score(X_test, y_test)))

from sklearn.feature_selection import SelectKBest 
from sklearn.feature_selection import chi2
X_new = SelectKBest(chi2, k=10).fit_transform(X_train, y_train)
Y_new = SelectKBest(chi2, k=10).fit_transform(X_test, y_test)
X_train_rfe = X_new
X_test_rfe = Y_new
#Logistic Regression
score = LogisticRegression().fit(X_train_rfe, y_train).score(X_test_rfe, y_test)
print("Logistic Regression Test score without select: {:.3f}".format(score))
print("Logistic Regression Test score with selection: {:.3f}".format(select.score(X_test, y_test)))
print("\n")

#decision tree
score = DecisionTreeClassifier(max_depth=10, random_state=0).fit(X_train_rfe, y_train).score(X_test_rfe, y_test)
print("Decision Tree Accuracy on test set without select: {:.3f}".format(score))
print("Decision Tree Accuracy on test selection: {:.3f}".format(select.score(X_test, y_test)))
print("\n")
#random forest
forest = RandomForestClassifier(n_estimators=100, random_state=0).fit(X_train_rfe, y_train).score(X_test_rfe, y_test)
print("Random Forest Accuracy on test set without select: {:.3f}".format(score))
print("Random Forest Accuracy on test selection: {:.3f}".format(select.score(X_test, y_test)))

fig, axes = plt.subplots(figsize=(8, 8))
training_accuracy = []
test_accuracy = []
# try n_neighbors from 1 to 25
nb = range(1, 25)

for n_neighbors in nb:
    
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X_train, y_train)
    training_accuracy.append(clf.score(X_train, y_train))
    test_accuracy.append(clf.score(X_test, y_test))
    
plt.plot(nb, training_accuracy, label="training accuracy")
plt.plot(nb, test_accuracy, label="test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()  
train_data.loc[:,train_data.columns[0:10]]
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(selected_features.loc[:,selected_features.columns != 'CARAVAN']  ,selected_features.loc[:,selected_features.columns == 'CARAVAN'])
predictions=knn.predict(test_data.loc[:,selected_features.columns!='CARAVAN'])
print("Accuracy on training set: {:.2f}".format(knn.score(selected_features.loc[:,selected_features.columns != 'CARAVAN'], selected_features.loc[:,selected_features.columns == 'CARAVAN'])))
print("Accuracy on test set: {:.2f}".format(knn.score(test_data.loc[:,selected_features.columns!='CARAVAN'], test_data.loc[:,selected_features.columns=='CARAVAN'])))
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(train_data.loc[:,train_data.columns[0:10]]  ,selected_features.loc[:,selected_features.columns == 'CARAVAN'])
predictions=knn.predict(test_data.loc[:,test_data.columns[0:10]])
print("Accuracy on training set: {:.2f}".format(knn.score(train_data.loc[:,train_data.columns[0:10]], selected_features.loc[:,selected_features.columns == 'CARAVAN'])))
print("Accuracy on test set: {:.2f}".format(knn.score(test_data.loc[:,test_data.columns[0:10]], test_data.loc[:,selected_features.columns=='CARAVAN'])))


knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(train_data,train_target)
predictions=knn.predict(test_data)
print("Accuracy on training set: {:.2f}".format(knn.score(train_data, train_target)))
print("Accuracy on test set: {:.2f}".format(knn.score(test_data, test_target)))
print("Mean square error " , metrics.mean_squared_error(test_target, predictions))
print("Precision " , metrics.precision_score(test_target, predictions))

fig = plt.figure(figsize=(10,10))
plt.subplot(3,1,1)
test_target['CARAVAN'].value_counts().plot(kind='bar', title='Classifying CARAVAN', color='steelblue', grid=True)
train_with_target.CARAVAN.value_counts()
# Class count
count_class_0, count_class_1 = train_with_target.CARAVAN.value_counts()

# Divide by class
df_class_0 = train_with_target[train_with_target['CARAVAN'] == 0]
df_class_1 = train_with_target[train_with_target['CARAVAN'] == 1]

df_class_0_under = df_class_0.sample(count_class_1)
df_test_under = pd.concat([df_class_0_under, df_class_1], axis=0)

print('Random under-sampling:')
print(df_test_under.CARAVAN.value_counts())

df_test_under.CARAVAN.value_counts().plot(kind='bar', title='Count (target)');

df_test_under.shape
df_class_1_over = df_class_1.sample(count_class_0, replace=True)
df_test_over = pd.concat([df_class_0, df_class_1_over], axis=0)

print('Random over-sampling:')
print(df_test_over.CARAVAN.value_counts())

df_test_over.CARAVAN.value_counts().plot(kind='bar', title='Count (target)');
df_test_over.shape
