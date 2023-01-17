# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline







# Modelling Algorithms

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier



# Modelling Helpers

from sklearn.preprocessing import Imputer , Normalizer , scale , normalize

from sklearn.model_selection import train_test_split , StratifiedKFold

#from sklearn.feature_selection import RFECV

# Helper functions 

# Reference: https://www.kaggle.com/helgejo/an-interactive-data-science-tutorial



def plot_distribution( df , var , target , **kwargs ):

    row = kwargs.get( 'row' , None )

    col = kwargs.get( 'col' , None )

    facet = sns.FacetGrid( df , hue=target , aspect=4 , row = row , col = col )

    facet.map( sns.kdeplot , var , shade= True )

    facet.set( xlim=( 0 , df[ var ].max() ) )

    facet.add_legend()



def plot_categories( df , cat , target , **kwargs ):

    row = kwargs.get( 'row' , None )

    col = kwargs.get( 'col' , None )

    facet = sns.FacetGrid( df , row = row , col = col )

    facet.map( sns.barplot , cat , target )

    facet.add_legend()



def plot_correlation_map( df ):

    corr = df.corr()

    _ , ax = plt.subplots( figsize =( 8 , 7 ) )

    cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )

    _ = sns.heatmap(

        corr, 

        cmap = cmap,

        square=True, 

        cbar_kws={ 'shrink' : 0.5}, 

        ax=ax, 

        annot = True, 

        annot_kws = { 'fontsize' : 11 }

    )





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
titanic_train=pd.read_csv('../input/train.csv')

titanic_test=pd.read_csv('../input/test.csv')
print('Size of training data:  rows:{}, cols:{}'.format(titanic_train.shape[0], titanic_train.shape[1]))

print('Size of test data:      rows:{}, cols:{}'.format(titanic_test.shape[0], titanic_test.shape[1]))

print('')

titanic_train_header = list(titanic_train)

titanic_test_header = list(titanic_test)



print('Train Header:', titanic_train_header)

print('Difference between train and test headers: ', [item for item in titanic_train_header if item not in titanic_test_header ])
titanic_train.head()
titanic_train.sample(10)
titanic_train.describe()
# Check how many unique values exist

titanic_train['Sex'].unique()
titanic_train["SexBin"] = (titanic_train.Sex == 'female').astype(int)

titanic_train.describe()
corr = titanic_train.corr()

cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )

_ = sns.heatmap(corr, 

            cmap=cmap, 

            annot=True)
plot_correlation_map(titanic_train)
#plot_distribution( titanic_train , var = 'Age' , target = 'Survived' , row = 'Sex' )

facet = sns.FacetGrid(titanic_train, col="Sex", row='Survived')

#_ = facet.map(sns.kdeplot , 'Age' , shade= True )

facet.map(plt.hist, "Age", density=True)
df_female = titanic_train.query('SexBin == 1')

plot_correlation_map(df_female)
df_male = titanic_train.query('SexBin == 0')

plot_correlation_map(df_male)
# Plot survival rate by Embarked

plot_categories( titanic_train , cat = 'Embarked' , target = 'Survived' )

plot_categories( df_male , cat = 'Embarked' , target = 'Survived' )

plot_categories( df_female , cat = 'Embarked' , target = 'Survived' )
plot_categories( titanic_train , cat = 'Pclass' , target = 'Survived' )

plot_categories( df_male , cat = 'Pclass' , target = 'Survived' )

plot_categories( df_female , cat = 'Pclass' , target = 'Survived' )
full_data = pd.concat([titanic_train.drop('Survived', axis=1).drop('SexBin', axis=1), titanic_test], axis = 0, sort=False) 

full_data.shape

full_data.sample(5)
sex = (full_data.Sex == 'female').astype(int) # Male = 0, Female = 1
# Create a new variable for every unique value of Embarked

#embarked = pd.get_dummies( full_data.Embarked , prefix='Embarked' ).idxmax(1)

embarked_list = full_data.Embarked.unique()

embarked_dict = pd.Series(range(len(embarked_list)), embarked_list)

print(embarked_dict)

embarked = pd.get_dummies( full_data.Embarked).idxmax(1)

embarked_id = embarked.map(embarked_dict).rename('Embarked').to_frame()

embarked.head()
cabin_num = full_data.Cabin.fillna('U')

cabin_num.head()
cabin  = cabin_num.str[0]

cabin_list = cabin.unique()

cabin_dict = pd.Series(range(len(cabin_list)), cabin_list)

print(cabin_dict)

cabin_id = pd.get_dummies( cabin ).idxmax(1)

cabin_id = cabin_id.map(cabin_dict).rename('Cabin').to_frame()

cabin_id.sample(5)
full_data.Age = full_data.Age.fillna(full_data.Age.mean())

full_data.Fare = full_data.Fare.fillna(full_data.Fare.mean())

full_data.isna().sum()
# Reference: https://medium.com/dunder-data/selecting-subsets-of-data-in-pandas-6fcd0170be9c

age = full_data.Age

age_sq = (age*age).rename('Age_sq')  # rename the series

age_sex = ((sex*2-1)*age).rename('Age_sex')

#embark_sex = ((sex*2-1)*embarked_id).rename('embark_sex')

class_sex = ((sex*2-1)*full_data.Pclass).rename('Class_sex')



full_x = pd.concat([full_data[['Pclass', 'Age', 'SibSp', 'Parch']], age_sq, sex, age_sex, class_sex, cabin_id, embarked_id], axis=1)





# Min-max normalization

full_x = (full_x - full_x.min())/(full_x.max()-full_x.min())

#full_x.Sex = full_x.Sex*2 - 1



train_valid_x = full_x[0:titanic_train.shape[0]]

train_valid_y = titanic_train['Survived']



test_x = full_x[titanic_train.shape[0]:]



train_valid_x.head()



train_x, valid_x, train_y, valid_y = train_test_split(train_valid_x, train_valid_y, train_size=0.8)
model_lr = LogisticRegression()

model_rf = RandomForestClassifier()

model_dt = DecisionTreeClassifier()

model_svc = LinearSVC()

model_knn = KNeighborsClassifier()

model_lr.fit(train_x, train_y)

model_rf.fit(train_x, train_y)  

model_dt.fit(train_x, train_y)

model_svc.fit(train_x, train_y)

model_knn.fit(train_x, train_y)



print('LR Scores:  Training:{} \t Valid:{}'.format(model_lr.score(train_x, train_y), model_lr.score(valid_x, valid_y)))

print('RF Scores:  Training:{} \t Valid:{}'.format(model_rf.score(train_x, train_y), model_rf.score(valid_x, valid_y)))

print('DT Scores:  Training:{} \t Valid:{}'.format(model_dt.score(train_x, train_y), model_dt.score(valid_x, valid_y)))

print('SVC Scores:  Training:{} \t Valid:{}'.format(model_svc.score(train_x, train_y), model_svc.score(valid_x, valid_y)))

print('KNN Scores:  Training:{} \t Valid:{}'.format(model_knn.score(train_x, train_y), model_knn.score(valid_x, valid_y)))
from sklearn.model_selection import cross_val_score

scores_rf = cross_val_score(model_rf, train_valid_x, train_valid_y, cv=10)

scores_knn = cross_val_score(model_knn, train_valid_x, train_valid_y, cv=10)



print('RF \t Mean: {} Std:{}'.format(np.mean(scores_rf), np.std(scores_rf)))

print('KNN \t Mean: {} Std:{}'.format(np.mean(scores_knn), np.std(scores_svc)))





print(train_valid_x.columns.shape)

print(model_rf.feature_importances_)



importances = pd.DataFrame({'feature':train_valid_x.columns,'importance':np.round(model_rf.feature_importances_,3)})

print(importances.sort_values(by=['importance'], ascending=False))

from sklearn.model_selection import cross_val_predict

from sklearn.metrics import confusion_matrix

predictions = cross_val_predict(model_rf, train_valid_x, train_valid_y, cv=10)

print(' Confusion Matrix \n', confusion_matrix(train_valid_y, predictions))



print('\n Weight of training set:', np.sum(train_valid_y==1)/len(train_valid_y))
#model_rf_improved = RandomForestClassifier()

model_rf_improved = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=42)

print('Weights:{}'.format(model_rf.class_weight))



model_rf_improved.fit(train_valid_x, train_valid_y)

print('oob_score:{}'.format(model_rf_improved.oob_score_))



scores_rf_improved = cross_val_score(model_rf_improved, train_valid_x, train_valid_y, cv=10)



min_estimators = 10

max_estimators = 200



error_rate = {}



for i in range(min_estimators, max_estimators + 1, 5):

    model_rf_improved.set_params(n_estimators=i)

    model_rf_improved.fit(train_valid_x, train_valid_y)



    oob_error = 1 - model_rf_improved.oob_score_

    error_rate[i] = oob_error

    

    #y_scores = model_rf_improved.predict_proba(train_valid_x)

    #y_scores = y_scores[:,1]



    #precision, recall, threshold = precision_recall_curve(train_valid_y, y_scores)

# Convert dictionary to a pandas series for easy plotting 

oob_series = pd.Series(error_rate)

fig, ax = plt.subplots(figsize=(10, 10))



oob_series.plot(kind='line',

                color = 'red')

plt.xlabel('n_estimators')

plt.ylabel('OOB Error Rate')

plt.title('OOB Error Rate Across various Forest sizes \n(From 15 to 1000 trees)')
model_rf_improved = RandomForestClassifier(n_estimators=125, oob_score=True, random_state=42, warm_start=True)

model_rf_improved.fit(train_valid_x, train_valid_y)
from sklearn.metrics import precision_recall_curve



# getting the probabilities of our predictions

y_scores = model_rf_improved.predict_proba(train_valid_x)

y_scores = y_scores[:,1]



precision, recall, threshold = precision_recall_curve(train_valid_y, y_scores)
def plot_precision_and_recall(precision, recall, threshold):

    plt.plot(threshold, precision[:-1], "r-", label="precision", linewidth=5)

    plt.plot(threshold, recall[:-1], "b", label="recall", linewidth=5)

    plt.xlabel("threshold", fontsize=19)

    plt.legend(loc="upper right", fontsize=19)

    plt.ylim([0, 1])



plt.figure(figsize=(14, 7))

plot_precision_and_recall(precision, recall, threshold)

plt.show()
def plot_precision_vs_recall(precision, recall):

    plt.plot(recall, precision, "g--", linewidth=2.5)

    plt.ylabel("recall", fontsize=19)

    plt.xlabel("precision", fontsize=19)

    plt.axis([0, 1.5, 0, 1.5])



plt.figure(figsize=(14, 7))

plot_precision_vs_recall(precision, recall)

plt.show()
test_data = full_data[titanic_train.shape[0]:]



threshold = 0.4



predicted = model_rf_improved.predict_proba(test_x)

#predicted [:,0] = (predicted [:,0] < threshold).astype('int')

#predicted [:,1] = (predicted [:,1] >= threshold).astype('int')



Y_prediction_default = model_rf_improved.predict(test_x)



Y_prediction = (predicted[:, 0] <= threshold).astype('int')



#print('Prediction probabilities which were changed by setting a different threshold')

#print(predicted[Y_prediction!=Y_prediction_default])





submission = pd.DataFrame({

        "PassengerId": test_data["PassengerId"],

        "Survived": Y_prediction

    })

submission.to_csv('submission.csv', index=False)