import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt 
#importing data

# ds --> dataset

ds = pd.read_csv("/kaggle/input/titanic/train.csv")
# getting the first ten rows of dataset

ds.head(10)
ds.info()
Y = ds.Survived

# drop the columns

X = ds.drop(['PassengerId','Name','Survived'],axis = 1)
print(X.isnull().sum())
# description

print(X.Age.describe())



# plus histogram

fig, ax = plt.subplots(figsize= (5,5))

ax.hist(x= X.Age,color= "blue",edgecolor = 'black')

plt.show()
X.columns
X.Sex.describe()
from sklearn.compose import ColumnTransformer # transforming

from sklearn.preprocessing import OrdinalEncoder # encoding

X_temp = X



Sex_Embarked = {"Sex":{"male": 1.,"female": 0.},

               "Embarked":{"S": 0.,"C": 1.,"Q": 2.}}

X_temp = X_temp.replace(Sex_Embarked, inplace=False)



CT = ColumnTransformer(transformers = [('encoder',OrdinalEncoder(),['Ticket'])],remainder = 'passthrough')

X_temp = pd.DataFrame(CT.fit_transform(X_temp.drop(['Cabin'],axis = 1,inplace = False).dropna(subset = ['Age','Embarked'],axis =0 ,inplace = False)),

                      columns=[ 'Ticket', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare','Embarked'],

                     dtype = 'float')
X_temp
import seaborn as sb

fig,ax = plt.subplots(figsize = (10,10))

sb.heatmap(pd.concat([X_temp,Y],axis=1).dropna().corr(),annot = True,linewidth = 0.5)
sb.set(style="whitegrid", palette="muted")



fig = plt.figure(figsize=(10,10))



ax = sb.swarmplot(x='Sex', y="Age", hue = "Survived",data=ds,size=6)



plt.show()
from sklearn.impute import SimpleImputer # missing values

imputer = SimpleImputer(missing_values=np.nan, strategy = 'mean')

ds.Age = imputer.fit_transform(ds.Age.values.reshape(-1,1))

X.Age = imputer.fit_transform(X.Age.values.reshape(-1,1))
X.Age
ds.Embarked = SimpleImputer(missing_values=np.nan, strategy = 'most_frequent').fit_transform(ds.Embarked.values.reshape(-1,1))

X.Embarked = SimpleImputer(missing_values=np.nan, strategy = 'most_frequent').fit_transform(X.Embarked.values.reshape(-1,1))
# get more information of the cabin attribute

X.Cabin.describe()
# what are the unique values of this attribute?

X.Cabin.unique()
# for instance, we'll get the not-null values of cabins whose names start with "C".

# which means those cabins belong to deck C.

X_Cabins = X.Cabin.dropna().str.contains('C',regex = False)

X_Cabins = X.Cabin[X_Cabins.loc[X_Cabins == True].index]

# X_Cabins = X.Cabin.dropna().str.extract(r'(^C.+)')

print(X_Cabins)
X_Cabins.describe()
# add a new column called Deck to our dataset

ds = ds.assign(Deck = lambda x: x.Cabin.str.extract(r'(.)'))

# reassign it to X

X = ds.drop(['PassengerId','Name','Survived'],axis = 1)

ds.Deck.describe()
import seaborn as sb





order = ['A','B','C','D','E','F','G','T']



fig, ax1 = plt.subplots()

fig.set_size_inches(11, 8)



sb.set(style = 'darkgrid')



ax = sb.countplot(ax =ax1 ,data=ds[ds.Deck.notnull()],x="Deck",hue ="Survived",

                  palette="dark",alpha=.6, order = order)

bars = ax.patches

half = int(len(bars)/2)

left_bars = bars[:half]

right_bars = bars[half:]



for left, right in zip(left_bars, right_bars):

    height_l = left.get_height()

    height_r = right.get_height()

    total = height_l + height_r



    ax.text(left.get_x() + left.get_width()/2., height_l, '{0:.0%}'.format(height_l/total), ha="center")

    ax.text(right.get_x() + right.get_width()/2., height_r, '{0:.0%}'.format(height_r/total), ha="center")





plt.show()
X = X.drop(['Cabin'],axis = 1)

X
X_temp = X



Sex_Embarked = {"Sex":{"male": 1.,"female": 0.},

               "Embarked":{"S": 0.,"C": 1.,"Q": 2.},

               "Deck":{"A": 1.,"B": 2.,"C": 3.,

                      "D": 4.,"E": 5.,"F": 6.,

                      "G": 7.,"T": 8.}}



X_temp = X_temp.replace(Sex_Embarked, inplace=False)



X_temp = pd.DataFrame(CT.fit_transform(X_temp),

                      columns=[ 'Ticket', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare','Embarked','Deck'],

                     dtype = 'float')
X_temp
fig,ax = plt.subplots(figsize = (10,10))

sb.heatmap(pd.concat([X_temp,Y],axis=1).dropna().corr(),annot = True,linewidth = 0.5)
X_temp = X_temp.drop(['Deck'],axis = 1)

X_temp
fig,ax = plt.subplots(figsize = (10,10))

sb.heatmap(pd.concat([X_temp,Y],axis=1).corr(),annot = True,fmt = '0.1',linewidth = 1)
# assign zero if they are alone

X_temp['family_Size'] = X_temp.Parch + X_temp.SibSp 

X_temp = X_temp.drop(['Parch','SibSp'],axis = 1)
X_temp
#mean of the Fare of the tickets for each class

mean = ['{0}, {1:.0f}'.format(pclass,np.nanmean(X.where(X.Pclass == pclass,inplace = False).Fare)) for pclass in X.Pclass.unique()]

print(mean)
# fig, ax = plt.subplots(1,2,figsize=(10,10))

# sb.set_theme(style="whitegrid")

fig, ax1 = plt.subplots(1,2)

fig.set_size_inches(15, 10)



sb.violinplot(ax = ax1[0],data=ds,x="Pclass",y ="Fare",hue="Survived",

              split=True, inner="quart", linewidth=1,

              palette={0: "b", 1: "r"},scale="area",saturation = 1)



sb.countplot(ax =ax1[1] ,data=ds,x="Pclass",hue ="Survived",

                  palette={0: "b", 1: "r"},alpha=1)





plt.show()
X_temp = X_temp.drop(['Ticket'],axis = 1)
X_temp
from sklearn.linear_model import LogisticRegression

from sklearn.feature_selection import SelectFromModel

from sklearn.model_selection import train_test_split



X_train, X_valid, Y_train, Y_valid = train_test_split(X_temp,Y,test_size = 0.25,random_state = 1)



# Set the regularization parameter C=1

logistic = LogisticRegression(C=1, penalty="l1", solver='liblinear', random_state=7).fit(X_train,Y_train )

model = SelectFromModel(logistic, prefit=True)



X_new = model.transform(X_train)

X_new
# Get back the kept features as a DataFrame with dropped columns as all 0s

selected_features = pd.DataFrame(model.inverse_transform(X_new), 

                                 index=X_train.index,

                                 columns=X_train.columns)



# Dropped columns have values of all 0s, keep other columns 

selected_columns = selected_features.columns[selected_features.var() != 0]

selected_columns
from sklearn.metrics import confusion_matrix, accuracy_score # estimating the model

cm2 = confusion_matrix(Y_valid,logistic.predict(X_valid))

log_acc = accuracy_score(Y_valid,logistic.predict(X_valid))

print(log_acc,'\n',cm2)
from sklearn.preprocessing import  MinMaxScaler# scaling

sc = MinMaxScaler()

scaled_X_temp = pd.DataFrame(sc.fit_transform(X_temp),columns = X_temp.columns)

scaled_X_temp
from sklearn.neighbors import KNeighborsClassifier



X_train, X_valid, Y_train, Y_valid = train_test_split(scaled_X_temp,Y,test_size = 0.25,random_state = 1)



knn_Classifier = KNeighborsClassifier(n_neighbors = 7, metric = 'minkowski', p=2)

knn_Classifier.fit(X_train,Y_train)



#predicting

Y_pred_knn = knn_Classifier.predict(X_valid)



#estimating

cm2 = confusion_matrix(Y_valid,Y_pred_knn)

knn_acc = accuracy_score(Y_valid,Y_pred_knn)

print(knn_acc)

print(cm2)
from sklearn.model_selection import cross_val_predict # used to predict

from sklearn.model_selection import cross_val_score #use to get the accuracy

cv_knn_score = cross_val_score(knn_Classifier,scaled_X_temp,Y,cv = 10,scoring='accuracy').mean()

print(cv_knn_score)
from sklearn.svm import SVC



svm_Classifier = SVC(kernel = 'rbf', random_state = 0)



cv_svm_score = cross_val_score(svm_Classifier,scaled_X_temp,Y,cv = 10,scoring='accuracy').mean()

print(cv_svm_score)
from sklearn.naive_bayes import GaussianNB

nb_Classifier = GaussianNB()



cv_nb_score = cross_val_score(nb_Classifier,scaled_X_temp,Y,cv = 10,scoring='accuracy').mean()

print(cv_nb_score)
from sklearn.tree import DecisionTreeClassifier

dt_Classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)



cv_dt_score = cross_val_score(dt_Classifier,scaled_X_temp,Y,cv = 10,scoring='accuracy').mean()

print(cv_dt_score)
from sklearn.ensemble import RandomForestClassifier



rf_Classifier = RandomForestClassifier(n_estimators=10, max_depth=None, random_state=0)

cv_rf_score = cross_val_score(rf_Classifier,scaled_X_temp, Y,cv = 8,scoring='accuracy').mean()

print(cv_rf_score)
from xgboost import XGBClassifier



XGB_Classifier = XGBClassifier(n_estimators = 1000,learning_rate = 0.01)

cv_XGB_score = cross_val_score(XGB_Classifier,scaled_X_temp, Y,cv = 18,scoring='accuracy').mean()

print(cv_XGB_score)
from sklearn.ensemble import VotingClassifier



eclf = VotingClassifier(

    estimators=[('xgb',XGB_Classifier), ('lr', logistic), ('rf', rf_Classifier), ('dt', dt_Classifier),('svm',svm_Classifier)],

    voting='hard'

)



# let's see all of our models accuracy til now

for clf, label in zip([XGB_Classifier, logistic, rf_Classifier, dt_Classifier,svm_Classifier, eclf], 

                      ['XGBooster Classifier', 'Logistic Regression', 'Random Forest', 'Decision Tree','SVM Classifier', 'Ensemble']):

    scores = cross_val_score(clf, scaled_X_temp, Y, scoring='accuracy', cv=10)

    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))

    
from sklearn.model_selection import RandomizedSearchCV

params = {'lr__C': [1.0, 100.0], 'rf__n_estimators': [20, 200],

          'xgb__n_estimators': [500,2000],'xgb__learning_rate': [0.01,0.1]}

rands = RandomizedSearchCV(estimator=eclf, param_distributions=params, cv=5)

rands = rands.fit(scaled_X_temp, Y)

#predicting

Y_pred_rands = rands.predict(X_valid)



#estimating

rands_cm = confusion_matrix(Y_valid,Y_pred_rands)

rands_acc = accuracy_score(Y_valid,Y_pred_rands)

print(rands_acc)

print(rands_cm)
test_set = pd.read_csv("/kaggle/input/titanic/test.csv")

test_set
test_set.info()
test_set.describe()
#Create a copy

test = test_set.copy()



# Creating family_Size

test['family_Size'] = test['SibSp'] + test['Parch']



# Dropping unused features

test = test.drop(['Name','SibSp','Parch','Cabin','Ticket'],axis = 1)



# Encoding

encoded_features = {'Sex' : {'male': 1,'female': 0 },

                   'Embarked':{'S': 0.,'C': 1.,'Q': 2.}}

test.replace(encoded_features, inplace = True)

test
fig,ax = plt.subplots(figsize = (10,10))

sb.heatmap(test.corr(),annot = True,fmt = '0.1',linewidth = 1)
from sklearn.linear_model import LinearRegression

# Filling null-values

# Age --> mean

test.Age = imputer.fit_transform(test.Age.values.reshape(-1,1))

# Fare --> predicting its value with linearregression

notnull_samples = test[test.columns].dropna()



X_set = notnull_samples.loc[:,['Pclass', 'Sex', 'Age', 'Embarked','family_Size']]

Y_set = notnull_samples.loc[:,['Fare']]



linreg = LinearRegression()

linreg.fit(X_set, Y_set)

fare_predict = linreg.predict(test.loc[test.Fare.isnull(),X_set.columns])



test.loc[test.Fare.isnull(),'Fare'] = fare_predict
test.info()
scaled_test = pd.concat([test.loc[:,['PassengerId']],pd.DataFrame(sc.fit_transform(test.drop(['PassengerId'],axis = 1)),

                                                        columns = test.drop(['PassengerId'],axis = 1).columns)],axis = 1)

scaled_test
eclf.fit(scaled_X_temp,Y)

Survived = pd.DataFrame(eclf.predict(scaled_test.drop(['PassengerId'],axis = 1)),columns = ['Survived'])

Survived
submission = pd.concat([scaled_test.loc[:,'PassengerId'],Survived],axis =1)

submission
submission.to_csv('submission.csv', index=False)