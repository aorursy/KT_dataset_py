import pandas as pd 

from sklearn.model_selection import cross_val_score



df_train = pd.read_csv('/kaggle/input/titanic/train.csv')

df_test = pd.read_csv('/kaggle/input/titanic/test.csv')



combined = df_train.append(df_test, ignore_index=True, sort=False)
df_train.head()
df_test.head()
def missing_values_table(df):

        mis_val = df.isnull().sum()

        

        mis_val_percent = 100 * df.isnull().sum() / len(df)



        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)



        mis_val_table_ren_columns = mis_val_table.rename(

        columns = {0 : 'Missing Values', 1 : '% of Total Values'})



        mis_val_table_ren_columns = mis_val_table_ren_columns[

            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(

        '% of Total Values', ascending=False).round(1)



        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      

            "There are " + str(mis_val_table_ren_columns.shape[0]) +

              " columns that have missing values.")

        

        return mis_val_table_ren_columns



print('\nUnique Values')

print(df_train.nunique(),'\n')

print(df_train.info(),'\n')

missing_values_table(combined)
df_train.describe()
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



fig = plt.figure(figsize=(15,15))

fig.suptitle('Distribution of features in my dataset', fontsize=16)



plt.subplot2grid((3,3),(0,0))

ax = sns.countplot(x="Survived", data=df_train)

for p in ax.patches:

    x=p.get_bbox().get_points()[:,0]

    y=p.get_bbox().get_points()[1,1]

    ax.annotate('{:.1f}%'.format(100.*y/len(df_train)), (x.mean(), y), ha='center', va='bottom') 

plt.title('Survived')



plt.subplot2grid((3,3),(0,1))

ax = sns.countplot(x="Embarked", data=df_train)

for p in ax.patches:

    x=p.get_bbox().get_points()[:,0]

    y=p.get_bbox().get_points()[1,1]

    ax.annotate('{:.1f}%'.format(100.*y/len(df_train)), (x.mean(), y), ha='center', va='bottom') 

plt.title('Embarked')



plt.subplot2grid((3,3),(0,2))

ax = sns.countplot(x="Pclass", data=df_train)

for p in ax.patches:

    x=p.get_bbox().get_points()[:,0]

    y=p.get_bbox().get_points()[1,1]

    ax.annotate('{:.1f}%'.format(100.*y/len(df_train)), (x.mean(), y), ha='center', va='bottom') 

plt.title('Pclass')



plt.subplot2grid((3,3),(1,0))

df_train.Age.hist(grid=False)

plt.title('Age')



plt.subplot2grid((3,3),(1,1))

ax = sns.countplot(x="SibSp", data=df_train)

for p in ax.patches:

    x=p.get_bbox().get_points()[:,0]

    y=p.get_bbox().get_points()[1,1]

    ax.annotate('{:.1f}%'.format(100.*y/len(df_train)), (x.mean(), y), ha='center', va='bottom') 

plt.title('SibSp')



plt.subplot2grid((3,3),(1,2))

ax = sns.countplot(x="Parch", data=df_train)

for p in ax.patches:

    x=p.get_bbox().get_points()[:,0]

    y=p.get_bbox().get_points()[1,1]

    ax.annotate('{:.1f}%'.format(100.*y/len(df_train)), (x.mean(), y), ha='center', va='bottom') 

plt.title('Parch')



plt.subplot2grid((3,3),(2,0))

df_train.Fare.hist(grid=False)

plt.title('Fare')
import numpy as np



corr = df_train.corr()

corr.style.background_gradient(cmap='coolwarm')
fig = plt.figure(figsize=(20,15))

fig.suptitle('Various relationships of survival rate', fontsize=16)



plt.subplot2grid((3,2),(0,0))

sns.scatterplot(x="Survived", y="Age", data=df_train,alpha = 0.1) 

plt.title('Age <> Survived')



plt.subplot2grid((3,2),(0,1))

for c in sorted(df_train.Pclass.unique()):

    sns.kdeplot(df_train.Age[df_train.Pclass == c],  label="%d class"%c)

plt.title('Age <> Class')



plt.subplot2grid((2,2),(1,0))

sns.barplot(x="Pclass", y="Survived", data=df_train)

plt.title('Class <> Survived')



plt.subplot2grid((2,2),(1,1))

sns.barplot(x="Embarked", y="Survived", data=df_train)

plt.title('Embarked <> Survived')



plt.show()
fig = plt.figure(figsize=(15,15))

fig.suptitle('Man,woman relationship wrt. wealth survival rate', fontsize=16)



plt.subplot2grid((2,2),(0,0))

ax =sns.countplot("Sex",hue="Survived", data=df_train)

for p in ax.patches:

    x=p.get_bbox().get_points()[:,0]

    y=p.get_bbox().get_points()[1,1]

    ax.annotate('{:.1f}%'.format(100.*y/len(df_train)), (x.mean(), y), ha='center', va='bottom') 



plt.subplot2grid((2,2),(0,1))

sns.barplot(x='Sex',y='Survived',hue="Pclass", data=df_train)

plt.title('Sex+class <> Survived')



plt.subplot2grid((2,2),(1,0))

sns.barplot(x='Sex',y='Survived', data=df_train)

plt.title('Sex <> Survived')



plt.subplot2grid((2,2),(1,1))

sns.barplot(x='Sex',y='Survived',hue="Embarked", data=df_train)

plt.title('Sex+class <> Survived')



plt.show()
imputer = SimpleImputer(strategy='median')

imputed = combined[['Age','Fare']]._get_numeric_data()

imputed = pd.DataFrame(imputer.fit_transform(imputed),columns=['Age', 'Fare'])

imputer.statistics_

imputed.head()
sex = pd.Series( np.where( combined.Sex == 'male' , 1 , 0 ) , name = 'Sex' )

sex.head()
embarked = pd.get_dummies(combined.Embarked , prefix='Embarked')

embarked.head()
pclass = pd.get_dummies(combined.Pclass , prefix='Pclass')

pclass.head()
title = pd.DataFrame()



title['Title'] = combined[ 'Name' ].map( lambda name: name.split( ',' )[1].split( '.' )[0].strip() )

title.Title.unique()
title_dict = { 'Mr':'Mr', 'Mrs':'Mrs', 'Miss':'Miss', 'Master':'Master', 'Don':'Royalty', 'Rev':'Officier', 'Dr':'Officier',

              'Mme':'Mrs', 'Ms':'Mrs', 'Major': 'Officier', 'Lady':'Royalty', 'Sir':'Royalty', 'Mlle':'Mrs', 

              'Col': 'Officier', 'Capt':'Officier', 'the Countess': 'Royalty','Jonkheer': 'Royalty', 'Dona': 'Royalty' }

title['Title'] = title['Title'].map(title_dict)

title = pd.get_dummies(title['Title'])

title.head()
cabin = pd.DataFrame()



# replacing missing cabins with U (for Uknown)

cabin['Cabin'] = combined.Cabin.fillna( 'U' )



# mapping each Cabin value with the cabin letter

cabin['Cabin'] = cabin[ 'Cabin' ].map( lambda c : c[0] )



# dummy encoding ...

cabin = pd.get_dummies( cabin['Cabin'] , prefix = 'Cabin' )



cabin.head()
# a function that extracts each prefix of the ticket, returns 'XXX' if no prefix (i.e the ticket is a digit)

def cleanTicket( ticket ):

    ticket = ticket.replace( '.' , '' )

    ticket = ticket.replace( '/' , '' )

    ticket = ticket.split()

    ticket = map( lambda t : t.strip() , ticket )

    ticket = list(filter( lambda t : not t.isdigit() , ticket ))

    if len( ticket ) > 0:

        return ticket[0]

    else: 

        return 'XXX'



ticket = pd.DataFrame()



# Extracting dummy variables from tickets:

ticket[ 'Ticket' ] = combined[ 'Ticket' ].map( cleanTicket )

ticket = pd.get_dummies( ticket[ 'Ticket' ] , prefix = 'Ticket' )



ticket.shape

ticket.head()
family = pd.DataFrame()



# introducing a new feature : the size of families (including the passenger)

family[ 'FamilySize' ] = combined[ 'Parch' ] + combined[ 'SibSp' ] + 1



# introducing other features based on the family size

family[ 'Family_Single' ] = family[ 'FamilySize' ].map( lambda s : 1 if s == 1 else 0 )

family[ 'Family_Small' ]  = family[ 'FamilySize' ].map( lambda s : 1 if 2 <= s <= 4 else 0 )

family[ 'Family_Large' ]  = family[ 'FamilySize' ].map( lambda s : 1 if 5 <= s else 0 )



family.head()
# Select which features/variables to include in the dataset from the list below:

# imputed , embarked , pclass , sex , family , cabin , ticket

# 0.76555 - imputed , pclass,embarked , family[['Family_Single','Family_Small','Family_Large']],title

# 0.77990 - imputed, pclass,embarked ,family['FamilySize'],title not poly

full_X = pd.concat( [ imputed, pclass,embarked ,family['FamilySize'],title ], axis=1 )

full_X.head()
from sklearn.model_selection import train_test_split



X = full_X[:len(df_train)]

Y = df_train.Survived



test_X = full_X[891:]



train_X , valid_X , train_y , valid_y = train_test_split( X , Y , train_size = .75 )





print (full_X.shape , train_X.shape , valid_X.shape , train_y.shape , valid_y.shape , test_X.shape)
def plot_model_var_imp( model , X , y ):

    imp = pd.DataFrame( 

        model.feature_importances_  , 

        columns = [ 'Importance' ] , 

        index = X.columns 

    )

    imp = imp.sort_values( [ 'Importance' ] , ascending = True )

    imp[ : 10 ].plot( kind = 'barh' )

    print (model.score( X , y ))
train_X
from sklearn.preprocessing import PolynomialFeatures



poly = PolynomialFeatures(degree=2)

poly_train_X = poly.fit_transform(train_X)

poly_valid_X = poly.fit_transform(valid_X)

poly_test_X = poly.fit_transform(test_X)
from sklearn.linear_model import LogisticRegression



log_reg = LogisticRegression(solver='lbfgs',max_iter=10000)

log_reg.fit(train_X, train_y )

acc_log = round(log_reg.score( train_X , train_y )) , round(log_reg.score( valid_X , valid_y ) * 100, 2)

print(acc_log)

log_reg.fit(poly_train_X, train_y )

acc_log_poly = round(log_reg.score( poly_train_X , train_y )) , round(log_reg.score( poly_valid_X , valid_y ) * 100, 2)

acc_log_poly
cross_val_score(log_reg, poly_train_X, train_y, scoring='accuracy',cv=15).mean()
tree = DecisionTreeClassifier( random_state = 99 )

tree.fit(train_X, train_y )

plot_model_var_imp(tree, train_X, train_y)



acc_tree = round(tree.score( train_X , train_y )) , round(tree.score( valid_X , valid_y ) * 100, 2)

print(acc_tree)

tree.fit(poly_train_X, train_y )

acc_tree_poly = round(tree.score( poly_train_X , train_y )) , round(tree.score( poly_valid_X , valid_y ) * 100, 2)

acc_tree_poly
cross_val_score(tree, poly_train_X, train_y, scoring='accuracy',cv=15).mean()
from sklearn.ensemble import RandomForestClassifier



forest = RandomForestClassifier(n_estimators=100)

forest.fit(train_X, train_y )

plot_model_var_imp(forest, train_X, train_y)



acc_forest = round(forest.score( train_X , train_y )) , round(forest.score( valid_X , valid_y ) * 100, 2)

print(acc_forest)



forest.fit(poly_train_X, train_y )

acc_forest_poly = round(forest.score( poly_train_X , train_y )) , round(forest.score( poly_valid_X , valid_y ) * 100, 2)

acc_forest_poly
cross_val_score(forest, poly_train_X, train_y, scoring='accuracy',cv=15).mean()
from sklearn import svm



svm_ = svm.LinearSVC()

svm_.fit(train_X, train_y )
acc_svm_ = round(svm_.score( train_X , train_y )) , round(svm_.score( valid_X , valid_y ) * 100, 2)



print(acc_svm_)



svm_.fit(poly_train_X, train_y )

acc_svm_poly = round(svm_.score( poly_train_X , train_y )) , round(svm_.score( poly_valid_X , valid_y ) * 100, 2)

acc_svm_poly
cross_val_score(svm_, poly_train_X, train_y, scoring='accuracy',cv=15).mean()
from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(train_X, train_y )
acc_knn = round(knn.score( train_X , train_y )) , round(knn.score( valid_X , valid_y ) * 100, 2)



print(acc_knn)



knn.fit(poly_train_X, train_y )

acc_knn_poly = round(knn.score( poly_train_X , train_y )) , round(knn.score( poly_valid_X , valid_y ) * 100, 2)

acc_knn_poly
cross_val_score(knn, poly_train_X, train_y, scoring='accuracy',cv=15).mean()
from sklearn.naive_bayes import GaussianNB



gnb = GaussianNB()

gnb.fit(train_X,train_y)
acc_gnb = round(gnb.score( train_X , train_y )) , round(gnb.score( valid_X , valid_y ) * 100, 2)



print(acc_gnb)



gnb.fit(poly_train_X, train_y )

acc_gnb_poly = round(gnb.score( poly_train_X , train_y )) , round(gnb.score( poly_valid_X , valid_y ) * 100, 2)

acc_gnb_poly
cross_val_score(gnb, poly_train_X, train_y, scoring='accuracy',cv=15).mean()
import xgboost as xgb

gbm = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05)

gbm.fit(train_X , train_y)
acc_gbm = round(gbm.score( train_X , train_y )) , round(gbm.score( valid_X , valid_y ) * 100, 2)



print(acc_gbm)



# gbm.fit(poly_train_X, train_y )

# acc_gbm_poly = round(gbm.score( poly_train_X , train_y )) , round(gbm.score( poly_valid_X , valid_y ) * 100, 2)

# acc_gbm_poly
cross_val_score(gbm, train_X, train_y, scoring='accuracy',cv=15).mean()
from sklearn.feature_selection import RFECV

from sklearn.model_selection import  StratifiedKFold



rfecv = RFECV( estimator = gbm , step = 1,cv=StratifiedKFold(2)  , scoring = 'accuracy' )

rfecv.fit( train_X , train_y )
print (rfecv.score( train_X , train_y ) , rfecv.score( valid_X , valid_y ))

print( "Optimal number of features : %d" % rfecv.n_features_ )



#Plot number of features VS. cross-validation scores

plt.figure()

plt.xlabel( "Number of features selected" )

plt.ylabel( "Cross validation score (nb of correct classifications)" )

plt.plot( range( 1 , len( rfecv.grid_scores_ ) + 1 ) , rfecv.grid_scores_ )

plt.show()
test_Y = rfecv.predict( test_X )

passenger_id = combined[891:].PassengerId

test = pd.DataFrame( { 'PassengerId': passenger_id , 'Survived': test_Y } )

test.shape

test.head()

test.to_csv( 'titanic_pred.csv' , index = False )