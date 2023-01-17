import numpy as np 

import pandas as pd 

from scipy import stats

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn import svm 

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, BaggingClassifier,AdaBoostClassifier

from sklearn.model_selection import train_test_split, GridSearchCV,cross_val_score

from sklearn.neighbors import KNeighborsClassifier  

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split  

from sklearn.preprocessing import StandardScaler  

from sklearn.linear_model import LogisticRegression



%matplotlib inline
train=pd.read_csv('../input/titanic/train.csv')

test=pd.read_csv('../input/titanic/test.csv')

gender_sub=pd.read_csv('../input/titanic/gender_submission.csv')
train.head()
test.head()
train.shape
test.shape
train.info()
test.info()
print('Is there a missing data in train dataset:')

train.isnull().sum()
print('Is there a missing data in test dataset:')

test.isnull().sum()
#return a series containing the counts of all the unique values in a column

train.nunique()
train.describe()
test.describe()
#train dataframe distribution

list_columns=list(train.iloc[:,[5,6,7,9]])



fig, ax = plt.subplots(nrows=1, ncols=4,figsize=(15,3)) 



for i, column in enumerate(list_columns): 

    ax[i].hist(train[column].dropna(),color='skyblue')



fig.suptitle('DATA DISTRIBUATION FOR FOUR COLUMNS\n ', fontsize=16);

print("Skewness of Age: %f" % train['Age'].skew())

print("Skewness of SibSp: %f" % train['SibSp'].skew())

print("Skewness of Parch: %f" % train['Parch'].skew())

print("Skewness of Fare: %f" % train['Fare'].skew())
#Observe the Ranges of some columns (difference between the maximum and minimum)



Age_range=np.ptp(train['Age'])



Fare_range=np.ptp(train['Fare'])



print('The Age range:',Age_range)

print('The Fare range:',Fare_range)
#plot pie to see the percentage of survived column

colors = [ 'lightcoral', 'lightskyblue','gold']



train['Survived'].value_counts().plot.pie( figsize=(10, 5),

     colors=colors,autopct='%1.1f%%', shadow=True,legend=True, startangle=120);



plt.title('ALL PASSENGERS \n',fontsize=17);





fig,ax = plt.subplots(ncols=2,figsize=(13,6))

fig.suptitle('THE PERCENTAGE OF GENDER IN ALL CLASSES ', fontsize=23)



#style the pie plor

colors = ['yellowgreen', 'lightskyblue','gold']

explode = (0,0,0.05)



# groupbed the data based on the gender and the class of the passenger

gender_class=train.groupby(['Sex','Pclass']).agg('count')



#plot the pie to display the percentage all the passengers [female,male] in all the classes

gender_class.iloc[0:3,[0]].plot.pie(subplots=True, colors=colors,autopct='%1.1f%%',

    shadow=True,legend=True, startangle=90,explode = explode,ax=ax[0])



gender_class.iloc[3:,[0]].plot.pie(subplots=True,colors=colors,autopct='%1.1f%%',

    shadow=True, startangle=90,explode = explode,ax=ax[1]);



#filtered the data based on survived passenger

# in the training data: 342 passenger survived from 891 passenger

train[(train['Survived'] == 1)].head()
#136 from class 1 survived

##survived by class percentage

train[(train['Survived'] == 1) & (train['Pclass'] == 1)].head()
# 119  survived from class 3

# poor mr.leo

train[(train['Survived'] == 1) & (train['Pclass'] == 3)].head()


#calculate mean of survived people in each class  

#to know how many survived out of total for each class



# from the table it's seems like the average of survived from class 1 is higher than the other two classes

# survived passengers of class 3 has the lowest average of the three classes

train[['Pclass','Survived']].groupby(['Pclass']).mean()


Sex_male=train[(train['Sex']=='male')]



print('Total males in the ship=',Sex_male['Sex'].count())



# to calculate the female numbers we subtract the males from all the passengers number

print('Total females in the ship=',891-Sex_male['Sex'].count())

sns.set(style="darkgrid")

sns.countplot(x="Survived", data=train,hue="Sex",palette="rocket",saturation=0.6);



plt.title('THE SURVIVED PASSENGERS BASED ON GENDER\n',fontsize=17)



plt.xlabel('Survived',fontsize=17)

plt.ylabel('Gender Count',fontsize=17);

 
sns.countplot(x="Pclass", hue="Survived", data=train,palette="rocket",saturation=0.6);



plt.title('THE SURVIVED PASSENGERS BASED ON CLASS\n',fontsize=17)

plt.xlabel('Survived',fontsize=17)

plt.ylabel('Class Count',fontsize=17);







Age_survived=train.pivot(columns='Survived').Age



colors = ['lightcoral', 'lightskyblue']

Age_survived.plot(kind = 'hist', stacked=True,bins=9,figsize=(6,4),color=colors)



plt.title('THE SURVIVED PASSENGERS BASED ON AGE\n',fontsize=17)

plt.xlabel('Age',fontsize=17)

plt.ylabel('Survived frequency',fontsize=17);

sns.lmplot('Age', 'Fare', data=train, fit_reg=False, hue="Pclass",

scatter_kws={"marker": "o", "s": 40},palette="magma",size=6);

plt.title('PASSENGER CLASS AGE AND FARE\n',fontsize=15);

#correlation of features with Survived 

print(train.corr()["Survived"].sort_values())



print('\nPclass is the highest negative correlation')

print('Fare is the Highest positive correlation')
#calculate the covariance to quantify the strength and direction of a relationship between two variables



X_column=train['Fare'].copy()

X_column_2=train['Pclass'].copy()



y_column=train['Survived'].copy()



cov_matrix_1 = np.cov(X_column, y_column)



cov_matrix_2 = np.cov(X_column_2, y_column)



print('The convariance between Fare and Survived:\n',cov_matrix_1)



print('\nThe convariance between Pclass and Survived:\n',cov_matrix_2)





#If the correlation is positive, then the covariance is positive, as well.



#If the correlation is negative, then the covariance is negative, as well.



#If the correlation is weak, then the covariance is close to zero.
## heatmeap to see the correlation between features. 



mask = np.zeros_like(train.corr())

mask[np.triu_indices_from(mask)] = True



plt.subplots(figsize = (8,7))

sns.heatmap(train.corr(), annot=True,mask = mask, cmap ='PRGn',linewidths=0.1,square=True)

plt.title("CORRELATIONS BETWEEN FEATURES\n", y = 1.03,fontsize = 20);



#sibSP and Parch is positively correlated

fig,ax = plt.subplots(1,figsize=(9, 3))



sns.boxplot(data=train['Fare'], orient='h', fliersize=8, 

linewidth=3, notch=True, saturation=0.5,showmeans=True,

meanline=True,medianprops={'linewidth': 2, 'color': 'lightskyblue'},

meanprops={'linewidth': 2, 'color': 'red'}, ax=ax);



#use the box plot to observe  the range, interquartile range,mean,median, and outliers



# the mean is the red line

# the median is the light blue line



# the fare column has multi fares, but in order to achieve the correct prediction we should ignore the outliers 
#observe the missing value in train

train.isnull().sum()

# fill the missing valus in Embarked column with the most frequent value using the mode

train['Embarked'] = train['Embarked'].fillna(train['Embarked'].mode().iloc[0])

train['Sex'] = train['Sex'].map({'female': 0, 'male': 1}).astype(int)
#replace each name with the passenger title only 



v=['Mr.','Miss.','Mrs.','Master.']

train['Name'] = train['Name'].apply(lambda x: ' '.join(np.array(x.split(' '))[np.in1d(x.split(' '),v)]))



train['Name'].replace('','NoTitle',inplace=True)
#get the mean for each name category



Age_mean = pd.pivot_table(train, values='Age',index=['Name'],aggfunc=np.mean)



display(Age_mean)
#define a function to fill the missing values in age

#we are going to use the mean from the above table to fill the age

def impute_age(age_name): 

    

    Age = age_name[0]

    Name = age_name [1]

    

    # if the values ==NaN than it's true and will enter the if-condition   

    if pd.isnull(Age):

        

        #if the name== Mr. than fill the missing value with it's mean

        if Name == 'Mr.':

            return 32

        

        #if the name== miss. than fill the missing value with it's mean

        elif Name == 'Miss.':

            return 22

        

        #if the name== Mrs. than fill the missing value with it's mean

        elif Name == 'Mrs.':

            return 35

        

        #if the name== Master. than fill the missing value with it's mean

        elif Name =='Master.':

            return 5



        else:

            return 42

    else:

        return Age
#call the function above with the age column

train.Age = train.apply(lambda x :impute_age(x[['Age', 'Name']] ) , axis = 1)
train['Age'].describe()
train.isnull().sum()
#Rerange the Age based on its distribution from its statistics.

# As the mean is almost 30, and standard deviation is almost 13, so the majority of passenger ages are 

#placed between 17 and 43..

# the uper level = mean+standard deviation = 30-13=17

# the lower level = mean-standard deviation = 30+13=43

# the other group will be by adding the standard deviation value



train.loc[ train['Age'] <= 17, 'Age'] = 0

train.loc[(train['Age'] > 17) & (train['Age'] <= 43), 'Age'] = 1

train.loc[(train['Age'] > 43) & (train['Age'] <= 56), 'Age'] = 2

train.loc[(train['Age'] > 56) & (train['Age'] < 81), 'Age'] = 3
#add new column contain the addition of two columns:[SibSp + Parch] +1

# we add +1 here because some of the passengers are alone in the ship with no siblings or parent. 

#so it get "the sibilings number" + "the parents number" + the passenger hemself

train['FamilyMember'] = train['SibSp'] + train['Parch'] + 1
#Drop SibSp,Parch and the Ticket after adding the family member column

train=train.drop(['SibSp','Parch','Ticket'],axis=1)

train.head(2)
DataFare=train.copy()

#use qcut to cut variable into equal-sized buckets based on sample quantiles

DataFare['Fare_Q'] = pd.qcut(train['Fare'], 4)



#group Fare and Survived columns and calculate it's mean

DataFare[['Fare_Q', 'Survived']].groupby(['Fare_Q'], as_index=False).mean()
#Rerange the Fare based on what we have from the table above



train.loc[ train['Fare'] <= 7.91, 'Fare'] = 0

train.loc[(train['Fare'] > 7.91) & (train['Fare'] <= 14.454), 'Fare'] = 1

train.loc[(train['Fare'] > 14.454) & (train['Fare'] <= 31), 'Fare']   = 2

train.loc[ train['Fare'] > 31, 'Fare'] = 3





train['Fare'] = train['Fare'].astype(int)

#take the unique values in Cabin column as a list in a new variable

train_cabin_fill=list(train['Cabin'].unique())



#since the first value in the list is nan, we will use pop() 

train_cabin_fill.pop(0)



print(train_cabin_fill)

len(train_cabin_fill)
#define a function to replace all the cabin characterwith number to a simple character 

#for example Cabin with name: C2,C34,C64 will be replaced with C

def search_cabin(data,character):

    

    Result = [idx for idx in data if idx.lower().startswith(character.lower())]

    return Result



train['Cabin'].replace(search_cabin(train_cabin_fill,'A'), 'a',inplace=True)



train['Cabin'].replace(search_cabin(train_cabin_fill,'B'), 'b',inplace=True)



train['Cabin'].replace(search_cabin(train_cabin_fill,'C'), 'c',inplace=True)



train['Cabin'].replace(search_cabin(train_cabin_fill,'D'), 'd',inplace=True)



train['Cabin'].replace(search_cabin(train_cabin_fill,'E'), 'e',inplace=True)



train['Cabin'].replace(search_cabin(train_cabin_fill,'G'), 'g',inplace=True)



train['Cabin'].replace(search_cabin(train_cabin_fill,'F'), 'f',inplace=True)


train_cabin_pivot = pd.pivot_table(train, index=['PassengerId','Pclass',

'Cabin','Name','Age','Embarked' ,'Survived' ,'Sex','FamilyMember'],values=['Fare'])
train_cabin_pivot = train_cabin_pivot.reset_index()



train_cabin_pivot.apply(pd.to_numeric,errors='ignore')



train_cabin_pivot.head()
train_cabin_pivot=pd.get_dummies(train_cabin_pivot, columns=['Name','Embarked'],drop_first=True)

train_cabin_pivot.head()
# get all rows of Cabin with the missing values in a variable

train_cabin_null = train[train['Cabin'].isnull()]



# dummies

train_cabin_null=pd.get_dummies(train_cabin_null, columns=['Name','Embarked'],drop_first=True)



train_cabin_null.head()


train_cabin_pivot.head()



#put the target in cabin_X_train without the ID and Cabin

cabin_X_train=train_cabin_null.drop(['PassengerId','Cabin'],axis=1)



#define X and y

X=train_cabin_pivot.drop(['PassengerId','Cabin'],axis=1).copy()

y=train_cabin_pivot['Cabin'].copy()



#using the random forest classifier

RandomForest = RandomForestClassifier(n_estimators=100,random_state=1)



#train the model

RandomForest.fit(X,y)
#get the prediction of the cabin

pred=RandomForest.predict(cabin_X_train)
#fill the null values in the train data with the prediction

train.loc[pd.isnull(train['Cabin']), 'Cabin'] = pred
#map each alphabet with number

train['Cabin']=train['Cabin'].map({'a': 0, 'b': 1,'c':2,'d':3,'e':4,'f':5,'g':6,'T':7})
train.isnull().sum()
train.head()
test.isnull().sum()
test.head(3)
#map each gender with with 0 or 1

test['Sex'] = test['Sex'].map({'female': 0, 'male': 1}).astype(int)
#replace each name with the passenger title only 

title=['Mr.','Miss.','Mrs.','Master.']

test['Name'] = test['Name'].apply(lambda x: ' '.join(np.array(x.split(' '))[np.in1d(x.split(' '),title)]))

test['Name'].replace('','NoTitle',inplace=True)
#get the mean for each name category

Age_mean_test = pd.pivot_table(test, values='Age',index=['Name'],aggfunc=np.mean)



display(Age_mean_test)
# fill the missing values in Age column with the most frequent value using the mode

test.Age = test.apply(lambda x :impute_age(x[['Age', 'Name']] ) , axis = 1)

test['Age'].isnull().sum()
# add the family member column with [SibSp + Parch + 1]

test['FamilyMember'] = test['SibSp'] + test['Parch'] + 1
# drop three columns

test= test.drop(['SibSp','Parch','Ticket'], axis=1)
test['Age'].describe()
#Rerange the Age based on its distribution from its statistics.

# As the mean is almost 30, and standard deviation is around 13, so the majority of passenger ages are 

#placed between 17 and 43..

# the uper level = mean+standard deviation = 30-13=17

# the lower level = mean-standard deviation = 30+13=43

# the other group will be by adding the standard deviation value



test.loc[ test['Age'] <= 17, 'Age'] = 0

test.loc[(test['Age'] > 17) & (train['Age'] <= 43), 'Age'] = 1

test.loc[(test['Age'] > 43) & (train['Age'] <= 56), 'Age'] = 2

test.loc[(test['Age'] > 56) & (train['Age'] < 81), 'Age'] = 3
test['Age'] = test['Age'].astype(int)
test['Fare'].sort_values().tail()
#fill the missing value with median

test['Fare'].fillna(test['Fare'].dropna().median(), inplace=True)
#use the same logic in the train cleaning for filling the Fare using the range of qcut

test.loc[ test['Fare'] <= 7.91, 'Fare'] = 0

test.loc[(test['Fare'] > 7.91) & (train['Fare'] <= 14.454), 'Fare'] = 1

test.loc[(test['Fare'] > 14.454) & (train['Fare'] <= 31), 'Fare']   = 2

test.loc[ test['Fare'] > 31, 'Fare'] = 3

test['Fare'] = test['Fare'].astype(int)
#find the unique values in cabin and add it as a list in new variable

test_cabin_fill=list(test['Cabin'].unique())



#pop the nan value

test_cabin_fill.pop(0)

print(test_cabin_fill)
#call the same function we define in train dataframe cleaning section





test['Cabin'].replace(search_cabin(test_cabin_fill,'A'), 'a',inplace=True)



test['Cabin'].replace(search_cabin(test_cabin_fill,'B'), 'b',inplace=True)



test['Cabin'].replace(search_cabin(test_cabin_fill,'C'), 'c',inplace=True)



test['Cabin'].replace(search_cabin(test_cabin_fill,'D'), 'd',inplace=True)



test['Cabin'].replace(search_cabin(test_cabin_fill,'E'), 'e',inplace=True)



test['Cabin'].replace(search_cabin(test_cabin_fill,'G'), 'g',inplace=True)



test['Cabin'].replace(search_cabin(test_cabin_fill,'F'), 'f',inplace=True)
test_cabin_pivot = pd.pivot_table(test, index=['PassengerId','Pclass',

'Cabin','Name','Age','Embarked' ,'Sex','FamilyMember'],values=['Fare'])
test_cabin_pivot = test_cabin_pivot.reset_index()



test_cabin_pivot.apply(pd.to_numeric,errors='ignore')



test_cabin_pivot.head()
test_cabin_pivot=pd.get_dummies(test_cabin_pivot, columns=['Name','Embarked'],drop_first=True)



test_cabin_pivot.head()
# find the rows with missing values in Cabin

test_cabin_null = test[test['Cabin'].isnull()]



#dummies

test_cabin_null=pd.get_dummies(test_cabin_null, columns=['Name','Embarked'],drop_first=True)



test_cabin_null.head()
# put the missing values dataframe in cabin_X_test as a target

cabin_X_test=test_cabin_null.drop(['PassengerId','Cabin'],axis=1)



#defin X & y

X=test_cabin_pivot.drop(['PassengerId','Cabin'],axis=1).copy()

y=test_cabin_pivot['Cabin'].copy()



#use Random Forest for classification

randomforest = RandomForestClassifier(n_estimators=100,random_state=1)

randomforest.fit(X,y)

#get the prediction of the Cabin

pred_test=randomforest.predict(cabin_X_test)
#fill hte missing values in the train dataframe with the prediction

test.loc[pd.isnull(test['Cabin']), 'Cabin'] = pred_test
#map each alphabet with number

test['Cabin']=test['Cabin'].map({'a': 0, 'b': 1,'c':2,'d':3,'e':4,'f':5,'g':6,'T':7})
test.isnull().sum()
test.head()
# we can see the correlation for some features with the target has improved after cleaning:[Age,Fare]

mask = np.zeros_like(train.corr())

mask[np.triu_indices_from(mask)] = True



plt.subplots(figsize = (8,7))

sns.heatmap(train.corr(), annot=True,mask = mask, cmap ='PRGn',linewidths=0.1,square=True)

plt.title("CORRELATIONS BETWEEN FEATURES\n", y = 1.03,fontsize = 20);



train = pd.get_dummies(train, columns=['Name','Embarked'],drop_first=True)



test = pd.get_dummies(test, columns=['Name','Embarked'],drop_first=True)
#drop the passenger Id from the test dataframe

test_prediction=test.drop(['PassengerId'],axis=1)

test_prediction.head()
X=train.drop(['PassengerId','Survived'],axis=1).copy()

y=train['Survived'].copy()



X_train, X_test, y_train, y_test = train_test_split(

    X, y, test_size=0.2, random_state=4)
X.columns
# # to find the best random_state,test_size and n_estimators will we use for loop with the range(1,6)



# for i in range(1,6):

#     for k in range(1,6):

#         X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=i,test_size=k/10)

#         rf = RandomForestClassifier(n_estimators=i*100,random_state=i)

#         rf.fit(X_train,y_train)

#         tr_score=rf.score(X_train, y_train)

#         tr_score

#         te_score=rf.score(X_test, y_test)

#         te_score

#         print("test size",k,"and","random state",i,"result of test score=",te_score)
# use the best 3 parameters we find from the loop above 

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=4,test_size=.1)

rf = RandomForestClassifier(n_estimators=100,random_state=4)



#train te model

rf.fit(X_train,y_train)



tr_score=rf.score(X_train, y_train)

print('The training score:',tr_score)



te_score=rf.score(X_test, y_test)

print('The testing score:',te_score)



print("test size","10%","and","random state",4,"result of test score=",te_score)

pred_4=rf.predict(test_prediction)

acc_rf= accuracy_score(gender_sub['Survived'],pred_4)



acc_rf
y_dataframe=pd.DataFrame(pred_4,columns=['Survivor_pred'])



dataset=pd.concat([gender_sub['PassengerId'],y_dataframe],axis=1)

dataset.columns=['PassengerId','Survived']

dataset.to_csv('titanic_sub_forest_.csv',index=False)
Rforest = RandomForestClassifier()



params = {'max_depth': [1, 2, 3, 4, 5, 6,8,9,10,13,15,16,17,18,20],

          'max_features':[1,.2,.3,.4,.6,.8,.9],

          'max_leaf_nodes': [5, 6, 7, 8, 9, 10],

          'min_samples_leaf': [1, 2, 3, 4]}

    

gs = GridSearchCV(Rforest, param_grid=params, cv=5)

gs.fit(X_train, y_train)
#print the best estimator

print(gs.best_estimator_)

#print the traning score

print('\n\ntraining score : ', gs.score(X_train, y_train))

#print the testing score

print('test score: ', gs.score(X_test, y_test))

print('The Baseline accuracy:\n',y_train.value_counts(normalize=True))
pred_survivor=gs.predict(test_prediction)

acc_rf_gs= accuracy_score(gender_sub['Survived'],pred_survivor)



print('The Random Forest accuracy:',acc_rf_gs)
# y_dataframe=pd.DataFrame(pred_survivor,columns=['Survivor_pred'])



# dataset=pd.concat([gender_sub['PassengerId'],y_dataframe],axis=1)

# dataset.columns=['PassengerId','Survived']

# dataset.to_csv('titanic_sub_forest_grid.csv',index=False)



model_adaboost = AdaBoostClassifier()



params_adaboost = {

          'n_estimators':[10,20,40,50,70,80],

            'random_state':[1,2,20,40],

           'learning_rate':[1,2,3,4,10,20]

          }





Boost_grid = GridSearchCV(model_adaboost, param_grid=params_adaboost, cv=5)



Boost_grid.fit(X_train, y_train)

#print the training score

print('Train accuracy:',Boost_grid.score(X_train, y_train))

#print the testing score

print('Test accuracy:',Boost_grid.score(X_test, y_test))

pred_survivor_boost=Boost_grid.predict(test_prediction)

acc_adboost= accuracy_score(gender_sub['Survived'],pred_survivor_boost)



print('The Adaboost classifier accuracy:', acc_adboost)
# y_dataframe=pd.DataFrame(pred_survivor_boost,columns=['Survivor_pred'])



# dataset=pd.concat([gender_sub['PassengerId'],y_dataframe],axis=1)

# dataset.columns=['PassengerId','Survived']

# dataset.to_csv('titanic_sub_adaboost.csv',index=False)



from sklearn import svm 





parm_grid = {'gamma': np.logspace(-5, 2, 20)}

svm = svm.SVC(kernel='rbf')



grid = GridSearchCV(svm, parm_grid, cv=5)



grid.fit(X_train, y_train)



grid.best_params_
grid.best_score_
pred_svm=grid.predict(test_prediction)

acc_svm= accuracy_score(gender_sub['Survived'],pred_svm)



print('The SVM accuracy:', acc_svm)

# y_dataframe=pd.DataFrame(pred_svm,columns=['Survivor_pred'])



# dataset=pd.concat([gender_sub['PassengerId'],y_dataframe],axis=1)

# dataset.columns=['PassengerId','Survived']

# dataset.to_csv('titanic_sub_svm.csv',index=False)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)  
scaler = StandardScaler()  

scaler.fit(X_train)



X_train = scaler.transform(X_train)  

X_test = scaler.transform(X_test)  



classifier = KNeighborsClassifier(n_neighbors=9)  

classifier.fit(X_train, y_train)

y_pred_knn = classifier.predict(test_prediction) 

y_pred_knn

acc_knn= accuracy_score(gender_sub['Survived'],y_pred_knn)

print("when knn=",2,"accuracy=",acc_knn)



decision_tree = DecisionTreeClassifier(max_depth=50)

decision_tree.fit(X_train, y_train)

y_pred_dt = decision_tree.predict(test_prediction)

acc_decision_tree = decision_tree.score(X_test,y_test)

print('The accuracy for Decision Tree prediction:',acc_decision_tree)
parameters = { 'max_features': [0.3, 0.6, 1],

        'n_estimators': [50, 150, 200], 

         'base_estimator__max_depth': [3, 5, 20]}

model_dec_tree = BaggingClassifier(base_estimator=DecisionTreeClassifier(), oob_score=True)

model_gs_dectree = GridSearchCV(model_dec_tree ,parameters, cv=4, n_jobs=-1 )



model_gs_dectree.fit(X_train, y_train)

print('The best parameters',model_gs_dectree.best_params_)

print('The best estimator and OBB score:',model_gs_dectree.best_estimator_.oob_score_)



dectree_pred=model_gs_dectree.predict(test_prediction)

y_dataframe=pd.DataFrame(dectree_pred,columns=['Survivor_pred'])

acc_dt= accuracy_score(gender_sub['Survived'],dectree_pred)



print('The accuracy for Decision Tree prediction:',acc_dt)
dt = DecisionTreeClassifier()

dt_en = BaggingClassifier(base_estimator=dt, n_estimators=200, oob_score=True )

dt_en.fit(X_train, y_train)

print('The train score:',dt_en.score(X_train, y_train))

print('The test score:',dt_en.score(X_test, y_test))

print('The OOB score:',dt_en.oob_score_)



bag_tree_pred=dt_en.predict(test_prediction)

print('The accuracy for Bagging classifier prediction:',bag_tree_pred)
param = { 'max_features': [0.3, 0.6, 1],

        'n_estimators': [50, 150, 200], 

         'base_estimator__max_depth': [3, 5, 20]}

model_bag_tree = BaggingClassifier(base_estimator=DecisionTreeClassifier(), oob_score=True)

model_gs_bagtree = GridSearchCV(model_bag_tree,param, cv=4, verbose=1, n_jobs=-1 )

model_gs_bagtree.fit(X_train, y_train)



print('The best parameters',model_gs_bagtree.best_params_)

print('The best estimator obb score',model_gs_bagtree.best_estimator_.oob_score_)





bagtree_pred=model_gs_bagtree.predict(test_prediction)

acc_bagtree= accuracy_score(gender_sub['Survived'],bagtree_pred)



print('The accuracy for Bagging classifier prediction:',acc_bagtree)
logreg = LogisticRegression()

logreg.fit(X_train, y_train)

y_pred_logreg = logreg.predict(test_prediction)



acc_log= accuracy_score(gender_sub['Survived'],y_pred_logreg)

print('The accuracy for Logistic Regression prediction:',acc_log)



y_dataframe=pd.DataFrame(y_pred_logreg,columns=['Survivor_pred'])



dataset=pd.concat([gender_sub['PassengerId'],y_dataframe],axis=1)

dataset.columns=['PassengerId','Survived']

dataset.to_csv('titanic_sub_logistic.csv',index=False)
list_scores = [acc_rf,acc_rf_gs,acc_adboost,acc_svm,acc_knn,acc_dt,acc_bagtree,acc_log]

list_classifiers = ['RandomForest','RandomForest_GS','Adboost','SVM','KNN','Decision Tree','Baggin-DT','Logistic Regression']

fig, ax = plt.subplots()

fig.set_size_inches(18,7)

sns.barplot(x=list_classifiers, y=list_scores, ax=ax)

plt.xlabel('Classifier',fontsize=20)

plt.ylabel('Accuracy',fontsize=20)

ax.tick_params(labelsize=15)



plt.show()