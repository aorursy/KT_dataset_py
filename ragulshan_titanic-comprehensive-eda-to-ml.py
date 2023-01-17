#importing libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



#To ignore Warnings

import warnings

warnings.filterwarnings("ignore")



#Data visualization Library

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

plt.style.use('fivethirtyeight')
#importing csv files

df_train = pd.read_csv('/kaggle/input/titanic/train.csv')

df_test =  pd.read_csv('/kaggle/input/titanic/test.csv')



#Saving test IDs for submission

ID_data = df_test['PassengerId']
#Shapes of train and test dataset

print("Shape of train dataset :",df_train.shape)

print("Shape of train dataset :",df_test.shape)
#Size of our datasets

print("Size of train dataset :",df_train.size) #size = rows * Columns

print("Size of test dataset :",df_test.size) #size = rows * Columns
#Displaying top5 rows training set

df_train.head()
#Displaying top rows testing set

df_test.head()
#infos

df_train.info()
#Types of datatypes 

df_train.dtypes
#Visualizing #of datatypes in train dataset

ax = df_train.dtypes.value_counts().sort_values().plot.barh(color='green')

ax.set(ylabel='Datatypes',xlabel='Counts');
#Length of datasets

print ("Length of train dataset :",len(df_train))

print ("Length of test dataset :",len(df_test))
#No of unique values in each feature(column)

df_train.nunique().to_frame()
#descriptive stats for numerical features(we can ues that to see text features too)

df_train.describe()
#Let's see above stat details for text features:

df_train.describe(include='object').transpose()
#Stats by age 

df_train.groupby('Survived').describe()['Age']
#Let's see correlation between features

df_train.corr()
#Heatmap

plt.figure(figsize=(14,8))

sns.heatmap(df_train.corr(),annot=True,cmap='coolwarm')

plt.title('Correaltion Co-eff Matrix',fontsize=16);

#Expore corrleation coeff__ for survived feature:

df_train.corr()['Survived'].sort_values().to_frame()
#Correlation with survival features

plt.figure(figsize=(14,6))

df_train.corr()['Survived'].sort_values()[:-1].plot.bar(color='r') #I'm leaving out survival feature 

plt.ylabel('Correlation Strength and Direction',fontsize=14)

plt.xlabel('Features',fontsize=14);
#Distribution of AGE feature

sns.distplot(df_train['Age'].dropna(),bins=30,color='m');

plt.title('AGE distribution ');
#outlier detection 

sns.boxplot(df_train['Survived'],df_train['Age']);
fig,ax= plt.subplots(1,3,figsize=(20,8))

sns.violinplot(df_train['Survived'],df_train['Age'],ax=ax[0])

sns.violinplot(df_train['Survived'],df_train['Age'],hue=df_train['Sex'],ax=ax[1],palette='winter')

sns.violinplot(df_train['Survived'],df_train['Age'],hue=df_train['Pclass'],ax=ax[2],palette='winter');
#Survived or not survived passengers

df_train.groupby(["Survived"]).agg({'Age':{'Min':'min','Max':'max','Avg':'mean'}})
#VIsualizing saeborn different kinds of plot

fig,ax= plt.subplots(1,3,figsize=(22,8))

sns.stripplot(df_train['Embarked'],df_train['Age'],ax=ax[0])

sns.violinplot(df_train['Embarked'],df_train['Age'],ax=ax[1],palette='plasma')

sns.swarmplot(df_train['Embarked'],df_train['Age'],ax=ax[2],palette='plasma');
df_train.groupby(["Embarked"]).agg({'Age':{'Min':'min','Max':'max','Avg':'mean'}})
df_train.groupby(["Embarked",'Survived']).agg({'Age':{'Min':'min','Max':'max','Avg':'mean'}})
#Explore Age distribution with respect to survived 

g = sns.FacetGrid(df_train,col='Survived')

g.map(sns.distplot,'Age',kde=False,);
#Avg age- survived or not survived by Passenger class

df_train.groupby(['Pclass']).mean()['Age'].to_frame()
#pandas crosstab...

pd.crosstab(df_train['Survived'],df_train['Embarked'],values=df_train['Age'],aggfunc=np.mean)
#Outlier detection with boxplot and stripplot

plt.figure(figsize=(12,6))

sns.boxplot(df_train['Survived'],df_train['Age'],palette='viridis')

sns.stripplot(df_train['Survived'],df_train['Age'],);
#Exploring pclass vs Age

g = sns.FacetGrid(df_train,col='Pclass')

g.map(sns.distplot,'Age',kde=False,color='indianred');
#Explore sibsp feature

plt.figure(figsize=(12,6))

sns.boxplot(df_train['SibSp'],df_train['Age'],palette='winter');
#Distribution of Fare price feature

plt.figure(figsize=(10,6))

sns.distplot(df_train['Fare'].dropna(),bins=30);

plt.title('Distribution of fare(Price)');
#Skewness 

df_train['Fare'].skew()
#seaborn swarmplot for Fare prices

sns.swarmplot(df_train['Survived'],y=df_train['Fare'],palette='Greens');
#Scatterplot - to plot two numerical values



sns.scatterplot(df_train['Fare'],df_train['Age'],color='c')

plt.title('Fare Vs Age');
#Joinplot - Visualize both histogram and scatterplot

fig,ax = plt.subplots(1,2,figsize=(18,6))

sns.scatterplot(df_train['Fare'],df_train['Age'],color='r',ax=ax[0])

sns.scatterplot(df_train['Fare'],df_train['Age'],color='r',hue=df_train['Survived'],ax=ax[1],palette='viridis');
#Outlier or anamolies detection 

fig,ax = plt.subplots(1,2,figsize=(14,6))

sns.boxplot(df_train['Survived'],df_train['Fare'],ax=ax[0])

sns.boxplot(df_train['Survived'],df_train['Fare'],hue=df_train['Sex'],ax=ax[1]);
#Explore fare by survival and gender wise:

df_train.groupby(['Survived','Sex']).agg({'Fare':{'Min_fare':'min','Max_fare':'max','Avg_Fare':'mean','Median_fare':'median'}})
g = sns.FacetGrid(df_train, hue="Survived", col="Pclass", margin_titles=True,

                  palette={1:"seagreen", 0:"gray"})

g=g.map(plt.scatter, "Fare", "Age",edgecolor="w").add_legend();
#Pclass fare details

df_train.groupby(['Pclass','Survived']).agg({'Fare':{'Min_fare':'min','Max_fare':'max','Avg_Fare':'mean','Median_fare':'median'}})
#box and stripplot for distribution of prices

plt.figure(figsize=(12,6))

sns.boxplot(df_train['Survived'],df_train['Fare'])

sns.stripplot(df_train['Survived'],df_train['Fare']);
#Exploring fare by embarked feature

df_train.groupby(['Embarked','Survived']).agg({'Fare':{'Min_fare':'min','Max_fare':'max','Avg_Fare':'mean','Median_fare':'median'}})
f,ax=plt.subplots(1,2,figsize=(18,8))

df_train['Survived'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)

ax[0].set_title('Survived')

ax[0].set_ylabel('')

sns.countplot('Survived',data=df_train,ax=ax[1])

ax[1].set_title('Survived');
#Male vs Female passengers ratio

df_train['Sex'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',shadow=True);
#Exploring survival features 

fig,ax= plt.subplots(1,3,figsize=(20,8))

sns.countplot(df_train['Survived'],ax=ax[0])

sns.countplot(df_train['Survived'],hue=df_train['Sex'] ,ax=ax[1])

sns.countplot(df_train['Survived'],hue=df_train['Pclass'] ,ax=ax[2],palette='winter');
#Exploring survival features 

fig,ax= plt.subplots(2,2,figsize=(20,8))

sns.countplot(df_train['Survived'],ax=ax[0,0])

sns.countplot(df_train['Survived'],hue=df_train['Sex'] ,ax=ax[0,1])

sns.countplot(df_train['Survived'],hue=df_train['Pclass'] ,ax=ax[1,0])

sns.countplot(df_train['Survived'],hue=df_train['Embarked'] ,ax=ax[1,1]);
print('Counts of survival by pclass :')

df_train.groupby(['Survived','Pclass']).count()[['PassengerId']].unstack()
#Lets visualize sibsp features 

sns.countplot(df_train['SibSp'],hue=df_train['Survived'],palette='magma')

plt.ylabel('Survival count')

plt.legend(loc=1);
sns.countplot(df_train['SibSp'],hue=df_train['Embarked'],palette='plasma')

plt.legend(loc=1);
#crosstab

pd.crosstab([df_train.Sex,df_train.Survived],df_train.Pclass,margins=True)
#factorplot

sns.factorplot('Pclass','Survived',hue='Sex',data=df_train);
fig,(ax1,ax2) = plt.subplots(1,2,figsize=(14,6))



sns.factorplot('Embarked','Survived',data=df_train,ax=ax1)



sns.factorplot('Embarked','Survived',hue='Sex',data=df_train,ax=ax2);
#merging train and test dataset

df = pd.concat([df_train.drop('Survived',axis=1),df_test],axis=0)
#data top rows

df.head(5)
#visualising missing values

sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis');
#Missing value percentage per variable 

mis_perc = round(df.isnull().sum().to_frame()/len(df),5)*100

mis_perc
#How many unique labels are on cabin?

print(df.Cabin.value_counts())

print(f"\n No of unique labels {df.Cabin.nunique()}")
#we could use numpy where function to create a variable taht captures null values

df['cabin_NA'] = np.where(df.Cabin.isnull(),1,0)
#ensure it worked

df.cabin_NA.value_counts(normalize=True) * 100
pd.crosstab(df.cabin_NA,df.Sex).plot.pie(subplots=True,autopct='%1.1f%%',shadow=True,figsize=(10,5));
#Fill missing values with median

df['Fare'] = df['Fare'].fillna(df['Fare'].median)
#When we filled missing values it converted to string 

df['Fare']= df['Fare'].astype('str')

df['Fare'] = pd.to_numeric(df['Fare'],errors='coerce')
#filling embarked feature

df.Embarked.fillna(df.Embarked.mode,inplace=True)
#we can drop cabin feature since most of data in that feature is missing

df =df.drop('Cabin',axis=1)
#Ensuring we dropped that column

df.head()
#Checking realtionship between age and other features

df.corrwith(df.Age).to_frame()
#Taking age mean for each class

df.groupby(['Pclass'])['Age'].mean().to_frame()
#Defining function to impute missing values

def impute_age(cols):

    Age = cols[0] #grabbing 1st column

    Pclass = cols[1] #Grabbing 2nd column

    

    if pd.isnull(Age):



        if Pclass == 1:

            return 39



        elif Pclass == 2:

            return 29



        else:

            return 25



    else:

        return Age
#Lets apply that function

df['Age'] = df[['Age','Pclass']].apply(impute_age,axis=1)
#Now let's check that heat map again!

sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis');
#Missing values 

df.isnull().sum()
#remove unnecessary columns

df = df.drop(['Name','PassengerId','Ticket'],axis=1)
#Lets checkout the frame

df.head()
#shape of dataset

df.shape
#we can convert age column to categoical feature using pandas cut method(7 categories)

bins = ( 0, 5,12,18,25,40,60,120)

group_names = [ 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']

df['age_cat'] = pd.cut(df.Age,bins=bins,labels=group_names)
#Percentage of different age passengers on board

df['age_cat'].value_counts(normalize=True).round(4).to_frame() *100
df.dtypes.to_frame()
#Discretising Fare variable

df['fare'] = pd.cut(df.Fare,bins=[0,10,30,60,1000],labels=['Low','medium','high','very high'])
#Lets see what we have done

df.head()
#Lets convert categorical features to numerical 

cat1 = pd.get_dummies(df['age_cat'],drop_first=True)

cat2 = pd.get_dummies(df['Sex'],drop_first=True)

cat3 = pd.get_dummies(df['fare'],drop_first=True)



#Concat those features to our original dataframe

df = pd.concat([df,cat1,cat2,cat3],axis=1)
#Embarked feature encoding

embark =pd.get_dummies(df['Embarked'].astype('str'),drop_first=True)

df = pd.concat([df,embark],axis=1)
df.head()
#Drop unneccessary columns

df = df.drop(['Embarked','Age','Fare','Sex','age_cat','fare'],axis=1)
#CHeck our final our dataframe

df.head()
#lets check shpae our final dataset

df.shape
#train test split

train = df.iloc[:891,:]

test = df.iloc[891:,:] # to submit our predictions

y = df_train['Survived']
#import Ml algos

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score,KFold,GridSearchCV,RandomizedSearchCV

from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,f1_score,precision_score,roc_auc_score

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.neighbors import KNeighborsClassifier

from xgboost import XGBClassifier

from sklearn.svm import SVC



#Split

x_train,x_val,y_train,y_val = train_test_split(train,y,test_size=.3,random_state=101)
#create a function to assses the model performance



def model_eval(model):

    

    train_preds = model.predict(x_train)

    val_preds = model.predict(x_val)

    scores = {"Training Accuracy": accuracy_score(y_train, train_preds),

              "Valid Accuracy": accuracy_score(y_val, val_preds),

              "Training auc_roc": roc_auc_score(y_train,model.predict_proba(x_train)[:,1]),

              "Valid auc_roc": roc_auc_score(y_val,model.predict_proba(x_val)[:,1]),

              }

    

    return scores
#set the reproductivity

np.random.seed(42)



#Traing our model(Without Hyper parameter optimization)

log_model = LogisticRegression()

log_model.fit(x_train,y_train)



#prediction

log_pred = log_model.predict(x_val)



#Model evaluation

model_eval(log_model)
# Setup random seed

np.random.seed(42)



#Searching best estiamtors using SKlearn gridsearch :

param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],'penalty':['l1','l2'] }

clf = GridSearchCV(LogisticRegression(), param_grid,cv=5,scoring='accuracy')



#Finding best parameters

clf.fit(x_train,y_train)



#Predictions

lmg = clf.predict(x_val)



#Find the model best parameter

print('Best parameters \n:' , clf.best_params_)

print ('Best score :', clf.best_score_ * 100)
#Model assessment

model_eval(clf)
#Trying different thresholds and assess their affect on results(experimental)

from sklearn.preprocessing import binarize



thresh = []

acc_score =[]

#predict probabiliy for each instances

probs = clf.predict_proba(x_val)

for i in np.arange(0.1,0.9,0.001):

    preds = binarize(probs,i)[:,1]

    thresh.append(i)

    score = accuracy_score(y_val,preds) *100

    acc_score.append(np.round(score,2))



#See top 5 thresholds values which acn increase boost the aacuracy

pd.DataFrame([thresh,acc_score]).T.sort_values(by=1,ascending=False).set_index(0).head(5)
#Coefficient of each features

co_eff = pd.DataFrame(log_model.coef_.T,index=x_train.columns,columns=['Co_eff']).sort_values(by='Co_eff',ascending=False)

co_eff
#Lets plot the feature coefficients to understand better features

co_eff.plot.barh(title='Logistic model Co-Efficient',figsize=(11,5));
fig,(ax1,ax2)= plt.subplots(nrows=1,ncols=2,sharex=True,sharey=True,figsize=(12,6))



#Barplot 

df_train.groupby(['Sex','Survived']).size().unstack().plot.bar(title='Survival ratio by Gender',ax=ax1)



#Stacked barplot

df_train.groupby(['Sex','Survived']).size().unstack().plot.bar(stacked=True,title='Stacked survival ratio by Gender',ax=ax2);
#Combining train x & y to get some insights

train_full = pd.concat([x_train,y_train],axis=1)

train_full.groupby(['Survived','male']).size().to_frame()
print(pd.crosstab(train_full.Survived,train_full.male))



#Groupby survived to get male&female survival ratio

survival_rat = train_full.groupby(['Survived'])['male'].mean().round(4).to_frame() *100

survival_rat.index = ['Not Survived','Survived']

survival_rat['female'] = 100 - survival_rat['male']

survival_rat
#Groupby gender

gender_ratio =train_full.groupby(['male'])['Survived'].mean().to_frame() *100 # 0 means female and 1 means male.

gender_ratio.index = ['Female','Male']

gender_ratio['Not Survived'] = 100 - gender_ratio['Survived']

gender_ratio.stack().to_frame()
#Compare cross validation score 

cv1 = cross_val_score(log_model,train,y,cv=10).round(5)

cv2 = cross_val_score(clf,train,y,cv=10).round(5)
#W/o tuning

print(f"Cv score : w/o optimasation -- Mean_score  {cv1.mean()*100} Std {cv1.std()*100}")

#with tuning

print(f"Cv score : with optimasation -- Mean_score  {cv2.mean()*100} Std {cv2.std()*100}")
#Kaggle doesn't support latest verison of sklearn modules which makes it hard to implement latest features



# from sklearn.metrics import plot_roc_curve



# #Plotting Roc curve for diff models

# fig,ax = plt.subplots(1,1,figsize=(10,5))

# plot_roc_curve(log_model,x_val,y_val,ax=ax)

# plot_roc_curve(clf,x_val,y_val,ax=ax);
tree = DecisionTreeClassifier()

tree.fit(x_train,y_train)



#Predictions

tree_pred = tree.predict(x_val)



#Model Evaluate

model_eval(tree)
#finding the best parameter

params = {'criterion': ['gini', 'entropy'],'max_depth':[1,2,3,4,5],'random_state': [0]}

tr_grid = GridSearchCV(tree,params)

tr_grid.fit(x_train,y_train)



#Make prediction

Gtr_pred = tr_grid.predict(x_val) 



#Find the model best parameter

print('Best parameters' , tr_grid.best_params_)

print ('\n Best score :', tr_grid.best_score_ * 100)
#model assessment

model_eval(tr_grid)
#Compare cross validation score 

cv1 = cross_val_score(tree,train,y,cv=10).round(5)

cv2 = cross_val_score(tr_grid,train,y,cv=10).round(5)



#W/o tuning

print(f"Cv score : w/o  optimasation -- Mean_score  {cv1.mean()*100} ---Std {cv1.std()*100}")

#with tuning

print(f"Cv score : with optimasation -- Mean_score  {cv2.mean()*100} ---Std {cv2.std()*100}")
# #Plotting Roc curve for diff curves

# fig,ax = plt.subplots(1,1,figsize=(10,5))

# #Model with deafult settings

# plot_roc_curve(tree,x_val,y_val,ax=ax)

# #Best model

# plot_roc_curve(tr_grid,x_val,y_val,ax=ax);
#Random forest 

rf = RandomForestClassifier(max_depth=4)

rf.fit(x_train,y_train)

rf_pred = rf.predict(x_val)



#model evaluate

model_eval(rf)
# Number of trees in random forest

n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 50)]



# Number of features to consider at every split

max_features = ['auto', 'sqrt']



# Maximum number of levels in tree

max_depth = [int(x) for x in range(1,15,2)]

max_depth.append(None)



# Minimum number of samples required to split a node

min_samples_split = [20,25,30,40]

# Minimum number of samples required at each leaf node

min_samples_leaf = [5,7,10,14]



# Create the grid

params = {     'criterion': ['gini', 'entropy'],

               'n_estimators': n_estimators,

               'max_features': max_features,

               'max_depth': max_depth,

               'min_samples_split': min_samples_split,

               'min_samples_leaf': min_samples_leaf,

                'random_state': [42]

         }



rf_grid = RandomizedSearchCV(rf,param_distributions=params)

rf_grid.fit(x_train,y_train)





#Find the model best parameter

print('Best parameters' , rf_grid.best_params_)

print ('\n Best score :', rf_grid.best_score_ * 100)
#Model assessment

model_eval(rf_grid)
# #Plotting Roc curve for diff curves

# fig,ax = plt.subplots(1,1,figsize=(10,5))

# #Model with deafult settings

# plot_roc_curve(rf,x_val,y_val,ax=ax)

# #Best model

# plot_roc_curve(rf_grid,x_val,y_val,ax=ax);
#PLotting feature importance

pd.DataFrame(rf.feature_importances_,index=x_train.columns).sort_values(by=0).plot.barh(

             color='salmon',

             title='Forest Feature Importance',figsize=(10,5));
#Lets combine to get a better idea



train_full = pd.concat([x_train,y_train],axis=1)



#Frequency analyse w.r.t survived

print(pd.crosstab(train_full.cabin_NA,train_full.Survived))



train_full.cabin_NA.value_counts().plot.pie(autopct='%1.1f%%',shadow=True);
# group data by Survived vs Non-Survived

# and find nulls for cabin

cab_ratio = train_full.groupby(['Survived'])['cabin_NA'].mean().round(4) *100

cab_ratio.index = ['Not Survived','Survived']

cab_ratio.to_frame()
#Visualizing those numbers to better understanding

fig,(ax1,ax2) =plt.subplots(1,2,figsize=(14,6))



pd.crosstab(train_full.cabin_NA,train_full.Survived).plot.bar(ax=ax1)

ax1.legend(['Not survived','survived'])



pd.crosstab(train_full.cabin_NA,train_full.Survived).plot.bar(stacked=True,ax=ax2,label=['Not survived','survived'])

ax2.legend(['Not survived','survived']);
gbm = XGBClassifier(

 #learning_rate = 0.02,

 n_estimators= 2000,

 max_depth= 3,

 min_child_weight= 2,

 #gamma=1,

 gamma=0.9,                        

 subsample=0.8,

 colsample_bytree=0.8,

 objective= 'binary:logistic',

 nthread= -1,

 scale_pos_weight=1).fit(x_train, y_train)
pd.DataFrame(gbm.feature_importances_,index=x_train.columns).sort_values(by=0).plot.barh(title='XGBoost Feature IMPortance');
#Xgb with optimisation

xg_pred = gbm.predict(test)



#Model assessment

model_eval(gbm)
#XGBoost with default settings

xgb = XGBClassifier()

xgb.fit(x_train,y_train)

xg_pred = xgb.predict(x_val)



#Model assessment

model_eval(xgb)
#SVM with default parameters

np.random.seed(42) #For reprodcutivity

svm = SVC()

svm.fit(x_train,y_train,)

sv_pred = svm.predict(x_val)



#Svm model assessment

print("Svm (with deafult parameters) training accuracy",accuracy_score(y_train,svm.predict(x_train))*100)

print("Svm (with deafult parameters) test accuracy    ",accuracy_score(y_val,sv_pred)*100)
#Searching best parametres to our model

params= {'C':[0.1,1,10,100],'gamma':[0.1,0.01,0.001,.0001],'random_state':[42]}



#Grid search

grid_svm = GridSearchCV(svm,param_grid=params,refit=True,scoring='accuracy',verbose=1)

grid_svm.fit(x_train,y_train)



print (f"Best hyperparameters \n:{grid_svm.best_params_}")

print (f"Grid search best Score {grid_svm.best_score_}")
#Make predictions with best SVM parameters

svm_pred = grid_svm.predict(x_val)



#Model evaluate



print(f"SVM after tuning parameters \n")

print("Accuary Score Train_set :",accuracy_score(y_train,grid_svm.predict(x_train))*100)

print(f"Accuary Score Test_set  : { accuracy_score(y_val,svm_pred)*100}")
#Making predictiions using SVM(after tuned model)

#submission1 = pd.DataFrame([ID_data,grid_svm.predict(test)]).T

#submission1.to_csv('sub_svm_tuned.csv',index=False)
#Soft voting

from sklearn.ensemble import VotingClassifier

ensemble_lin_rbf=VotingClassifier(estimators=[('KNN',KNeighborsClassifier(n_neighbors=15)),

                                              ('RBF',SVC(probability=True,kernel='rbf',C=1,gamma=0.1)),

                                              ('RFor',RandomForestClassifier(n_estimators=500,random_state=0)),

                                              ('LR',LogisticRegression(C=100)),

                                              ('DT',tr_grid)

                                             ], 

                       voting='soft').fit(x_train,y_train)

print('The accuracy for ensembled model is:',ensemble_lin_rbf.score(x_val,y_val))

cross=cross_val_score(ensemble_lin_rbf,train,y, cv = 10,scoring = "accuracy")

print('The cross validated score is',cross.mean())
#Hard Voting

from sklearn.ensemble import VotingClassifier

ensemble_lin_rbf=VotingClassifier(estimators=[('KNN',KNeighborsClassifier(n_neighbors=15)),

                                              ('RBF',SVC(probability=True,kernel='rbf',C=1,gamma=0.1)),

                                              ('RFor',RandomForestClassifier(n_estimators=500,random_state=0)),

                                              ('LR',LogisticRegression(C=100)),

                                              ('DT',tr_grid)

                                             ], 

                       voting='hard').fit(x_train,y_train)

print('The accuracy for ensembled model is:',ensemble_lin_rbf.score(x_val,y_val))

cross=cross_val_score(ensemble_lin_rbf,train,y, cv = 10,scoring = "accuracy")

print('The cross validated score is',cross.mean())
## Kaggle doesn't support latest verison of sklearn modules which makes it hard to implement latest features



#Stacking classifier

# from sklearn.ensemble import StackingClassifier



# #Base level estimators

# estimators =[('rf',rf_grid),('tr',tr_grid),('gbm',gbm), ('xgb',xgb),('log',log_model)]



# #create a stacking model

# stacking1 = StackingClassifier(estimators=estimators,

#                               final_estimator =SVC(kernel = 'rbf')

#                               )



# #Training the stackers

# stacking1.fit(x_train,y_train)
# #Stacking classifier

# from sklearn.ensemble import StackingClassifier



# #Base level estimators

# estimators =[('rf',rf_grid),('tr',tr_grid),('gbm',gbm), ('xgb',xgb),('log',log_model)]



# #create a stacking model

# stacking2 = StackingClassifier(estimators=estimators,

#                               final_estimator =LogisticRegression(C=100))



# #Training the stackers

# stacking2.fit(x_train,y_train)
# #Stacking model assessment(train-set)

# print('Stacking(svm) train-set score  ',stacking1.score(x_train,y_train)*100)

# print('Stacking(logit) train-set score',stacking2.score(x_train,y_train)*100)



# #Stacking model assessment(Test-set)

# print('\nStacking(svm) test-set score   ',stacking1.score(x_val,y_val)*100)

# print('Stacking(logit) test-set score ',stacking2.score(x_val,y_val)*100)
#define convenient function 

from sklearn.model_selection import cross_val_predict



def model_fit(algorithm, X_train, y_train, cv):

    

    #cv fold test

    y_pred =cross_val_predict(algorithm, X_train, y_train, cv = cv)

    

    #cv score

    cv_acc = accuracy_score(y_train, y_pred)

    

    #error rate

    error = np.mean(y_train != y_pred)

    

    return y_pred, cv_acc,error
#define algorithms

knn = KNeighborsClassifier(n_neighbors=15)

tree_entr = DecisionTreeClassifier(criterion = 'entropy')

svm_lin = SVC(kernel = 'linear')

svm_rbf = SVC(kernel = 'rbf')

log = LogisticRegression(C=100)

rf = RandomForestClassifier()
#run function to get scores for all above models

y_pred_knn, cv_acc_knn,err1 = model_fit(knn, x_train, y_train, 10)

y_pred_tree_gini, cv_acc_tree_best,err2 = model_fit(tr_grid, x_train, y_train, 10) #With best parameters

y_pred_tree_entr, cv_acc_tree_ent,err3 = model_fit(tree_entr, x_train, y_train, 10)

y_pred_svm_lin, cv_acc_svm_lin,err4 = model_fit(svm_lin, x_train, y_train, 10)

y_pred_svm_rbf, cv_acc_svm_rbf,err5 = model_fit(svm_rbf, x_train, y_train, 10)

y_pred_log, cv_acc_log,err6 = model_fit(log, x_train, y_train, 10)

y_pred_rf, cv_acc_rf,err7 = model_fit(rf, x_train, y_train, 10)
#create dataframe to view how our models score

models_eval = pd.DataFrame({

    'Model': ['KNN', 'Decision Tree (Grid_search)', 'Decision Tree (Entropy)', 

              'SVM (Linear)', 'SVM (RBF)', 

              'Logistic Growth', 'Random Forest'],

    'Score': [

        cv_acc_knn, 

        cv_acc_tree_best,      

        cv_acc_tree_ent, 

        cv_acc_svm_lin, 

        cv_acc_svm_rbf, 

        cv_acc_log,

        cv_acc_rf,

    ],'Error_Rate':[

        err1,err2,err3,err4,err5,err6,err7

    ]})

print('---Cross-validation Accuracy Scores (Train_set)---')

models_eval.sort_values(by='Score',ascending=False)
# #Kaggle doesn't support latest verison of sklearn modules which makes it hard to implement latest features



# #Lets plot confuison matrix (https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py)

# from sklearn.metrics import plot_confusion_matrix



# np.set_printoptions(precision=3)

# # Plot non-normalized confusion matrix

# titles_options = [("Confusion matrix, without normalization", None),

#                   ("Normalized confusion matrix", 'true')]



# #svm with default settings

# for title, normalize in titles_options:

#     disp = plot_confusion_matrix(svm, x_val,y_val,

#                                  display_labels=['Not Survived','Survived'],

#                                  cmap=plt.cm.Blues,

#                                  normalize=normalize)

#     disp.ax_.set_title(title)



#     print(title)

#     print(disp.confusion_matrix)
from sklearn.metrics import recall_score



#TPR or Sensitivity

recall_score(y_val,sv_pred)*100
# ## Plotting CM

# np.set_printoptions(precision=3)



# # Plot non-normalized confusion matrix

# titles_options = [("Confusion matrix, without normalization", None),

#                   ("Normalized confusion matrix", 'true')]



# #svm with default settings

# for title, normalize in titles_options:

#     disp = plot_confusion_matrix(grid_svm, x_val,y_val,

#                                  display_labels=['Not Survived','Survived'],

#                                  cmap=plt.cm.Blues,

#                                  normalize=normalize)

#     disp.ax_.set_title(title)



#     print(title)

#     print(disp.confusion_matrix)
#TPR or Sensitivity

recall_score(y_val,svm_pred)*100
# #compare precision-Recall score

# from sklearn.metrics import plot_precision_recall_curve



# #compare both models

# fig,ax = plt.subplots(1,1,figsize=(12,5))

# plot_precision_recall_curve(svm, x_val,y_val,ax=ax)

# ax.set_title('PR Curve')

# plot_precision_recall_curve(grid_svm, x_val,y_val,ax=ax);
#Making predictiions using SVM(after tuned model)

submission1 = pd.DataFrame([ID_data,grid_svm.predict(test)]).T



submission1.to_csv('sub_svm_tuned.csv',index=False)

submission1.columns = ['Id','Survived']

submission1.head(8)