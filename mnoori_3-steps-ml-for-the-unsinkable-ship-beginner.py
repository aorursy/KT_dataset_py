#importing modules and reading the train and test sets.

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import warnings

warnings.filterwarnings("ignore")





train=pd.read_csv('../input/train.csv')

test=pd.read_csv('../input/test.csv')



print('Shape of train set is:',train.shape)

print('Shape of test set is:',test.shape)

train.head()
#using pivot tables to get an idea how much of each sex survived.

sex_pivot=train.pivot_table(index='Sex',values='Survived')

sex_pivot
#pivot table of Pclass

class_pivot=train.pivot_table(index='Pclass',values='Survived')

class_pivot
train.Age.describe()
survived = train[train["Survived"] == 1]

died = train[train["Survived"] == 0]

fig, ax=plt.subplots(figsize=(8,6))

survived["Age"].plot.hist(alpha=0.5,color='red',bins=50)

died["Age"].plot.hist(alpha=0.5,color='blue',bins=50)

plt.legend(['Survived','Died'])

plt.show()
#I define a function, so I can re-use it on test set as well.

def cut_age(df,cut_limits,label_names):

    df['Age']=df['Age'].fillna(-.5)

    df['Age_cats']=pd.cut(df['Age'],cut_limits,labels=label_names)

    return df



cut_limits=[-1,0,5,12,18,35,60,100] #These limits are something to alter in the future

label_names=['Missing','Infant','Child','Teenager','Young Adult','Adult','Senior']



#we defined a function to apply to both train and test sets.

train=cut_age(train,cut_limits,label_names)

test=cut_age(test,cut_limits,label_names)



train.pivot_table(index='Age_cats',values='Survived').plot.bar()
#again, defining a function in order to be able to reuse on test set.

def create_dummies(df,col_name):

    dummies=pd.get_dummies(df[col_name],prefix=col_name)

    df=pd.concat([df,dummies],axis=1)

    return df

train = create_dummies(train,"Pclass")

test = create_dummies(test,"Pclass")

train = create_dummies(train,"Age_cats")

test = create_dummies(test,"Age_cats")

train = create_dummies(train,"Sex")

test = create_dummies(test,"Sex")



#let's see how our columns look now:

train.columns
#importing LogiticRegression class from sklearn

from sklearn.linear_model import LogisticRegression



#Although we have a test set, but that is only for submission purposes. We should still split our train set into...

#...two seperate sets. This helps us measure the accuracy of our model.

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score



lr=LogisticRegression()



#Let's only select sex, pclass, and age related columns.

features=['Pclass_1','Pclass_2', 'Pclass_3','Age_cats_Missing', 'Age_cats_Infant','Age_cats_Child', 'Age_cats_Teenager', 

          'Age_cats_Young Adult','Age_cats_Adult', 'Age_cats_Senior', 'Sex_female', 'Sex_male']

target='Survived'



all_X=train[features]

all_y=train[target]



#we will hold out our original test model

holdout=test



#splitting train set into two seperate sets. I use 80% of data for training and 20% for testing.

train_X,test_X,train_y,test_y=train_test_split(all_X,all_y,test_size=0.2,random_state=0)



#let's now fit the model and make predictions.

lr.fit(train_X,train_y)

predictions=lr.predict(test_X)



#calculating accuracy using sklearn function

accuracy=accuracy_score(test_y,predictions)

print('Accuracy of model is {0:.2f} percent'.format(accuracy*100))
#I use the cross validation score function of sklearn.

from sklearn.model_selection import cross_val_score

import numpy as np



lr=LogisticRegression()

scores=cross_val_score(lr,all_X,all_y,cv=10) #10 folds

accuracy=np.mean(scores)

print(scores)

print('Cross-validated accuracy of model is {0:.2f} percent'.format(accuracy*100))
lr=LogisticRegression()

lr.fit(all_X,all_y)

holdout_predictions=lr.predict(holdout[features])
holdout_ids=holdout['PassengerId']

submission_df={

    'PassengerId':holdout_ids,

    'Survived':holdout_predictions,

                  }



submission=pd.DataFrame(submission_df)

submission_file=submission.to_csv('TitanicSubmission.csv',index=False)
cols=['SibSp','Parch','Fare','Cabin','Embarked']

train[cols].describe(include='all',percentiles=[])
train['Embarked']=train['Embarked'].fillna('S')

#As you remember, whatever we do on train data set, we will do the same on test (holdout).

holdout['Embarked']=holdout['Embarked'].fillna('S')





#holdout has one missing value in Fare columns, let's replace it with mean of that column.

holdout['Fare']=holdout['Fare'].fillna(train['Fare'].mean())



holdout[cols].describe(include='all',percentiles=[])
#creating dummy variables for Embarked

train = create_dummies(train,"Embarked")

holdout = create_dummies(holdout,"Embarked")
#rescaling the numerical columns

from sklearn.preprocessing import minmax_scale

cols=['SibSp','Parch','Fare']

for col in cols:

    train[col + "_scaled"] = minmax_scale(train[col])

    holdout[col + "_scaled"] = minmax_scale(holdout[col])
train.columns
columns=['Pclass_1', 'Pclass_2', 'Pclass_3', 'Age_cats_Missing', 'Age_cats_Infant',

       'Age_cats_Child', 'Age_cats_Teenager', 'Age_cats_Young Adult',

       'Age_cats_Adult', 'Age_cats_Senior', 'Sex_female', 'Sex_male',

       'Embarked_C', 'Embarked_Q', 'Embarked_S', 'SibSp_scaled',

       'Parch_scaled', 'Fare_scaled']

       

lr=LogisticRegression()

lr.fit(train[columns],train['Survived'])



#finding the coefficients

coeffs=lr.coef_

importance_of_features=pd.Series(coeffs[0],index=train[columns].columns).abs().sort_values()

importance_of_features.plot.barh()
train['Fare'].hist(bins=20,range=(0,100))
# defining a function for binning fare column

def process_fare(df,cut_points,lebel_names):

    df['Fare_cats']=pd.cut(df['Fare'],cut_points,labels=label_names)

    return df



cut_points=[0,12,50,100,1000]

label_names=['0-12','12-50','50-100','100+']



#cutting the fare column using our function

train=process_fare(train,cut_points,label_names)

holdout=process_fare(test,cut_points,label_names)



#creating dummy columns:

train=create_dummies(train,'Fare_cats')

holdout=create_dummies(test,'Fare_cats')
train[['Name','Cabin']].head(10)
#creating a mapping dictionary

titles={

    "Mr" :         "Mr",

    "Mme":         "Mrs",

    "Ms":          "Mrs",

    "Mrs" :        "Mrs",

    "Master" :     "Master",

    "Mlle":        "Miss",

    "Miss" :       "Miss",

    "Capt":        "Officer",

    "Col":         "Officer",

    "Major":       "Officer",

    "Dr":          "Officer",

    "Rev":         "Officer",

    "Jonkheer":    "Royalty",

    "Don":         "Royalty",

    "Sir" :        "Royalty",

    "Countess":    "Royalty",

    "Dona":        "Royalty",

    "Lady" :       "Royalty"    

}



def titles_cabin_process(df):

    #extracting titles from 'Name' column

    extracted_titles=df['Name'].str.extract(' ([A-Za-z]+)\.',expand=False)

    df['Title']=extracted_titles.map(titles)

    

    #extracting first letter of 'Cabin' column

    df['Cabin_type']=df['Cabin'].str[0]

    df['Cabin_type']=df['Cabin_type'].fillna('Unknown')

    

    #creating dummy variables

    df=create_dummies(df,'Title')

    df=create_dummies(df,'Cabin_type')

    

    return df

    

train=titles_cabin_process(train)

holdout=titles_cabin_process(holdout)
train.columns
#writing a function that graphs nice heatmaps

def plot_corr_heatmap(df):

    import seaborn as sns

    corrs=df.corr()

    sns.set(style='white')

    mask=np.zeros_like(corrs,dtype=np.bool)

    mask[np.triu_indices_from(mask)]=True

    

    f,ax=plt.subplots(figsize=(11,9))

    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    

    sns.heatmap(corrs, mask=mask, cmap=cmap, vmax=.3, center=0,square=True, linewidths=.5, cbar_kws={"shrink": .5})

    plt.show()
ready_columns=['Pclass_1',

       'Pclass_2', 'Pclass_3', 'Age_cats_Missing', 'Age_cats_Infant',

       'Age_cats_Child', 'Age_cats_Teenager', 'Age_cats_Young Adult',

       'Age_cats_Adult', 'Age_cats_Senior', 'Sex_female', 'Sex_male',

       'Embarked_C', 'Embarked_Q', 'Embarked_S', 'SibSp_scaled',

       'Parch_scaled', 'Fare_cats_0-12',

       'Fare_cats_12-50', 'Fare_cats_50-100', 'Fare_cats_100+',

        'Cabin_type_A', 'Cabin_type_B', 'Cabin_type_C',

       'Cabin_type_D', 'Cabin_type_E', 'Cabin_type_F', 'Cabin_type_G',

       'Cabin_type_T', 'Cabin_type_Unknown', 'Title_Master', 'Title_Miss',

       'Title_Mr', 'Title_Mrs', 'Title_Officer', 'Title_Royalty']



plot_corr_heatmap(train[ready_columns])
final_ready_columns=['Pclass_1','Pclass_3', 'Age_cats_Missing', 'Age_cats_Infant',

       'Age_cats_Child', 'Age_cats_Teenager', 'Age_cats_Young Adult',

       'Age_cats_Adult', 'Embarked_C', 'Embarked_S', 'SibSp_scaled',

       'Parch_scaled', 'Fare_cats_0-12', 'Fare_cats_12-50', 'Fare_cats_50-100',

        'Cabin_type_A', 'Cabin_type_B', 'Cabin_type_C',

       'Cabin_type_D', 'Cabin_type_E', 'Cabin_type_F', 'Cabin_type_G',

       'Cabin_type_Unknown', 'Title_Master', 'Title_Miss',

       'Title_Mr', 'Title_Mrs', 'Title_Officer']
from sklearn.feature_selection import RFECV



all_X=train[final_ready_columns]

all_y=train['Survived']



lr=LogisticRegression()



#just like any other sklearn class, we will instantiate the class first, then fit the model

selector=RFECV(lr,cv=10)

selector.fit(all_X,all_y)



#usuing RFECV.support_ we can find the most import features. It provides a boolean list.

optimized_columns=all_X.columns[selector.support_]

optimized_columns
all_X = train[optimized_columns]

all_y = train["Survived"]

lr=LogisticRegression()

scores=cross_val_score(lr,all_X,all_y,cv=10)

accuracy=scores.mean()

print('Cross-validated accuracy of model is {0:.2f} percent'.format(accuracy*100))
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV



hyperparameters = {"criterion": ["entropy", "gini"],

                   "max_depth": [5, 10],

                   "max_features": ["log2", "sqrt"],

                   "min_samples_leaf": [1, 5],

                   "min_samples_split": [3, 5],

                   "n_estimators": [6, 9]

}



clf = RandomForestClassifier(random_state=1)

grid = GridSearchCV(clf,param_grid=hyperparameters,cv=10)



grid.fit(all_X, all_y)



best_params = grid.best_params_

best_score = grid.best_score_



print('Cross-validated accuracy of model is {0:.2f} percent'.format(best_score*100))