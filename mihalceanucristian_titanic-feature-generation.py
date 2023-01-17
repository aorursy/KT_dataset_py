import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline   

# this last command allows us to visualize plots in this notebook environment
df1=pd.read_csv('/kaggle/input/titanic/train.csv')   #save a dataframe of the csv files that were given to us 

df2=pd.read_csv('/kaggle/input/titanic/test.csv')
df1['train']=1  # we add a new column, which will later tell us what observations belong to which set(training/testing)

df2['train']=0
# df1 will be our training set, df2 the test set



df=pd.concat([df1,df2],axis=0,sort=False)



# we concatenate the two in order to get a better understanding of the data we are working with
df.head()
df.info()
df.describe().transpose()
df.isnull().sum()

#this way we check how many null values there are on each column
sns.heatmap(df.isnull(),cbar=False,yticklabels=False,cmap='viridis')



#in order to better visualize columns with missing values, instead of calculating percentage



# we shouldn't worry about the missing values on 'Survived' column, as they are the ones we are trying to predict
# the code for automatic EDA is the following, but in this project we'll focus on creating plots ourselves

'''

import pandas_profiling 

display(pandas_profiling.ProfileReport(df))

'''
sns.set_style('whitegrid')

plt.figure(figsize=(12,4))

df1['Fare'].plot(kind='hist',bins=70)

#As we can see, this is not a normal distribution, meaning majority of those who embarked purchased cheaper tickets
plt.figure(figsize=(12,8))

sns.boxplot(x='Pclass',y='Fare',hue='Survived',data=df1)

# As we might have guessed, a higher class ticket means a spike in 'Fare'
sns.distplot(df1['Age'].dropna())
sns.countplot(x='Survived',data=df1,hue='Sex',palette='RdBu_r')
sns.countplot(x='Pclass',data=df1,hue='Survived')
df1[df1['Fare']>500][['Name','Survived']]

#Out of pure curiosity, we checked if most expensive tickets ultimately lead to survival
sns.countplot(x='Embarked',data=df1,hue='Survived')
sns.countplot(x='SibSp',data=df1)
sns.countplot(x='Parch',data=df1)
sns.jointplot(x='Age',y='Fare',data=df,kind='hex',cmap='magma')

# Meaning there were many young adults on board, who opted for the 3rd class tickets
# A very useful practice is to gain a better understanding of the data through the means of pivot tables



pd.pivot_table(df1,index='Survived',columns='Pclass',values='Fare',aggfunc='count')



# Pivot table are a very powerful weapon of a data scientist, but require a little getting used to :))
sns.heatmap(df.corr())

# A heatmap is often useful to identify the correlations between various columns( how they influence each other )

# Lighter the color, the stronger the dependance
# Below, we'll separate numerical columns from categorical ones
numerical_columns=[col for col in df1.columns if df1.dtypes[col] in ['int64','float64'] and col not in['Survived','train']]

numerical_columns
categorical_columns=[col for col in df1.columns if df1.dtypes[col] == 'O']

categorical_columns
df[numerical_columns].isnull().sum()
sns.boxplot(data=df1,x='Pclass',y='Age')
#We'll imput age where is missing, based on mean age for respective Pclass 



def imput_age(cols):

    Age=cols[0]

    Pclass=cols[1]

    

    if pd.isnull(Age):

        if Pclass==1:

            return 37

        elif Pclass==2:

            return 29

        else:

            return 24

    else:

        return Age



df['Age']=df[['Age','Pclass']].apply(imput_age,axis=1)



# Create a new feature, 'child' if age is lower than 18



def is_child(x):

    if x>=18:

        return 0

    else: 

        return 1







df['child']=df['Age'].apply(is_child)
df['Fare']=df['Fare'].fillna(df[df['train']==1]['Fare'].mean())   # Doing it this way beacause of the problems 

# encountered using SimpleImputer 
# For some reason, using SimpleImputer resulted in an error which stated the kernel 

# could not perform an action on a slice of the dataframe



'''

df1=pd.read_csv('train.csv')

df2=pd.read_csv('test.csv')

df1['train']=1

df2['train']=0





from sklearn.impute import SimpleImputer

numerical_transformer=SimpleImputer(strategy='mean')

df1[numerical_columns] = numerical_transformer.fit_transform( df1[numerical_columns] )

df2[numerical_columns] = numerical_transformer.transform( df2[numerical_columns] )

df=pd.concat([df1,df2],axis=0,sort=False)

'''

# In this final version it seemed to work, but at the beginning it definitely did not work:



'''

from sklearn.impute import SimpleImputer

numerical_transformer=SimpleImputer(strategy='mean')

df[df[train]==1][numerical_columns] = numerical_transformer.fit_transform( df[df[train]==1][numerical_columns] )

df[df[train]==0][numerical_columns] = numerical_transformer.transform( df[df[train]==1][numerical_columns] )

df=pd.concat([df1,df2],axis=0,sort=False)

'''
df[numerical_columns].isnull().sum() # And we are good to go :))
categorical_columns
df[categorical_columns].isnull().sum()
df['Embarked']=df['Embarked'].fillna('S')

#Based on the plot created in the EDA section, 'S' is the most frequent value in the 'Embarked' column
df['rank']=df['Name'].apply(lambda x: x.split(',')[1].split('.')[0])



# We created a new column, 'rank', which contains the way certain people are adressed
df['rank'].value_counts(ascending=False)
df[df['rank']==' Capt']['Survived']



# Sadly, the Captain went down with the ship
def never_let_go(x):

    if 'Jack' in x:

        print(x)



df['Name'].apply(never_let_go)



# Well, it's not the Jack we were searching for, but it was worth trying :))
# Let's turn our attention over to the 'Ticket' column



# Through the listed functions, we are able to extract the letters and later the number of the tickets



def letter_separator(x):

    mylist=[letter for letter in x if letter.isalpha() ]

    if len(mylist):

        return ''.join(mylist)

    else:

        return 'None'

    

def digit_separator(x):

    if not str(x.split()[-1]).isdigit():

        return 0

    else:

        return int(str(x.split()[-1]))    

    

df['ticket_letters']=df['Ticket'].apply(letter_separator)

df['ticket_number']=df['Ticket'].apply(digit_separator)   
df.head()
plt.figure(figsize=(20,4))

sns.countplot(data=df,x='ticket_letters')
df['ticket_letters'].value_counts(ascending=False)[1:]  #avoid including 'None'
# Finally, the 'Cabin' column



df['Cabin'].unique()
df['cabin_letter']=df['Cabin'].apply(lambda x: str(x)[0]) 
pd.pivot_table(df,columns='cabin_letter',index='Survived',values='Ticket',aggfunc='count')



# missing cabin rows are marked by the 'n'
# New columns in place, we can afford to drop the former ones



df.drop(['Name','PassengerId'],axis=1,inplace=True)

df.drop('Cabin',axis=1,inplace=True)

df.drop('Ticket',axis=1,inplace=True)
# Our numerical and categorical columns have thus changed:



numerical_columns=[col for col in df.columns if df.dtypes[col] in ['int64','float64'] and col not in['Survived','train']]

numerical_columns
categorical_columns=[col for col in df.columns if df.dtypes[col] == 'O']

categorical_columns
import itertools
copy_df=df.copy()
interactions=pd.DataFrame(index=copy_df.index)
for col1, col2 in itertools.combinations(categorical_columns,2):

    new_col_name='_'.join([col1,col2])

    new_values=copy_df[col1].map(str)+'-'+copy_df[col2].map(str)

    interactions[new_col_name]=new_values
interactions



# We had 5 categorical columns, and we asociated every pair possible

#      ---> Combinations of 5 taken 2 at a time = (5!)/[(3!)*(2!)] = 10 new columns
# The only reason we created a copy earlier was because I feared I would cause damage to 

# the dataframe by concatenating something the wrong way :))
df=pd.concat([interactions,df],axis=1,sort=False)
df.head()
# Of course, the categorical columns have changed once again:



categorical_columns=[col for col in df.columns if df.dtypes[col] == 'O']

categorical_columns



# While numerical ones stayed the same
df.isnull().sum()

#Since only the values we want to predict are missing, we can go ahead and encode out categorical values
# Basically, we encode categorical features differently based on their associated cardinality



# Cardinality means the number of unique values in a column.



# We will use OneHotEncoder for features with low cardinality, because these ones do not add so many new columns to

# the dataset, which would make it difficult for the computer to process. 
# We first separate categorical columns in high cardinality ones vs low cardinality:



#high-cardinality categorical columns

hccc=[col for col in df.columns if df.dtypes[col] == 'O' and df[col].nunique()>10]

hccc
#low cardinality

lccc=[col for col in categorical_columns if col not in hccc]

lccc
trainset=df[df['train']==1]

testset=df[df['train']==0]
# I'll show with you my failures in trying to label encode hccc with the basic LabelEncoder



# The problem I had run into was that the testset contained new values on some columns



# For example, on column: 'Sex_rank' we might not have found the value 'female-the Countess' in the trainset, 

# but rather stumble upon it in the testset, in which case the encoder would not know what to do
# Failed Attempt nr1:

'''

from sklearn import preprocessing

for feature in hccc:

  encoded=preprocessing.LabelEncoder().fit(trainset[feature])

  df[feature+'_labels']=encoded.transform(df[feature])

#fit only on dataset to avoid leakage

#trebuie pr fiecare coloana in parte fit, la OneHot e la general ca pune doar 0 sau 1'''
# Failed Attempt nr2:

'''from sklearn import preprocessing

for feature in hccc:

  encoder=preprocessing.LabelEncoder()

  encoder.fit(trainset[feature])

  encoded_train_feature=encoder.transform(trainset[feature])



  trainset_feature_map=pd.DataFrame({'train_feature':trainset[feature],'encoded_train_feature':encoded_train_feature})'''
# Failed Attempt nr3:



'''

encoder=preprocessing.LabelEncoder()

encoder.fit(trainset['rank'])

encoded_train_feature=encoder.transform(trainset['rank'])



trainset_feature_map=pd.DataFrame({'train_feature':trainset['rank'],'encoded_train_feature':encoded_train_feature})

trainset_feature_map=trainset_feature_map.drop_duplicates()



print(trainset_feature_map)



testset_feature=pd.DataFrame({'testset_feature':testset['rank']})

testset_feature_unique=testset_feature.drop_duplicates()





print(testset_feature_unique)



#pana aici e bine



def select(x):

  if len(trainset_feature_map[trainset_feature_map['train_feature']==x]):

    return trainset_feature_map[trainset_feature_map['train_feature']==x]['encoded_train_feature']

  else:

    return len(trainset_feature_map)+1





testset_feature_unique['encoded_test_feature']=testset_feature_unique['testset_feature'].apply(select)



print(testset_feature_unique)







# print(trainset_feature_map[trainset_feature_map['train_feature']==' Mr']['encoded_train_feature'])

  # Be carefull! There is a spacebar right before!



'''
# So after serching for 3 days, I managed to find the solution on StackOverflow. Unfortunatelly, 

# I cannot trace it back to give credit to the author



# What the following function does, is basically, it does everything a LabelEncoder does, 

# only it assigns any new value in the testset, the highest label in the training set +1
from sklearn.preprocessing import LabelEncoder

import numpy as np





class LabelEncoderExt(object):

    def __init__(self):

        """

        It differs from LabelEncoder by handling new classes and providing a value for it [Unknown]

        Unknown will be added in fit and transform will take care of new item. It gives unknown class id

        """

        self.label_encoder = LabelEncoder()

        # self.classes_ = self.label_encoder.classes_



    def fit(self, data_list):

        """

        This will fit the encoder for all the unique values and introduce unknown value

        :param data_list: A list of string

        :return: self

        """

        self.label_encoder = self.label_encoder.fit(list(data_list) + ['Unknown'])

        self.classes_ = self.label_encoder.classes_



        return self



    def transform(self, data_list):

        """

        This will transform the data_list to id list where the new values get assigned to Unknown class

        :param data_list:

        :return:

        """

        new_data_list = list(data_list)

        for unique_item in np.unique(data_list):

            if unique_item not in self.label_encoder.classes_:

                new_data_list = ['Unknown' if x==unique_item else x for x in new_data_list]



        return self.label_encoder.transform(new_data_list)
for feature in hccc:

    encoded=LabelEncoderExt().fit(trainset[feature])

    df[feature+'_labels']=encoded.transform(df[feature])
df.drop(hccc,axis=1,inplace=True)
df.head()
from sklearn.preprocessing import OneHotEncoder

OH_encoder=OneHotEncoder(handle_unknown='ignore',sparse=False)

#'handle_unknown' argument  helps us avoid errors when test data contains new values, which aren't in the

# training data

#'sparse' agument return the output as an array, and not as a huge, hard to use sparse matrix



OH_encoder.fit(trainset[lccc])



OH_encoded=pd.DataFrame(OH_encoder.transform(df[lccc]))



OH_encoded.index=df.index

OH_encoded  # same as pd.get_dummies
df=pd.concat([df,OH_encoded],axis=1,sort=False)

df.drop(lccc,axis=1,inplace=True)
df.head()
# Great!



# Let's have a look which features impact the target variable

# ('Survived') positively and which affect it negatively
trainset=df[df['train']==1]

testset=df[df['train']==0]



trainset.corr()['Survived'].sort_values(ascending=False)[1:].plot(kind='bar')



# I trust this plot serves the purpose mentioned above. If not, I'd love to see your suggestions in the comments!
# Using this technique we are able to determine which features are most relevant to predicting survival



# We fill the less useful columns with zeros( causing vaariance to drop to zero), 

# and later select columns with val()!=0
from sklearn.feature_selection import SelectKBest, f_classif

feature_columns=df.drop('Survived',axis=1).columns





# We found that in this case it's better to kepp all 39 feature variables, 

# but I encourage you to play arround with different values for k



selector=SelectKBest(f_classif,k=39)

X_new=selector.fit_transform(trainset[feature_columns],trainset['Survived'])

# X_new



selected_features=pd.DataFrame(selector.inverse_transform(X_new),

                               index=trainset.index,

                               columns=feature_columns)

selected_columns = selected_features.columns[selected_features.var() != 0]

df[selected_columns]
# Naive Bayes 65% accuracy



'''

X=trainset[selected_columns].values

y=trainset['Survived'].values



from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid=train_test_split(X,y,test_size=0.2,random_state=101)



from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()

scaler.fit_transform(X_train)

scaler.transform(X_valid)



from sklearn.naive_bayes import GaussianNB

classifier=GaussianNB()

classifier.fit(X_train,y_train)



pred_naive_bayes=classifier.predict(X_valid)



from sklearn.metrics import classification_report, confusion_matrix



print(confusion_matrix(pred_naive_bayes,y_valid))

print(classification_report(pred_naive_bayes,y_valid))

'''
# Logistic Regression 66% accuracy



'''

X=trainset[selected_columns].values

y=trainset['Survived'].values



from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid=train_test_split(X,y,test_size=0.2,random_state=101)



from sklearn.linear_model import LogisticRegression

classifier=LogisticRegression()

classifier.fit(X_train,y_train)



pred_logistic_regression=classifier.predict(X_valid)



from sklearn.metrics import classification_report, confusion_matrix



print(confusion_matrix(pred_logistic_regression,y_valid))

print(classification_report(pred_logistic_regression,y_valid))

'''
# #KNearestNeighbors 68% accuracy

'''

X=trainset[selected_columns].values

y=trainset['Survived'].values



from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid=train_test_split(X,y,test_size=0.2,random_state=101)



from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()

scaler.fit_transform(X_train)

scaler.transform(X_valid)



from sklearn.neighbors import KNeighborsClassifier

classifier=KNeighborsClassifier(n_neighbors=1)

classifier.fit(X_train,y_train)



pred_KNN=classifier.predict(X_valid)



from sklearn.metrics import classification_report, confusion_matrix



print(confusion_matrix(pred_KNN,y_valid))

print(classification_report(pred_KNN,y_valid))

'''

#Choosing the suitable K value ( Elbow Method )

'''

error_rate=[]

for i in range(1,40):

    from sklearn.neighbors import KNeighborsClassifier

    classifier=KNeighborsClassifier(n_neighbors=i)

    classifier.fit(X_train,y_train)

    pred_i=classifier.predict(X_valid)

    error_rate.append(np.mean(pred_i!=y_valid))   # average error rate



plt.plot(range(1,40),error_rate)

plt.xlabel('K values')

plt.ylabel('error rate')

'''

# It's called the Elbow Method beacause we select the k value where the steepest drop occured,

# which causes the plot to look like an elbow

# In this case, I think the 'elbow' occures at k=21



'''

from sklearn.neighbors import KNeighborsClassifier

classifier=KNeighborsClassifier(n_neighbors=21)

classifier.fit(X_train,y_train)



pred_KNN=classifier.predict(X_valid)



print(confusion_matrix(pred_KNN,y_valid))

print(classification_report(pred_KNN,y_valid))

'''
# Kernel SVM  55% accuracy

'''

X=trainset[selected_columns].values

y=trainset['Survived'].values



from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid=train_test_split(X,y,test_size=0.2,random_state=101)



from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()

scaler.fit_transform(X_train)

scaler.transform(X_valid)



from sklearn.svm import SVC

classifier=SVC(kernel='rbf')

classifier.fit(X_train,y_train)



pred_Kernel_SVM=classifier.predict(X_valid)



from sklearn.metrics import classification_report, confusion_matrix



print(confusion_matrix(pred_Kernel_SVM,y_valid))

print(classification_report(pred_Kernel_SVM,y_valid))

'''
# Random Forest 82%



X=trainset[selected_columns].values

y=trainset['Survived'].values



from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid=train_test_split(X,y,test_size=0.2,random_state=101)



from sklearn.ensemble import RandomForestClassifier

classifier=RandomForestClassifier(n_estimators=400)

classifier.fit(X_train,y_train)



pred_rfc=classifier.predict(X_valid)



from sklearn.metrics import classification_report, confusion_matrix



print(confusion_matrix(pred_rfc,y_valid))

print(classification_report(pred_rfc,y_valid))



# Cross validation:



from sklearn.model_selection import cross_val_score

scores=(-1)*cross_val_score(classifier,X,y,cv=5,scoring='neg_mean_absolute_error')

print(scores.mean())



# GridSearch:



# The way an ML algorithm works is it usually calculates the parameters which would result in the lowest MAE possible

# However, there are some other parameters(hyperparameters) which we can tune ourselves



from sklearn.model_selection import GridSearchCV



classifier=RandomForestClassifier()

param_grid =  {'n_estimators': [800,1000,1200],

                'bootstrap': [True,False],

                'max_depth': [5, 10, 15],

                'min_samples_leaf': [1,2],

                'min_samples_split': [2,3]}

grid=GridSearchCV(classifier,param_grid,n_jobs=-1,verbose=3,scoring='accuracy',cv=10)

grid.fit(X_train,y_train)

print(grid.best_params_)

pred_grid_rfc=grid.predict(X_valid)

print(confusion_matrix(pred_grid_rfc,y_valid))

print(classification_report(pred_grid_rfc,y_valid))

#XGBoost  82%



X=trainset[selected_columns].values

y=trainset['Survived'].values



from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid=train_test_split(X,y,test_size=0.2,random_state=101)



from xgboost import XGBClassifier

classifier=XGBClassifier()



classifier.fit(X_train,y_train,

        early_stopping_rounds=5,

        eval_set=[(X_valid,y_valid)],

        verbose=False)



pred_xgb=classifier.predict(X_valid)



from sklearn.metrics import classification_report, confusion_matrix



print(confusion_matrix(pred_xgb,y_valid))

print(classification_report(pred_xgb,y_valid))



from sklearn.model_selection import cross_val_score

scores=(-1)*cross_val_score(classifier,X,y,cv=5,scoring='neg_mean_absolute_error')

print(scores.mean())



classifier=XGBClassifier()

param_grid={'n_estimators':[100,500,1000],

            'learning_rate':[0.01,0.05,0.1],

            'C':[0.25,0.5,0.75,1],

            'gamma':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]}

grid=GridSearchCV(classifier,param_grid,n_jobs=-1,verbose=3,scoring='accuracy')

grid.fit(X_train,y_train)

print(grid.best_params_)

pred_xgb=grid.predict(X_valid)

print(confusion_matrix(pred_xgb,y_valid))

print(classification_report(pred_xgb,y_valid))
X=trainset[selected_columns].values

y=trainset['Survived'].values



from xgboost import XGBClassifier

xgb=XGBClassifier(n_estimators=100,learning_rate=0.01,C=0.25,gamma=0.3)



xgb.fit(X,y) # we train it on the whole trainset now, we used a valid set earlier to determine accuracy



predictions=xgb.predict(testset[selected_columns].values).astype(int)

#                         !    !    !

# It is paramount that you use the .astype(int) method as Kaggle requires 

# the predictions to be integers (0 or 1), and they initially are float64

#                         !    !    !

submission = pd.DataFrame({

        "PassengerId": df2['PassengerId'],

        "Survived": predictions

    })

submission.to_csv('my_titanic.csv',index=False)



# we created a dataframe containing the passenger Id's from the dataframe initially given to as, and next to

# each id we predicted the person's survival
predictions