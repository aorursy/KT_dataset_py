# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



import matplotlib.pyplot as plt

%matplotlib inline
print('Data exploration: \n     1st step: list features')



#save test filepath to variable for easier access



test_titanic_file_path='../input/titanic/test.csv'

titanic_test_data=pd.read_csv(test_titanic_file_path)

titanic_test_data_df=pd.DataFrame(titanic_test_data)



# save train filepath to variable for easier access

titanic_file_path = '../input/titanic/train.csv'

# read the data and store data in DataFrame titled melbourne_data

titanic_data = pd.read_csv(titanic_file_path) 

# print a summary of the data in Melbourne data



df=pd.DataFrame(titanic_data)



titanic_data_df=pd.DataFrame(titanic_data)

titanic_data.describe(include='all')

#titanic_data.head()
print('Data exploration: 2nd step: list features by main type \n')



s = (titanic_data_df.dtypes == 'object')

object_cols = list(s[s].index)



t = (titanic_data_df.dtypes == 'int')

int_cols = list(t[t].index)



f = (titanic_data_df.dtypes == 'float')

float_cols = list(f[f].index)



numeric_cols=int_cols+float_cols



print('object cols:',object_cols)

print('numeric cols:',numeric_cols)

#Notes for data cleaning
def age_group(i):

    if   (i >= 0)&(i < 10):

        return '[0,9]'

    elif (i >= 10)&(i < 20):

        return '[10,19]'

    elif (i >= 20)&(i < 30):

        return '[20,29]' 

    elif (i >= 30)&(i < 40):

        return '[30,39]'    

    elif (i >= 40)&(i < 50):

        return '[40,49]'

    elif (i >= 50)&(i < 60):

        return '[50,59]'

    elif (i >= 60)&(i < 70):

        return '[60,69]' 

    elif (i >= 70)&(i < 80):

        return '[70,99]' 

    else:

        return '[NaN]'

    



def fare_group(i):

    if   (i >= 0)&(i < 10):

        return '[0,10['

    elif (i >= 10)&(i < 20):

        return '[10,20['

    elif (i >= 20)&(i < 30):

        return '[20,30[' 

    elif (i >= 30)&(i < 40):

        return '[30,40['    

    elif (i >= 40)&(i < 100):

        return '[40,90['

    else:

        return '[90,1000['
#Exploring the impact of the age on survival rate

titanic_data_df['Age_group']=titanic_data_df['Age'].apply(age_group)



features_age=['Age_group','Survived',]

df_survived_per_age=titanic_data_df[features_age].groupby(['Age_group']).sum()

df_age_count=titanic_data_df[features_age].groupby(['Age_group']).count()



df_survive_rate=df_survived_per_age.join(df_age_count, lsuffix='_survived', rsuffix='_total')



df_survive_rate['survival_rate']=df_survive_rate['Survived_survived']/df_survive_rate['Survived_total']



    

plt.plot(df_survive_rate['survival_rate'])

plt.show()



df_survive_rate



#ce graphique montre que les personnes dont l'age n'est pas déterminé ont un taux de survie (29%)

#nettement plus faible que la moyenne (39%) > a voir comment on fait avec ces cas
#survivor rate per gender

features_gender=['Sex','Survived']



df_survived_per_gender=titanic_data_df[features_gender].groupby(['Sex']).sum()

df_gender_count=titanic_data_df[features_gender].groupby(['Sex']).count()



df_survive_rate_gender=df_survived_per_gender.join(df_gender_count, lsuffix='_survived', rsuffix='_total')



df_survive_rate_gender['survival_rate']=df_survive_rate_gender['Survived_survived']/df_survive_rate_gender['Survived_total']

 



plt.plot(df_survive_rate_gender['survival_rate'])

plt.show()



df_survive_rate_gender

#survivors per fare group



#Exploring the impact of the age on survival rate

titanic_data_df['Fare_group']=titanic_data_df['Fare'].apply(fare_group)



features_fare=['Fare_group','Survived',]

df_survived_per_fare=titanic_data_df[features_fare].groupby(['Fare_group']).sum()

df_fare_count=titanic_data_df[features_fare].groupby(['Fare_group']).count()



df_survive_rate_fare=df_survived_per_fare.join(df_fare_count, lsuffix='_survived', rsuffix='_total')



df_survive_rate_fare['survival_rate']=df_survive_rate_fare['Survived_survived']/df_survive_rate_fare['Survived_total']



    

plt.plot(df_survive_rate_fare['survival_rate'])

plt.show()



df_survive_rate_fare
#impact pf Pclass on survival rate



features_pclass=['Pclass','Survived']



df_survived_per_pclass=titanic_data_df[features_pclass].groupby(['Pclass']).sum()

df_pclass_count=titanic_data_df[features_pclass].groupby(['Pclass']).count()



df_survive_rate_pclass=df_survived_per_pclass.join(df_pclass_count, lsuffix='_survived', rsuffix='_total')



df_survive_rate_pclass['survival_rate']=df_survive_rate_pclass['Survived_survived']/df_survive_rate_pclass['Survived_total']

 



plt.plot(df_survive_rate_pclass['survival_rate'])

plt.show()



df_survive_rate_pclass

#impact pf SibSp on survival rate



features_sibsp=['SibSp','Survived']



df_survived_per_sibsp=titanic_data_df[features_sibsp].groupby(['SibSp']).sum()

df_sibsp_count=titanic_data_df[features_sibsp].groupby(['SibSp']).count()



df_survive_rate_sibsp=df_survived_per_sibsp.join(df_sibsp_count, lsuffix='_survived', rsuffix='_total')



df_survive_rate_sibsp['survival_rate']=df_survive_rate_sibsp['Survived_survived']/df_survive_rate_sibsp['Survived_total']

 



plt.plot(df_survive_rate_sibsp['survival_rate'])

plt.show()



df_survive_rate_sibsp
#impact of Parch on survival rate



features_parch=['Parch','Survived']



df_survived_per_parch=titanic_data_df[features_parch].groupby(['Parch']).sum()

df_parch_count=titanic_data_df[features_parch].groupby(['Parch']).count()



df_survive_rate_parch=df_survived_per_parch.join(df_parch_count, lsuffix='_survived', rsuffix='_total')



df_survive_rate_parch['survival_rate']=df_survive_rate_parch['Survived_survived']/df_survive_rate_parch['Survived_total']

 



plt.plot(df_survive_rate_parch['survival_rate'])

plt.show()



df_survive_rate_parch
#impact of Parch on survival rate



features_embarked=['Embarked','Survived']



df_survived_per_embarked=titanic_data_df[features_embarked].groupby(['Embarked']).sum()

df_embarked_count=titanic_data_df[features_embarked].groupby(['Embarked']).count()



df_survive_rate_embarked=df_survived_per_embarked.join(df_embarked_count, lsuffix='_survived', rsuffix='_total')



df_survive_rate_embarked['survival_rate']=df_survive_rate_embarked['Survived_survived']/df_survive_rate_embarked['Survived_total']

 



plt.plot(df_survive_rate_embarked['survival_rate'])

plt.show()



df_survive_rate_embarked
from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import cross_val_score





features_without_embarked=['Age','Sex','Fare','Pclass','SibSp','Parch']

features_whole=['Age','Sex','Fare','Pclass','SibSp','Parch','Embarked']

features_age_sex_fare=['Age','Sex','Fare']



features=[features_whole,features_without_sibparch,features_age_sex_fare]



for m in range(len(features)):



    X=titanic_data_df[features[m]]

    y=titanic_data.Survived

    

    cols_with_missing = [col for col in X.columns

                         if X[col].isnull().any()]

    

    cols_with_missing

    #confirms that only Age has missing values

    

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, 

                                                                    train_size=0.8, test_size=0.2,

                                                                    random_state=0)

    #select categorical_cols for the future Pipeline

    categorical_cols = [cname for cname in X_train.columns if

                        X_train[cname].nunique() < 10 and 

                        X_train[cname].dtype == "object"]

    

    # Select numerical columns for the future Pipeline

    numerical_cols = [cname for cname in X_train.columns if 

                    X_train[cname].dtype in ['int64', 'float64']]

    

    

    # Preprocessing for numerical data

    numerical_transformer = SimpleImputer(strategy='mean')

    

    # Preprocessing for categorical data

    categorical_transformer = Pipeline(steps=[

        ('imputer', SimpleImputer(strategy='most_frequent')),

        ('onehot', OneHotEncoder(handle_unknown='ignore'))

    ])

    

    # Bundle preprocessing for numerical and categorical data

    preprocessor = ColumnTransformer(

        transformers=[

            ('num', numerical_transformer, numerical_cols),

            ('cat', categorical_transformer, categorical_cols)

        ])

    

    # Define model

    def randomfor(i,k): return RandomForestRegressor(n_estimators=i,criterion='mae',max_leaf_nodes=k, random_state=0)

    

    list_scores={}

    for l in range(90,120,10):

        for j in range (12,18,1):

            model = randomfor(l,j)

           # Bundle preprocessing and modeling code in a pipeline

             

            clf = Pipeline(steps=[('preprocessor', preprocessor),

                                  ('model', model)

                                   ])

            

            # Multiply by -1 since sklearn calculates *negative* MAE

            scores = -1 * cross_val_score(clf, X, y,

                               cv=5,

                                    scoring='neg_mean_absolute_error')

            

            #print('nbr_esti:',l,'max_l_nodes:',j,"\n MAE average score:", scores.mean())

            list_scores.update({tuple([l,j]) : scores.mean()} )

         

    best_combination=min(list_scores, key=lambda k: list_scores[k])

    print('features used:',features[m],'\n best combination:',best_combination,'\nMAE:',list_scores.get(best_combination))





     # SANS BOUCLE FOR PREPROCESSING ET PREDS

     # clf.fit(X_train, y_train)

     # Preprocessing of validation data, get predictions

     # preds = clf.predict(X_valid)

     # print('With n_estimators=',l,'max_leaf_nodes=',j,'mae=',mean_absolute_error(y_valid, preds))
from mpl_toolkits.mplot3d import Axes3D

list_scores.keys()



#GOAL: obtain the n_esti and then the max_leaf_nodes used to see whether or not there is a pattern

n_esti_axis=[]

max_leaf_node_axis=[]

mae_values_axis=[]

for k in list_scores.keys():

#Sequentially access each tuple in `tuple_list`



    n_esti_axis.append(k[0])

    max_leaf_node_axis.append(k[1])

    

for i in list_scores.values():

    mae_values_axis.append(i)

    

print('N_esti:\n',n_esti_axis,'\n\n Max Leaf Node:\n',max_leaf_node_axis,'\n\n Model mae:\n',mae_values_axis)

#Axes3D.scatter(xs=list_scores.keys[0], ys=list_scores.keys[1], zs=list_scores.values(), zdir='z', s=20, c=None, depthshade=True)
fig=plt.figure()

ax=fig.add_subplot(111,projection='3d')



X=n_esti_axis #n_estimators

Y=max_leaf_node_axis #max_leaf_node

Z=mae_values_axis #mae



ax.scatter(X,Y,Z,c=['g'],marker='o')



ax.set_xlabel('Nbr estimators')

ax.set_ylabel('max leaf node')

ax.set_zlabel('MAE')
X=titanic_data_df[features_whole]

y=titanic_data.Survived

X_test=titanic_test_data_df[features_whole]

    

cols_with_missing = [col for col in X.columns

                     if X[col].isnull().any()]



cols_with_missing

#confirms that only Age has missing values



#select categorical_cols for the future Pipeline

categorical_cols = [cname for cname in X.columns if

                    X[cname].nunique() < 10 and 

                    X[cname].dtype == "object"]



# Select numerical columns for the future Pipeline

numerical_cols = [cname for cname in X.columns if 

                X[cname].dtype in ['int64', 'float64']]





# Preprocessing for numerical data

numerical_transformer = SimpleImputer(strategy='mean')



# Preprocessing for categorical data

categorical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('onehot', OneHotEncoder(handle_unknown='ignore'))

])



# Bundle preprocessing for numerical and categorical data

preprocessor = ColumnTransformer(

    transformers=[

        ('num', numerical_transformer, numerical_cols),

        ('cat', categorical_transformer, categorical_cols)

    ])



# Define model

def randomfor(i,k): return RandomForestClassifier(n_estimators=i,criterion='mae',max_leaf_nodes=k, random_state=0)





my_model = randomfor(100,15)

      

# Bundle preprocessing and modeling code in a pipeline

         

clf = Pipeline(steps=[('preprocessor', preprocessor),

                              ('model', model)

                               ])



clf.fit(X,y)

#Preprocessing of validation data, get predictions

preds = clf.predict(X_test)

#print('With n_estimators=',l,'max_leaf_nodes=',j,'mae=',mean_absolute_error(y_valid, preds))



output = pd.DataFrame({'PassengerId': X_test.index,

                       'Survived': preds})

output.to_csv('submission_final_2.csv', index=False)