import numpy as np

import scipy.stats as stats

import seaborn as sns

import matplotlib.pyplot as plt

import pandas as pd



sns.set(font_scale=1.5)

%config InlineBackend.figure_format = 'retina'

%matplotlib inline
train=pd.read_csv('/kaggle/input/titanic/train.csv')
test=pd.read_csv('/kaggle/input/titanic/test.csv')
train.head()
test.head()
train.shape
test.shape
train.isnull().sum()
test.isnull().sum()
train.describe()
test.describe()
corr=train.corr()
sns.heatmap(corr, annot=True)
train['Embarked'].value_counts()
train['Embarked']=train['Embarked'].fillna('S')
train['Cabin']= train['Cabin'].fillna('Unknown')
train['Cabin']= train['Cabin'].apply(lambda x: x[0])
train['Cabin'].head()
train.groupby('Pclass')['Cabin'].value_counts()
train['Cabin'] = np.where((train.Pclass==1) & (train.Cabin=='U'),'T',

                                            np.where((train.Pclass==2) & (train.Cabin=='U'),'D',

                                                                        np.where((train.Pclass==3) & (train.Cabin=='U'),'E',train.Cabin

                                                                                                    )))
train['Sex']=train['Sex'].apply(lambda x: 1 if x=='female' else 0 )
guess_ages = np.zeros((2,3))

guess_ages


for i in range(0, 2):

    for j in range(0, 3):

        guess_df = train[(train['Sex'] == i) & (train['Pclass'] == j+1)]['Age'].dropna()



            # age_mean = guess_df.mean()

            # age_std = guess_df.std()

            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)



        age_guess = guess_df.median()



            # Convert random age float to nearest .5 age

        guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5

            

for i in range(0, 2):

    for j in range(0, 3):

        train.loc[ (train.Age.isnull()) & (train.Sex == i) & (train.Pclass == j+1),'Age'] = guess_ages[i,j]



train['Age'] = train['Age'].astype(int)



train.isnull().sum()
test['Cabin']= test['Cabin'].fillna('Unknown')

test['Cabin']= test['Cabin'].apply(lambda x: x[0])    
test.groupby('Pclass')['Cabin'].value_counts()
test['Cabin'] = np.where((test.Pclass==1) & (test.Cabin=='U'),'T',

                                            np.where((test.Pclass==2) & (test.Cabin=='U'),'D',

                                                                        np.where((test.Pclass==3) & (test.Cabin=='U'),'E',test.Cabin

                                                                                                    )))
test.groupby('Pclass')['Cabin'].value_counts()
guess_ages = np.zeros((2,3))

guess_ages
test['Sex']=test['Sex'].apply(lambda x: 1 if x=='female' else 0 )
for i in range(0, 2):

    for j in range(0, 3):

        guess_df = test[(test['Sex'] == i) & (test['Pclass'] == j+1)]['Age'].dropna()



            # age_mean = guess_df.mean()

            # age_std = guess_df.std()

            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)



        age_guess = guess_df.median()



            # Convert random age float to nearest .5 age

        guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5

            

for i in range(0, 2):

    for j in range(0, 3):

        test.loc[ (test.Age.isnull()) & (test.Sex == i) & (test.Pclass == j+1),'Age'] = guess_ages[i,j]



test['Age'] = test['Age'].astype(int)
test['Fare']=test['Fare'].fillna('unknown')
test.loc[test['Fare'] == 'unknown']
train.groupby('Pclass')['Fare'].mean()
test['Fare'].replace('unknown', 14, inplace=True)
test.isnull().sum()
def subplot_histograms(dataframe, list_of_columns, list_of_titles, list_of_xlabels):

    nrows = int(np.ceil(len(list_of_columns)/2)) # Makes sure you have enough rows

    fig, ax = plt.subplots(nrows=nrows, ncols=2, figsize=(14,24)) # You'll want to specify your figsize

    ax = ax.ravel() # Ravel turns a matrix into a vector, which is easier to iterate

    for i, column in enumerate(list_of_columns): # Gives us an index value to get into all our lists

        ax[i].hist(dataframe[column], color='skyblue') # feel free to add more settings

        ax[i].set_title(list_of_titles[i])

        ax[i].set_xlabel(list_of_xlabels[i])

       
cols=['Pclass','Sex','Age','SibSp','Parch','Fare','Survived']

tit=['Pclass','Sex','Age','Siblings','Parents and children','Fare','Survived']

xs=['Pclass','Sex','Age','SibSp','Parch','Fare','Survived']

subplot_histograms(train,cols,tit,xs)
surv=pd.DataFrame(train.loc[train['Survived'] == 1])




cols=['Pclass','Sex','Age','SibSp','Parch','Fare']

tit=['Pclass','Sex','Age','Siblings','Parents and children','Fare']

xs=['Pclass','Sex','Age','SibSp','Parch','Fare']

subplot_histograms(surv,cols,tit,xs)
sns.heatmap(train.corr(),annot=True)
fig, ax = plt.subplots()

ax.scatter(x = train['Survived'], y = train['Age'])

plt.ylabel('Age', fontsize=13)

plt.xlabel('Survived', fontsize=13)

plt.title('')

plt.show()
#Deleting outliers

train = train.drop(train[(train['Survived']== 1) & (train['Age']>79)].index)



#Check the graphic again

fig, ax = plt.subplots()

ax.scatter(x = train['Survived'], y = train['Age'])

plt.ylabel('Age', fontsize=13)

plt.xlabel('Survived', fontsize=13)

plt.title('')

plt.show()
def percentage(part):

    whole= train['Sex'].value_counts().sum()

    percentage= (part/whole)

    return percentage



percentage= train['Sex'].value_counts().apply(lambda x : percentage(x))

print (percentage)







labels = 'Male','Female' 

sizes = [65, 35]

c=['#b39ab0','#e6e6fa']

fig1, ax1 = plt.subplots()

ax1.pie(sizes, labels=labels, autopct='%1.1f%%',colors=c,

        shadow=True, startangle=90)

ax1.axis('equal')  

plt.title('Percentage of Male Vs. Female passengers')

plt.show()

def percentage(part):

    whole= surv['Sex'].value_counts().sum()

    percentage= (part/whole)

    return percentage



percentage= surv['Sex'].value_counts().apply(lambda x : percentage(x))

print (percentage)







labels = 'Female', 'Male'

sizes = [68, 32]

c=['#b39ab0','#e6e6fa']

fig1, ax1 = plt.subplots()

ax1.pie(sizes, labels=labels, autopct='%1.1f%%',colors=c,

        shadow=True, startangle=90)

ax1.axis('equal')  

plt.title('Percentage of Male Vs. Female survivors')

plt.show()

cat=train['Age']

cat=cat.apply(lambda x:'<10' if x<11 else '11-18' if x<18  else '19-35' if x<36  else '36-60' if x<61  else '61-80')



c= {'male': '#8c9fff' , 'female': '#ffb68c'}

sns.countplot(x='Pclass', data = train , hue=cat, palette= 'Set2' )

plt.title('Range of ages in each Pclass')

plt.legend(bbox_to_anchor=(1,1), loc=2)
cat=surv['Age']

cat=cat.apply(lambda x:'<10' if x<11 else '11-18' if x<18  else '19-35' if x<36  else '36-60' if x<61  else '61-80')



c= {'male': '#8c9fff' , 'female': '#ffb68c'}

sns.countplot(x='Pclass', data = surv , hue=cat, palette= 'Set2' )

plt.title('Range of ages of survivors in each Pclass')

plt.legend(bbox_to_anchor=(1,1), loc=2)
c= {'male': '#8c9fff' , 'female': '#ffb68c'}

sns.countplot(x='Survived', data = train , hue=cat, palette= 'Set2' )

plt.title('Range of ages for survivors')

plt.legend(bbox_to_anchor=(1,1), loc=2)
df_plot=train.groupby(['Pclass', 'Survived']).size().reset_index().pivot(columns='Pclass', index='Survived', values=0)

df_plot.plot(kind='bar', stacked=True,colormap='Set2')

plt.title('Number of survivors for each Pclass')

plt.legend(bbox_to_anchor=(1,1), loc=2)


sns.pairplot(train)
features_drop = ['PassengerId','Name', 'Ticket', 'Survived','Embarked','Cabin']

selected_features=[c for c in train if c not in features_drop]

selected_features
X_train = train[selected_features]

y_train = train['Survived']

X_test= test[selected_features]
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)

X_test = pd.DataFrame(scaler.fit_transform(X_test), columns=X_test.columns)
# from sklearn.preprocessing import MinMaxScaler

# scaler = MinMaxScaler(feature_range=(0, 1))



# x_train_scaled = scaler.fit_transform(X_train)

# X_train = pd.DataFrame(x_train_scaled)



# x_test_scaled = scaler.fit_transform(X_test)

# X_test = pd.DataFrame(x_test_scaled)
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score
baseline=y_train.value_counts(normalize=True)

baseline[0]
knn = KNeighborsClassifier(n_neighbors=5) 

knn.fit(X_train, y_train)

m_score= knn.score(X_train, y_train)

print('Model score: ', m_score)
predictions = pd.DataFrame(knn.predict(X_test))

predictions['PassengerId']=[i for i in range(892, 1310)]

predictions.rename(columns={0:'Survived'},inplace=True)

predictions.set_index('PassengerId',inplace=True)
predictions.to_csv('predictions_knn.csv')
from sklearn.model_selection import GridSearchCV

knn_params = {

    'n_neighbors': range(1,100),

    'weights':['uniform','distance'],

    'metric':['euclidean','manhattan']}

print('Initialized parameters for Grid Search')

print(knn_params)
knn_gridsearch = GridSearchCV(KNeighborsClassifier(), 

                              knn_params, 

                              n_jobs=1, cv=5) # try verbose!





knn_gridsearch.fit(X_train, y_train)

best_knn = knn_gridsearch.best_estimator_

best_knn.score(X_train, y_train)

predictions = pd.DataFrame(best_knn.predict(X_test))
predictions['PassengerId']=[i for i in range(892, 1310)]

predictions.rename(columns={0:'Survived'},inplace=True)

predictions.set_index('PassengerId',inplace=True)

predictions.to_csv('predictions_gs_knn_afteroutliers.csv')
from sklearn.tree import DecisionTreeClassifier
dtc_params = {

    'max_depth': range(1,20),

    'max_features': [None, 'log2', 'sqrt'],

    'min_samples_split': range(5,30),

    'max_leaf_nodes': [None],

    'min_samples_leaf': range(1,10)

}



from sklearn.model_selection import GridSearchCV

# set the gridsearch

dtc_gs = GridSearchCV(DecisionTreeClassifier(), dtc_params,  n_jobs=-1, cv=5)
dtc_gs.fit(X_train, y_train)
predictions = dtc_gs.best_estimator_.predict(X_test)

# predictions.to_csv('predictions_dt_gs.csv')
predictions = pd.DataFrame(predictions)

predictions['PassengerId']=[i for i in range(892, 1310)]

predictions.rename(columns={0:'Survived'},inplace=True)

predictions.set_index('PassengerId',inplace=True)

predictions.to_csv('predictions_dt_gs.csv')
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier



rf_params = {

      'n_estimators': range(1,100),

#     'max_features':[2, 3, 5, 7, 8],

      'max_depth': range(1,20),

     'criterion':['gini', 'entropy'],

}
rf_g = RandomForestClassifier() 
gs = GridSearchCV(rf_g, param_grid=rf_params, cv=5, verbose = 1)#, refit=False) 
gs=gs.fit(X_train, y_train)
predictions = gs.best_estimator_.predict(X_test)

predictions = pd.DataFrame(predictions)

predictions['PassengerId']=[i for i in range(892, 1310)]

predictions.rename(columns={0:'Survived'},inplace=True)

predictions.set_index('PassengerId',inplace=True)

predictions.to_csv('predictions_RF_gs.csv')
rf_params = {

      'n_estimators': range(1,100),

#     'max_features':[2, 3, 5, 7, 8],

      'max_depth': range(1,20),

     'criterion':['gini', 'entropy'],

}
et_g = ExtraTreesClassifier()
gs_et = GridSearchCV(rf_g, param_grid=rf_params, cv=5, verbose = 1)#, refit=False) 
gs_et =gs_et.fit(X_train, y_train)
predictions = gs_et.best_estimator_.predict(X_test)
predictions = pd.DataFrame(predictions)

predictions['PassengerId']=[i for i in range(892, 1310)]

predictions.rename(columns={0:'Survived'},inplace=True)

predictions.set_index('PassengerId',inplace=True)

predictions.to_csv('predictions_ET_gs.csv')
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

model = LogisticRegression()

params = {'C':np.logspace(-5,5,15),

          'penalty':['l1'],

          'fit_intercept':[True,False]}

gs = GridSearchCV(estimator=model,

                  param_grid=params,

                  cv=5,

                  scoring='accuracy',

                  return_train_score=True)

gs.fit(X_train,y_train)

print(gs.best_score_)

print(gs.score(X_train,y_train))

predictions= gs.predict(X_test)

predictions = pd.DataFrame(predictions)

predictions['PassengerId']=[i for i in range(892, 1310)]

predictions.rename(columns={0:'Survived'},inplace=True)

predictions.set_index('PassengerId',inplace=True)

predictions.to_csv('predictions_log_l1.csv')
model = LogisticRegression()

params = {'C':np.logspace(-5,5,15),

          'penalty':['l2'], #Ridge

          'fit_intercept':[True,False]}

gs = GridSearchCV(estimator=model,

                  param_grid=params,

                  cv=5,

                  scoring='accuracy',

                  return_train_score=True)

gs.fit(X_train,y_train)

print(gs.best_score_)

print(gs.score(X_train,y_train))
predictions= gs.predict(X_test)

predictions = pd.DataFrame(predictions)

predictions['PassengerId']=[i for i in range(892, 1310)]

predictions.rename(columns={0:'Survived'},inplace=True)

predictions.set_index('PassengerId',inplace=True)

predictions.to_csv('predictions_log_l2.csv')
X_train_s = train[['Sex','Pclass']]

y_train_s = train['Survived']

X_test_s= test[['Sex','Pclass']]
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_s = pd.DataFrame(scaler.fit_transform(X_train_s), columns=X_train_s.columns)

X_test_s = pd.DataFrame(scaler.fit_transform(X_test_s), columns=X_test_s.columns)
rf_params = {

      'n_estimators': range(1,100),

#     'max_features':[2, 3, 5, 7, 8],

      'max_depth': range(1,20),

     'criterion':['gini', 'entropy'],

}

rf_g = RandomForestClassifier()

gs = GridSearchCV(rf_g, param_grid=rf_params, cv=5, verbose = 1)#, refit=False) 

gs=gs.fit(X_train_s, y_train_s)

predictions = gs.best_estimator_.predict(X_test_s)
predictions = pd.DataFrame(predictions)

predictions['PassengerId']=[i for i in range(892, 1310)]

predictions.rename(columns={0:'Survived'},inplace=True)

predictions.set_index('PassengerId',inplace=True)

predictions.to_csv('predictions_RF_gs_s.csv')
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV



X_train_s = train[['Sex','Pclass','Age']]

y_train_s = train['Survived']

X_test_s= test[['Sex','Pclass','Age']]



scaler = StandardScaler()

X_train_s = pd.DataFrame(scaler.fit_transform(X_train_s), columns=X_train_s.columns)

X_test_s = pd.DataFrame(scaler.fit_transform(X_test_s), columns=X_test_s.columns)



rf_params = {

      'n_estimators': range(1,50),

#     'max_features':[2, 3, 5, 7, 8],

      'max_depth': range(1,20),

     'criterion':['gini', 'entropy'],

}

rf_g = RandomForestClassifier()

gs = GridSearchCV(rf_g, param_grid=rf_params, cv=5, verbose = 1)#, refit=False) 

gs=gs.fit(X_train_s, y_train_s)

predictions = gs.best_estimator_.predict(X_test_s)



predictions = pd.DataFrame(predictions)

predictions['PassengerId']=[i for i in range(892, 1310)]

predictions.rename(columns={0:'Survived'},inplace=True)

predictions.set_index('PassengerId',inplace=True)

predictions.to_csv('predictions_RF_gs_s2.csv')