import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



from wordcloud import WordCloud
path = '/kaggle/input/titanic/'



train = pd.read_csv(path + 'train.csv', sep=",")

test = pd.read_csv(path + "test.csv", sep=",")

test_sub = test.copy()
print('Shape:', train.shape)

train.info()
print('Shape:', test.shape)

test.info()
train['Survived'].value_counts()
class Plot_class():



    def __init__(self,feature, my_dataframe, my_table):

        self.feature = feature

        self.my_dataframe = my_dataframe

        

        

    def plot_bar(feature, my_dataframe):

        fig, axes = plt.subplots(nrows=1,ncols=1, figsize=(12,4))

        my_dataframe.groupby([feature,'Survived'])[feature].count().unstack().plot(kind='bar',stacked=True, ax=axes)

        plt.title('Frequency of {} feature vs  survived (target)'.format(feature))



    def plot_bar_table(feature, my_dataframe, my_table):

        fig = plt.figure()

        # definitions for the axes

        left, width = 0.10, 1.5

        bottom, height = 0.1, .8

        bottom_h = left_h = left + width + 0.02



        rect_cones = [left, bottom, width, height]

        rect_box = [left_h, bottom, 0.17, height]

        

        # plot

        ax1 = plt.axes(rect_cones)

        my_dataframe.groupby([feature,'Survived'])[feature].count().unstack().plot(kind='bar',stacked=True, ax=ax1)

        plt.title('Frequency of {} feature vs  survived (target)'.format(feature))

        

        # Table

        ax2 = plt.axes(rect_box)

        my_table = ax2.table(cellText = table_data, loc ='right')

        my_table.set_fontsize(40)

        my_table.scale(4,4)

        ax2.axis('off')

        plt.show()

    

    def distri(feature, my_dataframe):

        fig, axes = plt.subplots(nrows=1,ncols=1, figsize=(12,5))

        ax = sns.distplot(train.loc[(train['Survived'] == 0),feature].dropna(),color='orange',bins=40)

        ax = sns.distplot(train.loc[(train['Survived'] == 1),feature].dropna(),color='blue',bins=40)

        plt.legend(labels=['not survived','survived'])

        plt.title('{} Distribution Survived vs Non Survived'.format(feature))

        plt.ylabel("Frequency of Passenger Survived")

        plt.xlabel(feature)

        plt.show()

   
#values

Total = train.Survived.count()

Female = train[train['Sex'] == 'female'].Survived.count()

Male = train[train['Sex'] == 'male'].Survived.count()

P_Female = round(Female / Total,2)

P_Male = round(Male / Total,2)

P_Female_and_Survived = round(train[((train['Sex']=='female') & (train['Survived']==1))].Survived.count() / Total, 2)

P_Male_and_Survived = round(train[((train['Sex']=='male') & (train['Survived']==1))].Survived.count() / Total, 2)

P_Survived = round(train[train['Survived'] == 1].Survived.count()/Total, 2)

P_Survived_Female = round(P_Female_and_Survived / P_Female,2) #P(survived|female)

P_Survived_Male = round(P_Male_and_Survived / P_Male, 2) #P(survived|male)

P_Female_Survived = round(P_Female_and_Survived / P_Survived,2) #P(Female | Survived)

P_Male_Survived = round(P_Male_and_Survived / P_Survived,2) #P(Male | Survived)



table_data=[

    ["P(Female)", P_Female],

    ["P(Male)", P_Male],

    ["P(Survived | Female)", P_Survived_Female ],

    ["P(Survived | Male)", P_Survived_Male ],

    ["P(Female | Survived)", P_Female_Survived],

    ["P(Male | Survived)", P_Male_Survived ]

]
Plot_class.plot_bar_table('Sex',train,table_data)
#values

P_First = round(train[train['Pclass'] == 1].Survived.count() / Total, 2)

P_Middle = round(train[train['Pclass'] == 2].Survived.count() / Total, 2)

P_Third = round(train[train['Pclass'] == 3].Survived.count() / Total, 2)

P_First_and_Survived = round(train[((train['Pclass']==1) & (train['Survived']==1))].Survived.count() / Total, 2)#P(first and survived)

P_Middle_and_Survived = round(train[((train['Pclass']==2) & (train['Survived']==1))].Survived.count() / Total, 2)#P(middle and survived)

P_Third_and_Survived = round(train[((train['Pclass']==3) & (train['Survived']==1))].Survived.count() / Total, 2)#P(third and survived)

P_First_Survived = round(P_First_and_Survived / P_Survived, 2) #P(first | survived)

P_Middle_Survived = round(P_Middle_and_Survived / P_Survived, 2) #P(middle | survived)

P_Third_Survived = round(P_Third_and_Survived / P_Survived, 2) #P(first | survived)

P_Survived_First = round(P_First_and_Survived / P_First, 2) #P(Survived | First)

P_Survived_Middle = round(P_First_and_Survived / P_Middle, 2) #P(survived | middle)

P_Survived_Third = round(P_First_and_Survived / P_Third, 2) #P(survived | third)



table_data = [

    ["P(First)", P_First],

    ["P(Middle)", P_Middle],

    ["P(Third)", P_Third],

    ["P(First | Survived)", P_First_Survived],

    ["P(Middle | Survived)", P_Middle_Survived],

    ["P(Third | Survived)", P_Third_Survived],

    ["P(Survived | First)", P_Survived_First],

    ["P(Survived | Middle)", P_Survived_Middle],

    ["P(Survived | Third)", P_Survived_Third]

]
Plot_class.plot_bar_table('Pclass', train, table_data)
#values

P_C = round(train[train['Embarked'] == 'C'].Survived.count() / Total, 2)

P_Q = round(train[train['Embarked'] == 'Q'].Survived.count() / Total, 2)

P_S = round(train[train['Embarked'] == 'S'].Survived.count() / Total, 2)

P_C_and_Survived = round(train[((train['Embarked']=='C') & (train['Survived']==1))].Survived.count() / Total, 2)#P(first and survived)

P_Q_and_Survived = round(train[((train['Embarked']=='Q') & (train['Survived']==1))].Survived.count() / Total, 2)#P(middle and survived)

P_S_and_Survived = round(train[((train['Embarked']=='S') & (train['Survived']==1))].Survived.count() / Total, 2)#P(third and survived)

P_C_Survived = round(P_C_and_Survived / P_Survived, 2) #P(first | survived)

P_Q_Survived = round(P_Q_and_Survived / P_Survived, 2) #P(middle | survived)

P_S_Survived = round(P_S_and_Survived / P_Survived, 2) #P(first | survived)

P_Survived_C = round(P_C_and_Survived / P_C, 2) #P(Survived | First)

P_Survived_Q = round(P_Q_and_Survived / P_Q, 2) #P(survived | middle)

P_Survived_S = round(P_S_and_Survived / P_S, 2) #P(survived | third)



table_data = [

    ["P(C)", P_C],

    ["P(Q)", P_Q],

    ["P(S)", P_S],

    ["P(C | Survived)", P_C_Survived],

    ["P(Q | Survived)", P_Q_Survived],

    ["P(S | Survived)", P_S_Survived],

    ["P(Survived | C)", P_Survived_C],

    ["P(Survived | Q)", P_Survived_Q],

    ["P(Survived | S)", P_Survived_S]

]
Plot_class.plot_bar_table('Embarked', train, table_data)
train['Cabin_'] = train['Cabin'].astype(str).str[0]

train['Cabin_'] = train['Cabin_'].replace({'n':'No_value'})

Plot_class.plot_bar('Cabin_', train)
Plot_class.plot_bar('SibSp', train)
Plot_class.plot_bar('Parch',train)
train['Title'] = train['Name'].str.replace('(.*, )|(\..*)',"")

Plot_class.plot_bar('Title', train)

# although test is not plotting we create the featue

test['Title'] = test['Name'].str.replace('(.*, )|(\..*)',"")
Plot_class.distri('Fare',train)
Plot_class.distri('Age',train)
#f, ax = plt.subplots(figsize=(10, 8))

#Firstly, sex feature change numerical 0 for male and 1 for female

train_corr = train.replace({'Sex':{'male': 0, 'female':1}})

train_corr = train_corr.replace({'Embarked':{'C': 0, 'Q': 1 ,'S':2}})

corr=train_corr[['Survived','Sex', 'Pclass','Embarked','Age', 'SibSp', 'Parch', 'Fare']].corr()

#train_corr.corr()

corr.style.background_gradient().set_precision(2)
d = {'color': ['orange', 'b']}

g = sns.FacetGrid(train, col='Embarked')

g.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')

g.add_legend()
d = {'color': ['orange', 'b']}

g = sns.FacetGrid(train, row='Sex', col='Survived', hue_kws=d, hue='Survived')

g.map(plt.hist, 'Age')

g.add_legend()
d = {'color': ['orange', 'b']}

g = sns.FacetGrid(train, row='Sex', col='Survived', hue_kws=d, hue='Survived')

g.map(plt.hist, 'Fare', bins=20)

g.add_legend()
# Filling empty and NaNs values with NaN

train = train.fillna(np.nan)

# Checking for Null values

train.isnull().sum()
# Filling empty and NaNs values with NaN

test = test.fillna(np.nan)

# Checking for Null values

test.isnull().sum()
train['Embarked'].fillna(train.Embarked.mode()[0], inplace = True)
test['Fare'].fillna(test.Fare.mean(), inplace = True)
train = train.drop(['Cabin','Cabin_'], axis=1)

test = test.drop(['Cabin'], axis=1)
test.Title.unique()
def transform_title(dataset):

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\

                        'Don', 'Major', 'Rev', 'Sir', 'Jonkheer','the Countess', 'Dona'], 'Other')



    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
transform_title(train)

transform_title(test)
fig, axes = plt.subplots(nrows=2,ncols=3, figsize=(12,4))

sns.boxplot(data=train, x = 'Title', y = 'Age', ax=axes[0,0])

sns.boxplot(data=train, x = 'SibSp', y = 'Age', ax=axes[0,1])

sns.boxplot(data=train, x = 'Pclass', y = 'Age', ax=axes[0,2])

sns.boxplot(data=test, x = 'Title', y = 'Age', ax=axes[1,0])

sns.boxplot(data=test, x = 'SibSp', y = 'Age', ax=axes[1,1])

sns.boxplot(data=test, x = 'Pclass', y = 'Age', ax=axes[1,2])

fig.suptitle('Boxplot before filling missing values')

plt.show()
def filling_Age(dataset):

    dataset_aux = dataset.dropna()

    dataset_aux = dataset_aux.reset_index(drop=True)

    dataset_aux = dataset_aux.groupby(['Title','Pclass','SibSp'])['Age'].apply(lambda g: g.mean(skipna=True)).to_frame()

    

    aux = []



    for idx,row in dataset.iterrows():

        if row.isnull().sum() == 0:

            aux.append(dataset.loc[idx]['Age'])

        else:

            val_1 = dataset.loc[idx]['Title']

            val_2 = dataset.loc[idx]['Pclass']

            val_3 = dataset.loc[idx]['SibSp']

            if (val_1, val_2, val_3) in list(dataset_aux.index):

                val_sus = dataset_aux.loc[val_1, val_2, val_3][0]

                aux.append(val_sus)

            else:

                aux.append(dataset.Age.mean())

    

    dataset['Age']=aux
filling_Age(train)

filling_Age(test)
fig, axes = plt.subplots(nrows=2,ncols=3, figsize=(12,4))

plt.subplots_adjust(hspace = 0.8)





sns.boxplot(data=train, x = 'Title', y = 'Age', ax=axes[0,0])

sns.boxplot(data=train, x = 'SibSp', y = 'Age', ax=axes[0,1])

sns.boxplot(data=train, x = 'Pclass', y = 'Age', ax=axes[0,2])

sns.boxplot(data=test, x = 'Title', y = 'Age', ax=axes[1,0])

sns.boxplot(data=test, x = 'SibSp', y = 'Age', ax=axes[1,1])

sns.boxplot(data=test, x = 'Pclass', y = 'Age', ax=axes[1,2])

fig.suptitle('Boxplot after filling missing values', fontsize=14)



plt.show()
train = train.drop(['PassengerId', 'Name', 'Ticket'], axis=1)

test = test.drop(['PassengerId', 'Name', 'Ticket'], axis=1)
train['Sex'] = train['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

test['Sex'] = test['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
train['Embarked'] = train['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2 } ).astype(int)

test['Embarked'] = test['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2 } ).astype(int)
titles_numerics={"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Other":6}

train['Title'] = train['Title'].map(titles_numerics).astype(int)

test['Title'] = test['Title'].map(titles_numerics).astype(int)
def Ages_to_category(dataset):

    dataset_aux = dataset.copy()

    bins = [0, 3, 16, 25, 45, 60, dataset_aux.Age.max()]

    labels = ['Baby', 'Child', 'Young', 'Adult', 'Older Adult','Senior']

    dataset_aux['Age'] = pd.cut(dataset_aux['Age'], bins, labels = labels)

    Ages_numerics = {"Baby": 1, "Child": 2, "Young": 3, "Adult": 4, "Older Adult": 5, "Senior":6}

    dataset_aux['Age'] = dataset_aux['Age'].map(Ages_numerics).astype(int)

    return(dataset_aux)



def Fare_to_category(dataset):

    dataset_aux = dataset.copy()

    bins = [0, 50, 100, 150 , 200, 250, dataset_aux.Fare.max()]

    labels = [1, 2, 3, 4, 5, 6]

    dataset_aux['Fare'] = pd.cut(dataset_aux['Fare'], bins, labels = labels)

    dataset_aux['Fare'] = dataset_aux['Fare'].astype(int)

    return(dataset_aux)

  
train['FamilySize'] = train.SibSp + train.Parch + 1

test['FamilySize'] = test.SibSp + test.Parch + 1



train = train.drop(['SibSp', 'Parch'], axis=1)

test = test.drop(['SibSp', 'Parch'], axis=1)
######Prueba como funciona el modelo

from sklearn import preprocessing

from sklearn.model_selection import train_test_split



#making the dummy varaible of catagorical data

from sklearn.preprocessing import LabelEncoder, OneHotEncoder



# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn import metrics
class Models:

    def __init__(self,dataset, Model_name, X_train, X_test, y_train, y_test):

        self.dataset = dataset

        self.Model_name = Model_name

        self.X_train = X_train

        self.X_test = X_test

        self.y_train = y_train

        self.y_test = y_test

       

    def preparing_data(dataset):

        X = dataset.drop("Survived", axis=1)

        y = dataset["Survived"]

        X = preprocessing.StandardScaler().fit(X).transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=5)

        return(X_train, X_test, y_train, y_test)

    

    def one_hot_encoding(dataset):

        #OneHotEncoder

        OH_encoder = OneHotEncoder(handle_unknown = 'ignore', sparse = False)

        object_cols  = ['Pclass', 'Embarked', 'Title', 'FamilySize']

        OH_cols_dataset = pd.DataFrame(OH_encoder.fit_transform(dataset[object_cols]))

        #Remove categorical columns(will replace with one hot encoding)

        dataset = dataset.drop(object_cols, axis = 1)

        dataset = pd.concat([dataset, OH_cols_dataset], axis=1)

        return(dataset)

    

    def fitting_Model(Model_name, X_train, y_train):

        if Model_name == DecisionTreeClassifier:

            decision_tree = DecisionTreeClassifier(max_depth=4)

            return(decision_tree.fit(X_train, y_train))

        elif Model_name == KNeighborsClassifier:

            knn = KNeighborsClassifier(n_neighbors = 4)

            return(knn.fit(X_train, y_train))

        elif Model_name == RandomForestClassifier:

            random_forest = RandomForestClassifier(n_estimators=100, max_depth=4)

            return(random_forest.fit(X_train, y_train))

        elif Model_name == LogisticRegression:

            lg = LogisticRegression(solver='lbfgs')

            return(lg.fit(X_train, y_train))

        else:

            model = Model_name()

            return(model.fit(X_train, y_train))

    

    def predicting_Model(Model_name, X_train, y_train, X_test):

        model = Models.fitting_Model(Model_name, X_train, y_train)

        return(model.predict(X_test))

    

    def score_model(Model_name, X_train, y_train, X_test, y_test):

        model = Models.fitting_Model(Model_name, X_train, y_train)

        y_pred = model.predict(X_test)    

        acc_model = round(model.score(X_train, y_train) * 100, 2)

        acc_test = round(metrics.accuracy_score(y_test, y_pred)*100,2)

        return(acc_model, acc_test)
X_train, X_test, y_train, y_test = Models.preparing_data(train)
acc_train, acc_test = Models.score_model(DecisionTreeClassifier, X_train, y_train, X_test, y_test)

print('- Decision_tree:\n \t accuracy_train: {}\n \t accuracy_test:{}'.format(acc_train, acc_test))

acc_train, acc_test = Models.score_model(RandomForestClassifier, X_train, y_train, X_test, y_test)

print('- Random Forest:\n \t accuracy_train: {}\n \t accuracy_test:{}'.format(acc_train, acc_test))

acc_train, acc_test = Models.score_model(SGDClassifier, X_train, y_train, X_test, y_test)

print('- SGD_classifier:\n \t accuracy_train: {}\n \t accuracy_test:{}'.format(acc_train, acc_test))

acc_train, acc_test = Models.score_model(KNeighborsClassifier, X_train, y_train, X_test, y_test)

print('- KNN:\n \t accuracy_train: {}\n \t accuracy_test:{}'.format(acc_train, acc_test))

acc_train, acc_test = Models.score_model(GaussianNB, X_train, y_train, X_test, y_test)

print('- Gaussian:\n \t accuracy_train: {}\n \t accuracy_test:{}'.format(acc_train, acc_test))
#converting Age and Fare features, from continuous to categorical.

train_aux = Ages_to_category(train)

train_aux = Fare_to_category(train_aux)
X_train_aux, X_test_aux, y_train_aux, y_test_aux = Models.preparing_data(train_aux)
acc_train, acc_test = Models.score_model(DecisionTreeClassifier, X_train_aux, y_train_aux, X_test_aux, y_test_aux)

print('- Decision_tree:\n \t accuracy_train: {}\n \t accuracy_test:{}'.format(acc_train, acc_test))

acc_train, acc_test = Models.score_model(RandomForestClassifier, X_train_aux, y_train_aux, X_test_aux, y_test_aux)

print('- Random Forest:\n \t accuracy_train: {}\n \t accuracy_test:{}'.format(acc_train, acc_test))

acc_train, acc_test = Models.score_model(SGDClassifier, X_train_aux, y_train_aux, X_test_aux, y_test_aux)

print('- SGD_classifier:\n \t accuracy_train: {}\n \t accuracy_test:{}'.format(acc_train, acc_test))

acc_train, acc_test = Models.score_model(KNeighborsClassifier, X_train_aux, y_train_aux, X_test_aux, y_test_aux)

print('- KNN:\n \t accuracy_train: {}\n \t accuracy_test:{}'.format(acc_train, acc_test))

acc_train, acc_test = Models.score_model(GaussianNB, X_train_aux, y_train_aux, X_test_aux, y_test_aux)

print('- Gaussian:\n \t accuracy_train: {}\n \t accuracy_test:{}'.format(acc_train, acc_test))
# In order to use logistic regression, we applys hot encoding to Pclass, Embarked, Title, FamilySize features 

# We keep Age and Fare features as continuous features.

train_hot = Models.one_hot_encoding(train)

test_hot = Models.one_hot_encoding(test)
X_train_hot, X_test_hot, y_train_hot, y_test_hot = Models.preparing_data(train_hot)
acc_train, acc_test = Models.score_model(LogisticRegression, X_train_hot, y_train_hot,X_test_hot, y_test_hot)

print('- Logistic Regression:\n \t accuracy_train: {}\n \t accuracy_test:{}'.format(acc_train, acc_test))
#taking the LogisticRegression we have the follow prediction

test_hot = preprocessing.StandardScaler().fit(test_hot).transform(test_hot)
y_pred = Models.predicting_Model(LogisticRegression, X_train_hot, y_train_hot, test_hot)
gender_submission_LG = pd.DataFrame({

        "PassengerId": test_sub["PassengerId"],

        "Survived": y_pred

    })