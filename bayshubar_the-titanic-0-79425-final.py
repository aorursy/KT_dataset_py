import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
TRAIN_PATH = '../input/train.csv'
TEST_PATH = '../input/test.csv'
research_train = pd.read_csv(TRAIN_PATH)
research_train.head(2)
research_train.info()
pclass_group = research_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean();
plt.bar(pclass_group['Pclass'].values, pclass_group['Survived'].values);
plt.xticks(pclass_group['Pclass'].values, pclass_group['Pclass'].values);
plt.title('Pclass and Survived correlation')
plt.ylabel('Survived');
plt.xlabel('Pclass');
#Here we get title by searching them with regex wich will looking for dot before a title
def get_dot_title(full_name):
    result=re.search(' ([A-Za-z]+)\.', full_name)
    if result:
        return result.group(1)
    return ''
#Here we set new feature Title
research_train['Title'] = research_train['Name'].apply(get_dot_title)
research_train[['Title', 'Survived']].groupby(['Title'],  as_index=False).agg(['mean', 'count'])
#Here we combine other Titles
research_train['Title'] = research_train['Title'].replace(['Capt', 'Col','Don', 
                            'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Others')

research_train['Title'] = research_train['Title'].replace('Lady', 'Miss')
research_train['Title'] = research_train['Title'].replace('Countess', 'Miss')
research_train['Title'] = research_train['Title'].replace('Mlle', 'Miss')
research_train['Title'] = research_train['Title'].replace('Ms', 'Miss')
research_train['Title'] = research_train['Title'].replace('Mme', 'Mrs')

title_group = research_train[['Title', 'Survived']].groupby(['Title'],  as_index=False).mean()

plt.bar(title_group['Title'].values, title_group['Survived'].values);
plt.title('Title and Survived correlation')
plt.ylabel('Survived');
plt.xlabel('Title');

sex_group = research_train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean()
plt.bar(sex_group['Sex'].values, sex_group['Survived'].values);
plt.title('Sex and Survived correlation')
plt.ylabel('Survived');
plt.xlabel('Sex');
#Handling missing data
age_avg = research_train['Age'].mean()
age_std = research_train['Age'].std()
age_null_count = research_train['Age'].isnull().sum()
age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
research_train.loc[np.isnan(research_train['Age']), 'Age'] = age_null_random_list

#Categoryzing data
research_train['Age'] = research_train['Age'].astype(int)
research_train['AgeCategory'] = pd.cut(research_train['Age'], 5)
age_group = research_train[['AgeCategory', 'Survived']].groupby(['AgeCategory'], as_index=False).mean()
age_group
research_train['FamilySize'] = research_train['SibSp'] + research_train['Parch'] + 1
family_group = research_train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean()
plt.bar(family_group['FamilySize'].values, family_group['Survived'].values);
plt.title('Sex and Survived correlation')
plt.ylabel('Survived');
plt.xlabel('Sex');
research_train['FareCategory'] = pd.qcut(research_train['Fare'], 4)
research_train[['FareCategory', 'Survived']].groupby(['FareCategory'], as_index=False).mean()
research_train['Embarked'] = research_train['Embarked'].fillna('S')
research_train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean()

FEATURE_SIZE = 9
TEST_SIZE = 0.2
def preprocessing(df, is_survived = False):
    #Does a passanger has a Cabin. As we know it has only 22% percent of the data 
    #that is why I think that It is more resonable to ignore this feature

    
    #Name length of Passanger
    df['NameLength'] = df['Name'].apply(len)
    
    #The new feature generated by adding siblings number to parents number
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    
    #This binary feature indicate does a passanger alone or not
    df['IsAlone'] = 0
    df.loc[df['FamilySize']==1,'IsAlone'] = 1
    
    #Replacing Null with S in Embarked because S is common
    df['Embarked'] = df['Embarked'].fillna('S')
    
    #Replacing null values with median
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    
    #This is used to divide Fare to 4 category
    df['CategoryFare'] = pd.qcut(df['Fare'], 4)
    
    #Here we filling null values in age with random values in SET range
    age_avg = df['Age'].mean()
    age_std = df['Age'].std()
    age_null_count = df['Age'].isnull().sum()
    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
    df.loc[np.isnan(df['Age']), 'Age'] = age_null_random_list
    df['Age'] = df['Age'].astype(int)
    
    #This is used to divide Age to 5 category
    df['CategoryAge'] = pd.cut(df['Age'], 5)
    
    #Here we get title by searching them with regex wich will looking for dot before a title
    def get_dot_title(full_name):
        result = re.search(' ([A-Za-z]+)\.', full_name)
        if result:
            return result.group(1)
        return ''
    
    #Here we set new feature Title
    df['Title'] = df['Name'].apply(get_dot_title)
    #Here we combine other Titles
    df['Title'] = df['Title'].replace(['Capt', 'Col','Don', 
                            'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Others')

    df['Title'] = df['Title'].replace('Lady', 'Miss')
    df['Title'] = df['Title'].replace('Countess', 'Miss')
    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    
    
    
    
    
    #Mapping a titles to categories
    df['Title'] = df['Title'].map({"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Common": 5})
    df['Title'] = df['Title'].fillna(0).astype(int)
       
    
    #MAPPING STAGE
    
    #mapping Sex
    df['Sex'] = df['Sex'].map({'female':1, 'male':0}).astype(int)
    
    #mapping Embarked
    df['Embarked'] = df['Embarked'].map({'S':0, 'C':1, 'Q':2}).astype(int)
    
    #mapping Fare
    df.loc[df['Fare']<=7.91, 'Fare'] = 0
    df.loc[(df['Fare']>7.91) & (df['Fare']<=14.454), 'Fare'] = 1
    df.loc[(df['Fare']>14.454) & (df['Fare']<= 31), 'Fare'] = 2
    df.loc[df['Fare']>31, 'Fare'] = 3
    df['Fare'] = df['Fare'].astype(int)
    
    #mapping Age
    df.loc[df['Age']<=16, 'Age'] = 0
    df.loc[(df['Age']>16) & (df['Age']<=32), 'Age'] = 1
    df.loc[(df['Age']>32) & (df['Age']<=48), 'Age'] = 2
    df.loc[(df['Age']>48) & (df['Age']<=64), 'Age'] = 3
    df.loc[df['Age']>64, 'Age'] = 4
    df['Age'].astype(int)
    
    #Feature selecction
    features_to_delete = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp', 'CategoryAge', 'CategoryFare', 'Parch']
    #The preprocessing function will be used multiple times and It handle apsence of Survived column
    if not is_survived:
        features_to_delete.append('Survived')
        
    df = df.drop(features_to_delete, axis=1)
    df.head()
    
    return df


# #FUR FUNCTION DEVELOPMENT
# df_train = pd.read_csv('train.csv')
# df_train_preprocessed = preprocessing(df_train)
    
def split(df,y_train, size = 0):
    from sklearn.model_selection import train_test_split
    x_train = df.iloc[:, 0:FEATURE_SIZE].values
    return train_test_split(x_train, y_train, test_size = size, random_state = 3 )
def rf_classifier(train, test):
    from sklearn.ensemble import RandomForestClassifier
    classifier = RandomForestClassifier(
        random_state=0, 
        max_depth=20, 
        criterion='gini', 
        n_estimators=500, 
        min_samples_split=10,
    )
    return classifier.fit(train, test)
def svc_classifier(train, test):
    from sklearn.svm import SVC
    classifier = SVC(
        gamma='scale',
        kernel ='linear',
        probability = True,
    )
    return classifier.fit(train, test)
    
def generate_result(y_pred, passengerId):
    predicted = pd.DataFrame(y_pred)
    predicted.columns = ['Survived']
    result = pd.concat([passengerId, predicted], axis=1)
    result.to_csv('result.csv', index=False)
    
    
df_train = pd.read_csv(TRAIN_PATH)
#Preprocessing the input data
df_train_preprocessed = preprocessing(df_train)
#Dividing to train and test data
x_train, x_test, y_train, y_test = split(df_train_preprocessed,df_train['Survived'].values, TEST_SIZE)
#Creatin classifier for future use
classifier = rf_classifier(x_train, y_train)
df_train_preprocessed.head(2)
importance = classifier.feature_importances_
figure  = plt.figure(figsize=(20, 10))
plt.bar(df_train_preprocessed.columns.values, importance);
plt.title('Sex and Survived correlation')
plt.ylabel('Survived');
plt.xlabel('Sex');
from sklearn.metrics import confusion_matrix
import itertools
y_train_pred = classifier.predict(x_test)
cm = confusion_matrix(y_test, y_train_pred)

labels = ['Predicted NO', 'Predicted YES','Actual NO','Actual YES']
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
plt.title('TRAIN TEST EXAMPLE \n')

ax.set_xticklabels([''] + labels[0:2])
ax.set_yticklabels([''] + labels[2:4])

fmt = '.0f'

for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j], fmt),
        horizontalalignment="center",
        color="red", fontsize = 22)

plt.show()

from sklearn.metrics import accuracy_score
test_size = sum(sum(cm))
positive_result = cm[0][0] + cm[1][1]
accuracy = positive_result/test_size
accuracy
df_test = pd.read_csv(TEST_PATH)
df_test_preprocessed = preprocessing(df_test, is_survived = True)
PassengerId = df_test['PassengerId']

y_pred = classifier.predict(df_test_preprocessed.values)
generate_result(y_pred, PassengerId)
df_test_preprocessed.head(2)
