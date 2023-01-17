import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import plot_roc_curve
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
pd.options.display.float_format = '{:.2f}'.format
train = pd.read_csv('../input/titanic/train.csv')
train.head()
train.shape
train.columns
train.describe()
train.info()
train.isnull().sum()
sns.heatmap(train.isnull(),cmap='magma',cbar = False)
sur = train['Survived'].value_counts()
survival_rate = [sur[0]/len(train)*100,sur[1]/len(train)*100]
fig,ax = plt.subplots(nrows=1,ncols = 2,figsize = (15,5))
plt.subplot(1,2,1)
plt.pie(survival_rate,labels = ['Did Not Survive','Survived'],autopct='%1.1f%%',startangle = 90,)
plt.title('SURVIVAL PERCENTAGE')
plt.subplot(1,2,2)
sns.countplot('Survived',data = train,hue = 'Sex')
plt.title('SURVIVAL CHART')
survived_male = len(train[(train['Sex'] == 'male') & (train['Survived'] == 1)])/len(train[train['Sex'] == 'male'])*100
survived_female = len(train[(train['Sex'] == 'female') & (train['Survived'] == 1)])/len(train[train['Sex'] == 'female'])*100
male = [survived_male,100-survived_male]
female = [survived_female,100-survived_female]
fig = plt.subplots(nrows = 1,ncols = 2,figsize = (15,5))
plt.subplot(1,2,1)
plt.pie(male,labels = ['Survived','Did Not Survive'],autopct='%1.1f%%',startangle = 90)
plt.title('MALE SURVIVAL PERCENTAGE')
plt.subplot(1,2,2)
plt.pie(female,labels = ['Survived','Did Not Survive'],autopct='%1.1f%%',startangle = 90)
plt.title('FEMALE SURVIVAL PERCENTAGE')
plt.show()
plt.subplots(figsize = (15,5))
Pclass = [len(train[(train['Pclass'] == 1)])/len(train)*100,len(train[(train['Pclass'] == 2)])/len(train)*100,len(train[(train['Pclass'] == 3)])/len(train)*100]
plt.pie(Pclass,labels = ['Pclass 1','Pclass 2','Pclass 3'],autopct='%1.1f%%',startangle = 90)
plt.title('PCLASS DISTRIBUTION')
plt.show()
survived_class_1 = len(train[(train['Pclass'] == 1) & train['Survived'] == 1])/len(train[(train['Pclass'] == 1)])*100
survived_class_2 = len(train[(train['Pclass'] == 2) & train['Survived'] == 1])/len(train[(train['Pclass'] == 2)])*100
survived_class_3 = len(train[(train['Pclass'] == 3) & train['Survived'] == 1])/len(train[(train['Pclass'] == 3)])*100
class_1 = [survived_class_1,100-survived_class_1]
class_2 = [survived_class_2,100-survived_class_2]
class_3 = [survived_class_3,100-survived_class_3]
fig,ax = plt.subplots(nrows = 1, ncols = 2,figsize = (15,5))
plt.subplot(1,2,1)
sns.countplot('Pclass',data = train,hue = 'Survived')
plt.title('PASSENGER CLASS SURVIVAL',)
plt.subplot(1,2,2)
plt.pie(class_1,labels = ['Survived','Did Not Survive'],autopct='%1.1f%%',startangle = 90)
plt.title('PASSENGER CLASS 1 SURVIVAL PERCENTAGE')
plt.show()

fig,ax = plt.subplots(nrows = 1, ncols = 2,figsize = (15,5))
plt.subplot(1,2,1)
plt.pie(class_2,labels = ['Survived','Did Not Survive'],autopct='%1.1f%%',startangle = 90)
plt.title('PASSENGER CLASS 2 SURVIVAL PERCENTAGE')
plt.subplot(1,2,2)
plt.pie(class_3,labels = ['Survived','Did Not Survive'],autopct='%1.1f%%',startangle = 90)
plt.title('PASSENGER CLASS 3 SURVIVAL PERCENTAGE')
plt.show()
survived_S = len(train[(train['Embarked'] == 'S') & (train['Survived'] == 1)])/len(train[train['Embarked'] == 'S'])*100
survived_C = len(train[(train['Embarked'] == 'C') & (train['Survived'] == 1)])/len(train[train['Embarked'] == 'C'])*100
survived_Q = len(train[(train['Embarked'] == 'Q') & (train['Survived'] == 1)])/len(train[train['Embarked'] == 'Q'])*100
S = [survived_S,100-survived_S]
C = [survived_C,100-survived_C]
Q = [survived_Q,100-survived_Q]
fig,ax = plt.subplots(nrows = 1, ncols = 2,figsize = (15,5))
plt.subplot(1,2,1)
sns.countplot('Embarked',data = train,hue = 'Survived')
plt.title('EMBARKED CLASS SURVIVAL')
plt.subplot(1,2,2)
plt.pie(S,labels = ['Survived','Did Not Survive'],autopct='%1.1f%%',startangle = 90)
plt.title('EMBARKED S SURVIVAL PERCENTAGE')
plt.show()

fig,ax = plt.subplots(nrows = 1, ncols = 2,figsize = (15,5))
plt.subplot(1,2,1)
plt.pie(C,labels = ['Survived','Did Not Survive'],autopct='%1.1f%%',startangle = 90)
plt.title('EMBARKED C SURVIVAL PERCENTAGE')
plt.subplot(1,2,2)
plt.pie(Q,labels = ['Survived','Did Not Survive'],autopct='%1.1f%%',startangle = 90)
plt.title('EMBARKED Q SURVIVAL PERCENTAGE')
plt.show()
fig,ax = plt.subplots(figsize = (15,5))
sns.distplot(train['Fare'],label = 'Fare')
sns.distplot(train[train['Survived'] == 1]['Fare'],label = 'Survived')
sns.distplot(train[train['Survived'] == 0]['Fare'],label = 'Did Not Survive')
plt.title('DISTRIBUTION OF FARE')
plt.legend()
plt.show()
fig,ax = plt.subplots(nrows = 1,ncols = 2,figsize = (15,5))
plt.subplot(1,2,1)
sns.countplot('SibSp',data = train,hue = 'Survived')
plt.subplot(1,2,2)
sns.countplot('Parch',data = train,hue = 'Survived')
plt.show()
train['Age'].mode()
train['Age'].median()
train['Age'].mean()
train['Age_Mode'] = train['Age'].fillna(value = 24)
train['Age_Med'] = train['Age'].fillna(train['Age'].median())
train['Age_Mean'] = train['Age'].fillna(train['Age'].mean())
fig,ax = plt.subplots(figsize = (15,5))
sns.distplot(train['Age'],label = 'Age')
sns.distplot(train['Age_Mode'],label = 'Age_Mode')
sns.distplot(train['Age_Med'],label = 'Age_Median',)
sns.distplot(train['Age_Mean'],label = 'Age_Mean')
plt.legend()
plt.show()
train['Age'] = train['Age'].fillna(train['Age'].median())
train = train.drop(columns= ['Age_Mode','Age_Med','Age_Mean'])
train.head()
sns.heatmap(train.isnull(),cmap='magma',cbar = False)
def get_age_group(dataframe,column_name):
    
    dataframe[column_name] = dataframe[column_name].apply(np.ceil)
    age_group = {0:list(range(0,21)),1:list(range(21,41)),2:list(range(41,61)),3:list(range(61,81))}
    col = list(dataframe.columns)
    index = col.index(column_name)
    age = []
    
    for j in range(len(dataframe)):
        for k in age_group.keys():
            for i in range(len(age_group[k])):
                if (age_group[k][i] == dataframe.iloc[j,index]):
                    age.append(k)
    dataframe['Age_Group'] = age
get_age_group(train,'Age')
plt.subplots(figsize = (15,5))
sns.violinplot('Age_Group','Survived',data = train)
plt.show()
plt.subplots(figsize = (15,5))
sns.scatterplot('Age','Fare',data = train,hue = 'Survived')
def get_initials(dataframe,column_name):
    sub = []
    initials = ['Mrs.','Ms.','Mr.','Miss.','Master.','Lady.','Don.','Rev.','Dr.','Mme.','Major.','Sir.','Mlle.','Col.','Capt.','Countess.','Jonkheer.','Dona.']
    name = dataframe[column_name]
    for i in range(len(name)):
        split_names = name[i].split()
        for j in range(len(split_names)):
            if (split_names[j] in initials):
                sub.append(split_names[j])
    dataframe[column_name] = sub
get_initials(train,'Name')
train['Name'].value_counts()
le = LabelEncoder()
train['Name'] = le.fit_transform(train['Name'])
encoded_values = train['Name'].unique()
decoded_values = le.inverse_transform(encoded_values)
initials = {}
for i in range(len(encoded_values)):
    initials.setdefault(decoded_values[i],encoded_values[i])
plt.subplots(figsize = (15,5))
for keys,values in initials.items():
    sns.distplot(train[train['Name'] == values]['Age'],label = keys,kde = False)
plt.title('DISTRIBUTION OF INITIALS W.R.T AGE')
plt.legend()
plt.show()
plt.subplots(figsize = (15,5))
for keys,values in initials.items():
    sns.distplot(train[train['Name'] == values]['Age_Group'],label = keys,kde = False)
plt.title('DISTRIBUTION OF INITIALS W.R.T AGE_GROUP')
plt.legend()
plt.show()
train['Name'] = le.inverse_transform(train['Name'])
plt.subplots(figsize = (15,5))
sns.countplot('Name',data = train,hue = 'Survived')
plt.xlabel('Initials')
plt.title('SURVIVAL vs INITIALS')
plt.legend()
plt.show()
train['Name'] = le.fit_transform(train['Name'])
train['Sex'] = le.fit_transform(train['Sex'])
embarked = {0:'S',1:'C',2:'Q'}
train['Embarked'] = train['Embarked'].fillna('Q')
train['Embarked'] = le.fit_transform(train['Embarked'])
plt.subplots(figsize = (15,5))
sns.heatmap(train.corr(),cmap = 'RdBu',cbar = True,annot = True)
train = train.drop(columns = ['PassengerId','Ticket','Cabin','Age_Group'])
train.head()
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
features = train.iloc[:,1:]
target = train.iloc[:,0]
best_features = SelectKBest(score_func = chi2,k = 8)
fit = best_features.fit(features,target)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(features.columns)

featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Column','Score'] 
print(featureScores.nlargest(8,'Score'))
train.head()
train = train.drop(columns = ['Embarked','SibSp','Parch'])
train.head()
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
features = train[['Pclass','Sex','Name','Fare']].values
features = sc.fit_transform(features)
target = train['Survived'].values
x_train, x_test, y_train, y_test = train_test_split(features, target, test_size = 0.25, random_state = 2)
def model(classifier):
    
    classifier.fit(x_train,y_train)
    prediction = classifier.predict(x_test)
    print("ACCURACY : ",'{0:.2%}'.format(accuracy_score(y_test,prediction))) 
    print("CROSS VALIDATION SCORE : ",'{0:.2%}'.format(cross_val_score(classifier,x_train,y_train,cv = 10,scoring = 'accuracy').mean()))
    print("ROC_AUC SCORE : ",'{0:.2%}'.format(roc_auc_score(y_test,prediction)))
    plot_roc_curve(classifier, x_test,y_test)
    plt.title('ROC_AUC_PLOT')
    plt.show()
def model_evaluation(classifier):
    
    # CONFUSION MATRIX
    cm = confusion_matrix(y_test,classifier.predict(x_test))
    names = ['True Neg','False Pos','False Neg','True Pos']
    counts = [value for value in cm.flatten()]
    percentages = ['{0:.2%}'.format(value) for value in cm.flatten()/np.sum(cm)]
    labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(names,counts,percentages)]
    labels = np.asarray(labels).reshape(2,2)
    sns.heatmap(cm,annot = labels,cmap = 'Blues',fmt ='')
    
    tn,fp,fn,tp = cm.flatten()
    
    # PRECISION
    print('PRECISION : ','{0:.2%}'.format(tp/(tp + fp)))
    
    # RECALL
    print('RECALL : ','{0:.2%}'.format(tp/(tp + fn)))
def grid_search_cv(classifier,hyperparameters):
    
    GSCV = GridSearchCV(classifier,hyperparameters,cv = 10)
    model = GSCV.fit(x_test,y_test)
    print(model)
    print('HIGHEST ACCURACY : ','{0:.2%}'.format(model.best_score_))
    print('BEST PARAMETERS : ',model.best_params_)
from sklearn.linear_model import LogisticRegression
classifier_lr = LogisticRegression(random_state = 0,C=10,penalty= 'l2') 
model(classifier_lr)
model_evaluation(classifier_lr)
from sklearn.svm import SVC
classifier_svc = SVC(kernel = 'linear',C = 0.1)
hyperparameters = {'C' : [0.01,0.1,1,10,100]}
grid_search_cv(classifier_svc,hyperparameters)
model(classifier_svc)
model_evaluation(classifier_svc)
from sklearn.tree import DecisionTreeClassifier
classifier_dt = DecisionTreeClassifier(criterion = 'entropy')
model(classifier_dt)
model_evaluation(classifier_dt)
from sklearn.ensemble import RandomForestClassifier
classifier_rf = RandomForestClassifier(max_depth = 2,random_state = 0)
model(classifier_rf)
model_evaluation(classifier_rf)
from sklearn.neighbors import KNeighborsClassifier
classifier_knn = KNeighborsClassifier(leaf_size = 7, n_neighbors = 3,p = 1)
leaf_size = list(range(1,50))
n_neighbors = list(range(1,30))
p=[1,2]
hyperparameters = dict(leaf_size=leaf_size, n_neighbors=n_neighbors, p=p)
grid_search_cv(classifier_knn,hyperparameters)
model(classifier_knn)
model_evaluation(classifier_knn)
models = {'MODELS':['LOGISTIC REGRESSION','SUPPORT VECTOR CLASSIFIER','DECISION TREE CLASSIFIER','RANDOM FOREST CLASSIFIER','K-NEAREST NEIGHBORS'],
         'CROSS VAL ACCURACY (%)':[78.73,79.19,81.59,78.29,81.10]}
cross_val = pd.DataFrame(models)
cross_val.head()
test = pd.read_csv('../input/titanic/test.csv')
test.head()
passenger_id = test['PassengerId']
test = test.drop(columns = ['PassengerId','Age','SibSp','Parch','Ticket','Cabin','Embarked'])
test.head()
test.isnull().sum()
test['Fare'].mode()
test['Fare'] = test['Fare'].fillna(7.75)
get_initials(test,'Name')
test['Name'] = le.fit_transform(test['Name'])
test['Sex'] = le.fit_transform(test['Sex'])
test.head()
test = sc.fit_transform(test)
prediction = classifier_knn.predict(test)
submission = pd.DataFrame({'PassengerId':passenger_id,'Survived':prediction})
submission.to_csv('TITANIC_SUBMISSION.csv',index = False)
submission