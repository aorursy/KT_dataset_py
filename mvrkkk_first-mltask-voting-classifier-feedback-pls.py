# computation
import pandas as pd
import numpy as np

# visualization
import matplotlib.pyplot as plt
import seaborn as sns

# configuration
sns.set_style('whitegrid')
%matplotlib inline
# getting path to train and test data
train_data_path = '../input/train.csv'
test_data_path = '../input/test.csv'

train = pd.read_csv(train_data_path,index_col=None)
test = pd.read_csv(test_data_path,index_col=None)
print('====== Test data - info ======')
print(test.info())
print('\n')
print('====== Train data - info ======')
print(train.info())
fig = plt.figure(figsize=(10,5), dpi = 100)

axe1 = sns.heatmap(train.isnull(),
            cmap = 'coolwarm',
            yticklabels = False,
            cbar = False)
# define an age depending on Pclass
sns.boxplot(data = train,x = 'Pclass',y = 'Age')
# gettina a unique list of classes
classes = train.Pclass.unique()

# define a dict where class+age values will be stored
classes_mean_age = {}

# filling our dictionary with mean ages for a particular class
for _ in classes:
       classes_mean_age[_] = train[train['Pclass'] == _ ]['Age'].mean()

# making a function to map values from dataframe with values from dictionary
def fill_na_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):
        for _ in classes_mean_age:
            if Pclass == _:
                Age = classes_mean_age[_]
                return Age
    else:
        return Age

# performing cleaning on train and test data
train['Age'] = train[['Age','Pclass']].apply(fill_na_age,axis = 1)
test['Age'] = test[['Age','Pclass']].apply(fill_na_age,axis = 1)
# dropping cols and a na values
train.drop(['PassengerId','Ticket','Cabin'],axis = 1,inplace = True)
train.dropna(axis = 0,inplace=True)

test.drop(['Ticket','Cabin'],axis = 1,inplace = True)
test.dropna(axis = 0,inplace=True)

# encoding features 
from sklearn.preprocessing import LabelEncoder

# train
encoder = LabelEncoder()

train['Sex'] = encoder.fit_transform(train['Sex'])
train['Embarked'] = encoder.fit_transform(train['Embarked'].astype(str))
dummies_enbarked = pd.get_dummies(train['Embarked'],prefix='Emb',drop_first=True)
train.drop('Embarked',1,inplace=True)
train.join(dummies_enbarked)


test['Sex'] = encoder.fit_transform(test['Sex'])
test['Embarked'] = encoder.fit_transform(test['Embarked'])
dummies_enbarked_test = pd.get_dummies(test['Embarked'], prefix='Emb',drop_first=True)
test.drop('Embarked',1,inplace=True)
test.join(dummies_enbarked_test)



# Feature engineering and binning
train['age_bin'] = encoder.fit_transform(pd.cut(train.Age,6))
test['age_bin'] = encoder.fit_transform(pd.cut(test.Age,6))


title_list=['Mrs', 'Mr', 'Master', 'Miss', 'Major', 'Rev',
                    'Dr', 'Ms', 'Mlle','Col', 'Capt', 'Mme', 'Countess',
                    'Don', 'Jonkheer']

patern = '('+"|".join(title_list)+')'


train['title'] = train['Name'].str.extract(patern)
test['title'] = test['Name'].str.extract(patern)

def replace_titles(x):
    title=x['title']
    if title in ['Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col']:
        return 'Mr'
    elif title in ['Countess', 'Mme']:
        return 'Mrs'
    elif title in ['Mlle', 'Ms']:
        return 'Miss'
    elif title =='Dr':
        if x['Sex']=='Male':
            return 'Mr'
        else:
            return 'Mrs'
    else:
        return title
train['title']=train.apply(replace_titles, axis=1)
test['title']=test.apply(replace_titles, axis=1)

train['title'] = encoder.fit_transform(train['title'])
test['title'] = encoder.fit_transform(test['title'])



train.drop('Name',1,inplace=True)
test.drop('Name',1,inplace=True)
# setting style
plt.style.use('ggplot')

# creating figure
fig = plt.figure(figsize = (20,10))

# creating axis
ax1 = plt.subplot2grid((3,3),(0,0))
ax2 = plt.subplot2grid((3,3),(0,1))
ax3 = plt.subplot2grid((3,3),(1,0))
ax4 = plt.subplot2grid((3,3),(1,1))
ax5 = plt.subplot2grid((3,3),(2,0),colspan =2)
#ax6 = plt.subplot2grid((3,3),(0,2))
#ax7 = plt.subplot2grid((3,3),(1,2))

# Survival
train.groupby('Survived').size().plot(kind='bar',
                                      cmap = 'winter',
                                      width = 0.8,
                                      ax = ax1)
# Survival by sex
train.groupby(['Sex','Survived']).size().unstack().plot(kind='bar',
                                                        width = 0.8,
                                                        cmap = 'winter',
                                                        ax = ax2)
# Survival by Pclass
train.groupby(['Pclass','Survived'])\
                .size()\
                .unstack()\
                .plot(kind='Bar',stacked = True,
                width = 0.8,
                cmap = 'winter',
                ax = ax3)

# Survival by gender and age distribution
train[(train['Sex'] == 1) & (train['Survived'] == 0)]['Age'].plot('hist',
                                                                       color = 'Blue',
                                                                       alpha = 0.8,
                                                                       ax=ax4,
                                                                       label = 'Male / Died')
train[(train['Sex'] == 0 ) & (train['Survived'] == 0)]['Age'].plot('hist',
                                                                         color = 'lime',
                                                                         alpha = 0.8,
                                                                         ax=ax4,
                                                                         label = 'Female / Died')
ax4.set_xlabel('Age')
ax4.legend()

# Survival by sex and pclass
train.groupby(['Sex','Pclass','Survived'])\
                .size()\
                .unstack()\
                .plot(kind='Bar',
                stacked = True,width = 0.8,
                cmap = 'winter',
                ax = ax5)

train

# adding titles
ax1.set_title('Survivalness')
ax2.set_title('Survivalness by sex')
ax3.set_title('Survivalness by pclass')
ax4.set_title('Age distribution by sex - Died')
ax5.set_title('Survivalness by pclass and sex')

# adding axes labels
axes_list = [ax1,ax2,ax3,ax4,ax5]

for ax in axes_list:
    ax.set_ylabel('# Of records')
    
plt.tight_layout()
from sklearn.preprocessing import StandardScaler

norm = StandardScaler()

X,y = train.iloc[:,1:],train['Survived']

# scaling features
X_sc = norm.fit_transform(X,y)
X_sc_norm = pd.DataFrame(X_sc,columns=X.columns)
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

# split into train test
X_train,X_test,y_train,y_test = train_test_split(X_sc_norm,y,test_size = 0.28,random_state = 111)

# creating instances of models
cf_lg = LogisticRegression(penalty='l2',class_weight='balanced',random_state=1)
cf_rf = RandomForestClassifier(n_estimators=200,random_state=2)
cf_gb = GaussianNB()
cf_kn = KNeighborsClassifier(n_neighbors=3,p=2)

# training ensemble of models
elcf = VotingClassifier(estimators=[('Lg_cf',cf_lg),('rf',cf_lg),('gb',cf_gb),('kn',cf_kn)],voting='hard')
elcf.fit(X_train,y_train)

# making predictions
y_pred = elcf.predict(X_test)
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

acc = accuracy_score(y_test,y_pred)
report = classification_report(y_test,y_pred)
conf = confusion_matrix(y_test,y_pred)

print(conf)
print('\n')
print('==========================Classification report=========================')
print(report)
print('Prediction accuracy:{0}'.format(acc))
prediction = elcf.predict(test.drop('PassengerId',axis=1))
test['y_hat'] = prediction

submission = test[['PassengerId','y_hat']]
submission.head()
