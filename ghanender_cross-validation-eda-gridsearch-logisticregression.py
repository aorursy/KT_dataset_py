import pandas as pd
import numpy as np
import matplotlib
%matplotlib inline
from sklearn.metrics import confusion_matrix, accuracy_score
from scipy.stats import mode
import matplotlib.pyplot as plt 
train = pd.read_csv(r'...\train.csv')
test = pd.read_csv(r'....\test.csv')
full = pd.concat([train, test], keys=['train','test'])
#full = pd.concat([train, test])
full.head()

full['LastName'] = full.Name.str.split(',').apply(lambda x: x[0]).str.strip()
full['Title'] = full.Name.str.split("[\,\.]").apply(lambda x: x[1]).str.strip()
print(full.Title.value_counts())
##if the title is Dr and the sex is female, we'll update the Title as Miss
full.loc[(full.Title == 'Dr') & (full.Sex == 'female'), 'Title'] = 'Mrs'

##if the title is in any of the following, we'll update the Title as Miss
full.loc[full.Title.isin(['Lady','Mme','the Countess','Dona']), 'Title'] = 'Mrs'

##if the title is in any of the following, we'll update the Title as Miss
full.loc[full.Title.isin(['Ms','Mlle']), 'Title'] = 'Miss'

##if the title is Dr and the sex is female, we'll update the Title as Mr
full.loc[(full.Title == 'Dr') & (full.Sex == 'male'), 'Title'] = 'Mr'

##if the title is Rev and the sex is male, we'll update the Title as Mr
full.loc[(full.Title == 'Rev') & (full.Sex == 'male'), 'Title'] = 'Mr'

## Setting all the Rev, Col, Major, Capt, Sir --> Mr
full.loc[full.Title.isin(['Rev','Col','Major','Capt','Sir','Don','Jonkheer']) & (full.Sex == 'male'), 'Title'] = 'Mr'
def passenger_type (row):
   if row['Age'] < 2 :
      return 'Infant'
   elif (row['Age'] >= 2 and row['Age'] < 12):
      return 'Child'
   elif (row['Age'] >= 12 and row['Age'] < 18):
      return 'Youth'
   elif (row['Age'] >= 18 and row['Age'] < 65):
      return 'Adult'
   elif row['Age'] >= 65:
      return 'Senior'
   elif row['Title'] == 'Master':
      return 'Child'
   elif row['Title'] == 'Miss':
      return 'Child'
   elif row['Title'] == 'Mr' or row['Title'] == 'Mrs':
      return 'Adult'
   else:
      return 'Unknown'
full['PassengerType'] = full.apply(lambda row: passenger_type(row),axis=1)
full
#Now to see the distribution
full['PassengerType'].value_counts()
#factorize the PassengerType to make it numeric values
full['PassengerType'] = pd.factorize(full['PassengerType'])[0]
full['PassengerType'].value_counts()
#full = pd.get_dummies(full, columns=['PassengerType'])
#factorize the PassengerType to make it numeric values
full['Title'] = pd.factorize(full['Title'])[0]
full['Title'].value_counts()
full['Fare'].isnull().sum()
full.loc[full.Fare.isnull()]
full.loc[full.Fare.isnull(), 'Fare'] = full.loc[(full.Embarked == 'S') & (full.Pclass == 3),'Fare'].median()
full.head()
# Now let's check for nulls in the Embarked column.
full.Embarked.isnull().sum()
print(full.groupby(['Pclass', 'Embarked'])['Fare'].median())
full.loc[full.Embarked.isnull(), 'Embarked'] = 'C'
# We'll now create a bin for the Fare ranges. splitting into 6 groups seems to be a reasonable split.
full['Fare_bin'] = pd.qcut(full['Fare'], 6)
#Creating new family_size column
full['FamilySize'] = full['SibSp'] + full['Parch'] + 1
#The fare for the 2 rows is 80. Let's see which class and Embarked combination gives the closest Median Fare to 80
print(full.groupby(['Pclass', 'Embarked'])['Fare'].median())

#Boxplot to show the median values for different groups. (1,c) has a median value of 80
medianprops = dict(linestyle='-', linewidth=1, color='k')
full.boxplot(column='Fare',by=['Pclass','Embarked'], medianprops=medianprops, showmeans=False, showfliers=False)
#full = pd.get_dummies(full, columns=['Embarked'])
full['Embarked'] = pd.factorize(full['Embarked'])[0]
full['Gender'] = pd.factorize(full['Sex'])[0]
full.info()
full.rename(columns={"Fare_[0, 7.75]": "Fare_1"
                                ,"Fare_(7.75, 7.896]": "Fare_2"
                                ,"Fare_(7.896, 9.844]": "Fare_3"
                                ,"Fare_(9.844, 14.454]": "Fare_4"
                                ,"Fare_(14.454, 24.15]": "Fare_5"
                                ,"Fare_(24.15, 31.275]": "Fare_6"
                                ,"Fare_(31.275, 69.55]": "Fare_7"
                                ,"Fare_(69.55, 512.329]": "Fare_8"}, inplace=True)
full.info()
cols = full.columns.tolist()
cols
feature_cols = ['Fare', 'Parch', 'SibSp', 'Pclass', 'FamilySize', 'Title','PassengerType', 'Gender']
AgeNotNull = full.loc[full.Age.notnull(),:].copy()
AgeNull = full.loc[full.Age.isnull(),:].copy()
X = AgeNotNull[feature_cols]
y = AgeNotNull.Age

# follow the usual sklearn pattern: import, instantiate, fit
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X, y)

p = lm.predict(AgeNotNull[feature_cols])

# Now we can constuct a vector of errors
err = abs(p-y)
#print(y[:10])
#print(p[:10])
# Let's see the error on the first 10 predictions
print (err[:10])
# predict for a new observation
p1 = lm.predict(AgeNull[feature_cols])
print(p1[:10])
p1.shape
AgeNull.shape
AgeSer = full.loc[full.Age.notnull(),'Age']
plt.hist(AgeSer)
plt.ylabel("Count")
plt.xlabel("Age")
plt.show()
full.loc[full.Age.isnull(), 'Age'] = p1
train = full.loc['train']
test = full.loc['test']
y = train.loc[:,'Survived']
X = train.loc[:,['PassengerId','Age','Fare', 'Pclass','Title','PassengerType','FamilySize','Embarked','Gender']]
train_data = train.values
train_X = X.values
train_y = y.values
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.model_selection import GridSearchCV ,train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score ,roc_curve,auc
logreg = LogisticRegression(class_weight='balanced')
param = {'C':[0.001,0.003,0.005,0.01,0.03,0.05,0.1,0.3,0.5,1,2,3,3,4,5,10,20]}
clf = GridSearchCV(logreg,param,scoring='roc_auc',refit=True,cv=10)
clf.fit(X,y)
print('Best roc_auc: {:.4}, with best C: {}'.format(clf.best_score_, clf.best_params_))
kf = StratifiedKFold(n_splits=5,shuffle=True,random_state=20)
pred_test_full =0
cv_score =[]
i=1
for train_index,test_index in kf.split(X,y):
    print('{} of KFold {}'.format(i,kf.n_splits))
    xtr,xvl = X.loc[train_index],X.loc[test_index]
    ytr,yvl = y.loc[train_index],y.loc[test_index]
    
    #model
    lr = LogisticRegression(C=2)
    lr.fit(xtr,ytr)
    score = roc_auc_score(yvl,lr.predict(xvl))
    print('ROC AUC score:',score)
    
    i+=1
