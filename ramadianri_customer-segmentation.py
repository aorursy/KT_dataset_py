import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set()

import warnings
warnings.filterwarnings('ignore')
train = pd.read_csv('/kaggle/input/customer/Train.csv')
test = pd.read_csv('/kaggle/input/customer/Test.csv')
sample_submission = pd.read_csv('/kaggle/input/customer/sample_submission.csv')
train.head()
#combine train and test dataset
combined_data = pd.concat([train, test], ignore_index=True)

combined_data.info()
combined_data.describe(include='all')
print('Percentage of missing values:')
print('-----------------------------')
print(combined_data.isnull().sum().sort_values(ascending=False)[1:] / 10695 * 100)
#check for imbalance
sns.countplot(train['Segmentation'], order=['A','B','C','D']);
#plot the distribution of numerical features
train.hist(bins=50,figsize=(10,10),grid=False)
plt.tight_layout()
plt.show()
#check for duplicate
print('Duplicated value(s) on the train dataset : ', train.duplicated().sum())
print('Duplicated value(s) on the test dataset  : ', test.duplicated().sum())
sns.countplot('Family_Size', hue='Segmentation', data=train, hue_order=['A','B','C','D']);
#fill with mode
train['Family_Size'].fillna(train['Family_Size'].mode()[0], inplace=True)
test['Family_Size'].fillna(test['Family_Size'].mode()[0], inplace=True)
sns.countplot('Ever_Married', hue='Segmentation', data=train, hue_order=['A','B','C','D']);
#fill with mode
train['Ever_Married'].fillna(train['Ever_Married'].mode()[0], inplace=True)
test['Ever_Married'].fillna(test['Ever_Married'].mode()[0], inplace=True)
sns.countplot(y='Profession', hue='Segmentation', data=train, hue_order=['A','B','C','D']);
train.loc[(train['Profession'].isnull() & (train['Segmentation']=='C')),['Profession']] = 'Artist'
train.loc[(train['Profession'].isnull() & (train['Segmentation']=='D')),['Profession']] = 'Healthcare'
#fill with mode
train['Profession'].fillna(train['Profession'].mode()[0], inplace=True)
test['Profession'].fillna(test['Profession'].mode()[0], inplace=True)
sns.countplot('Var_1', hue='Segmentation', data=train, hue_order=['A','B','C','D']);
#fill with mode
train['Var_1'].fillna(train['Var_1'].mode()[0], inplace=True)
test['Var_1'].fillna(test['Var_1'].mode()[0], inplace=True)
sns.countplot('Graduated', hue='Segmentation', data=train, hue_order=['A','B','C','D']);
train.loc[(train['Graduated'].isnull() & (train['Segmentation']=='D')), ['Graduated']] = 'No'
train['Graduated'].fillna(train['Graduated'].mode()[0], inplace=True)
test['Graduated'].fillna(test['Graduated'].mode()[0], inplace=True)
#check for missing values
train.isnull().sum().sort_values(ascending=False)
#copy features that are needed later
target_array = train['Segmentation'].copy()
test_id = test['ID'].copy()

#drop features
train.drop(['Segmentation'], axis=1, inplace=True)

print('train shape: ', train.shape)
print('test shape: ', test.shape)
#create Work_Experience_given feature
train['Work_Experience_is_given']=train['Work_Experience'].notnull()*1
test['Work_Experience_is_given']=train['Work_Experience'].notnull()*1

#fill missing values
train['Work_Experience'].fillna(train['Work_Experience'].mode()[0], inplace=True)
test['Work_Experience'].fillna(test['Work_Experience'].mode()[0], inplace=True)
#convert age in bins
#train['Age']=pd.cut(train['Age'],bins=[10,20,30,40,50,60,70,80,90],labels=[15,25,35,45,55,65,75,85])
#test['Age']=pd.cut(test['Age'],bins=[10,20,30,40,50,60,70,80,90],labels=[15,25,35,45,55,65,75,85])

#train['Age']=train['Age'].astype('int')
#test['Age']=test['Age'].astype('int')
from sklearn import preprocessing

le = preprocessing.LabelEncoder()

for feature in ['Gender','Ever_Married','Graduated','Profession','Spending_Score','Var_1']:
    train[feature]=le.fit_transform(train[feature])
    test[feature]=le.transform(test[feature])
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

train = pd.DataFrame(scaler.fit_transform(train), columns=train.columns)
test = pd.DataFrame(scaler.transform(test), columns=test.columns)
#define a normality test function
def normalityTest(data, alpha=0.05):
    """data (array)   : The array containing the sample to be tested.
	   alpha (float)  : Significance level.
	   return True if data is normal distributed"""
    
    from scipy import stats
    
    statistic, p_value = stats.normaltest(data)
    
    #null hypothesis: array comes from a normal distribution
    if p_value < alpha:  
        #The null hypothesis can be rejected
        is_normal_dist = False
    else:
        #The null hypothesis cannot be rejected
        is_normal_dist = True
    
    return is_normal_dist
#check normality of all numericaal features and transform it if not normal distributed
for feature in train.columns:
    if (train[feature].dtype != 'object'):
        if normalityTest(train[feature]) == False:
            train[feature] = np.log1p(train[feature])
            test[feature] = np.log1p(test[feature])
X = train
y = target_array

X_to_be_predicted = test
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)
from lightgbm import LGBMClassifier

#tuning the model
model = LGBMClassifier(learning_rate=0.1,
                       n_estimators=1200,
                       max_depth=5,
                       min_child_weight=1,
                       gamma=0,
                       subsample=0.8,
                       colsample_bytree=0.8,
                       nthread=4,
                       scale_pos_weight=3,
                       seed=27)

#fitting
model.fit(X_train, y_train)
from sklearn.metrics import classification_report

#print a classification report
print(classification_report(y_test, model.predict(X_test)))
#make a prediction
y_predict = model.predict(X_to_be_predicted)
y_predict
#sava results to a file
results = pd.DataFrame({'ID': test_id, 'Segmentation': y_predict})
results.to_csv('my_submission.csv', index=False)
results.head()
