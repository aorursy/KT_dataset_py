#%matplotlib inline
import numpy as np
import pandas as pd
import re as re
import seaborn as sns
train = pd.read_csv('../input/train.csv', header = 0, dtype={'Age': np.float64})
test  = pd.read_csv('../input/test.csv' , header = 0, dtype={'Age': np.float64})
train.columns
train.head()
train.describe()
train.info()
'''
Append train and test data so that all the data manipulations are common to both
'''
full_data = train.append(test)
full_data.reset_index(inplace=True)

print (train.info())
pd.set_option('display.expand_frame_repr', False)
print (train.describe())
pd.set_option('display.expand_frame_repr', True)
print (train.head())
print (train.describe(include=['O']))
print (train.head())
print (train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean())
print (train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean())
print (train[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean())
print (train[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean())
full_data['FamilySize'] = full_data['SibSp'] + full_data['Parch'] + 1
print (full_data[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean())
full_data['IsAlone'] = 0
full_data.loc[full_data['FamilySize'] == 1, 'IsAlone'] = 1
print (full_data[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean())
print (full_data[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).count())
full_data['Embarked'] = full_data['Embarked'].fillna('S')
print (full_data[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean())
full_data['Fare'] = full_data['Fare'].fillna(full_data['Fare'].median())
full_data['CategoricalFare'] = pd.qcut(full_data['Fare'], 8)
print (full_data[['CategoricalFare', 'Survived']].groupby(['CategoricalFare'], as_index=False).mean())
age_avg 	   = full_data['Age'].mean()
age_std 	   = full_data['Age'].std()
age_null_count = full_data['Age'].isnull().sum()
    
age_null_random_list = np.random.randint(age_avg - 1.5*age_std, age_avg + 1.5*age_std, size=age_null_count)
full_data['Age'][np.isnan(full_data['Age'])] = age_null_random_list
full_data['Age'] = full_data['Age'].astype(int)
    
full_data['CategoricalAge'] = pd.cut(full_data['Age'], 8)

print (full_data[['CategoricalAge', 'Survived']].groupby(['CategoricalAge'], as_index=False).mean())
def get_title(name):
	title_search = re.search(' ([A-Za-z]+)\.', name)
	# If the title exists, extract and return it.
	if title_search:
		return title_search.group(1)
	return ""

full_data['Title'] = full_data['Name'].apply(get_title)

print(pd.crosstab(full_data['Title'], full_data['Sex']))
full_data['Title'] = full_data['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

full_data['Title'] = full_data['Title'].replace('Mlle', 'Miss')
full_data['Title'] = full_data['Title'].replace('Ms', 'Miss')
full_data['Title'] = full_data['Title'].replace('Mme', 'Mrs')

print (full_data[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())
#Turning cabin number into Deck
cabin_list = ['A', 'B', 'C', 'D', 'E', 'F', 'T', 'G', 'Unknown']
#full_data['Deck']=full_data['Cabin'].map(lambda x: substrings_in_string(x, cabin_list))
a= full_data['Cabin'].astype(str).str[0]
full_data['Cabin']=a.str.upper()
print (full_data[['Cabin','Survived']].groupby(['Cabin'], as_index=False).mean())
print (full_data[['Cabin','Pclass','Survived']].groupby(['Cabin','Pclass'], as_index=False).mean())
import matplotlib.pyplot as plt
full_data.columns,train.columns,train.index
#full_data.loc[train.index,:]
f, ax = plt.subplots(figsize=[10,10])
sns.heatmap(full_data.loc[train.index,:].corr(),
            annot=True, fmt=".2f",cbar_kws={'label': 'Percentage %'},cmap="plasma",ax=ax)
ax.set_title("Correlation Plot")
plt.show()
full_data['Sex'] = full_data['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
    
# Mapping titles
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
full_data['Title'] = full_data['Title'].map(title_mapping)
full_data['Title'] = full_data['Title'].fillna(0)
    
    # Mapping Embarked
full_data['Embarked'] = full_data['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
    
    # Mapping Fare

full_data.loc[ full_data['Fare'] <= 7.75, 'Fare'] 						            = 0
full_data.loc[(full_data['Fare'] > 7.75) & (full_data['Fare'] <= 7.896), 'Fare']    = 1
full_data.loc[(full_data['Fare'] > 7.896) & (full_data['Fare'] <= 10.008), 'Fare']  = 2
full_data.loc[(full_data['Fare'] > 10.008) & (full_data['Fare'] <= 14.454), 'Fare'] = 3
full_data.loc[(full_data['Fare'] > 14.454) & (full_data['Fare'] <= 24.15), 'Fare']  = 4
full_data.loc[(full_data['Fare'] > 24.15) & (full_data['Fare'] <= 31.275), 'Fare']  = 5    
full_data.loc[(full_data['Fare'] > 31.275) & (full_data['Fare'] <= 69.55), 'Fare']  = 6 
full_data.loc[(full_data['Fare'] > 69.55) , 'Fare']  = 7    

full_data['Fare'] = full_data['Fare'].astype(int)


    # Mapping Age

full_data.loc[ full_data['Age'] <= 10, 'Age'] 					         = 0
full_data.loc[(full_data['Age'] > 10) & (full_data['Age'] <= 20), 'Age'] = 1
full_data.loc[(full_data['Age'] > 20) & (full_data['Age'] <= 30), 'Age'] = 2
full_data.loc[(full_data['Age'] > 30) & (full_data['Age'] <= 40), 'Age'] = 3
full_data.loc[(full_data['Age'] > 40) & (full_data['Age'] <= 50), 'Age'] = 4
full_data.loc[(full_data['Age'] > 50) & (full_data['Age'] <= 60), 'Age'] = 5
full_data.loc[(full_data['Age'] > 60) & (full_data['Age'] <= 70), 'Age'] = 6
full_data.loc[ full_data['Age'] > 70, 'Age']                             = 7

# Mapping Cabin
cabin_mapping={"A": 1, "B": 2, "C": 3, "D": 4, "E": 5,"F": 6,"G": 7 , "T":8,"N":0}
full_data['Cabin'] = full_data['Cabin'].map(cabin_mapping)
full_data['Cabin'] = full_data['Cabin'].fillna(0)

full_data.head()

# Feature Selection
drop_elements = ['index','Name', 'Ticket', 'SibSp',\
                 'Parch', 'FamilySize']
full_data = full_data.drop(drop_elements, axis = 1)
full_data = full_data.drop(['CategoricalAge', 'CategoricalFare'], axis = 1)
train = full_data.iloc[:891]
test = full_data.iloc[891:]
targets = full_data['Survived'].iloc[:891]

test.drop(['Survived','PassengerId'],axis=1,inplace=True)
train.drop(['Survived','PassengerId'],axis=1,inplace=True)
print (full_data[full_data['Survived'].isnull()])
import matplotlib.gridspec as gridspec


from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier()
model.fit(train, targets)
print(model.feature_importances_)
feat_importances = pd.Series(model.feature_importances_, index=train.columns)
feat_importances.nlargest(15).plot(kind='barh')
plt.show()

df= full_data
gs = gridspec.GridSpec(28,1)
plt.figure(figsize=(6,28*4))
for i,col in enumerate(df[df.iloc[:,0:28].columns]):
    ax = plt.subplot(gs[i])
    sns.distplot(df[col][df.Survived ==1],bins=50,color='b')
    sns.distplot(df[col][df.Survived ==0],bins=50,color='r')
    #ax.set_title('feature '+str(col))
plt.show()
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
# encode Sex labels using one-hot encoding scheme
gen_ohe = OneHotEncoder()
gen_feature_arr = gen_ohe.fit_transform(full_data[['Sex']]).toarray()
#gen_feature_labels = list(gen_le.classes_)
gen_features = pd.DataFrame(gen_feature_arr)

df = pd.concat([full_data, gen_features], axis=1)
df.rename(columns={0:'female',1:'male'},inplace=True)
#df.drop('Sex',axis=1,inplace=True)

# encode Pclass status labels using one-hot encoding scheme
leg_ohe = OneHotEncoder()
leg_feature_arr = leg_ohe.fit_transform(df[['Pclass']]).toarray()
leg_features = pd.DataFrame(leg_feature_arr)

leg_features.rename(columns={0:'Pclass0',1:'Pclass1',2:'Pclass2'},inplace=True)

df = pd.concat([df, leg_features], axis=1)
df.drop('Pclass',axis=1,inplace=True)

# encode Title status labels using one-hot encoding scheme
leg_title = OneHotEncoder()
leg_feature_title = leg_title.fit_transform(df[['Title']]).toarray()
leg_features = pd.DataFrame(leg_feature_title)
leg_features.rename(columns={0:'Title0',1:'Title1',2:'Title2',3:'Title3',4:'Title4'},inplace=True)

df = pd.concat([df, leg_features], axis=1)
#df.drop('Title',axis=1,inplace=True)

'''
# encode Fare status labels using one-hot encoding scheme
leg_title = OneHotEncoder()
leg_feature_fare = leg_title.fit_transform(df[['Fare']]).toarray()
leg_features = pd.DataFrame(leg_feature_fare)
leg_features.rename(columns={0:'Fare0',1:'Fare1',2:'Fare2',3:'Fare3'},inplace=True)

df = pd.concat([df, leg_features], axis=1)
#df.drop('Fare',axis=1,inplace=True)
'''

'''
# encode Cabin status labels using one-hot encoding scheme
leg_title = OneHotEncoder()
leg_feature_fare = leg_title.fit_transform(df[['Cabin']]).toarray()
leg_features = pd.DataFrame(leg_feature_fare)
leg_features.rename(columns={1:'CabA', 2:'CabB', 3:'CabC', 4:'CabD', 5:'CabE',6:'CabF', 7:'CabG' , 8:'CabT',0:'CabN'},inplace=True)

df = pd.concat([df, leg_features], axis=1)
df.drop('Cabin',axis=1,inplace=True)
'''


# encode Embarked status labels using one-hot encoding scheme
leg_ohe = OneHotEncoder()
leg_feature_arr = leg_ohe.fit_transform(df[['Embarked']]).toarray()
leg_features = pd.DataFrame(leg_feature_arr)

leg_features.rename(columns={0:'Embarked0',1:'Embarked1',2:'Embarked2'},inplace=True)

df = pd.concat([df, leg_features], axis=1)
df.drop('Embarked',axis=1,inplace=True)
train = df.iloc[:891]
train.drop(['Survived','PassengerId','Title','Sex'],axis=1,inplace=True)
targets = df['Survived'].iloc[:891]

model = ExtraTreesClassifier()
model.fit(train, targets)
print(model.feature_importances_)
feat_importances = pd.Series(model.feature_importances_, index=train.columns)
feat_importances.nlargest(15).plot(kind='barh')
plt.show()
train.columns
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

print("Start classifer comparison")
col_selected = ['male','female','Fare','Age','Pclass0','Pclass1','Pclass2','Title0','Title1','Title2','Title3','Title4']

classifiers = [
    KNeighborsClassifier(3),
    SVC(probability=True),
    DecisionTreeClassifier(),
    RandomForestClassifier(n_estimators=1788,min_samples_split= 5, min_samples_leaf= 4, max_features= 'auto', max_depth= 32, bootstrap=True),
	AdaBoostClassifier(),
    GradientBoostingClassifier(),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis(),
    LogisticRegression(),
    XGBClassifier()]

log_cols = ["Classifier", "Accuracy"]
log 	 = pd.DataFrame(columns=log_cols)

sss = StratifiedShuffleSplit(n_splits=10, test_size=0.1, random_state=0)

full_data = df

train = full_data.iloc[:891]
test = full_data.iloc[891:]
targets = full_data['Survived'].iloc[:891]

test.drop('Survived',axis=1,inplace=True)
train.drop('Survived',axis=1,inplace=True)

test = test[col_selected]
train = train[col_selected]

#targets = pd.read_csv('../input/train.csv', usecols=['Survived'])['Survived'].values



clf = DecisionTreeClassifier()
clf = clf.fit(train, targets)
'''
clf = DecisionTreeClassifier(max_features ='sqrt',splitter='random',max_depth = 50 )
clf = clf.fit(train, targets)
'''
train_predictions = clf.predict(test).astype(int)

df_output = pd.DataFrame()
aux = pd.read_csv('../input/test.csv')
df_output['PassengerId'] = aux['PassengerId']
df_output['Survived'] = train_predictions
df_output[['PassengerId','Survived']].to_csv('titanic_submission_final.csv', index=False)
print("File saved")


X = train[col_selected]
y = targets
acc_dict = {}

for train_index, test_index in sss.split(X, y):
	X_train, X_test = X.iloc[train_index], X.iloc[test_index]
	y_train, y_test = y.iloc[train_index], y.iloc[test_index]
	  
	for clf in classifiers:
		name = clf.__class__.__name__
		clf.fit(train, targets)
		train_predictions = clf.predict(X_test)
		acc = accuracy_score(y_test, train_predictions)
		if name in acc_dict:
			acc_dict[name] += acc
		else:
			acc_dict[name] = acc

for clf in acc_dict:
	acc_dict[clf] = acc_dict[clf] / 10.0
	log_entry = pd.DataFrame([[clf, acc_dict[clf]]], columns=log_cols)
	log = log.append(log_entry)

plt.xlabel('Accuracy')
plt.title('Classifier Accuracy')

sns.set_color_codes("muted")
sns.barplot(x='Accuracy', y='Classifier', data=log, color="b")


clf = RandomForestClassifier(n_estimators=1788,min_samples_split= 5, min_samples_leaf= 4, max_features= 'auto', max_depth= 32, bootstrap=True)
clf.fit(X, y)
result = clf.predict(test).astype(int)

df_output = pd.DataFrame()
aux = pd.read_csv('../input/test.csv')
df_output['PassengerId'] = aux['PassengerId']
df_output['Survived'] = result
df_output[['PassengerId','Survived']].to_csv('titanic_submission_final_20190926.csv', index=False)
print("File saved")

log
'''
clf.get_params()
from sklearn.model_selection import RandomizedSearchCV
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 2000, num = 10)]
max_features = ['auto', 'log2']
max_depth = [int(x) for x in np.linspace(10, 110, num = 10)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

rf = RandomForestClassifier()
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
rf_random.fit(X, y)
print(rf_random.best_params_)
'''