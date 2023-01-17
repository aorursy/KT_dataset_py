%load_ext autoreload
%autoreload 2
%matplotlib inline

from fastai.imports import *
from fastai.structured import *
from pandas_summary import DataFrameSummary
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier,GradientBoostingClassifier
from IPython.display import display
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
import seaborn as sns
import pylab as plot
params = { 
    'axes.labelsize': "large",
    'xtick.labelsize': 'x-large',
    'legend.fontsize': 20,
    'figure.dpi': 150,
    'figure.figsize': [10, 6]
}
plot.rcParams.update(params)
PATH = "../input/"
train_raw=pd.read_csv(f'{PATH}train.csv',low_memory=False)
test_raw=pd.read_csv(f'{PATH}test.csv',low_memory=False)
#check the size of the train and test data

train_raw.shape, test_raw.shape
#check if the data loaded properly

train_raw.head()
train_raw.describe()
data = train_raw.copy()
data['Died']= 1 - data['Survived']
data.groupby('Sex').agg('sum')[['Survived','Died']].plot(kind='bar',stacked=True)
sns.violinplot(x='Sex', y='Age', hue='Survived',data=data,split=True)
#Check the Fare ticket of each passagener and see how it impact the survival.

figure = plt.figure(figsize=(32,16))
plt.hist([data[data['Survived'] == 1]['Fare'], data[data['Survived'] == 0]['Fare']], 
         stacked=True,
         bins = 50, label = ['Survived','Dead'])
plt.xlabel('Fare')
plt.ylabel('Number of Passengers')
plt.legend();
#combining age, fare and survival in one chart.

plt.figure(figsize=(25, 7))
ax = plt.subplot()

ax.scatter(data[data['Survived'] == 1]['Age'], data[data['Survived'] == 1]['Fare'], 
           c='green', s=data[data['Survived'] == 1]['Fare'])
ax.scatter(data[data['Survived'] == 0]['Age'], data[data['Survived'] == 0]['Fare'], 
           c='red', s=data[data['Survived'] == 0]['Fare']);
#ticket fare versues class

ax = plt.subplot()
ax.set_ylabel('Average fare')
data.groupby('Pclass').mean()['Fare'].plot(kind='bar',ax=ax)
#Combining the test and train data to prepare the data for modeling.

x_train = train_raw.drop(['Survived'],1)
y_train = train_raw['Survived']
x_test = test_raw
df_combined = x_train.append(x_test)
df_combined.shape
df_combined.head()
def display_all(df):
    with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000): 
        display(df)
display_all(df_combined.tail().T)
#train_cats module from fastai, which changes the strings in a dataframe to a 
#categorical values

train_cats(df_combined)
#Check the missing data %
display_all(df_combined.isnull().sum().sort_index()/len(df_combined))
#proc_df takes a data frame df and splits off the response variable, and
#changes the df into an entirely numeric dataframe. In this case am excluding the 
# fields in ignore_flds as they need further processing.

df,y,nas = proc_df(df_combined,y_fld=None,ignore_flds=['Age','Name','Embarked','Cabin','Parch',
                                                      'SibSp'])
df.head()
def process_family():
    
    global df
    # introducing a new feature : the size of families (including the passenger)
    df['FamilySize'] = df['Parch'] + df['SibSp'] + 1
    
    # introducing other features based on the family size
    df['Singleton'] = df['FamilySize'].map(lambda s: 1 if s == 1 else 0)
    df['SmallFamily'] = df['FamilySize'].map(lambda s: 1 if 2 <= s <= 4 else 0)
    df['LargeFamily'] = df['FamilySize'].map(lambda s: 1 if 5 <= s else 0)    
    return df
df = process_family()
def process_embarked():
    global df
    # two missing embarked values - filling them with the most frequent one in the train  set(S)
    df.Embarked.fillna('S', inplace=True)
    # dummy encoding 
    df_dummies = pd.get_dummies(df['Embarked'], prefix='Embarked')
    df = pd.concat([df, df_dummies], axis=1)
    df.drop('Embarked', axis=1, inplace=True)
#     status('embarked')
    return df
df = process_embarked()
def process_cabin():
    global df    
    # replacing missing cabins with U (for Uknown)
    df.Cabin.fillna('T', inplace=True)
    
    # mapping each Cabin value with the cabin letter
    df['Cabin'] = df['Cabin'].map(lambda c: c[0])
    
    # dummy encoding ...
    cabin_dummies = pd.get_dummies(df['Cabin'], prefix='Cabin')    
    df = pd.concat([df, cabin_dummies], axis=1)

    df.drop('Cabin', axis=1, inplace=True)
#     status('cabin')
    return df
df = process_cabin()
titles = set()
for name in df['Name']:
    titles.add(name.split(',')[1].split('.')[0].strip())
Title_Dictionary = {
    "Capt": "Officer",
    "Col": "Officer",
    "Major": "Officer",
    "Jonkheer": "Royalty",
    "Don": "Royalty",
    "Sir" : "Royalty",
    "Dr": "Officer",
    "Rev": "Officer",
    "the Countess":"Royalty",
    "Mme": "Mrs",
    "Mlle": "Miss",
    "Ms": "Mrs",
    "Mr" : "Mr",
    "Mrs" : "Mrs",
    "Miss" : "Miss",
    "Master" : "Master",
    "Lady" : "Royalty"
}

def get_titles():
    # we extract the title from each name
    df['Title'] = df['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())
    
    # a map of more aggregated title
    # we map each title
    df['Title'] = df.Title.map(Title_Dictionary)
#     status('Title')
    return df
df = get_titles()
df.head()
#summarize the Age grouped by sex, class and title
grouped_train = df.groupby(['Sex','Pclass','Title'])
grouped_median_train = grouped_train.median()
grouped_median_train = grouped_median_train.reset_index()[['Sex', 'Pclass', 'Title', 'Age']]
grouped_median_train.head()
df.head()
#Assing the value of age for missing values based on the group.
#If a title is miising then the age will be assigned based on sex and class.

def fill_age(row):
    condition = (
        (grouped_median_train['Sex'] == row['Sex']) & 
        (grouped_median_train['Title'] == row['Title']) & 
        (grouped_median_train['Pclass'] == row['Pclass'])
    ) 
    if np.isnan(grouped_median_train[condition]['Age'].values[0]):
        print('true')
        condition = (
            (grouped_median_train['Sex'] == row['Sex']) & 
            (grouped_median_train['Pclass'] == row['Pclass'])
        )

    return grouped_median_train[condition]['Age'].values[0]


def process_age():
    global df
    # a function that fills the missing values of the Age variable
    df['Age'] = df.apply(lambda row: fill_age(row) if np.isnan(row['Age']) else row['Age'], axis=1)
#     status('age')
    return df
df = process_age()
#Check for missing values.

display_all(df.isnull().sum().sort_index()/len(df))
df[df.Title.isnull()]
def process_names():
    global df
    # we clean the Name variable
    df.drop('Name', axis=1, inplace=True)
    
    # encoding in dummy variable
    titles_dummies = pd.get_dummies(df['Title'], prefix='Title')
    df = pd.concat([df, titles_dummies], axis=1)
    
    # removing the title variable
    df.drop('Title', axis=1, inplace=True)
    
#     status('names')
    return df
df = process_names()
df.head()
#Now no null vlaues
display_all(df.isnull().sum().sort_index()/len(df))
#Seperate out the train & test data

x_train = df[:891].copy()
x_test = df[891:].copy()
x_train.shape,x_test.shape
#split the tarin data into train and valid set
def split_vals(a,n): return a[:n], a[n:]
valid_count =60
n_trn = len(x_train)-valid_count
x_train1, x_valid1 = split_vals(x_train, n_trn)
y_train1, y_valid1 = split_vals(y_train, n_trn)
x_train1.shape,y_train1.shape,x_valid1.shape,y_valid1.shape
m = RandomForestClassifier(n_estimators=180,min_samples_leaf=4,max_features=0.5,n_jobs=-1)
m.fit(x_train1,y_train1)
m.score(x_train1,y_train1)
y_predict=m.predict(x_valid1)
from sklearn.metrics import accuracy_score
accuracy_score(y_valid1,y_predict)
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_valid1,y_predict))
#confusion Matrix
print(confusion_matrix(y_valid1,y_predict))
#Feature importance
fi = rf_feat_importance(m, x_train1); fi[:10]
def plot_fi(fi): return fi.plot('cols', 'imp', 'barh', figsize=(12,7), legend=False)
plot_fi(fi[:30]);
# Keeping only the variables which are significant for the model(>0.01)
to_keep = fi[fi.imp>0.01].cols; len(to_keep)
to_keep
#Now training the model on the entire data with only the important features.
x_train = x_train[to_keep]
x_train
m = RandomForestClassifier(n_estimators=200,min_samples_leaf=3,max_features=0.5,n_jobs=-1)
m.fit(x_train,y_train)
m.score(x_train,y_train)
x_test = x_test[to_keep]
output=m.predict(x_test).astype(int)
output.size
# aux=pd.read_csv(f'{PATH}test.csv',low_memory=False)
# df_output = pd.DataFrame()
# df_output['PassengerId'] = aux['PassengerId']
# df_output['Survived'] = output
# df_output[['PassengerId','Survived']].to_csv(f'{PATH}titanic_fastai2.csv', index=False)