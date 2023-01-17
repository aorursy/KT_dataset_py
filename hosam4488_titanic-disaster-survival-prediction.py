import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, Imputer
import warnings
warnings.filterwarnings('ignore')
import os 
print(os.listdir("../input"))
print(os.listdir("../input/")) 
app_train = pd.read_csv('../input/train.csv')
print('Training data shape: ', app_train.shape)
app_train.head() 
app_test = pd.read_csv('../input/test.csv')
print('Testing data shape: ', app_test.shape)
app_test.head() 
app_train['Survived'].value_counts()
plt.hist(app_train['Survived'])
plt.show() 
def missing_values_table(df):
        mis_val = df.isnull().sum()
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        return mis_val_table_ren_columns
    
missing_values = missing_values_table(app_train)
missing_values.head(20)
app_train.dtypes.value_counts() 
app_train.select_dtypes('object').apply(pd.Series.nunique, axis = 0)
le = LabelEncoder()
le_count = 0

for col in app_train:
    if app_train[col].dtype == 'object':
        if len(list(app_train[col].unique())) <= 2:
            le.fit(app_train[col])
            app_train[col] = le.transform(app_train[col])
            app_test[col] = le.transform(app_test[col])
            le_count += 1
            
print('%d columns were label encoded.' % le_count)
def sub_in_str(big_str, substr):
    for sub in range(len(substr)):
        for i in range(len(substr[sub])):
            if big_str.find(substr[sub][i]) != -1:
                return substr[sub]
    return np.nan

cabin_list = ['A', 'B', 'C', 'D', 'E', 'F', 'T', 'G', 'Unknown']
app_train['Deck']=(app_train['Cabin'].map(lambda x: sub_in_str(str(x),cabin_list ))).values
app_train['Deck'] = app_train['Deck'].fillna('Z')
app_train = app_train.drop(['Name', 'Ticket','PassengerId', 'Cabin'], axis=1)

app_test['Deck']=(app_test['Cabin'].map(lambda x: sub_in_str(str(x),cabin_list ))).values
app_test['Deck'] = app_test['Deck'].fillna('Z')
app_test = app_test.drop(['Name', 'Ticket','PassengerId', 'Cabin'], axis=1)

app_train.head()
app_train['Age*Class']=app_train['Age']*app_train['Pclass']
app_train['Family_Size']=app_train['SibSp']+app_train['Parch']
app_train['Fare_Per_Person']=app_train['Fare']/(app_train['Family_Size']+1)

app_test['Age*Class']=app_test['Age']*app_test['Pclass']
app_test['Family_Size']=app_test['SibSp']+app_test['Parch']
app_test['Fare_Per_Person']=app_test['Fare']/(app_test['Family_Size']+1)

app_train.head()
app_train = pd.get_dummies(app_train)
app_test = pd.get_dummies(app_test)

print('Training Features shape: ', app_train.shape)
print('Testing Features shape: ', app_test.shape)
app_train.head()
train_labels = app_train['Survived']
app_train, app_test = app_train.align(app_test, join = 'inner', axis = 1)
app_train['Survived'] = train_labels

print('Training Features shape: ', app_train.shape)
print('Testing Features shape: ', app_test.shape)
correlations = app_train.corr()['Survived'].sort_values()

print('Most Positive Correlations:\n', correlations.tail(15))
print('\nMost Negative Correlations:\n', correlations.head(15))
sns.kdeplot(app_train.loc[app_train['Survived'] == 0, 'Fare'], label = 'target == 0')
sns.kdeplot(app_train.loc[app_train['Survived'] == 1, 'Fare'], label = 'target == 1')
plt.xlabel('Fare'); plt.ylabel('Density'); plt.title('Distribution of Fares');
plt.show()
sns.kdeplot(app_train.loc[app_train['Survived'] == 0, 'Age'], label = 'target == 0')
sns.kdeplot(app_train.loc[app_train['Survived'] == 1, 'Age'], label = 'target == 1')
plt.xlabel('Age'); plt.ylabel('Density'); plt.title('Distribution of Ages');
plt.show()
sns.kdeplot(app_train.loc[app_train['Survived'] == 0, 'Pclass'], label = 'target == 0')
sns.kdeplot(app_train.loc[app_train['Survived'] == 1, 'Pclass'], label = 'target == 1')
plt.xlabel('Pclass'); plt.ylabel('Density'); plt.title('Distribution of Classes');
plt.show() 
age_data = app_train[['Survived', 'Age']]
age_data['Ybin'] = pd.cut(age_data['Age'], bins = np.linspace(0, 80, num = 9))
age_groups  = age_data.groupby('Ybin').mean()

plt.bar(age_groups.index.astype(str), 100 * age_groups['Survived'])
plt.xticks(rotation = 75); plt.xlabel('Age Group (years)'); plt.ylabel('Survival Rate')
plt.title('Titanic Survival by Age Group');
plt.show() 
imp_var = app_train[['Survived', 'Age', 'Fare', 'Pclass', 'Sex']]
imp_var_corr = imp_var.corr()
sns.heatmap(imp_var_corr, cmap = plt.cm.RdYlBu_r, vmin = -0.25, annot = True, vmax = 0.6)
plt.title('Correlation Heatmap');
plt.show() 
if 'Survived' in app_train:
    train = app_train.drop(['Survived'], axis=1)
else:
    train = app_train.copy()

features = list(train.columns)
test = app_test.copy()

imputer = Imputer(strategy = 'median')
imputer.fit(train)
train = imputer.transform(train)
test = imputer.transform(app_test)

scaler = MinMaxScaler(feature_range = (0, 1))
scaler.fit(train)
train = scaler.transform(train)
test = scaler.transform(test)

print('Training data shape: ', train.shape)
print('Testing data shape: ', test.shape) 
t = pd.read_csv('../input/test.csv')

from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(train, train_labels)
log_reg_pred = log_reg.predict(test) 

submit = t[['PassengerId']]
submit['Survived'] = log_reg_pred
submit.to_csv('log_reg.csv', index = False) 
from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators = 80, random_state = 50, verbose = 1, n_jobs = -1)
random_forest.fit(train, train_labels)

feature_importance_values = random_forest.feature_importances_
feature_importances = pd.DataFrame({'feature': features, 'importance': feature_importance_values})
randf_pred = random_forest.predict(test)

submit = t[['PassengerId']]
submit['Survived'] = randf_pred
submit.to_csv('random_forest.csv', index = False)