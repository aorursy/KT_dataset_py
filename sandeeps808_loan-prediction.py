# lets import the required libraries



import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns 

import warnings 

warnings.filterwarnings("ignore") # never prints matching warning.

df = pd.read_csv('../input/loan-prediction-problem-dataset/train_u6lujuX_CVtuZ9i.csv')
df.shape
df.info()  # for the detailed information about each column.

# We can aslo identify if we have missing values.

df.describe
df['Credit_History'] = df['Credit_History'].astype('O')
# describe categorical data ("object")



df.describe(include='O')
#Lets drop the 'Loan_ID' attribute as it doesn't make any impact. 

df.drop('Loan_ID', axis=1, inplace=True)
#To check if there's any duplicate rows.

df.duplicated().any()

plt.figure(figsize=(8,6)) #This creates a figure object, which has a width of 8 inches and 6 inches in height.

sns.countplot(df['Loan_Status']);
print('The percentage of Y class : %.2f' % (df['Loan_Status'].value_counts()[0] / len(df)))

print('The percentage of N class : %.2f' % (df['Loan_Status'].value_counts()[1] / len(df)))
df.columns
#Credit History



grid = sns.FacetGrid(df, col = 'Loan_Status', size = 3.2, aspect = 1.6)

grid.map(sns.countplot, 'Credit_History');
#Gender



grid  = sns.FacetGrid(df, col = 'Loan_Status', size = 3.2, aspect  = 1.6)

grid.map(sns.countplot, 'Gender');
#Married

grid = sns.FacetGrid(df , col = 'Loan_Status', size = 3.2, aspect = 1.6)

grid.map(sns.countplot, 'Married');
#dependent

grid = sns.FacetGrid(df, col = 'Loan_Status', size=3.2, aspect= 1.6)

grid.map(sns.countplot, 'Dependents');
#Education

grid = sns.FacetGrid(df, col = 'Loan_Status', size=3.2, aspect= 1.6)

grid.map(sns.countplot, 'Education');
#Self_Employed

grid = sns.FacetGrid(df, col = 'Loan_Status', size=3.2, aspect= 1.6)

grid.map(sns.countplot, 'Self_Employed');
#Property_Area

grid = sns.FacetGrid(df, col = 'Loan_Status', size=3.2, aspect= 1.6)

grid.map(sns.countplot, 'Property_Area');
#applicant income



plt.scatter(df['ApplicantIncome'], df['Loan_Status']);
df.groupby('Loan_Status').median()
df.isnull().sum().sort_values(ascending = False)
#Now let's start separating categorical and numerical data.



cat_data = []

num_data = []



for i, c in enumerate(df.dtypes) :

    if c == object:

        cat_data.append(df.iloc[:, i])

    else: 

        num_data.append(df.iloc[:, i])
cat_data = pd.DataFrame(cat_data).transpose()

num_data = pd.DataFrame(num_data).transpose()
cat_data.head()
num_data.head()
#for categorical data



cat_data = cat_data.apply(lambda x:x.fillna(x.value_counts().index[0]))



cat_data.isnull().sum().any()
num_data.fillna(method='bfill', inplace=True)



num_data.isnull().sum().any()
from sklearn.preprocessing import LabelEncoder 

le = LabelEncoder()

cat_data.head()
#transform the target column.

target_values = {'Y': 0 , 'N' : 1}



target = cat_data['Loan_Status']

cat_data.drop('Loan_Status', axis=1, inplace=True)



target = target.map(target_values)
#LETS transform all other columns.

for i in cat_data:

    cat_data[i] = le.fit_transform(cat_data[i])
target.head()
cat_data.head()
df = pd.concat([cat_data, num_data, target], axis =1)
df.head()
X = pd.concat([cat_data,num_data], axis= 1)

y = target
from sklearn.model_selection import StratifiedShuffleSplit

sss = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)



for train, test in sss.split(X, y):

    X_train, X_test = X.iloc[train], X.iloc[test]

    y_train, y_test = y.iloc[train], y.iloc[test]

    

print('X_train shape ', X_train.shape)

print('y_train shape', y_train.shape)

print('X_test shape ', X_test.shape)

print('y_test shape', y_test.shape)



# almost same ratio

print('\nratio of target in y_train :',y_train.value_counts().values/ len(y_train))

print('ratio of target in y_test :',y_test.value_counts().values/ len(y_test))

print('ratio of target in original_data :',df['Loan_Status'].value_counts().values/ len(df))



#we can use 4 different algorithms.



from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier



models = {

    'LogisticRegression: ' : LogisticRegression(random_state=42), 

    'KNeighborsClassifier :' : KNeighborsClassifier(),

    'SVC:' : SVC(random_state=42),

    'DecisionTreeClassifier: ': DecisionTreeClassifier(max_depth=1,random_state=42)

}
# loss



from sklearn.metrics import precision_score, recall_score, f1_score, log_loss, accuracy_score



def loss(y_true, y_pred, retu=False):

    pre = precision_score(y_true, y_pred)

    rec = recall_score(y_true, y_pred)

    f1 = f1_score(y_true, y_pred)

    loss = log_loss(y_true, y_pred)

    acc = accuracy_score(y_true, y_pred)

    

    if retu:

        return pre, rec, f1, loss, acc

    else:

        print('  pre: %.3f\n  rec: %.3f\n  f1: %.3f\n  loss: %.3f\n  acc: %.3f' % (pre, rec, f1, loss, acc))


def train_eval_train(models, X, y):

    for name, model in models.items():

        print(name,':')

        model.fit(X, y)

        loss(y, model.predict(X))

        print('-'*30)

        

train_eval_train(models, X_train, y_train)

X_train.shape
#cross_validation



from sklearn.model_selection import StratifiedKFold

skf= StratifiedKFold(n_splits=10, random_state=42, shuffle = True)



def train_eval_cross(models, X , y, folds):

    X = pd.DataFrame(X)

    y = pd.DataFrame(y)

    idx = ['pre', 'rec', 'f1', 'loss', 'acc'] 

    

    for name, model in models.items():

        ls = []

        print(name, ':')

        

        for train, test in folds.split(X, y):

            model.fit(X.iloc[train], y.iloc[train])

            y_pred = model.predict(X.iloc[test])

            ls.append(loss(y.iloc[test], y_pred, retu= True))

        print(pd.DataFrame(np.array(ls).mean(axis =0),index = idx)[0])

        print('-'*30)

        

train_eval_cross(models, X_train, y_train, skf)
data_corr = pd.concat([X_train, y_train], axis=1)

corr = data_corr.corr()

plt.figure(figsize=(10,7))

sns.heatmap(corr, annot=True);
X_train.head()
X_train['new_col'] = X_train['CoapplicantIncome'] / X_train['ApplicantIncome']  

X_train['new_col_2'] = X_train['LoanAmount'] * X_train['Loan_Amount_Term'] 
data_corr = pd.concat([X_train, y_train], axis=1)

corr = data_corr.corr()

plt.figure(figsize=(10,7))

sns.heatmap(corr, annot=True);
X_train.drop(['CoapplicantIncome', 'ApplicantIncome', 'Loan_Amount_Term', 'LoanAmount'], axis=1, inplace=True)
train_eval_cross(models, X_train, y_train, skf)

for i in range(X_train.shape[1]):

    print(X_train.iloc[:,i].value_counts(), end='\n------------------------------------------------\n')
from scipy.stats import norm



fig, ax = plt.subplots(1,2,figsize=(20,5))



sns.distplot(X_train['new_col_2'], ax=ax[0], fit=norm)

ax[0].set_title('new_col_2 before log')



X_train['new_col_2'] = np.log(X_train['new_col_2'])  # here we take the log of all these values.



sns.distplot(X_train['new_col_2'], ax=ax[1], fit=norm)

ax[1].set_title('new_col_2 after log');
train_eval_cross(models, X_train, y_train, skf)

# taking log has drastically improved our model.
print('before:')

print(X_train['new_col'].value_counts())



X_train['new_col'] = [x if x==0 else 1 for x in X_train['new_col']]

print('-'*50)

print('\nafter:')

print(X_train['new_col'].value_counts())
train_eval_cross(models, X_train, y_train, skf)
for i in range(X_train.shape[1]):

    print(X_train.iloc[:,i].value_counts(), end='\n------------------------------------------------\n')

sns.boxplot(X_train['new_col_2']);

plt.title('new_col_2 outliers', fontsize=15);

plt.xlabel('');
threshold= 0.1



new_col_2_out = X_train['new_col_2']

q25, q75 = np.percentile(new_col_2_out, 25), np.percentile(new_col_2_out, 75) # Q25, Q75

print('Quartile 25: {} , Quartile 75: {}'.format(q25, q75))



iqr = q75 - q25

print('iqr: {}'.format(iqr))



cut = iqr * threshold

lower, upper = q25 - cut, q75 + cut

print('Cut Off: {}'.format(cut))

print('Lower: {}'.format(lower))

print('Upper: {}'.format(upper))



outliers = [x for x in new_col_2_out if x < lower or x > upper]

print('Nubers of Outliers: {}'.format(len(outliers)))

print('outliers:{}'.format(outliers))



data_outliers = pd.concat([X_train, y_train], axis=1)

print('\nlen X_train before dropping the outliers', len(data_outliers))

data_outliers = data_outliers.drop(data_outliers[(data_outliers['new_col_2'] > upper) | (data_outliers['new_col_2'] < lower)].index)



print('len X_train before dropping the outliers', len(data_outliers))
X_train = data_outliers.drop('Loan_Status', axis=1)

y_train = data_outliers['Loan_Status']
sns.boxplot(X_train['new_col_2']);

plt.title('new_col_2 without outliers', fontsize=15);

plt.xlabel('');
train_eval_cross(models, X_train, y_train, skf)
# Lets check the correlation.

data_corr = pd.concat([X_train, y_train], axis=1)

corr = data_corr.corr()

plt.figure(figsize=(10,7))

sns.heatmap(corr, annot=True);
X_train.drop(['Self_Employed'], axis=1, inplace=True)



train_eval_cross(models, X_train, y_train, skf)

X_train.drop(['Dependents', 'new_col_2', 'Education', 'Gender', 'Property_Area','Married', 'new_col'], axis=1, inplace=True)
data_corr = pd.concat([X_train, y_train], axis=1)

corr = data_corr.corr()

plt.figure(figsize=(10,7))

sns.heatmap(corr, annot=True);
#X_test1 = pd.read_csv('test.csv')
X_test.head()
X_test_new = X_test.copy()
x = []



X_test_new['new_col'] = X_test_new['CoapplicantIncome'] / X_test_new['ApplicantIncome']  

X_test_new['new_col_2'] = X_test_new['LoanAmount'] * X_test_new['Loan_Amount_Term']

X_test_new.drop(['CoapplicantIncome', 'ApplicantIncome', 'Loan_Amount_Term', 'LoanAmount'], axis=1, inplace=True)



X_test_new['new_col_2'] = np.log(X_test_new['new_col_2'])



X_test_new['new_col'] = [x if x==0 else 1 for x in X_test_new['new_col']]



X_test_new.drop(['Self_Employed'], axis=1, inplace=True)



# drop all the features Except for Credit_History

#X_test_new.drop(['Self_Employed','Dependents', 'new_col_2', 'Education', 'Gender', 'Property_Area','Married', 'new_col'], axis=1, inplace=True)
for name,model in models.items():

    print(name, end=':\n')

    loss(y_test, model.predict(X_test_new))

    print('-'*40)
#So we can see that the logistic and decision tree performs well with the given data. 