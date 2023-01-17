import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import seaborn as sns
 
%matplotlib inline 
%config InlineBackend.figure_format = 'retina' ## This is preferable for retina display. 

import warnings
warnings.simplefilter(action='ignore')

import os ## imporing os
print(os.listdir("../input/"))
## Importing the datasets
model = pd.read_csv("../input/titanic/train.csv", index_col='PassengerId')
pred = pd.read_csv("../input/titanic/test.csv", index_col='PassengerId')
model.head(5)
dfs = [model,pred]

for i in dfs:
    print(i.isnull().sum())
    del i['Cabin']
    del i['Ticket']
    i['Age_copy'] = i['Age']
num_col = model.select_dtypes(exclude='object').columns
columns = len(num_col)/4+1

fg, ax = plt.subplots(figsize=(20, 5))

for i, col in enumerate(num_col):
    fg.add_subplot(columns, 4, i+1)
    sns.boxplot(model.select_dtypes(exclude='object')[col])
    plt.xlabel(col)
    plt.xticks(rotation=90)

plt.tight_layout()
plt.show()
model.shape
dfs = model

for i in dfs.columns:
    if dfs[i].dtype == 'int64' or dfs[i].dtype == 'float64':
        if dfs[i].value_counts().count() > 30:
            if i == 'Age_copy':
                pass
            else:
                Q1 = dfs[i].quantile(0.05)
                Q3 = dfs[i].quantile(0.95)
                IQR = Q3 - Q1
                dfs = dfs[~((dfs[i] < (Q1 - 1.5 * IQR)) | (dfs[i] > (Q3 + 1.5 * IQR)))]
                dfs[i] = dfs[i].fillna(dfs[i].median())
                print (i, '......Numeric Values, DROP OUTFITS, Q1:',(Q1 - 1.5 * IQR).round(2),'Q3:',(Q3 + 1.5 * IQR).round(2), '/ NAN: Mean')
        else:
            print (i, '......Numeric Categorical / NAN: 0')
            dfs[i] = dfs[i].fillna('0')
    else: 
        if dfs[i].value_counts().count() < 30:
            dfs[i] = dfs[i].fillna(dfs[i].mode().iloc[0]) #most frequent value
            print (i, '......String Categorical / NAN: Most frequent value') 
        else:
            dfs[i] = dfs[i].fillna(dfs[i].mode().iloc[0]) #most frequent value
            print (i, '......String Non Categorical / NAN: Most frequent value')
            
model = dfs
model.shape
dfs = pred

for i in dfs.columns:
    if dfs[i].dtype == 'int64' or dfs[i].dtype == 'float64':
        if dfs[i].value_counts().count() > 30:
            if i == 'Age_copy':
                pass
            else:
                dfs[i] = dfs[i].fillna(dfs[i].median())
                print (i, '......Numeric Values / NAN: mean')
        else:
            print (i, '......Numeric Categorical / NAN: 0')
            dfs[i] = dfs[i].fillna('0')
    else: 
        if dfs[i].value_counts().count() < 30:
            dfs[i] = dfs[i].fillna(dfs[i].mode().iloc[0]) #most frequent value
            print (i, '......String Categorical / NAN: Most frequent value')
        else:
            dfs[i] = dfs[i].fillna(dfs[i].mode().iloc[0]) #most frequent value
            print (i, '......String Non Categorical / NAN: Most frequent value')
            
pred = dfs
num_col = model.select_dtypes(exclude='object').columns
columns = len(num_col)/4+1

fg, ax = plt.subplots(figsize=(20, 5))

for i, col in enumerate(num_col):
    fg.add_subplot(columns, 4, i+1)
    sns.boxplot(model.select_dtypes(exclude='object')[col])
    plt.xlabel(col)
    plt.xticks(rotation=90)

plt.tight_layout()
plt.show()
dfs = [model,pred]

for i in dfs:
    print(i.isnull().sum())
for i in dfs:
    i['Title'] = i['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
    title_names = (i['Title'].value_counts() < 10)
    i['Title'] = i['Title'].apply(lambda x: 'Misc' if title_names.loc[x] == True else x)
    print(i['Title'].value_counts())
    
for i in dfs:    
    i['Age'] = i['Age'].round(0)
    i['Age'] = i['Age'].astype(np.int64)
    
    i['FamilySize'] = i['SibSp'] + i['Parch'] + 1
    i['RealFare'] = i['Fare']/i['FamilySize']
b = model.groupby(['Pclass','Title'])
b_median = b.median()
b = b_median.reset_index()[['Pclass', 'Title', 'Age']]

c = pred.groupby(['Pclass','Title'])
c_median = c.median()
c = c_median.reset_index()[['Pclass', 'Title', 'Age']]

b.head()
#fill nan in age_copy dependent to Sex, Pclass and Title, for model and pred.
for key, value in model['Age_copy'].iteritems(): 
    if pd.isna(model['Age_copy'][key]):
        for key2, value2 in b['Age'].iteritems():
            if model['Title'][key] == b['Title'][key2] and model['Pclass'][key] == b['Pclass'][key2]:
                model['Age_copy'][key] = b['Age'][key2]
    else:
        model['Age_copy'][key] = model['Age_copy'][key]
        
for key, value in pred['Age_copy'].iteritems(): 
    if pd.isna(pred['Age_copy'][key]):
        for key2, value2 in c['Age'].iteritems():
            if pred['Title'][key] == c['Title'][key2] and pred['Pclass'][key] == c['Pclass'][key2]:
                pred['Age_copy'][key] = c['Age'][key2]
    else:
        pred['Age_copy'][key] = pred['Age_copy'][key]

plt.figure(figsize=[14,4])
ax = sns.barplot(data=model, x='Pclass', y='Fare', hue='FamilySize')
ax = ax.set_title("Real Fare (Check Fare by Family members and Class)", fontsize=24)
plt.figure(figsize=[15,4])
ax = sns.barplot(data=model, x='Pclass', y='RealFare', hue='FamilySize')
ax = ax.set_title("Real Fare (Fare divided by Family members)", fontsize=24)
plt.figure(figsize=[15,4])
ax = sns.barplot(data=model, x='Pclass', y='RealFare', hue='Title')
ax = ax.set_title("Title related with RealFare paid", fontsize=24)
fig = plt.figure(figsize=(15,4))
ax = sns.barplot(y=model.corr()['Survived'],x=model.corr().index)
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
columns = ['Sex', 'Title']

for i in dfs:
    i['AgeBin'] = pd.cut(i['Age_copy'].astype(int), 4)
    i['FareBin'] = pd.qcut(i['RealFare'], 4)
    #i['Fare'] = i['Fare'].apply(np.log1p)
    del i['Name']
    del i['Age']
    #del i['Age_copy']
    #del i['AgeBin']
    #del i['SibSp']
    #del i['Parch']
    #del i['Fare']
    #del i['Embarked']
    #del i['FareBin']
    del i['RealFare']
    #i['Age_copy'] = i['Age_copy'].round(0)
    #i['Age_copy'] = i['Age_copy'].astype(np.int64)
    #i['Alone'] = i['FamilySize'].apply(lambda x: 0 if x > 1 else 1)
    del i['FamilySize']
    #i['AgeBinmix'] = pd.cut(i['Age_mix'].astype(int), 5)
    i['Sex'].replace([0,1],['Female','Male'],inplace=True)
    
    for col in columns:
        le.fit(i[col])
        i[col] = le.transform(i[col])
def dummies(df,column):
    titles_dummies = pd.get_dummies(df[column], prefix=column)
    df = pd.concat([df, titles_dummies], axis=1)
    df.drop(column, axis=1, inplace=True)
    return df

model = dummies(model,'Title')
pred = dummies(pred,'Title')
model = dummies(model,'Pclass')
pred = dummies(pred,'Pclass')
model = dummies(model,'Embarked')
pred = dummies(pred,'Embarked')
model = dummies(model,'FareBin')
model = dummies(model,'AgeBin')
pred = dummies(pred,'FareBin')
pred = dummies(pred,'AgeBin')
model.head(10)
fig = plt.figure(figsize=(18,4))
ax = sns.barplot(y=model.corr()['Survived'],x=model.corr().index)
ax = plt.xticks(rotation=90)
from pandas_profiling import ProfileReport
#profile = model.profile_report(title='Pandas Profiling Report')
#profile.to_notebook_iframe()
#profile.to_file("your_report.html")
# Importing the necessary library
from tpot import TPOTRegressor

mod = model
mY = mod['Survived']
del mod['Survived']
mX = mod
## It's a implementation of some customized models to do in future
tpot_config = {
    'sklearn.ensemble.GradientBoostingRegressor': {
        ''
    },
    'xgboost.XGBRegressor': {
        'alpha': [1e-3, 1e-2, 1e-1, 1., 10., 100.],
        'fit_prior': [True, False]
    },
    'sklearn.naive_bayes.MultinomialNB': {
        'alpha': [1e-3, 1e-2, 1e-1, 1., 10., 100.],
        'fit_prior': [True, False]
    }
}

# We will create our TPOT regressor with commonly used arguments
tpot = TPOTRegressor(verbosity=2, scoring='r2', cv=3, 
                      n_jobs=-1, generations=10, config_dict='TPOT light',
                      population_size=20, random_state=3,
                      early_stop = 5)
#fiting our tpot auto model
tpot.fit(mX, mY)


# Generate the predictions
submission = tpot.predict(pred)
submission = submission.round(0).astype(int)
# Create the submission file
final = pd.DataFrame({'PassengerId': pred.index, 'Survived': submission})
final.to_csv('submissionTPOT.csv', index = False)
import h2o
from h2o.automl import H2OAutoML

# Initialize your cluster
h2o.init()
model=h2o.H2OFrame(model)
pred=h2o.H2OFrame(pred)

#model = h2o.import_file("../input/titanic/train.csv")
#pred = h2o.import_file("../input/titanic/test.csv")

#all_train = h2o.H2OFrame(df['Survived'].notna()])
#all_test = h2o.H2OFrame(df['Survived'].isna()])
x = model.columns
y = "Survived"

# For binary classification, response should be a factor
model[y] = model[y].asfactor()
x.remove(y)
#no se si aixo fa algo!
#train, valid, test = model.split_frame(ratios=[0.7, 0.15], seed=42)

#Covert dtype to factor as per H2O implementation
#train[y] = train[y].asfactor()
#valid[y] = valid[y].asfactor()
#test[y] = test[y].asfactor()
aml = H2OAutoML(max_models = 20, max_runtime_secs=200, seed = 42, stopping_metric = 'RMSLE')
aml.train(x = x, y = y, training_frame = model)
lb = aml.leaderboard; lb
aml.leader.varimp_plot()
preds = aml.leader.predict(pred)
preds
predictions = preds[0].as_data_frame().values.flatten()
sample_submission = pd.read_csv("../input/titanic/gender_submission.csv")
sample_submission['Survived'] = predictions
sample_submission.to_csv('h2O_titanic_1.csv', index=False)