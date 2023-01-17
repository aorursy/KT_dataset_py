import pandas as pd

from sklearn.model_selection import train_test_split

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
data = pd.read_csv('../input/heart.csv')
data.head(3)
# Check for duplicities

data.duplicated().sum()
data = data.drop_duplicates()
data.shape
data.dtypes
df = data
# Sex

df['sex'][df['sex']==0] = 'female'

df['sex'][df['sex']==1] = 'male'

# cp: chest pain : name of cp from Reference a)

df['cp'][df['cp'] == 0] = 'typical angina'

df['cp'][df['cp'] == 1] = 'atypical angina'

df['cp'][df['cp'] == 2] = 'non-anginal pain'

df['cp'][df['cp'] == 3] = 'asymptomatic'

# fbs : binary variable

df['fbs'][df['fbs'] == 0] = '<120mg/dl'

df['fbs'][df['fbs'] == 1] = '>120mg/dl' 

# restecg 

df['restecg'][df['restecg'] == 0] = 'normal'

df['restecg'][df['restecg'] == 1] = 'ST-T wave abnormality'

df['restecg'][df['restecg'] == 2] = 'left ventricular hypertophy'

# exang

df['exang'][df['exang'] == 0] = 'no'

df['exang'][df['exang'] == 1] = 'yes'

# slope

df['slope'][df['slope'] == 0] = 'upsloping'

df['slope'][df['slope'] == 1] = 'flat'

df['slope'][df['slope'] == 2] = 'downsloping'

# thal

df['thal'][df['thal'] == 0] = 'unknown'

df['thal'][df['thal'] == 1] = 'normal'

df['thal'][df['thal'] == 2] = 'fixed defect'

df['thal'][df['thal'] == 3] = 'reversible defect'

# target

df['target'][df['target'] == 1 ] = 'yes'

df['target'][df['target'] == 0 ] = 'no'
df.dtypes
predictors = df.columns[:-1]

num_vars = ['age','trestbps','chol', 'thalach', 'oldpeak', 'ca']

cat_vars = []

for variable in predictors:

    if variable not in num_vars:

        cat_vars.append(variable)

y = df[['target']]

X = df[predictors]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=111)

X_train.shape # we have 226 training examples
training_data = pd.concat([X_train,y_train],axis=1)

validation_data = pd.concat([X_test,y_test],axis=1)
# summary statistics for numeric variables

X_train.describe()
# response variable : target

y_train['target'].value_counts()

sns.countplot(x = "target", data = y_train, palette = "hls")
# Summary statistics for indepdendent variables 

# Numeric Varibles

# a) Age

plt.figure(figsize = (9,8))

sns.distplot(X_train['age'], color = "g", bins = 10, hist_kws = {'alpha':0.4})
# b) trestbps

plt.figure(figsize = (9,8))

sns.distplot(X_train['trestbps'], color = "g", bins = 10, hist_kws = {'alpha':0.4})

# c) chol

plt.figure(figsize = (9,8))

sns.distplot(X_train['chol'], color = "g", bins = 16, hist_kws = {'alpha':0.4})
# d) thalach

plt.figure(figsize = (9,8))

sns.distplot(X_train['thalach'], color = "g", bins = 10 ,hist_kws = {'alpha':0.4})
# e) oldpeak

plt.figure(figsize = (9,8))

sns.distplot(X_train['oldpeak'], color = "g", bins = 10 ,hist_kws = {'alpha':0.4})
# f) ca

plt.figure(figsize = (9,8))

sns.distplot(X_train['ca'], color = "g", bins = 10 ,hist_kws = {'alpha':0.4})
# Categorical Variables 

for i,var in enumerate(X_train[cat_vars]):

#    print("{Variable}:{Counts}".format(Variable = var, Counts = X_train.groupby([var]).size()))

    plt.figure(i,figsize=(8,6))

    sns.countplot(x=var, data=X_train)
training_data.corr()
plt.figure(figsize=(10,6))

ax = sns.kdeplot(training_data['age'][training_data.target == 'yes'],color = 'r',shade=True)

sns.kdeplot(training_data['age'][training_data.target == 'no'],color = 'b',shade= True)

plt.legend(['yes','no'])

plt.title('Density Plot of Age of Patients - with heart disease (red) and without (blue)')

ax.set(xlabel = 'Age')

plt.xlim(20,80)

plt.show()
plt.figure(figsize=(10,6))

ax = sns.kdeplot(training_data['trestbps'][training_data.target == 'yes'],color = 'r',shade=True)

sns.kdeplot(training_data['trestbps'][training_data.target == 'no'],color = 'b',shade= True)

plt.legend(['yes','no'])

plt.title('Density Plot of Resting B.P. of Patients - with heart disease (red) and without (blue)')

ax.set(xlabel = 'Resting blood pressure')

plt.xlim(80,250)

plt.show()
plt.figure(figsize=(10,6))

ax = sns.kdeplot(training_data['chol'][training_data.target == 'yes'],color = 'r',shade=True)

sns.kdeplot(training_data['chol'][training_data.target == 'no'],color = 'b',shade= True)

plt.legend(['yes','no'])

plt.title('Density Plot of Cholestrol of Patients - with heart disease (red) and without (blue)')

ax.set(xlabel = 'Cholestrol')

plt.xlim(90,500)

plt.show()
plt.figure(figsize=(10,6))

ax = sns.kdeplot(training_data['thalach'][training_data.target == 'yes'],color = 'r',shade=True)

sns.kdeplot(training_data['thalach'][training_data.target == 'no'],color = 'b',shade= True)

plt.legend(['yes','no'])

plt.title('Density Plot of Max. heart rate achieved of Patients - with heart disease (red) and without (blue)')

ax.set(xlabel = 'Maximum heart rate achieved')

plt.xlim(50,250)

plt.show()
plt.figure(figsize=(10,6))

ax = sns.kdeplot(training_data['oldpeak'][training_data.target == 'yes'],color = 'r',shade=True)

sns.kdeplot(training_data['oldpeak'][training_data.target == 'no'],color = 'b',shade= True)

plt.legend(['yes','no'])

plt.title('Density Plot of S.T. depression induced in Patients - with heart disease (red) and without (blue)')

ax.set(xlabel = 'S.T. depression')

plt.xlim(-2,10)

plt.show()
# The following will show distribution of target variable for different groups of indepdendent variable.

for var in cat_vars:

     table=pd.crosstab(training_data[var],training_data['target'])   

     table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
