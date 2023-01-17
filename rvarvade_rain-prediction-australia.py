# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

df = pd.read_csv("../input/weather-dataset-rattle-package/weatherAUS.csv")

df.head()
df.shape
col_names = df.columns

col_names
df.drop(['RISK_MM'], axis = 1, inplace = True)

df.info()
# find categorical variables



categorical = [var for var in df.columns if df[var].dtype=='O']



print('There are {} categorical variables\n'.format(len(categorical)))



print('The categorical variables are :', categorical)
df[categorical].head()
df[categorical].isnull().sum()
cat1 = [var for var in categorical if df[var].isnull().sum()!=0]



print(df[cat1].isnull().sum())
# view frequency of categorical variables



for var in categorical: 

    

    print(df[var].value_counts())
# view frequency distribution of categorical variables



for var in categorical:

    

    print(df[var].value_counts()/np.float(len(df)))

    
for var in categorical:

    

    print(var, 'contains' , len(df[var].unique()),'labels')
df['Date'].dtypes
# parse the dates, currently coded as strings, into datetime format



df['Date'] = pd.to_datetime(df['Date'])

df['Year'] = df['Date'].dt.year

df['Year'].head()
df['Month'] = df['Date'].dt.month

df['Month'].head()
df['Day'] = df['Date'].dt.day



df['Day'].head()
df.info()
df.drop('Date', axis = 1, inplace = True)

df.head()
# find categorical variables



categorical = [var for var in df.columns if df[var].dtype=='O']



print('There are {} categorical variables\n'.format(len(categorical)))



print('The categorical variables are :', categorical)
# check for missing values in categorical variables

df[categorical].isnull().sum()
#print the number of location variables



print('Location counts',len(df.Location.unique()), 'labels')
df.Location.unique()
df.Location.value_counts()
pd.get_dummies(df.Location, drop_first =True).head()
print('WindGustDir contains', len(df['WindGustDir'].unique()), 'labels')
df['WindGustDir'].unique()
df.WindGustDir.value_counts()
pd.get_dummies(df.WindGustDir, drop_first = True, dummy_na = True).head()
pd.get_dummies(df.WindGustDir, drop_first = True, dummy_na = True).sum(axis=0)
print('WindDir9am contains', len(df['WindDir9am'].unique()), 'labels')
df.WindDir9am.unique()
df.WindDir9am.value_counts()
pd.get_dummies(df.WindDir9am, drop_first =True, dummy_na = True).head()
pd.get_dummies(df.WindDir9am, drop_first = True, dummy_na = True).sum(axis=0)
print('WindDir3pm contains', len(df['WindDir9am'].unique()), 'labels')
df.WindDir3pm.unique()
df.WindDir3pm.value_counts()
pd.get_dummies(df.WindDir3pm, drop_first =True, dummy_na = True).head()
pd.get_dummies(df.WindDir3pm, drop_first = True, dummy_na = True).sum(axis=0)
print('RainToday contains', len(df['RainToday'].unique()), 'labels')
df['RainToday'].unique()

df.RainToday.value_counts()
pd.get_dummies(df.RainToday, drop_first =True, dummy_na = True).head()
pd.get_dummies(df.RainToday, drop_first =True, dummy_na = True).sum(axis=0)
numerical = [var for var in df.columns if df[var].dtype!='O']



print('There are {} numerical variables \n'.format(len(numerical)))



print('The numerical variables are:', numerical)
df[numerical].head()
# check missing values in numerical variables



df[numerical].isnull().sum()
# view summary statistics in numerical variables



print(round(df[numerical].describe()),2)
# draw boxplots to visualize outliers

import matplotlib.pyplot as plt



plt.figure(figsize=(15,10))



plt.subplot(2,2,1)

fig = df.boxplot(column = 'Rainfall')

fig.set_title('')

fig.set_ylabel('Rainfall')





plt.subplot(2,2,2)

fig = df.boxplot(column = 'Evaporation')

fig.set_title('')

fig.set_ylabel('Evaporation')



plt.subplot(2,2,3)

fig = df.boxplot(column = 'WindSpeed9am')

fig.set_title('')

fig.set_ylabel('WindSpeed9am')



plt.subplot(2,2,4)

fig = df.boxplot(column = 'WindSpeed3pm')

fig.set_title('')

fig.set_ylabel('WindSpeed3pm')
# plot histogram to check distribution



plt.figure(figsize=(15,10))



plt.subplot(2,2,1)

fig = df.Rainfall.hist(bins=10)

fig.set_xlabel('Rainfall')

fig.set_ylabel('Rain Tomorrow')



plt.subplot(2,2,2)

fig = df.Evaporation.hist(bins=10)

fig.set_xlabel('Evaporation')

fig.set_ylabel('Rain Tommorow')



plt.subplot(2,2,3)

fig = df.WindSpeed9am.hist(bins=10)

fig.set_xlabel('WindSpeed9am')

fig.set_ylabel('Rain Tommorow')



plt.subplot(2,2,4)

fig = df.WindSpeed3pm.hist(bins=10)

fig.set_xlabel('WindSpeed3pm')

fig.set_ylabel('Rain Tommorow')
# find outliers for Rainfall variable

IQR = df.Rainfall.quantile(0.75) - df.Rainfall.quantile(0.25)

Lower_fence = df.Rainfall.quantile(0.25) - (IQR * 3)

Upper_fence = df.Rainfall.quantile(0.75) + (IQR * 3)

print('Rainfall outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence, upperboundary=Upper_fence))




IQR = df.Evaporation.quantile(0.75) - df.Evaporation.quantile(0.25)

Lower_fence = df.Evaporation.quantile(0.25) - (IQR*3)

Upper_fence = df.Evaporation.quantile(0.75) + (IQR*3)

print('Evaporation outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary = Lower_fence, upperboundary = Upper_fence))
IQR = df.WindSpeed9am.quantile(0.75) - df.WindSpeed9am.quantile(0.25)

Lower_fence = df.WindSpeed9am.quantile(0.25) - (IQR*3)

Upper_fence = df.WindSpeed9am.quantile(0.75) + (IQR*3)

print('WindSpeed9am outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary = Lower_fence, upperboundary = Upper_fence))

IQR = df.WindSpeed3pm.quantile(0.75) - df.WindSpeed3pm.quantile(0.25)

Lower_fence = df.WindSpeed3pm.quantile(0.25) - (IQR*3)

Upper_fence = df.WindSpeed3pm.quantile(0.75) + (IQR*3)

print('WindSpeed3pm outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary = Lower_fence, upperboundary = Upper_fence))

X = df.drop(['RainTomorrow'], axis=1)



y = df['RainTomorrow']
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
X_train.shape, X_test.shape
X_train.dtypes
# display categorical variables



categorical = [col for col in X_train.columns if X_train[col].dtypes == 'O']



categorical

numerical = [col for col in X_train.columns if X_train[col].dtypes != 'O']



numerical
# check missing values in numerical variables in X_train



X_train[numerical].isnull().sum()

X_test[numerical].isnull().sum()
for col in numerical:

    if X_train[col].isnull().mean()>0:

        print(col, round(X_train[col].isnull().mean(), 4))
# impute missing values in X_train and X_test with respective column median in X_train



for df1 in [X_train, X_test]:

    for col in numerical:

        col_median=X_train[col].median()

        df1[col].fillna(col_median, inplace=True)  
# check again missing values in numerical variables in X_train



X_train[numerical].isnull().sum()
X_test[numerical].isnull().sum()
X_train[categorical].isnull().mean()
for col in categorical:

    if X_train[col].isnull().mean()>0:

        print(col, (X_train[col].isnull().mean()))

        
# impute missing categorical variables with most frequent value



for df2 in [X_train, X_test]:

    df2['WindGustDir'].fillna(X_train['WindGustDir'].mode()[0], inplace=True)

    df2['WindDir9am'].fillna(X_train['WindDir9am'].mode()[0], inplace=True)

    df2['WindDir3pm'].fillna(X_train['WindDir3pm'].mode()[0], inplace=True)

    df2['RainToday'].fillna(X_train['RainToday'].mode()[0], inplace=True)
# check missing values in categorical variables in X_train



X_train[categorical].isnull().sum()
X_test[categorical].isnull().sum()
X_train.isnull().sum()
X_test.isnull().sum()
def max_value(df3, variable, top):

    return np.where(df3[variable]>top, top, df3[variable])



for df3 in [X_train, X_test]:

    df3['Rainfall'] = max_value(df3, 'Rainfall', 3.2)

    df3['Evaporation'] = max_value(df3,'Evaporation', 21.8)

    df3['WindSpeed9am'] = max_value(df3, 'WindSpeed9am', 55)

    df3['WindSpeed3pm'] = max_value(df3, 'WindSpeed3pm', 57)
X_train.Rainfall.max(), X_test.Rainfall.max()

X_train.Evaporation.max(), X_test.Evaporation.max()
X_train.WindSpeed9am.max(), X_test.WindSpeed9am.max()

X_train.WindSpeed3pm.max(), X_test.WindSpeed3pm.max()
X_train[numerical].describe()
categorical
X_train[categorical].head()
import category_encoders as ce



encoder = ce.BinaryEncoder(cols = ['RainToday'])



X_train = encoder.fit_transform(X_train)



X_test = encoder.transform(X_test)
X_train.head()

X_train = pd.concat([X_train[numerical], X_train[['RainToday_0', 'RainToday_1']],

                     pd.get_dummies(X_train.Location), 

                     pd.get_dummies(X_train.WindGustDir),

                     pd.get_dummies(X_train.WindDir9am),

                     pd.get_dummies(X_train.WindDir3pm)], axis=1)
X_train.head()
X_test = pd.concat([X_test[numerical], X_test[['RainToday_0', 'RainToday_1']],

                     pd.get_dummies(X_test.Location), 

                     pd.get_dummies(X_test.WindGustDir),

                     pd.get_dummies(X_test.WindDir9am),

                     pd.get_dummies(X_test.WindDir3pm)], axis=1)
X_test.head()
X_train.describe()
cols = X_train.columns
from sklearn.preprocessing import MinMaxScaler



scaler = MinMaxScaler()



X_train = scaler.fit_transform(X_train)



X_test = scaler.fit_transform(X_test)
X_train = pd.DataFrame(X_train, columns = [cols])



X_test = pd.DataFrame(X_test, columns = [cols])
X_train.describe()
from sklearn.linear_model import LogisticRegression



logreg = LogisticRegression(solver = 'liblinear', random_state = 0)



logreg.fit(X_train, y_train)
y_pred_test = logreg.predict(X_test)



y_pred_test
logreg.predict_proba(X_test)[:,0]
logreg.predict_proba(X_test)[:,1]
from sklearn.metrics import accuracy_score



print('Model accuracy score: {0:0.4f}'.format(accuracy_score(y_test, y_pred_test)))
y_pred_train = logreg.predict(X_train)



y_pred_train
print('Training set accuracy score: {0:0.4f}'.format(accuracy_score(y_train, y_pred_train)))
# print the scores on training and test set



print('Training set accuracy score: {:.4f}'.format(accuracy_score(y_train, y_pred_train)))



print('Test set score: {:.4f}'.format(logreg.score(X_test, y_test)))


from sklearn.metrics import confusion_matrix



cm = confusion_matrix(y_test, y_pred_test)



print('Confusion matrix\n\n', cm)



print('\nTrue Positives(TP) = ', cm[0,0])



print('\nTrue Negatives(TN) = ', cm[1,1])



print('\nFalse Positives(FP) = ', cm[0,1])



print('\nFalse Negatives(FN) = ', cm[1,0])
import seaborn as sns

sns.heatmap(cm, annot=True)
sns.heatmap(cm/np.sum(cm), annot=True, 

            fmt='.2%', cmap='Blues')
from sklearn.metrics import classification_report



print(classification_report(y_test, y_pred_test))
TP = cm[0,0]

TN = cm[1,1]

FP = cm[0,1]

FN = cm[1,0]
classification_accuracy = (TP + TN) / float(TP + TN + FP + FN)



print('Classification accuracy:{0:0.4f}'.format(classification_accuracy))
classification_error = (FP + FN) / float(TP + TN + FP + FN)



print('Classification error:{0:0.4f}'.format(classification_error))
y_pred_prob = logreg.predict_proba(X_test)[0:10]



y_pred_prob
# store the probabilities in dataframe



y_pred_prob_df = pd.DataFrame(data=y_pred_prob, columns=['Prob of - No rain tomorrow (0)', 'Prob of - Rain tomorrow (1)'])



y_pred_prob_df
logreg.predict_proba(X_test)[0:10, 1]
y_pred1 = logreg.predict_proba(X_test)[:,1]
# plot histogram of predicted probabilities





# adjust the font size 

plt.rcParams['font.size'] = 12





# plot histogram with 10 bins

plt.hist(y_pred1, bins = 10)





# set the title of predicted probabilities

plt.title('Histogram of predicted probabilities of rain')





# set the x-axis limit

plt.xlim(0,1)





# set the title

plt.xlabel('Predicted probabilities of rain')

plt.ylabel('Frequency')



from sklearn.model_selection import cross_val_score



scores = cross_val_score(logreg, X_train, y_train, cv = 5, scoring = 'accuracy')



print('Cross validation scores:{}'.format(scores))
# compute Average cross-validation score



print('Cross validation score:{}'.format(scores.mean()))
from sklearn.model_selection import GridSearchCV



scores = []



parameters = [{'penalty': ['l1','l2']},

             {'C': [0.1 , 1 , 10 , 100, 1000]}]



grid_search = GridSearchCV(estimator = logreg,

                          param_grid = parameters, 

                          scoring = 'accuracy',

                          cv = 5,

                          verbose = 0)



grid_search.fit(X_train, y_train)

scores.append({

        'model':model_name,

        'best_score':grid_search.best_score_,

        'best_params':grid_search.best_params_

    })