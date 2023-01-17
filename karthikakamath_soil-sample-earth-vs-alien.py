# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('../input/soil-sample/soil_samples_training.csv')
df.head()
df.shape
df['origin'].value_counts()
df.isnull().sum().sort_values(ascending=False)
df[df.origin=='earth'].describe()
df[df.origin=='alien'].describe()
df[df.origin=='earth'].grain_shape.value_counts()
df[df.origin=='alien'].grain_shape.value_counts()
df[df.origin=='earth'].particle_type.value_counts()
df[df.origin=='alien'].particle_type.value_counts()
import matplotlib.pyplot as plt

import seaborn as sns

from matplotlib.pyplot import xticks

%matplotlib inline



df.drop(['sample_id'],axis=1,inplace=True)

# Correlation matrix

plt.figure(figsize=(17,10))

sns.heatmap(df.corr(method='spearman').round(2), annot=True,cmap="YlGnBu")
# treating missing value

# for numeric variables, substituting with mean of the column grouped by origin 

# for categorical variables, substituting with mode of the column grouped by origin



def num_impute(df,var):

    df[var] = df.groupby("origin")[var].transform(lambda x: x.fillna(x.mean()))

        

def cat_impute(df,var):

    df[var] = df.groupby("origin")[var].transform(lambda x: x.fillna(x.mode().iloc[0]))

num_impute(df,'nitrate')

num_impute(df,'radioactivity')

num_impute(df,'optical_density')

num_impute(df,'chloride')

num_impute(df,'phosphate')

cat_impute(df,'grain_shape')

cat_impute(df,'particle_type')
df.isnull().sum().sort_values(ascending=False)
df.info()
df.head()
# selecting all categorical features

feature_cols = df.select_dtypes(include=['object']).columns



# selecting all numeric features

num_cols = df.select_dtypes(include = ['float64']).columns
print(feature_cols)

print(num_cols)
fig, axes = plt.subplots(ncols=2, nrows=6, figsize=(15,15))

xticks(rotation = 90)

for i,ax in zip(range(len(feature_cols)-1), axes.flat):

    chart = sns.countplot(x = feature_cols[i+1] , hue = "origin", data = df, ax=ax)

    chart.set_xticklabels(chart.get_xticklabels(), rotation=45)

plt.tight_layout()

plt.show()
fig, axes = plt.subplots(ncols=2, nrows=5, figsize=(15,15))

xticks(rotation = 90)

for i,ax in zip(range(len(num_cols)), axes.flat):

    chart = sns.boxplot(y = num_cols[i], x = 'origin', data = df, ax=ax)

    chart.set_xticklabels(chart.get_xticklabels(), rotation=45)

plt.tight_layout()

plt.show()
df['origin'] = df['origin'].map({'earth': 1, 'alien': 0})
# List of binary yes/no variables to map



varlist =  ['particle_attached', 'organics']



# Defining the map function

def binary_map(x):

    return x.map({'yes': 1, "no": 0})





df[varlist] = df[varlist].apply(binary_map)
df.head(10)
# Creating a dummy variable for some of the categorical variables and dropping the first one.

dummy1 = pd.get_dummies(df[['grain_shape','grain_surface','grain_color','particle_type','particle_color','particle_width','particle_distribution','solubles','isotope_diversity']], drop_first=True)

dummy1.head()
# Adding the results to the master dataframe

df_dm = pd.concat([df, dummy1], axis=1)

df_dm.head()
df_dm = df_dm.drop(['grain_shape','grain_surface','grain_color','particle_type','particle_color','particle_width','particle_distribution','solubles','isotope_diversity'], axis = 1)

df_dm.head()
from sklearn.model_selection import train_test_split



# Putting feature variable to X

X = df_dm.drop(['origin'], axis=1)
y = df_dm['origin']
from sklearn.preprocessing import MinMaxScaler



scaler = MinMaxScaler()



X[num_cols] = scaler.fit_transform(X[num_cols])



X.head()
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=42)
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()



from sklearn.feature_selection import RFE

rfe = RFE(logreg, 15)             # running RFE with 15 variables as output

rfe = rfe.fit(X_train, y_train)
list(zip(X_train.columns, rfe.support_, rfe.ranking_))
col = X_train.columns[rfe.support_]

col
import statsmodels.api as sm
X_train_sm = sm.add_constant(X_train[col])

logm = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm.fit()

res.summary()
col1 = col.drop('grain_shape_cylindrical',1)
X_train_sm = sm.add_constant(X_train[col1])

logm = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm.fit()

res.summary()
col2 = col1.drop('grain_color_green',1)
X_train_sm = sm.add_constant(X_train[col2])

logm = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm.fit()

res.summary()
col3 = col2.drop('solubles_medium',1)
X_train_sm = sm.add_constant(X_train[col3])

logm = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm.fit()

res.summary()
col4 = col3.drop('grain_color_red',1)
X_train_sm = sm.add_constant(X_train[col4])

logm = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm.fit()

res.summary()
# Getting the predicted values on the train set

y_train_pred = res.predict(X_train_sm)

y_train_pred[:10]
y_train_pred = y_train_pred.values.reshape(-1)

y_train_pred[:10]
y_train_pred_final = pd.DataFrame({'Actual':y_train.values, 'earth_prob':y_train_pred})

y_train_pred_final.head()
y_train_pred_final['Predicted'] = y_train_pred_final.earth_prob.map(lambda x: 1 if x > 0.5 else 0)





y_train_pred_final.head()
from sklearn import metrics



# Confusion matrix 

confusion = metrics.confusion_matrix(y_train_pred_final.Actual, y_train_pred_final.Predicted )

print(confusion)
# Let's check the overall accuracy.

print(metrics.accuracy_score(y_train_pred_final.Actual, y_train_pred_final.Predicted))
def draw_roc( actual, probs ):

    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,

                                              drop_intermediate = False )

    auc_score = metrics.roc_auc_score( actual, probs )

    plt.figure(figsize=(5, 5))

    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )

    plt.plot([0, 1], [0, 1], 'k--')

    plt.xlim([0.0, 1.0])

    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')

    plt.ylabel('True Positive Rate')

    plt.title('Receiver operating characteristic example')

    plt.legend(loc="lower right")

    plt.show()



    return None
fpr, tpr, thresholds = metrics.roc_curve( y_train_pred_final.Actual, y_train_pred_final.earth_prob, drop_intermediate = False )
draw_roc(y_train_pred_final.Actual, y_train_pred_final.earth_prob)
# Let's create columns with different probability cutoffs 

numbers = [float(x)/10 for x in range(10)]

for i in numbers:

    y_train_pred_final[i]= y_train_pred_final.earth_prob.map(lambda x: 1 if x > i else 0)

y_train_pred_final.head()
# Now let's calculate accuracy sensitivity and specificity for various probability cutoffs.

cutoff_df = pd.DataFrame( columns = ['prob','accuracy','sensi','speci'])

from sklearn.metrics import confusion_matrix



num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

for i in num:

    cm1 = metrics.confusion_matrix(y_train_pred_final.Actual, y_train_pred_final[i] )

    total1=sum(sum(cm1))

    accuracy = (cm1[0,0]+cm1[1,1])/total1

    

    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])

    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])

    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]

print(cutoff_df)
# Let's plot accuracy sensitivity and specificity for various probabilities.

cutoff_df.plot.line(x='prob', y=['accuracy','sensi','speci'])

plt.show()
#### From the curve above, 0.6 is the optimum point to take it as a cutoff probability.



y_train_pred_final['final_predicted'] = y_train_pred_final.earth_prob.map( lambda x: 1 if x > 0.6 else 0)



y_train_pred_final.head()
# Let's check the overall accuracy.

metrics.accuracy_score(y_train_pred_final.Actual, y_train_pred_final.final_predicted)
X_val = X_val[col4]

X_val.head()
X_val_sm = sm.add_constant(X_val)
y_val_pred = res.predict(X_val_sm)
y_pred_1 = pd.DataFrame(y_val_pred)

y_val_df = pd.DataFrame(y_val)

y_pred_1.reset_index(drop=True, inplace=True)

y_val_df.reset_index(drop=True, inplace=True)

y_pred_final = pd.concat([y_val_df, y_pred_1],axis=1)

y_pred_final= y_pred_final.rename(columns={ 0 : 'earth_prob'})

y_pred_final['final_predicted'] = y_pred_final.earth_prob.map(lambda x: 1 if x > 0.6 else 0)

y_pred_final.head()
# Let's check the overall accuracy.

metrics.accuracy_score(y_pred_final.origin, y_pred_final.final_predicted)
from sklearn.linear_model import LogisticRegression

from sklearn import metrics

from sklearn.metrics import accuracy_score
df.head()
X = df.loc[:,df.columns != 'origin']

y = df['origin']
# label encoding categorical features

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

X[feature_cols[1:]] = X[feature_cols[1:]].apply(LabelEncoder().fit_transform)
X.head()
scaler = MinMaxScaler()



X[num_cols] = scaler.fit_transform(X[num_cols])



X.head()
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.30, random_state=42)
X_train.head()
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred_logistic = logreg.predict(X_val)
round(accuracy_score(y_val,y_pred_logistic), 4)
from sklearn.tree import DecisionTreeClassifier

from sklearn import metrics



model_dt = DecisionTreeClassifier()
model_dt.fit(X_train, y_train)
y_pred_dt = model_dt.predict(X_val)
round(accuracy_score(y_val,y_pred_dt), 4)
from sklearn.ensemble import RandomForestClassifier



model_rf = RandomForestClassifier()
model_rf.fit(X_train,y_train)
y_pred_rf = model_rf.predict(X_val)
round(accuracy_score(y_val,y_pred_rf), 4)
#trainig the entire train data now on random forest before applying 

# it to test data so that val data that was set aside is also used for training



model_rf.fit(X,y)
df_test = pd.read_csv('../input/soil-sample/soil_samples_test.csv')
df_test.head()
df_test.isnull().sum().sort_values(ascending=False)
def num_impute_test(df,var):

    df[var] = df[var].transform(lambda x: x.fillna(x.mean()))

        

def cat_impute_test(df,var):

    df[var] = df[var].transform(lambda x: x.fillna(x.mode().iloc[0]))



num_impute_test(df_test,'nitrate')

num_impute_test(df_test,'radioactivity')

num_impute_test(df_test,'optical_density')

num_impute_test(df_test,'chloride')

num_impute_test(df_test,'phosphate')

cat_impute_test(df_test,'grain_shape')

cat_impute_test(df_test,'particle_type')
df_test.info()
# List of binary yes/no variables to map



varlist =  ['particle_attached', 'organics']



# Defining the map function

def binary_map(x):

    return x.map({'yes': 1, "no": 0})





df_test[varlist] = df_test[varlist].apply(binary_map)
X_test = df_test.drop('sample_id',axis=1)

X_test.head()
X_test[feature_cols[1:]] = X_test[feature_cols[1:]].apply(le.fit_transform)
X_test.head()
X_test[num_cols] = scaler.transform(X_test[num_cols])



X_test.head()
pred = model_rf.predict(X_test)
output = pd.DataFrame(pred)

output.head()
output = output.rename(columns={ 0 : 'predicted_origin'})
output['sample_id']=df_test['sample_id']

output.head()
output = output.reindex(columns=['sample_id','predicted_origin'])

output.head()
output['predicted_origin']=output['predicted_origin'].apply(lambda x: 'earth' if x==1 else 'alien')
output
output.to_csv('karthika_kamath.csv')
# Creating a dummy variable for some of the categorical variables and dropping the first one.

dummy2 = pd.get_dummies(df_test[['grain_shape','grain_surface','grain_color','particle_type','particle_color','particle_width','particle_distribution','solubles','isotope_diversity']], drop_first=True)

dummy2.head()
# Adding the results to the master dataframe

df_test = pd.concat([df_test, dummy2], axis=1)

df_test.head()
df_test = df_test.drop(['grain_shape','grain_surface','grain_color','particle_type','particle_color','particle_width','particle_distribution','solubles','isotope_diversity'], axis = 1)

df_test.head()
X_test = df_test.drop(['sample_id'],axis=1)
X_test[num_cols] = scaler.fit_transform(X_test[num_cols])



X_test.head()
X_test_sm = X_test[col4]

X_test_sm.head()
X_test_sm = sm.add_constant(X_test_sm)
y_test_pred = res.predict(X_test_sm)
y_test_pred[:10]