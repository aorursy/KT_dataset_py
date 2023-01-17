import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from scipy import stats

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline
import os

print(os.listdir("../input/heart-disease-uci"))
#bring in the six packs

df_train = pd.read_csv('../input/heart-disease-uci/heart.csv')
df_train.describe()
df_train.head()
df_train.info()
df_train.target.nunique()
df_train.shape
df_train.isnull().sum()
df_train.duplicated().sum()
df_train[df_train.duplicated(keep=False)]
idx = ["Heart Disease Not Present", "Heart Disease Present"]

vals = df_train.target.value_counts().values

fig, ax = plt.subplots()

explode = (0, 0.1)

ax.pie(vals, labels=idx, explode=explode, autopct='%1.1f%%')

ax.axis('equal')

plt.show()
# Drop the duplicates



print("Shape of dataset before dropping duplicates: ", df_train.shape)

df_train = df_train.drop_duplicates()

print("Shape of dataset after dropping duplicates: ", df_train.shape)

df_train.nunique()
df_train.describe(include='all')
cols = ['age', 'trestbps', 'chol', 'oldpeak', 'thalach']



fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 20))

fig.subplots_adjust(hspace=.5, wspace=.5)

fig.suptitle('Distributions of Numerical Feats')



for ax, feature in zip(axes.flatten(), cols):

    sns.distplot(df_train[feature], ax=ax, fit=norm, color='r')  

    

    plt.text(0.0, -0.15,'- Skewness: {0:.2f}'.format(df_train[feature].skew()), fontsize = 14,

             horizontalalignment='left',verticalalignment='center', transform = ax.transAxes)

    

    plt.text(0.0, -0.25,'- Kurtosis: {0:.2f}'.format(df_train[feature].kurtosis()), fontsize = 14, 

             horizontalalignment='left',verticalalignment='center', transform = ax.transAxes)

    

    ax.set(title=feature.upper(), xlabel=feature)

    ax.set_xlim([min(df_train[feature]), max(df_train[feature])])
# Plot side by side histograms



col = "chol"

def applyLog(x):

    return np.log(x+1)



log_col = df_train[col].apply(applyLog)



fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,4), )



sns.distplot(df_train[col], ax=ax[0], color='r')

ax[0].set_title('Distribution of {}'.format(col), fontsize=14)

ax[0].set_xlim([min(df_train[col]), max(df_train[col])])



sns.distplot(log_col, ax=ax[1], color='r')

ax[1].set_title('Distribution of log of {}'.format(col), fontsize=14)

ax[1].set_xlim([min(log_col), max(log_col)])



fig.show()
col = "oldpeak"

def applyLog(x):

    return np.log(x+1)



log_col = df_train[col].apply(applyLog)



fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,4), )



sns.distplot(df_train[col], ax=ax[0], color='r')

ax[0].set_title('Distribution of {}'.format(col), fontsize=14)

ax[0].set_xlim([min(df_train[col]), max(df_train[col])])



sns.distplot(log_col, ax=ax[1], color='r')

ax[1].set_title('Distribution of log of {}'.format(col), fontsize=14)

ax[1].set_xlim([min(log_col), max(log_col)])



fig.show()
var = 'age'

cat_column = 'target'



fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,5), )

sns.kdeplot(df_train[var][(df_train[cat_column] == 0) & (df_train[var].notnull())], 

            color="Red", shade = True, ax=ax)

sns.kdeplot(df_train[var][(df_train[cat_column] == 1) & (df_train[var].notnull())], 

            color="Blue", shade= True, ax=ax)

ax.set_xlabel("Age")

ax.set_ylabel("Frequency")

ax = ax.legend(["No Heart Disease","Heart Disease"])

fig.show()
df_train.age.describe()
tmp = df_train.copy()
tmp['age_cut'] = pd.cut(df_train.age,

                     bins=[20, 56, 81],

                     labels=["20-55", "55-80"])
var = 'chol'

cat_column = 'age_cut'



fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12,12), )





sns.kdeplot(df_train[var][(tmp[cat_column] == "20-55") & (df_train[var].notnull())], 

                color="Red", shade = True, ax=ax[0, 0])

sns.kdeplot(df_train[var][(tmp[cat_column] == "55-80") & (df_train[var].notnull())], 

                color="Blue", shade = True, ax=ax[0,0])

ax[0, 0].set_xlabel("Cholestrol")

ax[0, 0].set_ylabel("Frequency")

ax[0, 0] = ax[0, 0].legend(["Age Group 20-55","Age Group 55-80"])





var = 'trestbps'

cat_column = 'age_cut'



sns.kdeplot(df_train[var][(tmp[cat_column] == "20-55") & (df_train[var].notnull())], 

                color="Red", shade = True, ax=ax[0, 1])

sns.kdeplot(df_train[var][(tmp[cat_column] == "55-80") & (df_train[var].notnull())], 

                color="Blue", shade = True, ax=ax[0, 1])

ax[0, 1].set_xlabel("Blood Pressure")

ax[0, 1].set_ylabel("Frequency")

ax[0, 1] = ax[0, 1].legend(["Age Group 20-55","Age Group 55-80"])





var = 'thalach'

cat_column = 'age_cut'



sns.kdeplot(df_train[var][(tmp[cat_column] == "20-55") & (df_train[var].notnull())], 

                color="Red", shade = True, ax=ax[1,0])

sns.kdeplot(df_train[var][(tmp[cat_column] == "55-80") & (df_train[var].notnull())], 

                color="Blue", shade = True, ax=ax[1,0])

ax[1,0].set_xlabel("Maximum Heart Rate")

ax[1,0].set_ylabel("Frequency")

ax[1,0] = ax[1,0].legend(["Age Group 20-55","Age Group 55-80"])





var = 'oldpeak'

cat_column = 'age_cut'



sns.kdeplot(df_train[var][(tmp[cat_column] == "20-55") & (df_train[var].notnull())], 

                color="Red", shade = True, ax=ax[1,1])

sns.kdeplot(df_train[var][(tmp[cat_column] == "55-80") & (df_train[var].notnull())], 

                color="Blue", shade = True, ax=ax[1,1])

ax[1,1].set_xlabel("ST depression induced by exercise relative to rest")

ax[1,1].set_ylabel("Frequency")

ax[1,1] = ax[1,1].legend(["Age Group 20-55","Age Group 55-80"])





fig.show()
f, ax = plt.subplots(figsize=(6, 6))

sns.countplot(x="age_cut", hue="target", data=tmp)

ax.set_title('Age Group v/s Heart Disease')

ax.legend(["No heart disease","Have heart disease"])

f.show()
idx = ["Male", "Female"]

vals = df_train.sex.value_counts().values

fig, ax = plt.subplots()

explode = (0, 0.1)

ax.pie(vals, labels=idx, explode=explode, autopct='%1.1f%%')

ax.axis('equal')

plt.show()
tmp.groupby(by=['sex','age_cut', 'target']).size().unstack()
tmp.groupby(by=['sex','age_cut', 'target']).size().unstack().apply(lambda g: g / g.sum(), axis=1)
fig, ax = plt.subplots(figsize=(6, 6))

tmp.groupby(by=['sex','age_cut', 'target']).size().unstack().apply(lambda g: g / g.sum(), axis=1).plot.bar(ax=ax)

ax.set_title("Plot of Count of Sex and Age Group on Heart Disease")

ax.legend(["No heart disease","Have heart disease"])

ax.set_xticklabels(['Women between 20-55','Women between 55-80','Men between 20-55','Men between 55-80'])

fig.show()


cols = ['age', 'trestbps', 'chol', 'oldpeak', 'thalach']

fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 20))

fig.subplots_adjust(hspace=.5, wspace=.5)

fig.suptitle('Distributions of Numerical Feats with Target')



for ax, feature in zip(axes.flatten(), cols):

    sns.boxplot(x='target', y=feature, data=tmp, ax = ax)

    ax.set(title=feature.upper(), ylabel=feature)

    ax.set_ylim([min(tmp[feature]), max(tmp[feature])])



fig.show()
# Computing IQR

Q1 = tmp['chol'].quantile(0.25)

Q3 = tmp['chol'].quantile(0.75)

IQR = Q3 - Q1

print("IQR: ", IQR)



# Filtering Values between Q1-1.5IQR and Q3+1.5IQR

filtered = tmp.query('(@Q1 - 1.5 * @IQR) <= chol <= (@Q3 + 1.5 * @IQR)')



print("Skewness before removing outliers: ",tmp.chol.skew())

print("Skewness after removing outliers: ",filtered.chol.skew())



fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,6), )

sns.distplot(tmp.chol, fit=norm, color='r', ax=ax[0])

ax[0].set_title('Cholestrol', fontsize=14)

ax[0].set_xlim([min(tmp.chol), max(tmp.chol)])



sns.distplot(filtered.chol, fit=norm, color='r', ax=ax[1])

ax[1].set_title('Cholestrol - After outliers removed', fontsize=14)

ax[1].set_xlim([min(filtered.chol), max(filtered.chol)])

fig.show()
print("Number of rows before dropping cholestrol outliers: ", tmp.shape[0])

print("Number of rows after dropping cholestrol outliers: ", filtered.shape[0])

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

sns.boxplot(x='target', y='chol', data=tmp, ax = ax[0])

ax[0].set(title="Cholestrol", ylabel=feature)

ax[0].set_ylim([min(tmp['chol']), max(tmp['chol'])])



sns.boxplot(x='target', y='chol', data=filtered, ax = ax[1])

ax[1].set(title='Cholestrol - After outliers removed', ylabel=feature)

ax[1].set_ylim([min(filtered['chol']), max(filtered['chol'])])



fig.show()
cat_cols = [col for col in tmp.columns if col not in ['age', 'trestbps', 'chol', 'oldpeak', 

                                                      'thalach', 'target', 'age_cut']]

print(cat_cols)

cols = cat_cols

fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(20, 30))

fig.subplots_adjust(hspace=.5, wspace=.5)

fig.suptitle('Distributions of Categorical Feats with Target', fontsize=20)



for ax, feature in zip(axes.flatten(), cols):

    my_tab = pd.crosstab(index = tmp['target'], columns=tmp[feature])

    my_tab = my_tab/my_tab.sum()

    my_tab.plot.bar(ax=ax)

    ax.set(title=feature.upper(), ylabel="{} Count Percentage".format(feature.upper()), xlabel = "Target")

#     ax.legend(["No heart disease","Have heart disease"])



fig.show()

df_train["age_group"] = tmp["age_cut"]
df_train.columns
df_train.drop("trestbps", axis=1, inplace=True)
df_train.columns
print("Shape before dropping outliers in cholestrol: ", df_train.shape)



# Computing IQR

Q1 = tmp['chol'].quantile(0.25)

Q3 = tmp['chol'].quantile(0.75)

IQR = Q3 - Q1

print("IQR: ", IQR)



# Filtering Values between Q1-1.5IQR and Q3+1.5IQR

df_train = df_train.query('(@Q1 - 1.5 * @IQR) <= chol <= (@Q3 + 1.5 * @IQR)')



print("Shape after dropping outliers in cholestrol: ", df_train.shape)

df_train.head()
one_hot_df = pd.get_dummies(df_train[['sex', 'fbs', 'exang', 'age_group']] , prefix="onehot_")

df_train = pd.concat([df_train, one_hot_df], axis=1)
df_train.drop(['sex', 'fbs', 'exang', 'age_group'], axis=1, inplace=True)
df_train.head()
y = df_train.target.values

x_data = df_train.drop(['target'], axis = 1).values
from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(x_data, y,test_size = 0.2, random_state=0)
print(x_train.shape)

print(x_test.shape)
from sklearn.model_selection import GridSearchCV,train_test_split,cross_val_score

from sklearn.metrics import classification_report,confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import Imputer

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import roc_curve, auc
parameters=[

{

    'penalty':['l1','l2'],

    'C':[0.1,0.4,0.5],

    'random_state':[0]

    },

]



    

gslog=GridSearchCV(LogisticRegression(), parameters, scoring='accuracy')

gslog.fit(x_train, y_train)



print('Best parameters set:')

print(gslog.best_params_)

print()

predictions=[

(gslog.predict(x_train),y_train,'Train'),

(gslog.predict(x_test),y_test,'Test'),

]



for pred in predictions:

    print(pred[2] + ' Classification Report:')

    print("*"*50)

    print(classification_report(pred[1],pred[0]))

    print("*"*50)

    print(pred[2] + ' Confusion Matrix:')

    print(confusion_matrix(pred[1], pred[0]))

    print("*"*50)



print("*"*50)    

cross_val_step = cross_val_score(estimator=LogisticRegression(),X=x_data, y=y, cv=12)

print(cross_val_step.mean())

print(cross_val_step.std())

print("*"*50) 

def plot_roc_(false_positive_rate,true_positive_rate,roc_auc):

    plt.figure(figsize=(5,5))

    plt.title('Receiver Operating Characteristic')

    plt.plot(false_positive_rate,true_positive_rate, color='red',label = 'AUC = %0.2f' % roc_auc)

    plt.legend(loc = 'lower right')

    plt.plot([0, 1], [0, 1],linestyle='--')

    plt.axis('tight')

    plt.ylabel('True Positive Rate')

    plt.xlabel('False Positive Rate')

    plt.show()

    

def plot_feature_importances(gbm, X_train):

    n_features = X_train.shape[1]

    plt.barh(range(n_features), gbm.feature_importances_, align='center')

    plt.yticks(np.arange(n_features), X_train.columns)

    plt.xlabel("Feature importance")

    plt.ylabel("Feature")

    plt.ylim(-1, n_features)
lr=LogisticRegression(C=0.4,penalty='l1',random_state=0)

lr.fit(x_train, y_train)



y_pred=lr.predict(x_test)



y_proba=lr.predict_proba(x_test)



false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test,y_proba[:,1])

roc_auc = auc(false_positive_rate, true_positive_rate)

plot_roc_(false_positive_rate,true_positive_rate,roc_auc)





from sklearn.metrics import r2_score,accuracy_score



#print('Hata Oranı :',r2_score(y_test,y_pred))

print('Accurancy Oranı :',accuracy_score(y_test, y_pred))

print("Logistic TRAIN score with ",format(lr.score(x_train, y_train)))

print("Logistic TEST score with ",format(lr.score(x_test, y_test)))

print()



cm=confusion_matrix(y_test, y_pred)

print(cm)

sns.heatmap(cm,annot=True)

plt.show()
parameters = [

{

    'learning_rate': [0.01, 0.02, 0.002],

    'random_state': [0],

    'n_estimators': np.arange(3, 20)

    },

]



print("*"*50)



gbc = GridSearchCV(GradientBoostingClassifier(), parameters, scoring='accuracy')

gbc.fit(x_train, y_train)

print('Best parameters set:')

print(gbc.best_params_)

print("*"*50)

predictions = [

(gbc.predict(x_train), y_train, 'Train'),

(gbc.predict(x_test), y_test, 'Test1')

]

for pred in predictions:

    print(pred[2] + ' Classification Report:')

    print("*"*50)

    print(classification_report(pred[1], pred[0]))

    print("*"*50)

    print(pred[2] + ' Confusion Matrix:')

    print(confusion_matrix(pred[1], pred[0]))

    print("*"*50)



print("*"*50)    

cross_val_step =cross_val_score(estimator=GradientBoostingClassifier(),X=x_data, y=y ,cv=4)

print(cross_val_step.mean())

print(cross_val_step.std())

print("*"*50)
gbc=GradientBoostingClassifier(learning_rate=0.02, n_estimators=12, random_state=0)

gbc.fit(x_train, y_train)



y_pred=gbc.predict(x_test)



y_proba=gbc.predict_proba(x_test)

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test,y_proba[:,1])

roc_auc = auc(false_positive_rate, true_positive_rate)

plot_roc_(false_positive_rate,true_positive_rate,roc_auc)



from sklearn.metrics import r2_score,accuracy_score



print("GradientBoostingClassifier TRAIN score with ",format(gbc.score(x_train, y_train)))

print("GradientBoostingClassifier TEST score with ",format(gbc.score(x_test, y_test)))

print()



cm=confusion_matrix(y_test, y_pred)

print(cm)

sns.heatmap(cm,annot=True)

plt.show()
plot_feature_importances(gbc, df_train.drop(['target'], axis = 1))

plt.show()