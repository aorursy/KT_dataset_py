import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import scipy as sp

import warnings

warnings.filterwarnings('ignore')

plt.style.use('seaborn-deep')



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv('../input/pima-indians-diabetes-database/diabetes.csv')

original = df.copy()
print('Data has', df.shape[0], 'rows and', df.shape[1], 'columns')
df.info()
df.head()
df.describe()
import missingno as msna

msna.matrix(df)

plt.show()
df['Outcome'].value_counts(normalize = True)
plt.rcParams['figure.figsize'] = (18, 7)



def univariate_plot(x):

    plt.subplot(121)

    sns.distplot(x, color = 'seagreen')

    plt.title('Probability Distribution Function', fontsize = 15)

    plt.ylabel('Probability')

    

    n = len(x)

    a = np.sort(x)

    b = np.arange(1, 1 + n) / n

    plt.subplot(122)

    plt.plot(a, b, color = 'seagreen', marker = '.', linestyle = 'none')

    mean_x = np.mean(x)

    plt.axvline(mean_x, label = 'Mean', color = 'k')

    skew = '               Skew : ' + str(round(x.skew(), 2))

    plt.annotate(skew, xy = (mean_x, 0.5), fontsize = 16)

    plt.legend()

    plt.title('Empirical Cumulative Distribution Function', fontsize = 15)
univariate_plot(df['Age'])
univariate_plot(df['BMI'])
univariate_plot(df['Pregnancies'])
univariate_plot(df['Glucose'])
univariate_plot(df['BloodPressure'])
univariate_plot(df['SkinThickness'])
univariate_plot(df['Insulin'])
univariate_plot(df['DiabetesPedigreeFunction'])
df.loc[df['Pregnancies'] > 10, :]
df.loc[df['BMI'] == 0]
df.loc[df['Glucose'] == 0]
df.loc[df['BloodPressure'] == 0]
df.loc[df['SkinThickness'] == 0]
df.loc[df['Insulin'] == 0]
drop_index = df.loc[(df['BMI'] == 0) & (df['BloodPressure'] == 0) & (df['SkinThickness'] == 0) & (df['Insulin'] == 0), :].index

df.drop(drop_index, axis = 0, inplace = True)
for i in df.columns.tolist():

    print(i, '-', len(df.loc[df[i] == 0, :]))
df.sample(20)
df['Outcome'] = df['Outcome'].astype('str')
plt.rcParams['figure.figsize'] = (17, 6)



def plot_box(x):

    plt.subplot(121)

    sns.boxplot(y = x, x = 'Outcome', data = df)

    plt.title(x, fontsize = 16)

    

    plt.subplot(122)

    sns.violinplot(y = x, x = 'Outcome', data = df)

    plt.title(x, fontsize = 16)
plot_box('Age')
plot_box('Pregnancies')
plot_box('Insulin')
plot_box('BMI')
plot_box('BloodPressure')
plot_box('SkinThickness')
plot_box('DiabetesPedigreeFunction')
plot_box('Glucose')
df['BMI'] = np.where(df['BMI'] == 0, np.nan, df['BMI'])

df['Glucose'] = np.where(df['Glucose'] == 0, np.nan, df['Glucose'])

df['BloodPressure'] = np.where(df['BloodPressure'] == 0, np.nan, df['BloodPressure'])

df['SkinThickness'] = np.where(df['SkinThickness'] == 0, np.nan, df['SkinThickness'])
df['BMI'].fillna(27, inplace = True)

df['Glucose'].fillna(df['Glucose'].mean(), inplace = True)

df['BloodPressure'].fillna(df['BloodPressure'].mean(), inplace = True)

df['SkinThickness'].fillna(df['SkinThickness'].mean(), inplace = True)
df.describe()
df.loc[df['SkinThickness'] > 60]
df.drop(df.loc[df['SkinThickness'] > 60].index, axis = 0, inplace = True)
df.loc[df['BMI'] > 55]
df.drop(df.loc[df['BMI'] > 55].index, axis = 0, inplace = True)
df.describe()
for i in df.select_dtypes(['int64', 'float64']).columns.tolist():

    print(i, ':', df[i].skew())
def skew_visual():    

    plt.rcParams['figure.figsize'] = (20, 8)



    plt.subplot(241)

    sns.distplot(df['Pregnancies'], color = 'k')

    plt.title('PDF - Pregnancies')



    plt.subplot(242)

    sns.distplot(df['Glucose'], color = 'k')

    plt.title('PDF - Glucose')



    plt.subplot(243)

    sns.distplot(df['BloodPressure'], color = 'k')

    plt.title('PDF - BloodPressure')



    plt.subplot(244)

    sns.distplot(df['Insulin'], color = 'k')

    plt.title('PDF - Insulin')



    plt.subplot(245)

    sns.distplot(df['SkinThickness'], color = 'k')

    plt.title('PDF - SkinThickness')



    plt.subplot(246)

    sns.distplot(df['Age'], color = 'k')

    plt.title('PDF - Age')



    plt.subplot(247)

    sns.distplot(df['DiabetesPedigreeFunction'], color = 'k')

    plt.title('PDF - DiabetesPedigreeFunction')



    plt.subplot(248)

    sns.distplot(df['BMI'], color = 'k')

    plt.title('PDF - BMI')

    plt.tight_layout()
skew_visual()
for i in ['Pregnancies', 'Insulin', 'Age', 'DiabetesPedigreeFunction']:

    print(i, ':', np.sqrt(df[i]).skew())
for i in ['Pregnancies', 'Insulin', 'Age', 'DiabetesPedigreeFunction']:

    print(i, ':', np.log1p(df[i]).skew())
pd.DataFrame({'Feature' : ['Pregnancies', 'Insulin', 'Age', 'DiabetesPedigreeFunction'],

             'Actual' : [df[i].skew() for i in df[['Pregnancies', 'Insulin', 'Age', 'DiabetesPedigreeFunction']]],

             'Squared' : [np.sqrt(df[i]).skew() for i in df[['Pregnancies', 'Insulin', 'Age', 'DiabetesPedigreeFunction']]],

             'Cubed' : [(df[i] ** (1/3)).skew() for i in df[['Pregnancies', 'Insulin', 'Age', 'DiabetesPedigreeFunction']]],

             'Logged' : [np.log1p(df[i]).skew() for i in df[['Pregnancies', 'Insulin', 'Age', 'DiabetesPedigreeFunction']]]})
df_v1 = df.copy()
df['Pregnancies_trans'] = np.sqrt(df['Pregnancies'])

df['Insulin_trans'] = np.log1p(df['Insulin'])

df['Age_trans'] = np.log1p(df['Age'])

df['DiabetesPedigreeFunction_trans'] = df['DiabetesPedigreeFunction'] ** (1/3)
def skew_visual_trans():    

    plt.rcParams['figure.figsize'] = (20, 8)



    plt.subplot(241)

    sns.distplot(df['Pregnancies_trans'], color = 'k')

    plt.title('PDF - Pregnancies')



    plt.subplot(242)

    sns.distplot(df['Glucose'], color = 'k')

    plt.title('PDF - Glucose')



    plt.subplot(243)

    sns.distplot(df['BloodPressure'], color = 'k')

    plt.title('PDF - BloodPressure')



    plt.subplot(244)

    sns.distplot(df['Insulin_trans'], color = 'k')

    plt.title('PDF - Insulin')



    plt.subplot(245)

    sns.distplot(df['SkinThickness'], color = 'k')

    plt.title('PDF - SkinThickness')



    plt.subplot(246)

    sns.distplot(df['Age_trans'], color = 'k')

    plt.title('PDF - Age')



    plt.subplot(247)

    sns.distplot(df['DiabetesPedigreeFunction_trans'], color = 'k')

    plt.title('PDF - DiabetesPedigreeFunction')



    plt.subplot(248)

    sns.distplot(df['BMI'], color = 'k')

    plt.title('PDF - BMI')

    plt.tight_layout()
skew_visual_trans()
df['Pregnancies_bin'] = np.where(df['Pregnancies'] == 0, 0,

                                np.where((df['Pregnancies'] > 0) & (df['Pregnancies'] <= 5), 1,

                                        np.where((df['Pregnancies'] > 5) & (df['Pregnancies'] <= 10), 2, 3)))
df['Insulin_bin'] = np.where(df['Insulin'] == 0, 0,

                                np.where((df['Insulin'] > 0) & (df['Insulin'] <= 50), 1,

                                        np.where((df['Insulin'] > 50) & (df['Insulin'] <= 200), 2, 3)))
bins = np.arange(0, 100, 10)

names = [1, 2, 3, 4, 5, 6, 7, 8, 9]



df['Age_bin'] = pd.cut(df['Age'], bins = bins, labels = names)
df.sample(10)
df_v2 = df.copy()
df = df[['Glucose', 'BloodPressure', 'SkinThickness', 'BMI', 'DiabetesPedigreeFunction_trans', 'Age_bin', 'Insulin_bin', 'Pregnancies_bin', 'Outcome']]
df.sample(10)
df['Outcome'] = df['Outcome'].astype('int64')
df_scaled_mms = df.copy()
from sklearn.preprocessing import MinMaxScaler

mms = MinMaxScaler()
cols = df_scaled_mms.columns.tolist()



df_scaled_mms = pd.DataFrame(mms.fit_transform(df), columns = cols)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
df_scaled_sc = pd.DataFrame(sc.fit_transform(df), columns = cols)
df_scaled_mms.describe()
df_scaled_sc.describe()
plt.rcParams['figure.figsize'] = (10, 8)



sns.heatmap(df.corr() * 100, annot = True, cmap = 'coolwarm')

plt.title('Correlation - Before Scaling', fontsize = 16)

plt.show()
sns.heatmap(df_scaled_mms.corr() * 100, annot = True, cmap = 'plasma')

plt.title('Correlation - Normalized Data', fontsize = 16)

plt.show()
sns.heatmap(df_scaled_sc.corr() * 100, annot = True, cmap = 'Set1')

plt.title('Correlation - Standardized Data', fontsize = 16)

plt.show()
df_scaled_mms.head()
from sklearn.model_selection import cross_val_score, train_test_split, KFold

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, precision_score, recall_score
from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB
X = df_scaled_mms.drop(columns = ['Insulin_bin', 'Outcome'])

Y = df_scaled_mms['Outcome']
train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size = 0.25, stratify = Y, random_state = 42)
print('Train_X - ', train_x.shape)

print('Test_X - ', test_x.shape)

print('Train_Y - ', train_y.shape)

print('Test_Y - ', test_y.shape)
def model_build(x):

    model = x

    model.fit(train_x, train_y)

    pred = model.predict(test_x)

    print('Accuracy :', accuracy_score(test_y, pred))

    print('F1 :', f1_score(test_y, pred))

    kfold = KFold(n_splits = 5, random_state = 56)

    cv_res = cross_val_score(x, train_x, train_y, cv = kfold, scoring = 'accuracy')

    print('CV Accuracy Score - 5 Splits :', cv_res.mean())

    print('Precision :', precision_score(test_y, pred))

    print('Recall :', recall_score(test_y, pred))
model_build(LogisticRegression())
model_build(DecisionTreeClassifier())
model_build(RandomForestClassifier())
model_build(KNeighborsClassifier())
model_build(SVC())
model_build(GaussianNB())
rf = RandomForestClassifier()

rf.fit(train_x, train_y)

pred = rf.predict(test_x)
pd.DataFrame(confusion_matrix(test_y, pred), index = ["Actual 0's", "Actual 1's"], columns = ["Predicted 0's", "Predicted 1's"]).style.background_gradient(cmap = 'Set2')
print(rf.feature_importances_)
print(train_x.columns)
from sklearn.ensemble import VotingClassifier
estimators = [('DT', DecisionTreeClassifier()), ('RF', RandomForestClassifier())]

vc = VotingClassifier(estimators)

vc.fit(train_x, train_y)

pred = vc.predict(test_x)
print('Accuracy :', accuracy_score(test_y, pred))

print('Precision :', precision_score(test_y, pred))

print('Recall :', recall_score(test_y, pred))
pd.DataFrame(confusion_matrix(test_y, pred), index = ["Actual 0's", "Actual 1's"], columns = ["Predicted 0's", "Predicted 1's"]).style.background_gradient(cmap = 'Set2')
depth = []

estimators = []

recall = []



for i in np.arange(2, 10):

    for j in np.arange(100, 300, 10):

        rf = RandomForestClassifier(max_depth = i, n_estimators = j, n_jobs = -1)

        rf.fit(train_x, train_y)

        pred = rf.predict(test_x)

        r = recall_score(test_y, pred)

        depth.append(i)

        estimators.append(j)

        recall.append(r)
res = pd.DataFrame({'Depth' : depth,

                   'Estimators' : estimators,

                   'Recall Score' : recall})

res.sort_values(by = 'Recall Score', ascending = False).head()
rf = RandomForestClassifier(max_depth = 9, n_estimators = 120, n_jobs = -1)

rf.fit(train_x, train_y)

pred = rf.predict(test_x)
pd.DataFrame(confusion_matrix(test_y, pred), index = ["Actual 0's", "Actual 1's"], columns = ["Predicted 0's", "Predicted 1's"]).style.background_gradient(cmap = 'Set2')
prob = rf.predict_proba(test_x)[:, 1]



probabilities = pd.DataFrame({'Probability' : prob,

             'P(0.4)' : '',

             'P(0.5)' : '',

             'P(0.6)' : ''})
probabilities['P(0.4)'] = np.where(probabilities['Probability'] > 0.4, 1, 0)

probabilities['P(0.5)'] = np.where(probabilities['Probability'] > 0.5, 1, 0)

probabilities['P(0.6)'] = np.where(probabilities['Probability'] > 0.6, 1, 0)
probabilities.head()
pd.DataFrame(confusion_matrix(test_y, probabilities['P(0.4)']), index = ["Actual 0's", "Actual 1's"], columns = ["Predicted 0's", "Predicted 1's"]).style.background_gradient(cmap = 'Set2')
recall_score(test_y, probabilities['P(0.4)'])