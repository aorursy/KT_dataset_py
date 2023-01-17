import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Some mandatory Libraries

import string 

import warnings

import numpy as np

import pandas as pd



# plotting

import seaborn as sns;

import matplotlib.pyplot as plt



# features selection

from sklearn.feature_selection import chi2

from sklearn.feature_selection import SelectKBest



# scaling

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import StandardScaler



# model building

from sklearn.svm import SVC

from sklearn.svm import NuSVC

from sklearn.svm import LinearSVC

from sklearn.pipeline import make_pipeline

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split



# sccuracy

from sklearn.metrics import accuracy_score

from sklearn.model_selection import cross_val_score

from sklearn.metrics import classification_report, confusion_matrix, plot_roc_curve, plot_precision_recall_curve



# others

%matplotlib inline

warnings.filterwarnings("ignore")
df = pd.read_csv('../input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv')
df.head()
print('Shape of our Data:',df.shape)
# check datatypes



print(df.dtypes)
# check min, max and other details



df.describe()
# check the missing or null values.



print(df.isnull().sum())
print('DEATH_EVENT:')

print(df['DEATH_EVENT'].value_counts())
print('Distribution of DEATH_EVENT:')

print(df['DEATH_EVENT'].value_counts()/len(df))
ax = sns.countplot(x='DEATH_EVENT', data=df, facecolor=(0, 0, 0, 0), linewidth=5, edgecolor=sns.color_palette("dark", 3))
# define correlation matrice

corr = df.corr()



# Generate a mask for the upper triangle

mask = np.zeros_like(corr)

mask[np.triu_indices_from(mask)] = True



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



with sns.axes_style("white"):

    # Set up the matplotlib figure

    f, ax = plt.subplots(figsize=(10, 8))

    ax = sns.heatmap(corr, cmap=cmap, mask=mask, vmax=.3, square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)
#apply SelectKBest class to extract best features

X_train = df.drop(['DEATH_EVENT'], axis=1)

Y_test = df['DEATH_EVENT']

bestfeatures = SelectKBest(score_func=chi2, k=10)

fit = bestfeatures.fit(X_train, Y_test)

dfscores = pd.DataFrame(fit.scores_)

dfcolumns = pd.DataFrame(X_train.columns)



#concat two dataframes for better visualization 

featureScores = pd.concat([dfcolumns,dfscores],axis=1)

featureScores.columns = ['Featue','Score']

feature_imp = featureScores.nlargest(X_train.shape[1],'Score')
# plot top 5 features



print(feature_imp.head())
# plot each feature with it's importance



ax = sns.barplot(x='Score', y='Featue', data=feature_imp)
sns.pairplot(df, hue="DEATH_EVENT", palette="husl",diag_kind="kde")

plt.show()
for column in df.columns[:12]:

    sns.barplot(x='DEATH_EVENT',y=column, data=df, palette='Blues_d')

    plt.title('Death Event Vs. {}'.format(string.capwords(column.replace("_", " "))))

    plt.show()
# define two new dataframe for Survived & Non Servived



survived = df[df['DEATH_EVENT'] == 0]

not_survived = df[df['DEATH_EVENT'] == 1]
counts, bin_edges = np.histogram(survived['time'], bins=10, density = True)

pdf = counts/(sum(counts))

cdf = np.cumsum(pdf)

plt.plot(bin_edges[1:],pdf);

plt.plot(bin_edges[1:], cdf)



counts, bin_edges = np.histogram(not_survived['time'], bins=10, density = True)

pdf = counts/(sum(counts))

cdf = np.cumsum(pdf)

plt.plot(bin_edges[1:],pdf);

plt.plot(bin_edges[1:], cdf)



plt.xlabel('Time')

plt.title('PDF & CDF (Time)')

plt.legend(['PDF of Survived','CDF of Survived','PDF of Non-Survived','CDF of Non-Survived'])

plt.show()
counts, bin_edges = np.histogram(survived['serum_creatinine'], bins=10, density = True)

pdf = counts/(sum(counts))

cdf = np.cumsum(pdf)

plt.plot(bin_edges[1:],pdf);

plt.plot(bin_edges[1:], cdf)



counts, bin_edges = np.histogram(not_survived['serum_creatinine'], bins=10, density = True)

pdf = counts/(sum(counts))

cdf = np.cumsum(pdf)

plt.plot(bin_edges[1:],pdf);

plt.plot(bin_edges[1:], cdf)



plt.xlabel('Serum Creatinine')

plt.title('PDF & CDF (Serum Creatinine)')

plt.legend(['PDF of Survived','CDF of Survived','PDF of Non-Survived','CDF of Non-Survived'])

plt.show()
for column in df.columns[:12]:

    sns.boxplot(x='DEATH_EVENT',y=column, data=df, palette='Set3')

    plt.title('Death Event Vs. {}'.format(string.capwords(column.replace("_", " "))))

    plt.show()
# define features need to be scale

# select all numeric features except categorial

cols = ['age', 'creatinine_phosphokinase', 'ejection_fraction', 'platelets', 'serum_sodium', 'time']



# define object

scaler = MinMaxScaler()



# perform Min Max Scaling

for col in cols:

    scaler.fit(df[col].values.reshape(-1, 1))

    df['nrm_' + col] = scaler.transform(df[col].values.reshape(-1, 1))



# drop old columns

df.drop(['age', 'creatinine_phosphokinase', 'ejection_fraction', 'platelets', 'serum_sodium', 'time'], axis = 1, inplace=True)
X_train, X_test, y_train, y_test = train_test_split(df.drop(['DEATH_EVENT'], axis=1), df['DEATH_EVENT'], test_size=0.3, random_state=11)
# Define classifiers with default parameters.



classifiers = {

    'SVC': SVC(),

    'LinearSVC': LinearSVC(),

    'NuSVC': NuSVC()

}
for name, classifier in classifiers.items():

    classifier.fit(X_train, y_train) 

    training_score = cross_val_score(classifier, X_train, y_train, cv=3)

    print('Classifiers: ',name, 'has training score of', round(training_score.mean(),2) * 100)
# SVC



params = {

    'C':[10**-3, 10**-2, 10**-1, 1, 10, 10**2, 10**3], 

    'kernel':['linear', 'poly', 'rbf', 'sigmoid'],

    'gamma': ['scale', 'auto']

}



gs = GridSearchCV(SVC(), params, cv = 3, n_jobs=-1, scoring='accuracy')

gs_results = gs.fit(X_train, y_train)



SVC_best_estimator = gs.best_estimator_ # store best estimators for future analysis



print('Best Accuracy: ', gs_results.best_score_)

print('Best Parametrs: ', gs_results.best_params_)
# LinearSVC



params = {

    'C':[10**-3, 10**-2, 10**-1, 1, 10, 10**2, 10**3], 

    'penalty':['l1', 'l2'],

    'loss': ['hinge', 'squared_hinge']

}



gs = GridSearchCV(LinearSVC(), params, cv = 3, n_jobs=-1, scoring='accuracy')

gs_results = gs.fit(X_train, y_train)



LinearSVC_best_estimator = gs.best_estimator_ # store best estimators for future analysis



print('Best Accuracy: ', gs_results.best_score_)

print('Best Parametrs: ', gs_results.best_params_)
train_pred = SVC_best_estimator.predict(X_train)

print(classification_report(y_train,train_pred))
train_pred = LinearSVC_best_estimator.predict(X_train)

print(classification_report(y_train,train_pred))
print('FInal Test Accuracy:',LinearSVC_best_estimator.score(X_test,y_test))
plot_roc_curve(LinearSVC_best_estimator, X_test, y_test)

plt.show()
plot_precision_recall_curve(LinearSVC_best_estimator, X_test, y_test)

plt.show()
pred = LinearSVC_best_estimator.predict(X_test)

sns.heatmap(confusion_matrix(y_test,pred),annot=True)

plt.ylabel("Actual")

plt.xlabel("Prediction")