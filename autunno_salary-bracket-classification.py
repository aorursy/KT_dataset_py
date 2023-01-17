import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
%matplotlib inline

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import model_selection

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from statsmodels.stats.outliers_influence import variance_inflation_factor   
from scipy.stats import friedmanchisquare
from scipy.stats import wilcoxon
pd.options.display.max_columns = None
warnings.filterwarnings('ignore')
dataset = pd.read_csv('../input/survey_results_public.csv')
dataset.head()
# number of entries
print("Number of entries: " + str(len(dataset.index)))
pd.options.display.max_rows = None
dataset.isnull().sum()
all_data_na = (dataset.isnull().sum() / len(dataset)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
missing_data.head(20)
features = dataset[dataset.columns.difference(['Respondent','TimeAfterBootcamp', 'MilitaryUS', 'HackatonReasons', 'EngonomicDevices', 'AdBlockReasons', 'StackOverflowJobRecommend', 'SurveyTooLong', 'SurveyTooEasy', 'AdBlocker', 'AdBlockerDisable', 'AdBlockerReasons', 'AdsAgreeDisagree1', 'AdsAgreeDisagree2', 'AdsAgreeDisagree3', 'AdsActions', 'AdsPriorities1', 'AdsPriorities2', 'AdsPriorities3', 'AdsPriorities4', 'AdsPriorities5', 'AdsPriorities6', 'AdsPriorities7', 'HypotheticalTools1', 'HypotheticalTools2', 'HypotheticalTools3', 'HypotheticalTools4', 'HypotheticalTools5', 'CurrencySymbol', 'Salary', 'SalaryType'])]
features = features[features.columns.difference(['ErgonomicDevices', 'HackathonReasons', 'SurveyEasy', 'JobEmailPriorities1', 'JobEmailPriorities2', 'JobEmailPriorities3', 'JobEmailPriorities4', 'JobEmailPriorities5', 'JobEmailPriorities6', 'JobEmailPriorities7', 'JobContactPriorities1', 'JobContactPriorities2', 'JobContactPriorities3', 'JobContactPriorities4', 'JobContactPriorities5', 'AIDangerous', 'AIInteresting', 'AIResponsible', 'AIFuture', 'IDE', 'LanguageDesireNextYear', 'LanguageWorkedWith', 'PlatformDesireNextYear', 'PlatformWorkedWith', 'DatabaseDesireNextYear', 'DatabaseWorkedWith', 'FrameworkDesireNextYear', 'FrameworkWorkedWith', 'CommunicationTools', 'CheckInCode', 'VersionControl', 'UpdateCV',  'StackOverflowVisit', 'StackOverflowDevStory', 'StackOverflowHasAccount', 'StackOverflowParticipate', 'StackOverflowRecommend', 'StackOverflowJobs', 'StackOverflowJobsRecommend'])]
#features = dataset[dataset.columns.difference(['UpdateCV',  'StackOverflowVisit', 'StackOverflowDevStory', 'StackOverflowHasAccount', 'StackOverflowParticipate', 'StackOverflowRecommend', 'StackOverflowJobs', 'StackOverflowJobsRecommend'])]
features.nunique()
features = features[features.columns.difference(['Methodology', 'SelfTaughtTypes', 'DevType', 'EducationTypes'])]
f, ax = plt.subplots(figsize=(30, 7))
plt.xticks(rotation='90')
sns.countplot(features['Country'])
plt.title('Country distribution on the Stackoverflow 2018 survey', fontsize=12)
f, ax = plt.subplots(figsize=(30, 7))
plt.xticks(rotation='45')
sns.countplot(features['Currency'])
plt.title('Currency  distribution on the Stackoverflow 2018 survey', fontsize=15)
features = features[(features.Country == 'United States')]
features = features[features.columns.difference(['Currency', 'Country'])];
features = features.dropna();
print("Number of entries: " + str(len(features.index)))
features.columns
f, ax = plt.subplots(figsize=(14, 7))
plt.xticks(rotation='45')
sns.distplot(features['ConvertedSalary']);
plt.xlabel('Annual salary in US dollars', fontsize=15)
features['ConvertedSalary'].describe()
features = features[features['ConvertedSalary'] < 300000]
features = features[features['ConvertedSalary'] > 15000]
features['ConvertedSalary'].describe()
f, ax = plt.subplots(figsize=(14, 7))
plt.xticks(rotation='45')
sns.distplot(features['ConvertedSalary']);
plt.xlabel('Annual salary in US dollars', fontsize=15)
first_bracket = "From 15k to 60k"
second_bracket =  "From 60k to 110k"
third_bracket = "From 110k to 300k"

features['SalaryRange'] = pd.cut(features['ConvertedSalary'], bins=[15000, 60000, 110000, 300000], labels=[first_bracket, second_bracket, third_bracket])
features = features[features.columns.difference(['ConvertedSalary'])];
f, ax = plt.subplots(figsize=(14, 7))
plt.xticks(rotation='45')
sns.countplot(features['SalaryRange'], order=[first_bracket, second_bracket, third_bracket])
plt.title('Salary distribution', fontsize=15)
features = features.dropna();
print("Number of entries: " + str(len(features.index)))
features['SalaryRange'].describe()
f, ax = plt.subplots(figsize=(14, 7))
plt.xticks(rotation='45')
sns.countplot(features['CareerSatisfaction'], order=['Extremely dissatisfied', 'Slightly dissatisfied', 'Moderately dissatisfied', 'Neither satisfied nor dissatisfied', 'Slightly satisfied', 'Moderately satisfied', 'Extremely satisfied'])
plt.xlabel('Career satisfaction', fontsize=15)
plt.title('How people feel about their career', fontsize=15)
features['CareerSatisfaction'].describe()
labelencoder = LabelEncoder()
features['carrer_label'] = labelencoder.fit_transform(features['CareerSatisfaction'])

f, ax = plt.subplots(figsize=(14, 7))
sns.boxplot(x="SalaryRange", y="carrer_label", data=features)
plt.xlabel('Salary bracket', fontsize=15)
plt.ylabel('Career satisfaction', fontsize=15)
plt.title('Career satisfaction across salary brackets', fontsize=15)

features = features[features.columns.difference(['carrer_label'])]
# Define a set of graphs, 3 by 5, usin the matplotlib library
f, axes = plt.subplots(4, 3, figsize=(25, 25), sharex=False, sharey=False)
#f.subplots_adjust(hspace=0.4)
plt.subplots_adjust(left=0.2, wspace=0.3, top=0.95)
plt.suptitle('Salary bracket versus importance given to job opportunities aspects', fontsize=16)
axes[-1, -1].axis('off')
axes[-1, -2].axis('off')

sns.pointplot(x="SalaryRange", y="AssessJob1", data=features, ax=axes[0,0])
axes[0,0].set(ylabel='Importance given to industry you will be working in')

sns.pointplot(x="SalaryRange", y="AssessJob2", data=features, ax=axes[0,1])
axes[0,1].set(ylabel='Importance given to financial performance of the company/organization')

sns.pointplot(x="SalaryRange", y="AssessJob3", data=features, ax=axes[0,2])
axes[0,2].set(ylabel='Importance given to deparment or team you will be working on')

sns.pointplot(x="SalaryRange", y="AssessJob4", data=features, ax=axes[1,0])
axes[1,0].set(ylabel='Importance given to technologies you will be working with')

sns.pointplot(x="SalaryRange", y="AssessJob5", data=features, ax=axes[1,1])
axes[1,1].set(ylabel='Importance given to compensation and benefits offered')

sns.pointplot(x="SalaryRange", y="AssessJob6", data=features, ax=axes[1,2])
axes[1,2].set(ylabel='Importance given to the company culture')

sns.pointplot(x="SalaryRange", y="AssessJob7", data=features, ax=axes[2,0])
axes[2,0].set(ylabel='Importance given to the opportunity to work from home/remotely')

sns.pointplot(x="SalaryRange", y="AssessJob8", data=features, ax=axes[2,1])
axes[2,1].set(ylabel='Importance given to opportunities for professioal development')

sns.pointplot(x="SalaryRange", y="AssessJob9", data=features, ax=axes[2,2])
axes[2,2].set(ylabel='Importance given to the diversity of the company or organization')

sns.pointplot(x="SalaryRange", y="AssessJob10", data=features, ax=axes[3,0])
axes[3,0].set(ylabel='Importance given to the impact of the product/software you will be working with has')
features = features[features.columns.difference(['AssessJob3', 'AssessJob5', 'AssessJob7'])]
# Define a set of graphs, 3 by 5, usin the matplotlib library
f, axes = plt.subplots(4, 3, figsize=(25, 25), sharex=False, sharey=False)
#f.subplots_adjust(hspace=0.4)
plt.subplots_adjust(left=0.2, wspace=0.3, top=0.95)
plt.suptitle('Salary brackets versus importance given to benefits', fontsize=16)
axes[-1, -1].axis('off')

sns.pointplot(x="SalaryRange", y="AssessBenefits1", data=features, ax=axes[0,0])
axes[0,0].set(ylabel='Importance given to salary/bonuses')

sns.pointplot(x="SalaryRange", y="AssessBenefits2", data=features, ax=axes[0,1])
axes[0,1].set(ylabel='Importance given to stock options/shares')

sns.pointplot(x="SalaryRange", y="AssessBenefits3", data=features, ax=axes[0,2])
axes[0,2].set(ylabel='Importance given to healthcare')

sns.pointplot(x="SalaryRange", y="AssessBenefits4", data=features, ax=axes[1,0])
axes[1,0].set(ylabel='Importance given to parental leave')

sns.pointplot(x="SalaryRange", y="AssessBenefits5", data=features, ax=axes[1,1])
axes[1,1].set(ylabel='Importance given to fitness or wellness benefits')

sns.pointplot(x="SalaryRange", y="AssessBenefits6", data=features, ax=axes[1,2])
axes[1,2].set(ylabel='Importance given to retirement or pension savings matching')

sns.pointplot(x="SalaryRange", y="AssessBenefits7", data=features, ax=axes[2,0])
axes[2,0].set(ylabel='Importance given to company provided meals or snacks')

sns.pointplot(x="SalaryRange", y="AssessBenefits8", data=features, ax=axes[2,1])
axes[2,1].set(ylabel='Importance given to computer/office equipment allowance')

sns.pointplot(x="SalaryRange", y="AssessBenefits9", data=features, ax=axes[2,2])
axes[2,2].set(ylabel='Importance given to childcare benefit')

sns.pointplot(x="SalaryRange", y="AssessBenefits10", data=features, ax=axes[3,0])
axes[3,0].set(ylabel='Importance given to transportation benefit')

sns.pointplot(x="SalaryRange", y="AssessBenefits11", data=features, ax=axes[3,1])
axes[3,1].set(ylabel='Importance given to conference or education budget')
features = features[features.columns.difference(['AssessBenefits3', 'AssessBenefits4', 'AssessBenefits6', 'AssessBenefits9', 'AssessBenefits10'])]
output = features['SalaryRange']
features = features[features.columns.difference(['SalaryRange'])]
categorical = []
for col, value in features.iteritems():
    if value.dtype == 'object':
        categorical.append(col)

# Store the numerical columns in a list numerical
numerical = features.columns.difference(categorical)

print(categorical)
# get the categorical dataframe
features_categorical = features[categorical]
# one hot encode it
features_categorical = pd.get_dummies(features_categorical, drop_first=True)
# get the numerical dataframe
features_numerical = features[numerical]
# concatenate the features
features = pd.concat([features_numerical, features_categorical], axis=1)
labelencoder = LabelEncoder()
output = labelencoder.fit_transform(output)
def calculate_vif_(X, thresh=100):
    cols = X.columns
    variables = np.arange(X.shape[1])
    dropped=True
    while dropped:
        dropped=False
        c = X[cols[variables]].values
        vif = [variance_inflation_factor(c, ix) for ix in np.arange(c.shape[1])]
    
        maxloc = vif.index(max(vif))
        if max(vif) > thresh:
            print('dropping \'' + X[cols[variables]].columns[maxloc] + '\' at index: ' + str(maxloc))
            variables = np.delete(variables, maxloc)
            dropped=True

    print('Remaining variables:')
    print(X.columns[variables])
    return X[cols[variables]]
features = features[features.columns.difference(['WakeTime_Between 6:01 - 7:00 AM','SexualOrientation_Straight or heterosexual','OperatingSystem_Windows'])]
features_train, features_test, salary_train, salary_test = train_test_split(features, output, test_size = 0.3, random_state = 0)
def plot_confusion_matrix(cm, title):
    # building a graph to show the confusion matrix results
    cm_plot = pd.DataFrame(cm, index = [i for i in {first_bracket, second_bracket, third_bracket}],
                  columns = [i for i in {first_bracket, second_bracket, third_bracket}])
    plt.figure(figsize = (6,5))
    sns.heatmap(cm_plot, annot=True, vmin=5, vmax=90.5, cbar=False, fmt='g')
# Fit the classifier, get the prediction array and print the accuracy
def fit_and_pred_classifier(classifier, X_train, X_test, y_train, y_test):
    # Fit the classifier to the training data
    classifier.fit(X_train, y_train)

    # Get the prediction array
    y_pred = classifier.predict(X_test)
    
    # Get the accuracy %
    print("Accuracy: " + str(accuracy_score(y_test, y_pred) * 100) + "%") 
    
    return y_pred
# Build and fit the model
rf = RandomForestClassifier(n_estimators = 800, random_state = 0)
rf_salary_pred = fit_and_pred_classifier(rf, features_train, features_test, salary_train, salary_test)
cm = confusion_matrix(salary_test, rf_salary_pred)
plot_confusion_matrix(cm, 'Random Forest Confusion Matrix')
# Build and fit the model
svc = SVC(kernel = 'linear', probability=True, random_state = 0)
svc_salary_pred = fit_and_pred_classifier(svc, features_train, features_test, salary_train, salary_test)
cm = confusion_matrix(salary_test, svc_salary_pred)
plot_confusion_matrix(cm, 'Linear SVC Classifier Confusion Matrix')
lda = LinearDiscriminantAnalysis()
lda_salary_pred = fit_and_pred_classifier(lda, features_train, features_test, salary_train, salary_test)
cm = confusion_matrix(salary_test, lda_salary_pred)
plot_confusion_matrix(cm, 'OneVsRest Linear SVC Confusion Matrix')
ensemble = VotingClassifier(estimators=[('rf', rf), ('svc', svc), ('lda', lda)], voting='soft', weights=[2,3,2], flatten_transform=True)
ensemble_salary_pred = fit_and_pred_classifier(ensemble, features_train, features_test, salary_train, salary_test)
cm = confusion_matrix(salary_test, ensemble_salary_pred)
plot_confusion_matrix(cm, 'Ensemble Confusion Matrix')
# Friedman test
stat, p = friedmanchisquare(svc_salary_pred, lda_salary_pred, ensemble_salary_pred)
print('p=%.10f' % (p))
# interpret
def h0_test(p):
    alpha = 0.05
    if p > alpha:
        print('Same distributions (fail to reject H0)')
    else:
        print('Different distributions (reject H0)')

h0_test(p)
from scipy.stats import f_oneway
# compare samples
stat, p = f_oneway(svc_salary_pred, lda_salary_pred, ensemble_salary_pred)
print('p=%.3f' % (p))

# interpret
h0_test(p)
# compare samples
stat, p = f_oneway(svc_salary_pred, lda_salary_pred, ensemble_salary_pred, rf_salary_pred)
print('p=%.3f' % (p))

# interpret
h0_test(p)
# evaluate each model in turn
models = [['Linear SVC', svc], ['Random Forest', rf], ['Linear Discriminant Analysis', lda],['Ensemble', ensemble]]
results = []
names = []
scoring = 'accuracy'
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=7)
    cv_results = model_selection.cross_val_score(model, features_train, salary_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    print("%s accuracy: %0.4f (+/- %0.4f)" % (name, cv_results.mean(), cv_results.std() * 2))
# boxplot algorithm comparison
f, ax = plt.subplots(figsize=(15, 7))
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()