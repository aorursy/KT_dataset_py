import pandas as pd
diabetes = pd.read_csv('../input/pima-indians-diabetes-database//diabetes.csv')
diabetes.info()
diabetes.head()
import seaborn as sns
%matplotlib inline

sns.countplot(x='Outcome', data=diabetes, palette='hls')
diabetes.groupby('Outcome').mean()
import numpy as np
from sklearn import linear_model, datasets, metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
diabetes.isnull().sum()
sns.boxplot(x='Outcome', y='Glucose', data=diabetes, palette='hls')
sns.heatmap(diabetes.corr())
X_train, X_test, y_train, y_test = train_test_split(diabetes.drop('Outcome', 1), diabetes['Outcome'], test_size = .3, random_state=25)
LogReg = LogisticRegression()
LogReg.fit(X_train, y_train)
y_pred_quant = LogReg.predict_proba(X_test)[:, 1] #Only keep the first column, which is the 'pos' values
y_pred_bin = LogReg.predict(X_test)
from sklearn.metrics import confusion_matrix
confusion_matrix = metrics.confusion_matrix(y_test, y_pred_bin)
confusion_matrix
from sklearn.metrics import classification_report

total=sum(sum(confusion_matrix))

sensitivity = confusion_matrix[0,0]/(confusion_matrix[0,0]+confusion_matrix[0,1])
print('Sensitivity : ', sensitivity )

specificity = confusion_matrix[1,1]/(confusion_matrix[1,0]+confusion_matrix[1,1])
print('Specificity : ', specificity)
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_quant)
plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.rcParams['font.size'] = 12
plt.title('ROC curve for diabetes classifier')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)
metrics.auc(fpr, tpr)
liver = pd.read_csv('../input/indian-liver-patient-records/indian_liver_patient.csv')
liver.info()
liver.head()
liver.rename(columns = {'Dataset':'Outcome'}, inplace = True)
liver['Outcome'].replace(to_replace=2, value=0, inplace=True)
liver.groupby('Outcome').mean()
liver_sub = liver.iloc[:,[2,10]]
liver_sub.head()
sns.boxplot(x='Outcome', y='Total_Bilirubin', data=liver_sub, palette='hls')
sns.boxplot(x='Outcome', y='Total_Bilirubin', data=liver_sub, palette='hls', showfliers=False)
bins = np.linspace(0, 80, 100)

plt.hist(liver_sub['Total_Bilirubin'][liver_sub['Outcome'] == 0], bins, label='No disease',fc=(0, 0, 1, 1))
plt.hist(liver_sub['Total_Bilirubin'][liver_sub['Outcome'] == 1], bins, label='Disease',fc=(1, 0, 0, 0.4))
plt.legend(loc='upper right')
plt.show()
tb = liver_sub['Total_Bilirubin']
pred = np.zeros(tb.size)
tb_min = np.min(tb)
tb_max = np.max(tb)
tb_range = tb_max - tb_min
thresh_inc_size = 100
tb_inc = tb_range / thresh_inc_size
no_thresholds = tb_range / tb_inc.size
no_thresholds = np.ceil(no_thresholds)
no_thresholds = no_thresholds.astype(int)
sens = pd.Series(np.zeros(no_thresholds))
spec = pd.Series(np.zeros(no_thresholds))

tb_cutoff = tb_min
i=0
y=0
while (y <= thresh_inc_size):
    while (i<tb.size):
        if (tb[i] >= tb_cutoff):
            pred[i] = 1
        else:
            pred[i] = 0
        i=i+1
    confusion_matrix = metrics.confusion_matrix(liver_sub['Outcome'], pred)
    sens[y] = confusion_matrix[0,0]/(confusion_matrix[0,0]+confusion_matrix[0,1])
    spec[y] = confusion_matrix[1,1]/(confusion_matrix[1,0]+confusion_matrix[1,1])
    i=0    
    y=y+1
    tb_cutoff = tb_cutoff + tb_inc
fig, ax = plt.subplots()
ax.plot(1-spec, sens)
ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls="--", c=".3")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.rcParams['font.size'] = 12
plt.title('ROC curve for liver disease classifier')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
bcancer = pd.read_csv('../input/breast-cancer-wisconsin-data/data.csv')
bcancer.info()
bcancer.head()
bcancer.groupby('diagnosis').mean()
bcancer_sub = bcancer.iloc[:,[1,5,8]] #These features were found to be useful in other Kaggle kernels
bcancer_sub.head()
X_train, X_test, y_train, y_test = train_test_split(bcancer_sub.drop('diagnosis', 1), bcancer_sub['diagnosis'], test_size = .3, random_state=25)
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

np.random.seed(seed=100)
dtree = DecisionTreeClassifier(criterion = "gini", random_state = 10, max_depth=3, min_samples_leaf=3)
dtree.fit(X_train, y_train)
y_pred_quant = dtree.predict_proba(X_test)[:, 1] #Only keep the first column, which is the 'pos' values
y_pred_bin = dtree.predict(X_test)

confusion_matrix = metrics.confusion_matrix(y_test, y_pred_bin)
confusion_matrix
total=sum(sum(confusion_matrix))

sensitivity = confusion_matrix[0,0]/(confusion_matrix[0,0]+confusion_matrix[0,1])
print('Sensitivity : ', sensitivity )

specificity = confusion_matrix[1,1]/(confusion_matrix[1,0]+confusion_matrix[1,1])
print('Specificity : ', specificity)

fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_quant, pos_label='M')

plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.rcParams['font.size'] = 12
plt.title('ROC curve for Breast Cancer classifier')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)
#plt.axhline(y=sensitivity, color='r')
#plt.axvline(x=1-specificity, color='r')
metrics.auc(fpr, tpr)
from sklearn.utils import resample

specificity_orig = specificity

# configure bootstrap
n_iterations = 1000
n_size = int(len(bcancer_sub) * 0.25)

# run bootstrap
sens = list()
spec = list()

for i in range(n_iterations):
    # prepare train and test sets
    X_train, X_test, y_train, y_test = train_test_split(bcancer_sub.drop('diagnosis', 1), bcancer_sub['diagnosis'], test_size = .3)
    # fit model
    dtree = DecisionTreeClassifier(criterion = "gini", random_state = 10, max_depth=3, min_samples_leaf=3)
    dtree.fit(X_train, y_train)
    # evaluate model
    y_pred_quant = dtree.predict_proba(X_test)[:, 1] #Only keep the first column, which is the 'pos' values
    y_pred_bin = dtree.predict(X_test)
    confusion_matrix = metrics.confusion_matrix(y_test, y_pred_bin)
    sensitivity = confusion_matrix[0,0]/(confusion_matrix[0,0]+confusion_matrix[0,1])
    specificity = confusion_matrix[1,1]/(confusion_matrix[1,0]+confusion_matrix[1,1])
    sens.append(sensitivity)
    spec.append(specificity)
    
# plot scores
plt.hist(spec)
plt.axvline(x=specificity_orig, color='r')
plt.show()

# confidence intervals
alpha = 0.95
p = ((1.0-alpha)/2.0) * 100
lower = max(0.0, np.percentile(spec, p))
p = (alpha+((1.0-alpha)/2.0)) * 100
upper = min(1.0, np.percentile(spec, p))
print('%.1f confidence interval %.1f%% and %.1f%%' % (alpha*100, lower*100, upper*100))
def diagnostic_posterior(prior,sens,spec):
    lr_pos = sens / (1 - spec) #Positive likelihood ratio
    lr_neg = (1 - sens) / spec #Negative likelihood ratio
    
    pre_odds = prior / (1-prior) #Prior odds
    post_odds_pos = pre_odds * lr_pos #Positive posterior odds
    post_odds_neg = pre_odds * lr_neg #Negative posterior odds
    post_pos = post_odds_pos / (1+post_odds_pos) #Positive posterior probability
    post_neg = post_odds_neg / (1+post_odds_neg) #Negative posterior probability
    
    return(post_pos, post_neg)
lr_pos = 0.88 / (1 - 0.93) #Positive likelihood ratio
lr_neg = (1 - 0.88) / 0.93 #Negative likelihood ratio
print("The positive and negative likelihood ratios are %s, and %s, respectively" % (round(lr_pos,2), round(lr_neg,2)))
health = pd.read_csv('../input/key-indicators-of-annual-health-survey/Key_indicator_statewise.csv')
health.info()
health_sub = health.loc[:,['State_Name', 'UU_Children_Suffering_From_Acute_Respiratory_Infection_Total', 'UU_Children_Suffering_From_Acute_Respiratory_Infection_Rural', 'UU_Children_Suffering_From_Acute_Respiratory_Infection_Urban']]
health_sub
health_sub.loc[health_sub['UU_Children_Suffering_From_Acute_Respiratory_Infection_Total'].idxmin()]
health_sub.loc[health_sub['UU_Children_Suffering_From_Acute_Respiratory_Infection_Total'].idxmax()]
global_health = pd.read_csv('../input/health-nutrition-and-population-statistics/data.csv')
global_health.head()
global_health_sub = global_health[global_health['Indicator Name'].str.contains('Prevalence of HIV')]
global_health_sub = global_health_sub.loc[:,('Country Name', 'Indicator Name', '2005')]
global_health_sub = global_health_sub.dropna(subset = ['2005'])
global_health_sub.sort_values(['2005']).head()
global_health_sub.sort_values(['2005']).tail()
global_health_sub[global_health_sub['Country Name'] == 'Swaziland']
global_health_sub[global_health_sub['Country Name'] == 'Belarus']
print("A 20 year old man from Swaziland probability is {:0.2f}".format(diagnostic_posterior(0.181,0.99,0.99)[0]))
print("A 20 year old woman from Swaziland probability is {:0.2f}".format(diagnostic_posterior(0.067,0.99,0.99)[0]))
print("A 20 year old man from Belarus probability is {:0.2f}".format(diagnostic_posterior(0.001,0.99,0.99)[0]))
print("A 20 year old woman from Belarus probability is {:0.2f}".format(diagnostic_posterior(0.001,0.99,0.99)[0]))
post = list()
prev_options = np.arange(0.0, 1.0, 0.01)

for i in prev_options:
    post_temp = diagnostic_posterior(i,0.99,0.99)[0]
    post.append(post_temp)
fig, ax = plt.subplots()
ax.plot(prev_options, post)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.rcParams['font.size'] = 12
plt.title('Prevalence vs Posterior Probability (sens/spec = 0.99)')
plt.xlabel('Prevalence')
plt.ylabel('Posterior Probability')
plt.axvline(x=0.181, color='r')
plt.axvline(x=0.001, color='r')
post = list()
prev_options = np.arange(0.0, 1.0, 0.01)

for i in prev_options:
    post_temp = diagnostic_posterior(i,0.80,0.80)[0]
    post.append(post_temp)
    
fig, ax = plt.subplots()
ax.plot(prev_options, post)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.rcParams['font.size'] = 12
plt.title('Prevalence vs Posterior Probability (sens/spec = 0.80)')
plt.xlabel('Prevalence')
plt.ylabel('Posterior Probability')
diagnostic_posterior(0.01,0.92,0.85)[0]