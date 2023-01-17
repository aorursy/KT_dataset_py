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

df = pd.read_csv('../input/mental-health-unemployment/data.csv')
#renaming Columns
df.rename(columns={'I identify as having a mental illness':'MentallyIll'}, inplace =True)
df.rename(columns={'I am unemployed':'Unemployed'}, inplace =True)

# Show all column values
pd.set_option('display.max_columns', None)
#Delete Identifying columns and other unneeded info
cols = list(range(0,9))
df.drop(df.columns[cols], axis=1, inplace=True)
#Replace Yes and No's with 0s and 1s, replace NaNs
df = df.replace(to_replace=['No', 'Yes'], value=[0, 1])
df = df.fillna(0)
#remove first row of response
df = df.iloc[1:]
#df.head(5)
ill = pd.to_numeric(df.MentallyIll)
unempl = pd.to_numeric(df.Unemployed)
#renaming income columns
df.rename(columns={'Annual income (including any social welfare programs) in USD':'Income'}, inplace =True)
income = pd.to_numeric(df.Income)

#Income stats of all respondents
income.describe()
a4_dims = (20, 5.27)
fig, ax = plt.subplots(figsize=a4_dims)
sns.countplot(df.Education)
#sns.countplot(ill)
edu = df.Education.to_frame().to_records()
mental = df.MentallyIll.to_frame().to_records()
income = df.Income.to_frame().to_records()
unemployed = df.Unemployed.to_frame().to_records()
gender = df.Gender.to_frame().to_records()
age = df.Age.to_frame().to_records()
sixty, fifty, thirty, twenty = [], [], [], []
for i in age:
    if i[1] == '> 60':
        sixty.append((i[0], income[i[0]-1][1], unemployed[i[0]-1][1]))
    elif i[1] == '45-60':
        fifty.append((i[0], income[i[0]-1][1], unemployed[i[0]-1][1]))
    elif i[1] == '30-44':
        thirty.append((i[0], income[i[0]-1][1], unemployed[i[0]-1][1]))
    elif i[1] == '18-29':
        twenty.append((i[0], income[i[0]-1][1], unemployed[i[0]-1][1]))
        
age_list = [sixty, fifty, thirty, twenty]
title_age_list = ['Sixties', 'Fifties', 'Thirties', 'Twenties']
j = 0;
for i in age_list:
    #sort by income
    x_val = sorted([x[1] for x in i])
    unemp = [x[2] for x in i]

    #convert income to integers
    x_int = []
    for i in x_val:
        x_int.append(int(i))

    #Removing plots to reduce kernel size
    #a4_dims = (20, 4)
    #fig, ax = plt.subplots(figsize=a4_dims)
    #g = sns.scatterplot(x = x_int, y=unemp)
    #ax.set(xlabel='Income', ylabel='Unemployed')
    #plt.title(title_age_list[j])
    print(title_age_list[j], '-- Percent Unemployed:', sum(unemp)/len(unemp))
    j = j+1
#separate and regroup data based on education level
completed_masters, completed_undergrad, high_school, some_Phd, completed_Phd, some_highschool, some_undergrad, some_masters = [], [], [], [],[], [], [], [] 
for i in edu:
    if i[1] == 'Completed Masters':
        completed_masters.append((i[0], mental[i[0]-1][1], income[i[0]-1][1], unemployed[i[0]-1][1], gender[i[0]-1][1], age[i[0]-1][1]))
    elif i[1] == 'Completed Undergraduate':
        completed_undergrad.append((i[0], mental[i[0]-1][1], income[i[0]-1][1], unemployed[i[0]-1][1], gender[i[0]-1][1], age[i[0]-1][1]))
    elif i[1] == 'High School or GED':
        high_school.append((i[0], mental[i[0]-1][1], income[i[0]-1][1], unemployed[i[0]-1][1], gender[i[0]-1][1], age[i[0]-1][1]))
    elif i[1] == 'Some Phd':
        some_Phd.append((i[0], mental[i[0]-1][1], income[i[0]-1][1], unemployed[i[0]-1][1], gender[i[0]-1][1], age[i[0]-1][1]))
    elif i[1] == 'Completed Phd':
        completed_Phd.append((i[0], mental[i[0]-1][1], income[i[0]-1][1], unemployed[i[0]-1][1], gender[i[0]-1][1], age[i[0]-1][1]))
    elif i[1] == 'Some highschool':
        some_highschool.append((i[0], mental[i[0]-1][1], income[i[0]-1][1], unemployed[i[0]-1][1], gender[i[0]-1][1], age[i[0]-1][1]))
    elif i[1] == 'Some Undergraduate':
        some_undergrad.append((i[0], mental[i[0]-1][1], income[i[0]-1][1], unemployed[i[0]-1][1], gender[i[0]-1][1], age[i[0]-1][1]))
    else:
        some_masters.append((i[0], mental[i[0]-1][1], income[i[0]-1][1], unemployed[i[0]-1][1], gender[i[0]-1][1], age[i[0]-1][1]))
#Plot data, different graphs for different education levels
tot_list = [completed_masters, completed_undergrad, high_school, some_Phd, completed_Phd, some_highschool, some_undergrad, some_masters]
title_list = ['Completed Masters', 'Completed Undergrad', 'High School', 'Some PhD', 'Completed PhD', 'Some Highschool', 'Some Undergrad', 'Some Masters']
j = 0;
for i in tot_list:
    #sort by income
    x_val = sorted([x[2] for x in i])
    y_val = [x[1] for x in i]
    unemp = [x[3] for x in i]
    gend = [x[4] for x in i]
    age = [x[5] for x in i]

    #convert income to integers
    x_int = []
    for i in x_val:
        x_int.append(int(i))

    #convert binary employment to text labels
    for i in range(len(unemp)):
        if unemp[i] == 1:
            unemp[i] = 'Unemployed'
        elif unemp[i] == 0:
            unemp[i] = 'Employed'

    a4_dims = (20, 4)
    fig, ax = plt.subplots(figsize=a4_dims)
    g = sns.scatterplot(x = x_int, y=y_val, 
                        hue=unemp, hue_order = ('Employed', 'Unemployed'), 
                        style=gend, 
                        size=age, size_order = ('> 60', '45-60', '30-44','18-29'))
    ax.set(xlabel='Income', ylabel='Mentally Ill')
    plt.title(title_list[j])
    j = j+1
#use new dataframe for ML part
df2 = df
#remove last 3 irrelevant columns
df2 = df2.iloc[:, :-3]
#replace issues with binary representations
df2 = df2.replace(['Lack of concentration','Anxiety', 'Depression', 
                   'Obsessive thinking', 'Panic attacks', 'Compulsive behavior', 
                   'Mood swings', 'Tiredness'], 1)
#replace gender -> Male = 1, Female = 0
df2 = df2.replace(to_replace=['Female', 'Male'], value=[0, 1])
#replace age with median
df2 = df2.replace(to_replace=['> 60', '45-60', '30-44','18-29'], value=[65, 52, 37, 23])
#replace education with ranking, 0 is lowest educated 7 is highest
df2 = df2.replace(to_replace=['Completed Phd', 'Some Phd', 'Completed Masters','Some\xa0Masters', 
                              'Completed Undergraduate', 'Some Undergraduate', 'High School or GED', 'Some highschool']
                              , value=[7, 6, 5, 4, 3, 2, 1, 0])
#Fully cleaned data
df2.head(5)
#Covariance Matrix
corrmat = df2.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);
plt.show()
#Unemployed correlation matrix
k = 10 #number of variables for heatmap
cols = corrmat.nlargest(k, 'Unemployed')['Unemployed'].index
cm = np.corrcoef(df2[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()
#MentallyIll correlation matrix
k = 10 #number of variables for heatmap
cols = corrmat.nlargest(k, 'MentallyIll')['MentallyIll'].index
cm = np.corrcoef(df2[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()
#Begin Random Forest
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

#set target and drop from data
df3 = df2.copy()
df4 = df2.copy()
df5 = df2.copy()
target=df2['MentallyIll']
df2.drop(['MentallyIll'],axis=1,inplace=True)
X = df2
X_train, X_test, y_train, y_test = train_test_split(X,target, test_size=0.2, random_state=0)
forest = RandomForestClassifier(max_depth=10, random_state=0)
forest.fit(X_train, y_train)
print(forest.score(X_test,y_test))
#print(forest.score(X_train, y_train))
#Important Features
importances = pd.DataFrame({'feature':X_train.columns,'importance':np.round(forest.feature_importances_,3)})
importances = importances.sort_values('importance',ascending=False).set_index('feature')
importances.plot.bar()
#Confusion Matrix
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
predictions = cross_val_predict(forest, X_train, y_train, cv=3)
confusion_matrix(y_train, predictions)
#183 correctly classified as not mentally ill, 19 wrongly classified as not mentally ill
#26 people wrongly classified as mentally ill, 39 correctly classified as mentally ill
#Recall and Precision
from sklearn.metrics import precision_score, recall_score

print("Precision:", precision_score(y_train, predictions))
print("Recall:",recall_score(y_train, predictions))

from sklearn.metrics import f1_score
f1_score(y_train, predictions)
#Precision Recall Curve
from sklearn.metrics import precision_recall_curve

# getting the probabilities of our predictions
y_scores = forest.predict_proba(X_train)
y_scores = y_scores[:,1]

precision, recall, threshold = precision_recall_curve(y_train, y_scores)
def plot_precision_and_recall(precision, recall, threshold):
    plt.plot(threshold, precision[:-1], "orange", label="precision", linewidth=5)
    plt.plot(threshold, recall[:-1], "b", label="recall", linewidth=5)
    plt.xlabel("threshold", fontsize=19)
    plt.legend(loc="upper right", fontsize=19)
    plt.ylim([0, 1])

plt.figure(figsize=(14, 7))
plot_precision_and_recall(precision, recall, threshold)
plt.show()
#ROC AUC Curve
from sklearn.metrics import roc_curve
# compute true positive rate and false positive rate
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, y_scores)
# plotting them against each other
def plot_roc_curve(false_positive_rate, true_positive_rate, label=None):
    plt.plot(false_positive_rate, true_positive_rate, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'orange', linewidth=4)
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate (FPR)', fontsize=16)
    plt.ylabel('True Positive Rate (TPR)', fontsize=16)

plt.figure(figsize=(14, 7))
plot_roc_curve(false_positive_rate, true_positive_rate)
plt.show()
from sklearn.metrics import roc_auc_score
r_a_score = roc_auc_score(y_train, y_scores)
print("ROC-AUC-Score:", r_a_score)
#dropping said columns, leaving tiredness and lack of concentration
df3.drop(['I have been hospitalized before for my mental illness'],axis=1,inplace=True)
df3.drop(['How many days were you hospitalized for your mental illness'],axis=1,inplace=True)
df3.drop(['I am legally disabled'],axis=1,inplace=True)
df3.drop(['How many times were you hospitalized for your mental illness'],axis=1,inplace=True)
df3.drop([' depression'],axis=1,inplace=True)
df3.drop([' anxiety'],axis=1,inplace=True)
df3.drop([' Obsessive'],axis=1,inplace=True)
df3.drop([' panic attack'],axis=1,inplace=True)
df3.drop([' mood swings'],axis=1,inplace=True)
df3.drop([' compulsive'],axis=1,inplace=True)
#df3.head(5)
target=df3['MentallyIll']
df3.drop(['MentallyIll'],axis=1,inplace=True)
X = df3
X_train, X_test, y_train, y_test = train_test_split(X,target, test_size=0.2, random_state=0)
forest3 = RandomForestClassifier(max_depth=5, random_state=0)
forest3.fit(X_train, y_train)
print(forest3.score(X_test,y_test))
#print(forest.score(X_train, y_train))
#Important Features
importances = pd.DataFrame({'feature':X_train.columns,'importance':np.round(forest3.feature_importances_,3)})
importances = importances.sort_values('importance',ascending=False).set_index('feature')
importances.plot.bar()
#Confusion Matrix
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
predictions = cross_val_predict(forest3, X_train, y_train, cv=3)
confusion_matrix(y_train, predictions)
from sklearn.metrics import roc_auc_score
y_scores = forest3.predict_proba(X_train)
y_scores = y_scores[:,1]
r_a_score = roc_auc_score(y_train, y_scores)
print("ROC-AUC-Score:", r_a_score)
#Drop income and current employment features
df4.drop(['Income'],axis=1,inplace=True)
df4.drop(['I am currently employed at least part-time'],axis=1,inplace=True)
df4.drop(['Annual income from social welfare programs'],axis=1,inplace=True)
df4.drop(['I receive food stamps'],axis=1,inplace=True)
target=df4['Unemployed']
df4.drop(['Unemployed'],axis=1,inplace=True)
#df4.head(5)
X = df4
X_train, X_test, y_train, y_test = train_test_split(X,target, test_size=0.2, random_state=0)
forest4 = RandomForestClassifier(max_depth=15, random_state=0)
forest4.fit(X_train, y_train)
#Accuracy
print(forest4.score(X_test,y_test))
#Important Features
importances = pd.DataFrame({'feature':X_train.columns,'importance':np.round(forest4.feature_importances_,3)})
importances = importances.sort_values('importance',ascending=False).set_index('feature')
importances.plot.bar()
from sklearn.metrics import roc_auc_score
y_scores = forest4.predict_proba(X_train)
y_scores = y_scores[:,1]
r_a_score = roc_auc_score(y_train, y_scores)
print("ROC-AUC-Score:", r_a_score)
#Drop all non mental health features
df5.drop(['Income'],axis=1,inplace=True)
df5.drop(['I am currently employed at least part-time'],axis=1,inplace=True)
df5.drop(['Annual income from social welfare programs'],axis=1,inplace=True)
df5.drop(['I receive food stamps'],axis=1,inplace=True)
df5.drop(['Education'],axis=1,inplace=True)
df5.drop(['I have my own computer separate from a smart phone'],axis=1,inplace=True)
df5.drop(['I am legally disabled'],axis=1,inplace=True)
df5.drop(['I have my regular access to the internet'],axis=1,inplace=True)
df5.drop(['I live with my parents'],axis=1,inplace=True)
df5.drop(['I have a gap in my resume'],axis=1,inplace=True)
#df5.drop(['Total length of any gaps in my resume in months.'],axis=1,inplace=True)
df5.drop(df5.columns[3], axis=1, inplace=True)
df5.drop(['I read outside of work and school'],axis=1,inplace=True)
df5.drop(['I am on section 8 housing'],axis=1,inplace=True)
df5.drop(['Age'],axis=1,inplace=True)
df5.drop(['Gender'],axis=1,inplace=True)

target=df5['Unemployed']
df5.drop(['Unemployed'],axis=1,inplace=True)
#df5.head(5)
X = df5
X_train, X_test, y_train, y_test = train_test_split(X,target, test_size=0.2, random_state=0)
forest5 = RandomForestClassifier(max_depth=15, random_state=0)
forest5.fit(X_train, y_train)
#Accuracy
print(forest5.score(X_test,y_test))
#Important Features
importances = pd.DataFrame({'feature':X_train.columns,'importance':np.round(forest5.feature_importances_,3)})
importances = importances.sort_values('importance',ascending=False).set_index('feature')
importances.plot.bar()
from sklearn.metrics import roc_auc_score
y_scores = forest5.predict_proba(X_train)
y_scores = y_scores[:,1]
r_a_score = roc_auc_score(y_train, y_scores)
print("ROC-AUC-Score:", r_a_score)
# data to plot
n_groups = 4
accuracy = (0.82, 0.94, 0.82, 0.83)
auc_roc = (0.9899, 0.997, 0.98, 0.80)
 
# create plot
a4_dims = (20, 4)
fig, ax = plt.subplots(figsize=a4_dims)
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.8
 
rects1 = plt.bar(index, accuracy, bar_width,
alpha=opacity,
color='b',
label='Accuracy')
 
rects2 = plt.bar(index + bar_width, auc_roc, bar_width,
alpha=opacity,
color='orange',
label='ROC-AUC')
 
plt.xlabel('Model')
plt.ylabel('Scores')
plt.title('Scores by Model')
plt.xticks(index + bar_width, ('Mental Illness, no Indicators', 'Mental Illness, Indicators', 
                               'Unemployment, no Indicators', 'Unemployment based on Mental Health'))
plt.legend()
plt.figure(figsize=(30, 7))
plt.show()