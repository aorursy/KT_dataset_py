import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import warnings
import os, time
warnings.filterwarnings('ignore')
sns.set()

%matplotlib inline 
# useful so that we do not need to repeat the plt.show() command everytime we want a graph displayed

CV_Match = pd.read_excel("../input/data.xlsx", sheetname = "Match", index_col = 'Candidate_match') 
CV_Sourcing = pd.read_excel("../input/data.xlsx", sheetname = "Candidate", index_col = 'CV_ID')
# read in the data from a xlsx and define an index for each of the dataframes
print(len(CV_Match.columns),"columns in the Match table")
print(len(CV_Sourcing.columns),"columns in the Candidate table")
CV_Match["Status"].unique()
status_go_match = [
'A01 - Offer Preparation Started',
'A02 - Offer Sent to Candidate',
'A03 - Offer accepted',
'A03 - Offer declined',
'A03 - Process Cancelled',
'A04 - Hiring Request Started',
'A04a – Hiring Started / Collecting signatures',
'A04b – Hiring Started / Contract signed',
'A05 - Onboarding',
'A06 - Offboarding',
'B01 - Subcontracting Started',
'B02 - Subcontracting Signed',
'D01 – Resignation',
'T01- Terminated',
'Candidate validated']

CV_Match_go = CV_Match[CV_Match["Status"].isin(status_go_match)]

print("""CV_Match: Input data has {} lines, out of which we had {} GOs ({:.1f}%), Our target variable is highly skewed so when
computing the performance of our model we can't use accuracy as a measure of how good the model is. 
The reason why we don't use accuracy as a measure of performance here is because if our model predict y = 0 all the time 
it would still fare pretty decently: {:.1f} %
""".format(CV_Match.shape[0],CV_Match_go.shape[0],(CV_Match_go.shape[0]/CV_Match.shape[0])*100,(1-(CV_Match_go.shape[0]/CV_Match.shape[0]))*100))
'''
INDEX & MATCH EXPERIENCE AND FR LEVEL FROM THE CANDIDATE TABLE
'''

# build 2 series with the FR level and Exp levels we want to index into our Match dataframe
FR_level = CV_Sourcing["Language Level - French"][CV_Sourcing["Language Level - French"].notnull()]
Exp = CV_Sourcing["Experience"][CV_Sourcing["Experience"].notnull()]

# build an empty dataframe to join the 2 series
shell = pd.DataFrame(index = CV_Sourcing.index)
shell['FR_level'] = FR_level
shell['Experience'] = Exp
#shell.count()

#concatenate the two series into our Match table
CV_Match = pd.concat([CV_Match,shell], axis = 1, join_axes= [CV_Match.index])

#Now that we concatenated on the foreign key of the Match table let's set the primary key as the index 

CV_Match.set_index('Match_ID', inplace=True)
'''
ADD 2 NEW COMPUTED FEATURES: Response Time and Decision Time
'''

CV_Match['Response Time'] = CV_Match['4. Client initial validation date'] - CV_Match['4. CV sent to Client']
CV_Match['Response Time'] = CV_Match['Response Time'].dt.days

CV_Match['Decision Time'] = CV_Match['7. Client final approval date'] - CV_Match['4. Client initial validation date']
CV_Match['Decision Time'] = CV_Match['Decision Time'].dt.days

'''
MAP STATUS TO ZEROS AND ONES
YES GO --> 1 
NO  GO --> 0
'''

Status_map_1 = {
'Rejected':0,
'CV refused':0,
'A02 - Offer Sent to Candidate':1,
'Candidate refused':0,
'A03 - Offer declined':1,
'D01 – Resignation':1,
'A05 - Onboarding':1,
'Candidate dropped out':0,
'CV dropped out':0,
'T01- Terminated':1,
'A03 - Process Cancelled':1,
'Dropped out':0,
'Approved':0,
'CV sent to France':0,
'Matched':0,
'Candidate validated':1,
'A01 - Offer Preparation Started':1,
'A04b – Hiring Started / Contract signed':1,
'A03 - Offer accepted':1,
'CV approved':0,
'A04 - Hiring Request Started':1,
'Sent to Client':0
}


CV_Match['Match_Status_mapped'] = CV_Match['Status'].map(Status_map_1).astype(int)
'''
REMOVE ABSURD DATA
'''

drop_Response_Time = CV_Match.index[CV_Match['Response Time'] < 0].tolist()
drop_Decision_Time = CV_Match.index[CV_Match['Decision Time'] < 0].tolist()
'''
drop_Response_Time2 = CV_Match.index[CV_Match['Response Time'] > 300].tolist()
drop_Decision_Time2 = CV_Match.index[CV_Match['Decision Time'] > 300].tolist()
'''
# Append to list

c = drop_Response_Time + drop_Decision_Time# + drop_Response_Time2 + drop_Decision_Time2
e = drop_Response_Time# + drop_Response_Time2 
f = drop_Decision_Time# + drop_Decision_Time2

'''
Dropping the 98 missing values actually yields the same result on the prediction 
so we'll stick to replacing by the mean for now

Instead of replacing the 98 missing values by the mean of the column dropping the empty lines could be done by running:
CV_Match.drop(c, inplace = True)

'''
for i, item in enumerate(e):
    CV_Match.loc[item, "Response Time"] = CV_Match["Response Time"].mean()
for i, item in enumerate(f):
    CV_Match.loc[item, "Decision Time"] = CV_Match["Decision Time"].mean()

"""
DROP THE COLUMNS WE WONT USE IN OUR MODEL
"""
all_data = CV_Match[['Match_Status_mapped', 'FR_level', 'Experience', 'Response Time', 'Decision Time']] 
"""
DISPLAY THE MISSING DATA
"""
msno.matrix(all_data)
# Replace missing values with a 0
all_data['FR_level'] = all_data['FR_level'].fillna("0")

# drop data where the experience level of the candidate is missing
all_data = all_data[all_data['Experience'].notnull()]

all_data.skew()
print(all_data.describe())
# let's explore the relationships within our data by using scatter plots and frequency tables
plt.title('Response Time distribution against the hiring decision')
plt.scatter(all_data['Response Time'],all_data['Match_Status_mapped'],color='red')

plt.figure()
plt.title('Decision Time distribution against the hiring decision')
plt.scatter(all_data['Decision Time'],all_data['Match_Status_mapped'],color='blue')

"""
Getting rid of that outlier actually worsened the performance of the model so let's roll back and keep the outlier
all_data['Match_Status_mapped'][all_data['Match_Status_mapped']==1].groupby(all_data['Decision Time']).describe() 
lets get rid of that outlier at 150 as it might negatively impact our model's bility to generalize
all_data['Decision Time'][all_data['Match_Status_mapped']==1][all_data['Decision Time'][all_data['Match_Status_mapped']==1]>=150]
we found the outlier id = "10047" let's adjust our initial data munging operation 
to drop that line or replace it by some kind of proxy 

"""

table = pd.crosstab(all_data['Response Time'],all_data['Match_Status_mapped'])
table.columns = ['Candidate is a GO','Candidate is a NO GO']
table.head(10)

# Let's normalize the distributions to reduce the skew identified earlier

from scipy.stats import norm, skew
from scipy.special import boxcox1p

temp = CV_Match['Response Time'].dropna()+1
plt.title('Before the transformation')
sns.distplot(CV_Match['Response Time'].dropna(),fit = norm)

plt.figure()
plt.title('After the transformation')

sns.distplot(temp.apply(np.log),fit = norm)
#sns.distplot(boxcox1p(temp,0.2),fit = norm)
# Let's normalize the distribution
from scipy.stats import norm, skew

temp = CV_Match['Decision Time'].dropna()+1
plt.title('Before the transformation')

sns.distplot(CV_Match['Decision Time'].dropna(),fit = norm)

plt.figure()
plt.title('After the transformation')

sns.distplot(temp.apply(np.log),fit = norm)
#sns.distplot(boxcox1p(temp,0.2),fit = norm)
"""
LABEL ENCODE IN ORDER TO PREDICT ON CERTAIN CATEGORIES OF DATA
"""
# split the decision times into groups of 20 day periods
Decision_map = pd.concat(pd.Series(str(i+1), index=range(i*20,20+i*20)) for i in range(0, (np.ceil(all_data["Decision Time"].max()/20)).astype(int)))

all_data['Response_Time'] = all_data['Response Time'].map(Decision_map)
all_data['Decision_Time'] = all_data['Decision Time'].map(Decision_map)

# replace Decision time by the log of Decision time
temp = all_data['Decision_Time'].astype(float).dropna()
all_data['Decision_Time'] = temp.apply(np.log)

# replace Response time by the log of Response time
temp = all_data['Response_Time'].astype(float).dropna()
all_data['Response_Time'] = temp.apply(np.log)

#describe the data so that we get a sense of the distribution of the variable we just mapped
groupby = all_data['Decision_Time'].groupby(all_data['Decision_Time']).describe()
groupby['index'] = groupby.index.astype(int)
groupby = groupby.sort_values(by=['index'])
groupby.drop('index', axis=1, inplace=True)

# drop the original columns and rename the target/dependant variable
all_data.drop(['Response Time','Decision Time'], axis = 1, inplace = True)
all_data.rename(index=str, columns={"Match_Status_mapped": "y"}, inplace = True)

#display the groupby
groupby
"""
Map the sample data with this scikit-learn encoder which basically replaces the unique values by an int,
this operation does not create new columns

"""

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

le.fit(all_data['Experience'].unique())
all_data['Experience'] = le.transform(all_data['Experience'])

le.fit(all_data['FR_level'].unique())
all_data['FR_level'] = le.transform(all_data['FR_level'])
all_data['FR_level'] = all_data['FR_level'].fillna('0')
# remove missing values NaN from our columns

Response_Time_ID = []
Decision_Time_ID = []

Response_Time_ID = list(all_data["Response_Time"][all_data["Response_Time"].isnull()].index)
Decision_Time_ID = list(all_data["Decision_Time"][all_data["Decision_Time"].isnull()].index)
FR_level_ID = list(all_data["FR_level"][all_data["FR_level"].isnull()].index)

Response_Time_ID.extend(Decision_Time_ID) # adds Decision times elements at the end of the first list
Response_Time_ID.extend(FR_level_ID) # adds Decision times elements at the end of the first list

mylist = list(set(Response_Time_ID)) # convert to a set so that we remove the duplicates otherwise the drop lines method fails

all_data.drop(mylist, inplace = True)

"""
print(all_data["Response_Time"].isnull().any())
print(all_data["Decision_Time"].isnull().any())
print(all_data['FR_level'].isnull().any())
print(all_data['FR_level'].unique())
""";
#Keep a copy (without the dummy cariable so that we can graph the decision boundary later on

X_ = all_data.drop(['y'], axis = 1)
y_ = all_data['y']
"""
FURTHER ENCODE IN ORDER TO PREPARE FOR THE ONE HOT ENCODE
"""
Status_map_1 = {
0:0,
1:0,
2:0,
3:1,
4:1,
5:1,
6:1,
7:1
}

#don't skip values to map even if redundant otherwise the unmapped statuses will be mapped to NaN

Status_map_2 = {
0:0,
1:0,
2:1,
3:1

}

all_data['Experience'] = all_data['Experience'].map(Status_map_2)
all_data['FR_level'] = all_data['FR_level'].map(Status_map_1)

"""
print(all_data['Experience'].unique())
print(all_data['FR_level'].unique())
#print(all_data['Response_Time'].unique())
#print(all_data['Decision_Time'].unique())

print(all_data["Response_Time"].isnull().any())
print(all_data["Decision_Time"].isnull().any())
print(all_data["FR_level"].isnull().any())
print(all_data["Experience"].isnull().any())
""";
"""
One hot encode our features. 
This operation simply transforms our features into vectors of zeros and ones and creates new columns with those vectors

"""

def onehot(df, column):
    dummies = pd.get_dummies(df[column], prefix="_" + column)
    df = df.join(dummies, how='outer')
    df.drop([column], axis=1, inplace=True)
    return df

#col_list = all_data.columns.tolist()
col_list = ['FR_level', 'Experience']

for column in col_list:
    if column != "y":
        all_data = onehot(all_data, column)
print(all_data.head())
"""
Set the X and y matrices that will be fed to the model
"""
#all_data.rename(index=str, columns={"_Experience_1": "Experience",'_FR_level_1.0':'FR_level'}, inplace = True)

y = all_data['y']
X = all_data.drop(['y','_Experience_0','_FR_level_0'], axis = 1)

# adding an intercept negatively impacts our model so we'll skip this step
# manually add the intercept
#X['intercept'] = 1.0

"""
KEEP A DATASET FOR CROSSVALIDATION
"""

param = 100

X = X[:-param]
y = y[:-param]

X_CV = X[:param]
y_CV = y[:param]

"""
print(X_CV.index)
print(y_CV.index)
""";
sns.heatmap(all_data.corr(),cmap='coolwarm', ax=plt.subplots(figsize=(15,15))[1], annot = True, fmt = '.1f')
# Let's have a final look and feel for the data before building our model

from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FormatStrFormatter

for i, key in enumerate(all_data.columns):
    if key != 'y':
        fig, ax = plt.subplots(figsize=(20, 3))
        plt.title('Count of '+key)
        ax = sns.countplot(x=key, hue= 'y', data=all_data, palette='RdBu')
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, ["Candidate is a NO GO", "Candidate is a GO"])
        #ax.set_xticklabels(rotation=30)
        #plt.xticks(rotation=45)
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        #ax.set_ylabel('Example', fontsize=40)

import statsmodels.api as sm
sm.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)

logit_model=sm.Logit(y.astype(float),X.astype(float))
result=logit_model.fit()
print(result.summary())
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression # it is possible to fit a linear model to our classification problem but it is not recommended at all 
from sklearn.linear_model import LogisticRegression # scikit learn's log regression uses a certain amount of regularization so set C = 9e10 to suppress the applied regularization
from sklearn.svm import SVC # support vector machine model applied to regression
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

#randomly split our data into evenly distributed samples
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

"""
#We're not going to scale because it produces a worse classifier
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)
"""

#classifier = LogisticRegression(penalty='l1')
classifier = SVC(kernel ='linear', random_state=0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

print(y_pred.size,"""values were successfully predicted""")
from sklearn.metrics import confusion_matrix
import time
confusion_matrix = confusion_matrix(y_test, y_pred)
print("confusion_matrix:\n\n",confusion_matrix)

TP = confusion_matrix[0,0]
TN = confusion_matrix[1,1]
FN = confusion_matrix[1,0]
FP = confusion_matrix[0,1]
SUM_ = TP + TN + FN + FP

Accuracy = (TP+FP)/SUM_
Precision = TP/(TP+FP)
Recall = TP/(TP+FN)
F1_score = (2*Precision*Recall)/(Precision+Recall)

# Uncomment to print out the predictions
sub = pd.DataFrame() # cant concatenate on the index because the output from sklearn is a np array without an index
sub['TrueValue']=y_test
sub['LogPrediction']=y_pred
sub['EqualOrNot'] = np.where((sub['TrueValue'] == sub['LogPrediction']), True, False)

timer=time.strftime("%Y%m%d_%H%M%S") # create a variable with current date and time to name the output file

filename='Output_'+timer+'.xlsx'
print(sub.head())
writer = pd.ExcelWriter(filename, engine = 'openpyxl')
print(writer)
sub.to_excel(writer,'Log reg', header=True, index=True)
writer.save() # save the merged file into an excel 


print("""\nAccuracy:{:.2f}\nPrecision:{:.2f}\nRecall:{:.2f}\nF1_score:{:.2f}""".format(Accuracy,Precision,Recall,F1_score))
from matplotlib.colors import ListedColormap
import matplotlib.gridspec as gridspec

def get(classifier,X,y,Col1,Col2):
    fig = plt.figure(1)
    gridspec.GridSpec(4,4)
    plt.subplot2grid((4,4), (0,0), colspan=2, rowspan=2)
    plt.locator_params(axis='x', nbins=5)
    plt.locator_params(axis='y', nbins=5)
    X_tmp = X_[[Col1,Col2]].astype(np.float64)
    X_train, X_test, y_train, y_test = train_test_split(X_tmp, y_, test_size=0.3, random_state=0, stratify=y_)
    classifier.fit(X_train, y_train)
    X_set, y_set = X_train, y_train
    X1, X2 = np.meshgrid(np.arange(start = X_set.iloc[:, 0].min() - 1, stop = X_set.iloc[:, 0].max() + 1, step = 0.01),
                         np.arange(start = X_set.iloc[:, 1].min() - 1, stop = X_set.iloc[:, 1].max() + 1, step = 0.01))
    plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                 alpha = 0.75, cmap = ListedColormap(('red', 'green')))

    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())

    leg = ["Candidate is a NO GO", "Candidate is a GO"]

    for i, item in enumerate(np.unique(y_set)):
        plt.scatter(X_set.values[y_set == item, 0], X_set.values[y_set == item, 1], 
                    c = ListedColormap(('red', 'green'))(i), label = leg[i])
    title = str(classifier)
    index = title.find('(')
    title = title[:index]
    plt.title('Training set:'+str(title))
    plt.xlabel(Col1)
    plt.ylabel(Col2)
    plt.legend()
    
    # large subplot
    plt.subplot2grid((4,4), (0,2), colspan=2, rowspan=2)
    plt.locator_params(axis='x', nbins=5)
    plt.locator_params(axis='y', nbins=5)

    X_tmp = X_[[Col1,Col2]].astype(np.float64)
    X_train, X_test, y_train, y_test = train_test_split(X_tmp, y_, test_size=0.3, random_state=0)
    classifier.fit(X_test, y_test)
    y_pred = classifier.predict(X_test)
    X_set, y_set = X_test, y_test
    X1, X2 = np.meshgrid(np.arange(start = X_set.iloc[:, 0].min() - 1, stop = X_set.iloc[:, 0].max() + 1, step = 0.01),
                         np.arange(start = X_set.iloc[:, 1].min() - 1, stop = X_set.iloc[:, 1].max() + 1, step = 0.01))
    plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                 alpha = 0.75, cmap = ListedColormap(('red', 'green')))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    
    leg = ["Candidate is a NO GO", "Candidate is a GO"]

    for i, item in enumerate(np.unique(y_set)):
        plt.scatter(X_set.values[y_set == item, 0], X_set.values[y_set == item, 1], 
                    c = ListedColormap(('red', 'green'))(i), label = leg[i])
        
    title = str(classifier)
    index = title.find('(')
    title = title[:index]
    plt.title('Test set:'+str(title))
    plt.xlabel(Col1)
    plt.ylabel(Col2)
    plt.legend()
    
    # fit subplots and save fig
    fig.tight_layout()
    fig.set_size_inches(w=11,h=7)
    #fig_name = str(title)+'.png'
    #fig.savefig(fig_name)
get(SVC(kernel ='linear', random_state=0),X_,y_,'FR_level','Decision_Time')
get(LogisticRegression(random_state=0),X_,y_,'FR_level','Decision_Time')
get(RandomForestClassifier(criterion = 'entropy', n_estimators=10, random_state=0),X_,y_,'FR_level','Decision_Time')
get(DecisionTreeClassifier(criterion = 'entropy'),X_,y_,'FR_level','Decision_Time')
get(GaussianNB(),X_,y_,'FR_level','Decision_Time')
get(KNeighborsClassifier(n_neighbors = 5),X_,y_,'FR_level','Decision_Time')
"""
get(RandomForestClassifier(criterion = 'entropy', n_estimators=10, random_state=0),X_,y_,'Experience','Decision_Time')
get(DecisionTreeClassifier(criterion = 'entropy'),X_,y_,'Experience','Decision_Time')
get(GaussianNB(),X_,y_,'Experience','Decision_Time')
get(LogisticRegression(),X_,y_,'Experience','Decision_Time')
get(KNeighborsClassifier(n_neighbors = 5),X_,y_,'Experience','Decision_Time')
"""
get(SVC(kernel ='linear', random_state=0),X_,y_,'Experience','Decision_Time')

class_list = [RandomForestClassifier(criterion = 'entropy', n_estimators=10, random_state=0),
             DecisionTreeClassifier(criterion = 'entropy'),GaussianNB(),SVC(kernel ='linear', random_state=0),
             KNeighborsClassifier(n_neighbors = 5),LogisticRegression(random_state=0)]

class_dic = {}

def f_score(classifier):
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_CV)
    from sklearn.metrics import confusion_matrix
    confusion_matrix = confusion_matrix(y_CV, y_pred)
    TP = confusion_matrix[0,0]
    TN = confusion_matrix[1,1]
    FN = confusion_matrix[1,0]
    FP = confusion_matrix[0,1]
    SUM_ = TP + TN + FN + FP
    Accuracy = (TP+FP)/SUM_
    Precision = TP/(TP+FP)
    Recall = TP/(TP+FN)
    F1_score = (2*Precision*Recall)/(Precision+Recall)
    #print(F1_score)
    title = str(classifier)
    index = title.find('(')
    title = title[:index]
    class_dic[title]=F1_score

for i in class_list:
    f_score(i)
print(class_dic)
best = 0
best_class = ''

for k, value in class_dic.items():
    if value >= best:
        best_class = k
        best = value

        
print("\nThe best classifier is:",best_class,best)
"""
from sklearn.grid_search import GridSearchCV

list_test = np.arange(start = 0.01, stop = 10, step = 0.1).tolist()
penalty = ['l1', 'l2']

param_grid = {'C':list_test,'penalty':penalty}
grid = GridSearchCV(LogisticRegression(),param_grid, scoring='precision')
grid.fit(X,y)
print(grid.best_score_)
print(grid.best_params_)
#print(grid.best_estimator_)
#print(grid.grid_scores_)
""";
"""
from sklearn.grid_search import GridSearchCV
from sklearn import svm, grid_search

Cs = np.arange(start = 0.01, stop = 10, step = 0.1).tolist()
gammas = np.arange(start = 0.01, stop = 10, step = 0.1).tolist()
#nfolds = np.arange(start = 1, stop = 10, step = 1).tolist()

param_grid = {'C': Cs, 'gamma' : gammas}
grid = GridSearchCV(svm.SVC(kernel='rbf'), param_grid, cv=10)
grid.fit(X,y)
print(grid.best_score_)
print(grid.best_params_)
#print(grid.best_estimator_)
#print(grid.grid_scores_)
""";