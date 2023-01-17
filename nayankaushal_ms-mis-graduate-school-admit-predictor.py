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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from sklearn.preprocessing import StandardScaler, LabelEncoder
data = pd.read_csv("/kaggle/input/graduate-mis-student-data-yocketapp/Yocket-dataset.csv", na_values=['NA ', 'N.A. '], error_bad_lines = False)
data.head()
data.columns = data.columns.str.lower()
data.drop(data.columns[data.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)

data.head()
data.describe()
data[data.university=='University']
data.drop(data[data.university=='University'].index, axis=0, inplace=True)

data = data.reset_index(drop=True)
data = data.drop(columns='name')
data['university'].unique()
data[data.university.isin(['University of Washingto', 'Texas A&M; University, College Statio', 'Illinois Institute of Technolog'])]
data.loc[data.university.isin(['University of Washingto', 'Texas A&M; University, College Statio', 'Illinois Institute of Technolog']), ['university','course']] = [('Illinois Institute of Technology','Management Information System'), ('Texas A&M; University, College Station','Management Information System'), ('University of Washington','Management Information System')]
data['university'].unique()
data['course'].unique()
data = data.drop(columns='course')
data = data.drop(columns='year')
sns.heatmap(data.isna())
print("Percent of empty gre_scores = {} %".format(len(data[data.gre_score.isna()]["gre_score"])/len(data["gre_score"])*100))

print("Percent of empty test_score = {} %".format(len(data[data.test_score.isna()]["test_score"])/len(data["test_score"])*100))

print("Percent of empty work_ex = {} %".format(len(data[data.work_ex.isna()]["work_ex"])/len(data["work_ex"])*100))
data[data["gre_score"].isna()][["gre_score","test_score"]].isnull().count()
data = data.dropna()

data.head()
sns.heatmap(data.isna())
data.head()
data.dtypes
data[['gre_score', 'test_score']] = data[['gre_score', 'test_score']].apply(pd.to_numeric)
data.work_ex = data.work_ex.str.strip(to_strip='months ').apply(pd.to_numeric)
for i in data.loc[:, "undergrad_score"].index:

    if "CGPA" in data.loc[i, "undergrad_score"]:

        data.loc[i, "undergrad_score"] = pd.to_numeric(data.loc[i, "undergrad_score"].strip("CGPA "))*9.5

    elif "%" in data.loc[i, "undergrad_score"]:        

        data.loc[i, "undergrad_score"] = data.loc[i, "undergrad_score"].strip("%")

data.undergrad_score = pd.to_numeric(data.undergrad_score)
data.loc[data["eng_test"]=="IELTS", "test_score"] = data.loc[data["eng_test"]=="IELTS", "test_score"].replace({9.0:120, 8.5:117, 8.0:114, 7.5:109, 7.0:101, 6.5:93, 6.0:78, 5.5:59, 5.0:45, 4.5:34, 4.0:31})

data = data.drop(columns="eng_test")
data.head()
data.dtypes
data.describe()
data["status"].unique()
data["status"] = pd.to_numeric(data["status"].replace({"Reject": "0", "Admit": "1", "Applied": "2", "Interested": "3"}))
data.loc[data['work_ex']<0, 'work_ex'] = 0
data.loc[data['work_ex']<0, 'work_ex']
binInterval = [0, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120, 132, 144]

binLabels   = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

data['work_ex_yrs'] = pd.to_numeric(pd.cut(data['work_ex'], bins = binInterval, labels=binLabels, include_lowest = True))
data = data.drop(columns='work_ex').reset_index(drop=True)

data.head()
data.groupby('university')["status"].unique()
data = data[data['university']!='University of Iowa']
data = data[data['university']!='University of California, Los Angeles']
data = data.reset_index(drop=True)
############################################################################################################################
sns.distplot(data[data['status']==0]['test_score'], color = 'r')

sns.distplot(data[data['status']==1]['test_score'], color = 'g')

plt.legend(['Reject','Admit'])

plt.xlabel('Test Score')

plt.title('English Proficiency Test Score Distribution')
display('Test Scores with a value of zero: ',data[data['test_score']==0].shape[0])

sns.set_style('whitegrid')

sns.lmplot(x='gre_score', y='test_score', data=data).set(xlabel="GRE Score", ylabel = "Test Score", title = "GRE Score vs Test Score")
data = data.drop(index = data[data['test_score']==0].index).reset_index(drop=True)
sns.distplot(data[data['status']==0]['test_score'], color = 'r')

sns.distplot(data[data['status']==1]['test_score'], color = 'g')

plt.legend(['Reject','Admit'])

plt.xlabel('Test Score')

plt.title('English Proficiency Test Score Distribution')
sns.distplot(data[data['status']==0]['gre_score'], color = 'r')

sns.distplot(data[data['status']==1]['gre_score'], color = 'g')

plt.legend(['Reject','Admit'])

plt.xlabel('GRE Score')

plt.title('GRE Score Distribution')
sns.distplot(data[data['status']==0]['work_ex_yrs'], color = 'r')

sns.distplot(data[data['status']==1]['work_ex_yrs'], color = 'g')

plt.legend(['Reject','Admit'])

plt.xlabel('Work Experience (in Years)')

plt.title('Work Experience (in Years) Distribution')
sns.distplot(data[data['status']==0]['undergrad_score'], color = 'r')

sns.distplot(data[data['status']==1]['undergrad_score'], color = 'g')

plt.legend(['Reject','Admit'])

plt.xlabel('Undergrad Score')

plt.title('Undergrad Scores Distribution')
univ = data[data['status']==1].groupby('university').mean()['gre_score'].sort_values(ascending=False).reset_index()

univ
bins = 6

binLength = (univ.describe().loc['max'][0] - univ.describe().loc['min'][0]) / bins

binLength
list(range(int(univ.describe().loc['min'][0]), int(univ.describe().loc['max'][0]), int(binLength)))
binInterval = list(range(int(univ.describe().loc['min'][0]), int(univ.describe().loc['max'][0]), int(binLength)))

binLabels   = list(range(1,len(binInterval)))

univ['rating'] = pd.cut(univ['gre_score'], bins = binInterval, labels=binLabels, right=True)
univ=univ[['university','rating']]
univ.loc[:,'rating']= pd.to_numeric(univ.loc[:,'rating']).fillna(len(binLabels)+1)
temp=[]

for i in data['university'].index:

    for j in univ['university'].index:

        if data.iloc[i,0]==univ.iloc[j,0]:

            temp.append(univ.iloc[j,1])
data['rating'] = pd.Series(temp)
test = data[data["status"].isin([2,3])]

univ_backup = test['university']

test = test.drop(columns=["status","university"])

test
train = data[data["status"].isin([0,1])].drop(columns=["university"])

train
y = train["status"]
X = train.drop(columns='status')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)

print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)
#Logistic Regression:

LR = LogisticRegression(random_state = 0, solver = 'lbfgs', multi_class='ovr', max_iter = 10000).fit(X_train, y_train)

y_pred = LR.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)*100

display("Accuracy of Logistic Regression = {:.2f}%".format(accuracy))

print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test,y_pred))
sns.distplot(y_test,label = 'Actual')

sns.distplot(y_pred, label ='Predicted')

plt.legend(loc="upper left")

plt.xlabel('Prediction level')
display('Accuracy with the training model: ', accuracy)

display('Accuracy on setting status to Admit: ',train['status'].value_counts(normalize=True)[1]*100)
importance = LR.coef_[0]

feature_importance = pd.DataFrame(sorted(zip(importance, X.columns)), columns=['Value','Feature'])

sns.barplot(x="Feature", y="Value", data=feature_importance.sort_values(by="Value", ascending=False))

plt.xticks(rotation=90)
test["status"] = LR.predict(test)

test['status'] = test['status'].replace({0:'Reject', 1:'Admit'})

test['university'] = univ_backup

test = test[['university', 'gre_score', 'test_score', 'undergrad_score', 'work_ex_yrs', 'status']]

test.to_csv("output.csv")

test