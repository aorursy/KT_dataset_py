# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#Providing a passingmark criteria which will be used to categorize the students

passmarks = 40 

#Reading the Data from a local repo in the system

df = pd.read_csv("../input/StudentsPerformance.csv")
# Let's Get some max,min,standard deviation for the Data Frame

df.describe()
#Also let's check for any missing values if in Data Set

df.isnull().sum()

# We find that no such missing values are there which will not be the case everytime. 

#But for Now since there are None....Let's continue with it.
# Let Us Explore Math Score at First Instance# Let U 

p = sns.countplot(x="math score" , data=df , palette = "muted")

_ = plt.setp(p.get_xticklabels(),rotation = 90)
# Let's find the number of students Passed and failed according to the passing Score:

df['MathPassingStatus'] = np.where(df['math score'] < passmarks , 'Failed!' , 'Passed!')

df.MathPassingStatus.value_counts()
#Let's Plot a Graph for Passed Students:

p = sns.countplot(x='parental level of education' , data = df , hue = 'MathPassingStatus' , palette = 'bright')

_ = plt.setp(p.get_xticklabels(), rotation = 90)

#Here we plot the graph in context to the Parental level of Education and depending upon that, showing the Number of Students passed or failed.
#Now exploring the Writing Score:

p= sns.countplot(x = "writing score" , data = df , palette = "muted")

_ = plt.setp(p.get_xticklabels(),rotation = 90)
#Here We are Analyzing on the Attribute of Lunch:#Here We 

p = sns.countplot(x='lunch' , data =df , hue = 'MathPassingStatus' , palette = 'bright')

_ = plt.setp(p.get_xticklabels(),rotation = 90)
#Similarly going for race/ethnicity:

p = sns.countplot(x='race/ethnicity' , data = df , hue = 'MathPassingStatus' , palette = 'bright')

_ = plt.setp(p.get_xticklabels(), rotation = 90)
# Now students passing the Writing Exam:

df['WritingPassingStatus'] = np.where(df['writing score']<passmarks , 'Failed!','Passed!')

df.WritingPassingStatus.value_counts()
#Plot for the Passed or failed, and seeing the Variation w.r.t Parental Level of Education:

p = sns.countplot(x='parental level of education' , data = df, hue = 'WritingPassingStatus', palette = 'bright')

_ = plt.setp(p.get_xticklabels(),rotation =90)
#Now exploring the Writing Score:

p= sns.countplot(x = "writing score" , data = df , palette = "muted")

_ = plt.setp(p.get_xticklabels(),rotation = 65)
#Here We are Analyzing on the Attribute of Lunch:

p = sns.countplot(x='lunch' , data =df , hue = 'WritingPassingStatus' , palette = 'bright')

_ = plt.setp(p.get_xticklabels(),rotation = 65)
#Similarly going for race/ethnicity:#Similar 

p = sns.countplot(x='race/ethnicity' , data = df , hue = 'WritingPassingStatus' , palette = 'bright')

_ = plt.setp(p.get_xticklabels(), rotation = 60)
# Similarly for the Reading Score:

p=sns.countplot(x="reading score" , data =df,palette = "muted")

plt.show()
# Number of Students Passed??

df['ReadingPassStatus'] = np.where(df['reading score'] < passmarks , 'Failed!' , 'Passed!')

df.ReadingPassStatus.value_counts()
#Finding % of Marks:

df['Total_Marks'] = df['math score'] + df['reading score'] + df['writing score']

df['Percent'] = df['Total_Marks']/3
#Let us Check how many Students totally passed in All Subjects:

df['OverAllPassingStatus'] = np.where(df.Total_Marks < 215 , 'Failed' , 'Passed!')

df.OverAllPassingStatus.value_counts()
p =  sns.countplot(x="Percent" , data = df , palette = "muted")

_ = plt.setp(p.get_xticklabels(),rotation = 0)
#Let us do the grading for the students now:

def GetGrade(Percent,OverAllPassingStatus):

    if(OverAllPassingStatus == 'Failed!'):

        return 'Failed'

    if(Percent >= 80):

        return 'A'

    if(Percent >= 70):

        return 'B'

    if(Percent >= 60):

        return 'C'

    if(Percent >= 50):

        return 'D'

    if(Percent >= 40):

        return 'E'

    else:

        return 'Failed!'
df['Grade'] = df.apply(lambda x: GetGrade(x['Percent'], x['OverAllPassingStatus']),axis =1)

df.Grade.value_counts()
#Plotting Grades in an Obtained Order

sns.countplot(x="Grade" , data=df,order = ['A','B','C','D','E','F'] , palette = "muted")

plt.show()
#Plotting with variation of Perental Education:

p = sns.countplot(x='parental level of education', data=df,hue='Grade',palette = 'bright')

_ = plt.setp(p.get_xticklabels(),rotation = 30)
#Lunch Variation

p = sns.countplot(x='lunch', data=df,hue='Grade',palette = 'bright')

_ = plt.setp(p.get_xticklabels(),rotation = 30)
#Test Prep Course Variation

p = sns.countplot(x='test preparation course', data=df,hue='Grade',palette = 'bright')

_ = plt.setp(p.get_xticklabels(),rotation = 30)
#Race/Ethnicity Variation

p = sns.countplot(x='race/ethnicity', data=df,hue='Grade',palette = 'bright')

_ = plt.setp(p.get_xticklabels(),rotation = 30)
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier



#Getting Data Points for All Scores availbale

ML_DataPoints = pd.read_csv(filepath_or_buffer = "../input/StudentsPerformance.csv",header = 0,

                           usecols = ['math score','reading score','writing score'])

#Getting Test Prep Course Values

ML_Labels = pd.read_csv(filepath_or_buffer = "../input/StudentsPerformance.csv",header = 0,usecols=['test preparation course'])



#Load MinMaxScaler

MNScaler = MinMaxScaler()

MNScaler.fit(ML_DataPoints) #Fitting the Scores

T_DataPoints = MNScaler.transform(ML_DataPoints) #Transform the Scores





#Load Label Encoder#Load La 

LEncoder = LabelEncoder()

LEncoder.fit(ML_Labels)

T_Labels = LEncoder.transform(ML_Labels)



#Split the DATA SET

XTrain,XTest,YTrain,YTest = train_test_split(T_DataPoints,T_Labels,random_state=10)



#Apply Random Forest Classifier:

RandomForest = RandomForestClassifier(n_estimators = 10,random_state=5)



RandomForest.fit(XTrain,YTrain)
RandomForest.fit(XTest,YTest)
RandomForest.score(XTrain,YTrain)
RandomForest.score(XTest,YTest)

#We see the Model is Underfitting..!!
model_now = LogisticRegression()

model_now.fit(XTrain,YTrain)
y_pred = model_now.predict(XTest)
from sklearn.metrics import accuracy_score

ac = accuracy_score(YTest,y_pred)

print(ac)
from sklearn.svm import SVC
model = SVC()

model.fit(XTrain,YTrain)
model.score(XTrain,YTrain)
model.score(XTest,YTest)

#We found Almost Same Accuracy..!!
model_tree = DecisionTreeClassifier()

model_tree.fit(XTrain,YTrain)
model_tree.score(XTrain,YTrain)
model_tree.score(XTest,YTest)
df.head()
df.nunique()
# Since we need to create dummy variables and from my inference, we don't need MathPassingStatus/ReadingPassingStatus and WritingPassingStatus, so we will drop these.

# Next we will dummy code the variables -> gender/lunch/test preparation course.

# Next we will Label Encode the variables -> race/ethinicity / parental level of education / Grade

# The we will standardize the remaining Numerical Variables to bring them onto one scale for modelling.

marks_df = df.drop(['MathPassingStatus','WritingPassingStatus','ReadingPassStatus'],1)

marks_df.head()
# Ok, as mentioned we have dropped the non-required columns, now let's perform some visualization for the dataframe

sns.pairplot(marks_df);
# So we see a Linear Relationship between the variables, let's plot their correlation

sns.heatmap(marks_df.corr(),annot=True);

plt.title('Correaltion for Marks Data Frame');
marks_df = marks_df.drop('Total_Marks',axis=1)

# Dropping highly correlated variables
# we see that Total Marks has high correlation for all the individual subjects and obviously, it is because it has been derived from the sum of all subjects, so we will keep it for modelling.

# Next we will convert our categorical variables to Numerical Variables.

dummy_df = pd.get_dummies(marks_df[['gender','test preparation course','lunch']],drop_first=True)

marks_df = pd.concat([marks_df,dummy_df],axis = 1)

marks_df.head(50)
marks_df.info()

# So now, we will drop the coulmns from which we have got dummies as they are now insignificant
marks_df = marks_df.drop(['gender','lunch','test preparation course'],axis=1)

marks_df.head()
# Next we go onto Label Enocding for the Variables -> race/ethinicity , parental level of education , Grade and Overall Passing Status

marks_df['race/ethnicity'] = marks_df['race/ethnicity'].astype('category')

marks_df['parental level of education'] = marks_df['parental level of education'].astype('category')

marks_df['Grade'] = marks_df['Grade'].astype('category')

marks_df['OverAllPassingStatus'] = marks_df['OverAllPassingStatus'].astype('category')

marks_df.info()
#Group A->0,Group B->1,#Group C->2,Group D->3,Group E->4,

marks_df['race/ethnicity'] = marks_df['race/ethnicity'].cat.codes

#associate's degree -> 0 , bachelor's degree -> 1, high school -> 2, master's degree - >3 , some college ->4 , some high school ->5

marks_df['parental level of education'] = marks_df['parental level of education'].cat.codes

# A->0 , B->1,C->2,D->3,E->4,Failed ->5

marks_df['Grade'] = marks_df['Grade'].cat.codes

marks_df.head()
marks_df.OverAllPassingStatus = marks_df.OverAllPassingStatus.cat.codes

# Passed->1,Failed - >0

marks_df.head()
sns.heatmap(marks_df.corr(),annot=True)
# Dropping negatively high correlated variables!

marks_df = marks_df.drop(['Grade','Percent'],axis=1)

marks_df.head()
# Now we will Standardize the untouches variables, which are: match score/writing score/readin score/Total_Marks/Percent

# But after splitting the data!

y = marks_df.OverAllPassingStatus

X = marks_df.drop('OverAllPassingStatus',axis=1)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)

X_train.head() 
cols_to_standardize = ['math score','writing score','reading score']

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train[cols_to_standardize] = scaler.fit_transform(X_train[cols_to_standardize])
X_train.head()
X_train.corr()
plt.figure(figsize=(16,9))

sns.heatmap(X_train.corr(),annot=True);

plt.title('Correlation for the training set');
#Let's see what is our passing rate

passed = round(sum(marks_df.OverAllPassingStatus/len(marks_df.OverAllPassingStatus.index))*100,2)

passed
from sklearn.linear_model import LogisticRegression

from sklearn.feature_selection import RFE

logistic_reg = LogisticRegression()

rfe = RFE(logistic_reg,5)

rfe = rfe.fit(X_train,y_train)
rfe.support_
cols_we_need = X_train.columns[rfe.support_]

cols_we_need
# Let's now access the Stats Model as we have found the columns that are to be used. Also, we have standardised the columns too.

import statsmodels.api as sm

X_train_sm = sm.add_constant(X_train[cols_we_need])

model = sm.GLM(y_train,X_train_sm,family = sm.families.Binomial())

res = model.fit()