# Importing libraries

import pandas as pd

import numpy as np



import seaborn as sns

import matplotlib.pyplot as plt

#sns.set_style('darkgrid')
import warnings

warnings.filterwarnings('ignore')
# Loading the train data

df = pd.read_csv('/kaggle/input/customer/Train.csv')



# Looking top 10 rows

df.head(10)
# Looking the bigger picture

df.info()
# Checking the number of missing values in each column

df.isnull().sum()
# Removing all those rows that have 3 or more missing values

df = df.loc[df.isnull().sum(axis=1)<3]
# Looking random 10 rows of the data

df.sample(10)
print('The count of each category\n',df.Var_1.value_counts())
# Checking for null values

df.Var_1.isnull().sum()
# Filling the missing values w.r.t other attributes underlying pattern 

df.loc[ (pd.isnull(df['Var_1'])) & (df['Graduated'] == 'Yes'),"Var_1"] = 'Cat_6'

df.loc[ (pd.isnull(df['Var_1'])) & (df['Graduated'] == 'No'),"Var_1"] = 'Cat_4'

df.loc[ (pd.isnull(df["Var_1"])) & ((df['Profession'] == 'Lawyer') | (df['Profession'] == 'Artist')),"Var_1"] = 'Cat_6'

df.loc[ (pd.isnull(df["Var_1"])) & (df['Age'] > 40),"Var_1"] = 'Cat_6'
# Counting Var_1 in each segment

ax1 = df.groupby(["Segmentation"])["Var_1"].value_counts().unstack().round(3)



# Percentage of category of Var_1 in each segment

ax2 = df.pivot_table(columns='Var_1',index='Segmentation',values='ID',aggfunc='count')

ax2 = ax2.div(ax2.sum(axis=1), axis = 0).round(2)



#count plot

fig, ax = plt.subplots(1,2)

ax1.plot(kind="bar",ax = ax[0],figsize = (15,4))

ax[0].set_xticklabels(labels = ['A','B','C','D'],rotation = 0)

ax[0].set_title(str(ax1))



#stacked bars

ax2.plot(kind="bar",stacked = True,ax = ax[1],figsize = (15,4))

ax[1].set_xticklabels(labels = ['A','B','C','D'],rotation = 0)

ax[1].set_title(str(ax2))

plt.show()
print('The count of gender\n',df.Gender.value_counts())
# Checking the count of missing values

df.Gender.isnull().sum()
# Counting male-female in each segment

ax1 = df.groupby(["Segmentation"])["Gender"].value_counts().unstack().round(3)



# Percentage of male-female in each segment

ax2 = df.pivot_table(columns='Gender',index='Segmentation',values='ID',aggfunc='count')

ax2 = ax2.div(ax2.sum(axis=1), axis = 0).round(2)



#count plot

fig, ax = plt.subplots(1,2)

ax1.plot(kind="bar",ax = ax[0],figsize = (15,4))

ax[0].set_xticklabels(labels = ['A','B','C','D'],rotation = 0)

ax[0].set_title(str(ax1))



#stacked bars

ax2.plot(kind="bar",stacked = True,ax = ax[1],figsize = (15,4))

ax[1].set_xticklabels(labels = ['A','B','C','D'],rotation = 0)

ax[1].set_title(str(ax2))

plt.show()
print('Count of married vs not married\n',df.Ever_Married.value_counts())
# Checking the count of missing values

df.Ever_Married.isnull().sum()
# Filling the missing values w.r.t other attributes underlying pattern

df.loc[ (pd.isnull(df["Ever_Married"])) & ((df['Spending_Score'] == 'Average') | (df['Spending_Score'] == 'High')),"Ever_Married"] = 'Yes'

df.loc[ (pd.isnull(df["Ever_Married"])) & (df['Spending_Score'] == 'Low'),"Ever_Married"] = 'No'

df.loc[ (pd.isnull(df["Ever_Married"])) & (df['Age'] > 40),"Ever_Married"] = 'Yes'

df.loc[ (pd.isnull(df["Ever_Married"])) & (df['Profession'] == 'Healthcare'),"Ever_Married"] = 'No'
# Counting married and non-married in each segment

ax1 = df.groupby(["Segmentation"])["Ever_Married"].value_counts().unstack().round(3)



# Percentage of married and non-married in each segment

ax2 = df.pivot_table(columns='Ever_Married',index='Segmentation',values='ID',aggfunc='count')

ax2 = ax2.div(ax2.sum(axis=1), axis = 0).round(2)



#count plot

fig, ax = plt.subplots(1,2)

ax1.plot(kind="bar",ax = ax[0],figsize = (15,4))

ax[0].set_xticklabels(labels = ['A','B','C','D'],rotation = 0)

ax[0].set_title(str(ax1))



#stacked bars

ax2.plot(kind="bar",stacked = True,ax = ax[1],figsize = (15,4))

ax[1].set_xticklabels(labels = ['A','B','C','D'],rotation = 0)

ax[1].set_title(str(ax2))

plt.show()
df.Age.describe(percentiles=[0.25,0.5,0.75,0.9,0.95,0.99])
# Checking the count of missing values

df.Age.isnull().sum()
# Looking the distribution of column Age

plt.figure(figsize=(10,5))



skewness = round(df.Age.skew(),2)

kurtosis = round(df.Age.kurtosis(),2)

mean = round(np.mean(df.Age),0)

median = np.median(df.Age)



plt.subplot(1,2,1)

sns.boxplot(y=df.Age)

plt.title('Boxplot\n Mean:{}\n Median:{}\n Skewness:{}\n Kurtosis:{}'.format(mean,median,skewness,kurtosis))



plt.subplot(1,2,2)

sns.distplot(df.Age)

plt.title('Distribution Plot\n Mean:{}\n Median:{}\n Skewness:{}\n Kurtosis:{}'.format(mean,median,skewness,kurtosis))



plt.show()
# Looking the distribution of column Age w.r.t to each segment

a = df[df.Segmentation =='A']["Age"]

b = df[df.Segmentation =='B']["Age"]

c = df[df.Segmentation =='C']["Age"]

d = df[df.Segmentation =='D']["Age"]



plt.figure(figsize=(15,5))



plt.subplot(1,2,1)

sns.boxplot(data = df, x = "Segmentation", y="Age")

plt.title('Boxplot')



plt.subplot(1,2,2)

sns.kdeplot(a,shade= False, label = 'A')

sns.kdeplot(b,shade= False, label = 'B')

sns.kdeplot(c,shade= False, label = 'C')

sns.kdeplot(d,shade= False, label = 'D')

plt.xlabel('Age')

plt.ylabel('Density')

plt.title("Mean\n A: {}\n B: {}\n C: {}\n D: {}".format(round(a.mean(),0),round(b.mean(),0),round(c.mean(),0),round(d.mean(),0)))



plt.show()
# Converting the datatype from float to int

df['Age'] = df['Age'].astype(int)
df.Age.describe(percentiles=[0.25,0.5,0.75,0.9,0.95,0.99])
# Divide people in the 4 age group

df['Age_Bin'] = pd.cut(df.Age,bins=[17,30,45,60,90],labels=['17-30','31-45','46-60','60+'])
# Counting different age group in each segment

ax1 = df.groupby(["Segmentation"])["Age_Bin"].value_counts().unstack().round(3)



# Percentage of age bins in each segment

ax2 = df.pivot_table(columns='Age_Bin',index='Segmentation',values='ID',aggfunc='count')

ax2 = ax2.div(ax2.sum(axis=1), axis = 0).round(2)



#count plot

fig, ax = plt.subplots(1,2)

ax1.plot(kind="bar",ax = ax[0],figsize = (15,4))

ax[0].set_xticklabels(labels = ['A','B','C','D'],rotation = 0)

ax[0].set_title(str(ax1))



#stacked bars

ax2.plot(kind="bar",stacked = True,ax = ax[1],figsize = (15,4))

ax[1].set_xticklabels(labels = ['A','B','C','D'],rotation = 0)

ax[1].set_title(str(ax2))

plt.show()
print('Count of each graduate and non-graduate\n',df.Graduated.value_counts())
# Checking the count of missing values

df.Graduated.isnull().sum()
# Filling the missing values w.r.t other attributes underlying pattern

df.loc[ (pd.isnull(df["Graduated"])) & (df['Spending_Score'] == 'Average'),"Graduated"] = 'Yes'

df.loc[ (pd.isnull(df["Graduated"])) & (df['Profession'] == 'Artist'),"Graduated"] = 'Yes'

df.loc[ (pd.isnull(df["Graduated"])) & (df['Age'] > 49),"Graduated"] = 'Yes'

df.loc[ (pd.isnull(df["Graduated"])) & (df['Var_1'] == 'Cat_4'),"Graduated"] = 'No'

df.loc[ (pd.isnull(df["Graduated"])) & (df['Ever_Married'] == 'Yes'),"Graduated"] = 'Yes'



# Replacing remaining NaN with previous values

df['Graduated'] = df['Graduated'].fillna(method='pad')
# Counting graduate and non-graduate in each segment

ax1 = df.groupby(["Segmentation"])["Graduated"].value_counts().unstack().round(3)



# Percentage of graduate and non-graduate in each segment

ax2 = df.pivot_table(columns='Graduated',index='Segmentation',values='ID',aggfunc='count')

ax2 = ax2.div(ax2.sum(axis=1), axis = 0).round(2)



#count plot

fig, ax = plt.subplots(1,2)

ax1.plot(kind="bar",ax = ax[0],figsize = (15,4))

ax[0].set_xticklabels(labels = ['A','B','C','D'],rotation = 0)

ax[0].set_title(str(ax1))



#stacked bars

ax2.plot(kind="bar",stacked = True,ax = ax[1],figsize = (15,4))

ax[1].set_xticklabels(labels = ['A','B','C','D'],rotation = 0)

ax[1].set_title(str(ax2))

plt.show()
print('Count of each profession\n',df.Profession.value_counts())
# Checking the count of missing values

df.Profession.isnull().sum()
# Filling the missing values w.r.t other attributes underlying pattern

df.loc[ (pd.isnull(df["Profession"])) & (df['Work_Experience'] > 8),"Profession"] = 'Homemaker'

df.loc[ (pd.isnull(df["Profession"])) & (df['Age'] > 70),"Profession"] = 'Lawyer'

df.loc[ (pd.isnull(df["Profession"])) & (df['Family_Size'] < 3),"Profession"] = 'Lawyer'

df.loc[ (pd.isnull(df["Profession"])) & (df['Spending_Score'] == 'Average'),"Profession"] = 'Artist'

df.loc[ (pd.isnull(df["Profession"])) & (df['Graduated'] == 'Yes'),"Profession"] = 'Artist'

df.loc[ (pd.isnull(df["Profession"])) & (df['Ever_Married'] == 'Yes'),"Profession"] = 'Artist'

df.loc[ (pd.isnull(df["Profession"])) & (df['Ever_Married'] == 'No'),"Profession"] = 'Healthcare'

df.loc[ (pd.isnull(df["Profession"])) & (df['Spending_Score'] == 'High'),"Profession"] = 'Executives'
# Count of segments in each profession

ax1 = df.groupby(["Profession"])["Segmentation"].value_counts().unstack().round(3)



# Percentage of segments in each profession

ax2 = df.pivot_table(columns='Segmentation',index='Profession',values='ID',aggfunc='count')

ax2 = ax2.div(ax2.sum(axis=1), axis = 0).round(2)



#count plot

fig, ax = plt.subplots(1,2)

ax1.plot(kind="bar",ax = ax[0],figsize = (16,5))

label = ['Artist','Doctor','Engineer','Entertainment','Executives','Healthcare','Homemaker','Lawyer','Marketing']

ax[0].set_xticklabels(labels = label,rotation = 45)



#stacked bars

ax2.plot(kind="bar",stacked = True,ax = ax[1],figsize = (16,5))

ax[1].set_xticklabels(labels = label,rotation = 45)



plt.show()
df.Work_Experience.describe(percentiles=[0.25,0.5,0.75,0.9,0.95,0.99])
# Checking the count of missing values

df.Work_Experience.isnull().sum()
# Replacing NaN with previous values

df['Work_Experience'] = df['Work_Experience'].fillna(method='pad')
# Looking the distribution of column Work Experience

plt.figure(figsize=(15,10))



skewness = round(df.Work_Experience.skew(),2)

kurtosis = round(df.Work_Experience.kurtosis(),2)

mean = round(np.mean(df.Work_Experience),0)

median = np.median(df.Work_Experience)



plt.subplot(1,2,1)

sns.boxplot(y=df.Work_Experience)

plt.title('Boxplot\n Mean:{}\n Median:{}\n Skewness:{}\n Kurtosis:{}'.format(mean,median,skewness,kurtosis))



plt.subplot(2,2,2)

sns.distplot(df.Work_Experience)

plt.title('Distribution Plot\n Mean:{}\n Median:{}\n Skewness:{}\n Kurtosis:{}'.format(mean,median,skewness,kurtosis))



plt.show()
# Looking the distribution of column Work_Experience w.r.t to each segment

a = df[df.Segmentation =='A']["Work_Experience"]

b = df[df.Segmentation =='B']["Work_Experience"]

c = df[df.Segmentation =='C']["Work_Experience"]

d = df[df.Segmentation =='D']["Work_Experience"]



plt.figure(figsize=(15,5))



plt.subplot(1,2,1)

sns.boxplot(data = df, x = "Segmentation", y="Work_Experience")

plt.title('Boxplot')



plt.subplot(1,2,2)

sns.kdeplot(a,shade= False, label = 'A')

sns.kdeplot(b,shade= False, label = 'B')

sns.kdeplot(c,shade= False, label = 'C')

sns.kdeplot(d,shade= False, label = 'D')

plt.xlabel('Work Experience')

plt.ylabel('Density')

plt.title("Mean\n A: {}\n B: {}\n C: {}\n D: {}".format(round(a.mean(),0),round(b.mean(),0),round(c.mean(),0),round(d.mean(),0)))



plt.show()
# Changing the data type

df['Work_Experience'] = df['Work_Experience'].astype(int)
df.Work_Experience.describe(percentiles=[0.25,0.5,0.75,0.9,0.95,0.99])
# Dividing the people into 3 category of work experience 

df['Work_Exp_Category'] = pd.cut(df.Work_Experience,bins=[-1,1,7,15],labels=['Low Experience','Medium Experience','High Experience'])
# Counting different category of work experience in each segment

ax1 = df.groupby(["Segmentation"])["Work_Exp_Category"].value_counts().unstack().round(3)



# Percentage of work experience in each segment

ax2 = df.pivot_table(columns='Work_Exp_Category',index='Segmentation',values='ID',aggfunc='count')

ax2 = ax2.div(ax2.sum(axis=1), axis = 0).round(2)



#count plot

fig, ax = plt.subplots(1,2)

ax1.plot(kind="bar",ax = ax[0],figsize = (15,4))

ax[0].set_xticklabels(labels = ['A','B','C','D'],rotation = 0)

ax[0].set_title(str(ax1))



#stacked bars

ax2.plot(kind="bar",stacked = True,ax = ax[1],figsize = (15,4))

ax[1].set_xticklabels(labels = ['A','B','C','D'],rotation = 0)

ax[1].set_title(str(ax2))

plt.show()
print('Count of spending score\n',df.Spending_Score.value_counts())
# Checking the count of missing values

df.Spending_Score.isnull().sum()
# Counting different category of spending score in each segment

ax1 = df.groupby(["Segmentation"])["Spending_Score"].value_counts().unstack().round(3)



# Percentage of spending score in each segment

ax2 = df.pivot_table(columns='Spending_Score',index='Segmentation',values='ID',aggfunc='count')

ax2 = ax2.div(ax2.sum(axis=1), axis = 0).round(2)



#count plot

fig, ax = plt.subplots(1,2)

ax1.plot(kind="bar",ax = ax[0],figsize = (15,4))

ax[0].set_xticklabels(labels = ['A','B','C','D'],rotation = 0)

ax[0].set_title(str(ax1))



#stacked bars

ax2.plot(kind="bar",stacked = True,ax = ax[1],figsize = (15,4))

ax[1].set_xticklabels(labels = ['A','B','C','D'],rotation = 0)

ax[1].set_title(str(ax2))

plt.show()
df.Family_Size.describe(percentiles=[0.25,0.5,0.75,0.9,0.95,0.99])
# Checking the count of missing values

df.Family_Size.isnull().sum()
# Filling the missing values w.r.t other attributes underlying pattern

df.loc[ (pd.isnull(df["Family_Size"])) & (df['Ever_Married'] == 'Yes'),"Family_Size"] = 2.0

df.loc[ (pd.isnull(df["Family_Size"])) & (df['Var_1'] == 'Cat_6'),"Family_Size"] = 2.0

df.loc[ (pd.isnull(df["Family_Size"])) & (df['Graduated'] == 'Yes'),"Family_Size"] = 2.0



# Fill remaining NaN with previous values

df['Family_Size'] = df['Family_Size'].fillna(method='pad')
# Looking the distribution of column Work Experience

plt.figure(figsize=(15,10))



skewness = round(df.Family_Size.skew(),2)

kurtosis = round(df.Family_Size.kurtosis(),2)

mean = round(np.mean(df.Family_Size),0)

median = np.median(df.Family_Size)



plt.subplot(1,2,1)

sns.boxplot(y=df.Family_Size)

plt.title('Boxplot\n Mean:{}\n Median:{}\n Skewness:{}\n Kurtosis:{}'.format(mean,median,skewness,kurtosis))



plt.subplot(2,2,2)

sns.distplot(df.Family_Size)

plt.title('Distribution Plot\n Mean:{}\n Median:{}\n Skewness:{}\n Kurtosis:{}'.format(mean,median,skewness,kurtosis))



plt.show()
# Looking the distribution of column Family Size w.r.t to each segment

a = df[df.Segmentation =='A']["Family_Size"]

b = df[df.Segmentation =='B']["Family_Size"]

c = df[df.Segmentation =='C']["Family_Size"]

d = df[df.Segmentation =='D']["Family_Size"]



plt.figure(figsize=(15,5))



plt.subplot(1,2,1)

sns.boxplot(data = df, x = "Segmentation", y="Family_Size")

plt.title('Boxplot')



plt.subplot(1,2,2)

sns.kdeplot(a,shade= False, label = 'A')

sns.kdeplot(b,shade= False, label = 'B')

sns.kdeplot(c,shade= False, label = 'C')

sns.kdeplot(d,shade= False, label = 'D')

plt.xlabel('Family Size')

plt.ylabel('Density')

plt.title("Mean\n A: {}\n B: {}\n C: {}\n D: {}".format(round(a.mean(),0),round(b.mean(),0),round(c.mean(),0),round(d.mean(),0)))



plt.show()
# Changing the data type

df['Family_Size'] = df['Family_Size'].astype(int)
df.Family_Size.describe(percentiles=[0.25,0.5,0.75,0.9,0.95,0.99])
# Divide family size into 3 category

df['Family_Size_Category'] = pd.cut(df.Family_Size,bins=[0,4,6,10],labels=['Small Family','Big Family','Joint Family'])
# Counting different category of family size in each segment

ax1 = df.groupby(["Segmentation"])["Family_Size_Category"].value_counts().unstack().round(3)



# Percentage of family size in each segment

ax2 = df.pivot_table(columns='Family_Size_Category',index='Segmentation',values='ID',aggfunc='count')

ax2 = ax2.div(ax2.sum(axis=1), axis = 0).round(3)



#count plot

fig, ax = plt.subplots(1,2)

ax1.plot(kind="bar",ax = ax[0],figsize = (15,4))

ax[0].set_xticklabels(labels = ['A','B','C','D'],rotation = 0)

ax[0].set_title(str(ax1))



#stacked bars

ax2.plot(kind="bar",stacked = True,ax = ax[1],figsize = (15,4))

ax[1].set_xticklabels(labels = ['A','B','C','D'],rotation = 0)

ax[1].set_title(str(ax2))

plt.show()
print('Count of each category of segmentation\n',df.Segmentation.value_counts())
segments = df.loc[:,"Segmentation"].value_counts()

plt.xlabel("Segment")

plt.ylabel('Count')

sns.barplot(segments.index , segments.values).set_title('Segments')

plt.show()
df.reset_index(drop=True, inplace=True)

df.info()
# number of unique ids

df.ID.nunique()
df.describe(include='all')
df = df[['ID','Gender', 'Ever_Married', 'Age', 'Age_Bin', 'Graduated', 'Profession', 'Work_Experience', 'Work_Exp_Category',

         'Spending_Score', 'Family_Size', 'Family_Size_Category','Var_1', 'Segmentation']]

df.head(10)
df1 = df.copy()

df1.head()
# Separating dependent-independent variables

X = df1.drop('Segmentation',axis=1)

y = df1['Segmentation']
# import the train-test split

from sklearn.model_selection import train_test_split



# divide into train and test sets

df1_trainX, df1_testX, df1_trainY, df1_testY = train_test_split(X,y, train_size = 0.7, random_state = 101, stratify=y)
# converting binary variables to numeric

df1_trainX['Gender'] = df1_trainX['Gender'].replace(('Male','Female'),(1,0))

df1_trainX['Ever_Married'] = df1_trainX['Ever_Married'].replace(('Yes','No'),(1,0))

df1_trainX['Graduated'] = df1_trainX['Graduated'].replace(('Yes','No'),(1,0))

df1_trainX['Spending_Score'] = df1_trainX['Spending_Score'].replace(('High','Average','Low'),(3,2,1))



# converting nominal variables into dummy variables

pf = pd.get_dummies(df1_trainX.Profession,prefix='Profession')

df1_trainX = pd.concat([df1_trainX,pf],axis=1)



vr = pd.get_dummies(df1_trainX.Var_1,prefix='Var_1')

df1_trainX = pd.concat([df1_trainX,vr],axis=1)



# scaling continuous variables

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

df1_trainX[['Age','Work_Experience','Family_Size']] = scaler.fit_transform(df1_trainX[['Age','Work_Experience','Family_Size']])



df1_trainX.drop(['ID','Age_Bin','Profession','Work_Exp_Category','Family_Size_Category','Var_1'], axis=1, inplace=True)
# converting binary variables to numeric

df1_testX['Gender'] = df1_testX['Gender'].replace(('Male','Female'),(1,0))

df1_testX['Ever_Married'] = df1_testX['Ever_Married'].replace(('Yes','No'),(1,0))

df1_testX['Graduated'] = df1_testX['Graduated'].replace(('Yes','No'),(1,0))

df1_testX['Spending_Score'] = df1_testX['Spending_Score'].replace(('High','Average','Low'),(3,2,1))



# converting nominal variables into dummy variables

pf = pd.get_dummies(df1_testX.Profession,prefix='Profession')

df1_testX = pd.concat([df1_testX,pf],axis=1)



vr = pd.get_dummies(df1_testX.Var_1,prefix='Var_1')

df1_testX = pd.concat([df1_testX,vr],axis=1)



# scaling continuous variables

df1_testX[['Age','Work_Experience','Family_Size']] = scaler.transform(df1_testX[['Age','Work_Experience','Family_Size']])



df1_testX.drop(['ID','Age_Bin','Profession','Work_Exp_Category','Family_Size_Category','Var_1'], axis=1, inplace=True)
df1_trainX.shape, df1_trainY.shape, df1_testX.shape, df1_testY.shape
# Correlation matrix

plt.figure(figsize=(17,10))

sns.heatmap(df1_trainX.corr(method='spearman').round(2),linewidth = 0.5,annot=True,cmap="YlGnBu")

plt.show()
df2 = df.copy()

df2.head()
# Separating dependent-independent variables

X = df2.drop('Segmentation',axis=1)

y = df2['Segmentation']
# import the train-test split

from sklearn.model_selection import train_test_split



# divide into train and test sets

df2_trainX, df2_testX, df2_trainY, df2_testY = train_test_split(X,y, train_size = 0.7, random_state = 101, stratify=y)
# Converting binary to numeric

df2_trainX['Gender'] = df2_trainX['Gender'].replace(('Male','Female'),(1,0))

df2_trainX['Ever_Married'] = df2_trainX['Ever_Married'].replace(('Yes','No'),(1,0))

df2_trainX['Graduated'] = df2_trainX['Graduated'].replace(('Yes','No'),(1,0))



# Converting nominal variables to dummy variables

ab = pd.get_dummies(df2_trainX.Age_Bin,prefix='Age_Bin')

df2_trainX = pd.concat([df2_trainX,ab],axis=1)



pf = pd.get_dummies(df2_trainX.Profession,prefix='Profession')

df2_trainX = pd.concat([df2_trainX,pf],axis=1)



we = pd.get_dummies(df2_trainX.Work_Exp_Category,prefix='WorkExp')

df2_trainX = pd.concat([df2_trainX,we],axis=1)



sc = pd.get_dummies(df2_trainX.Spending_Score,prefix='Spending')

df2_trainX = pd.concat([df2_trainX,sc],axis=1)



fs = pd.get_dummies(df2_trainX.Family_Size_Category,prefix='FamilySize')

df2_trainX = pd.concat([df2_trainX,fs],axis=1)



vr = pd.get_dummies(df2_trainX.Var_1,prefix='Var_1')

df2_trainX = pd.concat([df2_trainX,vr],axis=1)



df2_trainX.drop(['ID','Age','Age_Bin','Profession','Work_Experience','Work_Exp_Category','Spending_Score',

               'Family_Size','Family_Size_Category','Var_1'],axis=1,inplace=True)
# Converting binary to numeric

df2_testX['Gender'] = df2_testX['Gender'].replace(('Male','Female'),(1,0))

df2_testX['Ever_Married'] = df2_testX['Ever_Married'].replace(('Yes','No'),(1,0))

df2_testX['Graduated'] = df2_testX['Graduated'].replace(('Yes','No'),(1,0))



# Converting nominal variables to dummy variables

ab = pd.get_dummies(df2_testX.Age_Bin,prefix='Age_Bin')

df2_testX = pd.concat([df2_testX,ab],axis=1)



pf = pd.get_dummies(df2_testX.Profession,prefix='Profession')

df2_testX = pd.concat([df2_testX,pf],axis=1)



we = pd.get_dummies(df2_testX.Work_Exp_Category,prefix='WorkExp')

df2_testX = pd.concat([df2_testX,we],axis=1)



sc = pd.get_dummies(df2_testX.Spending_Score,prefix='Spending')

df2_testX = pd.concat([df2_testX,sc],axis=1)



fs = pd.get_dummies(df2_testX.Family_Size_Category,prefix='FamilySize')

df2_testX = pd.concat([df2_testX,fs],axis=1)



vr = pd.get_dummies(df2_testX.Var_1,prefix='Var_1')

df2_testX = pd.concat([df2_testX,vr],axis=1)



df2_testX.drop(['ID','Age','Age_Bin','Profession','Work_Experience','Work_Exp_Category','Spending_Score',

               'Family_Size','Family_Size_Category','Var_1'],axis=1,inplace=True)
df2_trainX.shape, df2_trainY.shape, df2_testX.shape, df2_testY.shape
# Correlation matrix

plt.figure(figsize=(17,10))

sns.heatmap(df2_trainX.corr(method='spearman').round(2),linewidth = 0.5,annot=True,cmap="YlGnBu")

plt.show()
train_dt1_x = df1_trainX.copy()

train_dt1_x.head()
train_dt1_y = df1_trainY.copy()

train_dt1_y.head()
# importing decision tree classifier 

from sklearn.tree import DecisionTreeClassifier



# creating the decision tree function

model_dt1 = DecisionTreeClassifier(random_state=10,criterion='gini')



#fitting the model

model_dt1.fit(train_dt1_x, train_dt1_y)



# depth of the decision tree

print('Depth of the Decision Tree: ', model_dt1.get_depth())



#checking the training score

print('Accuracy on training: ',model_dt1.score(train_dt1_x, train_dt1_y))



# predict the target on the train dataset

yhat1 = model_dt1.predict(train_dt1_x)



from sklearn.metrics import confusion_matrix

cm1 = confusion_matrix(train_dt1_y.values, yhat1, labels=["A","B","C","D"])

print('-------The confusion matrix for this model is-------')

print(cm1)



from sklearn.metrics import classification_report

print('\n\n-------Printing the whole report of the model-------')

print(classification_report(train_dt1_y.values, yhat1))
X1 = train_dt1_x.copy()

y1 = pd.DataFrame({'Seg':train_dt1_y})

y1['Seg'] = y1['Seg'].replace(('A','B','C','D'),(1,2,3,4))
# Implementing grid search



parameter_grid = {

    'max_depth' : [24,25,26,27,28,29,30],

    'max_features': [0.3, 0.5, 0.7]

    }



from sklearn.model_selection import GridSearchCV

gridsearch = GridSearchCV(estimator=model_dt1, param_grid=parameter_grid, scoring='neg_mean_squared_error', cv=5)



gridsearch.fit(X1, y1)



print(gridsearch.best_params_)
# Implementing random search



parameter_grid = {

    'max_depth' : [24,25,26,27,28,29,30],

    'max_features': [0.3, 0.5, 0.7,0.9]

    }



from sklearn.model_selection import RandomizedSearchCV



randomsearch = RandomizedSearchCV(estimator=model_dt1, param_distributions=parameter_grid, n_iter= 10, cv=5)

randomsearch.fit(X1, y1)



print(randomsearch.best_params_)
# final model

model_dt1 = DecisionTreeClassifier(max_depth=26, max_features=0.9 ,random_state=10)



# fitting the model

model_dt1.fit(train_dt1_x, train_dt1_y)



# Training score

print(model_dt1.score(train_dt1_x, train_dt1_y).round(4))
from sklearn import tree



fig = plt.figure(figsize=(25,20))

_ = tree.plot_tree(model_dt1, feature_names=train_dt1_x.columns, max_depth=2, filled=True)
test_dt1_x = df1_testX.copy()

test_dt1_x.head()
test_dt1_y = df1_testY.copy()

test_dt1_y.head()
y_dt1 = model_dt1.predict(test_dt1_x)

y_dt1
from sklearn.metrics import confusion_matrix

print('-------The confusion matrix for test data is-------\n')

print(confusion_matrix(test_dt1_y.values, y_dt1, labels=["A","B","C","D"]))



from sklearn.metrics import classification_report

print('\n\n-------Printing the report of test data-------\n')

print(classification_report(test_dt1_y.values, y_dt1))
pd.Series(y_dt1).value_counts()
train_dt2_x = df2_trainX.copy()

train_dt2_x.head()
train_dt2_y = df2_trainY.copy()

train_dt2_y.head()
# importing decision tree classifier 

from sklearn.tree import DecisionTreeClassifier



# creating the decision tree function

model_dt2 = DecisionTreeClassifier(random_state=10,criterion='gini')



#fitting the model

model_dt2.fit(train_dt2_x, train_dt2_y)



# depth of the decision tree

print('Depth of the Decision Tree: ', model_dt2.get_depth())



#checking the training score

print('Accuracy on training: ',model_dt2.score(train_dt2_x, train_dt2_y))



# predict the target on the train dataset

yhat2 = model_dt2.predict(train_dt2_x)



from sklearn.metrics import confusion_matrix

cm2 = confusion_matrix(train_dt2_y.values, yhat2, labels=["A","B","C","D"])

print('-------The confusion matrix for this model is-------')

print(cm2)



from sklearn.metrics import classification_report

print('\n\n-------Printing the whole report of the model-------')

print(classification_report(train_dt2_y.values, yhat2))
X2 = train_dt2_x.copy()

y2 = pd.DataFrame({'Seg':train_dt2_y})

y2['Seg'] = y2['Seg'].replace(('A','B','C','D'),(1,2,3,4))
# Implementing grid search



parameter_grid = {

    'max_depth' : [24,25,26,27,28,29,30],

    'max_features': [0.3, 0.5, 0.7]

    }



from sklearn.model_selection import GridSearchCV

gridsearch = GridSearchCV(estimator=model_dt2, param_grid=parameter_grid, scoring='neg_mean_squared_error', cv=5)



gridsearch.fit(X2, y2)



print(gridsearch.best_params_)
# Implementing random search



parameter_grid = {

    'max_depth' : [24,25,26,27,28,29,30],

    'max_features': [0.3, 0.5, 0.7,0.9]

    }



from sklearn.model_selection import RandomizedSearchCV



randomsearch = RandomizedSearchCV(estimator=model_dt2, param_distributions=parameter_grid, n_iter= 10, cv=5)

randomsearch.fit(X2, y2)



print(randomsearch.best_params_)
# final model

model_dt2 = DecisionTreeClassifier(max_depth=25, max_features=0.7, random_state=10)



#fitting the model

model_dt2.fit(train_dt2_x, train_dt2_y)



#Training score

print(model_dt2.score(train_dt2_x, train_dt2_y).round(4))
from sklearn import tree



fig = plt.figure(figsize=(25,20))

_ = tree.plot_tree(model_dt2, feature_names=train_dt2_x.columns, max_depth=2, filled=True)
test_dt2_x = df2_testX.copy()

test_dt2_x.head()
test_dt2_y = df2_testY.copy()

test_dt2_y.head()
y_dt2 = model_dt2.predict(test_dt2_x)

y_dt2
from sklearn.metrics import confusion_matrix

print('-------The confusion matrix for test data is-------')

print(confusion_matrix(test_dt2_y.values, y_dt2, labels=["A","B","C","D"]))



from sklearn.metrics import classification_report

print('\n\n-------Printing the report of test data-------')

print(classification_report(test_dt2_y.values, y_dt2))
pd.Series(y_dt2).value_counts()
print('************************  MODEL-1 REPORT  *********************************\n')

print('Train data')

print(classification_report(train_dt1_y.values, yhat1))

print('\nTest data')

print(classification_report(test_dt1_y.values, y_dt1))
print('************************  MODEL-2 REPORT  *********************************\n')

print('Train data')

print(classification_report(train_dt2_y.values, yhat2))

print('\nTest data')

print(classification_report(test_dt2_y.values, y_dt2))
train_rf1_x = df1_trainX.copy()

train_rf1_x.head()
train_rf1_y = df1_trainY.copy()

train_rf1_y.head()
# Importing the library

from sklearn.ensemble import RandomForestClassifier



# Instantiate the classifier with 20 decision tree

rfc1 = RandomForestClassifier(random_state=0,n_estimators=20)



# Train model

model_rfc1 = rfc1.fit(train_rf1_x, train_rf1_y)



# Predicting the classes

yhat3 = rfc1.predict(train_rf1_x)



# view the feature scores

feature_scores = pd.Series(rfc1.feature_importances_, index=train_rf1_x.columns).sort_values(ascending=False)

print('The importance of features ranked from high to low:\n',feature_scores)



from sklearn.metrics import confusion_matrix

cm3 = confusion_matrix(train_rf1_y.values, yhat3, labels=["A","B","C","D"])

print('\n\n-------The confusion matrix for this model is-------')

print(cm3)



from sklearn.metrics import classification_report

print('\n\n-------Printing the whole report of the model-------')

print(classification_report(train_rf1_y.values, yhat3))
# Creating bar plot of scores of variables importance

plt.figure(figsize=(10,8))

sns.barplot(x=feature_scores, y=feature_scores.index)

plt.xlabel('Feature Importance Score')

plt.ylabel('Features')

plt.title("Visualizing Important Features")

plt.show()
test_rf1_x = df1_testX.copy()

test_rf1_x.head()
test_rf1_y = df1_testY.copy()

test_rf1_y.head()
y_rf1 = rfc1.predict(test_rf1_x)

y_rf1
from sklearn.metrics import confusion_matrix

print('-------The confusion matrix for test data is-------\n')

print(confusion_matrix(test_rf1_y.values, y_rf1, labels=["A","B","C","D"]))



from sklearn.metrics import classification_report

print('\n\n-------Printing the report of test data-------\n')

print(classification_report(test_rf1_y.values, y_rf1))
pd.Series(y_rf1).value_counts()
train_rf2_x = df2_trainX.copy()

train_rf2_x.head()
train_rf2_y = df2_trainY.copy()

train_rf2_y.head()
# Importing the library

from sklearn.ensemble import RandomForestClassifier



# Instantiate the classifier with 20 decision tree

rfc2 = RandomForestClassifier(random_state=0,n_estimators=20)



# Train model

model_rfc2 = rfc2.fit(train_rf2_x, train_rf2_y)



# Predicting the classes

yhat4 = rfc2.predict(train_rf2_x)



# view the feature scores

feature_scores = pd.Series(rfc2.feature_importances_, index=train_rf2_x.columns).sort_values(ascending=False)

print('The importance of features ranked from high to low:\n',feature_scores)



from sklearn.metrics import confusion_matrix

cm4 = confusion_matrix(train_rf2_y.values, yhat4, labels=["A","B","C","D"])

print('\n\n-------The confusion matrix for this model is-------')

print(cm4)



from sklearn.metrics import classification_report

print('\n\n-------Printing the whole report of the model-------')

print(classification_report(train_rf2_y.values, yhat4))
# Creating bar plot of scores of variables importance

plt.figure(figsize=(10,8))

sns.barplot(x=feature_scores, y=feature_scores.index)

plt.xlabel('Feature Importance Score')

plt.ylabel('Features')

plt.title("Visualizing Important Features")

plt.show()
test_rf2_x = df2_testX.copy()

test_rf2_x.head()
test_rf2_y = df2_testY.copy()

test_rf2_y.head()
y_rf2 = rfc2.predict(test_rf2_x)

y_rf2
from sklearn.metrics import confusion_matrix

print('-------The confusion matrix for test data is-------\n')

print(confusion_matrix(test_rf2_y.values, y_rf2, labels=["A","B","C","D"]))



from sklearn.metrics import classification_report

print('\n\n-------Printing the report of test data-------\n')

print(classification_report(test_rf2_y.values, y_rf2))
pd.Series(y_rf2).value_counts()
print('************************  MODEL-1 REPORT  *********************************\n')

print('Train data')

print(classification_report(train_rf1_y.values, yhat3))

print('\nTest data')

print(classification_report(test_rf1_y.values, y_rf1))
print('************************  MODEL-2 REPORT  *********************************\n')

print('Train data')

print(classification_report(train_rf2_y.values, yhat4))

print('\nTest data')

print(classification_report(test_rf2_y.values, y_rf2))