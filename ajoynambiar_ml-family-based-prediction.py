# Suppressing Warnings|

import warnings

warnings.filterwarnings('ignore')



# Importing Pandas and NumPy

import pandas as pd, numpy as np



import matplotlib.pyplot as plt

import seaborn as sns





# set seaborn theme if you prefer

sns.set()
pd.set_option('display.max_columns', 50)  #Nice to have if if workign with large data

pd.set_option('display.max_rows', 50)  
# Importing all datasets



#Importing the input files

titanic = pd.read_csv('/kaggle/input/titanic/train.csv')

Test = pd.read_csv('/kaggle/input/titanic/test.csv')

#titanic = pd.read_csv("train.csv")

#Test=pd.read_csv("test.csv")

titanic.head()
# Let's check the dimensions of the dataframe

titanic.shape
# let's look at the statistical aspects of the dataframe

titanic.describe()
# Let's see the type of each column

titanic.info()
#Make a editable copy

df_train=titanic.copy()
#make an editable copy of test. whatever features we create in train will make in test

df_test=Test.copy()
df_test['Survived']='NA'
df=df_train.append(df_test,ignore_index = True)

#survived col becomes object column after append
df.shape
#Drop any duplicate rows

df= df.drop_duplicates(keep='first')
df.shape #no duplicates as 891 rows
# Create function to check distribution/ variation in each column - name, unique values, isnull, dtype

def unique_col_values(d):

    for column in d:

        print("{}  | {} | {} | {} ".format(d[column].name, len(d[column].unique()), d[column].isnull().sum(), d[column].dtype))

        

unique_col_values(df)



#several Nulls in age and cabin and couple embarked
df.describe(include='all')
df[df.Age.isna()].describe(include='all')

#check any pattern when age is null
base_survival_rate=df_train['Survived'].value_counts()[1]/df_train['Survived'].count()

base_survival_rate  # Save the overall baseline survival rate. I guess the survival rate in test data should be same
#Impute single 2 null in mebarked

df.Embarked=df['Embarked'].fillna('S')
unique_col_values(df)



#Several null in cavin and Age
#Add family Name

df['Family_name']=df['Name'].str.split(', ').str[0]



df
#Add familysize

df['FamilySize']= df['SibSp']+df['Parch']+1



df.head()

#Extrac cabin information

cabin_only = df[["Cabin"]].copy()

cabin_only["Cabin_Data"] = cabin_only["Cabin"].isnull().apply(lambda x: not x) # extract rows that do not contain null Cabin data.

cabin_only.head()

#Slice cabin into two columns Desk and room

cabin_only["Deck"] = cabin_only["Cabin"].str.slice(0,1)

cabin_only["Room"] = cabin_only["Cabin"].str.slice(1,5).str.extract("([0-9]+)", expand=False).astype("float") #int does not work for NAN

cabin_only[cabin_only["Cabin_Data"]]

cabin_only.head()



#retain room and cabin columns and replace null with N and median

cabin_only.drop(["Cabin", "Cabin_Data"], axis=1, inplace=True, errors="ignore")

cabin_only["Deck"] = cabin_only["Deck"].fillna("N") # assign 'N' for the deck name of the null Cabin value. 

cabin_only["Room"] = cabin_only["Room"].fillna("N") # use mean to fill null Room values.

cabin_only.head()

cabin_only=cabin_only.join(pd.get_dummies(cabin_only['Deck'], prefix='Deck'))

cabin_only=cabin_only.drop(['Deck'], axis=1)

cabin_only



df=pd.concat([df,cabin_only],axis=1)

df.shape
cabin_only.head()
# extract numbers from the ticket

df['Ticket_numerical'] = df.Ticket.apply(lambda s: s.split()[-1]) #First character/ digit

df[["Ticket","Ticket_numerical"]] .head(20)
# dtype shows this is object. Replace non numerical will null

df['Ticket_numerical'] = np.where(df.Ticket_numerical.str.isdigit(), df.Ticket_numerical, np.nan)



df["Ticket_numerical"] = df["Ticket_numerical"].fillna(0) # some tickets have string values only, so we will assign a 0 for their ticket_numerical.

df['Ticket_numerical'] = df['Ticket_numerical'].astype('int64')

df["Ticket_numerical"] .tail()
# Similarly extract the first part of ticket as category

df['Ticket_categorical'] = df.Ticket.apply(lambda s: s.split()[0])

df['Ticket_categorical'] = np.where(df.Ticket_categorical.str.isdigit(), np.nan, df.Ticket_categorical)

df["Ticket_categorical"] = df["Ticket_categorical"].fillna("NONE") # some tickets have digit values only, so we will assign 'NONE' for their ticket_categorical.

df['Ticket_numerical'].tolist()

df[['Ticket', 'Ticket_numerical', 'Ticket_categorical']].head()



pd.DataFrame(df.Ticket_categorical.value_counts())

#Several categorgical value. Lets bucket them to similar sounding

#Hack: printed in dataframe so I can copy into excel and match them
df['Ticket_cat']=np.where(df['Ticket_categorical']!='NONE', 1, 0) #Binary label for ticket category

df.head()
pd.DataFrame(df.Ticket_cat.value_counts())
df.Name.head()

#Can extract title/ salutation
#Name Salutation

df['Title'] = df['Name'].str.strip()

df['Title'] = df['Name'].str.split('.').str[0]

df['Title'] = df['Title'].str.split(',').str[1]

df['Title']= df['Title'].str.strip()

df['Title']



df_test['Title'] = df_test['Name'].str.strip()

df_test['Title'] = df_test['Name'].str.split('.').str[0]

df_test['Title'] = df_test['Title'].str.split(',').str[1]

df_test['Title']= df_test['Title'].str.strip()
df['Title'].value_counts()
#Check dataframe where we have uncommon title

df[(df['Title']!='Mr') & (df['Title']!='Miss') & (df['Title']!='Mrs')&(df['Title']!='Master')&(df['Title']!='Dr')&(df['Title']!='Rev')]
#Do some bucketing for uncommon names - can do 

df['Title']=df['Title'].replace(['Don','Lady','Sir','the Countess','Jonkheer','Mme','Mlle','Dona'],'Titled') #Titled

df['Title']=df['Title'].replace(['Major', 'Col','Capt','Dr','Rev'],'Titled') #officer bucket

df['Title']=df['Title'].replace(['Ms'],'Miss') #ms and miss are same



df['Title'].value_counts()
df.head()
df['Singleton'] = df['FamilySize'].map(lambda s: 1 if s == 1 else 0)

df['SmallFamily'] = df['FamilySize'].map(lambda s: 1 if 2 <= s <= 4 else 0)

df['LargeFamily'] = df['FamilySize'].map(lambda s: 1 if 5 <= s else 0)

df.head()
unique_col_values(df)
df[df.Age.notna()].describe(include='all')
df[df.Age.isna()].describe(include='all')
#Add a flag where age is null

df["Age_Null"]= np.where(df.Age.notna(), 'No', 'Yes')

df[["Age","Age_Null"]].head(10)
#Storing Numerical and categorical vairable names

numerical=df.select_dtypes(exclude=['object']).columns #Pulls numerical columns

numerical=numerical[numerical!='PassengerId'] #PassengerId is not independent variable i.e. has no bearing 

numerical=numerical[numerical!='Age'] #Remove age 

#print(numerical)







#Plot Numerical features violin plot

#https://dev.to/nexttech/how-to-perform-exploratory-data-analysis-with-seaborn-29eo

fig, ax = plt.subplots(5, 3, figsize=(20, 20))

for variable, subplot in zip(numerical, ax.flatten()):

    sns.violinplot(x=df["Age_Null"], y=df[variable],split=True,inner='quartile' , ax=subplot, palette="Set2")

    for label in subplot.get_xticklabels():

        label.set_rotation(0)

        #print(label)

        

#Most significant difference is Pclass. More likelihood of Age null when Plass=3

#Violin plot are best of both worlds- power of histrogram to show probability density and conciseness of boxplot
# Crosstab help to create frequency table. I think of it similar to pivot table in excel

#Adding normalize='index' helps normalize row wise to i.e. sum equal 1 for row

variable_age_null=pd.crosstab(columns=df["Age_Null"], index=df["Title"],normalize='index')

variable_age_null



#Here it show if title is Mr 23% times name will be null. for Miss nulls are 19.6%. Very rare to have age null with titled folks
categorical=list(df.select_dtypes(include=['object']).columns)  #pulls categorical columns

print(categorical)

categorical=[ 'Sex',   'Embarked',  'Title']

#Keep which categories we want to explore

#fig, ax = plt.subplots(2, 3, figsize=(20, 10))

for variable, subplot in zip(categorical, ax.flatten()):

    variable_age_null=pd.crosstab(columns=df["Age_Null"], index=df[variable],normalize='index')

    variable_age_null.plot.bar(stacked=False,figsize=(7,3), rot=0)

    print(variable)

    plt.show()

    

    #for label in subplot.get_xticklabels():

        #label.set_rotation(0)

        #print(label)

        #plt.show()
### Most Striking difference is when age will be null is from embarked and title



##So using above we can impute age based on Pclass, embarked, Sex and Title
#Age imuation

#Lets slice up grpupos to see age statistics





Age_map=pd.DataFrame(df.groupby(['Title', 'Pclass','Embarked','Sex'])['Age'].agg(['median','mean','count'])).reset_index()

median_age=df.Age.median()
Age_map #use Age map to impure
median_age #Or use simple median
# Updating the missing age based on above lookup table. Create a function

#https://www.kaggle.com/amritachatterjee09/predicting-survival-in-titanic-disaster

def impute_age(x):

    try:    

        return Age_map[(Age_map.Pclass==x.Pclass)&(Age_map.Title==x.Title)&(Age_map.Embarked==x.Embarked)&(Age_map.Sex==x.Sex)]['mean'].values[0]

    except:

        pass



df['Age_imp'] = df.apply(lambda x: impute_age(x) if np.isnan(x['Age']) else  x['Age'], axis=1) 



df['Age_imp_med']=df.Age #simple median impute

df['Age_imp_med']=df['Age_imp_med'].fillna(median_age)



#compare the raw age column with nulls (nulls dont show), map impute and simple median impute in a histogram



figure = plt.figure(figsize=(15, 7))

plt.hist([df['Age'], df['Age_imp'], df['Age_imp_med']], color = ['g','r','y'], stacked=False, bins = 30, label = ['Age','Age_Imp','Age_imp_med'])

plt.xlabel('Age')

plt.ylabel('Count')

plt.legend();



#Its seems the mapping impute is better than simple median impute
df['Age']=df['Age_imp'] #Replace 

df=df.drop(['Age_imp','Age_imp_med'], axis=1) #Drop other age impute
unique_col_values(df) #Fare has null. Cabin which we have tried to feature



#Note: Here I have append test and train. This helped impute but technically wrong as I looked in to test data statistics

#In real world we will not have test data access. I appended so my code is simpler





df.Fare=df.Fare.fillna(df_test.Fare.median())

df.describe()
df[df.Fare==0]  #Why would some passengers pay zero fare? Did they work on the ship. Only 1 survived.

#Few had common ticket number. 
df=df.drop(['Cabin','Age_Null'],axis=1) #Cabin has been featured and Age null not needed any more (age imputed)
unique_col_values(df) #no nulls except Cabin
#Storing Numerical and categorical vairable names

numerical=df.select_dtypes(exclude=['object']).columns #Pulls numerical columns

numerical=numerical[numerical!='PassengerId'] #PassengerId is not independent variable i.e. has no bearing 

numerical=numerical[numerical!='Survived']

#print(numerical)







#Plot Numerical features violin plot

#https://dev.to/nexttech/how-to-perform-exploratory-data-analysis-with-seaborn-29eo

fig, ax = plt.subplots(7, 3, figsize=(20, 50))

for variable, subplot in zip(numerical, ax.flatten()):

    sns.violinplot(x=df[df.Survived!='NA']["Survived"], y=df[variable],split=True,inner='quartile' , ax=subplot, palette="Set2")

    for label in subplot.get_xticklabels():

        label.set_rotation(0)

        #print(label)

        

#Difference of violin in Age, Fare, PClass, Familysize,deckN
categorical=list(df.select_dtypes(include=['object']).columns ) #pulls categorical columns

print(categorical)
categorical=[ 'Sex',   'Embarked',    'Title']

#Only categories of interest
#fig, ax = plt.subplots(2, 3, figsize=(20, 10))

for variable, subplot in zip(categorical, ax.flatten()):

    variable_survive=pd.crosstab(columns=df["Survived"], index=df[variable],normalize='index')

    variable_lift=variable_survive[1.0]/base_survival_rate

    

    variable_lift.plot.bar(stacked=False,figsize=(7,3), rot=0)

    print(variable)

    plt.show()

    #for label in subplot.get_xticklabels():

        #label.set_rotation(0)

        #print(label)

        #plt.show()

        

#Observation

#Female have high chance of survival - makes sense

#Women and girls have high chance, then Boys (Masters) and then titled people. Mr. have lowest chance of survival - ok

#EMbarked from C had higher change of survival - not sure what



#Concatenate Family name and ticket to group family

df['FamilyID']=df['Family_name'].map(str) +   '-' + df['Ticket'].map(str)#\

                #+'-' + df['Fare'].map(str)+'-' + df['Embarked'].map(str)+'-' + df['Pclass'].map(str)
family=df[['PassengerId','FamilyID','Age','Sex','Title','Fare','FamilySize','Singleton','SmallFamily','LargeFamily','Survived']].sort_values('FamilyID',ascending=False)

family
len(family.FamilyID.unique()) #1024 family - can have singletons
family.FamilySize.value_counts() #790 singletons. Ltes remove them
a=pd.DataFrame(family.groupby('FamilyID')['PassengerId'].agg(['count']).reset_index())  #temp df

single_people=set(a[a['count']==1].FamilyID)



family[family['FamilyID'].isin(single_people)].FamilySize.value_counts() #shows family size is not correct
#https://stackoverflow.com/questions/35905335/aggregation-over-partition-pandas-dataframe

#calculate AVG(value) OVER (PARTITION BY group) - transfor is similar to SQL over

#df['mean_value'] = df.groupby('group').value.transform(np.mean)



family['FamilySize_new'] = family.groupby('FamilyID').PassengerId.transform('count')

family.head()
family[family.FamilySize!=family.FamilySize_new].info()

family_size_incorrect=set(family[family.FamilySize!=family.FamilySize_new].FamilyID)





family[family.FamilyID.isin(family_size_incorrect)]  #New FamilySize seems better
df=pd.merge(df,family[['PassengerId','FamilySize_new']], on='PassengerId', how='left')

df.head()
df[df.FamilySize!=df.FamilySize_new].info()  #check ok
df['Singleton_n'] = df['FamilySize_new'].map(lambda s: 1 if s == 1 else 0)

df['SmallFamily_n'] = df['FamilySize_new'].map(lambda s: 1 if 2 <= s <= 4 else 0)

df['LargeFamily_n'] = df['FamilySize_new'].map(lambda s: 1 if 5 <= s else 0)

df.head()
family=family[(family.FamilySize_new>1)]

family
len(family.FamilyID.unique()) #176 family 
#Lets check family groups having women and children



family['W_C']=np.where((family.Title=='Mrs') | (family.Title=='Master')| (family.Title=='Miss'), 1, 0)

family#[family['W_C']==1]
#Treat titlted folks as well

family['W_C']=np.where((family.Sex=='female')| (family.Age<14), 1, 0)

family[(family.Title=='Titled')]
family.W_C.value_counts() #188 men in 240 families
family[(family.W_C==1) ].FamilyID.value_counts()  #1 count is women with 1 man- 1 woman or 1man - 1 child combination
W_C_small_family=pd.DataFrame(family[(family.W_C==1) ].FamilyID.value_counts()).reset_index() #covert above to pandas

W_C_small_family.columns = ['FamilyID', 'count']

W_C_small_family.tail()
wc_group_names=set(W_C_small_family[W_C_small_family['count'] !=1].FamilyID)     #Remove 1 count as they are part of 1man-1woman couple 

len(wc_group_names)  #74  women child family combination


df[df['Ticket_numerical'].isin(['11752'])]  #Observe instances wherwe Familysize is not accurate

df[df['Ticket_numerical'].isin(['347091'])] 



df[df['Family_name'].isin(['Andersson'])] 



#df[df['Ticket_numerical'].isin(['3101281'])] 
family[~family['FamilyID'].isin(wc_group_names)]
family_WC=family[family['FamilyID'].isin(wc_group_names)] #df where 2 or more W-C in a single family

family_WC=family_WC[family_WC['W_C']==1]

family_WC.head()
len(family_WC.FamilyID.unique()) #74 such families
family_WC_test=family_WC[(family_WC.Survived=='NA')   ]

len(family_WC_test.FamilyID.unique()) #47 W-C family need prediction who are themselves women or children
test_family=set(family_WC_test.FamilyID)

test_family
family_WC_train=family_WC[(~family_WC['FamilyID'].isin(test_family)) ]  #Families where we know if W-C survived. No prediction required

len(set(family_WC_train.FamilyID))  #We have information about 24 family out of 50

#In other words we know if someone lived or died from 24 families. We culd use it to predict from test family set
family_WC_train.info()
family_WC_train.Survived=family_WC_train.Survived.astype("float")
family_WC_train.head()
#https://stackoverflow.com/questions/35905335/aggregation-over-partition-pandas-dataframe

#calculate AVG(value) OVER (PARTITION BY group) - transform is similar to SQL over

#df['mean_value'] = df.groupby('group').value.transform(np.mean)



family_WC_train['family_survive'] = family_WC_train.groupby('FamilyID').Survived.transform(np.mean)

family_WC_train
#Rank family member. Senior more member given rank 1 etc

family_WC_train['Age_rank'] = family_WC_train.groupby('FamilyID').Age.rank(method='dense', ascending=False)

#Dense- How to rank the group of records that have the same value (i.e. ties):



family_WC_train
family_WC_train.family_survive.value_counts()  #Interesting. The W-C fmaily group survive together
family_WC_train.groupby('FamilyID')['family_survive'].agg(['max']).reset_index()

#1 means everyone survicved 0- means all W_C did not survive

#this is remarkable all women and children in families either survive or die together. Only 1 exception is Allison-I-113781
#Family groups which have show in test 

family_WC_test=family_WC[family_WC['FamilyID'].isin(test_family)]

family_WC_test



#Remove rows which have Survived-NA to come up with family survival aggregate

family_WC_test_notNA=family_WC_test[family_WC_test.Survived!='NA']

family_WC_test_notNA.Survived=family_WC_test_notNA.Survived.astype('float')

family_WC_test_notNA.head()

#Assign the aggregate survival rate

family_WC_test_notNA['family_survive'] = family_WC_test_notNA.groupby('FamilyID').Survived.transform(np.mean)

family_WC_test_notNA
family_survive_map=family_WC_test_notNA.groupby('FamilyID')['family_survive'].agg(['max']).reset_index()

family_survive_map.columns = ['FamilyID', 'family_survive'] #Rename the clumn for mapping df

family_survive_map

#again only one family which show up in test set and train set show they survive togther. only Asplund-III-347077-31.3875-S has one child who dies.

#Will use this as a key to fill up NA
family_WC_test_NA=family_WC[(family_WC['FamilyID'].isin(test_family)) & (family_WC['Survived']=='NA')]

family_WC_test_NA.info()

#Can susbstitue the survival information of these NA by above mapping table - assuming they also survive together
family_WC_test_NA.head()
# Updating the missing age based on above lookup table

def survive_family(x):

    try:    

        return family_survive_map[family_survive_map.FamilyID==x.FamilyID]['family_survive'].values[0]

    except:

        return x.Survived



family_WC_test_NA['Survived'] = family_WC_test_NA.apply(lambda x: survive_family(x) if (x['Survived']=='NA') else  x['Survived'], axis=1) 


family_WC_test_NA.Survived.value_counts()
family_WC_test_NA.Survived=np.where(family_WC_test_NA.Survived==.75, 1, family_WC_test_NA.Survived)

#family_WC_test_NA.Survived=np.where(family_WC_test_NA.Survived==0.5, 0, family_WC_test_NA.Survived)



family_WC_test_NA.Survived.value_counts()

#successfuly predictred 52 passengers basedon their family survival
cannot_predict_family=set(family_WC_test_NA[family_WC_test_NA.Survived=='NA'].FamilyID)
family[family['FamilyID'].isin(cannot_predict_family)]

#Except van Billiard all where travelling without men. Perhaps we can profile how families without men did
family_survival_pred=family_WC_test_NA[family_WC_test_NA.Survived!='NA'][['PassengerId','Survived']]



family_survival_pred.Survived=family_survival_pred.Survived.astype(int)

family_survival_pred.info()



#So overall we have 51 prediction just based on if other W-C survived in their famlies
df.head()
# Creating a dummy variable for some of the categorical variables and dropping the first one.

dummy1 = pd.get_dummies(df[['Sex', 'Embarked', 'Title']], drop_first=True)



# Adding the results to the master dataframe

df1 = pd.concat([df, dummy1], axis=1)
df1.head()
# We have created dummies for the below variables, so we can drop them

df1 = df1.drop(['Embarked','Sex', 'Title'], 1)
df1 = df1.drop(['Family_name','Room'], 1)
df1 = df1.drop(['Ticket', 'Name'], 1)
df1 = df1.drop(['Ticket_categorical'], 1)
df1 = df1.drop(['Ticket_numerical'], 1)
df1 = df1.drop(['FamilyID'], 1)
df1.head()
df1 = df1.drop(['FamilySize', 'Singleton','SmallFamily','LargeFamily'], 1)
df1.info()
df1.columns
df1['Fare_pp']=df1.Fare/df1.FamilySize_new  #Fare per person

df1=df1.drop('Fare', axis=1)
df1_test=df1[df1.Survived=='NA']

df1_test = df1_test.drop('Survived', 1)





df1_train=df1[df1.Survived!='NA']

df1_train.Survived=df1_train.Survived.apply(pd.to_numeric)
#Scale the numerical variables

from sklearn.preprocessing import StandardScaler



std_scaler = StandardScaler()

df1_train[['Age','Fare_pp']] = std_scaler.fit_transform(df1_train[['Age','Fare_pp']])  #Fit and transform





df1_test[['Age','Fare_pp']] = std_scaler.transform(df1_test[['Age','Fare_pp']] )
plt.figure(figsize = (22, 10))

sns.heatmap(df1_train.corr(), annot = True, cmap="YlGnBu")

plt.show()



#correlation going. If it were regresion would have to manage multi-collinearity
#Implort ML libraries

from sklearn.model_selection import train_test_split, KFold, cross_val_score # to split the data

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, fbeta_score #To evaluate our model

from sklearn import metrics

from imblearn.metrics import sensitivity_specificity_support

#from imblearn.over_sampling import SMOTE



from sklearn.model_selection import GridSearchCV



# Algorithmns models to be compared

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from xgboost import XGBClassifier
#Creating the X and y variables - Using test data

X = df1_train.drop(['Survived','PassengerId'], 1)

y = df1_train["Survived"]



# Spliting X and y into train and test version

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)




clf = RandomForestClassifier().fit(X, y)



pred_class = clf.predict(X)

pred_prob=clf.predict_proba(X)[:,1]



sensitivity, specificity, _ = sensitivity_specificity_support(y, pred_class, average='binary')



print('AUC',round(metrics.roc_auc_score(y, pred_prob),2))

print('f1', round(metrics.f1_score(y, pred_class),2))

print('Accuracy', round(metrics.accuracy_score(y, pred_class),2))

print('Sensi',round(sensitivity,2))

print('speci',round(specificity),2)



cm=confusion_matrix(y, pred_class)

print(cm)







features = pd.DataFrame()

features['feature'] = X.columns

features['importance'] = clf.feature_importances_

features.sort_values(by=['importance'], ascending=True, inplace=True)

features.set_index('feature', inplace=True)



features.plot(kind='barh', figsize=(25, 25))
features.tail(15)
#Important feature - I have removed Male as title should be sufficient



feature_col=['Age','Fare_pp','Title_Mr','Pclass','FamilySize_new', 'Title_Miss','Title_Mrs','Deck_N', 'Ticket_cat' ]
[feature_col]
#Lets plot the heatmap with just feature col

temp=feature_col.copy()

temp.append('Survived') #Add Survived 

plt.figure(figsize = (22, 10))

sns.heatmap(df1_train[temp].corr(), annot = True, cmap="YlGnBu")

plt.show()


# to feed the random state

seed = 7



# prepare models

models = []

models.append(('LR', LogisticRegression()))

models.append(('LDA', LinearDiscriminantAnalysis()))

models.append(('KNN', KNeighborsClassifier()))

models.append(('CART', DecisionTreeClassifier()))

models.append(('NB', GaussianNB()))

models.append(('RF', RandomForestClassifier()))

models.append(('XGB', XGBClassifier()))

models.append(('SVM', SVC(probability=True, gamma='auto')))



# evaluate each model in turn

results = []

names = []

scoring = 'accuracy'



for name, model in models:

        kfold = 5 #KFold(n_splits=3, random_state=seed)

        cv_results = cross_val_score(model, X, y, cv=kfold, scoring=scoring) #[feature_col]

        results.append(cv_results)

        names.append(name)

        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

        print(msg)

        

        

        

# boxplot algorithm comparison

fig = plt.figure(figsize=(11,6))

fig.suptitle('Algorithm Comparison - Accuracy')

ax = fig.add_subplot(111)

plt.boxplot(results)

ax.set_xticklabels(names)

plt.show()



[feature_col]
#Run corss validation on unbalance data

# to feed the random state

seed = 7



# prepare models

models = []

models.append(('LR', LogisticRegression()))

models.append(('LDA', LinearDiscriminantAnalysis()))

models.append(('KNN', KNeighborsClassifier()))

models.append(('CART', DecisionTreeClassifier()))

models.append(('NB', GaussianNB()))

models.append(('RF', RandomForestClassifier()))

models.append(('XGB', XGBClassifier()))

models.append(('SVM', SVC(probability=True, gamma='auto')))



# evaluate each model in turn

results = []

names = []

scoring = 'accuracy'



for name, model in models:

        kfold = 5 #KFold(n_splits=3, random_state=seed)

        cv_results = cross_val_score(model, X[feature_col], y, cv=kfold, scoring=scoring) #[feature_col]

        results.append(cv_results)

        names.append(name)

        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

        print(msg)

        

        

        

# boxplot algorithm comparison

fig = plt.figure(figsize=(11,6))

fig.suptitle('Algorithm Comparison - Accuracy')

ax = fig.add_subplot(111)

plt.boxplot(results)

ax.set_xticklabels(names)

plt.show()



from sklearn.pipeline import Pipeline

from sklearn.decomposition import PCA

from imblearn.metrics import sensitivity_specificity_support

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report, confusion_matrix
# Tune Logistic reg trained on balanced data with  class_weight estimator

logistic = LogisticRegression(class_weight='balanced')



# create pipeline

steps = [("logistic", logistic)        ]



# compile pipeline

logistic = Pipeline(steps)



# hyperparameter space

params = {'logistic__C': [0.03,0.05,0.07,0.1,0.2,0.3,0.4, 0.5, 1, 2, 3, 4], 'logistic__penalty': ['l1', 'l2']}



# create 5 folds

folds = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 4)



# create gridsearch object

LR_bal_tuned = GridSearchCV(estimator=logistic, cv=folds, param_grid=params, scoring='accuracy')



LR_bal_tuned.fit(X, y) #_reduced



# print best hyperparameters

print("Best Accuracy: ", LR_bal_tuned.best_score_)

print("Best hyperparameters: ", LR_bal_tuned.best_params_)
# Tune Logistic reg trained on balanced data with  class_weight estimator

logistic = LogisticRegression(class_weight='balanced')



# create pipeline

steps = [("logistic", logistic)        ]



# compile pipeline

logistic = Pipeline(steps)



# hyperparameter space

params = {'logistic__C': [0.03,0.05,0.07,0.1,0.2,0.3,0.4, 0.5, 1, 2, 3, 4], 'logistic__penalty': ['l1', 'l2']}



# create 5 folds

folds = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 4)



# create gridsearch object

LR_bal_red_tuned = GridSearchCV(estimator=logistic, cv=folds, param_grid=params, scoring='accuracy')



LR_bal_red_tuned.fit(X[feature_col], y) #_reduced



# print best hyperparameters

print("Best Accuracy: ", LR_bal_red_tuned.best_score_)

print("Best hyperparameters: ", LR_bal_red_tuned.best_params_)
# Tune Logistic reg trained on smote balanced data 

rf = RandomForestClassifier(class_weight='balanced')



# create pipeline

#steps = [("rf", rf)        ]



# compile pipeline

#rf_tune = Pipeline(steps)



# hyperparameter space

params = {

                 'max_depth' : [4, 6, 8],

                 'n_estimators': [50, 10],

                 'max_features': ['sqrt', 'auto', 'log2'],

                 'min_samples_split': [2, 3, 10],

                 'min_samples_leaf': [1, 3, 10],

                 'bootstrap': [True, False],

                 }



#{'n_estimators': [500,550,600,650],

# 'min_samples_split': [2,3,4],

# 'min_samples_leaf': [2,3,4],

# 'max_depth': [100,105,110,115]

#    }







# create 5 folds

folds = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 4)



# create gridsearch object

rf_bal_tuned= GridSearchCV(estimator=rf, cv=folds, param_grid=params, scoring='accuracy', n_jobs = -1, verbose = 2)



rf_bal_tuned.fit(X, y)

parameters = rf_bal_tuned.best_params_



# print best hyperparameters

print("Best Accuracy: ", rf_bal_tuned.best_score_)

print("Best hyperparameters: ", rf_bal_tuned.best_params_)
# Tune Logistic reg trained on smote balanced data 

rf = RandomForestClassifier(class_weight='balanced')



# create pipeline

#steps = [("rf", rf)        ]



# compile pipeline

#rf_tune = Pipeline(steps)



# hyperparameter space

params = {

                 'max_depth' : [4, 6, 8],

                 'n_estimators': [50, 10],

                 'max_features': ['sqrt', 'auto', 'log2'],

                 'min_samples_split': [2, 3, 10],

                 'min_samples_leaf': [1, 3, 10],

                 'bootstrap': [True, False],

                 }



#{'n_estimators': [500,550,600,650],

# 'min_samples_split': [2,3,4],

# 'min_samples_leaf': [2,3,4],

# 'max_depth': [100,105,110,115]

#    }







# create 5 folds

folds = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 4)



# create gridsearch object

rf_bal_red_tuned= GridSearchCV(estimator=rf, cv=folds, param_grid=params, scoring='accuracy', n_jobs = -1, verbose = 2)



rf_bal_red_tuned.fit(X[feature_col], y)

parameters = rf_bal_red_tuned.best_params_



# print best hyperparameters

print("Best Accuracy: ", rf_bal_red_tuned.best_score_)

print("Best hyperparameters: ", rf_bal_red_tuned.best_params_)
print(parameters)# = {'bootstrap': True, 'max_depth': 8, 'max_features': 'log2', 'min_samples_leaf': 1, 'min_samples_split': 3, 'n_estimators': 50}






params = {

 'max_depth':range(3,10,2),

 'min_child_weight':range(1,10,2)

}

xg_bal_tuned = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=5,

 min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,

 objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27), 

 param_grid = params, scoring='accuracy',n_jobs=4,iid=False, cv=5)





xg_bal_tuned.fit(X,y)

#gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_



# print best hyperparameters

print("Best Accuracy: ", xg_bal_tuned.best_score_)

print("Best hyperparameters: ", xg_bal_tuned.best_params_)







params = {

 'max_depth':range(3,10,2),

 'min_child_weight':range(1,6,2)

}

xg_bal_red_tuned = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=5,

 min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,

 objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27), 

 param_grid = params, scoring='accuracy',n_jobs=4,iid=False, cv=5)





xg_bal_red_tuned.fit(X[feature_col],y)

#gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_



# print best hyperparameters

print("Best Accuracy: ", xg_bal_red_tuned.best_score_)

print("Best hyperparameters: ", xg_bal_red_tuned.best_params_)
param_grid = {'C': [0.1, 1, 10, 100, 1000],  

              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 

              'kernel': ['rbf']}  

  

svm_bal_tuned = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3) 



svm_bal_tuned.fit(X, y) 



# print best hyperparameters

print("Best Accuracy: ", svm_bal_tuned.best_score_)

print("Best hyperparameters: ", svm_bal_tuned.best_params_)









param_grid = {'C': [0.1, 1, 10, 100, 1000],  

              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 

              'kernel': ['rbf']}  

  

svm_bal_red_tuned = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3) 



svm_bal_red_tuned.fit(X[feature_col], y) 



# print best hyperparameters

print("Best Accuracy: ", svm_bal_red_tuned.best_score_)

print("Best hyperparameters: ", svm_bal_red_tuned.best_params_)
df2_test=df1_test.drop('PassengerId',axis=1)
#Run model trained on full data



# prepare models

models = []



models.append(('LR_bal_tuned', LR_bal_tuned))

models.append(('RF_tuned', rf_bal_tuned))

models.append(('XG_tuned', xg_bal_tuned))

models.append(('SVM_tuned', svm_bal_tuned))





for name, model in models:

        prediction = model.predict(df2_test)

        Test['Survived_Predict']=prediction



        print(Test.Survived_Predict.sum()/Test.Survived_Predict.count()) #Survivial Rate

            

        output = pd.DataFrame({'PassengerId': Test.PassengerId, 'Survived': Test.Survived_Predict})

        output.to_csv(name+'.csv', index=False)

        

print("Your submission was successfully saved!")

        

        
#Run model trained on reduced feature data



# prepare models

models = []

models.append(('LR_red__tuned', LR_bal_red_tuned))

models.append(('RF_red_tuned', rf_bal_red_tuned))

models.append(('XG_red_tuned', xg_bal_red_tuned))

models.append(('SVM_red_tuned', svm_bal_red_tuned))





for name, model in models:

        prediction = model.predict((df2_test)[feature_col])  #predict on reduced feature test data

        Test['Survived_Predict']=prediction



        print(Test.Survived_Predict.sum()/Test.Survived_Predict.count()) #Survivial Rate

            

        output = pd.DataFrame({'PassengerId': Test.PassengerId, 'Survived': Test.Survived_Predict})

        output.to_csv(name+'.csv', index=False)

        

print("Your submission was successfully saved!")

        

        
break1
#overwrite Mahcine learning model with our family based prediction of women and children

family_survival_pred.head()
#Check where machine learning and family based classification disgaree- use right merge

temp1=pd.merge(Test,family_survival_pred , on='PassengerId', how='right')

temp1[temp1.Survived_Predict!=temp1.Survived] #8 chnages to ML model using family survival prediction

#Do complete merge using left join

temp2=pd.merge(Test,family_survival_pred , on='PassengerId', how='left')

temp2['Survived_Predict']=np.where(temp2.Survived.notna(), temp2.Survived, temp2.Survived_Predict) 

#Overwite from family based prediction where is it NOT null

temp2.head()
temp2.info()
temp2.Survived_Predict=temp2.Survived_Predict.astype(int)
temp2.Survived_Predict.sum()/temp2.Survived_Predict.count()

#Survivial Rate
output = pd.DataFrame({'PassengerId': temp2.PassengerId, 'Survived': temp2.Survived_Predict})

output.to_csv('submission_Family+svm_bal_red_tuned_05092020.csv', index=False)

print("Your submission was successfully saved!")
#So there is slight improvment over my best mahcine learning model. 0.00957 increase to be exact => 0.00957*418=4

#to be exact from the 8 people prediction we overwrote ML model. So it means Family model had 6 correct and 2 incorrect.
