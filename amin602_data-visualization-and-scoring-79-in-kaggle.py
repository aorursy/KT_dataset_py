x=1 # hello from the other side :|
import pandas as pd 
import numpy as np
import sklearn
import seaborn as sns 
import matplotlib.pyplot as plt 
%matplotlib inline 
from sklearn.linear_model import LogisticRegression
plt.rc("font", size=14)
sns.set(style="dark") #white background style for seaborn plots
sns.set(style="whitegrid", color_codes=True)#style="whitegrid"
training_data = pd.read_csv('../input/train.csv')
testing_data = pd.read_csv('../input/test.csv')
training_data.head()
training_data.head()
a,ax=plt.subplots(figsize=(9,9))
sns.heatmap(training_data.corr(),annot=True,linewidths=0.5,fmt='.1f',ax=ax)
#there is correlation between survived and (Pclass,Fare,parch,age)
training_data_length=len(training_data)# for splitting later to the origional state 
print('the length of the training data is :',training_data_length)
df=pd.concat(objs=[training_data,testing_data],axis=0).copy()
df=df.reset_index(drop=True)
df.head(2)
print('the Cabins unique values count is :',len(training_data['Cabin'].unique()))
print('the null values on Cabin equal:',pd.isnull(training_data['Cabin']).sum())
#filling the Fare nans(which is just one row)
df.loc[pd.isnull(df['Fare'])==True,'Fare']=df.Fare.median()

#converting male to 1 , female to 0
df.loc[df['Sex']=='male','Sex'],df.loc[df['Sex']=='female','Sex']=1,0

#renaming the data 'Cabins' from Characters to numbers

df.Cabin.fillna(8,inplace=True) 
df.loc[df.Cabin.str.contains('T',na=False),'Cabin']=8 #the only Cabin that have T in all of the data 

# from A to 1, from B to 2 , etc..
a=list('ABCDEFG') ; b=list(range(1,len(a)+1))  
for x in range(len(a)):
    df.loc[df.Cabin.str.contains(a[x],na=False),'Cabin']=b[x]
    
    
#reducing the Tickets unique vales         
print('the Ticket length befor the change is :',len(df.Ticket.unique()))
unique_tickets=df.Ticket.unique()
for x in range((len(df)-1)):
    unique_tickets=df.loc[x,'Ticket'].split(' ')
    if(len(unique_tickets)==1):
        
        if unique_tickets[0]=='1'or'2'or'3'or'4'or'5'or'6'or'7'or'8'or'9':
            unique_tickets=(unique_tickets[0][0]+unique_tickets[0][1]+unique_tickets[0][2])
            
        else:
            pass
        
    elif len(unique_tickets)==2:
        unique_tickets=unique_tickets[0]

    elif len(unique_tickets)==3 :
        unique_tickets=unique_tickets[0]
    df.loc[x,'Ticket']=unique_tickets

#chaging the 'Embarked' into numbers 
    
df['Embarked']=df['Embarked'].replace('S',np.int32(1))
df['Embarked']=df['Embarked'].replace('Q',np.int32(2))
df['Embarked']=df['Embarked'].replace('C',np.int32(3))


print('the Ticket length after the change is  :',len(df.Ticket.unique()))
print('how the Tickets numbers look like after the change :',unique_tickets[0:4])
print('the unique Cabin values are :',df['Cabin'].unique())
print('the unique Pclass values are :',df['Pclass'].unique())
df.head(2)
# Define get_title function to extract titles from passenger names
import re
def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1) # 1 to erase the space befor the title
    return ""

#testing it..
print(get_title('Braund, Mr. Owen Harris'))
#working

# Create a new feature 'Title' that contains the titles of passenger names


df['Title'] = df['Name'].apply(get_title)
del df['Name'] # delete the origional column 'Name' since we don't need it anymore 

df['Title'] = df['Title'].replace('Mlle', 'Miss')
df['Title'] = df['Title'].replace('Ms', 'Miss')
df['Title'] = df['Title'].replace('Mme', 'Mrs') 
df.groupby(['Sex','Pclass','Title']).aggregate(np.median).head(10)  # groub them by the features and get the median value
# so the title actually matters !
#( but survived changes the value when Pclass changes: as you can see Miss survival changed when she was on Pclass 1 and 2 )
# we have to search ages by using the 3 of them ('Sex','Pclass','Title')
# filling the Age nans by using the 3 features 
for a in df.Sex.unique():
        for b in df.Pclass.unique():
            for c in df.loc[(df['Sex']==a) & (df['Pclass']==b),'Title'].unique():
                # get the median Age of the not-null Ages that fit the condition 
                the_median=df.loc[(pd.notnull(df['Age'])) & (df['Pclass']==b) &
                                       (df['Sex']==a) & (df['Title']==c),'Age'].median()

               # set the median Age on the null Ages that fit the condition 
                df.loc[(pd.isnull(df['Age'])) & (df['Pclass']==b) & 
                                  (df['Sex']==a) &(df['Title']==c),'Age']=the_median

#filling the nans on the 'Embarked'
#df.loc[pd.isnull(df['Embarked'])]#61,829 # less than training_data_length(891) so the nulls on the training data 
# dropping them 
df=df.drop((df.index[pd.isnull(df['Embarked'])])) #61,829
training_data_length=training_data_length-2                 

#(df[pd.isnull(df['Age'])])#no nulls 
#df[pd.isnull(df['Embarked'])]#no nulls 
df.columns
for i in df.columns:
     print ('for the column %s :'%i,df[i].unique())
    
print('survived : ', df.Survived.unique())
print('Sex : ',df.Sex.unique())  # 1 for male , 0 for female 
print('Pclass : ',df.Pclass.unique())#Ticket class
print('SibSp :',df.SibSp.unique())# of siblings / spouses aboard the Titanic
print('Embarked : ',df.Embarked.unique()) # Port of Embarkation C = Cherbourg, Q = Queenstown, S = Southampton
print('Parch : ',df.Parch.unique())# of parents / children aboard the Titanic
print('Cabin :',df.Cabin.unique())
print('Age :',df.Age.unique())
#print(df['Fare']) # the ticket cost  # bad idea ... (alot of unique values)
# the only nans are the survived on the testing data 
fig,ax=plt.subplots(1,5,figsize=(17,5))
sns.barplot(df['Embarked'],df['Survived'],ci=False,ax=ax[0])
sns.barplot(df['SibSp'],df['Survived'],ci=False,ax=ax[1]);ax[1].set_ylabel('')
sns.barplot(df['Parch'],df['Survived'],ci=False,ax=ax[2]);ax[2].set_ylabel('')

df['relations']=df['SibSp']+df['Parch']
sns.barplot(df['relations'],df['Survived'],ci=False,ax=ax[3]);ax[3].set_ylabel('')

ax[4].hist2d(x=training_data['Survived'],y=training_data['Pclass']);ax[4].set_title('Survival and death desity due to Pclass')

print('so having relations have an effective percentage of the survival ')
df[:training_data_length].head(2)
fig,ax=plt.subplots(1,5,figsize=(18,4))
sns.barplot(df['Cabin'],df['Survived']==1,ci=False,ax=ax[0]); ax[0].set_title('who survived');ax[0].set_ylabel(' ')
sns.barplot(df['Cabin'],df['Survived']==0,ci=False,ax=ax[1]); ax[1].set_title('who didn\'t survive');ax[1].set_ylabel(' ')
#df.Sex.plot('hist')
ax[2].hist(df.Sex);ax[2].set_title('the sex count')
ax[3].hist(training_data.Survived);ax[3].set_title('the Survived count on training data ')
ax[4].scatter(df['Pclass'],df['Cabin'],s=200) # s : the size of the dotts
ax[4].set_xlabel('The Pclass');ax[4].set_ylabel('Cabins');ax[4].set_title('Pclass regression  in cabins')
fig,ax=plt.subplots(2,2,figsize=(6,6))
fig.suptitle('plotting the Sex vs (putted Cabins) vs survival')
sns.countplot('Cabin',data=df.loc[((df.Cabin!=8) & (df.Sex==0) & (df.Survived==0))],ax=ax[0,0]);ax[0,0].set_ylabel('didn\'t survive count');ax[0,0].set_xlabel('');ax[0,0].set_title('females')
sns.countplot('Cabin',data=df.loc[((df.Cabin!=8) & (df.Sex==1)& (df.Survived==0))],ax=ax[0,1]);ax[0,1].set_ylabel('');ax[0,1].set_xlabel('');ax[0,1].set_title('males')
sns.countplot('Cabin',data=df.loc[((df.Cabin!=8) & (df.Sex==0)& (df.Survived==1))],ax=ax[1,0]);ax[1,0].set_ylabel('Survival count');ax[1,0].set_xlabel('')
sns.countplot('Cabin',data=df.loc[((df.Cabin!=8) & (df.Sex==1)& (df.Survived==1))],ax=ax[1,1]);ax[1,1].set_ylabel('');ax[1,1].set_xlabel('')
# making a dataFrame to plot the people who didn't Survive
temp=df[:training_data_length]
temp2=temp.copy() # replace 0 to 1 and 1 to 0 for the plotting 
temp2['Survived']=temp2['Survived'].replace(0,np.int32(2))
temp2['Survived']=temp2['Survived'].replace(1,np.int32(0))
temp2['Survived']=temp2['Survived'].replace(2,np.int32(1))
temp2['Age_Classification']=(temp2.loc[:,'Age']/40).astype(int)
temp=df[:training_data_length] # i made it because there is nulls on the age for the training 
fig,ax=plt.subplots(1,4,figsize=(18,6))

a=sns.regplot(data=temp,x='Age',y='Survived',ci=False,order=5,ax=ax[0])
a.set(xlim=(0.43, 70),ylim=(0,1.1))
w=sns.barplot(temp['Age'],temp['Survived']==1,ci=None,ax=ax[1]) # survived !
w.set(ylabel='Survived')


w=sns.barplot(temp['Age'],temp['Survived']==0,ci=None,ax=ax[2])
w.set( ylabel='didn\'t survive')
a=sns.regplot(data=temp2,x='Age',y='Survived',ci=False,order=5,ax=ax[3])#,set_ylabel('dd'))
a.set_ylabel('didn\'t survive')
a.set(xlim=(0.43, 70),ylim=(0,1.1))
#a.legend(['survived','didn\'t survive'])# 
print(np.max(temp.Age))
print(np.min(temp.Age))
sns.distplot(temp['Age'][temp['Survived']==1],label='Survived', hist=False)
sns.distplot(temp['Age'][temp['Survived']==0], hist=False,label='Didn\'t survive')
grid =sns.FacetGrid(training_data,row='Survived',col='Sex',margin_titles=True) # FaceGrid : it makes the histograms ready in seaborn 
                                                # margin_titles : to make the titles show on like its a data frames ( on the side and top )
grid.map(plt.hist,'Pclass')
# is the titanic movie right about the captin death tho ? 
df[df['Title']=='Capt']
#the captin of the titanic didn't survive...
# well, that was sad lol 
#plt.scatter(training_data['Survived'],training_data['Fare'])
# and ladies and gentlements , we take from this plot that the people who payed about 500 pounds simply survived (at least in the training data )
# unhash code to see it with your eyes
dummies_titles = pd.get_dummies(df['Title'],prefix='Title')
df = pd.concat([df,dummies_titles],axis=1)


dummies_titles = pd.get_dummies(df['Cabin'],prefix='Cabin')
df = pd.concat([df,dummies_titles],axis=1)


dummies_titles = pd.get_dummies(df['Pclass'],prefix='Pclass')
df = pd.concat([df,dummies_titles],axis=1)


dummies_titles = pd.get_dummies(df['Embarked'],prefix='Embarked')
df = pd.concat([df,dummies_titles],axis=1)

# deleting the columns that we changed 
del df['Title'] ; del df['Cabin'] ; del df['Pclass'] ; del df['Embarked']
del df['PassengerId']

del df['Ticket']

training_data = df[:training_data_length].copy()
training_data=training_data.reset_index(drop=True)

testing_data = df[training_data_length:].copy()
testing_data=testing_data.reset_index(drop=True)

del testing_data['Survived'] # it is generated as nan by the first concat (while gathering the data )

print ('the length of the training data is ',len(training_data.columns)) # the normal columns and the "Survived" column
print (' the length of the testing data is ',len(testing_data.columns))
del training_data['Cabin_8']
del testing_data['Cabin_8']


training_x=training_data.copy()
del training_x['Survived']
training_y =training_data['Survived'].copy()

testing=testing_data.copy()
from sklearn.linear_model import LogisticRegression

logistic=LogisticRegression()
ss=logistic.fit(training_x,training_y)
result=logistic.predict(testing)

print(result[:3]) # ok,everything is good

df=pd.concat([pd.read_csv('../input/test.csv',usecols=[0]),pd.DataFrame({'Survived':result})],axis=1).set_index('PassengerId')
df.Survived=df.Survived.astype(int)

df.to_csv('result.csv')
# and from that we got 0.78947 on kaggle 
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression
logistic =LogisticRegression()

from sklearn.neighbors import KNeighborsClassifier
kNeibours=KNeighborsClassifier(n_neighbors=3)

from sklearn.tree import DecisionTreeClassifier
tree=DecisionTreeClassifier()

from sklearn.tree import DecisionTreeRegressor
regressor=DecisionTreeRegressor()

from sklearn.ensemble import RandomForestClassifier
forest=RandomForestClassifier()

from sklearn.linear_model import LinearRegression
linearRegression=LinearRegression()

from sklearn.svm import SVR
svr=SVR()

from sklearn.svm import SVC
svc=SVC()


logistic_scores=cross_val_score(logistic,training_x,training_y,cv=5)
kNeibours_scores=cross_val_score(kNeibours,training_x,training_y,cv=5)
tree_scores=cross_val_score(tree,training_x,training_y,cv=5)
forest_scores=cross_val_score(forest,training_x,training_y,cv=5)
linearRegression_scores=cross_val_score(linearRegression,training_x,training_y,cv=5)
svc_scores=cross_val_score(svc,training_x,training_y,cv=5)
svr_scores=cross_val_score(svr,training_x,training_y,cv=5)


print ('the logistic_scores accuracy score is ',np.average(logistic_scores))
print ('kNeibours_scores accuracy score is ',np.average(kNeibours_scores))
print ('the tree_scores accuracy score is ',np.average(tree_scores))
print ('the forest_scores accuracy score is ',np.average(forest_scores))
print ('the linearRegression_scores accuracy score is ',np.average(linearRegression_scores))
print ('the svc_scores accuracy score is ',np.average(svc_scores))
print ('the svr_scores accuracy score is ',np.average(svr_scores))
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression

logistic =LogisticRegression()
logistic = logistic.fit(training_x, training_y)

# print(logistic.transform)


a=logistic.coef_
b=training_x.columns
a[0]=np.abs(a[0]) # getting the absoulte values so we get the strong relations despite if it's a negative or positive relation

a=pd.DataFrame()
a['Title']=training_x.columns
a['Values']=logistic.coef_[0]
a.sort_values('Values')
a.set_index('Title', inplace=True)
a.sort_values(by=['Values'], ascending=True, inplace=True)
a.plot(kind='barh', figsize=(10,10))
# so it got the age as a really bad feature because it is basically not catagorical .... 
# so as you know the decision trees ADORE the catagorical data since it is build on such logic .... 
# it basically can't use the age because it will overfit when it uses it so that sucks for it :) 
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
clf = RandomForestClassifier(n_estimators=5)#, max_features='sqrt')
clf = clf.fit(training_x, training_y)

features = pd.DataFrame()
features['feature'] = training_x.columns
features['importance'] = clf.feature_importances_
features.sort_values(by=['importance'], ascending=True, inplace=True)
features.set_index('feature', inplace=True)

features.plot(kind='barh', figsize=(10,10))
from sklearn.linear_model import LogisticRegression

logistic=LogisticRegression()
ss=logistic.fit(training_x,training_y)
result=logistic.predict(testing)

print(result[:3]) # ok,everything is good



df=pd.concat([pd.read_csv('../input/test.csv',usecols=[0]),pd.DataFrame({'Survived':result})],axis=1).set_index('PassengerId')
df.Survived=df.Survived.astype(int)

df.to_csv('result.csv')
# and from that we got 0.78947 on kaggle 