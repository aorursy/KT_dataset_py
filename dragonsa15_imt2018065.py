import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import datetime
from sklearn import preprocessing 
alcdata = pd.read_csv("../input/iiitb-ai511ml2020-assignment-1/Assignment/alcoholism/student-mat.csv")
fifadata = pd.read_csv("../input/iiitb-ai511ml2020-assignment-1/Assignment/fifa18/data.csv")
accidata1 = pd.read_csv("../input/iiitb-ai511ml2020-assignment-1/Assignment/accidents/accidents_2005_to_2007.csv")
accidata2 = pd.read_csv("../input/iiitb-ai511ml2020-assignment-1/Assignment/accidents/accidents_2009_to_2011.csv")
accidata3 = pd.read_csv("../input/iiitb-ai511ml2020-assignment-1/Assignment/accidents/accidents_2012_to_2014.csv")
alcdata.isnull().sum()
alcdata.info()
#Function to get a sorted correlation plot, based on the target column specified ( decreasing)
def CorrPlotLargest(df, target):
    k = 10
    numerical_feature_columns = list(df._get_numeric_data().columns)
    cols = df[numerical_feature_columns].corr().nlargest(k, target)[target].index
    cm = df[cols].corr()
    plt.figure(figsize=(10,6))
    return sns.heatmap(cm, annot=True, cmap = 'viridis')

#Function to get a sorted correlation plot, based on the target column specified (increasing)
def CorrPlotSmallest(df, target):
    k = 10
    numerical_feature_columns = list(df._get_numeric_data().columns)
    cols = df[numerical_feature_columns].corr().nsmallest(k-1, target)[target].index
    cols = cols.insert(0,target)
    cm = df[cols].corr()
    plt.figure(figsize=(10,6))
    return sns.heatmap(cm, annot=True, cmap = 'viridis')

alcdata['G_avg'] = (alcdata['G1'] + alcdata['G2'] + alcdata['G3'])/3

#extracting features
alcdata_features = alcdata.iloc[:,:-4]

#extracting target
alcdata_target = alcdata.iloc[:,-1]

#drawing a correlation matrix for the numeric data or non object data sorted on the basis of max correlation with g_avg
subplot1 = CorrPlotLargest(alcdata,'G_avg')

#drawing a correlation matrix for the numeric data or non object data sorted on the basis of negative correlation with g_avg
subplot2 = CorrPlotSmallest(alcdata,'G_avg')

#As the amount of ppl in same age is different lets compare expectation value of marks in each age gp
ExpectedMarks_With_Age = alcdata.groupby('age').apply(lambda x : x['G_avg']/len(x)).groupby('age').sum()
sns.jointplot(ExpectedMarks_With_Age.index,ExpectedMarks_With_Age.values)
ExpectedMarks_With_Age[20] 
alcdata[alcdata['age'] == 20]
rel = ['Medu','Fedu','studytime']
for col in rel:
    ex = alcdata.groupby(col)['G_avg'].mean()
    #print(col)
    #print(ex.index,ex.values)
    sns.jointplot(ex.index,ex.values)
    
 
alcdata.describe()
obj_alcdata = alcdata_features.select_dtypes(include=['object']).copy()
#obj_alcdata

onehot_objects = pd.get_dummies(obj_alcdata)
onehot_objects
#These are all the categorical items
#for col in onehot_objects.columns:
means = []

#print(alcdata.iloc[onehot_objects[onehot_objects[col] == 1].index]['G_avg'].sum())
for col in onehot_objects.columns:
    #print(alcdata.iloc[onehot_objects[onehot_objects[col] == 1].index]['G_avg'].sum(),len(onehot_objects[col]))
    means.append(alcdata.iloc[onehot_objects[onehot_objects[col] == 1].index]['G_avg'].sum()/len(onehot_objects[onehot_objects[col]==1]))

print(len(means))
plt.figure(figsize = (20,8))
g = sns.barplot(data=onehot_objects,ci=None)
plt.xticks(rotation = 'vertical')
plt.ylabel("Probability of occurence of a particular categorical gp")
#onehot_objects[onehot_objects[onehot_objects.columns] == 1].sum()[onehot_objects.columns[1]]/len(onehot_objects[onehot_objects.columns[1]])

for index, row in alcdata.iterrows():
    g.text(index,onehot_objects[onehot_objects[onehot_objects.columns] == 1].sum()[onehot_objects.columns[index]]/len(onehot_objects[onehot_objects.columns[index]]), round(means[index],1), color='black', ha="center")
#A plot of probabilities of the entity being in one of the following categories
le = preprocessing.LabelEncoder()
cols = alcdata.nunique()
new_cols = []
other_cols = []


for col in alcdata.columns:
        if alcdata.nunique()[col] == 2:
            new_cols.append(col)
        else:
            if(alcdata[col].dtype == 'object'):
                other_cols.append(col)
le = preprocessing.LabelEncoder()
for col in new_cols:
    alcdata[col] = le.fit_transform(alcdata[col].astype(str))

new_obj_data = pd.get_dummies(alcdata[other_cols])

new_obj_data

alcdata = alcdata.merge(new_obj_data,on = alcdata.index)

alcdata.drop(other_cols,inplace = True)

alcdata
alcdata['Pstatus'] = alcdata['Pstatus'].apply(lambda x: x.replace('A','1')).apply(lambda x: x.replace('T','0')).astype(int)
rel_alcdata = alcdata[['famrel','Pstatus','G3']].groupby('famrel')


final_val = rel_alcdata['Pstatus'].sum()/rel_alcdata['Pstatus'].size()
final_val
sns.barplot(final_val.index,final_val.values)
plt.ylabel('PStatus')
#enter code/answer in this cell. You can add more code/markdown cells below for your answer. 

#enter code/answer in this cell. You can add more code/markdown cells below for your answer.
#alcdata_features.dtypes
#alcdata_features['health']
#sns.barplot(alcdata_target.index,alcdata_target['G3'].values)
#plt.figure(figsize = (15,5))
#alcdata_features.mean()
possible_skews = alcdata.loc[:, alcdata.dtypes != 'object']
abs(possible_skews.kurtosis()) > 0.5
sns.pairplot(data = possible_skews)

#Features which show that they have a skew when saw kurtosis
#   traveltime,failures,famrel,Dalc,Walc,absences

plt.show()


#possible_skews['absences'].apply(lambda x: np.power(x,0.02)).apply(lambda x:x - possible_skews['absences'].mean()).hist(bins = 50,figsize = (5,5))
#sns.boxplot(alcdata_features.absences,ax=ax[1])
plt.hist(possible_skews['absences'].apply(lambda x: np.power(x,0.45)),bins=40)

#Since cotegoricl data don;t come from the normal distribution therefore there is no concept of skew in these variables
#Thus the only variable here witha skew is absence/ all others either are categorical / have minimal kurtosis / 
#Graph doesnt show much skew
#plt.hist(np.power(alcdata.Walc,1),bins=40)


possible_skews['absences'].apply(lambda x: np.power(x,0.45)).kurtosis()

possible_skews.hist(bins = 50,figsize = (15,15))
plt.show()
sns.pairplot(alcdata)
fifadata.iloc[fifadata['Value'].apply(lambda x: x.replace('€','')).apply(lambda x: x.replace('M','e06')).apply(lambda x: x.replace('K','e03')).astype(float).sort_values(ascending = False).index[10:]][['Name','Value','Release Clause','Club']][fifadata['Club'] == "FC Bayern München"].head(100)
#fifadata['Release Clause'].apply(lambda x: x.replace('€','')).apply(lambda x: x.replace('M','e06')).apply(lambda x: x.replace('K','e03')).astype(float)
#fifadata['Release Clause'] = fifadata['Release Clause'].apply(lambda x: x.replace('€','')).apply(lambda x: x.replace('M','e06')).apply(lambda x: x.replace('K','e03')).astype(float)

#remove_nan = fifadata.copy()
null_data = fifadata[fifadata['Release Clause'].isnull()]
non_null_data = fifadata[fifadata['Release Clause'].notnull()]
non_null_data['Release Clause'] = non_null_data['Release Clause'].apply(lambda x: x.replace('€','')).apply(lambda x: x.replace('M','e06')).apply(lambda x: x.replace('K','e03')).astype(float)
gped_data = non_null_data.groupby('Overall')
#non_null_data.head()
gped_data = gped_data['Release Clause'].mean()
#gped_data
#null_data['Release Clause'] = gped_data[null_data['Overall']].values
#null_data['Release Clause']
#fifadata[fifadata['Release Clause'].isnbull()].apply(lambda x: x)
#fifadata["Release Clause"].fillna(null_data["Release Clause"], inplace=True)

#gped_data[88]

#fifadata
#CorrPlotLargest(remove_nan,'Release Clause')
#So i guess International Reputation and Overall are pretty good standards to get the release clause

#So my intution for the most economical club would be one which has enough money for it to have 
#highest release clause when summed over all the players which yields us the following list


fifadata['Release Clause'].replace(np.nan,'€0.0M',inplace=True)
#fifadata[fifadata['Release Clause'] == '€0.0M']
testdata = fifadata.copy()
testdata.head()
testdata.head()
testdata['Release Clause'] = testdata['Release Clause'].apply(lambda x: x.replace('€','')).apply(lambda x: x.replace('M','e06')).apply(lambda x: x.replace('K','e03')).astype(float)
testdata['Value'] = testdata['Value'].apply(lambda x: x.replace('€','')).apply(lambda x: x.replace('M','e06')).apply(lambda x: x.replace('K','e03')).astype(float)
testdata['Wage'] = testdata['Wage'].apply(lambda x: x.replace('€','')).apply(lambda x: x.replace('M','e06')).apply(lambda x: x.replace('K','e03')).astype(float)

null_data['Release Clause'] = gped_data[null_data['Overall']].values
#null_data['Release Clause']
#fifadata[fifadata['Release Clause'].isnull()].apply(lambda x: x)
#fifadata["Release Clause"].fillna(null_data["Release Clause"], inplace=True)

testdata[testdata['Release Clause'] == 0]['Release Clause'] = null_data['Release Clause']
#testdata.head()
club_data = testdata[['Wage','Value','Release Clause','Club']].groupby('Club')
AmountPerClub = club_data['Release Clause'].apply(lambda x: x.sum()) - club_data['Value'].apply(lambda x: x.sum()) - club_data['Wage'].apply(lambda x: x.sum())   
AmountPerClub
final_values = AmountPerClub.apply(lambda x: (x - AmountPerClub.values.min())/AmountPerClub.values.max())
final_values =AmountPerClub.sort_values(ascending = False)
fig, axs = plt.subplots(3,1,figsize=(17,15))
fig.tight_layout()
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=1.5)
axs[0].tick_params(axis='x', rotation=90)
sns.barplot(final_values[:100].index,final_values[:100].values,ax=axs[0])
axs[1].tick_params(axis='x', rotation=90)
sns.barplot(final_values[100:200].index,final_values[100:200].values,ax=axs[1])
axs[2].tick_params(axis='x', rotation=90)
sns.barplot(final_values[200:300].index,final_values[200:300].values,ax=axs[2])

fifadata['Club'].isnull().sum()
corrMatrix = testdata[['Age','Potential','Value']].corr()
#plt.figure(figsize = (20,20))
sns.heatmap(corrMatrix, annot=True)
plt.show()


#fifadata['Age'].isnull().sum()
gp_data = fifadata[['Age','Potential']].groupby('Age')
fv = gp_data['Potential'].max()
sns.barplot(fv.index,fv.values)
sns.jointplot(fifadata['Potential'],fifadata['Age'])
sns.jointplot(testdata['Age'],testdata['Value'])
fifadata.columns
#'Acceleration','SprintSpeed', 'Agility' ,  'Stamina'
fifadata['Acceleration'].fillna(fifadata['Acceleration'].median(),inplace = True)
fifadata['SprintSpeed'].fillna(fifadata['SprintSpeed'].median(),inplace = True)
fifadata['Agility'].fillna(fifadata['Agility'].median(),inplace = True)
fifadata['Stamina'].fillna(fifadata['Stamina'].median(),inplace = True)
fifadata[['Acceleration','SprintSpeed', 'Agility' ,  'Stamina']].isnull().sum()
#fifadata[['SprintSpeed','Age']].groupby('Age').mean()
testdata = fifadata.copy()
#fifadata['Wage'].isnull().sum()
plt.figure(figsize = (30,4))
testdata['Wage'] = testdata['Wage'].apply(lambda x: x.replace('€','')).apply(lambda x: x.replace('M','e06')).apply(lambda x: x.replace('K','e03')).astype(float)

#fifadata.info()
CorrPlotLargest(testdata,'Potential')
#CorrPlotSmallest(testdata,'Potential')

cols = testdata.columns
fig,ax = plt.subplots(5,1,figsize = (20,20))
sns.barplot(testdata['Overall'],testdata['Potential'],ax = ax[0])
sns.barplot(testdata['Reactions'],testdata['Potential'],ax = ax[1])
sns.barplot(testdata['Composure'],testdata['Potential'],ax = ax[2])
sns.barplot(testdata['International Reputation'],testdata['Potential'],ax = ax[3])
#sns.barplot(testdata['Special'],testdata['Potential'],ax = ax[4])

cols = testdata.columns
fig,ax = plt.subplots(5,1,figsize = (20,20))
sns.barplot(testdata['BallControl'],testdata['Potential'],ax = ax[0])
sns.barplot(testdata['Skill Moves'],testdata['Potential'],ax = ax[1])
sns.barplot(testdata['LongPassing'],testdata['Potential'],ax = ax[2])
sns.barplot(testdata['Dribbling'],testdata['Potential'],ax = ax[3])
CorrPlotLargest(testdata,'Wage')
#CorrPlotSmallest(testdata,'Wage')

#enter code/answer in this cell. You can add more code/markdown cells below for your answer. 
club_data = fifadata[['Age','Club']].groupby('Club')
young_age = 21
ageVclub = club_data['Age'].apply((lambda x: len(x[x<=young_age])))
final_values = ageVclub.sort_values(ascending = False)
fig, axs = plt.subplots(3,1,figsize=(17,18))
fig.tight_layout()
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=2)
axs[0].tick_params(axis='x', rotation=90)
sns.barplot(final_values[:100].index,final_values[:100].values,ax=axs[0])
axs[1].tick_params(axis='x', rotation=90)
sns.barplot(final_values[100:200].index,final_values[100:200].values,ax=axs[1])
axs[2].tick_params(axis='x', rotation=90)
sns.barplot(final_values[200:300].index,final_values[200:300].values,ax=axs[2])
plt.show()
print(accidata1.columns == accidata2.columns)
print(accidata2.columns == accidata1.columns)
print(accidata2.columns == accidata3.columns)
print(accidata3.columns == accidata2.columns)
accidata = pd.concat([accidata1,accidata2,accidata3])
#Since on initial analysis it displayed itself as a entirely null column
accidata = accidata.drop('Junction_Detail',axis = 1)
#len(accidata)
#accidata.isnull().sum()
#accidata.info()




#Again the number of null values in Longitude and Latitude seemed to be pretty low so I just straight away removed them 
#so that it doesn't cause trouble in later stages
accidata = accidata.dropna(axis=0,thresh = 28)

#Removing the null values of those variable that are very hard to predict and are pretty low
accidata = accidata[accidata['Special_Conditions_at_Site'].notna()]
#accidata = accidata[accidata['LSOA_of_Accident_Location'].notna()]

#Creating a new category unknown for categorical variable that have a high amount of nulll values
accidata['LSOA_of_Accident_Location'] = np.where(accidata['LSOA_of_Accident_Location'].isnull(),"Unknown_location",accidata['LSOA_of_Accident_Location'])
accidata['Junction_Control'] = np.where(accidata['Junction_Control'].isnull(),"Unknown_Junction",accidata['Junction_Control'])

#Filling with the max imputed value
accidata['Pedestrian_Crossing-Human_Control'] = accidata['Pedestrian_Crossing-Human_Control'].fillna('None within 50 metres')
accidata['Pedestrian_Crossing-Physical_Facilities'] = accidata['Pedestrian_Crossing-Physical_Facilities'].fillna('No physical crossing within 50 meters')

#Removing all the rest of the null values
accidata = accidata.dropna()

#Setting date to be a python timestamp
accidata['Date'] = accidata['Date'].apply(lambda x: time.mktime(datetime.datetime.strptime(str(x),"%d/%m/%Y").timetuple()))

#Very high correlation with date so can be removed
accidata = accidata.drop('Year',axis = 1)

accidata.notna().sum()
test = accidata['Accident_Severity']
#enter code/answer in this cell. You can add more code/markdown cells below for your answer. 
accidata['LSOA_of_Accident_Location'].value_counts()
accidata['Junction_Control'].value_counts()

plt.figure(figsize = (15,15))
corrMatrix = accidata.corr()
sns.heatmap(corrMatrix, annot=True)
#Checking for skew
accidata[['Location_Easting_OSGR']]
accidata['Location_Easting_OSGR'].kurtosis()
accidata[['Location_Northing_OSGR']]
accidata['Location_Northing_OSGR'].kurtosis()

#Location Northing Shows a bit of skew so updating it to not have skew
accidata['Location_Northing_OSGR'] = accidata['Location_Northing_OSGR'].apply(lambda x: np.log(x))

#accidata
#removing longitude and latitude as they are the same as the other 2 and above we have update the skew in both and stored
#here in this column
accidata = accidata.loc[:, accidata.columns != 'Longitude']
accidata = accidata.loc[:, accidata.columns != 'Latitude']
accidata
#Location
accidata['Location_Easting_OSGR'].apply(lambda x : np.power(x,1)).hist(bins = 1000,figsize = (5,5))
pd.set_option('display.max_columns', 50)
accidata.head(150)
#Pretty much a simple group by, sum and then sort
Weekday_Deaths = accidata.groupby('Day_of_Week')['Number_of_Casualties'].sum()
Weekday_Deaths = Weekday_Deaths.sort_values(ascending = False)
labels = ['Mon','Tue','Wed','Thur','Fri','Sat','Sun']
g = sns.barplot(Weekday_Deaths.index,Weekday_Deaths.values,order = Weekday_Deaths.index)



#Here you can change the 'Road_Type' column to see how exactly that feature changes speed limit.
MaxSpeedLimit_Week = accidata.groupby(['Day_of_Week','Road_Type'])['Speed_limit'].max()
MinSpeedLimit_Week = accidata.groupby(['Day_of_Week','Road_Type'])['Speed_limit'].min()

fig, axs = plt.subplots(14,1,figsize=(15,50))

for i in range(1,8):
    if((i-1)%2 == 0):
        sns.barplot(MaxSpeedLimit_Week[i].index,MaxSpeedLimit_Week[i].values,ax = axs[i-1])
        axs[i-1].set_ylabel('Max Speed on Week Day' + str(i))
        sns.barplot(MinSpeedLimit_Week[i].index,MinSpeedLimit_Week[i].values,ax = axs[i])
        axs[i].set_ylabel('Min Speed on Week Day' + str(i))

for i in range(1,8):
    if((i-1)%2 != 0):
        sns.barplot(MaxSpeedLimit_Week[i].index,MaxSpeedLimit_Week[i].values,ax = axs[7 + i-1])
        axs[7 + i-1].set_ylabel('Max Speed on Week Day' + str(i))
        sns.barplot(MinSpeedLimit_Week[i].index,MinSpeedLimit_Week[i].values,ax = axs[7 + i])
        axs[7 + i].set_ylabel('Min Speed on Week Day' + str(i))

val = accidata['Light_Conditions'].value_counts()
sns.barplot(val.index,val.values)
plt.xticks(rotation = 'vertical')
val = accidata['Weather_Conditions'].value_counts()
sns.barplot(val.index,val.values)
plt.xticks(rotation = 'vertical')
Severity_Light = accidata.groupby('Accident_Severity')['Light_Conditions'].value_counts()
fig,axs = plt.subplots(2,1,figsize = (10,10))
#fig.tight_layout()
#plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=2)
severity_index = [item[0] for item in Severity_Light.index] 
light_index = [item[1] for item in Severity_Light.index] 
sns.barplot(severity_index[:-10],Severity_Light.values[:-10],light_index[:-10],ax = axs[0])
sns.barplot(severity_index,Severity_Light.values,light_index,ax = axs[1])

plt.xticks(rotation = 90)
plt.figure(figsize = (20,5))
CorrPlotLargest(accidata,'Accident_Severity')
accidata_categorical = accidata[accidata.columns[accidata.dtypes == object]]
target = accidata['Accident_Severity']
#accidata_categorical
df = pd.merge(accidata_categorical,target,on = accidata_categorical.index)
gped_data = df.groupby('Accident_Severity')
fig,axs = plt.subplots(len(accidata_categorical.columns),1,figsize = (10,10))
fig.tight_layout()
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=2)
key = 0
for i in accidata_categorical.columns:
    val = gped_data[i].value_counts()
    sns.barplot(val.index,val.values,ax = axs[key])
    key += 1
plt.show()
#Severity_Light = accidata_categorical.groupby('Road_Type')['Light_Conditions'].value_counts()
#fig,axs = plt.subplots(2,1,figsize = (10,10))
#fig.tight_layout()
#plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=2)
#severity_index = [item[0] for item in Severity_Light.index] 
#light_index = [item[1] for item in Severity_Light.index] 
#sns.barplot(severity_index[:-10],Severity_Light.values[:-10],light_index[:-10],ax = axs[0])
#sns.barplot(severity_index,Severity_Light.values,light_index,ax = axs[1])

#plt.xticks(rotation = 90)
accidata['Number_of_Casualties'].isnull().sum()

#below is a set of data that needs to be removed
#accidata['Date'] = pd.to_datetime(val, errors='coerce', cache=False).strftime('%m/%d/%Y')
accidata['Date'] = accidata['Date'].apply(lambda x: time.mktime(datetime.datetime.strptime(str(x),"%d/%m/%Y").timetuple()))

#too much corelation with date
accidata = accidata.drop('Year',axis = 1)
#accidata = accidata.drop('Year',axis = 1)

#There is no need for index or time there is realy low correaltion with them
accidata = accidata.drop('Accident_Index',axis = 1)
accidata = accidata.drop('Time',axis = 1)

#Im using label encoder for all of the features
le = preprocessing.LabelEncoder()
cols = accidata.select_dtypes(include=['object']).columns
for col in cols:
    accidata[col] = le.fit_transform(accidata[col].astype(str))

accidata.info()
testdata = accidata.copy()
testdata1 = accidata.copy()
testdata2 = accidata.copy()

testdata[testdata['Accident_Severity'] < 3] = 0
testdata[testdata['Accident_Severity'] == 3] = 1
testdata['Accident_Severity'].value_counts()

testdata1[testdata1['Accident_Severity'] != 2] = 0
testdata1[testdata1['Accident_Severity'] == 2] = 1
testdata1['Accident_Severity'].value_counts()


testdata2[testdata2['Accident_Severity'] != 1] = 0
testdata2[testdata2['Accident_Severity'] == 1] = 1
testdata2['Accident_Severity'].value_counts()

#testdata1 = accidata[accidata['Accident_Severity'] < 3]
testdata['Accident_Severity'].value_counts()
#testdata2 = accidata[accidata['Accident_Severity'] < 3]
#accidata[accidata['Accident_Severity'] > 1] = 1
from sklearn.model_selection import train_test_split

testdata['Accident_Severity'].value_counts()
y = accidata['Accident_Severity']
X = accidata.loc[:, accidata.columns != 'Accident_Severity']
X_train,X_test,Y_train,Y_test = train_test_split(X,y,test_size=0.3)

X_scaled = preprocessing.scale(X_train)
len(Y_test3)
#enter code/answer in this cell. You can add more code/markdown cells below for your answer. 
from sklearn.linear_model import LogisticRegression

logreg1 = LogisticRegressionCV(cv=5, n_jobs = -1,random_state=0,multi_class = "multinomial").fit(X_scaled,Y_train)


X_test = preprocessing.scale(X_test)
X_test
logreg1.score(X_test,Y_test)




