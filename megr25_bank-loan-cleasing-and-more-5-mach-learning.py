#Elemental Library

import pandas as pd 

import numpy as np 

import math 



import re 

from scipy import stats

from scipy.stats import norm, skew

import string



#Visualization

import matplotlib.pyplot as plt 

import seaborn as sns 



import matplotlib.gridspec as gridspec

%matplotlib inline

sns.set_style('whitegrid')

import cufflinks as cf

cf.go_offline()

from IPython.display import display

from PIL import Image



import warnings

warnings.filterwarnings("ignore")
path="../input/loanimages/os.png"

display(Image.open(path))
train = pd.read_csv('../input/my-dataset/credit_train.csv')

print("----------Technical Information-------------")

print('Data Set Shape = {}'.format(train.shape))

print('Data Set Memory Usage = {:.2f} MB'.format(train.memory_usage().sum()/1024**2))

print("Data columns type""\n""{}".format(train.dtypes.value_counts()))

train.describe()
path="../input/loanimages/welcome.jpg"

display(Image.open(path))
# General View

print('Target Column is = {}'.format(train.columns[2]))

print('-------------')

print('Variable Target =\n{}'.format(train['Loan Status'].value_counts().index.to_list()))

print('-------------')

print('Columns in train are ={}'.format(train.columns.to_list()))



# Missing Values



train.isnull().sum()
print('Train_Data')

train.head(3)

missing = (train.isnull().sum()/ len(train))*100

Total = missing.drop(missing[missing==0].index).sort_values(ascending= False)

print("Missing Data in %\n")

print(Total)
print("This Result shows all the row with Nan Values but Im intersting in those\n which the entire row is nan" )

train[train.isna().any(axis=1)]
print("For a better Comprenhension I will drop the columns mentioned Above \nWe can easily identify that the missing Values in Credit Score - Annual Income matches by index")

train.drop(['Loan ID','Customer ID'], axis = 1 , inplace= True)



# Heat Map

fig, ax = plt.subplots(figsize=(12,7))

sns.heatmap(train.isnull(), yticklabels=False, cmap="viridis", cbar=False,ax=ax)

plt.show()
print(train.isnull().sum(axis=1).value_counts())

print('\n')

print("Look this, numbers of Nan values by row, in other words there are\n514 rows with 17 Missng Values\n1 row with 4 Missng Values\n...etc")

print('\n')

print("Total number of columns are 19, even though we can fill them by many method\nit will be complicated and they represent only 0.5% of the Total Data ")
# Finding number of Missing values by row 

Nan_values_row = train.isnull().sum(axis=1)



# Finding Index NAN values == 17 

index_NAN_Greater_17 = Nan_values_row[Nan_values_row==17].index





print("Here we have the 514 Row with Nan Values")

train.iloc[index_NAN_Greater_17][:5]  #Showing them 
train.drop(index_NAN_Greater_17,axis = 0 , inplace = True)

train.reset_index(drop=True)
t_object = train.dtypes[train.dtypes == 'object'].index

t_float = train.dtypes[train.dtypes == 'float'].index





def nan_col (data):

    for i in data:

        MV = train[i].isna().sum()

        if MV > 0:

            print(i,MV)



print("Here we have Missing Values by Columns - type = object")

nan_col(t_object)

print('\n')

print("Here we have Missing Values by Columns - type = float")

nan_col(t_float)

print('\n')

print("I will go Colum by Column from the Lowest to Greatest")
def corr_heat (frame):   #<---Heat Map

    correlation = frame.corr()

    f,ax = plt.subplots(figsize=(15,10))

    mask = np.triu(correlation)

    sns.heatmap(correlation, annot=True, mask=mask,ax=ax,cmap='viridis')

    bottom,top = ax.get_ylim()

    ax.set_ylim(bottom+ 0.5, top - 0.5)





# Heat Map

corr_heat(train)

print("I plotted a Heat map to find any relation among variables to start filling them out")
print("there is no a variable which has strong relation with Maximum Open Credit, there fore I will search for a person who has similar behaviour and fill the value")



train[train['Maximum Open Credit'].isna()][['Maximum Open Credit','Credit Score','Annual Income','Monthly Debt','Years of Credit History','Number of Open Accounts','Current Credit Balance']]
Value_1 =train[(train['Years of Credit History']==15.3) & (train['Number of Open Accounts'] == 3 )]['Maximum Open Credit'].mean()

Value_2 = train[(train['Credit Score']>=7030.0) & (train['Number of Open Accounts'] == 9 ) & (train['Years of Credit History'] >= 22 )]['Maximum Open Credit'].mean()



print("Ok 1. Pandas find someone who's Years of Credit History = 15.3,Open Accounts = 3 and then Tell me the Maximun open Credit average\nHere you are")

print(round(Value_1,1))

print('\n')

print("Ok 2. Pandas now find someone who's Years of Credit Score => 7030 ,Open Accounts = 9 , Years of Credit History = 9  then Tell me the Maximun open Credit average\nHere you are")

print(round(Value_2 ,1))



# then Filling by index 



train.loc[30180,'Maximum Open Credit'] =Value_1

train.loc[98710,'Maximum Open Credit'] =Value_2



print('\nFilling  Nan Values ................... Done..!!') 
print('Tax Liens has strong relation with Number of Credit Problems , this might help us to fill them out')

train.loc[train['Tax Liens'].isna()][['Number of Credit Problems','Annual Income','Tax Liens']]
print("Here something happened , Tax liens is imposed as a guarantee so people have to pay or they lose their houses, If they have had 1 or less Credit Problems We can assume that those 10 people didnt reach that point")

print("\nfill Nan Values in Tex liens with 0")

train['Tax Liens'].fillna(0,inplace=True)

print('\nFilling  Nan Values ................... Done..!!') 
print("Bankruptcies and Number Of credit Problems are related, Im almost sure that we will have the same escenario as in Tax lines")

train.loc[train['Bankruptcies'].isna()][['Number of Credit Problems','Annual Income','Tax Liens','Bankruptcies']].tail(10)
print("As I thought No Credit Problems , no Tax liens = No Bankruptcies")

print("\nfill Nan Values in Bankruptciess with 0")

train['Bankruptcies'].fillna(0,inplace=True)

print('\nFilling  Nan Values ................... Done..!!') 
train[train['Months since last delinquent'].isna()][['Credit Score','Months since last delinquent','Years of Credit History']]
print("please Notice that in USA 'Late payments generally wont end up on your credit reports for at least 30 days after you miss the payment. so you have almost 60 day to pay ")

print("and if you notice the Credit Score is greater that 700,meaning that they have good - Excellent credit score")

print("Which made me think of those Nan Values as 0 , I mean 0 month having a late payment")

print("\nfill Nan Values in Months since last delinquent with 0")

train['Months since last delinquent'].fillna(0,inplace=True)

print('\nFilling  Nan Values ................... Done..!!') 
# finding outliers

def Outliers (data,column):

    mean_ = data[column].mean()

    Sdev_= data[column].std()

    Upper_limit= mean_+ (3*Sdev_)

    lower_limit= mean_- (3*Sdev_) #error 

    out= data[(data[column]>Upper_limit)|(data[column]<lower_limit)].index

    

    data.drop(out ,inplace=True)

    



train.reset_index(drop=True)

train.reset_index(drop=True)
def find_ranges(data,column):

    count= data[column].count()

    max_ = max(data[column])

    min_ = min(data[column])

    element = math.trunc(np.sqrt(count))

    interval =math.trunc(max_/element)

    print("count=",count,"max=",max_,"min=",min_,"N-elements=",element,"Intervals=",interval)

    

find_ranges(train,'Annual Income')



#Range monthly debt 

lower_limit = np.arange(0,4621560.0,16867)

upper_limit = np.arange(16866.99,4638427.0,16867)





#Replacing Values

income = []

index = []

              

print("\n-------------Starting --------------\n")

for i, j in zip(lower_limit,upper_limit):

    value= round(train[(train['Annual Income'].notna())&(train['Monthly Debt']>=i)&(train['Monthly Debt']<=j)]['Annual Income'].mean(),2)

    income.append(value)

    ind = train[(train['Monthly Debt']>=i)&(train['Monthly Debt']<=j)&(train['Annual Income'].isna())]['Annual Income'].index

    index.append(ind)

    

    #print("From",i,"to",j,"Annual Income =",value,", Values to be replaced", ind[:5])

    

for i, j in zip (index,income):

    train.loc[i,'Annual Income']= j



print("\nReplacing Multiples Values in 'Annual Income' ..............Done")

print("\nThe following Rows couldnt be replaced because there were not data between the ranges...They will be dropped")

train[train['Annual Income'].isna()]

train.drop([11648,68650], inplace=True)
# Fixing Digits



def fixing (data,digits=3): 

    if data >0:

        new_credit_score = int(str(data)[:digits])

        return new_credit_score

    else:

        return data

    

#Fixing Digits

print("\n------- Before Replacing Values------\n")

print("Credit Score Min =",min(train['Credit Score']),"\nCredit Score Max=",max(train['Credit Score']),"\nCredit Score Average=",round(np.mean(train['Credit Score']),2))

train['Credit Score'].fillna(720, inplace = True)

train['Credit Score']=train['Credit Score'].apply(lambda x: fixing(x))

print("\n------- After Replacing Values------\n")

print(train['Credit Score'].describe())



# Extracting numbers



def extract_numbers (data):

    if data == str(data):

        text = [w for w in data if w in string.digits]

        years = int(''.join(text))

        return years



train['Years in current job']= train['Years in current job'].apply(lambda x: extract_numbers(x))

print('\nFollowing some information from the web the Average time in the same job in USA is between 4.2 and 5.6 years , I will use 5 to fill all missing values ')

train['Years in current job'].fillna(5,inplace=True)

print('\nFilling  Nan Values ................... Done..!!\n') 

missing = (train.isnull().sum()/ len(train))*100

Total = missing.drop(missing[missing==0].index).sort_values(ascending= False)

print("Missing Data in %\n")

print(Total)



train['Years in current job']=train['Years in current job'].astype(int)
print('Train_Data')

train.head(3)

train.reset_index(drop=True)
path="../input/loanimages/see.jpg"

display(Image.open(path))
train.drop(train[train['Current Loan Amount']==99999999.0].index, inplace=True)

train.reset_index(drop=True)
fig = plt.figure(constrained_layout=True, figsize=(20,10))

grid = gridspec.GridSpec(ncols=6, nrows=2, figure=fig)



#bar plot Horizontal

ax1 = fig.add_subplot(grid[0, :2])

ax1.set_title('Loan Status')

sns.countplot(y='Loan Status',hue ='Term',data=train, ax=ax1,) #Paid no paid



#bar plot Vertical

ax2 = fig.add_subplot(grid[1, :2])

ax2.set_title('Purpose segmented by Fully Paid/Charged Off')

bar = sns.barplot(x='Purpose', y='Current Loan Amount', hue = 'Loan Status',data=train, ax = ax2)

bar.set_xticklabels(bar.get_xticklabels(),  rotation=90, horizontalalignment='right') #fixing the Names



#box plot Credit Score

ax3 = fig.add_subplot(grid[:, 2])

ax3.set_title('Credit Score')

sns.boxplot(train.loc[:,'Credit Score'], orient='v', ax = ax3)





#box plot Monthly payment

ax4 = fig.add_subplot(grid[:,3])

ax4.set_title("Amount paid Monthly")

sns.boxplot(train['Monthly Debt'], orient='v' ,ax=ax4)



#Displot Distribution

ax5 = fig.add_subplot(grid[0, 4:6])

ax5.set_title("Amount borrowed 'Blue= fully Paid, red=Charged Off'")

#---> Segmenting fully Paid /Charge Off

full_paid = train[train['Loan Status']=="Fully Paid"]

charged_off = train[train['Loan Status']=="Charged Off"]

sns.distplot(full_paid['Current Loan Amount'], color = 'Blue' , rug=False, ax=ax5) 

sns.distplot(charged_off['Current Loan Amount'], color = 'Red',rug=False, ax=ax5) 



#Displot Distribution

ax6 = fig.add_subplot(grid[1, 4:6])

ax6.set_title("Annual Income 'Blue= fully Paid, red=Charged Off'")

#---> Segmenting fully Paid /Charge Off

short = train[train['Term']=='Short Term']

long = train[train['Term']=='Long Term']

sns.distplot(short['Current Loan Amount'], color = 'Blue' , ax=ax6) 

sns.distplot(long['Current Loan Amount'], color = 'Green', ax=ax6)



plt.show()



print("The Loan Status chart shows most of the fully payments are short term, if we take a look of the 'current loan amount chart' 'Blue=Short' show that most of the loan are between 0-200K\nComparing to 'long Term loan'which are from 200K to almost 6000K")

print("The Purpose chart show that 'Business Loan, Consildation,buy a house and Improvement' are the Top 4 for reason for loana application,However the purpose could not tell us if someone will pay in full or not")

print("\nMost of the applicants have credit Scores between 675 - 750 which can be Considered Fair~Good")

print("\nThe average payment recieved monthly is almost 200K")

print("\nThe Institution who borrows the money usually borrows arround 0-400K ")
from sklearn.preprocessing import OneHotEncoder



print("The following Columns have string Values\n")

t_object = train.dtypes[train.dtypes == 'object'].index

for i in t_object:

    a= train[i].nunique()

    print(i,a)



print("\n-------------Starting---------------\nEncoding with pd.get_dummies\n-------Done..!!")

for i in t_object:

    econ = pd.get_dummies(train[i], prefix =i, drop_first=True)

    train = pd.concat([train,econ],axis = 1)

    train.drop(i,axis=1,inplace=True)

    

train.head()
from sklearn.model_selection import train_test_split

from sklearn.feature_selection import VarianceThreshold, f_classif , f_regression, SelectKBest, SelectPercentile



#Defining 

X=train.drop('Loan Status_Fully Paid',axis=1)

y=train['Loan Status_Fully Paid']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)



# Removing Constant , QuasiConstant

CF = VarianceThreshold(threshold=0.01)

CF.fit(X_train)

X_train_ = CF.transform(X_train)

X_test_ = CF.transform(X_test)





X_data = VarianceThreshold(threshold = 0.01).fit_transform(X)

print("before Removing Constant and QuasiConstant values the data set has",X.shape)

print("after Removing The data set has",X_data.shape)
from sklearn import svm

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split,cross_val_predict

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LinearRegression, LogisticRegression

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import accuracy_score





MX=MinMaxScaler()

X_data =MX.fit_transform(X_data)



X_train, X_test, y_train, y_test = train_test_split(X_data, y, test_size=0.33, random_state=42)





# Model initilization

knn = KNeighborsClassifier()

tree_gin = DecisionTreeClassifier(criterion = 'gini')

tree_ent = DecisionTreeClassifier(criterion = 'entropy')

#svm_lin = svm.SVC(kernel = 'linear')

#svm_rbf = svm.SVC(kernel = 'rbf')

log = LogisticRegression()

rf = RandomForestClassifier()

xgb = XGBClassifier()



models = [knn,tree_gin,tree_ent,log,rf,xgb]#,svm_rbf,svm_lin]
def model_fit_predict(model,X_train, X_test, y_train, y_test):

    model = model.fit(X_train,y_train) #fitting

    y_pred = model.predict(X_test) #predicting

    model_acurracy=accuracy_score(y_test, y_pred) #Evaluating

    accuracy.append(model_acurracy)

    

accuracy=[]



for i in models:

    model_fit_predict(i,X_train, X_test, y_train, y_test)

    

#print(accuracy)



#Visulization for the best model 

model_eval = pd.DataFrame(accuracy, 

                          index=['knn','tree_gin','tree_ent','log','rf','xgb'],

                         columns= ['Accuracy'])

print('--- Accuracy Scores---')

model_eval.sort_values(by='Accuracy', ascending=False)
path="../input/loanimages/upvote.jpg"

display(Image.open(path))