# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Reading the datasets.

codebook=pd.read_excel('/kaggle/input/customer-purchase-journey-netherlands/codebook.xlsx')

data=pd.read_csv('/kaggle/input/customer-purchase-journey-netherlands/TravelData.csv')

demo=pd.read_csv('/kaggle/input/customer-purchase-journey-netherlands/TravelDataDemos.csv')
#Chcking few columns.

pd.set_option('display.max_colwidth', -1)

codebook.head()
#Merging the two major datasets.

master=pd.merge(demo,data)

master.head(1)
#Changing columns name from dutch to english.

master.rename(columns={'SPSS_Regio5':'Region','RESP_GEM_GROOTTE':'Size_of_Manucipality','BAS_huishoudgrootte':'Household_Size','GenderID':'Gender','BAS_werkzaamheid_resp':'Kind_of_work','BAS_bruto_jaarinkomen':'Gross_income','afg_kinderen_huishouden':'no_of_children','AFG_sk2015':'Social_Class','BAS_voltooide_opleiding8_resp':'Education','SPSS_Lifestage':'Lifestage'},inplace=True)

demo.rename(columns={'SPSS_Regio5':'Region','RESP_GEM_GROOTTE':'Size_of_Manucipality','BAS_huishoudgrootte':'Household_Size','GenderID':'Gender','BAS_werkzaamheid_resp':'Kind_of_work','BAS_bruto_jaarinkomen':'Gross_income','afg_kinderen_huishouden':'no_of_children','AFG_sk2015':'Social_Class','BAS_voltooide_opleiding8_resp':'Education','SPSS_Lifestage':'Lifestage'},inplace=True)

master.head(1)
#Checking the relation between different features of customers.

d=demo.corr()

plt.figure(figsize=(15,10))

sns.heatmap(d,linewidth=.02,cmap="YlGnBu",annot=True)

plt.show()
master['TIMESPSS']=pd.to_datetime(master['TIMESPSS'])

master['DayOfWeekNum'] = master['TIMESPSS'].dt.dayofweek

master['week_name'] = master['TIMESPSS'].dt.weekday_name

master['Day'] = master['TIMESPSS'].dt.day

master['Hour'] = master['TIMESPSS'].dt.hour

master['Month'] = master['TIMESPSS'].dt.month

master.head()
#Count of device used by customers with respect to gender.

mylabels=['Male','Female']

lines=sns.countplot(data=master,x='DEVICE_TYPE',hue='Gender')

plt.legend(labels=mylabels)

plt.ylabel('Count of Device used')

plt.show()
#Binning the touchpoints into Customer Initiated and Firm Initiated Touchpoints.

master['touch']=pd.cut(master['type_touch'],[0,16,22],right=False,labels=['CIT ','FIT'])
#Quering to get only the rows where purchase happened and initiated by Counsumer.

m1=master.query('purchase_any==1 or purchase_own==1 and touch=="CIT"')
plt.figure(figsize=(10,7))

ax=sns.countplot(data=master,x='Hour')

sns.set(style="darkgrid",color_codes=True)

ax.set(xlabel='Count Of Purchases', ylabel='Hour',title='Traffic on online websites on different hours of a day')

plt.show()
plt.figure(figsize=(10,7))

ax=sns.countplot(data=master,x='week_name',order=master['week_name'].value_counts().index)

sns.set(style="darkgrid",color_codes=True)

ax.set(xlabel='Count Of Purchases', ylabel='Weekday',title='Traffic on online websites on different days of a week')

plt.show()
plt.figure(figsize=(10,7))

ax=sns.countplot(data=m1,x='Region',palette='Set3',order=master['Region'].value_counts().index)

sns.set(style="darkgrid")

ax.set(xlabel='Count Of Purchases', ylabel='Region',title='CIT who got converted with respect to region')

plt.xticks(np.arange(5),('Amsterdam, Rotterdam, Den Haag','West','North','East','South'))

plt.xticks(rotation=90)

plt.show()
sns.set(style="darkgrid", palette="Set1", color_codes=True)

plt.figure(figsize=(10,5))

ax=sns.countplot(x='Gross_income',data=m1,hue='Gender',order = master['Gross_income'].value_counts().index)

ax.set(xlabel='Count Of Purchases', ylabel='Region',title='Customers who got converted with respect to income and gender')

plt.legend(labels=mylabels)

plt.show()
newfig=m1.groupby('Lifestage')[['UserID']].count().reset_index()

explode = (0.09,0.09,0.09,0.09,0.09,0.09,0.09,0.09,0.05) 

plt.figure(figsize=(10,7))

plt.pie(newfig['Lifestage'],autopct='%1.1f%%', startangle=90, pctdistance=0.85,shadow=True,explode=explode)

plt.title='% of customers who bought the products in different Lifestages'

centre_circle = plt.Circle((0,0),0.6,fc='white')

fig=plt.gcf()

fig.gca().add_artist(centre_circle)

plt.legend(newfig['Lifestage'])

plt.axis('equal')

plt.title='% of customers who bought the products in different Lifestages'

plt.show()
sns.set(style="darkgrid", palette="magma_r", color_codes=True)

plt.figure(figsize=(10,5))

b=sns.countplot(x='Education',data=m1,hue='Gender',order=m1['Education'].value_counts().index)

b.set(ylabel='Count of customers', xlabel='Education',title='Most common education background of customer who bought the product with respect to gender.')

plt.legend(labels=mylabels)

plt.show()
prange=m1.groupby('type_touch')[['UserID']].count().sort_values(by='UserID',ascending=False).head().reset_index()

explode = (0.08, 0.08, 0.08,0.08,0.08)

plt.figure(figsize=(10,5))

ax=plt.pie(prange['UserID'], explode=explode, labels=['Accomodations Website','Touroperator / Travel agent Website Competitor','Touroperator / Travel agent Website Focus brand','Information / comparison Website','Flight tickets Website'],

autopct='%1.1f%%',shadow=True, startangle=90) 

plt.title='Most sucessfull touchpoints'

plt.gca().legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.tight_layout()

plt.show()
sns.set( palette="tab20c", color_codes=True)

plt.figure(figsize=(10,5))

a=sns.countplot(x='purchase_any',data=master,hue='touch')

a.set(xlabel='Conversion_status', ylabel='Count',title='Ratio of customers who purchased and not purchased with reference to touchtype')

plt.xticks(np.arange(2),('Not Purchased','Purchased'))

plt.show()
master.isnull().sum()
#Nan values were there in every column so dropped them.

master.dropna(inplace=True)
from sklearn.preprocessing import LabelEncoder

lb=LabelEncoder()

master['DEVICE_TYPE']=lb.fit_transform(master['DEVICE_TYPE'])
X=master.drop(columns=['UserID','PurchaseID','Duration','purchase_own','purchase_any','TIMESPSS'])

Y=master['purchase_any']
from sklearn.preprocessing import MinMaxScaler

mm=MinMaxScaler()

X_data=mm.fit_transform(X)

X_df=pd.DataFrame(X_data,columns=X.columns)
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error,r2_score

from sklearn.model_selection import train_test_split

#Training and testing

x_train,x_test,y_train,y_test=train_test_split(X_df,Y,test_size=0.3,random_state=42)

print(len(x_train),len(x_test))
#Getting regression and prediction

reg = LinearRegression()

reg.fit(x_train,y_train)

y_pred = reg.predict(x_test)
#Checking RMSE and R-score.

rmse=np.sqrt(mean_squared_error(y_test,y_pred))

r2=r2_score(y_test,y_pred)

print('RMSE =',rmse)

print('R-squared =',r2*100,'%')