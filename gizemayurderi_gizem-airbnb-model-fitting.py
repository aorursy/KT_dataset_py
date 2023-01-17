# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from xgboost.sklearn import XGBClassifier

from datetime import datetime

from sklearn.preprocessing import LabelEncoder 

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train=pd.read_csv('../input/airbnb-recruiting-new-user-bookings/train_users_2.csv.zip')

test = pd.read_csv('../input/airbnb-recruiting-new-user-bookings/test_users.csv.zip')

#sessions=pd.read_csv('../input/airbnb-recruiting-new-user-bookings/sessions.csv.zip')

#countries=pd.read_csv('../input/airbnb-recruiting-new-user-bookings/countries.csv.zip')

#age_gender=pd.read_csv('../input/airbnb-recruiting-new-user-bookings/age_gender_bkts.csv.zip')
Id = 'id' # Submission'da kullanmak üzere id kısmını değişkene atayarak datamdan sileceğm.



mysubmission_ID = test.loc[:,Id]



train.drop(Id, axis=1, inplace=True)

test.drop(Id, axis=1, inplace=True)



# Test ve Train datasını işaretleyebilmek için Train kolonu oluşturdum.

train.loc[:,'Train'] = 1

test.loc[:,'Train'] = 0



# Tek bir df üzerinden gitmek için birleştiriyorum.



df = pd.concat([train,test], ignore_index=True)

df = df.drop(['date_first_booking'], axis=1)
df.info() # Boş değeri olan sütunları belirlemek ve data tiplerini anlamak için info'ya bakıyorum. 
train.info() 
test.info()
train.describe()
dft=df[df['Train']==1]
df.drop(dft[(dft['age']<=17) & (dft['country_destination']=='NDF')].sort_values(by=['age']).index, inplace=True)#18 yaşından küçük ve bir yere gitmemiş kişileri kaldırdım 
df.loc[df['age']<=17, 'age']=18 # 18 yaşından küçük ve bir yere gidebilmiş kişiler kaldı, gitmeyen herkesi datamdan kaldırdığım için. 

                                #Küçük görünen kişileri 18 yaşına getirdim.
def ageconverter(age):

    if (age<1997) & (age>1000):

        return (2014-age)

    else:

        return age
df['aged'] =df['age'].apply(ageconverter) #yeni bir sütuna aldım 'age' özelliğini

df.loc[df['aged']>=1000, 'aged']=np.nan # geride kalan ve yaşı 1997'den büyük kişilerin yaşlarını NaN yaptım.
df['aged'].describe() 
plt.figure(figsize=(30,15))

sns.countplot(x='aged', data=df)
df.loc[df['aged']>=85, 'aged']=np.nan #grafiğe bakınca 85'ten sonrasını almamaya karar verdim, onları da NaN ile doldurdum.
df['aged'].describe()
dft=df[df['Train']==1]

dft.groupby('affiliate_channel')['aged'].mean()
def agefiller(param):

    age = param[0]

    ac = param[1]

    if pd.isnull(age)==True:

        if ac == 'api':

            return 33

        elif ac== 'content':

            return 41

        elif ac== 'direct':

            return 36

        elif ac=='other':

            return 37

        elif ac== 'remarketing':

            return 40

        elif ac== 'sem-brand':

            return 39

        elif ac== 'sem-non-brand':

            return 40

        elif ac== 'seo':

            return 35

    else:

        return param[0]

      
df['aged'] =df[['aged', 'affiliate_channel']].apply(agefiller, axis=1)
df['aged'].describe()
df['aged']=df['aged'].astype(int) # sayıları integer'a çevirdim daha okunabilir olması için.

df = df.drop(['age'], axis=1)
df.isnull().sum() # test datası dışında hala boş hücresi olan 'first_affiliate_tracked' görünüyor.
dft=df[df['Train']==1]

dft.pivot_table(values='gender',index='affiliate_channel', columns='first_affiliate_tracked',aggfunc='count')
def fatfiller(param):

    fat= param[0]

    ac = param[1]

    if pd.isnull(fat)==True:

        if ac == 'seo':

            return 'linked'

        elif ac== 'sem-non-brand':

            return 'omg'

        else:

            return 'untracked'

    else:

        return param[0]
df['first_affiliate_tracked'] =df[['first_affiliate_tracked', 'affiliate_channel']].apply(fatfiller, axis=1)
df['first_browser'].replace('-unknown-',np.nan,inplace=True)
dft.pivot_table(values='language',index='first_browser', columns='first_device_type',aggfunc='count')
def fbfiller(param):

    fb= param[0]

    fdt = param[1]

    if pd.isnull(fb)==True:

        if fdt == 'Android Phone':

            return 'Android Browser'

        elif fdt =='Android Tablet':

            return 'IE'

        elif fdt =='SmartPhone (Other)':

            return 'BlackBerry Browser'

        elif fdt =='iPad':

            return 'Mobile Safari'

        elif fdt =='iPhone':

            return 'Mobile Safari'

        elif fdt =='Mac Desktop':

            return 'Safari'

        else:

            return 'Chrome'

    else:

        return param[0]
df['first_browser'] =df[['first_browser', 'first_device_type']].apply(fbfiller, axis=1) # fonksiyona göre datamı doldurdum.
first_active_date =[]

for i in df['timestamp_first_active']:

    d = datetime.strptime(str(i),'%Y%m%d%H%M%S')

    day_string = d.strftime('%Y-%m-%d')

    first_active_date.append(day_string)
first_active_time =[]

for i in df['timestamp_first_active']:

    d = datetime.strptime(str(i),'%Y%m%d%H%M%S')

    time_string = d.strftime('%H:%M:%S')

    first_active_time.append(time_string)

    
df['first_active_date']=first_active_date # timestamp_first_active'den first active date ve time'ı ayırarak ayrı iki sütun oluşturdum

df['first_active_time']=first_active_time

df=df.drop(['timestamp_first_active'], axis=1)
fad = np.vstack(df['first_active_date'].astype(str).apply(lambda x: list(map(int, x.split('-')))).values)

df['fad_year'] = fad[:, 0] # Yıl - ay - gün olarak ayırıyorum

df['fad_month'] = fad[:, 1]

df['fad_day'] = fad[:, 2]

df = df.drop(['first_active_date'], axis=1)
fad = np.vstack(df['first_active_time'].astype(str).apply(lambda x: list(map(int, x.split(':')))).values)

df['fad_hour'] = fad[:, 0] # saati ayrı bir sütuna aldım

df = df.drop(['first_active_time'], axis=1)
dac = np.vstack(df['date_account_created'].astype(str).apply(lambda x: list(map(int, x.split('-')))).values)

df['dac_year'] = dac[:, 0] #airbng hesabının ilk oluşturduğu yıl-ay-gün sütunları oluşturuyorum.

df['dac_month'] = dac[:, 1]

df['dac_day'] = dac[:, 2]

df = df.drop(['date_account_created'], axis=1)
df.select_dtypes("object").columns


le = LabelEncoder() 

df['signup_method']= le.fit_transform(df['signup_method']) 

df['language']= le.fit_transform(df['language'])

df['affiliate_channel']= le.fit_transform(df['affiliate_channel'])

df['affiliate_provider']= le.fit_transform(df['affiliate_provider'])

df['signup_app']= le.fit_transform(df['signup_app'])

df['first_device_type']= le.fit_transform(df['first_device_type'])

df['gender']= le.fit_transform(df['gender'])

df['first_browser']= le.fit_transform(df['first_browser'])

df['first_affiliate_tracked']= le.fit_transform(df['first_affiliate_tracked'])
df['country_destination'].replace('NDF',0, inplace=True)

df['country_destination'].replace('US',1, inplace=True)

df['country_destination'].replace('other',2, inplace=True)

df['country_destination'].replace('FR',3, inplace=True)

df['country_destination'].replace('CA',4, inplace=True)

df['country_destination'].replace('GB',5, inplace=True)

df['country_destination'].replace('ES',6, inplace=True)

df['country_destination'].replace('IT',7, inplace=True)

df['country_destination'].replace('PT',8, inplace=True)

df['country_destination'].replace('DE',9, inplace=True)

df['country_destination'].replace('NL',10, inplace=True)

df['country_destination'].replace('AU',11, inplace=True)
dft=df[df['Train']==1] #train datasını oluşturmak için train olanları alıyorum

X=dft.drop(['country_destination','Train'], axis=1) #feature tanımlama

Y = dft['country_destination'] # target tanımlama
x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.3, random_state = 42)  #data ayırma
from sklearn.tree import DecisionTreeClassifier
dtree=DecisionTreeClassifier()

dtree.fit(x_train,y_train)

predictions=dtree.predict(x_test)

print(confusion_matrix(y_test, predictions))

print(classification_report(y_test, predictions))
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(criterion='entropy', max_depth= 8, max_leaf_nodes=30, min_samples_leaf=30, n_estimators= 100, random_state=0)



rfc.fit(x_train, y_train)

prediction = pd.DataFrame(data=rfc.predict(x_test), index = x_test.index)

print(classification_report(y_test, prediction))

xgb = XGBClassifier(max_depth=5, learning_rate=0.2, n_estimators=50,

                    objective='multi:softprob', subsample=0.6, colsample_bytree=0.6, seed=0, silent=0)                  



xgb.fit(x_train, y_train)
y_predicted=xgb.predict(x_test)

print(confusion_matrix(y_test, y_predicted))

print(classification_report(y_test, y_predicted))
pred_country={0:'NDF' ,1:"US", 2:"other", 3:"FR", 4:"CA", 5:"GB", 6:"ES", 7:"IT", 8:"PT", 9:"DE", 10:"NL", 11:"AU"}
dftest=df[df['Train']==0]# submission yapmak için hazırlık

testX=dftest.drop(['country_destination','Train'],axis=1) # submission yapmak için hazırlık

tested = xgb.predict(testX)# submission yapmak için hazırlık
results=[]

for i in tested:

    results.append(pred_country[i])

print(results)
len(tested) #Kontrol

len(mysubmission_ID)
len(mysubmission_ID)
my_submission = pd.DataFrame({'id': mysubmission_ID, 'country':results})

my_submission.to_csv('submission.csv', index=False)