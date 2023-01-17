import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns 

%matplotlib inline



from sklearn.model_selection import train_test_split
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
RANDOM_SEED = 42
!pip freeze > requirements.txt
#df_train = pd.read_csv('main_task.csv')

#df_test = pd.read_csv('kaggle_task.csv')

#sample_submission = pd.read_csv('sample_submission.csv')

df_train = pd.read_csv('/kaggle/input/kaggle-sf-dst-through-1/main_task.csv/main_task.csv')

df_test = pd.read_csv('/kaggle/input/kaggle-sf-dst-through-1/kaggle_task.csv')

sample_submission = pd.read_csv('/kaggle/input/kaggle-sf-dst-through-1/sample_submission.csv/sample_submission.csv')
df_train.info()
df_test.info()
sample_submission.info()
df_train['sample'] = 1 

df_test['sample'] = 0 

df_test['Rating'] = 0 
data = df_test.append(df_train, sort=False).reset_index(drop=True)
data.info()
#del data['Name']
data.sample(5)
data.Reviews[1]
data['Number_of_Reviews_isNAN'] = pd.isna(data['Number of Reviews']).astype('uint8')

data['Number of Reviews'].fillna(data['Number of Reviews'].mean(), inplace=True)
data.nunique(dropna=False)
#data = pd.get_dummies(data, columns=[ 'City',], dummy_na=True)
import re



data['Cuisine Style']=data['Cuisine Style'].fillna('No')

lcs = data['Cuisine Style'].tolist()

y=[]

for i in lcs:

    y.append(re.findall('\w[A-Za-z\s]+\w',i))



cus_count=[]

for i in y:

    cus_count.append(len(i))

data['Cusine count']=cus_count



set_set=set()

for i in y:

    for e in i:        

        set_set.add(e)



for i in set_set:

    data[i]=0

    for e in range(0,len(data)):

        if i in y[e]:

            data[i][e]=1
a = data['City'].unique()

a = list(a)

cap ={}

for i in a:

    cap[i]=1



cap['Barcelona'] = 0

cap['Munich'] = 0

cap['Oporto'] = 0

cap['Milan'] = 0

cap['Lyon'] = 0

cap['Hamburg'] = 0

cap['Krakow'] = 0

cap['Geneva'] = 0

cap['Zurich'] = 0

z=[]

for i in data['City']:

    for k,v in cap.items():

        if i == k:

            z.append(v)

cap

new_cap=cap

new_cap

new_cap['Paris']='French'

new_cap['Stockholm']='Swedish'

new_cap['London']='British'

new_cap['Berlin']='German'

new_cap['Munich']='German'

new_cap['Oporto']='Portuguese'

new_cap['Milan']='Italian'

new_cap['Bratislava']='Slovenian'

new_cap['Vienna']='Austrian'

new_cap['Rome']='Italian'

new_cap['Barcelona']='Spanish'

new_cap['Madrid']='Spanish'

new_cap['Dublin']='Irish'

new_cap['Brussels']='Belgian'

new_cap['Zurich']='Swiss'

new_cap['Warsaw']='Polish'

new_cap['Budapest']='Hungarian'

new_cap['Copenhagen']='Danish'

new_cap['Amsterdam']='Dutch'

new_cap['Lyon']='French'

new_cap['Hamburg']='German'

new_cap['Lisbon']='Portuguese'

new_cap['Prague']='Czech'

new_cap['Oslo']='Norwegian'

new_cap['Helsinki']='Finnish'

new_cap['Edinburgh']='Scottish'

new_cap['Geneva']='Swiss'

new_cap['Ljubljana']='Slovenian'

new_cap['Athens']='Greek'

new_cap['Luxembourg']='Mediterranean'

new_cap['Krakow']='Polish'



ct = data['City'].tolist()



x=[]

for i in new_cap.keys():

    x.append(i)

cousines={}

for i in range(0,len(y)):

    cousines[ct[i]]=y[i]

    

nation_count=[]

for i in data['City']:

    if new_cap[i] in cousines[i]:

        nation_count.append(1)

    else:

        nation_count.append(0)

data['Nationlity cousine']=nation_count
data['Price Range']=data['Price Range'].fillna('$$ - $$$')

pr=[]

for i in data['Price Range']:

    if i == '$$ - $$$':

        pr.append(2)

    elif i == '$$$$':

        pr.append(3)

    elif i == '$':

        pr.append(1)

data['Price Range']=pr 
lis={}

for i in y:

    for e in i:

        if e in lis:

            lis[e]+=1

        else:

            lis[e]=0

data['Restaurant_id']= data['Restaurant_id'].str.split('_').apply(lambda x:x[1])

data['Restaurant_id']= [float(x) for x in data['Restaurant_id']]



a = data['City'].unique()

a = list(a)

cap ={}

for i in a:

    cap[i]=1



cap['Barcelona'] = 0

cap['Munich'] = 0

cap['Oporto'] = 0

cap['Milan'] = 0

cap['Lyon'] = 0

cap['Hamburg'] = 0

cap['Krakow'] = 0

cap['Geneva'] = 0

cap['Zurich'] = 0

z=[]

for i in data['City']:

    for k,v in cap.items():

        if i == k:

            z.append(v)

data['Capitals']= z
from datetime import datetime





data['Reviews']=data['Reviews'].fillna('No')



cst=data['Reviews'].tolist()

p=[]

for i in cst:

    p.append(re.findall('\d\d\D\d\d\D\d\d\d\d',i))

new_st=[]        

for i in p:

    u=[]

    if i == []:

        u.append(datetime.today())

    else:

        for e in i:

            if e[2]=='/':

                u.append(datetime.strptime(e,'%m/%d/%Y'))            

            elif e[4]=='-':

                u.append(datetime.strptime(e,'%Y-%m-%d'))

            elif e[2] == '.':

                u.append(datetime.strptime(e,'%d.%m.%Y'))

    new_st.append(u)



last_time = []

for i in new_st:

    last_time.append(max(i))

last_time



z=[]    

for i in last_time:    

    z.append((datetime.today()-i).days)

data['Delta time revie']=z
data = pd.get_dummies(data, columns=[ 'City',], dummy_na=True)
data.head()

data.info()
data.drop(['Restaurant_id','ID_TA',], axis = 1, inplace=True)

data.drop([ 'URL_TA','Reviews','Cuisine Style',], axis = 1, inplace=True)
train_data = data.query('sample == 1').drop(['sample'], axis=1)



y = train_data.Rating.values  

X = train_data.drop(['Rating'], axis=1)





X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)

train_data.shape, X.shape, X_train.shape, X_test.shape
from sklearn.ensemble import RandomForestRegressor 

from sklearn import metrics 
model = RandomForestRegressor(n_estimators=100, verbose=1, n_jobs=-1, random_state=RANDOM_SEED)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
test_data = data.query('sample == 0').drop(['sample'], axis=1)

test_data = test_data.drop(['Rating'], axis=1)
predict_submission = model.predict(test_data)
sample_submission['Rating'] = predict_submission

sample_submission.to_csv('submission.csv', index=False)

sample_submission.head()