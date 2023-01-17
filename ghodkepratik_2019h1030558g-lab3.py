# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/eval-lab-3-f464/train.csv')

df.head()
df.describe()
df.describe(include='object')
df.info()
df['TotalCharges'].replace(' ', np.nan, inplace = True)


df['TotalCharges'].fillna('0', inplace = True) 

df
gender = {'Male':0,'Female':1}

df['gender'] = df['gender'].map(gender)

Married = {'No':0,'Yes':1}

df['Married'] = df['Married'].map(Married)

Children = {'No':0,'Yes':1}

df['Children'] = df['Children'].map(Children)

TVConnection = {'No':0,'Cable':1,'DTH':2}

df['TVConnection'] = df['TVConnection'].map(TVConnection)

Channel1 = {'No':0,'Yes':1,'No tv connection':2}

df['Channel1'] = df['Channel1'].map(Channel1)

Channel2 = {'No':0,'Yes':1,'No tv connection':2}

df['Channel2'] = df['Channel2'].map(Channel2)

Channel3 = {'No':0,'Yes':1,'No tv connection':2}

df['Channel3'] = df['Channel3'].map(Channel3)

Channel4 = {'No':0,'Yes':1,'No tv connection':2}

df['Channel4'] = df['Channel4'].map(Channel4)

Channel5 = {'No':0,'Yes':1,'No tv connection':2}

df['Channel5'] = df['Channel5'].map(Channel5)

Channel6 = {'No':0,'Yes':1,'No tv connection':2}

df['Channel6'] = df['Channel6'].map(Channel6)

Internet = {'No':0,'Yes':1}

df['Internet'] = df['Internet'].map(Internet)

HighSpeed = {'No':0,'Yes':1,'No internet':2}

df['HighSpeed'] = df['HighSpeed'].map(HighSpeed)

AddedServices = {'No':0,'Yes':1}

df['AddedServices'] = df['AddedServices'].map(AddedServices)

Subscription = {'Monthly':0,'Biannually':1,'Annually':2}

df['Subscription'] = df['Subscription'].map(Subscription)

PaymentMethod = {'Cash':0,'Credit card':1,'Net Banking':2,'Bank transfer':3}

df['PaymentMethod'] = df['PaymentMethod'].map(PaymentMethod)

df
df.corr()
from sklearn.cluster import KMeans

X = np.array(df.drop(['Satisfied','custId','SeniorCitizen','Married','Children','Internet','HighSpeed','tenure','MonthlyCharges'], 1).astype(float))
y = np.array(df['Satisfied'])
kmeans = KMeans(n_clusters=2) 

kmeans.fit(X)

kmeans.labels_
correct = 0

for i in range(len(X)):

    predict_me = np.array(X[i].astype(float))

    predict_me = predict_me.reshape(-1, len(predict_me))

    prediction = kmeans.predict(predict_me)

    if prediction[0] == y[i]:

        correct += 1



print(correct/len(X))

df = pd.read_csv('/kaggle/input/eval-lab-3-f464/test.csv')

df.head()
df['TotalCharges'].replace(' ', np.nan, inplace = True)

df['TotalCharges'].fillna('0', inplace = True) 

df
gender = {'Male':0,'Female':1}

df['gender'] = df['gender'].map(gender)

Married = {'No':0,'Yes':1}

df['Married'] = df['Married'].map(Married)

Children = {'No':0,'Yes':1}

df['Children'] = df['Children'].map(Children)

TVConnection = {'No':0,'Cable':1,'DTH':2}

df['TVConnection'] = df['TVConnection'].map(TVConnection)

Channel1 = {'No':0,'Yes':1,'No tv connection':2}

df['Channel1'] = df['Channel1'].map(Channel1)

Channel2 = {'No':0,'Yes':1,'No tv connection':2}

df['Channel2'] = df['Channel2'].map(Channel2)

Channel3 = {'No':0,'Yes':1,'No tv connection':2}

df['Channel3'] = df['Channel3'].map(Channel3)

Channel4 = {'No':0,'Yes':1,'No tv connection':2}

df['Channel4'] = df['Channel4'].map(Channel4)

Channel5 = {'No':0,'Yes':1,'No tv connection':2}

df['Channel5'] = df['Channel5'].map(Channel5)

Channel6 = {'No':0,'Yes':1,'No tv connection':2}

df['Channel6'] = df['Channel6'].map(Channel6)

Internet = {'No':0,'Yes':1}

df['Internet'] = df['Internet'].map(Internet)

HighSpeed = {'No':0,'Yes':1,'No internet':2}

df['HighSpeed'] = df['HighSpeed'].map(HighSpeed)

AddedServices = {'No':0,'Yes':1}

df['AddedServices'] = df['AddedServices'].map(AddedServices)

Subscription = {'Monthly':0,'Biannually':1,'Annually':2}

df['Subscription'] = df['Subscription'].map(Subscription)

PaymentMethod = {'Cash':0,'Credit card':1,'Net Banking':2,'Bank transfer':3}

df['PaymentMethod'] = df['PaymentMethod'].map(PaymentMethod)

df
X = np.array(df.drop(['SeniorCitizen','custId','Married','Children','Internet','HighSpeed','tenure','MonthlyCharges','PaymentMethod','AddedServices'], 1).astype(float))

X
kmeans = KMeans(n_clusters=2,max_iter=900)

kmeans.fit(X)
arr=[]

for i in range(len(X)):

    predict_me = np.array(X[i].astype(float))

    predict_me = predict_me.reshape(-1, len(predict_me))

    prediction = kmeans.predict(predict_me)

    arr.append(prediction[0])

    

print(arr)    
df['Satisfied'] = arr 

df1 = df[['custId','Satisfied']]

df1.head()
df1.to_csv(r'satisfied5.csv',index=False)
df2 = pd.read_csv('satisfied5.csv')

df2.head()