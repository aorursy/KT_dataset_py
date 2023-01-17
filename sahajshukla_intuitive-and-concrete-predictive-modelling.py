

# importing essential libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import LabelEncoder #encoder

from sklearn.svm import SVC

from sklearn.preprocessing import OneHotEncoder # encoding to overcome the trap of assigning values according to category number

from scipy import stats

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import confusion_matrix

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVR

from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LinearRegression





df = pd.read_csv('../input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')    
encoder = LabelEncoder()

encoder1 = LabelEncoder()

df['status'] = encoder.fit_transform(df['status'])

df.groupby('ssc_b').mean()
df.groupby('gender').mean()
print(df.groupby('hsc_b').mean()) # grouped by HSC board

df.groupby('hsc_s').mean() # grouped by HSC Stream ie Science, comm and arts

df.groupby('degree_t').mean()
df.groupby('specialisation').mean()
df.iloc[:,1] = encoder1.fit_transform(df.iloc[:,1])

df.iloc[:,3] = encoder1.fit_transform(df.iloc[:,3])

df.iloc[:,5] = encoder1.fit_transform(df.iloc[:,5])

df.iloc[:,6] = encoder1.fit_transform(df.iloc[:,6])

df.iloc[:,8] = encoder1.fit_transform(df.iloc[:,8])

df.iloc[:,9] = encoder1.fit_transform(df.iloc[:,9])

df.iloc[:,11] = encoder1.fit_transform(df.iloc[:,11])
df = df.iloc[:,1:]

df_chance = df.iloc[:,:13]

enc = OneHotEncoder()

enc1 = OneHotEncoder()
array1 = df_chance.iloc[:,5].values

array2 = df_chance.iloc[:,5].values

enc.fit(array1.reshape(-1,1))

enc1.fit(array1.reshape(-1,1))

enc1.categories_

enc.categories_

array1 = enc.transform(array1.reshape(-1,1)).toarray()

array2 = enc.transform(array2.reshape(-1,1)).toarray()

df_new = pd.DataFrame(array1)

df_chance = df_chance.join(df_new)

df_chance = df_chance.drop(columns = 0, axis = 1)

df_chance = df_chance.rename(columns={1: 'new_1', 2: "new_2"})

df_new = pd.DataFrame(array2)

df_chance = df_chance.join(df_new)

df_chance = df_chance.drop(columns = 0, axis = 1)

df_chance = df_chance.drop(columns = 'hsc_s', axis = 1)

df_chance = df_chance.drop(columns = 'degree_t', axis = 1)

df_chance = df_chance.rename(columns={1: 'new_3', 2: "new_4"})

#df_chance.iloc[:,5] = enc.fit_transform(df_chance.iloc[:,5]).toarray()

result = df['status'].values

df_chance = df_chance.drop(columns = 'status', axis = 1)



scaler = StandardScaler()

df_chance = df_chance.values

df_chance[:,[1,3,5,7,9]] = scaler.fit_transform(df_chance[:,[1,3,5,7,9]])
clf = SVC(gamma = 'auto')



X_train = df_chance[:172, :]

X_test = df_chance[172:, :]

y_train = result[:172]

y_test = result[172:]



clf.fit(X_train, y_train)

clf1 = RandomForestClassifier(n_estimators = 200)

clf1.fit(X_train, y_train)

preds = clf.predict(X_test)

pred1 = clf1.predict(X_test)

cm = confusion_matrix(y_test, preds)

cm1 = confusion_matrix(y_test, pred1)

print(cm)

print(cm1)
array2 = df.iloc[:,13].values



results2 = []

factors = []



for i in range(len(result)):

    if(result[i] == 1):

        results2.append(array2[i])

        factors.append(df_chance[i])

        

factors = np.array(factors)

results2 = np.array(results2)

results2 = scaler.fit_transform(results2.reshape(-1,1))



X_train = factors[:113, :]

X_test = factors[113:, :]

y_train = results2[:113]

y_test = results2[113:]

reg = LinearRegression()

reg.fit(X_train, y_train)

pred_lin = reg.predict(X_test)



regressor = SVR()

regressor.fit(X_train, y_train)

pred = regressor.predict(X_test)

score = mean_squared_error(y_test, pred)

score_1 = mean_squared_error(y_test, pred_lin)
print(score)

print(score_1)