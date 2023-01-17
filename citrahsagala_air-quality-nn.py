import pandas as pd
df = pd.read_csv('../input/AirQualityUCI.csv')
df.head()
df.info()
#reduce columns

df = df.drop('Date', axis=1)

df = df.drop('Time', axis=1)

df = df.drop('C6H6(GT)', axis=1)
#check null value

df.isnull().sum()
#check minimum values each attribute

df.describe()
for i in df.ix[df['CO(GT)']<0].index:

    df.loc[i, 'CO(GT)'] = 0

for i in df.ix[df['PT08.S1(CO)']<0].index:

    df.loc[i, 'PT08.S1(CO)'] = 0

for i in df.ix[df['NMHC(GT)']<0].index:

    df.loc[i, 'NMHC(GT)'] = 0

for i in df.ix[df['PT08.S2(NMHC)']<0].index:

    df.loc[i, 'PT08.S2(NMHC)'] = 0

for i in df.ix[df['NOx(GT)']<0].index:

    df.loc[i, 'NOx(GT)'] = 0

for i in df.ix[df['PT08.S3(NOx)']<0].index:

    df.loc[i, 'PT08.S3(NOx)'] = 0

for i in df.ix[df['NO2(GT)']<0].index:

    df.loc[i, 'NO2(GT)'] = 0

for i in df.ix[df['PT08.S4(NO2)']<0].index:

    df.loc[i, 'PT08.S4(NO2)'] = 0

for i in df.ix[df['PT08.S5(O3)']<0].index:

    df.loc[i, 'PT08.S5(O3)'] = 0

for i in df.ix[df['RH']<0].index:

    df.loc[i, 'RH'] = 0

for i in df.ix[df['AH']<0].index:

    df.loc[i, 'AH'] = 0

#this case because it's possible the Temperature have minus degree,

#so i just fill 0 that have -200 value

for i in df.ix[df['T']==-200].index:

    df.loc[i, 'T'] = 0
#if you check min value has change

df['CO(GT)'].min()
#mean values has change

df.describe()
df['CO(GT)'].median()
df['PT08.S1(CO)'].median()
df['NMHC(GT)'].median()
df['PT08.S2(NMHC)'].median()
df['NOx(GT)'].median()
df['PT08.S3(NOx)'].median()
df['NO2(GT)'].median()
df['PT08.S4(NO2)'].median()
df['PT08.S5(O3)'].median()
df['T'].median()

#17.2
df['RH'].median()

#48.6
df['AH'].median()

#0.9768
for i in df.ix[df['CO(GT)']==0].index:

    df.loc[i, 'CO(GT)'] = df['CO(GT)'].mean()

for i in df.ix[df['PT08.S1(CO)']==0].index:

    df.loc[i, 'PT08.S1(CO)'] = df['PT08.S1(CO)'].mean()

for i in df.ix[df['NMHC(GT)']==0].index:

    df.loc[i, 'NMHC(GT)'] = df['NMHC(GT)'].mean()

for i in df.ix[df['PT08.S2(NMHC)']==0].index:

    df.loc[i, 'PT08.S2(NMHC)'] = df['PT08.S2(NMHC)'].mean()

for i in df.ix[df['NOx(GT)']==0].index:

    df.loc[i, 'NOx(GT)'] = df['NOx(GT)'].median()

for i in df.ix[df['PT08.S3(NOx)']==0].index:

    df.loc[i, 'PT08.S3(NOx)'] = df['PT08.S3(NOx)'].mean()

for i in df.ix[df['NO2(GT)']==0].index:

    df.loc[i, 'NO2(GT)'] = df['NO2(GT)'].mean()

for i in df.ix[df['PT08.S4(NO2)']==0].index:

    df.loc[i, 'PT08.S4(NO2)'] = df['PT08.S4(NO2)'].median()

for i in df.ix[df['PT08.S5(O3)']==0].index:

    df.loc[i, 'PT08.S5(O3)'] = df['PT08.S5(O3)'].median()

for i in df.ix[df['T']==0].index:

    df.loc[i, 'T'] = df['T'].mean()

for i in df.ix[df['RH']==0].index:

    df.loc[i, 'RH'] = df['RH'].mean()

for i in df.ix[df['AH']==0].index:

    df.loc[i, 'AH'] = df['AH'].mean()
#and if we check minimum value has change

df['CO(GT)'].min()
print(df[df['AH'].isnull()])
#There is 1 null value in AH attributes. Fill it with AH mean.

df['AH'].iloc[4887] = df['AH'].mean()
from sklearn.cluster import KMeans



km = KMeans(n_clusters=2, random_state=1)

new = df._get_numeric_data().dropna(axis=1)

km.fit(new)

predict=km.predict(new)
#fill dataframe with cluster result

df['Class'] = pd.Series(predict, index=df.index)
df.head()
X = df.drop('Class', axis=1)

y = df.loc[:,'Class'].values
from sklearn.cross_validation import train_test_split

#split data with 70%, 30%

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,  random_state =0)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

sc.fit(X_train)

X_train_std = sc.transform(X_train)

X_test_std = sc.transform(X_test)
#use nn for classify

from sklearn.linear_model import Perceptron

ppn =  Perceptron(n_iter=100, eta0=0.1, random_state=0)

ppn.fit(X_train_std, y_train)
y_pred = ppn.predict(X_test_std)

print('Misclassified sample : %d' % (y_test != y_pred).sum())



from sklearn.metrics import accuracy_score

print('accuracy : %.2f' % accuracy_score(y_test, y_pred))