import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df = pd.read_csv('../input/students-performance-in-exams/StudentsPerformance.csv')
df.head()
df.info()
df.describe()
sns.distplot(df['math score'])

plt.title('Distribution of Math scores')
sns.distplot(df['reading score'])

plt.title('Distribution of Reading scores')
sns.distplot(df['writing score'])

plt.title('Distribution of Writing Scores')
exam_list = ['math score', 'reading score', 'writing score']

for exam in exam_list:

    print(exam + ':')

    print('Percentage of students scoring between 0 & 50: {}%'.format(100 * len(df[df[exam] <= 50]) / len(df)))

    print('Percentage of students scoring between 51 & 60: {}%'.format(100 * len(df[(df[exam] >= 51) & (df[exam] <= 60)]) / len(df)))

    print('Percentage of students scoring between 61 & 70: {}%'.format(100 * len(df[(df[exam] >= 61) & (df[exam] <= 70)]) / len(df)))

    print('Percentage of students scoring between 71 & 80: {}%'.format(100 * len(df[(df[exam] >= 71) & (df[exam] <= 80)]) / len(df)))

    print('Percentage of students scoring between 81 & 90: {}%'.format(100 * len(df[(df[exam] >= 81) & (df[exam] <= 90)]) / len(df)))

    print('Percentage of students scoring between 91 & 100: {}%'.format(100 * len(df[(df[exam] >= 91)]) / len(df)))

    print('-' * 40)

  
sns.pairplot(df[['math score','writing score','reading score']])
sns.heatmap(df[['math score','reading score','writing score']].corr(), annot=True)
df[df['gender'] == 'male'].describe()
df[df['gender'] == 'female'].describe()
sns.boxplot(x='gender',y='math score',data=df)
sns.boxplot(x='gender',y='reading score',data=df)
sns.boxplot(x='gender',y='writing score',data=df)
df[df['test preparation course'] == 'completed'].describe()
df[df['test preparation course'] == 'none'].describe()
plt.figure(figsize=(12,8))

plt.subplot(1,3,1)

sns.boxplot(x='test preparation course', y='math score', data=df)



plt.subplot(1,3,2)

sns.boxplot(x='test preparation course', y='reading score', data=df)



plt.subplot(1,3,3)

sns.boxplot(x='test preparation course', y='writing score', data=df)



plt.suptitle('How does the Test Preparation Course effect Test Scores?')
order = ['group A', 'group B', 'group C', 'group D', 'group E']



plt.figure(figsize=(15,8))

plt.subplot(1,3,1)

sns.boxplot(x='race/ethnicity', y='math score', data=df,order=order)



plt.subplot(1,3,2)

sns.boxplot(x='race/ethnicity', y='reading score', data=df, order=order)



plt.subplot(1,3,3)

sns.boxplot(x='race/ethnicity', y='writing score', data=df,order=order)



plt.suptitle('How does race/ethnicity effect Test Scores?')
sns.countplot(x='race/ethnicity',data=df, order=order)
plt.figure(figsize=(10,6))

sns.countplot(x='parental level of education',data=df)
df['parental level of education'] = df['parental level of education'].apply(lambda x: 'high school' if 'high school' in x else x)
plt.figure(figsize=(10,6))

sns.countplot(x='parental level of education',data=df)
education_order = ["high school", "some college", "associate's degree", "bachelor's degree", "master's degree"]

plt.figure(figsize=(10,6))

sns.boxplot(x='parental level of education', y='math score', data=df, order=education_order)
plt.figure(figsize=(10,6))

sns.boxplot(x='parental level of education', y='reading score', data=df, order=education_order)
plt.figure(figsize=(10,6))

sns.boxplot(x='parental level of education', y='writing score', data=df,order=education_order)
df[df['lunch'] == 'standard'].describe()
df[df['lunch'] == 'free/reduced'].describe()
plt.figure(figsize=(15,8))

plt.subplot(1,3,1)

sns.boxplot(x='lunch', y='math score', data=df)



plt.subplot(1,3,2)

sns.boxplot(x='lunch', y='reading score', data=df)



plt.subplot(1,3,3)

sns.boxplot(x='lunch', y='writing score', data=df)



plt.suptitle('How does the lunch package each student recieves effect Test Scores?')
race_v_pared = df.groupby(['parental level of education','race/ethnicity']).size().reset_index(name="Count").pivot(index='parental level of education',columns='race/ethnicity',values='Count')

race_v_pared.index = pd.CategoricalIndex(race_v_pared.index, categories=["high school", "some college", "associate's degree", "bachelor's degree", "master's degree"])

race_v_pared.sort_index(level=0, inplace=True)

sns.heatmap(race_v_pared,annot=True,fmt='d')
for group in ['group A', 'group B', 'group C', 'group D', 'group E']:

    print(group)

    for edu in education_order:

        print("Percentage with {} education: {}%".format(edu, 100 * race_v_pared[group][edu]/race_v_pared[group].sum()))

    print('-' * 25)
race_v_lunch = df.groupby(['race/ethnicity','lunch']).size().reset_index(name="Count").pivot(index='lunch',columns='race/ethnicity',values='Count')

race_v_lunch.index = pd.CategoricalIndex(race_v_lunch.index, categories=['free/reduced','standard'])

race_v_lunch.sort_index(level=0, inplace=True)

sns.heatmap(race_v_lunch,annot=True, fmt='d')
for group in ['group A', 'group B', 'group C', 'group D', 'group E']:

    print(group)

    for lunch in ['free/reduced','standard']:

        print("Percentage with {} lunch: {}%".format(lunch,100 * race_v_lunch[group][lunch] / race_v_lunch[group].sum()))

    print('-' * 40)
race_v_prep = df.groupby(['race/ethnicity','test preparation course']).size().reset_index(name="Count").pivot(index='test preparation course',columns='race/ethnicity',values='Count')

race_v_prep.index = pd.CategoricalIndex(race_v_prep.index, categories=['none','completed'])

race_v_prep.sort_index(level=0, inplace=True)

sns.heatmap(race_v_prep,annot=True,fmt='d')
for group in ['group A', 'group B', 'group C', 'group D', 'group E']:

    print("Percentage of Students in ethnic {} who completed the test preparation course: {}%".format(group, 100 * race_v_prep[group]['completed'] / race_v_prep[group].sum()))
parvprep = df.groupby(['parental level of education', 'test preparation course']).size().reset_index(name='Count').pivot(index='parental level of education',columns='test preparation course',values='Count')

parvprep.index = pd.CategoricalIndex(parvprep.index, categories=["high school", "some college", "associate's degree", "bachelor's degree", "master's degree"])

parvprep.sort_index(level=0, inplace=True)

sns.heatmap(parvprep,annot=True,fmt='d')
for edu in ["high school", "some college", "associate's degree", "bachelor's degree", "master's degree"]:

    print("Percentage of students who parents achieved {} level education that completed the test preparation course: {}%".format(edu, (100 * parvprep['completed'][edu] / (parvprep['completed'][edu] + parvprep['none'][edu]) )))
full_marks = df[(df['math score'] == 100) & (df['reading score'] == 100) & (df['writing score'] == 100)]

full_marks
zero_marks = df[(df['math score'] == 0) & (df['reading score'] == 0) & (df['writing score'] == 0)]

zero_marks
lessthan40 = df[(df['math score'] < 20) & (df['reading score'] < 20) & (df['writing score'] < 20)]

lessthan40
df.describe().T
m_and_r = df[((df['math score'] > 77) & (df['reading score'] < 59)) | ((df['reading score'] > 79) & (df['math score'] < 57))]

m_and_r
m_and_w = df[((df['math score'] > 77) & (df['writing score'] < 58)) | ((df['writing score'] > 79) & (df['math score'] < 57))]

m_and_w
r_and_w = df[((df['reading score'] > 79) & (df['writing score'] < 58)) | ((df['writing score'] > 79) & (df['reading score'] < 59))]

r_and_w
def outside_range(df, column):

    global lower,upper

    q25, q75 = np.quantile(df[column], 0.25), np.quantile(df[column], 0.75)

    

    # calculate the IQR

    iqr = q75 - q25

    

    # calculate the outlier cutoff

    cut_off = iqr * 1.5

    

    # calculate the lower and upper bound value of the range

    lower, upper = q25 - cut_off, q75 + cut_off

    print('The IQR for {} is {}'.format(column,iqr))

    print('The lower bound value is', lower)

    print('The upper bound value is', upper)

    

    

    # Calculate the number of records below and above lower and above bound value respectively

    df1 = df.index[(df[column] > upper) | (df[column] < lower)]

    

    print("The number of outliers for {} is {}".format(column, len(df1)))

    

    # show the two data frames where the values are outside the range

    return df.iloc[df1]
outside_range(df,'math score')
outside_range(df,'reading score')
outside_range(df, 'writing score')
df.iloc[842]
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
df['parental level of education'] = label_encoder.fit_transform(df['parental level of education'])

df['lunch'] = label_encoder.fit_transform(df['lunch'])

df['test preparation course'] = label_encoder.fit_transform(df['test preparation course'])
df['race/ethnicity'] = df['race/ethnicity'].replace('group A', 1)

df['race/ethnicity'] = df['race/ethnicity'].replace('group B', 2)

df['race/ethnicity'] = df['race/ethnicity'].replace('group C', 3)

df['race/ethnicity'] = df['race/ethnicity'].replace('group D', 4)

df['race/ethnicity'] = df['race/ethnicity'].replace('group E', 5)
gender = pd.get_dummies(df['gender'],drop_first=True)

df = pd.concat([df,gender],axis=1)

df.head()
df = df.drop('gender',axis=1)
df.head()
df['average score'] = (df['math score'] + df['reading score'] + df['writing score']) / 3
df.head()
X = df[['race/ethnicity','parental level of education','lunch','test preparation course','male']]

y = df['average score']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test_ave = train_test_split(X, y, test_size=0.3, random_state=101)
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(X_train,y_train)

lr_ave_pred = lr.predict(X_test)
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout

from tensorflow.keras.optimizers import Adam
ave_deep_model = Sequential()



# Input Layer

ave_deep_model.add(Dense(5,activation='relu'))

ave_deep_model.add(Dropout(0.25))



# Hidden Layer 1

ave_deep_model.add(Dense(10,activation='relu'))

ave_deep_model.add(Dropout(0.25))



# Hidden Layer 2

ave_deep_model.add(Dense(20,activation='relu'))

ave_deep_model.add(Dropout(0.25))



# Hidden Layer 3

ave_deep_model.add(Dense(10,activation='relu'))

ave_deep_model.add(Dropout(0.25))



# Output Layer

ave_deep_model.add(Dense(1,activation='relu'))



# Compile Model

ave_deep_model.compile(optimizer='adam',loss='mse')
from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=250)
ave_deep_model.fit(x=X_train, 

          y=y_train.values, 

          epochs=1000,

          validation_data=(X_test, y_test_ave), verbose=1,

          batch_size=64,

          callbacks=[early_stop]

          )
ave_deep_model_pred = ave_deep_model.predict(X_test)
X = df[['race/ethnicity','parental level of education','lunch','test preparation course','male']]

y = df[['math score','reading score','writing score']]
X_train, X_test, y_train, y_test_indiv = train_test_split(X, y, test_size=0.3, random_state=101)
scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
lr3 = LinearRegression()

lr3.fit(X_train,y_train)

lr3_pred = lr3.predict(X_test)
multi_deep_model = Sequential()



# Input Layer

multi_deep_model.add(Dense(5,activation='relu'))

multi_deep_model.add(Dropout(0.25))



# Hidden Layer 1

multi_deep_model.add(Dense(10,activation='relu'))

multi_deep_model.add(Dropout(0.25))



# Hidden Layer 2

multi_deep_model.add(Dense(20,activation='relu'))

multi_deep_model.add(Dropout(0.25))



# Hidden Layer 3

multi_deep_model.add(Dense(10,activation='relu'))

multi_deep_model.add(Dropout(0.25))



# Output Layer

multi_deep_model.add(Dense(3,activation='relu'))



# Compile Model

multi_deep_model.compile(optimizer='adam',loss='mse')
multi_deep_model.fit(x=X_train, 

          y=y_train.values, 

          epochs=1000,

          validation_data=(X_test, y_test_indiv), verbose=1,

          batch_size=64,

          callbacks=[early_stop]

          )
multi_deep_preds = multi_deep_model.predict(X_test)
from sklearn.metrics import mean_absolute_error, mean_squared_error
print("Linear Regression MAE: {}".format(mean_absolute_error(y_test_ave,lr_ave_pred)))

print("Deep Learning Model MAE: {}".format(mean_absolute_error(y_test_ave,ave_deep_model_pred)))

print("-" * 40)

print("Linear Regression MSE: {}".format(mean_squared_error(y_test_ave,lr_ave_pred)))

print("Deep Learning Model MSE: {}".format(mean_squared_error(y_test_ave,ave_deep_model_pred)))
print("Multi Target Linear Regression MAE: {}".format(mean_absolute_error(y_test_indiv,lr3_pred)))

print("Multi Target Deep Network MAE: {}".format(mean_absolute_error(y_test_indiv,multi_deep_preds)))

print("-" * 50)

print("Multi Target Linear Regression MSE: {}".format(mean_squared_error(y_test_indiv,lr3_pred)))

print("Multi Target Deep Network MAE: {}".format(mean_squared_error(y_test_indiv,multi_deep_preds)))
def average_to_grade(x):

    

    if 0 <= x <= 40:

        return 0

    elif 40 < x <= 50:

        return 1

    elif 50 < x <= 60:

        return 2

    elif 60 < x <= 70:

        return 3

    elif 70 < x <= 80:

        return 4

    elif 80 < x <= 90:

        return 5

    else:

        return 6
df['grade'] = df['average score'].apply(average_to_grade)
df['grade'].value_counts()
df = df.drop(['math score','reading score','writing score','average score'],axis=1)
df.head()
from imblearn.over_sampling import SMOTENC
data = df.values

X = data[:, :-1]

y = data[:, -1]

X_columns = df.columns[:-1]

y_columns = df.columns[-1]



oversample = SMOTENC([0,1,2,3])

X, y = oversample.fit_sample(X, y)

X_sampled = pd.DataFrame(X, columns=X_columns)

y_sampled = pd.DataFrame(y, columns=[y_columns])



df = pd.concat([X_sampled,y_sampled],axis=1)
df['grade'].value_counts()
X = df.drop('grade',axis=1)

y = df['grade']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,random_state=101)
from sklearn.linear_model import LogisticRegression

log_model = LogisticRegression(multi_class='multinomial',max_iter=2000)
log_model.fit(X_train,y_train)
log_model_preds = log_model.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix

print(confusion_matrix(y_test,log_model_preds))

print("\n")

print(classification_report(y_test,log_model_preds))
from sklearn.neighbors import KNeighborsClassifier

error_rate = []



# Will take some time

for i in range(1,40):

    

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(X_train,y_train)

    pred_i = knn.predict(X_test)

    error_rate.append(np.mean(pred_i != y_test))
plt.figure(figsize=(10,6))

plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',

         markerfacecolor='red', markersize=10)

plt.title('Error Rate vs. K Value')

plt.xlabel('K')

plt.ylabel('Error Rate')
knn = KNeighborsClassifier(n_neighbors=17)

knn.fit(X_train,y_train)

knn_preds = knn.predict(X_test)

print(confusion_matrix(y_test,knn_preds))

print("\n")

print(classification_report(y_test,knn_preds))
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=100)

rfc.fit(X_train,y_train)
rfc_preds = rfc.predict(X_test)

print(confusion_matrix(y_test,rfc_preds))

print("\n")

print(classification_report(y_test,rfc_preds))