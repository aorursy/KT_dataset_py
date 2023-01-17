import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv ('../input/train.csv')

#info of our training set
df.info()
#since we have 5 columns with "object" type, we have to make them numerical type
df.columns [df.dtypes == 'object']
#We did not need Id and idhogar for training, I am not sure about idhogar though
#lets look at depencdency column
df['dependency'].unique()
#we can take care of yes and no values with the help of "SQBdependency" columns which is "dependency squared"
df[['dependency', 'SQBdependency']].head()
#it does not hace yes or no
#it appears, you can also replace every yes with 1 and every no with 0
#Volla
df['dependency'] = np.sqrt (df['SQBdependency'])

#lets look at other columns
df['edjefe'].unique()
#similarily
df['edjefe'] = np.sqrt( df['SQBedjefe'])
#no related column is given for 'edjefa'.
#lets replace every yes with 1 and no with 0
df['edjefa'] = df['edjefa'].replace ('no' , 0)
df['edjefa'] = df['edjefa'].replace ('yes' , 1)
df['edjefa'] = df['edjefa'].astype(dtype='float64')
#now lets check for the null values
n = df.isna().any()
n[n == True]
#v2a1 is the Monthly rent payment
#lets look at columns related to it

df[['v2a1','tipovivi1', 'tipovivi2', 'tipovivi3', 'tipovivi4', 'tipovivi5']]
#v2a1 has NAN only for tipvovivi1, tipvovivi4 and tipvovivi5. It appears they pay 0 monthly rent.
#So we can replace it by 0
#tipovivi1, =1 own and fully paid house
#tipovivi2, =1 own,  paying in installments
#tipovivi3, =1 rented
#tipovivi4, =1 precarious
#tipovivi5, =1 other(assigned,  borrowed)

df['v2a1'] = df['v2a1'].fillna (value = 0)
#let's look at "v18q1" => number of tablets household owns
#luckily it also have a related variable i.e., v18q => owns a tablet 

df[['v18q1', 'v18q']]
#it appears only those value are NAN for whihc the person did not own a Tablet
#it means number of tablets = 0

df['v18q1'] = df['v18q1'].fillna (value = 0)
#rez_esc => Years behind in school
#lets look at it and its related variabels

df[['rez_esc', 'age', 'escolari']]
#it appears that rez_esc is NAN for all the persons whose age >= 18

df[['rez_esc', 'age', 'escolari']][df['age'] >= 18]
#leta take a look at persons with age < 18

df[['rez_esc', 'age', 'escolari']][df['age'] < 18]
#maximum age for which rez_esc i.e., school behind can be 0
#it appears that the ideal age to go to the school is 7 (take a look at the observations)

df[['rez_esc', 'age', 'escolari']][df['age'] == 7]
#unique values present 
df['rez_esc'].unique()
#persons having maximum School Behind values
df[['age', 'escolari', 'rez_esc']][df['rez_esc'] == 5]
#filling with 0's for now
df['rez_esc'] = df['rez_esc'].fillna (value=0)
#lets take a look at meaneduc and SQBmeaned 

df[['meaneduc', 'SQBmeaned']][df['meaneduc'].isnull()]
#As there are very few null values,
#lets fill them with the mean value
df['meaneduc'] = df['meaneduc'].fillna (df['meaneduc'].mean())
df['SQBmeaned'] = df['meaneduc']**2
#I am not using this column for classification
del df['idhogar']

row, col = df.shape

X = df.iloc[: , 1:col-1].values
y = df.iloc[: , -1]

#its time to fit our model :)
#I am using NAIVE BYES because it is good when the training data is less
#Also it is quite simple
from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB ()
classifier.fit (X, y)

#Bingo :)
#data preprocessing will be same for test set too

tf = pd.read_csv ('../input/test.csv')

#making all columns numerical type (int/float)
tf['dependency'] = np.sqrt(tf['SQBdependency'])
tf['edjefe'] = np.sqrt(tf['SQBedjefe'])

tf['edjefa'] = tf['edjefa'].replace ('no' , 0)
tf['edjefa'] = tf['edjefa'].replace ('yes' , 1)
tf['edjefa'] = tf['edjefa'].astype(dtype='float64')

#removing NAN values
tf['v2a1'] = tf['v2a1'].fillna (value=0)
tf['v18q1'] = tf['v18q1'].fillna (value=0)
tf['rez_esc'] = tf['rez_esc'].fillna (value=0)
tf['meaneduc'] = tf['meaneduc'].fillna (tf['meaneduc'].mean())
tf['SQBmeaned'] = tf['meaneduc']**2

del tf['idhogar']
X_test = tf.iloc[: , 1:].values
#its time for prediction
y_pred = classifier.predict (X_test)

tf['Target'] = y_pred
#store the result
tf[['Id','Target']].to_csv ('result.csv' , index = False)