import numpy as np 

import pandas as pd 

import math

import matplotlib.pyplot as plt 

import seaborn as sns 

from scipy import stats



# naive bayes classsifier 
df=pd.read_csv('../input/credit-approval.csv')
df.info()
df.head()
df.count()/df.shape[0]
df['class'].value_counts().plot(kind='bar')
# Fill the null values (Values Imputing)

f,ax=plt.subplots(2,3,figsize=(16,6))



df['A1'].value_counts().plot(kind='bar',ax=ax[0][0])

df['A4'].value_counts().plot(kind='bar',ax=ax[0][1])

df['A5'].value_counts().plot(kind='bar',ax=ax[0][2])

df['A6'].value_counts().plot(kind='bar',ax=ax[1][0])

df['A7'].value_counts().plot(kind='bar',ax=ax[1][1])
 # df['A7'].value_counts().plot(kind='bar')

# Replace the categorical values with most frequently repeated one

df['A1'].replace(np.nan,'b',inplace=True)

df['A4'].replace(np.nan,'u',inplace=True)

df['A5'].replace(np.nan,'g',inplace=True)

df['A6'].replace(np.nan,'c',inplace=True)

df['A7'].replace(np.nan,'v',inplace=True)
# df.info()

# Check for continous missing values in the features.

df.describe()
f,ax=plt.subplots(1,3,figsize=(13,7))

sns.boxplot(df["A15"],ax=ax[0])

sns.distplot(df["A15"],ax=ax[1])

stats.probplot(df["A15"],plot=ax[2])
# fill the null values of A2 with the mean of the column...

df['A2'].replace(np.nan,df['A2'].mean(),inplace=True)

df['A3'].replace(np.nan,df['A3'].mean(),inplace=True)

df['A8'].replace(np.nan,df['A8'].median(),inplace=True)

df['A11'].replace(np.nan,df['A11'].median(),inplace=True)

df['A14'].replace(np.nan,df['A14'].mean(),inplace=True)



# df=df[df['A15']<20000]



df_pos=df[df['class']=='+']

df_neg=df[df['class']=='-']



df_pos.describe()
df_neg.describe()
df.loc[(df["A15"].isnull())&(df["class"]=="+"),"A15"]==df_pos['A15'].mean()

df.loc[(df["A15"].isnull())&(df["class"]=="-"),"A15"]==df_neg['A15'].mean()

# df.count()/df.shape[0]*100

# No rows are empty and all the column values are imputed with some related values.
df.info()
df_approved=df[df['class']=='+']

df_notapproved=df[df['class']=='-']
# Analysis on the Data Set (A1 column)



df_A1=pd.crosstab(df['A1'],df['class'])

df_A1=df_A1.reset_index()

df_A1.columns=['A1','approved','not approved']

l=[]

for x in range(len(df['A1'].unique())):

    l.append(df_A1['approved'][x]/(df_A1['approved'][x]+ df_A1['not approved'][x]))

df_A1['percent approved']=[x*100 for x in np.array(l)]

df_A1['percent not approved']=[100-x for x in df_A1['percent approved']]
# df_approved.groupby('A1')['A1'].value_counts().plot(kind='bar')

df_A1.plot(x="A1", y=["percent approved","percent not approved"], kind="bar")
print(df['A9'].value_counts())

print(df['A10'].value_counts())

print(df['A12'].value_counts())
df_A4=pd.crosstab(df['A4'],df['class'])

df_A4=df_A4.reset_index()

df_A4.columns=['A4','approved','not approved']

l=[]

for x in range(len(df['A4'].unique())):

    l.append(df_A4['approved'][x]/(df_A4['approved'][x]+ df_A4['not approved'][x]))

df_A4['percent approved']=[x*100 for x in np.array(l)]

df_A4['percent not approved']=[100-x for x in df_A4['percent approved']]

df.A4.value_counts()
df_A4.plot(x="A4", y=["percent approved","percent not approved"], kind="bar")
df_A5=pd.crosstab(df['A5'],df['class'])

df_A5=df_A5.reset_index()

df_A5.columns=['A5','approved','not approved']

l=[]

for x in range(len(df['A5'].unique())):

    l.append(df_A5['approved'][x]/(df_A5['approved'][x]+ df_A5['not approved'][x]))

df_A5['percent approved']=[x*100 for x in np.array(l)]

df_A5['percent not approved']=[100-x for x in df_A5['percent approved']]

df.A5.value_counts()
df_A5.plot(x="A5", y=["percent approved","percent not approved"], kind="bar")
df.A7.value_counts()
df_A6=pd.crosstab(df['A6'],df['class'])

df_A6=df_A6.reset_index()

df_A6.columns=['A6','approved','not approved']

l=[]

for x in range(len(df['A6'].unique())):

    l.append(df_A6['approved'][x]/(df_A6['approved'][x]+ df_A6['not approved'][x]))

df_A6['percent approved']=[x*100 for x in np.array(l)]

df_A6['percent not approved']=[100-x for x in df_A6['percent approved']]

df.A6.value_counts()
df_A6.plot(x="A6", y=["percent approved","percent not approved"], kind="bar")
df_A9=pd.crosstab(df['A9'],df['class'])

df_A9=df_A9.reset_index()

df_A9.columns=['A9','approved','not approved']

l=[]

for x in range(len(df['A9'].unique())):

    l.append(df_A9['approved'][x]/(df_A9['approved'][x]+ df_A9['not approved'][x]))

df_A9['percent approved']=[x*100 for x in np.array(l)]

df_A9['percent not approved']=[100-x for x in df_A9['percent approved']]

df.A9.value_counts()
df_A9.plot(x="A9", y=["percent approved","percent not approved"], kind="bar")
sns.catplot(x="A9", y="A11", hue="class", data=df,kind="violin")
sns.catplot(x="A9", y="A11", hue="class", kind="bar", data=df)
plt.figure(figsize = (12, 6))

ax = sns.boxplot(x='class', y='A11', data=df)

plt.setp(ax.artists, alpha=.5, linewidth=2, edgecolor="k")

plt.xticks(size=10)
sns.set(style="white", palette="deep", font_scale=1.2, 

        rc={"figure.figsize":(15,9)})

ax = sns.scatterplot(x="A2", y="A8", hue="class",data=df)
df_temp=df.copy()

df_temp['class'].replace({'+':1,'-':0},inplace=True)

sns.heatmap(df_temp.corr(),annot=True)
from collections import Counter

import scipy.stats as ss



df_new=df.drop(columns=['A2','A3','A8','A11','A14','A15'])



def conditional_entropy(x,y):

    # entropy of x given y

    y_counter = Counter(y)

    xy_counter = Counter(list(zip(x,y)))

    total_occurrences = sum(y_counter.values())

    entropy = 0

    for xy in xy_counter.keys():

        p_xy = xy_counter[xy] / total_occurrences

        p_y = y_counter[xy[1]] / total_occurrences

        entropy += p_xy * math.log(p_y/p_xy)

    return entropy



def theil_u(x,y):

    s_xy = conditional_entropy(x,y)

    x_counter = Counter(x)

    total_occurrences = sum(x_counter.values())

    p_x = list(map(lambda n: n/total_occurrences, x_counter.values()))

    s_x = ss.entropy(p_x)

    if s_x == 0:

        return 1

    else:

        return (s_x - s_xy) / s_x



theilu = pd.DataFrame(index=['class'],columns=df_new.columns)

columns = df_new.columns

for j in range(0,len(columns)):

    u = theil_u(df_new['class'].tolist(),df_new[columns[j]].tolist())

    theilu.loc[:,columns[j]] = u

theilu.fillna(value=np.nan,inplace=True)

plt.figure(figsize=(20,1))

sns.heatmap(theilu,annot=True,fmt='.2f')

plt.show()
df['class']=df['class'].replace({'+':1,'-':0})

df['class'].unique()

df_model=df.copy()

df_model=df_model.drop(columns=['A2','A14','A15','A1','A4','A5','A6','A7','A10','A12','A13'])

df_model=pd.get_dummies(df_model)
df_model.info()
# features for predictive modelling 

from sklearn.model_selection import train_test_split



# A9,A3,A8,A11

y=df_model['class']



X=pd.DataFrame(columns=['A3','A8',

                        'A11','A9_f'],data=df_model)

# Split dataset into training set and test set

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3,random_state=109) # 70% training and 30% test
# Naive bayes Classifier 

from sklearn.model_selection import train_test_split



from sklearn.naive_bayes import GaussianNB



#Create a Gaussian Classifier

gnb = GaussianNB()



#Train the model using the training sets

gnb.fit(X_train, y_train)



#Predict the response for test dataset

y_pred = gnb.predict(X_test)



from sklearn import metrics



# Model Accuracy, how often is the classifier correct?

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))



print("Precision:",metrics.precision_score(y_test, y_pred))



# Model Recall: what percentage of positive tuples are labelled as such?

print("Recall:",metrics.recall_score(y_test, y_pred))


