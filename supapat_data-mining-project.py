# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
!pip install apyori
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns 
from apyori import apriori

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
csvdf=pd.read_csv("/kaggle/input/hotel-booking-demand/hotel_bookings.csv")
csvdf = csvdf.drop(['company','agent'],axis=1)
csvdf.corr()['is_canceled'][:-1].sort_values().plot(kind='bar')
nf = csvdf.select_dtypes(exclude=['int','float'])
nf = nf.dropna(subset=['country'])
nf.columns = range(nf.shape[1])
print(nf.isna().sum())
transactions = []
for i in range(0,len(nf)):
    transactions.append([str(nf.values[i,j]) for j in range(0,11) if str(nf.values[i,j])!='0'])
transactions[0]
rules = apriori(transactions,min_support=0.1,min_confidance=0.3,min_lift=3,min_length=2)
Results = list(rules)
df_results = pd.DataFrame(Results)
df_results
support = df_results.support
first_values = []
second_values = []
third_values = []
fourth_value = []

# loop number of rows time and append 1 by 1 value in a separate list.. first and second element was frozenset which need to be converted in list..
for i in range(df_results.shape[0]):
    single_list = df_results['ordered_statistics'][i][0]
    first_values.append(list(single_list[0]))
    second_values.append(list(single_list[1]))
    third_values.append(single_list[2])
    fourth_value.append(single_list[3])
#convert all four list into dataframe for further operation..
lhs = pd.DataFrame(first_values)
rhs= pd.DataFrame(second_values)
confidance=pd.DataFrame(third_values,columns=['Confidance'])
lift=pd.DataFrame(fourth_value,columns=['lift'])
#concat all list together in a single dataframe
df_final = pd.concat([lhs,rhs,support,confidance,lift], axis=1)
df_final
print("Nan in each columns" , csvdf.isna().sum(), sep='\n')
csvdf['is_canceled'].value_counts()
plt.figure(figsize=(15,10))
plt.hist(csvdf['lead_time'].dropna(), bins=30,color = 'paleturquoise' )

plt.ylabel('Count')
plt.xlabel('Time (days)')
plt.title("Lead time distribution ", fontdict=None, position= [0.48,1.05], size = 'xx-large')
plt.show()
corr = csvdf.corr()
f, ax = plt.subplots(figsize=(15, 8))
cmap = sns.diverging_palette(10, 10, as_cmap=True)
sns.heatmap(corr, annot=True)
corr_matrix = csvdf.corr().abs()

#the matrix is symmetric so we need to extract upper triangle matrix without diagonal (k = 1)
sol = (corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
                 .stack()
                 .sort_values(ascending=False))
sol
labels = csvdf['arrival_date_month'].unique().tolist()
plt.rcParams['figure.figsize'] = 15,8

height = csvdf['is_canceled'].value_counts().tolist()
bars =  ['Not Cancel','Cancel']
y_pos = np.arange(len(bars))
color = ['lightgreen','salmon']
plt.bar(y_pos, height , width=0.7 ,color= color)
plt.xticks(y_pos, bars)
plt.xticks(rotation=90)
plt.title("How many booking was cancel", fontdict=None, position= [0.48,1.05], size = 'xx-large')
plt.show()
csvdf['arrival_date_month_p'] = csvdf['arrival_date_month'].map({'January':1, 'February': 2, 'March':3, 'April':4, 'May':5, 'June':6, 'July':7,'August':8, 'September':9, 'October':10, 'November':11, 'December':12})
csvdf = csvdf.sort_values(by=['arrival_date_month_p'])
ct = pd.crosstab(csvdf.arrival_date_month_p, csvdf.is_canceled)
ct.plot.bar(stacked=True)
plt.legend(title='is_cancle')
plt.title("How many booking was cancel per month", fontdict=None, position= [0.48,1.05], size = 'xx-large')
plt.show()
from sklearn.model_selection import train_test_split

df = csvdf.select_dtypes(exclude=['object'])
df["children"].replace(np.nan,0,inplace=True)
X = df.drop(['is_canceled','children'], axis = 1)
y = df['is_canceled']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
df.isnull().sum()
csvdf
from sklearn.tree import DecisionTreeClassifier

#Create tree
decision_tree = DecisionTreeClassifier(criterion = 'entropy',max_depth = 4)
decision_tree.fit(X_train, y_train)
print('The accuracy of the Decision Tree classifier on test data is {:.2f}'.format(decision_tree.score(X_test, y_test)))
import sklearn.tree as tree
from sklearn.externals.six import StringIO 
from IPython.display import Image

dot_data = StringIO()
tree.export_graphviz(decision_tree, 
 out_file='tree_limited.dot', 
 class_names=df['is_canceled'].map({0:'False',1:'True'}).unique().tolist(), # the target names.
 feature_names=X_train.columns.tolist(), # the feature names.
 filled=True, 
 rounded=True, 
 special_characters=True)
!dot -Tpng tree_limited.dot -o tree_limited.png -Gdpi=600
Image(filename = 'tree_limited.png')
df=csvdf.drop('reservation_status',axis=1)
df['is_canceled']=df['is_canceled'].replace([0,1],["no","yes"])
cols = df.columns
num_cols = df._get_numeric_data().columns
cat_cols=list(set(cols) - set(num_cols))
df_cat=df[cat_cols]

X_cat = df_cat.drop("is_canceled", axis=1)
y_cat = df_cat["is_canceled"].eq('yes').mul(1)

X_cat['country'].fillna("No Country", inplace = True)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
def labelencode(df):
    le = LabelEncoder()
    return df.apply(le.fit_transform)

def onehotencode(df):
    onehot = OneHotEncoder()
    return onehot.fit_transform(df).toarray()

X_2 = labelencode(X_cat)
onehotlabels = onehotencode(X_2)
X_2.head().transpose()
#getting the numerical feature columns one more time
cols = csvdf.columns
num_cols = csvdf._get_numeric_data().columns

#selecting numerical features
df_num=df[num_cols].drop('is_canceled',axis=1)

#selecting target ('is_canceled' column)
y_num=y_cat
df_num.columns
df_num=df_num.fillna(df_num.median())
from sklearn.preprocessing import StandardScaler
# Standardizing the features
df_num_standard = StandardScaler().fit_transform(df_num.values)

#replacing the X_num dataframe with the standardized dataframe
df_num[:] = df_num_standard
#concatenating numerically converted categorical and numerical feature arrays
X_arr=np.concatenate((onehotlabels, df_num_standard), axis=1)
y_arr = df['is_canceled'].values
X_arr
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_arr,y_arr,test_size=0.25,random_state=2019)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation,Dropout
X_train.shape
model = Sequential()

#adding dropout layers for improved learning
model.add(Dense(units=30,activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(units=20,activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(units=10,activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(units=1,activation='sigmoid'))

# For a binary classification problem
model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
#Putting early_stop in to prevent overfitting
from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)

model.fit(x=X_train, 
          y=y_train, 
          epochs=100,
          validation_data=(X_test, y_test), verbose=1,callbacks=[early_stop]
          )