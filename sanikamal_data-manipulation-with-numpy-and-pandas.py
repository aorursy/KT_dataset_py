#load the library and check its version, just to make sure we aren't using an older version
import numpy as np
np.__version__
#create a list comprising numbers from 0 to 20
L = list(range(21))
L
#converting integers to string - this style of handling lists is known as list comprehension.
[str(c) for c in L]
#List comprehension offers a versatile way to handle list manipulations tasks easily.
[type(item) for item in L]
#creating arrays
np.zeros(10, dtype='int')
#creating a 3 row x 8 column matrix
np.ones((3,8), dtype=float)
#creating a matrix with a predefined value
np.full((3,5),1.23)
#create an array with a set sequence
np.arange(0, 20, 2)
#create an array of even space between the given range of values
np.linspace(0, 1, 5)
#create a 3x3 array with mean 0 and standard deviation 1 in a given dimension
np.random.normal(0, 1, (3,3))
#create an identity matrix
np.eye(5)
#set a random seed
np.random.seed(0)


x1 = np.random.randint(10, size=4) #one dimension
x2 = np.random.randint(10, size=(3,5)) #two dimension
x3 = np.random.randint(10, size=(3,5,5)) #three dimension


print("x1 ndim:", x1.ndim)
print("x1 shape:", x1.shape)
print("x1 size: ", x1.size)
print("x2 ndim:", x2.ndim)
print("x2 shape:", x2.shape)
print("x2 size: ", x2.size)
print("x3 ndim:", x3.ndim)
print("x3 shape:", x3.shape)
print("x3 size: ", x3.size)
x1 = np.array([4, 3, 4, 4, 8, 4])
x1
#assess value to index zero
x1[0]
#assess fifth value
x1[4]
#get the last value
x1[-1]
#get the second last value
x1[-2]

#in a multidimensional array, we need to specify row and column index
x2=np.array([[3, 7, 5, 5],[0, 1, 5, 9],[3, 0, 5, 0]])
x2
#3rd row and last value from the 3rd column
x2[2,-1]
#replace value at 0,0 index
x2[0,0] = 12
x2
x = np.arange(20)
x
#from start to 4th position
x[:5]
#from 4th position to end
x[4:]
#from 4th to 6th position
x[4:7]
#return elements at even place
x[ : : 2]
#return elements from first position step by two
x[1::2]
#reverse the array
x[::-1]
#You can concatenate two or more arrays at once.
x = np.array([1, 2, 3])
y = np.array([3, 2, 1])
z = [21,21,21]
np.concatenate([x, y,z])
#You can also use this function to create 2-dimensional arrays.
grid = np.array([[1,2,3],[4,5,6]])
np.concatenate([grid,grid])
#Using its axis parameter, you can define row-wise or column-wise matrix
np.concatenate([grid,grid],axis=1)
x = np.array([3,4,5])
grid = np.array([[1,2,3],[17,18,19]])
np.vstack([x,grid])
#Similarly, you can add an array using np.hstack
z = np.array([[9],[9]])
np.hstack([grid,z])
x = np.arange(10)
x
x1,x2,x3 = np.split(x,[3,6])
print(x1,x2,x3)
grid = np.arange(16).reshape((4,4))
grid
upper,lower = np.vsplit(grid,[2])
print (upper, lower)
#load library - pd is just an alias. 
import pandas as pd
#create a data frame - dictionary is used here where keys get converted to column names and values to row values.
data = pd.DataFrame({'Country': ['India','Nepal','Pakistan','Bangladesh','Bhutan'],
                    'Rank':[11,40,100,130,101]})
data
#We can do a quick analysis of any data set using:
data.describe()
data.info()
#Let's create another data frame.
data = pd.DataFrame({'group':['x', 'x', 'x', 'y','y', 'y', 'z', 'z','z'],'ounces':[4, 3, 12, 6, 7.5, 8, 3, 5, 6]})
data
#Let's sort the data frame by ounces - inplace = True will make changes to the data
data.sort_values(by=['ounces'],ascending=True,inplace=False)
data.sort_values(by=['group','ounces'],ascending=[True,False],inplace=False)
#create another data with duplicated rows
data = pd.DataFrame({'k1':['one']*3 + ['two']*4, 'k2':[3,2,1,3,3,4,4]})
data
#sort values 
data.sort_values(by='k2')
#remove duplicates - ta da! 
data.drop_duplicates()
data.drop_duplicates(subset='k1')
data = pd.DataFrame({'food': ['bacon', 'pulled pork', 'bacon', 'Pastrami','corned beef', 'Bacon', 'pastrami', 'honey ham','nova lox'],
                 'ounces': [4, 3, 12, 6, 7.5, 8, 3, 5, 6]})
data
meat_to_animal = {
'bacon': 'pig',
'pulled pork': 'pig',
'pastrami': 'cow',
'corned beef': 'cow',
'honey ham': 'pig',
'nova lox': 'salmon'
}

def meat_2_animal(series):
    if series['food'] == 'bacon':
        return 'pig'
    elif series['food'] == 'pulled pork':
        return 'pig'
    elif series['food'] == 'pastrami':
        return 'cow'
    elif series['food'] == 'corned beef':
        return 'cow'
    elif series['food'] == 'honey ham':
        return 'pig'
    else:
        return 'salmon'


#create a new variable
data['animal'] = data['food'].map(str.lower).map(meat_to_animal)
data
#another way of doing it is: convert the food values to the lower case and apply the function
lower = lambda x: x.lower()
data['food'] = data['food'].apply(lower)
data['animal2'] = data.apply(meat_2_animal, axis='columns')
data
data.assign(new_variable = data['ounces']*14)
data.drop('animal2',axis='columns',inplace=True)
data
#Series function from pandas are used to create arrays
data = pd.Series([1., -999., 2., -999., -1000., 3.])
data
#replace -999 with NaN values
data.replace(-999, np.nan,inplace=True)
data
#We can also replace multiple values at once.
data = pd.Series([1., -999., 2., -999., -1000., 3.])
data.replace([-999,-1000],np.nan,inplace=True)
data
data = pd.DataFrame(np.arange(12).reshape((3, 4)),index=['Ohio', 'Colorado', 'New York'],columns=['one', 'two', 'three', 'four'])
data
#Using rename function
data.rename(index = {'Ohio':'SanF'}, columns={'one':'one_p','two':'two_p'},inplace=True)
data
#You can also use string functions
data.rename(index = str.upper, columns=str.title,inplace=True)
data
ages = [20, 22, 25, 27, 21, 23, 37, 31, 61, 45, 41, 32]
#Understand the output - '(' means the value is included in the bin, '[' means the value is excluded
bins = [18, 25, 35, 60, 100]
cats = pd.cut(ages, bins)
cats
#To include the right bin value, we can do:
pd.cut(ages,bins,right=False)
#Let's check how many observations fall under each bin
pd.value_counts(cats)
bin_names = ['Youth', 'YoungAdult', 'MiddleAge', 'Senior']
new_cats = pd.cut(ages, bins,labels=bin_names)

pd.value_counts(new_cats)
#we can also calculate their cumulative sum
pd.value_counts(new_cats).cumsum()
df = pd.DataFrame({'key1' : ['a', 'a', 'b', 'b', 'a'],
                   'key2' : ['one', 'two', 'one', 'two', 'one'],
                   'data1' : np.random.randn(5),
                   'data2' : np.random.randn(5)})
df
#calculate the mean of data1 column by key1
grouped = df['data1'].groupby(df['key1'])
grouped.mean()
dates = pd.date_range('20130101',periods=6)
df = pd.DataFrame(np.random.randn(6,4),index=dates,columns=list('ABCD'))
df
#get first n rows from the data frame
df[:3]
#slice based on date range
df['20130101':'20130104']
#slicing based on column names
df.loc[:,['A','B']]
#slicing based on both row index labels and column names
df.loc['20130102':'20130103',['A','B']]
#slicing based on index of columns
df.iloc[3] #returns 4th row (index is 3rd)
#returns a specific range of rows
df.iloc[2:4, 0:2]
#returns specific rows and columns using lists containing columns or row indexes
df.iloc[[1,5],[0,2]] 
df[df.A > 0.5]
#we can copy the data set
df2 = df.copy()
df2['E']=['one', 'one','two','three','four','three']
df2
#select rows based on column values
df2[df2['E'].isin(['two','four'])]
#select all rows except those with two and four
df2[~df2['E'].isin(['two','four'])]
#list all columns where A is greater than C
df.query('A > C')
#using OR condition
df.query('A < B | C > A')
#create a data frame
data = pd.DataFrame({'group': ['a', 'a', 'a', 'b','b', 'b', 'c', 'c','c'],
                 'ounces': [4, 3, 12, 6, 7.5, 8, 3, 5, 6]})
data
#calculate means of each group
data.pivot_table(values='ounces',index='group',aggfunc=np.mean)
#calculate count by each group
data.pivot_table(values='ounces',index='group',aggfunc='count')
#load the data
train  = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
#check data set
train.info()
test.info()
print ("The train data has",train.shape)
print ("The test data has",test.shape)
#Let have a glimpse of the data set
train.head()
nans = train.shape[0] - train.dropna().shape[0]
print ("%d rows have missing values in the train data" %nans)

nand = test.shape[0] - test.dropna().shape[0]
print ("%d rows have missing values in the test data" %nand)
#only 3 columns have missing values
train.isnull().sum()
cat = train.select_dtypes(include=['O'])
cat.apply(pd.Series.nunique)
#Education
train.workclass.value_counts(sort=True)
train.workclass.fillna('Private',inplace=True)


#Occupation
train.occupation.value_counts(sort=True)
train.occupation.fillna('Prof-specialty',inplace=True)


#Native Country
train['native.country'].value_counts(sort=True)
train['native.country'].fillna('United-States',inplace=True)
train.isnull().sum()
#check proportion of target variable
train.target.value_counts()/train.shape[0]
pd.crosstab(train.education, train.target,margins=True)/train.shape[0]
#load sklearn and encode all object type variables
from sklearn import preprocessing

for x in train.columns:
    if train[x].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train[x].values))
        train[x] = lbl.transform(list(train[x].values))
train.head()
#<50K = 0 and >50K = 1
train.target.value_counts()
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

y = train['target']
del train['target']

X = train
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=1,stratify=y)

#train the RF classifier
clf = RandomForestClassifier(n_estimators = 500, max_depth = 6)
clf.fit(X_train,y_train)
clf.predict(X_test)
#make prediction and check model's accuracy
prediction = clf.predict(X_test)
acc =  accuracy_score(np.array(y_test),prediction)
print ('The accuracy of Random Forest is {}'.format(acc))