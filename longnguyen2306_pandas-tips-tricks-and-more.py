import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("../input"))
print(pd.__version__)

print(np.__version__)
import warnings

warnings.filterwarnings("ignore")
data = pd.read_csv("../input/titanic/train.csv")
#peek top

data.head()
#peek tail

data.tail()
values = {}

arr = []

print('values is a ' ,type(values))

type(arr)
def getColumnsWithMissingValuesList(df):

    return [col for col in df.columns if df[col].isnull().any()] 



getColumnsWithMissingValuesList(data)
def getObjectColumnsList(df):

    return [cname for cname in df.columns if df[cname].dtype == "object"]



cat_cols = getObjectColumnsList(data)

cat_cols
def getNumericColumnsList(df):

    return [cname for cname in df.columns if df[cname].dtype in ['int64', 'float64']]



num_cols = getNumericColumnsList(data)

num_cols
def getLowCardinalityColumnsList(df,cardinality):

    return [cname for cname in df.columns if df[cname].nunique() < cardinality and df[cname].dtype == "object"]



LowCardinalityColumns = getLowCardinalityColumnsList(data,10)

LowCardinalityColumns
data['Embarked'].nunique()
def PerformOneHotEncoding(df,columnsToEncode):

    return pd.get_dummies(df,columns = columnsToEncode)



oneHotEncoded_df = PerformOneHotEncoding(data,getLowCardinalityColumnsList(data,10))

oneHotEncoded_df.head()
# select only int64 & float64 columns

numeric_data = data.select_dtypes(include=['int64','float64'])



# select only object columns

categorical_data = data.select_dtypes(include='object')
numeric_data.head()
categorical_data.head()
def missingValuesInfo(df):

    total = df.isnull().sum().sort_values(ascending = False)

    percent = round(df.isnull().sum().sort_values(ascending = False)/len(df)*100, 2)

    temp = pd.concat([total, percent], axis = 1,keys= ['Total', 'Percentage'])

    return temp.loc[(temp['Total'] > 0)]



missingValuesInfo(data)
# for Object columns fill using 'UNKOWN'

# for Numeric columns fill using median

def HandleMissingValues(df):

    num_cols = [cname for cname in df.columns if df[cname].dtype in ['int64', 'float64']]

    cat_cols = [cname for cname in df.columns if df[cname].dtype == "object"]

    values = {}

    for a in cat_cols:

        values[a] = 'UNKOWN'



    for a in num_cols:

        values[a] = df[a].median()

        

    df.fillna(value=values,inplace=True)

    

    

HandleMissingValues(data)

data.head()
#check for NaN values

data.isnull().sum().sum()
def logistic(X,y):

    X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42,test_size=0.2)

    lr=LogisticRegression()

    lr.fit(X_train,y_train)

    y_pre=lr.predict(X_test)

    print('Accuracy : ',accuracy_score(y_test,y_pre))
series = data['Fare']

d = {series.name : series}

df = pd.DataFrame(d) 

df.head()
PassengerClass = data['Pclass'].astype('category')

PassengerClass.describe()