import numpy as np

import pandas as pd

data_term_dep = pd.read_csv("../input/bank-marketing-term-deposit/bank_customer_survey.csv")

#Top 5 rows of the dataset is shown below.

data_term_dep.head()
#Dataset has 17columns and around 45k rows as shown below:

data_term_dep.shape
def memory_management(train_identity):

    """ iterate through all the columns of a dataframe and modify the data type

    to reduce memory usage."""



    df=train_identity

    start_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:

        col_type = df[col].dtype



        if col_type in [int,float]:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)

       # else:

        #    df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))

    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    print("*******************************************************************************************")

    train_identity=df

    return df
# size reduced for data and named as data1

data_term_dep1=memory_management(data_term_dep)
def null_values(base_dataset):

    print(base_dataset.isna().sum())

    ## null value percentage     

    null_value_table=(base_dataset.isna().sum()/base_dataset.shape[0])*100

    ## null value percentage beyond threshold drop , else treat the columns    

    retained_columns=null_value_table[null_value_table<30].index

    # if any variable as null value greater than input(like 30% of the data) value than those variable are consider as drop

    drop_columns=null_value_table[null_value_table>30].index

    base_dataset.drop(drop_columns,axis=1,inplace=True)

    len(base_dataset.isna().sum().index)

    cont=base_dataset.describe().columns

    cat=[i for i in base_dataset.columns if i not in base_dataset.describe().columns]

    for i in cat:

        base_dataset[i].fillna(base_dataset[i].value_counts().index[0],inplace=True)

    for i in cont:

        base_dataset[i].fillna(base_dataset[i].median(),inplace=True)

    print(base_dataset.isna().sum())

    return base_dataset,cat,cont
data_term_dep2,cat,cont=null_values(data_term_dep1)
def outliers_transform(base_dataset):

    for i in base_dataset.var().sort_values(ascending=False).index[0:10]:

        x=np.array(base_dataset[i])

        qr1=np.quantile(x,0.25)

        qr3=np.quantile(x,0.75)

        iqr=qr3-qr1

        utv=qr3+(1.5*(iqr))

        ltv=qr1-(1.5*(iqr))

        y=[]

        for p in x:

            if p <ltv or p>utv:

                y.append(np.median(x))

            else:

                y.append(p)

        base_dataset[i]=y
outliers_transform(data_term_dep2)
#Display the columns after outlier treatment

data_term_dep2.columns
dummy_columns=[]

for i in data_term_dep2.columns:

    if (data_term_dep2[i].nunique()>=3) & (data_term_dep2[i].nunique()<5):

        dummy_columns.append(i)
dummy_columns
#Dummy Variable

dummies_tables=pd.get_dummies(data_term_dep2[dummy_columns])
for i in dummies_tables.columns:

    data_term_dep2[i]=dummies_tables[i]
#Displaying columns after dummy variable creation

data_term_dep2.columns
#Drop the existing columns after the creation of dummy variable for those

data_term_dep2=data_term_dep2.drop(dummy_columns,axis=1)
data_term_dep2.columns
from sklearn.preprocessing import LabelEncoder

def label_encoders(data,cat):

    le=LabelEncoder()

    for i in cat:

        le.fit(data[i])

        x=le.transform(data[i])

        data[i]=x

    return data
data_new=data_term_dep2

cat=data_term_dep2.describe(include='object').columns
label_encoders(data_new,cat).head()
data_new.columns
import seaborn as sns

import matplotlib.pyplot as plt

plt.figure(figsize=(20,10))

for i in data_new.var().index:

    sns.distplot(data_new[i],kde=False)

    plt.show()
plt.figure(figsize=(20,10))

sns.heatmap(data_new.corr())
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import BaggingClassifier
y=data_new['y']

x=data_new.drop('y',axis=1)
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=43)
print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)
models=[DecisionTreeClassifier(),RandomForestClassifier(),BaggingClassifier()]
from sklearn.metrics import confusion_matrix,accuracy_score

final_accuracy_scores=[]

for i in models:

    dt=i

    dt.fit(X_train,y_train)

    dt.predict(X_test)

    dt.predict(X_train)

    print(confusion_matrix(y_test,dt.predict(X_test)))

    print(accuracy_score(y_test,dt.predict(X_test)))

    print(confusion_matrix(y_train,dt.predict(X_train)))

    print(accuracy_score(y_train,dt.predict(X_train)))

    print(i)

    final_accuracy_scores.append([i,confusion_matrix(y_test,dt.predict(X_test)),accuracy_score(y_test,dt.predict(X_test)),confusion_matrix(y_train,dt.predict(X_train)),accuracy_score(y_train,dt.predict(X_train))])

    from sklearn.model_selection import cross_val_score

    print(cross_val_score(i,X_train,y_train,cv=10))
final_accuracy_scores1=pd.DataFrame(final_accuracy_scores)
final_accuracy_scores1
from sklearn.metrics import accuracy_score

accuracy_score(y_test,dt.predict(X_test))