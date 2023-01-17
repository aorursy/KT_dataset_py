import pandas as pd

import numpy as np
base_dataset = pd.read_csv("../input/Automobile_data.csv")
base_dataset
base_dataset.isna().sum()
from sklearn import preprocessing



def variables_creation(base_dataset,unique):

    

    cat=base_dataset.describe(include='object').columns

    

    cont=base_dataset.describe().columns

    

    x=[]

    

    for i in base_dataset[cat].columns:

        if len(base_dataset[i].value_counts().index)<unique:

            x.append(i)

    

    dummies_table=pd.get_dummies(base_dataset[x])

    encode_table=base_dataset[x]

    

    le = preprocessing.LabelEncoder()

    lable_encode=[]

    

    for i in encode_table.columns:

        le.fit(encode_table[i])

        le.classes_

        lable_encode.append(le.transform(encode_table[i]))

        

    lable_encode=np.array(lable_encode)

    lable=lable_encode.reshape(base_dataset.shape[0],len(x))

    lable=pd.DataFrame(lable)

    return (lable,dummies_table,cat,cont)
(lable,dummies_table,cat,cont)=variables_creation(base_dataset,8)
cont
base_dataset = base_dataset[base_dataset.describe().columns]
base_dataset.shape
base_dataset.drop('highway-mpg',axis=1).columns
base_dataset.shape
def outliers(df):

    import numpy as np

    import statistics as sts



    for i in df.describe().columns:

        x=np.array(df[i])

        p=[]

        Q1 = df[i].quantile(0.25)

        Q3 = df[i].quantile(0.75)

        IQR = Q3 - Q1

        LTV= Q1 - (1.5 * IQR)

        UTV= Q3 + (1.5 * IQR)

        for j in x:

            if j <= LTV or j>=UTV:

                p.append(sts.median(x))

            else:

                p.append(j)

        df[i]=p

    return df
outliers_treated=outliers(base_dataset[base_dataset.drop('highway-mpg',axis=1).columns])
outliers_treated.shape
import matplotlib.pyplot as plt

for i in outliers_treated:

    plt.hist(outliers_treated[i])

    plt.show()
outliers_treated=outliers_treated[outliers_treated.describe().columns]

outliers_treated['const']=1
outliers_treated
outliers_treated['target']=base_dataset['highway-mpg']

y=outliers_treated['target']

x=outliers_treated.drop('target',axis=1)
x
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
from sklearn.linear_model import LinearRegression

lm=LinearRegression()

lm.fit(X_train,y_train)

predicted_values=lm.predict(X_test)
sum(abs(predicted_values-y_test.values))
Final = pd.DataFrame(predicted_values)

y_test=y_test.reset_index()

y_test = y_test.drop('index',axis = 1)

Final['y_test'] = y_test

print(Final)
from sklearn.metrics import mean_absolute_error

MAE=mean_absolute_error(y_test.values,predicted_values)
from sklearn.metrics import mean_squared_error

MSE=mean_squared_error(y_test.values,predicted_values)
from sklearn.metrics import mean_squared_error

RMSE=np.sqrt(mean_squared_error(y_test.values,predicted_values))
MAPE=sum(abs((y_test.values-predicted_values)/(y_test.values)))/X_test.shape[0]
error_table=pd.DataFrame(predicted_values,y_test.values)
error_table.reset_index(inplace=True)
error_table.columns=['pred','actual']
error_table.plot(figsize=(20,8))