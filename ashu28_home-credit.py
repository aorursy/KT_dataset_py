# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd
base_dataset=pd.read_csv("../input/application_train.csv")
base_dataset.head()
base_dataset=base_dataset.sample(1000)
base_dataset.reset_index(inplace=True)
base_dataset.drop(['index','SK_ID_CURR'],axis=1,inplace=True)
base_dataset.shape
base_dataset.head()
base_dataset.shape
def nullvalue_function(base_dataset,percentage):
    
    # Checking the null value occurance
    
    print(base_dataset.isna().sum())

    # Printing the shape of the data 
    
    print(base_dataset.shape)
    
    # Converting  into percentage table
    
    null_value_table=pd.DataFrame((base_dataset.isna().sum()/base_dataset.shape[0])*100).sort_values(0,ascending=False )
    
    null_value_table.columns=['null percentage']
    
    # Defining the threashold values 
    
    null_value_table[null_value_table['null percentage']>percentage].index
    
    # Drop the columns that has null values more than threashold 
    base_dataset.drop(null_value_table[null_value_table['null percentage']>30].index,axis=1,inplace=True)
    
    # Replace the null values with median() # continous variables 
    for i in base_dataset.describe().columns:
        base_dataset[i].fillna(base_dataset[i].median(),inplace=True)
    # Replace the null values with mode() #categorical variables
    for i in base_dataset.describe(include='object').columns:
        base_dataset[i].fillna(base_dataset[i].value_counts().index[0],inplace=True)
  
    print(base_dataset.shape)
    
    return base_dataset
base_dataset_null=nullvalue_function(base_dataset,30)
base_dataset_null.shape
base_dataset_null
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
import numpy as np
(lable,dummies_table,cat,cont)=variables_creation(base_dataset_null,8)
cat
lable.rename(columns={0:'NAME_CONTRACT_TYPE'},inplace=True)
lable.rename(columns={1:'CODE_GENDER'},inplace=True)
lable.rename(columns={2:'FLAG_OWN_CAR'},inplace=True)
lable.rename(columns={3:'FLAG_OWN_REALTY'},inplace=True)
lable.rename(columns={4:'NAME_TYPE_SUITE'},inplace=True)
lable.rename(columns={5:'NAME_INCOME_TYPE'},inplace=True)
lable.rename(columns={6:'NAME_EDUCATION_TYPE'},inplace=True)
lable.rename(columns={7:'NAME_FAMILY_STATUS'},inplace=True)
lable.rename(columns={8:'NAME_HOUSING_TYPE'},inplace=True)
lable.rename(columns={9:'WEEKDAY_APPR_PROCESS_START'},inplace=True)
lable.rename(columns={10:'ORGANIZATION_TYPE'},inplace=True)
for i in lable.columns:
    base_dataset_null[i]=lable[i]
base_dataset_null.shape
base_dataset_null.drop('AMT_CREDIT',axis=1).columns
base_dataset_null.var().sort_values(ascending=False).head(20).index[1:]
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
base_dataset_null.shape
outliers_treated=outliers(base_dataset_null[base_dataset_null.drop('AMT_CREDIT',axis=1).columns])
outliers_treated.shape
def univariate_analysis(base_null_value_treated):
    import matplotlib.pyplot as plt
    col=[]
    for i in base_null_value_treated.describe().columns:
        var=base_null_value_treated[i].value_counts().values.var()
        col.append([i,var])
        variance_table=pd.DataFrame(col)
        variance_table[variance_table[1]>100][0].values
    return variance_table[variance_table[1]>100][0].values
viz=outliers_treated[['AMT_INCOME_TOTAL','AMT_ANNUITY','AMT_GOODS_PRICE']]
df_columns=univariate_analysis(viz)
import matplotlib.pyplot as plt
for i in df_columns:
    plt.hist(outliers_treated[i])
    plt.show()
outliers_treated.columns
import matplotlib.pyplot as plt
import seaborn as sns
for i in df_columns:
    for j in df_columns:
        if i!=j:
            sns.jointplot(outliers_treated[i],outliers_treated[j])
            plt.show()
outliers_treated=outliers_treated[outliers_treated.describe().columns]
outliers_treated['const']=1
outliers_treated['target']=base_dataset_null['AMT_CREDIT']
y=outliers_treated['target']
x=outliers_treated.drop('target',axis=1)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

from sklearn.linear_model import LinearRegression
lm=LinearRegression()
lm.fit(X_train,y_train)
lm.predict(X_test)
predicted_values=lm.predict(X_test)
sum(abs(predicted_values-y_test.values))
from sklearn.metrics import mean_absolute_error
MAE=mean_absolute_error(y_test.values,predicted_values)
from sklearn.metrics import mean_squared_error
MSE=mean_squared_error(y_test.values,predicted_values)
from sklearn.metrics import mean_squared_error
RMSE=np.sqrt(mean_squared_error(y_test.values,predicted_values))
MAPE=sum(abs((y_test.values-predicted_values)/(y_test.values)))/X_test.shape[0]
def regression_model(predicted_values,y_test):
    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import r2_score
    total_error=sum(abs(predicted_values-y_test.values))
    MSE=mean_absolute_error(y_test.values,predicted_values)
    MAE=mean_squared_error(y_test.values,predicted_values)
    RMSE=np.sqrt(mean_squared_error(y_test.values,predicted_values))
    MAPE=sum(abs((y_test.values-predicted_values)/(y_test.values)))/X_test.shape[0]
    r2=r2_score(predicted_values,y_test)
    print("total error",total_error)
    print("MSE",MSE)
    print("MAE",MAE)
    print("RMSE",RMSE)
    print("MAPE",MAPE)
    print("R2",r2)
regression_model(predicted_values,y_test)
error_table=pd.DataFrame(lm.predict(X_test),y_test.values)
error_table.reset_index(inplace=True)
error_table.columns=['pred','actual']
error_table.plot(figsize=(20,8))
