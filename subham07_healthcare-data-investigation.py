# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df_train=pd.read_csv('/kaggle/input/av-healthcare-analytics-ii/healthcare/train_data.csv')

df_test=pd.read_csv('/kaggle/input/av-healthcare-analytics-ii/healthcare/test_data.csv')
print(df_train.dtypes)
df_train=df_train.drop(columns=['case_id','patientid'])

y_submit_1=df_test['case_id'].values # storing for final submission

df_test=df_test.drop(columns=['case_id','patientid'])
df_train.describe()
df_test.describe()
print(df_train.isna().sum())
print(df_test.isna().sum())
print(df_train['Bed Grade'].unique())



print(df_train['City_Code_Patient'].unique())
df_train['Bed Grade']=df_train['Bed Grade'].replace(np.nan,2)

df_test['Bed Grade']=df_test['Bed Grade'].replace(np.nan,2)



df_train['City_Code_Patient']=df_train['City_Code_Patient'].replace(np.nan,7)

df_test['City_Code_Patient']=df_test['City_Code_Patient'].replace(np.nan,7)
print(df_train.isna().sum())

print(df_test.isna().sum())
import plotly.express as px

fig = px.histogram(df_train, x="Hospital_code")

fig.show()
import plotly.express as px

fig = px.histogram(df_test, x="Hospital_code")

fig.show()
import plotly.express as px

fig = px.histogram(df_train, x="Hospital_type_code").update_xaxes(categoryorder="total descending")

fig.show()
import plotly.express as px

fig = px.histogram(df_test, x="Hospital_type_code").update_xaxes(categoryorder="total descending")

fig.show()
import plotly.express as px

fig = px.histogram(df_train, x="Hospital_region_code").update_xaxes(categoryorder="total descending")

fig.show()



import plotly.express as px

fig = px.histogram(df_test, x="Hospital_region_code").update_xaxes(categoryorder="total descending")

fig.show()
import plotly.express as px

fig = px.histogram(df_train, x="Department").update_xaxes(categoryorder="total descending")

fig.show()



import plotly.express as px

fig = px.histogram(df_test, x="Department").update_xaxes(categoryorder="total descending")

fig.show()
import plotly.express as px

fig = px.histogram(df_train, x="Ward_Type").update_xaxes(categoryorder="total descending")

fig.show()



import plotly.express as px

fig = px.histogram(df_test, x="Ward_Type").update_xaxes(categoryorder="total descending")

fig.show()
import plotly.express as px

fig = px.histogram(df_train, x="Type of Admission").update_xaxes(categoryorder="total descending").update_xaxes(categoryorder="total descending")

fig.show()



import plotly.express as px

fig = px.histogram(df_test, x="Type of Admission").update_xaxes(categoryorder="total descending")

fig.show()
import plotly.express as px

fig = px.histogram(df_train, x="Severity of Illness").update_xaxes(categoryorder="total descending")

fig.show()



import plotly.express as px

fig = px.histogram(df_test, x="Severity of Illness").update_xaxes(categoryorder="total descending")

fig.show()
import plotly.express as px

fig = px.histogram(df_train, x="Age").update_xaxes(categoryorder="total descending")

fig.show()



import plotly.express as px

fig = px.histogram(df_test, x="Age").update_xaxes(categoryorder="total descending")

fig.show()
import plotly.express as px

fig = px.histogram(df_train, x="Stay").update_xaxes(categoryorder="total descending")

fig.show()
import seaborn as sns

sns.set(style="white")



g=sns.barplot(y="Stay", x="Admission_Deposit", data=df_train,order=['0-10','11-20','21-30','31-40','41-50','51-60','61-70','71-80','81-90','91-100','More than 100 Days'])

import seaborn as sns

sns.set(style="white")



g=sns.barplot(y="Stay", x="Visitors with Patient", data=df_train,order=['0-10','11-20','21-30','31-40','41-50','51-60','61-70','71-80','81-90','91-100','More than 100 Days'])

df_train["Stay"].unique()
df_train["Age"].unique()
y_val=df_train.groupby(['Age','Stay']).count().reset_index()
y_val['count']=y_val['Hospital_code']
import plotly.express as px

df = px.data.gapminder()



fig = px.scatter(y_val, x="Age", y="Stay",

      size="count")

fig.show()