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
df = pd.read_excel("/kaggle/input/covid19-symptoms-dataset/covid19-symptoms-dataset.xlsx")

df.head()
x=df.iloc[:,:-1]

y=df.iloc[:,-1]

y = y.map({'Yes':1,'No':0})

print(x.head())

print(y.head())
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y)

print(x_train,y_train)
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_train)

from IPython.display import display_html 

y_pred=regressor.predict(x_train)

df1_styler = pd.DataFrame(y_train).style.set_table_attributes("style='display:inline'")

df2_styler = pd.DataFrame(y_pred,columns=['Predicted Outcome']).style.set_table_attributes("style='display:inline'")

display_html(df1_styler._repr_html_()+"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"+df2_styler._repr_html_(), raw=True)
y_pred_test=regressor.predict(x_test)

df1_styler = pd.DataFrame(y_test).style.set_table_attributes("style='display:inline'")

df2_styler = pd.DataFrame(y_pred_test,columns=['Predicted Outcome']).style.set_table_attributes("style='display:inline'")

display_html(df1_styler._repr_html_()+"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"+df2_styler._repr_html_(), raw=True)