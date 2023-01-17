import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from collections import Counter
%matplotlib inline
import matplotlib.pyplot as plt # 
import seaborn as sns # visualization tool

import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.figure_factory as ff
import os
from sklearn.tree import DecisionTreeRegressor

test = "../input/BlackFriday.csv"
test_data = pd.read_csv(test)
print(test_data)
test_data.info()
test_data.columns
test_data.describe()
test_data.head(15)
print("Are there missing values? {}".format(test_data.isnull().any().any()))
#missing value control in features
test_data.isnull().sum()

total_miss = test_data.isnull().sum()
perc_miss = total_miss/test_data.isnull().count()*100

missing_data = pd.DataFrame({'Total missing':total_miss,
                            '% missing':perc_miss})

missing_data.sort_values(by='Total missing',
                         ascending=False).head(3)
print("Maximum Purchase: $ {0} ".format( test_data.Purchase.max()))
print("Minimum Purchase: $ {0} ".format( test_data.Purchase.min()))
print("Average Purchase: $ {0:.2f}".format( test_data.Purchase.mean()))
ax2 = test_data.add_subplot(122)
ax2.bar( ['Male', 'Female'], [male_purch/male_user, female_purch/female_user], color='#d6404c')
plt.title('Average Purchase by Gender')
plt.ylabel('Purchase')

plt.show()
test_data.Gender.replace(('M','F'),(0,1), inplace =True)
test_data.head()
Male = test_data[test_data['Gender'] ==0]['Purchase'].values.sum()
Female = test_data[test_data['Gender'] ==1]['Purchase'].values.sum()

print("Total Sales by Male: ", Male)
print("The Ratio of Male to Total: ", Male/(Male+Female))
print("Total Sales by Female:", Female)
print("The Ratio of Female to Total: ", Female/(Male+Female))
print("Data Based on Occupation:")
occupationStat = test_data['Occupation'].value_counts(dropna = False)
occupationStat.plot(kind='pie', figsize=(10,10))
test_data.Product_Category_1 = test_data.Product_Category_1.fillna(0)
test_data.Product_Category_2 = test_data.Product_Category_2.fillna(0)
test_data.Product_Category_3 = test_data.Product_Category_3.fillna(0)