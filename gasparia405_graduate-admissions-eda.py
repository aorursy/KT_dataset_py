"""

The dataset contains several parameters which are considered important during 

the application for Masters Programs in India.

The parameters included are : 

1. GRE Scores ( out of 340 ) 

2. TOEFL Scores ( out of 120 ) 

3. University Rating ( out of 5 ) 

4. Statement of Purpose

5. Letter of Recommendation Strength ( out of 5 ) 

6. Undergraduate GPA ( out of 10 ) 

7. Research Experience ( either 0 or 1 ) 

8. Chance of Admit ( ranging from 0 to 1 )



I borrowed some of my ideas for data cleaning and exploration from Jonas Erthal's pipeline notebook at

https://www.kaggle.com/apollopower/grad-admissions-full-ml-pipeline-96-accuracy/notebook

"""



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set()
df = pd.read_csv("../input/Admission_Predict.csv")

df.head()
df.dtypes
df.set_index('Serial No.', inplace=True)

df.rename(index=str, columns={"Chance of Admit ": "Chance_of_Admit",

                             "GRE Score": "GRE_Score",

                             "TOEFL Score": "TOEFL_Score",

                             "University Rating": "University_Rating",

                             "SOP": "Statement_of_Purpose",

                             "LOR ": "Letter_of_Recommendation",

                             "CGPA": "College_GPA"}, inplace=True)

df.head()
df['Chance_of_Admit'].describe()
df.hist(column="Chance_of_Admit", figsize=(8,4))
plt.scatter(df['GRE_Score'], df['Chance_of_Admit'])

plt.title('Chance of Admit vs GRE Score')

plt.show()

plt.scatter(df['TOEFL_Score'], df['Chance_of_Admit'])

plt.title('Chance of Admit vs TOEFL Score')

plt.show()

plt.scatter(df['University_Rating'], df['Chance_of_Admit'])

plt.title('Chance of Admit vs University Rating')

plt.show()

plt.scatter(df['College_GPA'], df['Chance_of_Admit'])

plt.title('Chance of Admit vs CGPA')

plt.show()
for col in df.columns[:-1]:

    print('Chance of Admit vs {} : {}'.format(col, round(df['Chance_of_Admit'].corr(df[col]), 4)))
corr_matrix = df.corr()



sns.heatmap(corr_matrix)
sns.lmplot('College_GPA', 'Chance_of_Admit', data=df, hue='Research')
from sklearn.linear_model import LinearRegression

from sklearn import metrics
features = list(df.columns[:-1])



train_x = df[features]

train_y = df['Chance_of_Admit']



print(features)
linear_regression_model = LinearRegression()



linear_regression_model.fit(train_x, train_y)
pd.DataFrame(linear_regression_model.coef_, train_x.columns, columns=['Coefficient'])