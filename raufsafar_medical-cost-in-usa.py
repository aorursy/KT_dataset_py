import numpy as np # linear algebra

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv('../input/insurance/insurance.csv')
df.head()
df.columns
# Looking for nomber of missing value in each column

df.isnull().sum()
#Calculate the average of the column

#Lets calculate average of age.

avg_age = df['age'].mean()

avg_age
#Replace NaN by mean value

df['age'].replace(np.nan, avg_age, inplace=True)
# Lets check how many regions we have

df['region'].unique()
#df['region'] = df.region you can use both 

df.region.value_counts()
df['age'].value_counts().sort_values(ascending=False)

#as we can see 18 and 19 years old people are in the majority
df['smoker'].value_counts()
df.dtypes
#We will use numerical data, so we should convert 'sex','smoker' amd 'region' to numerical data

# Encoding the data with map function



df['sex'] = df['sex'].map({'female':0,'male':1})

df['smoker'] = df['smoker'].map({'yes':1,'no':0})

df['region'] = df['region'].map({'southeast':0,'southwest':1,'northwest':2,'northeast':3})
df.head()
df.describe().T
sns.regplot(x ='age', y = 'charges', data =df)

# Weak Linear Relationship
df['region'].value_counts()
#Let's repeat the above steps but save the results to the dataframe "region_value_counts" and rename the column 

#'region' to 'value_counts'.

region_value_counts = df['region'].value_counts().to_frame()

region_value_counts.rename(columns={'region': 'value_counts'}, inplace=True)

region_value_counts
smoker_value_counts = df['smoker'].value_counts().to_frame()

smoker_value_counts.rename(columns ={'smoker': 'value_counts'},inplace =True)

smoker_value_counts
smoker_value_counts.index.name = 'Smoker'
smoker_value_counts
df['age'].unique()
#If we want to know, on average, which age_group and 'sex'  are charged more,

df_group_one = df[['age','sex','charges']]
df_group_one = df_group_one.groupby(['age','sex'], as_index =False).mean()
df_group_one
group_pivot = df_group_one.pivot(index = 'age',columns = 'sex')
group_pivot.T
df.corr()

#weak correlation
sns.boxplot(x="sex", y="charges", data=df)
from scipy import stats

#Let's calculate the Pearson Correlation Coefficient and P-value of 'age' and 'charges'.

pearson_coef, p_value = stats.pearsonr(df['age'], df['charges'])

print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value) 
#sex vs charges

#Let's calculate the Pearson Correlation Coefficient and P-value of 'sex' and 'charges'.

pearson_coef, p_value = stats.pearsonr(df['sex'], df['charges'])

print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value) 
#Binning the age column.

bins = [17,35,55,1000]

slots = ['Young adult','Senior Adult','Elder']



df['Age_range']=pd.cut(df['age'],bins=bins,labels=slots)
df.head()
# I can check the number of unique values is a column

# If the number of unique values <=40: Categorical column

# If the number of unique values in a columns> 50: Continuous



df.nunique().sort_values()
plt.figure(figsize=(25, 16))

plt.subplot(2,3,1)

sns.boxplot(x = 'smoker', y = 'charges', data = df)

plt.title('Smoker vs Charges',fontweight="bold", size=20)

plt.subplot(2,3,2)

sns.boxplot(x = 'children', y = 'charges', data = df,palette="husl")

plt.title('Children vs Charges',fontweight="bold", size=20)

plt.subplot(2,3,3)

sns.boxplot(x = 'sex', y = 'charges', data = df, palette= 'husl')

plt.title('Sex vs Charges',fontweight="bold", size=20)

plt.subplot(2,3,4)

sns.boxplot(x = 'region', y = 'charges', data = df,palette="bright")

plt.title('Region vs Charges',fontweight="bold", size=20)

plt.subplot(2,3,5)

sns.boxplot(x = 'Age_range', y = 'charges', data = df, palette= 'husl')

plt.title('Age vs Charges',fontweight="bold", size=20)

plt.show()
plt.figure(figsize=(12,6))

sns.barplot(x='region', y='charges', hue='sex', data=df, palette='Paired')

plt.show()
plt.figure(figsize=(12,6))

sns.barplot(x = 'region', y = 'charges',hue='smoker', data=df, palette='cool')

plt.show()
plt.figure(figsize=(12,6))

sns.barplot(x='region', y='charges', hue='children', data=df, palette='Set1')

plt.show()
plt.figure(figsize=(12,6))

sns.violinplot(x = 'children', y = 'charges', data=df, hue='smoker', palette='inferno')

plt.show()
#Heatmap to see correlation between variables

plt.figure(figsize=(12, 8))

sns.heatmap(df.corr(), cmap='RdYlGn', annot = True)

plt.title("Correlation between Variables")

plt.show()
sns.scatterplot(x=df['bmi'], y=df['charges'], hue=df['smoker'])
sns.lmplot(x="bmi", y="charges", hue="smoker", data=df)
sns.swarmplot(x=df['smoker'],y=df['charges'])
df_group_two=df_group_one[['age', 'charges']].groupby(['age'])

df_group_two.head()
df_group_two.get_group(18)['charges']

#we see the 18 years old female(0) and male(1) charges
# ANOVA

f_val, p_val = stats.f_oneway(df_group_two.get_group(20)['charges'], df_group_two.get_group(40)['charges'], df_group_two.get_group(60)['charges'])  

 

print( "ANOVA results: F=", f_val, ", P =", p_val)  
from sklearn.linear_model import LinearRegression
lm = LinearRegression()

lm
X = df[['age','sex','bmi','region']]

Y = df['charges']
lm.fit(X,Y)
Yhat = lm.predict(X)

Yhat[0:5]
lm.intercept_
lm.coef_
width = 6

height = 4

plt.figure(figsize=(width, height))

sns.regplot(x="bmi", y="charges", data=df)

plt.ylim(0,)
y_data = df['charges']
x_data =df.drop('charges',axis =1)
from sklearn.model_selection import train_test_split





x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.15, random_state=1)





print("number of test samples :", x_test.shape[0])

print("number of training samples:",x_train.shape[0])
#Use the function "train_test_split" to split up the data set such that 40% of the data samples will be utilized for testing, set the parameter "random_state" equal to zero. The output of the function should be the following: "x_train_1" , "x_test_1", "y_train_1" and "y_test_1".


x_train1, x_test1, y_train1, y_test1 = train_test_split(x_data, y_data, test_size=0.4, random_state=0) 

print("number of test samples :", x_test1.shape[0])

print("number of training samples:",x_train1.shape[0])
from sklearn.linear_model import LinearRegression
lre=LinearRegression()  
lre.fit(x_train[['bmi']], y_train)   # we fit the model using the feature bmi
lre.score(x_test[['bmi']], y_test)   # claculates th R^2 on the test data
lre.score(x_train[['bmi']], y_train)  # we can see the R^2 is much smaller using the test data
x_train1, x_test1, y_train1, y_test1 = train_test_split(x_data, y_data, test_size=0.1, random_state=0)

lre.fit(x_train1[['bmi']],y_train1)

lre.score(x_test1[['bmi']],y_test1)