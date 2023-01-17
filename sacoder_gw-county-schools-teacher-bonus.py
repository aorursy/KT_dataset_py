# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy.random as np
import sys
import matplotlib 
import seaborn as sns
import numpy as np
from subprocess import check_output
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

%matplotlib inline

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# Location of file
Location = '../input/Gwinnett County Teacher Bonus2.csv'

df = pd.read_csv(Location)

df.info()
df.head(5)
#Clean up data

#Title 1 school has Yes and NaN, replace NaN with 0, change Yes to 1
df["Title 1 School"].fillna(0, inplace=True)
df.loc[(df['Title 1 School'] == 'Yes'),'Title 1 School']='1'
df["Title 1 School"] = df["Title 1 School"].astype(str).astype(int)

#remove NaN valued rows
df.dropna(subset = ["% of Hispanic+Black"], inplace=True)

#change the % Bonus @ School column to a numeric value by taking out the % sign
#df["% Bonus @ School"]= df["% Bonus @ School"].astype(str).replace("%", "") 
df["% Bonus @ School"]= df["% Bonus @ School"].astype(str).replace('[\%,]','',regex=True)
df["% Bonus @ School"] = df["% Bonus @ School"].astype(str).astype(float)

#clean up the Total Bonus value as well
df["Total School Bonus"] = df["Total School Bonus"].astype(str).replace(",", "") 
df["Total School Bonus"] = df["Total School Bonus"].astype(str).replace('[\$,]','',regex=True)
df["Total School Bonus"] = df["Total School Bonus"].astype(str).astype(float)

df.info()
df.isna().sum()
df.dtypes
df.describe()
df.head(20)
plt.figure(figsize = (10,6))

#see the distribution of % Bonus wrt % Hispanic+Black
plt.scatter(df['% of Hispanic+Black'], df['% Bonus @ School'])
plt.title("Scatter plot: Hispanic+Black % vs % Bonus at School")
plt.xlabel('% of Hispanic+Black', fontsize=10)
plt.ylabel('% Bonus @ School', fontsize=10)

z = np.polyfit(df['% of Hispanic+Black'], df['% Bonus @ School'], 1)
p = np.poly1d(z)
plt.plot(df['% of Hispanic+Black'], p(df['% of Hispanic+Black']), 'm-')
plt.show()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,6))
fig.suptitle('Scatter plot of % School Bonus')

#see the distribution of % Bonus wrt % Hispanic+Black
ax1.scatter(df['% of Hispanic+Black'], df['% Bonus @ School'])
ax1.set(xlabel='% of Hispanic+Black', ylabel='% Bonus @ School')

z = np.polyfit(df['% of Hispanic+Black'], df['% Bonus @ School'], 1)
p = np.poly1d(z)
ax1.plot(df['% of Hispanic+Black'], p(df['% of Hispanic+Black']), 'm-')
#plt.show()

#see the distribution of % Bonus wrt % Eligible Free Lunch
ax2.scatter(df['% Eligible Free Lunch'], df['% Bonus @ School'])
ax2.set(xlabel='% Eligible Free Lunch', ylabel='% Bonus @ School')

z = np.polyfit(df['% Eligible Free Lunch'], df['% Bonus @ School'], 1)
p = np.poly1d(z)
ax2.plot(df['% Eligible Free Lunch'], p(df['% Eligible Free Lunch']), 'm-')
#plt.show()
#create bins for the % Hispanic+Black and see the Average % Bonus for each bin
df['% of Hispanic+Black Bin'] = np.where((df['% of Hispanic+Black'] > 0) & (df['% of Hispanic+Black'] <= 20), '0 - 20', '')
df['% of Hispanic+Black Bin'] = np.where((df['% of Hispanic+Black'] > 20) & (df['% of Hispanic+Black'] <= 40), '20 - 40', df['% of Hispanic+Black Bin'] )
df['% of Hispanic+Black Bin'] = np.where((df['% of Hispanic+Black'] > 40) & (df['% of Hispanic+Black'] <= 60), '40 - 60', df['% of Hispanic+Black Bin'] )
df['% of Hispanic+Black Bin'] = np.where((df['% of Hispanic+Black'] > 60) & (df['% of Hispanic+Black'] <= 80), '60 - 80', df['% of Hispanic+Black Bin'] )
df['% of Hispanic+Black Bin'] = np.where((df['% of Hispanic+Black'] > 80) & (df['% of Hispanic+Black'] <= 1000), '80 - 100', df['% of Hispanic+Black Bin'] )

df1 = df[['% of Hispanic+Black', '% of Hispanic+Black Bin']]

df2 = df.groupby('% of Hispanic+Black Bin', as_index=False)['% Bonus @ School'].mean()
#df2.plot.figure(figsize = (10,10))
df2.plot.bar(x="% of Hispanic+Black Bin", y="% Bonus @ School", rot=70, title="Average % School Bonus", figsize=(10,4));
#plt.show();

#see the Average bonus for School Types 
df3 = df.groupby('Type', as_index=False)['% Bonus @ School'].mean().round(2)
df3.plot.barh(x="Type", y="% Bonus @ School", rot=70, title="Average % School Bonus", figsize=(10,6));

for index, value in enumerate(df3['% Bonus @ School']):
    plt.text(value, index, str(value))
    
#plt.show();

#see the Average bonus for Cluster
df4 = df.groupby('Cluster', as_index=False)['% Bonus @ School'].mean()
df4.plot.barh(x="Cluster", y="% Bonus @ School", rot=70, title="Average % School Bonus", figsize=(10,10));
#plt.show();

#see the Average bonus for Title 1 school vs non-Title 1 school
df5 = df.groupby('Title 1 School', as_index=False)['% Bonus @ School'].mean()
df5.plot.bar(x="Title 1 School", y="% Bonus @ School", rot=70, title="Average % School Bonus", figsize=(10,4));
#plt.show();
print(df['Type'].unique())
print(df['Cluster'].unique())
#Analyze cluster data further 
df6 = df.groupby(
   ['Cluster']
).agg(
    {
         '% of Hispanic+Black':'mean',    # get the mean of % Black+Hispanic
         '% Bonus @ School': 'mean',  # get the mean of % Bonus
    }
)

sorted_df = df6.sort_values(['% of Hispanic+Black'], ascending = [True])    

sorted_df.plot.barh(rot=70, title="Average % School Bonus and Average % Black+Hispanic Across clusters", figsize=(10,10));
#Analyze cluster data further 
df6 = df.groupby(
   ['Cluster']
).agg(
    {
         '% Eligible Free Lunch':'mean',    # get the mean of % Black+Hispanic
         '% Bonus @ School': 'mean',  # get the mean of % Bonus
    }
)

sorted_df = df6.sort_values(['% Eligible Free Lunch'], ascending = [True])    

sorted_df.plot.barh(rot=70, title="Average % School Bonus and Average % Ecnomically Disadvantaged Across clusters", figsize=(10,10));
#perform a linear regression and examine the coefficients
X = df[['# of Teachers', 'Title 1 School', '% of Hispanic+Black', '% Eligible Free Lunch']]
print(X.head())
X.info()


#X_std = StandardScaler().fit_transform(X)
X_std = MinMaxScaler().fit_transform(X)
y = df['% Bonus @ School']

print(X_std[:10])
print(y.head(10))

model = LinearRegression()
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.18, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model.fit(X_train,y_train)

print(model.coef_) 

plt.barh([0,1,2,3],model.coef_)
plt.yticks(range(4),X.columns[0:4], fontsize = 10)
plt.title('Regression Coefficients: Bonus')

plt.show()

#print(X.columns[0:4])
print('R^2 on training...',model.score(X_train,y_train))
print('R^2 on test...',model.score(X_test,y_test))

print('Model Coefficients',model.coef_)
print('Model Intercept',model.intercept_)  
#Examine the beeswarm plot

plt.figure(figsize = (10,6))
ax = sns.swarmplot(x='Title 1 School', y='% Bonus @ School', data=df, size=6)

plt.figure(figsize = (10,6))
ax = sns.boxplot(x="Title 1 School", y="% Bonus @ School", data=df, palette="Set3")
df.corr()
def ecdf(data):
    return np.sort(data), np.arange(1, len(data)+1) / len(data)

plt.figure(figsize = (10,6))

X1, y1 = ecdf(df[df['Title 1 School']==1]['% Bonus @ School'])
plt.plot(X1, y1, marker='D', linestyle='none', label='Title 1 School')

X0, y0 = ecdf(df[df['Title 1 School']==0]['% Bonus @ School'])
plt.plot(X0, y0, marker='D', linestyle='none', label='Non-Title 1 School')

plt.xlabel('% Bonus at School')
plt.ylabel('ECDF')
plt.legend()
plt.show()


#perform a permutation test to see if the difference of mean bonuses at school for Title 1 vs non-Title1 is significant

!pip install dc_stat_think

import dc_stat_think as dcst

diff_means_exp = np.mean(df[df['Title 1 School']==0]['% Bonus @ School']) - np.mean(df[df['Title 1 School']==1]['% Bonus @ School'])
print(diff_means_exp)

perm_reps = dcst.draw_perm_reps(df[df['Title 1 School']==0]['% Bonus @ School'], df[df['Title 1 School']==1]['% Bonus @ School'], dcst.diff_of_means, size=10000)

# Compute the p-value: p-val
p_val = np.sum(perm_reps >= diff_means_exp) / len(perm_reps)

# Print the result
print('p =', p_val)