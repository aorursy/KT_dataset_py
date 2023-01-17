import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn.linear_model import LinearRegression

from sklearn.svm import SVR

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor









# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/fifa19/data.csv')

df.head()
df.info()
#makes the wage and value objects into floats

def fix_value(Value):

    out = Value.replace('€', '')

    if 'M' in out:

        out = float(out.replace('M', ''))*1000000

    elif 'K' in Value:

        out = float(out.replace('K', ''))*1000

    return float(out)



df['Wage'] = df['Wage'].apply(lambda x: fix_value(x))

df['Value'] = df['Value'].apply(lambda x: fix_value(x))
df.describe().T
stats = ['Wage','Value','Overall','Potential', 'Special', 'Age','Weak Foot','Crossing', 'Finishing',

         'HeadingAccuracy','Volleys','Dribbling', 'Curve', 'BallControl','Vision',

         'ShortPassing','LongPassing', 'Acceleration', 'Aggression', 'ShotPower', 'Agility', 'Positioning', 

         'Composure', 'Marking', 'StandingTackle']



main_df = df[stats]



print(df[stats].isnull().sum())
for stat in stats: 

    main_df[stat].fillna(main_df[stat].median(), inplace=True)
fig = plt.figure(figsize=(10,10))

sns.distplot(a=df['Overall'], color = 'g', bins = 49)

plt.title('Distribution of Overall Ratings')

plt.show()
main_df = main_df[main_df['Overall'] > 70]
corr = main_df.corr()

fig, ax = plt.subplots(figsize=(15, 15))

mask = np.zeros_like(corr)

mask[np.triu_indices_from(mask)] = True

sns.heatmap(corr,annot=False, ax=ax,mask = mask,cmap="Greens").set(

    title='Feature Correlations')

plt.show()
#makes new dataframe for each position group

forwards = ['ST','LS', 'RS', 'LF','RF', 'CAM', 'CF','RW', 'LW']

midfield = ['CM', 'LCM', 'RCM', 'CDM', 'RDM', 'LDM','RM','LM','RAM','LAM']

defense = ['LB', 'RB', 'RCB', 'LCB', 'RWB', 'LWB']

goalie = ['GK']



forward_df = df[df['Position'].isin(forwards)]

mid_df = df[df['Position'].isin(midfield)]

defense_df = df[df['Position'].isin(defense)]

goalies = df[df['Position'].isin(goalie)]



#Makes new column in original df to have general position group

def group_positions(Position):

    if Position in forwards:

        return 'Forward'

    elif Position in midfield:

        return 'Midfield'

    elif Position in defense:

        return 'Defense'

    elif Position in goalie:

        return 'Goalkeeper'

    

main_df['Position Group'] = df['Position'].apply(lambda x: group_positions(x))
g = sns.pairplot(data=main_df, vars=['Overall','Wage', 'Age', 'Value', 'Dribbling', 'Agility','Composure'], hue='Position Group',palette="Paired")

g.fig.suptitle('Feature Relations')

plt.show()
fig = plt.figure(figsize=(10,10))

sns.violinplot(x="Position Group", y="Overall", data=main_df)

plt.title('Overall Rating by Position Group')

fig = plt.figure(figsize=(10,10))

sns.violinplot(x="Position Group", y="Wage", data=main_df)

plt.title('Wage by Position Group')

plt.show()
fig = plt.figure(figsize=(15,10))

sns.lineplot(x="Age", y="Overall", data=main_df,color='g')

plt.title('Age vs Overall Rating')

plt.show()
df.loc[(df['Overall']==71) & (df['Age']==45)]
fig = plt.figure(figsize=(15,10))

sns.lineplot(x="Age", y="Wage", data=main_df,color='g')

plt.title('Age vs Wage')

plt.show()
fig = plt.figure(figsize=(10,10))

sns.distplot(a=main_df['Wage'],kde=False, color = 'g')

plt.title('Distribution of Wages')

plt.show()
main_df = main_df[main_df.Wage != 0] #removes players with 0 wage, which allows us to scale logarithmically

main_df['logwage'] = np.log(main_df['Wage'])
fig = plt.figure(figsize=(10,10))

sns.distplot(a=main_df['logwage'],kde=False,color='g')

plt.title('Distribution of Log Wages')

plt.show()
regs = ['Overall','Special','Composure', 'Dribbling', 'Vision', 'ShortPassing', 'BallControl']

x = main_df[regs]

y = main_df['logwage']



X_train, X_test, y_train, y_test = train_test_split(x,y)
lr = LinearRegression()

lr.fit(X_train, y_train)

lr_prediction = lr.predict(X_test)



print('Linear Regression Performance ')

print('MAE: ', metrics.mean_absolute_error(y_test, lr_prediction))

print('RMSE: ', metrics.mean_squared_error(y_test, lr_prediction))

print('R2: ', metrics.r2_score(y_test, lr_prediction))
print('Coefficients')

for num, col in enumerate(x.columns,start = 0):

    print(col, ':', lr.coef_[num])
fig = plt.figure(figsize=(12,6))

plt.scatter(y_test, lr_prediction, color='teal')

plt.xlabel('Actual Player Log Wage')

plt.ylabel('Predicted Player Log Wage')

plt.title('Linear Regression Performance')

plt.show()
forward_df = forward_df[forward_df.Wage != 0]

forward_df['logwage'] = np.log(forward_df['Wage'])



main_stats = ['Overall', 'Special', 'Composure']

x2 = forward_df[main_stats]

y2 = forward_df['logwage']

X2_train, X2_test, y2_train, y2_test = train_test_split(x2,y2)



lr2 = LinearRegression()

lr2.fit(X2_train, y2_train)

lr2_prediction = lr2.predict(X2_test)



print('Linear Regression with Selected Variables Performance ')

print('MAE: ', metrics.mean_absolute_error(y2_test, lr2_prediction))

print('RMSE: ', metrics.mean_squared_error(y2_test, lr2_prediction))

print('R2: ', metrics.r2_score(y2_test, lr2_prediction))
print('Coefficients for Second Linear Model')

for num, col in enumerate(x2.columns,start = 0):

    print(col, ':', lr2.coef_[num])
svm = SVR(kernel = 'rbf')

svm.fit(X_train, y_train)

svm_prediction = svm.predict(X_test)



print('SVM Performance')

print('MAE: ', metrics.mean_absolute_error(y_test, svm_prediction))

print('RMSE: ', metrics.mean_squared_error(y_test, svm_prediction))

print('R2: ', metrics.r2_score(y_test, svm_prediction))
fig = plt.figure(figsize=(12,6))

plt.scatter(y_test, svm_prediction, color='teal')

plt.title('SVM Performance')

plt.xlabel('Actual Player Log Wage')

plt.ylabel('Predicted Player Log Wage')

plt.show()
tree_stats = ['Overall', 'Special', 'Composure', 'Dribbling', 'Vision', 'ShortPassing', 'LongPassing', 'Agility', 'BallControl', 'Dribbling', 'ShotPower', 'Positioning']

x3 = main_df[tree_stats]

y3 = main_df['logwage']

X3_train, X3_test, y3_train, y3_test = train_test_split(x3,y3)

dt = DecisionTreeRegressor(random_state=1)

dt.fit(X3_train,y3_train)

dt_prediction = dt.predict(X3_test)



print('Decision Tree Performance')

print('MAE: ', metrics.mean_absolute_error(y3_test, dt_prediction))

print('RMSE: ', metrics.mean_squared_error(y3_test, dt_prediction))

print('R2: ', metrics.r2_score(y3_test, dt_prediction))
fig = plt.figure(figsize=(12,6))

plt.scatter(y3_test, dt_prediction, color = 'teal')

plt.xlabel('Actual Player Log Wage')

plt.ylabel('Predicted Player Log Wage')

plt.title('Decision Tree Performance')

plt.show()
rf = RandomForestRegressor(n_estimators = 100)

rf.fit(X3_train,y3_train)

rf_prediction = rf.predict(X3_test)



print('Random Forest Performance')

print('MAE: ', metrics.mean_absolute_error(y3_test, rf_prediction))

print('RMSE: ', metrics.mean_squared_error(y3_test, rf_prediction))

print('R2: ', metrics.r2_score(y3_test, rf_prediction))
fig = plt.figure(figsize=(12,6))

plt.scatter(y3_test, rf_prediction, color= 'teal')

plt.xlabel('Actual Player Log Wage')

plt.ylabel('Predicted Player Log Wage')

plt.title('Random Forest Performance')

plt.show()