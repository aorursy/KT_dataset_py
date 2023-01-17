df_train = pd.read_csv('C:/Users/teren/Documents/Python/Practice/US Births 2018 (Regression)/US_Births(2018).csv')
def drop_rows(df_train):
    '''
    Dropping rows where missing: 'DBWT', 'BMI', 'DBWT', 'WTGAIN', 'PWgt_R', 'DLMP_MM', 'DLMP_YY'
    '''
    df_train.drop(df_train[df_train['DBWT'].eq(9999)].index, inplace=True)    
    df_train.drop(df_train[df_train['BMI'].eq(99.9)].index, inplace=True)
    df_train.drop(df_train[df_train['DBWT'].eq(9999)].index, inplace=True)
    df_train.drop(df_train[df_train['WTGAIN'].eq(99)].index, inplace=True)
    df_train.drop(df_train[df_train['PWgt_R'].eq(999)].index, inplace=True)
    df_train.drop(df_train[df_train['DLMP_MM'].eq(99)].index, inplace=True)
    df_train.drop(df_train[df_train['DLMP_YY'].eq(9999)].index, inplace=True)
    
    df_train.drop(columns=['IMP_SEX'], inplace=True)
    return df_train
df_train = drop_rows(df_train)
#creating new column 'pregnancy_length': An estimation of the gestation period by subtracting the month/year of last menses from month/year of baby born
conditions = [(df_train['DOB_MM'] > df_train['DLMP_MM']) & (2018 == df_train['DLMP_YY']),
                  (df_train['DOB_MM'] > df_train['DLMP_MM']) & (2018 > df_train['DLMP_YY']),
                  (df_train['DOB_MM'] < df_train['DLMP_MM']) & (2018 > df_train['DLMP_YY'])]
choices = [df_train['DOB_MM'] - df_train['DLMP_MM'],
               ((df_train['DOB_YY'] - df_train['DLMP_YY'])* 12) + df_train['DOB_MM'] - df_train['DLMP_MM'],
               ((df_train['DOB_YY'] - df_train['DLMP_YY'])* 12) - df_train['DLMP_MM'] + df_train['DOB_MM']]
df_train['pregnancy_length'] = np.select(conditions,choices, 12)
df_train.describe()
sns.distplot(df_train['DBWT'])
plt.figure(figsize=(6,5))
sns.distplot(df_train[df_train['SEX'] == 'F']['DBWT'], label = 'Female')
sns.distplot(df_train[df_train['SEX'] == 'M']['DBWT'], label = 'Male')
plt.title('Distribution of Baby Weight Separated by Gender')
plt.xlabel('Baby Weight in Grams')
plt.legend();
#Find the distirbution of weights between the sexes
stats.f_oneway(df_train[df_train['SEX'] == 'F']['DBWT'],
               df_train[df_train['SEX'] == 'M']['DBWT'])
#Assign the value 1 if male in variable Sex_M
df_train['SEX_M'] = np.where(df_train['SEX'] == 'M', 1, 0)
df_train['SEX_M'].describe()
#Let's determine if smoking is a factor in baby weight
#First, let's create a dummy variable with var CIG_0
df_train.loc[df_train['CIG_0'].between(1, 98), 'Smoking Habit'] = 1
df_train.loc[df_train['CIG_0'].eq(0), 'Smoking Habit'] = 0
df_train['DBWT'].groupby(df_train['Smoking Habit']).describe()
#Find the distirbution of baby weights by the mothers' previous smoking habits
plt.figure(figsize=(6,5))
sns.distplot(df_train[df_train['Smoking Habit'].eq(0)]['DBWT'], label = 'Never Smoked')
sns.distplot(df_train[df_train['Smoking Habit'].eq(1)]['DBWT'], label = 'Smoked Daily')
plt.title('Distribution of Baby Weight Separated by Mother Previous Smoking Habit')
plt.xlabel('Baby Weight in Grams')
plt.legend();

stats.f_oneway(df_train[df_train['Smoking Habit'].eq(0)]['DBWT'],
               df_train[df_train['Smoking Habit'].eq(1)]['DBWT'])
fig, ax = plt.subplots(figsize=(24,10))
sns.boxplot(x='MRAVE6',y='DBWT', data=df_train, palette='muted')
ax.set_title('Distribution of Baby Weight Separated by Race')
ax.set_xlabel('Mother\'s Race')
ax.set_ylabel('Baby Weight in Grams')
ax.set_xticklabels(['White(only)','Black(only)','AIAN(only)','Asian(only)','NHOPI(only)','More than one race']);
_X =pd.get_dummies(df_train, columns=[ 'MRAVE6', 'RDMETH_REC'
                                ]).copy()
pd.set_option('display.max_columns', 500)
_X.head()
# null: Baby Weights of Moms of different Race are equal
# alt: Baby Weights of Moms of different Race are NOT equal
# alpha: 0.05
stats.f_oneway(_X[_X['MRAVE6_1'].eq(1)]['DBWT'],
              _X[_X['MRAVE6_2'].eq(1)]['DBWT'],
              _X[_X['MRAVE6_3'].eq(1)]['DBWT'],
              _X[_X['MRAVE6_4'].eq(1)]['DBWT'],
              _X[_X['MRAVE6_5'].eq(1)]['DBWT'],
              _X[_X['MRAVE6_6'].eq(1)]['DBWT'])
# reject null. There is significant evidence to suggest that the all race babies are not the same.
fig, ax = plt.subplots(figsize=(24,10))
sns.boxplot(x='pregnancy_length',y='DBWT', data=df_train, palette='muted')
ax.set_title('Distribution of Baby Weight Separated by Gestation Period')
ax.set_xlabel('Mother\'s Gestation Period')
ax.set_ylabel('Baby Weight in Grams')
#dropping rows that with gestation period greater than 12 and less than 5, treating them as outliers
df_train.drop(df_train[df_train['pregnancy_length'].gt(12)].index,inplace=True)
df_train.drop(df_train[df_train['pregnancy_length'].lt(5)].index,inplace=True)
fig, ax = plt.subplots(figsize=(14,10))
sns.boxplot(x='pregnancy_length',y='DBWT', data=df_train, palette='muted')
ax.set_title('Distribution of Baby Weight Separated by Gestation Period')
ax.set_xlabel('Mother\'s Gestation Period')
ax.set_ylabel('Baby Weight in Grams')
# null: Baby weights of all gestation periods are equal
# alt: Baby weights of all gestation periods are NOT equal
# alpha: 0.05
stats.f_oneway(df_train[df_train['pregnancy_length'].eq(5)]['DBWT'],
              df_train[df_train['pregnancy_length'].eq(6)]['DBWT'],
              df_train[df_train['pregnancy_length'].eq(7)]['DBWT'],
              df_train[df_train['pregnancy_length'].eq(8)]['DBWT'],
              df_train[df_train['pregnancy_length'].eq(9)]['DBWT'],
               df_train[df_train['pregnancy_length'].eq(10)]['DBWT'],
               df_train[df_train['pregnancy_length'].eq(11)]['DBWT'],
               df_train[df_train['pregnancy_length'].eq(12)]['DBWT'])
# reject null. There is significant evidence to suggest that that the length of gestation has an effect on baby weight
df_train['DBWT'].groupby(df_train['pregnancy_length']).describe()
fig, ax = plt.subplots(figsize=(14,10))
sns.boxplot(x='pregnancy_length',y='DBWT',data=_X, hue = 'RDMETH_REC_3',palette = 'muted', ax=ax)
handles, _ = ax.get_legend_handles_labels()
ax.legend(loc='upper right', handles = handles, labels = ['No Cesar', 'Yes Cesar'])
ax.set_title('Baby Weight vs Total Months of Gestation with-without C-Section')
ax.set_ylabel('Baby Weight by Grams')
ax.set_xlabel('Total Months of Gestation');
# defining the variables 
x = _X[['MRAVE6_1', 'MRAVE6_2', 'MRAVE6_3']]
y = _X['DBWT']
# adding the constant term 
x = sm.add_constant(x) 
# performing the regression 
# and fitting the model 
result = sm.OLS(y, x).fit() 
  
# printing the summary table 
print(result.summary()) 
