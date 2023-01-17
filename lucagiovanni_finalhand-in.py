import numpy as np

import pandas as pd

import scipy as sp

import seaborn as sns

from sklearn import preprocessing

from sklearn.model_selection import KFold

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import RandomizedSearchCV





#import your classifiers here



import matplotlib

import matplotlib.pyplot as plt

%matplotlib inline
#Train

df = pd.read_csv('data/flu_train.csv')

df = df[~np.isnan(df['flu'])]

df.head()
#Test

df_test = pd.read_csv('data/flu_test.csv')

df_test.head()
#What's up in each set



x = df.values[:, :-1]

y = df.values[:, -1]



x_test = df_test.values[:, :-1]



print('x train shape:', x.shape)

print('x test shape:', x_test.shape)

print('train class 0: {}, train class 1: {}'.format(len(y[y==0]), len(y[y==1])))
df.describe()
df.dtypes
df['Gender'].isna().sum()
df['Gender'].value_counts().plot.bar(rot=0)

plt.show()
df['Age'].isna().sum()
df['Age'].hist(bins=16)

plt.show()
df['Race1'].isna().sum()
df['Race1'].value_counts().plot.bar(rot=0)

plt.show()
df['Education'].isna().sum()
df['Education'].value_counts().plot.bar(rot=45)

plt.show()
df.loc[df['Education'].isna()]['Age'].hist(bins=16)

plt.show()
dummies = pd.get_dummies(df['Education'])

pd.concat([dummies, df['Age']], axis=1).corr()['Age']
df['MaritalStatus'].isna().sum()
df['MaritalStatus'].value_counts().plot.bar(rot=45)

plt.show()
df.loc[df['MaritalStatus'].isna()]['Age'].hist(bins=16)

plt.show()
df['HHIncome'].isna().sum()
df['HHIncome'].value_counts().plot.bar(rot=90)

plt.show()
df['HHIncomeMid'].isna().sum()
df['HHIncomeMid'].value_counts().sort_index().plot.bar(rot=90)

plt.show()
df['Poverty'].isna().sum()
df['Poverty'].hist(bins=10)

plt.show()
df['HomeRooms'].isna().sum()
df['HomeRooms'].hist(bins=12)

plt.show()
df['HomeOwn'].isna().sum()
df['HomeOwn'].value_counts().plot.bar(rot=0)

plt.show()
df.loc[df['HomeOwn'].isna()]['Age'].hist(bins=8)

plt.show()
df['Work'].isna().sum()
df['Work'].value_counts().plot.bar(rot=0)

plt.show()
len(df[(df['Work'].isna()) & (df['Education'].isna()) & (df['Age']<=20)])
df['Weight'].isna().sum()
df['Weight'].hist(bins=20)

plt.show()
df.loc[df['Weight'].isna()]['Age'].hist(bins=8)

plt.show()
df['Length'].isna().sum()
df['Length'].hist()

plt.show()
df.loc[df['Length'].notna()]['Age'].value_counts().plot.bar()

plt.show()
df['HeadCirc'].notna().sum()
df['HeadCirc'].hist()

plt.show()
df.loc[df['HeadCirc'].notna()]['Age'].value_counts().plot.bar(rot=0)

plt.show()
len(df.loc[df['Age']==0])
df['Height'].isna().sum()
df['Height'].hist(bins=30)

plt.show()
df.loc[df['Height'].isna()]['Age'].hist(bins=32)

plt.show()
df.loc[(df['Height'].isna()) & (df['Length'].isna())]['Age'].hist(bins=16)

plt.show()
print('Values present in both Height and Length', len(df[(df['Height'].notna()) & (df['Length']).notna()]))

print('Values present in neither Height and Length', len(df[(df['Height'].isna()) & (df['Length']).isna()]))
df['BMI'].isna().sum()
df['BMI'].hist()

plt.show()
df.loc[df['BMI'].isna()]['Age'].hist(bins=16)

plt.show()
df.loc[df['Age'] < 20]['BMICatUnder20yrs'].isna().sum()
df['BMICatUnder20yrs'].value_counts().plot.bar(rot=0)

plt.show()
df['BMI_WHO'].isna().sum()
df['BMI_WHO'].value_counts().plot.bar(rot=0)

plt.show()
df['Pulse'].isna().sum()
df['Pulse'].hist()

plt.show()
df.loc[df['Pulse'].isna()]['Age'].hist(bins=32)

plt.show()
df.plot.scatter('Age', 'Pulse')

plt.show()
df['BPSysAve'].isna().sum()
df.loc[df['BPSysAve'].isna()]['Age'].hist(bins=32)

plt.show()
df['BPDiaAve'].isna().sum()
df.loc[df['BPDiaAve'].isna()]['Age'].hist(bins=32)

plt.show()
df['Testosterone'].isna().sum()
df['Testosterone'].hist()

plt.show()
df.loc[df['Testosterone'].isna()]['Age'].hist(bins=16)

plt.show()
df['DirectChol'].isna().sum()
df.loc[df['DirectChol'].isna()]['Age'].hist(bins=16)

plt.show()
df.loc[df['DirectChol'].notna(), 'DirectChol'].head()
df['TotChol'].isna().sum()
df.loc[df['TotChol'].isna()]['Age'].hist(bins=16)

plt.show()
df['UrineVol1'].isna().sum()
df['UrineVol1'].hist(bins=20)

plt.show()
df['UrineFlow1'].isna().sum()
df['UrineVol1'].hist(bins=20)

plt.show()
df['UrineVol2'].isna().sum()
df['UrineFlow2'].isna().sum()
df['Diabetes'].isna().sum()
df['Diabetes'].value_counts().plot.bar(rot=0)

plt.show()
df.loc[df['Age'] < 1]['Diabetes'].isna().sum()
df.loc[df['Diabetes'].isna()]['Age'].hist(bins=32)

plt.show()
len(df[(df['DiabetesAge'].isna()) & (df['Diabetes'].isna())])
df['DiabetesAge'].hist(bins=16)

plt.show()
df['HealthGen'].isna().sum()
df['HealthGen'].value_counts().plot.bar(rot=0)

plt.show()
df.loc[df['HealthGen'].isna()]['Age'].hist(bins=8)

plt.show()
df.loc[df['Age'] <= 11]['HealthGen'].notna().sum()
df.loc[df['Age'] <= 20]['HealthGen'].value_counts().plot.bar(rot=0)

plt.show()
df['DaysMentHlthBad'].isna().sum()
df['DaysMentHlthBad'].hist()

plt.show()
df.loc[df['DaysMentHlthBad'].isna()]['Age'].hist(bins=16)

plt.show()
df['LittleInterest'].isna().sum()
df['LittleInterest'].value_counts().plot.bar(rot=0)

plt.show()
df.loc[df['LittleInterest'].isna()]['Age'].hist(bins=16)

plt.show()
df['Depressed'].isna().sum()
df['Depressed'].value_counts().plot.bar(rot=0)

plt.show()
df.loc[df['Depressed'].isna()]['Age'].hist(bins=16)

plt.show()
len(df[(df['nPregnancies'].isna()) & (df['Gender']=='female')])
len(df[df['nPregnancies']==0])
len(df[(df['nPregnancies'].isna()) & (df['nBabies']>0)])
df['nPregnancies'].value_counts().plot.bar(rot=0)

plt.show()
len(df[(df['nPregnancies'].notna()) & (df['Gender']=='male')])
len(df[(df['nPregnancies'].notna()) & (df['nBabies']).isna()])
len(df[df['nBabies']==0])
df['nBabies'].value_counts().plot.bar(rot=0)

plt.show()
len(df[(df['nPregnancies'] != df['nBabies']) & (df['nBabies'].notna())])
len(df[(df['nPregnancies'] < df['nBabies']) & (df['nBabies'].notna())])
len(df[(df['nPregnancies'].notna()) & (df['nBabies'].isna())])
len(df[(df['nBabies']!=0) & (df['Age1stBaby']).isna()])
df['Age1stBaby'].hist(bins=12)

plt.show()
df.loc[(df['nBabies'].notna()) & (df['Age1stBaby'].isna())]['Age'].hist(bins=16)

plt.show()
df['SleepHrsNight'].isna().sum()
df['SleepHrsNight'].hist()

plt.show()
df.loc[df['SleepHrsNight'].isna()]['Age'].hist(bins=14)

plt.show()
age_count = list()

for i in range(df['Age'].max()):

    age_count += [(i, len(df.loc[(df['SleepHrsNight'].notna()) & (df['Age']==i)]))]

print(age_count)
df['SleepTrouble'].isna().sum()
df['SleepTrouble'].value_counts().plot.bar(rot=0)

plt.show()
df.loc[df['SleepTrouble'].isna()]['Age'].hist(bins=14)

plt.show()
df['PhysActive'].isna().sum()
df['PhysActive'].value_counts().plot.bar(rot=0)

plt.show()
df.loc[df['PhysActive'].isna()]['Age'].hist(bins=14)

plt.show()
age_count = list()

for i in range(df['Age'].max()):

    age_count += [(i, len(df.loc[(df['PhysActive'].notna()) & (df['Age']==i)]))]

print(age_count)
df['PhysActiveDays'].isna().sum()
df['PhysActiveDays'].hist(bins=6)

plt.show()
df['TVHrsDay'].isna().sum()
df['TVHrsDay'].value_counts().plot.bar(rot=45)

plt.show()
df['CompHrsDay'].isna().sum()
df['CompHrsDay'].value_counts().plot.bar(rot=45)

plt.show()
df['TVHrsDayChild'].isna().sum()
df['TVHrsDayChild'].value_counts().plot.bar(rot=0)

plt.show()
df['TVHrsDayChild'].isna().sum()
df['TVHrsDayChild'].hist(bins=6)

plt.show()
df['Alcohol12PlusYr'].isna().sum()
df['Alcohol12PlusYr'].value_counts().plot.bar(rot=0)

plt.show()
df.loc[df['Alcohol12PlusYr'].isna()]['Age'].hist(bins=14)

plt.show()
age_count = list()

for i in range(df['Age'].max()):

    age_count += [(i, len(df.loc[(df['Alcohol12PlusYr'].notna()) & (df['Age']==i)]))]

print(age_count)
df['AlcoholDay'].isna().sum()
df['AlcoholDay'].hist(bins=20)

plt.show()
df.loc[df['AlcoholDay'].notna()]['Age'].hist(bins=14)

plt.show()
df['AlcoholYear'].isna().sum()
df['AlcoholYear'].hist()

plt.show()
df['SmokeNow'].isna().sum()
df['SmokeNow'].value_counts().plot.bar(rot=0)

plt.show()
df.loc[df['SmokeNow'].isna()]['Age'].hist(bins=14)

plt.show()
df.loc[df['SmokeNow']=='Yes']['Age'].hist(bins=8)

plt.show()
df['Smoke100'].isna().sum()
df['Smoke100'].value_counts().plot.bar(rot=0)

plt.show()
len(df.loc[(df['Smoke100'].isna()) & (df['Age']>=20)])
df['Smoke100n'].isna().sum()
df['Smoke100n'].value_counts().plot.bar(rot=0)

plt.show()
df['SmokeAge'].isna().sum()
df['SmokeAge'].hist(bins=30)

plt.show()
len(df.loc[(df['SmokeNow']=='Yes') & (df['SmokeAge'].isna())])
df['Marijuana'].isna().sum()
df['Marijuana'].value_counts().plot.bar(rot=0)

plt.show()
df.loc[df['Marijuana'].notna()]['Age'].hist(bins=14)

plt.show()
age_count = list()

for i in range(df['Age'].max()):

    age_count += [(i, len(df.loc[(df['Marijuana'].notna()) & (df['Age']==i)]))]

print(age_count)
len(df.loc[(df['Marijuana'].isna()) & (df['SmokeNow']=='Yes')])
df['AgeFirstMarij'].isna().sum()
df['AgeFirstMarij'].hist(bins=20)

plt.show()
len(df.loc[(df['Marijuana']=='Yes') & (df['AgeFirstMarij'].isna())])
df['RegularMarij'].isna().sum()
df['RegularMarij'].value_counts().plot.bar(rot=0)

plt.show()
len(df.loc[(df_test['Marijuana']=='Yes') & (df['RegularMarij'].isna())])
df['AgeRegMarij'].isna().sum()
df['AgeRegMarij'].hist(bins=20)

plt.show()
len(df[(df['AgeRegMarij'].isna()) & (df['Marijuana'] == 'Yes')])
df['HardDrugs'].isna().sum()
df['HardDrugs'].value_counts().plot.bar(rot=0)

plt.show()
df.loc[df['HardDrugs'].isna()]['Age'].hist(bins=14)

plt.show()
age_count = list()

for i in range(df['Age'].max()):

    age_count += [(i, len(df.loc[(df['HardDrugs'].notna()) & (df['Age']==i)]))]

print(age_count)
df['SexEver'].isna().sum()
df['SexEver'].value_counts().plot.bar(rot=0)

plt.show()
df.loc[df['SexEver'].isna()]['Age'].hist(bins=14)

plt.show()
df['SexAge'].isna().sum()
df['SexAge'].hist(bins=20)

plt.show()
len(df.loc[(df['SexAge'].isna()) & (df['SexEver'].notna())])
df['SexNumPartnLife'].isna().sum()
df['SexNumPartnLife'].hist(bins=20)

plt.show()
len(df[(df['SexNumPartnLife'].isna()) & (df['SexEver'] == 'Yes')])
df.loc[df['SexNumPartnLife'].isna()]['Age'].hist(bins=14)

plt.show()
age_count = list()

for i in range(df['Age'].max()):

    age_count += [(i, len(df.loc[(df['SexNumPartnLife'].notna()) & (df['Age']==i)]))]

print(age_count)
df['SexNumPartYear'].isna().sum()
df['SexNumPartYear'].hist(bins=20)

plt.show()
len(df[(df['SexNumPartYear'].isna()) & (df['SexEver'] == 'Yes')])
df.loc[(df['SexNumPartYear'].isna()) & (df['SexEver'] == 'Yes')]['Age'].hist(bins=14)

plt.show()
df['SameSex'].isna().sum()
len(df.loc[(df['SameSex'].isna()) & (df['SexEver']=='Yes')])
df['SameSex'].value_counts().plot.bar(rot=0)

plt.show()
df.loc[df['SameSex'].isna()]['Age'].hist(bins=16)

plt.show()
len(df.loc[(df['SexOrientation'].isna()) & (df['Age'] >= 14)])
df['SexOrientation'].value_counts().plot.bar(rot=0)

plt.show()
df.loc[df['SexOrientation'].isna()]['Age'].hist(bins=16)

plt.show()
len(df.loc[(df['PregnantNow'].isna()) & (df['Gender'] == 'female')])
df['PregnantNow'].value_counts().plot.bar(rot=0)

plt.show()
df.loc[(df['PregnantNow'].isna()) & (df['Gender'] == 'female')]['Age'].hist(bins=16)

plt.show()
df['flu'].isna().sum()
df['flu'].value_counts().plot.bar(rot=0)

plt.show()
def fill_bin_num(dataframe, feature, bin_feature, bin_size, stat_measure, min_bin=None, max_bin=None, default_val='No'):

    if min_bin is None:

        min_bin = dataframe[bin_feature].min()

    if max_bin is None:

        max_bin = dataframe[bin_feature].max()

    new_dataframe = dataframe.copy()

    df_meancat = pd.DataFrame(columns=['interval', 'stat_measure'])

    for num_bin, subset in dataframe.groupby(pd.cut(dataframe[bin_feature], np.arange(min_bin, max_bin+bin_size, bin_size), include_lowest=True)):

        if stat_measure is 'mean':

            row = [num_bin, subset[feature].mean()]

        elif stat_measure is 'mode': 

            mode_ar = subset[feature].mode().values

            if len(mode_ar) > 0:

                row = [num_bin, mode_ar[0]]

            else:

                row = [num_bin, default_val]

        else:

            raise Exception('Unknown statistical measure: ' + stat_measure)

        df_meancat.loc[len(df_meancat)] = row

    for index, row_df in dataframe[dataframe[feature].isna()].iterrows():

        for _, row_meancat in df_meancat.iterrows():

            if row_df[bin_feature] in row_meancat['interval']:

                new_dataframe.at[index, feature] = row_meancat['stat_measure']

    return new_dataframe





def make_dummy_cols(dataframe, column, prefix, drop_dummy):

    dummy = pd.get_dummies(dataframe[column], prefix=prefix)

    dummy = dummy.drop(columns=prefix+'_'+drop_dummy)

    dataframe = pd.concat([dataframe, dummy], axis=1)

    dataframe = dataframe.drop(columns=column)

    return dataframe





def cleaning(dataframe_raw):

    dataframe = dataframe_raw.copy()



    dataframe = dataframe.set_index('ID')



    dataframe.loc[(dataframe['Age']<=13) & (dataframe['Education'].isna()), 'Education'] = 'Lower School/Kindergarten'

    dataframe.loc[(dataframe['Age']==14) & (dataframe['Education'].isna()), 'Education'] = '8th Grade'

    dataframe.loc[(dataframe['Age']<=17) & (dataframe['Education'].isna()), 'Education'] = '9 - 11th Grade'

    dataframe.loc[(dataframe['Age']<=21) & (dataframe['Education'].isna()), 'Education'] = 'High School'

    dataframe['Education'] = dataframe['Education'].fillna('Some College')



    dataframe.loc[(dataframe['Age']<=20) & (dataframe['MaritalStatus'].isna()), 'MaritalStatus'] = 'NeverMarried'

    dataframe.at[dataframe['MaritalStatus'].isna(), 'MaritalStatus'] = fill_bin_num(dataframe, 'MaritalStatus', 'Age', 5, 'mode',20)



    dataframe = dataframe.drop(columns=['HHIncome'])



    dataframe.loc[dataframe['HHIncomeMid'].isna(), 'HHIncomeMid'] = dataframe['HHIncomeMid'].median()



    dataframe.loc[dataframe['Poverty'].isna(), 'Poverty'] = dataframe['Poverty'].median()



    dataframe.loc[dataframe['HomeRooms'].isna(), 'HomeRooms'] = dataframe['HomeRooms'].mean()



    dataframe.loc[dataframe['HomeOwn'].isna(), 'HomeOwn'] = dataframe['HomeOwn'].mode().values[0]



    dataframe.loc[(dataframe['Work'].isna()) & (dataframe['Education'].isna()) & (dataframe['Age']<=20), 'Work'] = 'NotWorking'



    dataframe.loc[dataframe['Work'].isna(), 'Work'] = dataframe['Work'].mode().values[0]



    dataframe = fill_bin_num(dataframe, 'Weight', 'Age', 2, 'mean')



    dataframe = dataframe.drop(columns=['HeadCirc'])



    for index, row in dataframe.iterrows():

        if np.isnan(row['Height']) and not np.isnan(row['Length']):

            dataframe.at[index, 'Height'] = row['Length']

    dataframe = fill_bin_num(dataframe, 'Height', 'Age', 2, 'mean')



    dataframe = dataframe.drop(columns=['Length'])



    for index, row in dataframe[dataframe['BMI'].isna()].iterrows():

        dataframe.at[index, 'BMI'] = row['Weight'] / ((row['Height']/100)**2)



    dataframe = dataframe.drop(columns='BMICatUnder20yrs')



    dataframe = dataframe.drop(columns='BMI_WHO')



    dataframe = fill_bin_num(dataframe, 'Pulse', 'Age', 10, 'mean')



    dataframe.loc[(dataframe['Age']<10) & (dataframe['BPSysAve'].isna()), 'BPSysAve'] = 105

    dataframe = fill_bin_num(dataframe, 'BPSysAve', 'Age', 5, 'mean', 10)



    dataframe.loc[(dataframe['Age']<10) & (dataframe['BPDiaAve'].isna()), 'BPDiaAve'] = 60

    dataframe = fill_bin_num(dataframe, 'BPDiaAve', 'Age', 5, 'mean', 10)



    dataframe = dataframe.drop(columns='BPSys1')



    dataframe = dataframe.drop(columns='BPDia1')



    dataframe = dataframe.drop(columns='BPSys2')



    dataframe = dataframe.drop(columns='BPDia2')



    dataframe = dataframe.drop(columns='BPSys3')



    dataframe = dataframe.drop(columns='BPDia3')



    dataframe = dataframe.drop(columns=['Testosterone'])



    dataframe.loc[(dataframe['Age']<10) & (dataframe['DirectChol'].isna()), 'DirectChol'] = 0 

    dataframe = fill_bin_num(dataframe, 'DirectChol', 'Age', 5, 'mean', 10)



    dataframe.loc[(dataframe['Age']<10) & (dataframe['TotChol'].isna()), 'TotChol'] = 0

    dataframe = fill_bin_num(dataframe, 'TotChol', 'Age', 5, 'mean', 10)

    

    dataframe.loc[dataframe['UrineVol1'].isna(), 'UrineVol1'] = dataframe['UrineVol1'].median()



    dataframe.loc[dataframe['UrineFlow1'].isna(), 'UrineFlow1'] = dataframe['UrineFlow1'].median()



    dataframe = dataframe.drop(columns=['UrineVol2'])



    dataframe = dataframe.drop(columns=['UrineFlow2'])



    dataframe['Diabetes'] = dataframe['Diabetes'].fillna('No')



    dataframe['DiabetesAge'] = dataframe['DiabetesAge'].fillna(0)



    dataframe.loc[(dataframe['Age']<=12) & (dataframe['HealthGen'].isna()), 'HealthGen'] = 'Good'

    dataframe = fill_bin_num(dataframe, 'HealthGen', 'Age', 5, 'mode', 10)



    dataframe.loc[(dataframe['Age']<=12) & (dataframe['DaysMentHlthBad'].isna()), 'DaysMentHlthBad'] = 0

    dataframe = fill_bin_num(dataframe, 'DaysMentHlthBad', 'Age', 5, 'mean', 10)



    dataframe.loc[(dataframe['Age']<=15) & (dataframe['LittleInterest'].isna()), 'LittleInterest'] = 'None'

    dataframe = fill_bin_num(dataframe, 'LittleInterest', 'Age', 5, 'mode', 15)



    dataframe.loc[(dataframe['Age']<=12) & (dataframe['DaysMentHlthBad'].isna()), 'DaysMentHlthBad'] = 0

    dataframe = fill_bin_num(dataframe, 'DaysMentHlthBad', 'Age', 5, 'mean', 10)



    for index, row in dataframe.iterrows():

        if np.isnan(row['nBabies']) and not np.isnan(row['nPregnancies']):

            dataframe.at[index, 'nBabies'] = row['nPregnancies']

    dataframe['nBabies'] = dataframe['nBabies'].fillna(0)



    dataframe['nPregnancies'] = dataframe['nPregnancies'].fillna(0)



    dataframe['Age1stBaby'] = dataframe['Age1stBaby'].fillna(0)



    dataframe.loc[(dataframe['Age']==0) & (dataframe['SleepHrsNight'].isna()), 'SleepHrsNight'] = 14

    dataframe.loc[(dataframe['Age']<=2) & (dataframe['SleepHrsNight'].isna()), 'SleepHrsNight'] = 12

    dataframe.loc[(dataframe['Age']<=5) & (dataframe['SleepHrsNight'].isna()), 'SleepHrsNight'] = 10

    dataframe.loc[(dataframe['Age']<=10) & (dataframe['SleepHrsNight'].isna()), 'SleepHrsNight'] = 9

    dataframe.loc[(dataframe['Age']<=15) & (dataframe['SleepHrsNight'].isna()), 'SleepHrsNight'] = 8

    dataframe['SleepHrsNight'] = dataframe['SleepHrsNight'].fillna(dataframe_raw['SleepHrsNight'].mean())



    dataframe['SleepTrouble'] = dataframe['SleepTrouble'].fillna('No')



    dataframe.loc[(dataframe['Age']<=4) & (dataframe['PhysActive'].isna()), 'PhysActive'] = 'No'

    dataframe = fill_bin_num(dataframe, 'PhysActive', 'Age', 2, 'mode', 16)

    dataframe['PhysActive'] = dataframe['PhysActive'].fillna('Yes') # Big assumption here. All kids between 4 and 16 are physically active



    dataframe = dataframe.drop(columns=['PhysActiveDays'])



    dataframe = dataframe.drop(columns=['TVHrsDay'])



    dataframe = dataframe.drop(columns=['TVHrsDayChild'])



    dataframe = dataframe.drop(columns=['CompHrsDay'])



    dataframe = dataframe.drop(columns=['CompHrsDayChild'])



    dataframe.loc[(dataframe['Age']<18) & (dataframe['Alcohol12PlusYr'].isna()), 'Alcohol12PlusYr'] = 'No'

    dataframe = fill_bin_num(dataframe, 'Alcohol12PlusYr', 'Age', 5, 'mode', 18)



    dataframe.loc[(dataframe['Age']<18) & (dataframe['AlcoholDay'].isna()), 'AlcoholDay'] = 0

    dataframe = fill_bin_num(dataframe, 'AlcoholDay', 'Age', 5, 'mean', 18)



    dataframe.loc[(dataframe['Age']<18) & (dataframe['AlcoholYear'].isna()), 'AlcoholYear'] = 0

    dataframe = fill_bin_num(dataframe, 'AlcoholYear', 'Age', 5, 'mean', 18)



    dataframe.loc[(dataframe['Age']<20) & (dataframe['SmokeNow'].isna()), 'SmokeNow'] = 'No'

    dataframe = fill_bin_num(dataframe, 'SmokeNow', 'Age', 5, 'mode', 20)



    dataframe['Smoke100'] = dataframe['Smoke100'].fillna('No')



    dataframe['Smoke100n'] = dataframe['Smoke100n'].fillna('No')



    dataframe.loc[(dataframe['SmokeNow']=='No') & (dataframe['SmokeAge'].isna()), 'SmokeAge'] = 0

    dataframe = fill_bin_num(dataframe, 'SmokeAge', 'Age', 5, 'mean', 20)



    dataframe.loc[(dataframe['Age']<18) & (dataframe['Marijuana'].isna()), 'Marijuana'] = 'No'

    dataframe.loc[(dataframe['Marijuana'].isna()) & (dataframe['SmokeNow']=='No'), 'Marijuana'] = 'No'

    dataframe = fill_bin_num(dataframe, 'Marijuana', 'Age', 5, 'mode', 20)



    dataframe.loc[(dataframe['Marijuana']=='No') & (dataframe['AgeFirstMarij'].isna()), 'AgeFirstMarij'] = 0

    dataframe = fill_bin_num(dataframe, 'AgeFirstMarij', 'Age', 5, 'mean', 20)



    dataframe.loc[(dataframe['Marijuana']=='No') & (dataframe['RegularMarij'].isna()), 'RegularMarij'] = 'No'

    dataframe = fill_bin_num(dataframe, 'RegularMarij', 'Age', 5, 'mode', 20)



    dataframe.loc[(dataframe['RegularMarij']=='No') & (dataframe['AgeRegMarij'].isna()), 'AgeRegMarij'] = 0

    dataframe = fill_bin_num(dataframe, 'AgeRegMarij', 'Age', 5, 'mean', 20)



    dataframe.loc[(dataframe['Age']<18) & (dataframe['HardDrugs'].isna()), 'HardDrugs'] = 'No'

    dataframe = fill_bin_num(dataframe, 'HardDrugs', 'Age', 5, 'mode', 18)



    mode_sex_age = dataframe['SexAge'].mode()[0]

    dataframe.loc[(dataframe['Age']<=mode_sex_age) & (dataframe['SexEver'].isna()), 'SexEver'] = 'No'

    dataframe['SexEver'] = dataframe['SexEver'].fillna('Yes')



    dataframe.loc[(dataframe['SexEver']=='No') & (dataframe['SexAge'].isna()), 'SexAge'] = 0

    dataframe['SexAge'] = dataframe['SexAge'].fillna(mode_sex_age)



    dataframe.loc[(dataframe['SexEver']=='No') & (dataframe['SexNumPartnLife'].isna()), 'SexNumPartnLife'] = 0

    dataframe = fill_bin_num(dataframe, 'SexNumPartnLife', 'Age', 5, 'mean')

    dataframe['SexNumPartnLife'] = dataframe_raw.loc[(dataframe_raw['Age'] >= 60) & (dataframe_raw['Age'] <= 70), 'SexNumPartnLife'].mode()[0] # Missing values for the elderly. Assumed that lifetime sex partners do not increase after 60.



    dataframe.loc[(dataframe['SexEver']=='No') & (dataframe['SexNumPartYear'].isna()), 'SexNumPartYear'] = 0

    dataframe = fill_bin_num(dataframe, 'SexNumPartYear', 'Age', 10, 'mean')

    dataframe['SexNumPartYear'] = dataframe['SexNumPartYear'].fillna(0)



    dataframe['SameSex'] = dataframe['SameSex'].fillna('No')



    dataframe = dataframe.drop(columns=['SexOrientation'])



    dataframe['PregnantNow'] = dataframe['PregnantNow'].fillna('No')





    # Making dummy variables

    dataframe['male'] = 1*(dataframe['Gender'] ==  'male')

    dataframe = dataframe.drop(columns=['Gender'])



    dataframe['white'] = np.where(dataframe['Race1'] == 'white',1,0)

    dataframe = dataframe.drop(columns=['Race1'])



    dataframe = make_dummy_cols(dataframe, 'Education', 'education', '8th Grade')



    dataframe = make_dummy_cols(dataframe, 'MaritalStatus', 'maritalstatus', 'Separated')



    dataframe = make_dummy_cols(dataframe, 'HomeOwn', 'homeown', 'Other')



    dataframe = make_dummy_cols(dataframe, 'Work', 'work', 'Looking')



    dataframe['Diabetes'] = np.where(dataframe['Diabetes'] == 'Yes',1,0)



    dataframe = make_dummy_cols(dataframe, 'HealthGen', 'healthgen', 'Poor')



    dataframe = make_dummy_cols(dataframe, 'LittleInterest', 'littleinterest', 'None')



    dataframe = make_dummy_cols(dataframe, 'Depressed', 'depressed', 'None')



    dataframe['SleepTrouble'] = np.where(dataframe['SleepTrouble'] == 'Yes',1,0)



    dataframe['PhysActive'] = np.where(dataframe['PhysActive'] == 'Yes',1,0)



    dataframe['Alcohol12PlusYr'] = np.where(dataframe['Alcohol12PlusYr'] == 'Yes',1,0)



    dataframe['SmokeNow'] = np.where(dataframe['SmokeNow'] == 'Yes',1,0)

    

    dataframe['Smoke100'] = np.where(dataframe['Smoke100'] == 'Yes',1,0)



    dataframe['Smoke100n'] = np.where(dataframe['Smoke100n'] == 'Yes',1,0)



    dataframe['Marijuana'] = np.where(dataframe['Marijuana'] == 'Yes',1,0)



    dataframe['RegularMarij'] = np.where(dataframe['RegularMarij'] == 'Yes',1,0)



    dataframe['HardDrugs'] = np.where(dataframe['HardDrugs'] == 'Yes',1,0)



    dataframe['SexEver'] = np.where(dataframe['SexEver'] == 'Yes',1,0)



    dataframe['SameSex'] = np.where(dataframe['SameSex'] == 'Yes',1,0)



    dataframe['PregnantNow'] = np.where(dataframe['PregnantNow'] == 'Yes',1,0)



    return dataframe
from sklearn import preprocessing

data = cleaning(df).select_dtypes(include = 'number')

norm = preprocessing.MinMaxScaler()

data_n = norm.fit_transform(data.drop('flu', axis=1))

ndata = pd.DataFrame(norm.fit_transform(data.drop('flu', axis=1)), index=data.index)

ndata['flu'] = data['flu']

num_test = cleaning(df_test).select_dtypes(include='number')

ntest = pd.DataFrame(norm.fit_transform(num_test), index=num_test.index)

train, test = train_test_split(ndata, stratify=ndata['flu'], test_size=0.1)



X_train = train.drop('flu', axis=1)

X_test = test.drop('flu', axis=1)

y_train = train['flu']

y_test = test['flu']
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import RandomizedSearchCV





cw = []

for i in np.linspace(start = 0.006, stop = 0.08, num = 5):

    cw.append({0:i, 1:1-i})

cw.append('balanced')

C = [x for x in np.linspace(start = 0.2, stop = 1.5, num = 5)]

C.append(1)



param_grid = {

    'C':C,

    'kernel':['linear', 'rbf', 'poly', 'sigmoid'],

    'degree':[2,3,4,5,6,7,8],

    'gamma':['auto'],

    'shrinking':[True, False],

    'class_weight': cw

}
sv = SVC()

sv_r = RandomizedSearchCV(sv, param_grid, scoring=scorel, cv=3, return_train_score=True, verbose=2, random_state=42, n_jobs=-2, n_iter=300)

sv_r.fit(X_train, y_train)
params = sv_r.best_params_

print('The best parameters are {} giving an average Balanced Accuract of {:.4f}'.format(params, sv_r.best_score_))
a = np.array(sv_r.best_estimator_.predict(ntest))

result = pd.DataFrame(np.array([num_test.index, a], dtype=np.int32).T, columns=['ID', 'Prediction'])

result.to_csv('result_svm.csv', index=False)
import telegram 

import json

import os

def notify_me(message='Done'):

    filename = os.environ['HOME']+'/.telegram'

    with open(filename) as f:

        json_blob = f.read()

        credentials = json.loads(json_blob)

    bot = telegram.Bot(token=credentials['api_key'])

    bot.send_message(chat_id=credentials['chat_id'], text=message)
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression



cw = []

for i in np.linspace(start = 0.001, stop = 0.4, num = 20):

    cw.append({0:i, 1:1-i})

cw.append('balanced')
w0 = 0.0599

param_grid = {

    'C':[x for x in np.linspace(start = 0.001, stop = 20, num = 40)],

    'penalty':['l1', 'l2', 'elasticnet'],

    'max_iter':[10, 100, 1000, 10000],

    'class_weight': cw

}
lr = LogisticRegression()

lr_r = GridSearchCV(lr, param_grid, scoring='balanced_accuracy', cv=3, return_train_score=True, verbose=0, n_jobs=-1)

lr_r.fit(ndata.drop('flu', axis=1), ndata['flu'])
params = lr_r.best_params_



notify_me('The best parameters are {} giving an average ROC AUC score of {:.4f}'.format(params, lr_r.best_score_))
pd.DataFrame(lr_r.cv_results_).sort_values(by='rank_test_score')
w = 0.064

fin = LogisticRegression(class_weight={0:w,1:1-w}, C=2.02, penalty='l2')

fin.fit(X_train, y_train)

np.mean(cross_val_score(fin, X_train, y_train, scoring='balanced_accuracy', cv=3))
lo = lr_r.best_estimator_
a = np.array(lr_r.best_estimator_.predict(ntest))

result = pd.DataFrame(np.array([num_test.index, a], dtype=np.int32).T, columns=['ID', 'Prediction'])

result.to_csv('result_lr_n.csv', index=False)
def scorel(model, X_test, y_test):

    return 0.6*np.mean(cross_val_score(model,X_train,y_train,scoring='balanced_accuracy', cv=5))+0.4*score(model,X_test, y_test)[2]
import xgboost as xgb



xg_c = xgb.XGBClassifier(max_depth=3)

param_grid = {

    'objective':['reg:squarederror', 'reg:logistic', 'binary:logistic'],

    'scale_pos_weight':[20,21,22],

    'colsample_bytree':[0.3],

    'eval_metric':['aucpr', 'auc', 'mae', 'map'],

    'alpha':[5, 10, 20],

    'n_estimators': [5, 10, 25, 40, 50, 100, 125],

    'learning_rate': [0.05, 0.1, 0.15]

}

xg_s = GridSearchCV(xg_c, param_grid, scoring='balanced_accuracy', cv=3, return_train_score=True)
xg_s.fit(ndata.drop('flu', axis=1), ndata['flu'])
params = xg_s.best_params_

print('The best parameters are {} giving an average ROC AUC score of {:.4f}'.format(params, xg_s.best_score_))

xg = xg_s.best_estimator_
a = np.array(xg.predict(ntest.values))

result = pd.DataFrame(np.array([num_test.index, a], dtype=np.int32).T, columns=['ID', 'Prediction'])

result.to_csv('result_xg.csv', index=False)
from sklearn.ensemble import RandomForestClassifier



w0 = 0.0599



cw = []

for i in np.linspace(start = 0.001, stop = 0.15, num = 10):

    cw.append({0:i, 1:1-i})

cw.append('balanced')



param_grid = {

    'n_estimators' : [20,50,70,110, 130, 150, 200],

    'max_features' : ['auto', 'sqrt'], 

    'max_depth':[3, 5, 7, 10, 15, None],

    'criterion' : ['gini', 'entropy'],

    'min_samples_split' : [2, 3, 5, 7],

    'min_samples_leaf' : [2, 3, 5, 7],

    'class_weight': cw

}
rfs = RandomForestClassifier()

rfs_random = RandomizedSearchCV(rfs, param_grid, scoring='balanced_accuracy', cv=3, return_train_score=True, random_state=42, n_jobs=-1, n_iter=1000)

rfs_random.fit(X_train, y_train)
params = rfs_random.best_params_

notify_me('The best parameters are {} giving an average ROC AUC score of {:.4f}'.format(params, rfs_random.best_score_))
rf = RandomForestClassifier(**rfs_random.best_params_)

rf.fit(ndata.drop('flu', axis=1), ndata['flu'])
a = np.array(rf.predict(ntest))

result = pd.DataFrame(np.array([num_test.index, a], dtype=np.int32).T, columns=['ID', 'Prediction'])

result.to_csv('result_rf.csv', index=False)
param_grid = {

    'max_features' : ['auto', 'sqrt'], 

    'max_depth':[3, 4, 5,6, 7, 10, None],

    'criterion' : ['gini', 'entropy'],

    'min_samples_split' : [2, 3,4, 5, 7],

    'min_samples_leaf' : [2, 3,4, 5, 7],

    'class_weight': cw

}

clf = tree.DecisionTreeClassifier()
clf_r = RandomizedSearchCV(clf, param_grid, scoring='balanced_accuracy', cv=3, return_train_score=True, verbose=0, n_iter=2000)

clf_r.fit(X_train, y_train)
params = clf_r.best_params_

print('The best parameters are {} giving an average ROC AUC score of {:.4f}'.format(params, clf_r.best_score_))
a = np.array(clf_r.best_estimator_.predict(ntest))

result = pd.DataFrame(np.array([num_test.index, a], dtype=np.int32).T, columns=['ID', 'Prediction'])

result.to_csv('result_dt.csv', index=False)
pd.DataFrame({'LR':0.69452, 'SVM':0.69214, 'XGBoost':0.67610}, index=[0])
###AUROC locally



#score = roc_auc_score(real_labels, predicted_labels)



#real_labels: the ground truth (0 or 1)

#predicted_labels: labels predicted by your algorithm (0 or 1)
def extended_score(model, x_test, y_test):

    overall = 0

    class_0 = 0

    class_1 = 0

    for i in range(100):

        sample = np.random.choice(len(x_test), len(x_test))

        x_sub_test = x_test[sample]

        y_sub_test = y_test[sample]

        

        overall += model.score(x_sub_test, y_sub_test)

        class_0 += model.score(x_sub_test[y_sub_test==0], y_sub_test[y_sub_test==0])

        class_1 += model.score(x_sub_test[y_sub_test==1], y_sub_test[y_sub_test==1])



    return pd.Series([overall / 100., 

                      class_0 / 100.,

                      class_1 / 100.],

                      index=['overall accuracy', 'accuracy on class 0', 'accuracy on class 1'])
#same job as before, but faster?



score = lambda model, x_val, y_val: pd.Series([model.score(x_val, y_val), 

                                                 model.score(x_val[y_val==0], y_val[y_val==0]),

                                                 model.score(x_val[y_val==1], y_val[y_val==1])], 

                                                index=['overall accuracy', 'accuracy on class 0', 'accuracy on class 1'])
def scorel(model, X_test, y_test):

    return 0.8*np.mean(cross_val_score(model,X_train,y_train,scoring='balanced_accuracy', cv=2))+0.2*score(model,X_test, y_test)[2]