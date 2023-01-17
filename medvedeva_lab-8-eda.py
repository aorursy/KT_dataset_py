# Импорт нужных библиотек

import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(); # более красивый внешний вид графиков по умолчанию
df = pd.read_csv('../input/cardio_train.csv', sep=';')
df.head(10)
if df.groupby('gender')['height'].mean()[1] > df.groupby('gender')['height'].mean()[2]:

    print('Количество мужчин:', df['gender'].value_counts()[1], '(закодировано цифрой 1).', 

          'Количество женщин:', df['gender'].value_counts()[2], '(закодировано цифрой 2).')

else:

    print('Количество мужчин:', df['gender'].value_counts()[2], '(закодировано цифрой 2).',

          'Количество женщин:', df['gender'].value_counts()[1], '(закодировано цифрой 1).')

    
m = df['gender'].value_counts()[2]

m
f = df['gender'].value_counts()[1]

f
f_m_alco = pd.crosstab(df['gender'], df['alco'])

f_m_alco
f_alco = f_m_alco.loc[1][1]/df['gender'].value_counts().loc[1]

m_alco = f_m_alco.loc[2][1]/df['gender'].value_counts().loc[2]

if f_alco < m_alco:

    print('Верно, что мужчины более склонны к употреблению алкоголя, чем женщины')

else:

    print('Не верно, что мужчины более склонны к употреблению алкоголя, чем женщины')
f_alco = df[df['gender'] == 1]['alco'].value_counts(normalize=True)[1]

f_alco
m_alco = df[df['gender'] == 2]['alco'].value_counts(normalize=True).loc[1]

m_alco
if f_alco < m_alco:

    print('Верно, что мужчины более склонны к употреблению алкоголя, чем женщины')

else:

    print('Не верно, что мужчины более склонны к употреблению алкоголя, чем женщины')
df[df['gender'] == 2]['smoke'].value_counts(normalize=True)
f_smoke = df[df['gender'] == 1]['smoke'].value_counts(normalize=True).loc[1]

f_smoke
m_smoke = df[df['gender'] == 2]['smoke'].value_counts(normalize=True).loc[1]

m_smoke
print('Pазличие между процентами курящих мужчин и женщин:', str(round(abs(m_smoke - f_smoke)*100, 1))+'%')
df[df['smoke'] == 1]['age'].mean()/365
df[df['smoke'] == 0]['age'].mean()/365
df['bmi'] = round(df['weight'] / ((df['height']/100)**2), 1)

df.head()
df['bmi'].mean()
f_bmi = df[df['gender'] == 1]['bmi'].mean()

f_bmi
m_bmi = df[df['gender'] == 2]['bmi'].mean()

m_bmi
cardio_0_bmi = df[df['cardio'] == 0]['bmi'].mean()

cardio_0_bmi
cardio_1_bmi = df[df['cardio'] == 1]['bmi'].mean()

cardio_1_bmi
f00_bmi = df[(df['gender'] == 1) & (df['cardio'] == 0) & (df['alco'] == 0)]['bmi'].mean()

f00_bmi
m00_bmi = df[(df['gender'] == 2) & (df['cardio'] == 0) & (df['alco'] == 0)]['bmi'].mean()

m00_bmi
ap_df = df.drop(df[df['ap_lo'] < df['ap_hi']].index)

print(str(round((ap_df.shape[0] / df.shape[0])*100, 1))+'%')
cardio_1_65_160_3 = df[(df['age']/365 > 60) & (df['gender'] == 2) & (df['smoke'] == 1) & (df['ap_hi'] > 160) & (df['ap_hi'] < 180) & (df['cholesterol'] == 3) & (df['cardio'] == 1)].shape[0]

cardio_1_65_160_3
cardio_1_65_160_3 = df[(df['age']/365 > 60) & (df['gender'] == 2) & (df['smoke'] == 1) & (df['ap_hi'] > 160) & (df['ap_hi'] < 180) & (df['cholesterol'] == 3)].shape[0]

cardio_1_65_160_3
risk_cardio_65_160_3 = (cardio_1_65_160_3 / cardio_65_160_3)*100

risk_cardio_65_160_3
cardio_1_65_120_1 = df[(df['age']/365 > 60) & (df['gender'] == 2) & (df['smoke'] == 1) & (df['ap_hi'] < 120) & (df['cholesterol'] == 1) & (df['cardio'] == 1)].shape[0]

cardio_1_65_120_1
cardio_65_120_1 = df[(df['age']/365 > 60) & (df['gender'] == 2) & (df['smoke'] == 1) & (df['ap_hi'] < 120) & (df['cholesterol'] == 1)].shape[0]

cardio_65_120_1
risk_cardio_65_120_1 = (cardio_1_65_120_1/cardio_65_120_1)*100

risk_cardio_65_120_1
risk_cardio_65_160_3/risk_cardio_65_120_1
df['age'] = (df['age'] / 365).round()
plt.figure(figsize=(15, 8))

sns.countplot(y='age', hue='cholesterol', data=df);
sns.boxplot(df['bmi']);
df.groupby('cardio')['bmi'].mean().plot(kind='bar') 

plt.ylabel('BMI')

plt.show();