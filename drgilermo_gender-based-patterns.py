import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

df = pd.read_csv('../input/No-show-Issue-Comma-300k.csv')
df = df[df.Status == 'Show-Up']

range_df = pd.DataFrame()

range_df['Age'] = range(100)

men = range_df.Age.apply(lambda x: len(df[(df.Age == x) & (df.Gender == 'M')]))

women = range_df.Age.apply(lambda x: len(df[(df.Age == x) & (df.Gender == 'F')]))

plt.plot(range(100),men, 'b')

plt.plot(range(100),women, color = 'r')

plt.legend(['M','F'])

plt.xlabel('Age')

plt.title('Women visit the doctor more often')
men_smoke = range_df[range_df.columns[0]].apply(lambda x: len(df[(df.Age == x) & (df.Gender == 'M') & (df.Smokes == 1)]))

women_smoke = range_df[range_df.columns[0]].apply(lambda x: len(df[(df.Age == x) & (df.Gender == 'F') & (df.Smokes == 1)]))



men_tension = range_df[range_df.columns[0]].apply(lambda x: len(df[(df.Age == x) & (df.Gender == 'M') & (df.HiperTension == 1)]))

women_tension = range_df[range_df.columns[0]].apply(lambda x: len(df[(df.Age == x) & (df.Gender == 'F') & (df.HiperTension == 1)]))



men_Diabetes = range_df[range_df.columns[0]].apply(lambda x: len(df[(df.Age == x) & (df.Gender == 'M') & (df.Diabetes == 1)]))

women_Diabetes = range_df[range_df.columns[0]].apply(lambda x: len(df[(df.Age == x) & (df.Gender == 'F') & (df.Diabetes == 1)]))



men_Tuber = range_df[range_df.columns[0]].apply(lambda x: len(df[(df.Age == x) & (df.Gender == 'M') & (df.Tuberculosis == 1)]))

women_Tuber = range_df[range_df.columns[0]].apply(lambda x: len(df[(df.Age == x) & (df.Gender == 'F') & (df.Tuberculosis == 1)]))



plt.figure(figsize = (10,10))

plt.subplot(2,2,1)

plt.plot(range(100),men_smoke/men)

plt.plot(range(100),women_smoke/women, color = 'r')

plt.title('Smoking')

plt.legend(['M','F'], loc = 2)



plt.subplot(2,2,2)

plt.plot(range(100),men_tension/men)

plt.plot(range(100),women_tension/women, color = 'r')

plt.title('Hiper Tension')

plt.legend(['M','F'], loc = 2)



plt.subplot(2,2,3)

plt.plot(range(100),men_Diabetes/men)

plt.plot(range(100),women_Diabetes/women, color = 'r')

plt.title('Diabetes')

plt.legend(['M','F'], loc = 2)

plt.xlabel('Age')



plt.subplot(2,2,4)

plt.plot(range(100),men_Tuber/men)

plt.plot(range(100),women_Tuber/women, color = 'r')

plt.legend(['M','F'], loc = 2)

plt.xlabel('Age')



plt.title('Tuberculosis')
plt.hold(True)

sns.kdeplot(df.AwaitingTime[df.Gender == 'M'],marker = '*', color = 'b')

sns.kdeplot(df.AwaitingTime[df.Gender == 'F'],marker ='o',color = 'r')

plt.xlim([-50,0])

plt.title('Awaiting Time')

plt.xlabel('Awaiting days before appointment')

plt.legend(['Men','Women'])

plt.show()
Days = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']

Days_df =  pd.DataFrame(Days)



men_days = Days_df[Days_df.columns[0]].apply(lambda x: len(df[(df.DayOfTheWeek == x) & (df.Gender == 'M')]))

women_days = Days_df[Days_df.columns[0]].apply(lambda x: len(df[(df.DayOfTheWeek == x) & (df.Gender == 'F')]))



plt.bar(range(7), men_days/len(df[df.Gender == 'M']), width = 0.5)

plt.bar(range(7)+0.5*np.ones(len(range(7))), women_days/len(df[df.Gender == 'F']), width = 0.5, color = 'r')

plt.xticks(range(7) + 0.25*np.ones(len(range(7))),Days)



plt.title('Visit Day')

plt.xlabel('Day')

plt.legend(['Men','Women'])
df['Time'] = df['AppointmentRegistration'].apply(lambda x: int(x[11:13]))

sns.distplot(df.Time[df.Gender == 'M'], bins = range(24),norm_hist = True, kde = False)#, barwidth = 0.5)#, kde = False)

sns.distplot(df.Time[df.Gender == 'F'], bins = range(24),norm_hist = True, kde = False, color = 'r')

plt.xlim([6,22])

plt.title('Registration Hour')

plt.xlabel('Hour')

plt.legend(['Men','Women'])