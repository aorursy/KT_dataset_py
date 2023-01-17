# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import scipy as sp

import matplotlib.pyplot as plt

import seaborn as sns

from numpy import average

from numpy import median

import sqlite3



frame = pd.read_csv("../input/celebrity_deaths_3.csv", header=0, sep=',')



df = frame.drop_duplicates()
sui = df[(df['cause_of_death'].fillna('unknown').str.contains('suicide'))]



sns.countplot(x='death_year', data = sui)



plt.xlabel('Year of death')

plt.ylabel('Amount')

plt.title('Celebrity deaths by suicide per year')

plt.show()
sui = df[(df['cause_of_death'].fillna('unknown').str.contains('suicide'))]



sns.countplot(x='death_month', data = sui, order=['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 

                                                  'September', 'October', 'November', 'December'])



plt.xlabel('Month of death')

plt.ylabel('Amount')

plt.xticks(rotation='vertical')

plt.title('Celebrity deaths by suicide per month')

plt.show()
lite3conn = sqlite3.connect(':memory:')

sui.to_sql('sui',lite3conn)
query = '''

select *,

     case

         when death_month in ('December', 'January', 'February') then

             'Winter'

         when death_month in ('March', 'April', 'May') then

             'Spring'

         when death_month in ('June', 'July', 'August') then

             'Summer'

         when death_month in ('September', 'October', 'November') then

             'Autumn'

     end as season

from sui

'''



season_suicide = pd.read_sql(query,lite3conn)

#season_suicide



sns.countplot(x='season', data = season_suicide, order = ['Spring', 'Summer', 'Autumn', 'Winter'])



#plt.xlabel('Season of death')

plt.xlabel('')

plt.ylabel('Amount')

plt.title('Celebrity deaths by suicide per season in overall')
sns.countplot(x='death_year', hue='season', data=season_suicide)



plt.xlabel('')

plt.ylabel('Amount')

plt.title('Suicide per season by year')
query = '''

select kind_of_suicide, count(*) as cnt

from

    (select 

        case

            when cause_of_death like '%gunshot%' then 'Gunshot'

            when cause_of_death like '%hanging%' then 'Hanging'

            when cause_of_death like '%jumping%' then 'Jumping'

            when cause_of_death like '%overdose%' then 'Drug/Alcohol Overdose'

            when cause_of_death like '%train%' then 'By train'

            when cause_of_death like '%bombing%' then 'Bombing'

            when cause_of_death like '%defenestration%' then 'Defenestration'

            when cause_of_death like '%drowning%' then 'Drowning'

            when cause_of_death like '%assisted%' then 'Assisted'

            when cause_of_death like '%possible%' then 'Possible'

            when cause_of_death like '%asphyxiation%' then 'Asphyxiation'

            when cause_of_death like '%helium inhalation%' then 'Helium inhalation'

            when cause_of_death like '%poisoning%' then 'Poisoning'

            when cause_of_death like '%wrist cutting%' then 'Wrist cutting'

            else 'Without kind'

        end kind_of_suicide 

    from sui) as sui

group by kind_of_suicide

order by cnt desc

'''



suicide_type = pd.read_sql(query,lite3conn)



sns.barplot(y="kind_of_suicide", x="cnt", data=suicide_type, palette = "Blues_d")

plt.ylabel('')

plt.xticks(range(0,61,5))

plt.xlabel('Amount')

plt.title('Kind of suicide')
query = '''

select

     nationality, 

     count(*) as cnt

from

    (select

        case

            when nationality = 'South' and famous_for like '%African%' then 'South African'

            when nationality = 'South' and famous_for like '%Korean%' then 'South Korean'

            else nationality

        end nationality

    from sui) as suinat

group by nationality

order by cnt desc

'''



nation_suicide = pd.read_sql(query,lite3conn).head(10)

nation_suicide



sns.barplot(y="nationality", x="cnt", data=nation_suicide, palette = "Greens_d")

plt.ylabel('')

plt.xticks(range(0,66,5))

plt.xlabel('Amount')

plt.title('TOP 10 Nationality suicide')
sui = df[(df['cause_of_death'].fillna('unknown').str.contains('suicide'))]

nat = df[(df['cause_of_death'].fillna('unknown').str.contains('natural'))]



sns.set_style('dark',{"axes.facecolor": "1"})

sns.countplot(nat.death_year,palette=sns.color_palette("Reds_d",n_colors=1),label='Natural')

sns.countplot(sui.death_year,palette=sns.color_palette("Greys_d",n_colors=1),label='Suicide')

plt.legend(loc=2)

plt.xlabel('Year of death')

plt.ylabel('Amount')

plt.title('Celebrity deaths by suicide per year')

plt.show()
sui = df[(df['cause_of_death'].fillna('unknown').str.contains('suicide'))]

sns.stripplot(x="death_year", y="age", data=sui);

plt.xlabel('Year of death')

plt.ylabel('Age')

plt.title('Deaths ages by suicide per year')

plt.show()
sns.regplot(x="death_year", y="age", data=sui);

plt.xlabel('Year of death')

plt.ylabel('Age')

plt.title('Regression line and a 95% confidence interval for deaths ages by suicide per year')

plt.show()
sns.pointplot(x="death_year", y="age", data=sui, estimator=average, color='red');

sns.pointplot(x="death_year", y="age", data=sui, estimator=median, color='green');

plt.xlabel('Year of death')

plt.ylabel('Amount')

plt.yticks(range(20,71,5))

plt.title('Average(red) and median(green) for deaths ages by suicide per year')

plt.show()