# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.




#criar dataframe from excel



df = pd.read_excel('/kaggle/input/novoexport/exames citologia 1.10 a 15.10 DF - novo.xlsx',

                   header=None, names=['cod', 'datadia', 'ubs'], 

                   skipfooter=33)



# convert data lançamento para tipo date

# df['Data de lançamento'] = pd.to_datetime(df['Data de lançamento'])



df.head()











df = df.drop([1])

print(df.info())

print(df.head())


from IPython.display import Image

Image("/kaggle/input/imagens-importacao/Screen Shot 2019-10-28 at 11.20.29.png")


Image("/kaggle/input/imagens-importacao/Screen Shot 2019-10-28 at 11.20.40.png")



Image("/kaggle/input/imagens-importacao/Screen Shot 2019-10-28 at 11.20.58.png")


Image("/kaggle/input/imagens-importacao/Screen Shot 2019-10-28 at 11.20.50.png")


Image("/kaggle/input/imagens-importacao/Screen Shot 2019-10-28 at 11.21.34.png")


Image("/kaggle/input/imagens-importacao/Screen Shot 2019-10-28 at 11.23.02.png")


Image("/kaggle/input/imagens-importacao/Screen Shot 2019-10-28 at 11.23.36.png")
# df.groupby(month('data'))

df.datadia.value_counts()




# df['data'] =  pd.to_datetime(df['data'], format='%d/%m/%Y', errors='coerce')



print(df[df['datadia'] == '01/10/2019\n'])

print(df[df['datadia'] == '01/10/2019'])

print(df[df['datadia'] == '11/10/2019\n'])

print(df[df['datadia'] == '15/10/2019'])



df = df.drop([2430, 2429, 1763, 1764])
df.head()



df2 = df
df2.drop('cod', axis=1)
df2['total'] = df2.groupby(['datadia', 'ubs']).transform('count')
# df2.drop('cod', axis=1, inplace=True)


print(df2.head(100))

df2.drop_duplicates(subset=['datadia', 'ubs'], inplace=True)



df2.info()
df2
df.total.sum()


# import seaborn as sns

# import matplotlib.pyplot as plt

# sns.barplot(data=df2, x="data", y="total")



# x_dates = df2['data'].df2.strftime('%Y-%m-%d').sort_values().unique()

# plt.set_xticklabels(labels=x_dates, rotation=45, ha='right')



# # plt.xticks(rotation=45)

# plt.show()





import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns



# RANDOM DATA

np.random.seed(62918)

emp = pd.DataFrame({'uniqueClientExits': [np.random.randint(15) for _ in range(50)],

                    '12monthsEnding': pd.to_datetime(

                                          np.random.choice(

                                              pd.date_range('2019-08-01', periods=50), 

                                          50)

                                      )

                   }, columns = ['uniqueClientExits','12monthsEnding'])



# PLOTTING

fig, ax = plt.subplots(figsize = (12,6))    

fig = sns.barplot(x = "12monthsEnding", y = "uniqueClientExits", data = emp, 

                  estimator = sum, ci = None, ax=ax)



x_dates = emp['12monthsEnding'].dt.strftime('%Y-%m-%d').sort_values().unique()

ax.set_xticklabels(labels=x_dates, rotation=45, ha='right')







import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns



# RANDOM DATA

# np.random.seed(62918)

# emp = pd.DataFrame({'uniqueClientExits': [np.random.randint(15) for _ in range(50)],

#                     '12monthsEnding': pd.to_datetime(

#                                           np.random.choice(

#                                               pd.date_range('2018-01-01', periods=50), 

#                                           50)

#                                       )

#                    }, columns = ['uniqueClientExits','12monthsEnding'])



# PLOTTING

# df['datadia'] = df['datadia'].str[:10]

# df['dia'] = df['datadia'].str[:10]

fig, ax = plt.subplots(figsize = (12,6))    

fig = sns.barplot(x = "datadia", y = "cod", data = df, 

                  estimator = lambda x: len(x), ci = None, ax=ax)



x_dates = df['datadia'].sort_values().unique()

ax.set_xticklabels(labels=x_dates, rotation=45, ha='right')

plt.title('Quantidade de Exames Diários', fontsize=24)



plt.ylabel('Quantidade', fontsize=24)

plt.xlabel('Data', fontsize=24)





fig.ticklabel_format(style='plain', axis='y',useOffset=False)


