# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
excel_file = '/kaggle/input/movies.xls'

movies = pd.read_excel(excel_file)
movies.head()
movies_sheet1 = pd.read_excel(excel_file, sheet_name=0, index_col=0)

movies_sheet1.head()
movies_sheet2 = pd.read_excel(excel_file, sheet_name=1, index_col=0)

movies_sheet2.head()
movies_sheet3 = pd.read_excel(excel_file, sheet_name=2, index_col=0)

movies_sheet3.head()
movies = pd.concat([movies_sheet1, movies_sheet2, movies_sheet3])
movies.shape
xlsx = pd.ExcelFile(excel_file)

movies_sheets = []

for sheet in xlsx.sheet_names:

    movies_sheets.append(xlsx.parse(sheet))

movies = pd.concat(movies_sheets)
movies.tail()

sorted_by_gross = movies.sort_values(['Gross Earnings'], ascending=False)
sorted_by_gross["Gross Earnings"].head(10)
import matplotlib.pyplot as plt
sorted_by_gross['Gross Earnings'].head(10).plot(kind="barh")

plt.show()
movies['IMDB Score'].plot(kind="hist")

plt.show()
movies.describe()
movies["Gross Earnings"].mean()
movies_subset_columns = pd.read_excel(excel_file, parsecols=6)

movies_subset_columns.head()
movies["Net Earnings"] = movies["Gross Earnings"] - movies["Budget"]
sorted_movies = movies[['Net Earnings']].sort_values(['Net Earnings'], ascending=[False])

sorted_movies.head(10)['Net Earnings'].plot.barh()

plt.show()
movies_subset = movies[['Year', 'Gross Earnings']]

movies_subset.head()
earnings_by_year = movies_subset.pivot_table(index=['Year'])

earnings_by_year.head()
earnings_by_year.plot()

plt.show()
movies_subset = movies[['Country', 'Language', 'Gross Earnings']]

movies_subset.head()
earnings_by_co_lang = movies_subset.pivot_table(index=['Country', 'Language'])

earnings_by_co_lang.head()
earnings_by_co_lang.head(20).plot(kind='bar', figsize=(20,8))

plt.show()
movies.to_excel('output.xlsx')
movies.to_excel('output.xlsx', index=False)
writer = pd.ExcelWriter('output.xlsx', engine='xlsxwriter')

movies.to_excel(writer, index=False, sheet_name='report')

workbook = writer.book

worksheet = writer.sheets['report']
header_fmt = workbook.add_format({'bold': True})

worksheet.set_row(0, None, header_fmt)
writer.save()