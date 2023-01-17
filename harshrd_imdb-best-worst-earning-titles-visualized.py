# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # data visualization package

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# dropping duplicate movie titles and na values
df = pd.read_csv("/kaggle/input/movie_metadata.csv")
# print(df.head())
df = df.dropna()
df = df.drop_duplicates(subset=['movie_title'])
y_ranges=np.arange(1990,2000)
print(y_ranges)
df_counts=[]

# splitting dataset based on release year
df_years=[df.loc[df['title_year']==y] for y in y_ranges]

# store number of movies released in each year
for i in df_years:
    df_counts.append(i.count())
# storing avg duration of movies of each year
dur_means=[]
for j in df_years:
    print(j['title_year'].mean())
    dur_means.append(j['duration'].mean())
print(dur_means)
# plot movie year with avg duration of that year
plt.scatter(y_ranges,dur_means,color='blue')
plt.xticks(np.arange(y_ranges.min(),y_ranges.max()+1,1))
plt.xlabel("Title Year")
plt.ylabel("Avg Duration")
plt.show()
# plot top earning movie titles with gross collections and budget
# green dots depict gross collections
# red dots depict budget
plt.rcParams['figure.figsize'] = [20,6]
for i in df_years:
    sorted_i=i.sort_values(by='gross',ascending=False).head(5)
    plt.scatter(sorted_i['movie_title'],sorted_i['gross'],color='green',label='Gross')
    plt.scatter(sorted_i['movie_title'],sorted_i['budget'],color='red',label='Budget')
    plt.xlabel("Movie Title",fontsize=20)
    plt.show()
    top_earners=pd.DataFrame({'Year':sorted_i['title_year'],
                              'Movie Title':sorted_i['movie_title'],
                              'Earnings':(sorted_i['gross']-sorted_i['budget'])
                             })
    print('Top 5 Movies')
    print(top_earners.sort_values(by='Earnings',ascending=False))
# plot movie titles with net revenue
# plots with green dots indicate top earning titles
# plots with red dots indicate worst earning titles
plt.rcParams['figure.figsize'] = [20,6]
for i in df_years:
    sorted_i_desc=i.sort_values(by='gross',ascending=False).head(5)
    sorted_i_asc=i.sort_values(by='gross',ascending=True).head(5)
    revenue_best=sorted_i_desc['gross']-sorted_i_desc['budget']
    revenue_worst=sorted_i_asc['gross']-sorted_i_asc['budget']
    plt.xlabel("Movie Title",fontsize=20)
    plt.ylabel("Net Revenue",fontsize=20)
    plt.scatter(sorted_i_desc['movie_title'],revenue_best,c='green',s=100)
    plt.show()
    plt.scatter(sorted_i_asc['movie_title'],revenue_worst,c='red',s=100)
    plt.show()
    top_earners=pd.DataFrame({'Year':sorted_i_desc['title_year'],
                              'Movie Title':sorted_i_desc['movie_title'],
                              'Earnings':(sorted_i_desc['gross']-sorted_i_desc['budget'])
                             })
    worst_earners=pd.DataFrame({'Year':sorted_i_asc['title_year'],
                                'Movie Title':sorted_i_asc['movie_title'],
                              'Earnings':(sorted_i_asc['gross']-sorted_i_asc['budget'])
                             })
    print('Top Earning Movies\n')
    print(top_earners.sort_values(by='Earnings',ascending=False))
    print('\nWorst Earning Movies\n')
    print(worst_earners.sort_values(by='Earnings',ascending=True))