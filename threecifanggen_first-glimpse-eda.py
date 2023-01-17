# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import functools as fts



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
albumlist = pd.read_csv('../input/albumlist.csv', encoding='cp1256')
albumlist.head()
sns.distplot(albumlist['Year'])
albumlist['Artist'].value_counts()[:15].plot('bar')
genre_list = fts.reduce(lambda x, y: x + y,

                        list(albumlist['Genre']

                             .apply(lambda x: x.replace('ت', ' ').replace(', & ', ', ').split(', '))))



pd.Series(genre_list).value_counts()[:15].plot('bar')
sub_genre_list = fts.reduce(lambda x, y: x + y,

                        list(albumlist['Subgenre'].apply(lambda x: x.split(', '))))



pd.Series(sub_genre_list).value_counts()[:15].plot('bar')
albumlist['decade'] = albumlist['Year'].apply(lambda x: round(x, -1))
decade_group = albumlist.groupby(['decade'])['Genre']





def return_counts(grouped):

    return pd.DataFrame({

        'Genre': pd.Series(fts.reduce(lambda x, y: x + y, list(grouped.apply(lambda x: x

                               .replace('ت', ' ')

                               .replace(', & ', ', ')

                               .split(', '))))).value_counts().index,

        'Percentage': pd.Series(fts.reduce(lambda x, y: x + y, list(grouped.apply(lambda x: x

                               .replace('ت', ' ')

                               .replace(', & ', ', ')

                               .split(', '))))).value_counts()/len(fts.reduce(lambda x, y: x + y, list(grouped.apply(lambda x: x

                               .replace('ت', ' ')

                               .replace(', & ', ', ')

                               .split(', ')))))

    })



decade_genre = decade_group.apply(return_counts)

decade_genre.reset_index(level=0, inplace=True)

              
decade_genre.head()
sns.set_color_codes("pastel")

sns.factorplot(x="decade", y="Percentage", data=decade_genre, hue = 'Genre')