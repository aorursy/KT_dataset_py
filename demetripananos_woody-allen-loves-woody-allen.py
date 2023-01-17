# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
def print_full(x):

    pd.set_option('display.max_rows', len(x))

    print(x)

    pd.reset_option('display.max_rows')





df = pd.read_csv('../input/movie_metadata.csv')

r = pd.melt(df,id_vars=['director_name'], value_vars=['actor_1_name','actor_2_name','actor_3_name'], value_name='actor').dropna()

s = r[['director_name','actor']].groupby(['director_name','actor']).size().sort_values(ascending = False)



print_full(s)
s = pd.DataFrame(s).reset_index()



s.columns = ['Director','Actor','NomMovies']



s = s.sort_values(by = ['Director','Actor','NomMovies'],ascending=[True,True,False]).set_index(['Director','Actor'])



print_full(s)