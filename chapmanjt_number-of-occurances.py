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
import matplotlib.pyplot as plt
df = pd.read_csv('/kaggle/input/data-police-shootings/fatal-police-shootings-data.csv')



df.sample(20)
df.describe()
def get_occ(data, feat, return_df=False, verb=False) :

    

    """

    Takes a DataFrame and feature of interest.

    Returns the number of occurances for each value of the specified feature.

    OR

    Returns a DataFrame

    """

    

    # get unique

    uniq = data[feat].unique()

    if verb: print(uniq, type(uniq))

    

    

    try :

        uniq = np.sort(uniq)

        

    except :

        print("Warning: will not sort.")

        

    finally :

    

        occ = []



        for i in uniq :

            occ.append([i])

            occ[-1].append( data[feat][data[feat]==i].count() )

        

        if return_df :



            df_feat = pd.DataFrame(occ)

            df_feat.columns = [feat, 'occ']

            df_feat = df_feat.set_index(feat)

            df_feat = df_feat.sort_values('occ', ascending=False)



            return df_feat



        return np.array(occ)



        





# usage

# df_occ = get_occ(df, 'number')

# df_occ
def reduce(data, feat, value) :

    

    """

    shorthand.

    Takes a DataFrame, a feature of interest, and a vlue of that feature.

    Returns the DataFrame filtered by <feat>==<'value'>.

    """

    

    d = data[data[feat]==value]



    # dbg

    # print('\ti-reduce:', end='')

    

    return d

    

# usage 

# d15 = reduce(df, 'year', value=2015)

# d15

# d15B = reduce(d15, 'race', 'B')

# d15B
# splice year month day

df['dt'] = pd.to_datetime(df.date)



df['year'] = df.dt.apply(lambda x: x.year)

df['month'] = df.dt.apply(lambda x: x.month)

df['day'] = df.dt.apply(lambda x: x.day)

df['number'] = df.dt.apply(lambda x: x.dayofyear)

df
from scipy.optimize import curve_fit



# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html



# linear

def func_lin(t, b, e): 

    

    return b * t + e



# polynomial $Ax^4+Bx^3$

def func_poly(t, a, b, c , d, i, e): 

       

    return  a * t ** 4 + b * t ** 3 +  c * t ** 2  + d * t + i * t ** .5 + e



def plot_curve_fitting(func, dat, axes, title, verb=False):



    # get scipy curve fitting params

    pop, pcn = curve_fit(func, dat.index.values, dat.values.flatten())

    if verb : print(pop)

    

    axes.plot(dat.index.values, func(dat.index.values, *pop), 'r--')

    axes.plot(dat)
df_doy = get_occ(df, 'number', return_df=True)

df_doy[:15].T
fig = plt.figure(figsize=(20,7))



# curve fitting

d = get_occ(df, 'number', return_df=True).sort_index()



# plot with fitted linear curve

ax = fig.add_subplot(211)

plot_curve_fitting(func_lin, d, ax, 'test', verb=True)



# plot with fitted polynomial curve

ax = fig.add_subplot(212)



plot_curve_fitting(func_poly, d, ax, 'test', verb=True)
stats = []



fig = plt.figure(figsize=(30,20))

plt_idx = 1





for y in range(2015, 2021) :

    

    y_df = reduce(df, 'year', y)

    

    # plot

    ax = fig.add_subplot(6,1,plt_idx)

    ax.set_title(str(y) + ' Occ')

    occperdat = [ y_df[y_df.number==d].id.count() for d in range(366) ]

    ax.plot(occperdat)

    plt_idx += 1

    

    y_occ = get_occ(y_df, 'number')

    

    v = y_occ[:,1]

    

    # dbg

    # print(v)

    

    stats.append([y, v.min(), v.mean(), v.max(), v.std()])

    



    idx =  np.argsort(y_occ[:,1])

    hi = y_occ[idx[-10:]]

  

    [ stats[-1].append(x) for x in hi ]

    

# print(stats, his)



df_y_occ = pd.DataFrame(stats)

df_y_occ.columns = ['year', 'min', 'mean', 'max', 'std', 'hi', 'hi', 'hi', 'hi', 'hi', 'hi', 'hi', 'hi', 'hi', 'hi']

df_y_occ



# for each 'hi' column, starting with the highest for that year, is the day-of-year and number of occurances
get_occ(df, 'manner_of_death', return_df=True)
df_arms = get_occ(df, 'armed', return_df=True)

df_arms.T
# armed values with only one occurrance

# df_arms[df_arms.occ==1].T
# armed values with over 10 occurances

df_arms[df_arms.occ>10].T
# df_age = get_occ(df, 'age', return_df=True)

df_age = get_occ(df, 'age')



df_age = pd.DataFrame(df_age[:-1])

df_age.columns = ('age','occ')

df_age = df_age.set_index('age')



df_age.iloc[:15].T
# slice to get occurnaces for age > 35, but only the first 10

df_age.loc[35:].iloc[:10].T
get_occ(df, 'gender', return_df=True)
get_occ(df, 'race', return_df=True).T
get_occ(df, 'state', return_df=True).T
# The 15 states with the most occurances

df_st = get_occ(df, 'state', return_df=True)

df_st[:15].T
get_occ(df, 'signs_of_mental_illness', return_df=True)
get_occ(df, 'threat_level', return_df=True)
get_occ(df, 'flee', return_df=True).T
get_occ(df, 'body_camera', return_df=True)
def cross(data, feat1, feat2, verb=False) :

    """

    Takes the DataFrame and two features to reduce it by.

    Returns DataFrame of the number of occurances of the cross of the two features.

    """

    

    # storage

    tab = []



    

    # for each value of feat1

    for x in data[feat1].unique() :

        

        d={}

        

        # reduce based on value x of feat1

        dd = reduce(data, feat1, x)

              

        # record value

        d.update({'feat1' : x})

        

        # for each value of feat2

        for y in dd[feat2].unique() :

            

            # reduce based on value y of feat2

            ddd = reduce(dd, feat2, y)

            

            # record number of occurances

            d.update({y : ddd.id.count() })



        # add to list of dictionaries per feat1

        tab.append(d)

            

    if verb : print(tab)



    # make DataFrame from np.array

    d = pd.DataFrame(tab)

    

    # label columns        

    d.columns = [feat1] + list(data[feat2].unique()) #list(data[feat2].unique())

    

    # set index

    d = d.set_index(feat1)

        

    # replace NaN with zeros

    d = d.replace(np.NaN, 0)

    

    # sort

    d = d.sort_values(d.columns[0], ascending=False)

            

    return d



# # usage

# df_axm = cross(df, 'armed', 'manner_of_death')

# df_axm
def make_heatmap(data, show_vals=False) :

    """

    Takes a DataFrame, the result of the cross(df,feat1,feat2).

    Returns the heatmap of the number of occurances of the cross of the two features.

    """

    

    # instance figure and axes

    figure = plt.figure(figsize=(20,20))

    ax = figure.add_subplot()    

    

    # plot data

    ax.imshow(data.values)

    

    # x-axis

    ax.set_xticks(np.arange(len(data.columns)))

    ax.set_xticklabels(data.columns, rotation=90)



    # y-axis

    ax.set_yticks(np.arange(len(data.index)))

    ax.set_yticklabels(data.index)

    

    # add values to plot

    if show_vals :

        for i in range(len(data)) :

            for j in range(len(data.columns)) :



                text = ax.text(j,i, data[data.columns[j]].iloc[i], ha='center', va='center', color='w')



                # verify

                # print(i, j, text)

                

    return ax

   

# # usage

# # df_axm = cross(df, 'armed', 'manner_of_death')

# make_heatmap(df_axm[:10].T, show_vals=True)
# usage - nested

make_heatmap(cross(df, 'armed', 'signs_of_mental_illness')[:10].T, show_vals=True)
considering = ['manner_of_death', 'race', 'signs_of_mental_illness', 'threat_level', 'month', 'state']



for f in df[considering] :

    for g in df[considering] :

        if f < g :

            print('eval:', f, '_x_', g)

            

            # get number of occurances

            d = cross(df, f,g )

            

            # size up and limit data for displaying               

            if d.values.shape[0] > d.values.shape[1] : d = d.T

                

            if d.shape[0] > 15 : d = d[:15]

            elif d.shape[1] > 15 : d = d[d.columns[:15]]

                        

            make_heatmap(d, show_vals=True)

            plt.savefig(f+'_x_'+g+'.png')