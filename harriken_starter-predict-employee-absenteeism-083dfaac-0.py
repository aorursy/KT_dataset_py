from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

print(os.listdir('../input'))
# Distribution graphs (histogram/bar graph) of column data

def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):

    nunique = df.nunique()

    df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 50]] # For displaying purposes, pick columns that have between 1 and 50 unique values

    nRow, nCol = df.shape

    columnNames = list(df)

    nGraphRow = (nCol + nGraphPerRow - 1) / nGraphPerRow

    plt.figure(num = None, figsize = (6 * nGraphPerRow, 8 * nGraphRow), dpi = 80, facecolor = 'w', edgecolor = 'k')

    for i in range(min(nCol, nGraphShown)):

        plt.subplot(nGraphRow, nGraphPerRow, i + 1)

        columnDf = df.iloc[:, i]

        if (not np.issubdtype(type(columnDf.iloc[0]), np.number)):

            valueCounts = columnDf.value_counts()

            valueCounts.plot.bar()

        else:

            columnDf.hist()

        plt.ylabel('counts')

        plt.xticks(rotation = 90)

        plt.title(f'{columnNames[i]} (column {i})')

    plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)

    plt.show()

# Correlation matrix

def plotCorrelationMatrix(df, graphWidth):

    filename = df.dataframeName

    df = df.dropna('columns') # drop columns with NaN

    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values

    if df.shape[1] < 2:

        print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')

        return

    corr = df.corr()

    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')

    corrMat = plt.matshow(corr, fignum = 1)

    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)

    plt.yticks(range(len(corr.columns)), corr.columns)

    plt.gca().xaxis.tick_bottom()

    plt.colorbar(corrMat)

    plt.title(f'Correlation Matrix for {filename}', fontsize=15)

    plt.show()

# Scatter and density plots

def plotScatterMatrix(df, plotSize, textSize):

    color_wheel = {1: "#aa0000", 

               2: "#0000aa", 

               3: "#0000ff"}

    colors = df["sick"].map(lambda x: color_wheel.get(x + 1))



    

    df = df.select_dtypes(include =[np.number]) # keep only numerical columns

    # Remove rows and columns that would lead to df being singular

    df = df.dropna('columns')

    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values

    columnNames = list(df)

    if len(columnNames) > 10: # reduce the number of columns for matrix inversion of kernel density plots

        columnNames = columnNames[:10]

    df = df[columnNames]

    

    ax = pd.plotting.scatter_matrix(df,color=colors, alpha=0.15, figsize=[plotSize, plotSize], diagonal='kde',)

    corrs = df.corr().values

    for i, j in zip(*plt.np.triu_indices_from(ax, k = 1)):

        ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', ha='center', va='center', size=textSize)

    plt.suptitle('Scatter and Density Plot')

    plt.show()

nRowsRead = None # specify 'None' if want to read whole file

# commentInteractions.csv has 93816 rows in reality, but we are only loading/previewing the first 1000 rows

df1 = pd.read_csv('../input/commentInteractions.csv', delimiter=',', nrows = nRowsRead)

df1.dataframeName = 'commentInteractions.csv'

nRow, nCol = df1.shape

print(f'There are {nRow} rows and {nCol} columns')
df1.head(5)
plotPerColumnDistribution(df1, 10, 5)
nRowsRead = None # specify 'None' if want to read whole file

# comments_by_employees_in_anonymous_forum.csv has 5072 rows in reality, but we are only loading/previewing the first 1000 rows

df2 = pd.read_csv('../input/comments_by_employees_in_anonymous_forum.csv', delimiter=',', nrows = nRowsRead)

df2.dataframeName = 'comments_by_employees_in_anonymous_forum.csv'

nRow, nCol = df2.shape

print(f'There are {nRow} rows and {nCol} columns')
df2.head(5)
plotPerColumnDistribution(df2, 10, 5)
plotCorrelationMatrix(df2, 8)
df4 = pd.read_csv('../input/employeeAbsenteeism.csv', delimiter=',')

df4.dataframeName = 'employeeAbsenteeism'

nRow, nCol = df4.shape

print(f'There are {nRow} rows and {nCol} columns')

df4.head(5)
sickSet = set(df4['employee'])

a = len(df4['employee'])

b =  (len(sickSet))

print(f'In {df4.dataframeName}, there are {a} employee ids entries and  {b} of those are unique')



def is_sick(id_str):

    return id_str in sickSet



df2['sick'] =  df2['employee'].apply(is_sick)



a = len(df2['employee'])

b =  (len(set(df2['employee'])))

print(f'In {df2.dataframeName}, there are {a} employee ids entries and  {b} of those are unique')



b =  (len(set(df2['employee'])))

d = df2.loc[df2['employee'].isin(sickSet)]

c = (len(set(d['employee'])))

    



print(f'In {df2.dataframeName}, of the  {b}  unique employees {c} were sick at some point')



b =  (len(sickSet))

print(f'So of the sick, only {c} out of {b} wrote a comment in {df2.dataframeName} ; ( {c/b}%)')





print(f'the ratio in non sick is...')



# check if any duplciated rows... none so far...

#df2[df2.duplicated(keep=False)]



df5 = pd.read_csv('../input/lastParticipationExists.csv', delimiter=',')

df5.dataframeName = 'lastParticipationExists.csv'

nRow, nCol = df5.shape

print(f'There are {nRow} rows and {nCol} columns')

df5.head(5)

all =(set(df5['employee']))

len(all)
wrotesomething = set (df2['employee'])

print(f'the ratio for all emplyees is { len(wrotesomething) } wrote a comment out of {len(all)}... {(len(wrotesomething)/len(all))} %')

print(f'inconclusive')

print(f'inconclusive')



df4.drop_duplicates(inplace=True)

df4['employee'].value_counts().plot.hist(bins=12, alpha=0.5)



df4['employee'].value_counts()

df4.loc[df4['employee']=='yKX']

#mask = (df['date'] > start_date) & (df['date'] <= end_date)

plotScatterMatrix(df2, 9, 10)
df4['from'] = pd.to_datetime(df4['from'])

df4['to']   = pd.to_datetime(df4['to'])

df2['commentDate'] =pd.to_datetime(df2['commentDate'])





def calculatePostsPerWeekAllTimeAndWB4(id_str,sick_date, n):

    #select posts from df2

    # test with df4.loc[df4['employee']=='yKX']

    xx = df2.loc[df2['employee']==id_str]

    if (xx.empty):

        return (0)

    else:

        #print(print(xx.dtypes))   

        ndays = -(xx['commentDate'].min() - pd.to_datetime(sick_date) )

        nposts = len(xx.index)

        #print(ndays.days)

        WB4mask = (xx['commentDate'] > (sick_date- pd.DateOffset(days=7)) ) & (xx['commentDate'] <= sick_date)

        WB4 = xx.loc[WB4mask]

        if not (WB4.empty):

            WB4_c =  len(WB4.index)

            if not (WB4_c > -1):

                WB4_c = -1

        else:

            WB4_c = 0



        MB4mask = (xx['commentDate'] > (sick_date- pd.DateOffset(days=7*4)) ) & (xx['commentDate'] <= sick_date)

        MB4 = xx.loc[MB4mask]

        if not (MB4.empty):

            MB4_c =  len(MB4.index)

            if not (MB4_c > -1 ):

                MB4_c = -1

        else:

            MB4_c = 0

        Ndays_since_first_comment_to_sick = ndays.days   

        res = [(nposts/(ndays.days/7)),WB4_c,MB4_c, nposts,Ndays_since_first_comment_to_sick]

        return res[n]

        #return pd.Series([(nposts/(ndays.days/7)),WB4_c,MB4_c, nposts,ndays.days])





# test the function

# print(df4.dtypes)

id_str = "qKO"

sick_date = df4.iloc[3]["from"]

#print(sick_date)

print(calculatePostsPerWeekAllTimeAndWB4(id_str,sick_date,1))



#calculatePostsPerWeekAllTimeAndWB4(df4['employee'], df4['from'])

df4['npostsperweek'] = df4.apply(lambda x: calculatePostsPerWeekAllTimeAndWB4(x['employee'], x['from'],0) , axis=1)

df4['npostsWB4Sick'] = df4.apply(lambda x: calculatePostsPerWeekAllTimeAndWB4(x['employee'], x['from'],1) , axis=1)

df4['npostsMB4Sick'] = df4.apply(lambda x: calculatePostsPerWeekAllTimeAndWB4(x['employee'], x['from'],2) , axis=1)

df4['nposts'] = df4.apply(lambda x: calculatePostsPerWeekAllTimeAndWB4(x['employee'], x['from'],3) , axis=1)

df4['Ndays_since_first_comment_to_sick'] = df4.apply(lambda x: calculatePostsPerWeekAllTimeAndWB4(x['employee'], x['from'],4) , axis=1)

df4.head(5)

#df4['postsWB4Sick'] = df4.apply(lambda x: calculatePostsPerWeekAllTimeAndWB4(df4['employee'], df4['from'])[2], axis=1)

#df4['ratio'] = df4['postsWB4Sick'] / df4['Alltimepostperweek']

dd = df4.groupby(['npostsWB4Sick','npostsMB4Sick']).size()

print(dd)



df4.head(100)

plotPerColumnDistribution(df4[1:144], 10, 5)
import matplotlib.pyplot as plt

plt.scatter(df4.npostsWB4Sick, df4.npostsMB4Sick/4

, s=df4.nposts)

plt.title('Employee posts less the week before getting sick')

plt.xlabel('Number of posts the week before sick leave day')

plt.ylabel('Number of posts per week the month before sick leave day')



# sample points 

X = df4.npostsWB4Sick

Y = df4.npostsMB4Sick/4

#YSum = df4.groupby(['npostsWB4Sick'],['npostsMB4Sick']).count()

#print(YSum)

# solve for a and b

def best_fit(X, Y):



    xbar = sum(X)/len(X)

    ybar = sum(Y)/len(Y)

    n = len(X) # or len(Y)



    numer = sum([xi*yi for xi,yi in zip(X, Y)]) - n * xbar * ybar

    denum = sum([xi**2 for xi in X]) - n * xbar**2



    b = numer / denum

    a = ybar - b * xbar



    print('best fit line:\ny = {:.2f} + {:.2f}x'.format(a, b))



    return a, b



# solution

a, b = best_fit(X, Y)

#best fit line:

#y = 0.80 + 0.92x



# plot points and fit line

import matplotlib.pyplot as plt

plt.scatter(X, Y)

yfit = [a + b * xi for xi in X]

plt.plot(X, yfit)



plt.title('Employee posts less the week before getting sick')

plt.xlabel('Number of posts the week before sick leave day')

plt.ylabel('Number of posts per week the month before sick leave day')



nRowsRead = None # specify 'None' if want to read whole file

# comments_by_employees_in_anonymous_forum.csv has 5072 rows in reality, but we are only loading/previewing the first 1000 rows

df6 = pd.read_csv('../input/commentInteractions.csv', delimiter=',', nrows = nRowsRead)

df6.dataframeName = 'commentInteractions.csv'

nRow, nCol = df2.shape

print(f'There are {nRow} rows and {nCol} columns')

#df6.head(55)

df6.dtypes

dd = df6.groupby(['interaction']).size()

print(dd)
df6['actionDate'] = pd.to_datetime(df6['actionDate'])





def calculateLikesPerWeekAllTimeAndWB4(id_str,sick_date, n):

    #select posts from df2

    # test with df4.loc[df4['employee']=='yKX']

    xx = df6.loc[df6['employee']==id_str]

    if (xx.empty):

        return (0)

    else:

        #print(print(xx.dtypes))   

        ndays = -(xx['actionDate'].min() - pd.to_datetime(sick_date) )

        nposts = len(xx.index)

        #print(ndays.days)

        WB4mask = (xx['actionDate'] > (sick_date- pd.DateOffset(days=7)) ) & (xx['actionDate'] <= sick_date)

        WB4 = xx.loc[WB4mask]

        if not (WB4.empty):

            WB4_c =  len(WB4.index)

            if not (WB4_c > -1):

                WB4_c = -1

        else:

            WB4_c = 0



        MB4mask = (xx['actionDate'] > (sick_date- pd.DateOffset(days=7*4)) ) & (xx['actionDate'] <= sick_date)

        MB4 = xx.loc[MB4mask]

        if not (MB4.empty):

            MB4_c =  len(MB4.index)

            if not (MB4_c > -1 ):

                MB4_c = -1

        else:

            MB4_c = 0

        Ndays_since_first_comment_to_sick = ndays.days   

        res = [(nposts/((0.001 + ndays.days)/7)),WB4_c,MB4_c, nposts,Ndays_since_first_comment_to_sick]

        return res[n]

        #return pd.Series([(nposts/(ndays.days/7)),WB4_c,MB4_c, nposts,ndays.days])





# test the function

# print(df4.dtypes)

id_str = "qKO"

sick_date = df4.iloc[3]["from"]

#print(sick_date)

print(calculatePostsPerWeekAllTimeAndWB4(id_str,sick_date,1))



#calculatePostsPerWeekAllTimeAndWB4(df4['employee'], df4['from'])

df4['nlikesperweek'] = df4.apply(lambda x: calculateLikesPerWeekAllTimeAndWB4(x['employee'], x['from'],0) , axis=1)

df4['nlikesWB4Sick'] = df4.apply(lambda x: calculateLikesPerWeekAllTimeAndWB4(x['employee'], x['from'],1) , axis=1)

df4['nlikesMB4Sick'] = df4.apply(lambda x: calculateLikesPerWeekAllTimeAndWB4(x['employee'], x['from'],2) , axis=1)

df4['nposts']        = df4.apply(lambda x: calculateLikesPerWeekAllTimeAndWB4(x['employee'], x['from'],3) , axis=1)

df4['Ndays_since_first_like_to_sick'] = df4.apply(lambda x: calculateLikesPerWeekAllTimeAndWB4(x['employee'], x['from'],4) , axis=1)



#df4['postsWB4Sick'] = df4.apply(lambda x: calculatePostsPerWeekAllTimeAndWB4(df4['employee'], df4['from'])[2], axis=1)

#df4['ratio'] = df4['postsWB4Sick'] / df4['Alltimepostperweek']
df4.head(5)
dd = df4.groupby(['nlikesWB4Sick','nlikesMB4Sick']).size()

print(dd)

# sample points 

X = df4.nlikesWB4Sick

Y = df4.nlikesMB4Sick/4

#YSum = df4.groupby(['npostsWB4Sick'],['npostsMB4Sick']).count()

#print(YSum)

# solve for a and b

def best_fit(X, Y):



    xbar = sum(X)/len(X)

    ybar = sum(Y)/len(Y)

    n = len(X) # or len(Y)



    numer = sum([xi*yi for xi,yi in zip(X, Y)]) - n * xbar * ybar

    denum = sum([xi**2 for xi in X]) - n * xbar**2



    b = numer / denum

    a = ybar - b * xbar



    print('best fit line:\ny = {:.2f} + {:.2f}x'.format(a, b))



    return a, b



# solution

a, b = best_fit(X, Y)

#best fit line:

#y = 0.80 + 0.92x



# plot points and fit line

import matplotlib.pyplot as plt

plt.scatter(X, Y)

yfit = [a + b * xi for xi in X]

plt.plot(X, yfit)



plt.title('Employee likes less the week before getting sick?')

plt.xlabel('Number of (likes+dislikes) the week before sick leave day')

plt.ylabel('Number of (likes+dislikes) per week the month before sick leave day')

# unlikes

df6 = df6.loc[df6['interaction']<0 ]

df4['nUNlikesperweek'] = df4.apply(lambda x: calculateLikesPerWeekAllTimeAndWB4(x['employee'], x['from'],0) , axis=1)

df4['nUNlikesWB4Sick'] = df4.apply(lambda x: calculateLikesPerWeekAllTimeAndWB4(x['employee'], x['from'],1) , axis=1)

df4['nUNlikesMB4Sick'] = df4.apply(lambda x: calculateLikesPerWeekAllTimeAndWB4(x['employee'], x['from'],2) , axis=1)

df4['nposts']        = df4.apply(lambda x: calculateLikesPerWeekAllTimeAndWB4(x['employee'], x['from'],3) , axis=1)

df4['Ndays_since_first_UNlike_to_sick'] = df4.apply(lambda x: calculateLikesPerWeekAllTimeAndWB4(x['employee'], x['from'],4) , axis=1)



X = df4.nUNlikesWB4Sick

Y = df4.nUNlikesMB4Sick/4

plt.scatter(X, Y)

yfit = [a + b * xi for xi in X]

plt.plot(X, yfit)

plt.title('Employee UNlike less the week before getting sick?')

plt.xlabel('Number of (dislikes) the week before sick leave day')

plt.ylabel('Number of (dislikes) per week the month before sick leave day')

df2['sick'].describe()
df2.head()
df22 = df2[['likes','sick']]

df22.describe()

df2.groupby(['sick']).mean()
