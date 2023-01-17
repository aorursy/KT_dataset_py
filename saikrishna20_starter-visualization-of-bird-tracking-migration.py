from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

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
    df = df.select_dtypes(include =[np.number]) # keep only numerical columns
    # Remove rows and columns that would lead to df being singular
    df = df.dropna('columns')
    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
    columnNames = list(df)
    if len(columnNames) > 10: # reduce the number of columns for matrix inversion of kernel density plots
        columnNames = columnNames[:10]
    df = df[columnNames]
    ax = pd.plotting.scatter_matrix(df, alpha=0.75, figsize=[plotSize, plotSize], diagonal='kde')
    corrs = df.corr().values
    for i, j in zip(*plt.np.triu_indices_from(ax, k = 1)):
        ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', ha='center', va='center', size=textSize)
    plt.suptitle('Scatter and Density Plot')
    plt.show()

nRowsRead = None  # specify 'None' if want to read whole file
# bird_tracking.csv may have more rows in reality, but we are only loading/previewing the first 1000 rows
df1 = pd.read_csv('/kaggle/input/bird_tracking.csv',index_col= 0, delimiter=',', nrows = nRowsRead)
df1.dataframeName = 'bird_tracking.csv'
nRow, nCol = df1.shape
print(f'There are {nRow} rows and {nCol} columns')
df1.head(5)
plotPerColumnDistribution(df1, 10, 5)
plotCorrelationMatrix(df1, 8)
plotScatterMatrix(df1, 18, 10)
df1.info()
df1.describe()
df1.bird_name.value_counts()
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize= (15,15))
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Latitude vs Longitude')
sns.lineplot(x='longitude' , y= 'latitude', hue = 'bird_name', data= df1, legend= 'full', alpha =  0.7)
plt.show()
plt.figure(figsize= (10,10))
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Latitude vs Longitude')
sns.scatterplot(x='longitude' , y= 'latitude', hue = 'bird_name', data= df1, legend= 'full', alpha = 0.3 )
plt.show()
df1.speed_2d.isna().sum()
x = df1.speed_2d[df1.bird_name == 'Eric']
x1 = df1.speed_2d[df1.bird_name == 'Nico']
x2 = df1.speed_2d[df1.bird_name == 'Sanne']
#print(x,x1,x2)
plt.figure(figsize=(15,9))
sns.distplot(x,bins =30,rug = True )
plt.show();
sns.distplot(x1,bins =20,rug = True )
plt.show();
sns.distplot(x2,bins =10,rug = True )
plt.show();

df1.head()
import datetime
#convert column to Date time format
df1.date_time = pd.to_datetime(df1.date_time)
df1.info()
df1.head()
df1.date_time.iloc[-1] - df1.date_time.iloc[0]
time_Eric = (df1.date_time[df1.bird_name == 'Eric']).astype('datetime64[ns]')
time_Eric.iloc[-1] - time_Eric.iloc[0]
time_Nico = (df1.date_time[df1.bird_name == 'Nico']).astype('datetime64[ns]')

time_Nico.iloc[-1] - time_Nico.iloc[0]
time_Sanne = (df1.date_time[df1.bird_name == 'Sanne']).astype('datetime64[ns]')
time_Sanne.head()
#time_Sanne.iloc[-1] - time_Sanne.iloc[0]
plt.title('Eric')
plt.xlabel('Time taken in DateTime format')
plt.ylabel('Number of observations')
plt.plot(time_Eric)
plt.show()

plt.title('Nico')
plt.xlabel('Time taken in DateTime format')
plt.ylabel('Number of observations')
plt.plot(time_Nico)
plt.show()


plt.title('Sanne')
plt.xlabel('Time taken in DateTime format')
plt.ylabel('Number of observations')
plt.plot(time_Sanne)
plt.show()


plt.figure(figsize=(30,12))
plt.plot( time_Eric, x, linestyle= '-', marker = 'o')
plt.show()
df1.describe()
#time_Eric.iloc[1] - time_Eric.iloc[0]
#time_Eric.iloc[2] - time_Eric.iloc[1]
time_Eric.iloc[3] - time_Eric.iloc[2]
elapsed_time = [time - time_Eric[0] for time in time_Eric]
elapsed_time[:5]
elapsed_days= np.array(elapsed_time) / datetime.timedelta(days=1)
elapsed_days[:5]
# it will convert the datetime series into number of days as we need to iterate we converted it to a array as we can't iterate over list
'''for (i,t) in enumerate(elapsed_days):
    print(i,t)
'''
next_day = 1
indeces=[]
daily_mean_speed = []
for (i,t) in enumerate(elapsed_days):
    # Here i is the count or index we use for later gtting the spped_2d and 
    # t is the datetime in the time delta RATIO OF timedata so it says 
    if t< next_day:
        indeces.append(i)
        # we get a list of indeces and those speed_2d are with in the same day
        #we get multiple list of each day indeces as a list
        #print(indeces)
    else:
        daily_mean_speed.append(np.mean(df1.speed_2d[indeces]))
        #using the list of indeces of day 1 and getting their values from Speed_2d and
        #cal mean of that day and storing in daiy_mean_speed by append
        # so now we get mean speed of day 1 as one value and the ssame continues for all days
        #print(indeces)
        next_day += 1
        indeces = []
daily_mean_speed[:10]
plt.figure(figsize=(15,9))
plt.plot(daily_mean_speed)
plt.xlabel('Days')
plt.ylabel('Mean Speed per Day')
plt.title('Mean speed per Day of Eric')
plt.show()


#!pip install cartopy
import cartopy.crs as ccrs
import cartopy.feature as cft
proj = ccrs.Mercator()
plt.figure(figsize=(12,12))
ax = plt.axes(projection = proj)
ax.set_extent((-25.0, 20.0, 52.0, 10.0))
#to set the long and lat take the min and max of lat and long and add some degree to max 
#and subtract some degrees to min and give the range by trial and error method.
ax.add_feature(cft.LAND)
ax.add_feature(cft.OCEAN)
ax.add_feature(cft.COASTLINE)
ax.add_feature(cft.BORDERS)
# adding features to show on the map like land, ocean, coastline, borders
for name in df1.bird_name:
    ix = df1.bird_name == name
    x, y = df1.longitude.loc[ix], df1.latitude.loc[ix] 
ax.plot(x,y, '.', transform = ccrs.Geodetic(), label = name )
plt.legend(loc= 'upper left' )
plt.show()