import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

%matplotlib inline

plt.style.use('seaborn')



# we will skip 2001 - 2005 due to bad quality



crimes1 = pd.read_csv('../input/Chicago_Crimes_2005_to_2007.csv',error_bad_lines=False)

crimes2 = pd.read_csv('../input/Chicago_Crimes_2008_to_2011.csv',error_bad_lines=False)

crimes3 = pd.read_csv('../input/Chicago_Crimes_2012_to_2017.csv',error_bad_lines=False)

crimes = pd.concat([crimes1, crimes2, crimes3], ignore_index=False, axis=0)



del crimes1

del crimes2

del crimes3



print('Dataset ready..')



print('Dataset Shape before drop_duplicate : ', crimes.shape)

crimes.drop_duplicates(subset=['ID', 'Case Number'], inplace=True)

print('Dataset Shape after drop_duplicate: ', crimes.shape)
crimes.drop(['Unnamed: 0', 'Case Number', 'IUCR','Updated On','Year', 'FBI Code', 'Beat','Ward','Community Area', 'Location'], inplace=True, axis=1)
#Let's have a look at the first 3 records and see if we see what we expect

crimes.head(3)
# convert dates to pandas datetime format

crimes.Date = pd.to_datetime(crimes.Date, format='%m/%d/%Y %I:%M:%S %p')

# setting the index to be the date will help us a lot later on

crimes.index = pd.DatetimeIndex(crimes.Date)
# of records X # of features

crimes.shape
crimes.info()
loc_to_change  = list(crimes['Location Description'].value_counts()[20:].index)

desc_to_change = list(crimes['Description'].value_counts()[20:].index)

#type_to_change = list(crimes['Primary Type'].value_counts()[20:].index)



crimes.loc[crimes['Location Description'].isin(loc_to_change) , crimes.columns=='Location Description'] = 'OTHER'

crimes.loc[crimes['Description'].isin(desc_to_change) , crimes.columns=='Description'] = 'OTHER'

#crimes.loc[crimes['Primary Type'].isin(type_to_change) , crimes.columns=='Primary Type'] = 'OTHER'
# we convert those 3 columns into 'Categorical' types -- works like 'factor' in R

crimes['Primary Type']         = pd.Categorical(crimes['Primary Type'])

crimes['Location Description'] = pd.Categorical(crimes['Location Description'])

crimes['Description']          = pd.Categorical(crimes['Description'])
plt.figure(figsize=(11,5))

crimes.resample('M').size().plot(legend=False)

plt.title('Number of crimes per month (2005 - 2016)')

plt.xlabel('Months')

plt.ylabel('Number of crimes')

plt.show()
plt.figure(figsize=(11,4))

crimes.resample('D').size().rolling(365).sum().plot()

plt.title('Rolling sum of all crimes from 2005 - 2016')

plt.ylabel('Number of crimes')

plt.xlabel('Days')

plt.show()
crimes_count_date = crimes.pivot_table('ID', aggfunc=np.size, columns='Primary Type', index=crimes.index.date, fill_value=0)

crimes_count_date.index = pd.DatetimeIndex(crimes_count_date.index)

plo = crimes_count_date.rolling(365).sum().plot(figsize=(12, 30), subplots=True, layout=(-1, 3), sharex=False, sharey=False)
days = ['Monday','Tuesday','Wednesday',  'Thursday', 'Friday', 'Saturday', 'Sunday']

crimes.groupby([crimes.index.dayofweek]).size().plot(kind='barh')

plt.ylabel('Days of the week')

plt.yticks(np.arange(7), days)

plt.xlabel('Number of crimes')

plt.title('Number of crimes by day of the week')

plt.show()
crimes.groupby([crimes.index.month]).size().plot(kind='barh')

plt.ylabel('Months of the year')

plt.xlabel('Number of crimes')

plt.title('Number of crimes by month of the year')

plt.show()
plt.figure(figsize=(8,10))

crimes.groupby([crimes['Primary Type']]).size().sort_values(ascending=True).plot(kind='barh')

plt.title('Number of crimes by type')

plt.ylabel('Crime Type')

plt.xlabel('Number of crimes')

plt.show()
plt.figure(figsize=(8,10))

crimes.groupby([crimes['Location Description']]).size().sort_values(ascending=True).plot(kind='barh')

plt.title('Number of crimes by Location')

plt.ylabel('Crime Location')

plt.xlabel('Number of crimes')

plt.show()
hour_by_location = crimes.pivot_table(values='ID', index='Location Description', columns=crimes.index.hour, aggfunc=np.size).fillna(0)

hour_by_type     = crimes.pivot_table(values='ID', index='Primary Type', columns=crimes.index.hour, aggfunc=np.size).fillna(0)

hour_by_week     = crimes.pivot_table(values='ID', index=crimes.index.hour, columns=crimes.index.weekday_name, aggfunc=np.size).fillna(0)

hour_by_week     = hour_by_week[days].T # just reorder columns according to the the order of days

dayofweek_by_location = crimes.pivot_table(values='ID', index='Location Description', columns=crimes.index.dayofweek, aggfunc=np.size).fillna(0)

dayofweek_by_type = crimes.pivot_table(values='ID', index='Primary Type', columns=crimes.index.dayofweek, aggfunc=np.size).fillna(0)

location_by_type  = crimes.pivot_table(values='ID', index='Location Description', columns='Primary Type', aggfunc=np.size).fillna(0)
from sklearn.cluster import AgglomerativeClustering as AC



def scale_df(df,axis=0):

    '''

    A utility function to scale numerical values (z-scale) to have a mean of zero

    and a unit variance.

    '''

    return (df - df.mean(axis=axis)) / df.std(axis=axis)



def plot_hmap(df, ix=None, cmap='bwr'):

    '''

    A function to plot heatmaps that show temporal patterns

    '''

    if ix is None:

        ix = np.arange(df.shape[0])

    plt.imshow(df.iloc[ix,:], cmap=cmap)

    plt.colorbar(fraction=0.03)

    plt.yticks(np.arange(df.shape[0]), df.index[ix])

    plt.xticks(np.arange(df.shape[1]))

    plt.grid(False)

    plt.show()

    

def scale_and_plot(df, ix = None):

    '''

    A wrapper function to calculate the scaled values within each row of df and plot_hmap

    '''

    df_marginal_scaled = scale_df(df.T).T

    if ix is None:

        ix = AC(4).fit(df_marginal_scaled).labels_.argsort() # a trick to make better heatmaps

    cap = np.min([np.max(df_marginal_scaled.as_matrix()), np.abs(np.min(df_marginal_scaled.as_matrix()))])

    df_marginal_scaled = np.clip(df_marginal_scaled, -1*cap, cap)

    plot_hmap(df_marginal_scaled, ix=ix)

    

def normalize(df):

    result = df.copy()

    for feature_name in df.columns:

        max_value = df[feature_name].max()

        min_value = df[feature_name].min()

        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)

    return result
plt.figure(figsize=(15,12))

scale_and_plot(hour_by_type)
plt.figure(figsize=(15,7))

scale_and_plot(hour_by_location)
plt.figure(figsize=(12,4))

scale_and_plot(hour_by_week, ix=np.arange(7))
plt.figure(figsize=(17,17))

scale_and_plot(dayofweek_by_type)
plt.figure(figsize=(15,12))

scale_and_plot(dayofweek_by_location)
df = normalize(location_by_type)

ix = AC(3).fit(df.T).labels_.argsort() # a trick to make better heatmaps

plt.figure(figsize=(17,13))

plt.imshow(df.T.iloc[ix,:], cmap='Reds')

plt.colorbar(fraction=0.03)

plt.xticks(np.arange(df.shape[0]), df.index, rotation='vertical')

plt.yticks(np.arange(df.shape[1]), df.columns)

plt.title('Normalized location frequency for each crime')

plt.grid(False)

plt.show()
crimes.iloc[(crimes[['Longitude']].values < -88.0).flatten(), crimes.columns=='Longitude'] = 0.0

crimes.iloc[(crimes[['Longitude']].values > -87.5).flatten(), crimes.columns=='Longitude'] = 0.0

crimes.iloc[(crimes[['Latitude']].values < 41.60).flatten(),  crimes.columns=='Latitude'] = 0.0

crimes.iloc[(crimes[['Latitude']].values > 42.05).flatten(),  crimes.columns=='Latitude'] = 0.0

crimes.replace({'Latitude': 0.0, 'Longitude': 0.0}, np.nan, inplace=True)

crimes.dropna(inplace=True)
import seaborn as sns
crimes_new
crimes_new = crimes[(crimes['Primary Type'] == 'SEX OFFENSE') | (crimes['Primary Type'] == 'HOMICIDE') | (crimes['Primary Type'] == 'ARSON')]

ax = sns.lmplot('Longitude', 'Latitude',

                data= crimes_new[['Longitude','Latitude']],

                fit_reg=False,

                size=4, 

                scatter_kws={'alpha':.1})

ax = sns.kdeplot(crimes_new[['Longitude','Latitude']],

                 cmap="jet", 

                 bw=.005,

                 #n_levels=10,

                 cbar=True, 

                 shade=False, 

                 shade_lowest=False)

ax.set_xlim(-87.9,-87.5)

ax.set_ylim(41.60,42.05)

#ax.set_axis_off()
# crimes_new['by'] = crimes_new.index.month == 7

# g = sns.FacetGrid(crimes_new[['Longitude', 'Latitude', 'by', 'Primary Type']],

#               col='by', row = 'Primary Type')

# g.map(sns.kdeplot,'Longitude', 'Latitude')
ctypes = ['THEFT', 'BATTERY', 'CRIMINAL DAMAGE', 'NARCOTICS', 'ASSAULT', 'BURGLARY', 'DECEPTIVE PRACTICE', 'MOTOR VEHICLE THEFT', 'ROBBERY', 'CRIMINAL TRESPASS', 'WEAPONS VIOLATION', 'PUBLIC PEACE VIOLATION', 'OFFENSE INVOLVING CHILDREN', 'PROSTITUTION', 'CRIM SEXUAL ASSAULT', 'INTERFERENCE WITH PUBLIC OFFICER', 'SEX OFFENSE', 'HOMICIDE', 'ARSON', 'GAMBLING', 'LIQUOR LAW VIOLATION', 'KIDNAPPING', 'STALKING', 'INTIMIDATION']
fig = plt.figure(figsize=(15,35))

for i, crime_type in enumerate(ctypes):

    ax = fig.add_subplot(int(np.ceil(float(len(ctypes)) / 4)), 4, i+1)

    crimes_ = crimes[crimes['Primary Type']==crime_type]

    sns.regplot('Longitude', 'Latitude',

               data= crimes_[['Longitude','Latitude']],

               fit_reg=False,

               scatter_kws={'alpha':.1, 'color':'grey'},

               ax=ax)

    sns.kdeplot(X='Longitude', Y='Latitude',

                data= crimes_[['Longitude','Latitude']],

                cmap="jet", 

                bw=.005,

                #n_levels=10,

                cbar=True, 

                shade=True, 

                shade_lowest=False,

                ax = ax)

    ax.set_title(crime_type)

    ax.set_xlim(-87.9,-87.5)

    ax.set_ylim(41.60,42.05)

    ax.set_axis_off()    

plt.show()
from numpy.linalg import svd





class CA(object):

    """Simple corresondence analysis.

    

    Inputs

    ------

    ct : array_like

      Two-way contingency table. If `ct` is a pandas DataFrame object,

      the index and column values are used for plotting.

    Notes

    -----

    The implementation follows that presented in 'Correspondence

    Analysis in R, with Two- and Three-dimensional Graphics: The ca

    Package,' Journal of Statistical Software, May 2007, Volume 20,

    Issue 3.

    """



    def __init__(self, ct):

        self.rows = ct.index.values if hasattr(ct, 'index') else None

        self.cols = ct.columns.values if hasattr(ct, 'columns') else None

        

        # contingency table

        N = np.matrix(ct, dtype=float)



        # correspondence matrix from contingency table

        P = N / N.sum()



        # row and column marginal totals of P as vectors

        r = P.sum(axis=1)

        c = P.sum(axis=0).T



        # diagonal matrices of row/column sums

        D_r_rsq = np.diag(1. / np.sqrt(r.A1))

        D_c_rsq = np.diag(1. / np.sqrt(c.A1))



        # the matrix of standarized residuals

        S = D_r_rsq * (P - r * c.T) * D_c_rsq



        # compute the SVD

        U, D_a, V = svd(S, full_matrices=False)

        D_a = np.asmatrix(np.diag(D_a))

        V = V.T



        # principal coordinates of rows

        F = D_r_rsq * U * D_a



        # principal coordinates of columns

        G = D_c_rsq * V * D_a



        # standard coordinates of rows

        X = D_r_rsq * U



        # standard coordinates of columns

        Y = D_c_rsq * V



        # the total variance of the data matrix

        inertia = sum([(P[i,j] - r[i,0] * c[j,0])**2 / (r[i,0] * c[j,0])

                       for i in range(N.shape[0])

                       for j in range(N.shape[1])])



        self.F = F.A

        self.G = G.A

        self.X = X.A

        self.Y = Y.A

        self.inertia = inertia

        self.eigenvals = np.diag(D_a)**2



    def plot(self):

        """Plot the first and second dimensions."""

        xmin, xmax = None, None

        ymin, ymax = None, None

        if self.rows is not None:

            for i, t in enumerate(self.rows):

                x, y = self.F[i,0], self.F[i,1]

                plt.text(x, y, t, va='center', ha='center', color='r')

                xmin = min(x, xmin if xmin else x)

                xmax = max(x, xmax if xmax else x)

                ymin = min(y, ymin if ymin else y)

                ymax = max(y, ymax if ymax else y)

        else:

            plt.plot(self.F[:, 0], self.F[:, 1], 'ro')



        if self.cols is not None:

            for i, t in enumerate(self.cols):

                x, y = self.G[i,0], self.G[i,1]

                plt.text(x, y, t, va='center', ha='center', color='b')

                xmin = min(x, xmin if xmin else x)

                xmax = max(x, xmax if xmax else x)

                ymin = min(y, ymin if ymin else y)

                ymax = max(y, ymax if ymax else y)

        else:

            plt.plot(self.G[:, 0], self.G[:, 1], 'bs')



        if xmin and xmax:

            pad = (xmax - xmin) * 0.1

            plt.xlim(xmin - pad, xmax + pad)

        if ymin and ymax:

            pad = (ymax - ymin) * 0.1

            plt.ylim(ymin - pad, ymax + pad)



        plt.grid()

        plt.xlabel('Dim 1')

        plt.ylabel('Dim 2')



    def scree_diagram(self, perc=True, *args, **kwargs):

        """Plot the scree diagram."""

        eigenvals = self.eigenvals

        xs = np.arange(1, eigenvals.size + 1, 1)

        ys = 100. * eigenvals / eigenvals.sum() if perc else eigenvals

        plt.plot(xs, ys, *args, **kwargs)

        plt.xlabel('Dimension')

        plt.ylabel('Eigenvalue' + (' [%]' if perc else ''))
#crime type x district

#crime type x location

#crime type x hour

#crime type x month

#crime type x dayofweek

#crime type x year



ctypexdistrict = crimes.pivot_table(values='ID', index='Primary Type', columns='District', aggfunc=np.size).fillna(0)

ctypexlocation = crimes.pivot_table(values='ID', index='Primary Type', columns='Location Description', aggfunc=np.size).fillna(0)

ctypexhour = crimes.pivot_table(values='ID', index='Primary Type', columns=crimes.index.hour, aggfunc=np.size).fillna(0)

ctypexmonth = crimes.pivot_table(values='ID', index='Primary Type', columns=crimes.index.month, aggfunc=np.size).fillna(0)

ctypexyear = crimes.pivot_table(values='ID', index='Primary Type', columns=crimes.index.year, aggfunc=np.size).fillna(0)

ctypexdayofweek = crimes.pivot_table(values='ID', index='Primary Type', columns=crimes.index.dayofweek, aggfunc=np.size).fillna(0)
ctypes_short = ctypes[:8]

ca = CA(ctypexlocation.loc[ctypes_short])



plt.figure(100)

ca.plot()



# plt.figure(101)

# ca.scree_diagram()

# plt.show()
# days = ['Monday','Tuesday','Wednesday',  'Thursday', 'Friday', 'Saturday', 'Sunday']



# df1 = pd.DataFrame(ca.F[:,0], columns=['value'])

# df1['type'] = ca.rows

# df1.sort_values('value',inplace=True)

# df1.reset_index()





# df2 = pd.DataFrame(ca.G[:,0], columns=['value'])

# df2['type'] = ca.cols

# df2.sort_values('value',inplace=True)

# df2.reset_index()



# fig, ax = plt.subplots(1, 2)



# ax1 = sns.stripplot(y="type", x="value", data=df1, order=df1.type, ax=ax[0])

# ax2 = sns.stripplot(y="type", x="value", orient='h', data=df2,order=df2.type, ax=ax[1])


