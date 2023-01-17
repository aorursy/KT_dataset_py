import pandas as pd

import numpy as np

import scipy as scp



from multiprocessing import Pool

import time



from sklearn.preprocessing import LabelEncoder

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer



from sklearn.decomposition import PCA



from sklearn.ensemble import GradientBoostingRegressor

from sklearn.linear_model import Lasso



import warnings; warnings.simplefilter('ignore')
# path = '../input/web-traffic-time-series-forecasting/'

# path=''

path = '../input/'

kfile = '{}key_1.csv'.format(path)

sfile = '{}sample_submission_1.csv'.format(path)

tfile = '{}train_1.csv'.format(path)
feature_keys = {'Access': {'all-access': 0, 'desktop': 1, 'mobile-web': 2},

                'Agent': {'all-agents': 0, 'spider': 1},

                'Domain': {'mediawiki.org': 0, 'wikimedia.org': 1, 'wikipedia.org': 2},

                'Language': {'commons': 0, 'de': 1, 'en': 2,

                             'es': 3, 'fr': 4,'ja': 5, 'ru': 6,

                             'www': 7,'zh': 8}

               }
dateCols = ['quarter',

            'is_month_start','is_month_end',

            'is_quarter_start','is_quarter_end',

            'is_year_start','is_year_end',

            'dayofweek','month']

pageCols = ['Name','Language', 'Domain', 'Access', 'Agent']



extractDate = lambda col: (np.uint8(col.quarter),

                                      np.uint8(col.is_month_start),

                                      np.uint8(col.is_month_end),

                                      np.uint8(col.is_quarter_start),

                                      np.uint8(col.is_quarter_end),

                                      np.uint8(col.is_year_start),

                                      np.uint8(col.is_year_end),

                                      np.uint8(col.dayofweek),

                                      np.uint8(col.month)

                                      )
def fun(x):

    return page_dict[x]

def fun_d(x):

    return date_dict[x]

    

def parallel_map(ser,n=20,isDate=False):

    """

    looks up the input pandas series values to in dictionary d

    return res a series with the mapped values and the same index as ser

    """

#     res = ser.copy()

    t0 = time.time()

    

    try:

        p = Pool(n)

        if not isDate:

            res = p.map(fun,ser)

        else:

            res = p.map(fun_d,ser)

    except Exception as e:

        print('failed',e)

        p.close()

        

    p.close()

    p=None

    print('Time:',time.time()-t0)

    return res
# parallel_map(pd.Series(pd.date_range('2015-01-01',periods=1000,freq='D').strftime('%Y-%m-%d').tolist()),n=20,isDate=True)
def create_dictionary(isDate=False):

    """

    if isDate=True

    return a dictionary with the unique dates as keys and their features as tuple value

    if isDate=False

    return dictionary with the unique pages as keys and their features as tuple value

    """

    fnames = ['Name','Language','Domain','Access','Agent']

    domain = '([A-Za-z0-9\-]+\.org)'

    language = '([A-Za-z0-9\-]+)'

    access = '([A-Za-z0-9\-]+)'

    agent = '([A-Za-z0-9\-]+)'

    name = '(.+)'

    pattern = '^{:}_{:}\.{:}_{:}_{:}$'.format(name, language,

                                              domain, access,

                                              agent)

    if not isDate:

        keys = pd.read_csv(kfile,

                           usecols=['Page'],

                           converters={0:lambda p:p[:-11]},

                           index_col='Page')

        keys['Page'] = keys.index.tolist()

        keys.drop_duplicates(inplace=True)

        keys[fnames] = keys['Page'].str.extract(pattern)

        keys[fnames[1:]] = keys[fnames[1:]].apply(

            lambda col: col.map(feature_keys[col.name]).astype(np.uint8))

        keys.drop('Page',axis=1,inplace=True)

        keys = dict(zip(keys.index,map(tuple,keys.values)))

        return keys

    else:

        keys = pd.read_csv(kfile,

                           usecols=['Page'],

                           converters={0:lambda p:p[-10:]},

                           index_col='Page')

        keys['Date'] = keys.index.tolist()

        keys.drop_duplicates(inplace=True)

        keys['Date'] = pd.to_datetime(keys['Date']).map(extractDate)

        keys = dict(zip(keys.index,map(tuple,keys['Date'].values)))

        return keys

page_dict = create_dictionary()

date_dict = create_dictionary(True)

print(list(page_dict.items())[:5])

print(list(date_dict.items())[:5])
def load_validation_set(keyfile,samplefile):

    """

    returns the validation set

    typical use: load_validation_set(kfile,sfile)

    ### Optimise for large files

    #### read_csv Parameters

    ###### na_filter : boolean, default True

        Detect missing value markers (empty strings and the value of na_values). In

        data without any NAs, passing na_filter=False can improve the performance

        of reading a large file)

    ###### memory_map : boolean, default False

        If a filepath is provided for `filepath_or_buffer`, map the file object

        directly onto memory and access the data directly from there. Using this

        option can improve performance because there is no longer any I/O overhead.

     ###### engine : {'c', 'python'}, optional

        Parser engine to use. The C engine is faster while the python engine is

        currently more feature-complete.

    """

    keys = pd.read_csv(keyfile,

                   index_col = 'Id',

                   converters = {

                                'Page':lambda p: 

                                 {'Page': p[:-11],'Date':p[-10:],}},     

                   engine = 'c',

                   na_filter = False,

                   memory_map = True)

    keys['Date'] = parallel_map(keys['Page'].apply(lambda d: d['Date']),isDate=True,n=50)

    keys['Page'] = keys['Page'].apply(lambda d: d['Page'])

    

    sample = pd.read_csv(samplefile,

                         index_col='Id',

                         usecols=['Id'],

                         engine = 'c',

                         na_filter = False,

                         memory_map = True)

    df = pd.concat([keys,sample],join_axes=[sample.index],axis=1).to_sparse()

    df['Page'] = parallel_map(df['Page'],n=50)

    return df

vData = load_validation_set(kfile,sfile)

vData.head()   

vData.info()
# vData.to_pickle('vData.csv')
def load_train_set(train_file):

    df = pd.read_csv(train_file, 

                    index_col=0,

                   engine = 'c',

                   memory_map = True).rename(

        columns=pd.to_datetime).groupby(

        extractDate,axis=1).mean().unstack().dropna().astype(int).to_frame().reset_index()

    df['Page'] = parallel_map(df['Page'],n=50)

    return df.rename(columns={0:'Visits','level_0':'Date'}).to_sparse()



tData = load_train_set(tfile)

tData.head()
tData.info()
# tData.to_pickle('tData.csv')
def getXsparse(df,addText=False,test=False):

    """

    """

    cv = TfidfVectorizer()

    if addText:

        X = sp.sparse.hstack([scp.sparse.csr_matrix(df[cols].values),

                      cv.fit_transform(df['Name'].astype('str'))],'csr')

    else:

        X =scp.sparse.csr_matrix(pd.get_dummies(df[cols],columns=cols).values)

    return X
# X_sp = getXsparse(tData)
# X_sp = getXsparse(X)

# X_Validate_sp = getXsparse(X_Validate)

# X=None

# X_Validate = None

# X_sp.shape,X_Validate_sp.shape