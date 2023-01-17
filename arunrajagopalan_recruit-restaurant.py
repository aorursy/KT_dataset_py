import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



plt.rcParams['axes.labelsize'] = 14

plt.rcParams['xtick.labelsize'] = 12

plt.rcParams['ytick.labelsize'] = 12



%matplotlib inline



np.random.seed(42)





from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.utils import check_array

from sklearn.preprocessing import LabelEncoder

from scipy import sparse



# Definition of the CategoricalEncoder class, copied from PR #9151.



from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.utils import check_array

from sklearn.preprocessing import LabelEncoder

from scipy import sparse



class CategoricalEncoder(BaseEstimator, TransformerMixin):

    """Encode categorical features as a numeric array.

    The input to this transformer should be a matrix of integers or strings,

    denoting the values taken on by categorical (discrete) features.

    The features can be encoded using a one-hot aka one-of-K scheme

    (``encoding='onehot'``, the default) or converted to ordinal integers

    (``encoding='ordinal'``).

    This encoding is needed for feeding categorical data to many scikit-learn

    estimators, notably linear models and SVMs with the standard kernels.

    Read more in the :ref:`User Guide <preprocessing_categorical_features>`.

    Parameters

    ----------

    encoding : str, 'onehot', 'onehot-dense' or 'ordinal'

        The type of encoding to use (default is 'onehot'):

        - 'onehot': encode the features using a one-hot aka one-of-K scheme

          (or also called 'dummy' encoding). This creates a binary column for

          each category and returns a sparse matrix.

        - 'onehot-dense': the same as 'onehot' but returns a dense array

          instead of a sparse matrix.

        - 'ordinal': encode the features as ordinal integers. This results in

          a single column of integers (0 to n_categories - 1) per feature.

    categories : 'auto' or a list of lists/arrays of values.

        Categories (unique values) per feature:

        - 'auto' : Determine categories automatically from the training data.

        - list : ``categories[i]`` holds the categories expected in the ith

          column. The passed categories are sorted before encoding the data

          (used categories can be found in the ``categories_`` attribute).

    dtype : number type, default np.float64

        Desired dtype of output.

    handle_unknown : 'error' (default) or 'ignore'

        Whether to raise an error or ignore if a unknown categorical feature is

        present during transform (default is to raise). When this is parameter

        is set to 'ignore' and an unknown category is encountered during

        transform, the resulting one-hot encoded columns for this feature

        will be all zeros.

        Ignoring unknown categories is not supported for

        ``encoding='ordinal'``.

    Attributes

    ----------

    categories_ : list of arrays

        The categories of each feature determined during fitting. When

        categories were specified manually, this holds the sorted categories

        (in order corresponding with output of `transform`).

    Examples

    --------

    Given a dataset with three features and two samples, we let the encoder

    find the maximum value per feature and transform the data to a binary

    one-hot encoding.

    >>> from sklearn.preprocessing import CategoricalEncoder

    >>> enc = CategoricalEncoder(handle_unknown='ignore')

    >>> enc.fit([[0, 0, 3], [1, 1, 0], [0, 2, 1], [1, 0, 2]])

    ... # doctest: +ELLIPSIS

    CategoricalEncoder(categories='auto', dtype=<... 'numpy.float64'>,

              encoding='onehot', handle_unknown='ignore')

    >>> enc.transform([[0, 1, 1], [1, 0, 4]]).toarray()

    array([[ 1.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  0.],

           [ 0.,  1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.]])

    See also

    --------

    sklearn.preprocessing.OneHotEncoder : performs a one-hot encoding of

      integer ordinal features. The ``OneHotEncoder assumes`` that input

      features take on values in the range ``[0, max(feature)]`` instead of

      using the unique values.

    sklearn.feature_extraction.DictVectorizer : performs a one-hot encoding of

      dictionary items (also handles string-valued features).

    sklearn.feature_extraction.FeatureHasher : performs an approximate one-hot

      encoding of dictionary items or strings.

    """



    def __init__(self, encoding='onehot', categories='auto', dtype=np.float64,

                 handle_unknown='error'):

        self.encoding = encoding

        self.categories = categories

        self.dtype = dtype

        self.handle_unknown = handle_unknown



    def fit(self, X, y=None):

        """Fit the CategoricalEncoder to X.

        Parameters

        ----------

        X : array-like, shape [n_samples, n_feature]

            The data to determine the categories of each feature.

        Returns

        -------

        self

        """



        if self.encoding not in ['onehot', 'onehot-dense', 'ordinal']:

            template = ("encoding should be either 'onehot', 'onehot-dense' "

                        "or 'ordinal', got %s")

            raise ValueError(template % self.handle_unknown)



        if self.handle_unknown not in ['error', 'ignore']:

            template = ("handle_unknown should be either 'error' or "

                        "'ignore', got %s")

            raise ValueError(template % self.handle_unknown)



        if self.encoding == 'ordinal' and self.handle_unknown == 'ignore':

            raise ValueError("handle_unknown='ignore' is not supported for"

                             " encoding='ordinal'")



        X = check_array(X, dtype=np.object, accept_sparse='csc', copy=True)

        n_samples, n_features = X.shape



        self._label_encoders_ = [LabelEncoder() for _ in range(n_features)]



        for i in range(n_features):

            le = self._label_encoders_[i]

            Xi = X[:, i]

            if self.categories == 'auto':

                le.fit(Xi)

            else:

                valid_mask = np.in1d(Xi, self.categories[i])

                if not np.all(valid_mask):

                    if self.handle_unknown == 'error':

                        diff = np.unique(Xi[~valid_mask])

                        msg = ("Found unknown categories {0} in column {1}"

                               " during fit".format(diff, i))

                        raise ValueError(msg)

                le.classes_ = np.array(np.sort(self.categories[i]))



        self.categories_ = [le.classes_ for le in self._label_encoders_]



        return self



    def transform(self, X):

        """Transform X using one-hot encoding.

        Parameters

        ----------

        X : array-like, shape [n_samples, n_features]

            The data to encode.

        Returns

        -------

        X_out : sparse matrix or a 2-d array

            Transformed input.

        """

        X = check_array(X, accept_sparse='csc', dtype=np.object, copy=True)

        n_samples, n_features = X.shape

        X_int = np.zeros_like(X, dtype=np.int)

        X_mask = np.ones_like(X, dtype=np.bool)



        for i in range(n_features):

            valid_mask = np.in1d(X[:, i], self.categories_[i])



            if not np.all(valid_mask):

                if self.handle_unknown == 'error':

                    diff = np.unique(X[~valid_mask, i])

                    msg = ("Found unknown categories {0} in column {1}"

                           " during transform".format(diff, i))

                    raise ValueError(msg)

                else:

                    # Set the problematic rows to an acceptable value and

                    # continue `The rows are marked `X_mask` and will be

                    # removed later.

                    X_mask[:, i] = valid_mask

                    X[:, i][~valid_mask] = self.categories_[i][0]

            X_int[:, i] = self._label_encoders_[i].transform(X[:, i])



        if self.encoding == 'ordinal':

            return X_int.astype(self.dtype, copy=False)



        mask = X_mask.ravel()

        n_values = [cats.shape[0] for cats in self.categories_]

        n_values = np.array([0] + n_values)

        indices = np.cumsum(n_values)



        column_indices = (X_int + indices[:-1]).ravel()[mask]

        row_indices = np.repeat(np.arange(n_samples, dtype=np.int32),

                                n_features)[mask]

        data = np.ones(n_samples * n_features)[mask]



        out = sparse.csc_matrix((data, (row_indices, column_indices)),

                                shape=(n_samples, indices[-1]),

                                dtype=self.dtype).tocsr()

        if self.encoding == 'onehot-dense':

            return out.toarray()

        else:

            return out
data = {

    'air_reserve': pd.read_csv('../input/air_reserve.csv'),

    'air_store_info': pd.read_csv('../input/air_store_info.csv'),

    'air_visit_data': pd.read_csv('../input/air_visit_data.csv'),

    'date_info': pd.read_csv('../input/date_info.csv'),

    'hpg_reserve': pd.read_csv('../input/hpg_reserve.csv'),

    'hpg_store_info': pd.read_csv('../input/hpg_store_info.csv'),

    'store_id_relation': pd.read_csv('../input/store_id_relation.csv'),

    'sample_submission': pd.read_csv('../input/sample_submission.csv'),

}
data['air_reserve'].info()
data['air_reserve']['visit_datetime'] = pd.to_datetime(data['air_reserve']['visit_datetime']) #Converting date object

data['air_reserve']['reserve_datetime'] = pd.to_datetime(data['air_reserve']['reserve_datetime']) #Converting date object

data['air_reserve']['air_store_id'].nunique() #Unique stores in air reserve data
data['air_store_info'].info()
data['air_store_info']['air_store_id'].nunique()
data['air_visit_data'].info()
data['air_visit_data']['air_store_id'].nunique()
data['date_info'].info()
data['date_info']['calendar_date'] = pd.to_datetime(data['date_info']['calendar_date'])

#data['date_info'].describe()

#data['date_info']['calendar_date'].max()
data['hpg_reserve'].info()
data['hpg_reserve']['visit_datetime'] = pd.to_datetime(data['hpg_reserve']['visit_datetime']) #Converting date object

data['hpg_reserve']['reserve_datetime'] = pd.to_datetime(data['hpg_reserve']['reserve_datetime']) #Converting date object

data['hpg_reserve']['hpg_store_id'].nunique() #Unique stores in hpg reserve data
data['hpg_store_info'].info()
data['hpg_store_info']['hpg_store_id'].nunique() #Unique stores in hpg store info
data['store_id_relation'].info()
data['sample_submission'].info()
data['submission_prep'] = data['sample_submission'].copy()

data['submission_prep']['visit_date'] = data['submission_prep']['id'].map(lambda x: str(x).split('_')[2])

data['submission_prep']['visit_date'] = pd.to_datetime(data['submission_prep']['visit_date'])

data['submission_prep']['air_store_id'] = data['submission_prep']['id'].map(lambda x: '_'.join(str(x).split('_')[:2]))
data['submission_prep']['air_store_id'].nunique()
data['air_hpg_store_info'] = pd.merge(data['store_id_relation'], data['hpg_store_info'], how="inner")
data['hpg_air_reserve'] =pd.merge(data['store_id_relation'], data['hpg_reserve'], how="inner")
data['air_reserve_prep'] = data['air_reserve'].copy()

data['air_reserve_prep']['visit_date'] = data['air_reserve_prep']['visit_datetime'].dt.date

data['air_reserve_prep']['reserve_date'] = data['air_reserve_prep']['reserve_datetime'].dt.date

data['air_reserve_prep'].drop(['visit_datetime', 'reserve_datetime'], axis=1, inplace=True)
data['air_reserve_prep'] = data['air_reserve_prep'].groupby(['air_store_id', 'visit_date', 'reserve_date']).sum()

data['air_reserve_prep'] = data['air_reserve_prep'].reset_index()
data['hpg_air_reserve_prep'] = data['hpg_air_reserve'].copy()

data['hpg_air_reserve_prep']['visit_date'] = data['hpg_air_reserve_prep']['visit_datetime'].dt.date

data['hpg_air_reserve_prep']['reserve_date'] = data['hpg_air_reserve_prep']['reserve_datetime'].dt.date

data['hpg_air_reserve_prep'].drop(['visit_datetime', 'reserve_datetime', 'hpg_store_id'], axis=1, inplace=True)

data['hpg_air_reserve_prep'] = data['hpg_air_reserve_prep'].groupby(['air_store_id', 'visit_date', 'reserve_date']).sum()

data['hpg_air_reserve_prep'] = data['hpg_air_reserve_prep'].reset_index()
data['air_reserve_final'] = pd.concat([data['air_reserve_prep'], data['hpg_air_reserve_prep']], axis=0) 

data['air_reserve_final'] = data['air_reserve_final'].groupby(['air_store_id', 'visit_date', 'reserve_date']).sum()

data['air_reserve_final'] = data['air_reserve_final'].reset_index()
data['air_reserve_final']['visit_date'] = pd.to_datetime(data['air_reserve_final']['visit_date'])

data['air_reserve_final']['day_n_of_week'] = data['air_reserve_final']['visit_date'].dt.dayofweek

data['air_reserve_final']['day'] = data['air_reserve_final']['visit_date'].dt.day

data['air_visit_data']['visit_date'] = pd.to_datetime(data['air_visit_data']['visit_date'])
data['visit_reserve_final'] = data['air_reserve_final'].drop('reserve_date', axis=1)

data['visit_reserve_final'] = data['visit_reserve_final'].groupby(['air_store_id', 'visit_date']).sum()

data['visit_reserve_final'] = data['visit_reserve_final'].reset_index()

data['visit_final'] = pd.merge(data['visit_reserve_final'], data['air_visit_data'], how="right")

data['visit_final'].shape
data['air_store_info']['air_genre_name'] = data['air_store_info']['air_genre_name'].map(lambda x: str(str(x).replace('/',' ')))

data['air_store_info']['air_area_name'] = data['air_store_info']['air_area_name'].map(lambda x: str(str(x).replace('-',' ')))
data['final_data'] = pd.merge(data['air_store_info'], data['visit_final'])

data['final_data'] = pd.merge(data['final_data'], data['date_info'], how = "left", 

                              right_on ='calendar_date', left_on='visit_date') #Merge with date information



#Id Change

air_store_id_reshaped = data['final_data']['air_store_id'].values.reshape(-1, 1)

cat_encoder = CategoricalEncoder(encoding="ordinal")

data['final_data']['store_id'] = cat_encoder.fit_transform(air_store_id_reshaped)



#Area name split

data['final_data']['region_0'] = data['final_data']['air_area_name'].map(lambda x: str(x).split(' ')[0])

data['final_data']['region_1'] = data['final_data']['air_area_name'].map(lambda x: str(x).split(' ')[1])

data['final_data']['region_2'] = data['final_data']['air_area_name'].map(lambda x: str(x).split(' ')[2])

data['final_data']['region_3'] = data['final_data']['air_area_name'].map(lambda x: str(x).split(' ')[3])



#Genre name split

data['final_data']['genre_0'] = data['final_data']['air_genre_name'].map(lambda x: str(x).split(' ')[0])

data['final_data']['genre_1'] = data['final_data']['air_genre_name'].map(lambda x: str(x).split(' ')[1] 

                                                                         if len(str(x).split(' ')) > 1 else "")



#data['final_data'].shape

data['final_data']['lat_long'] = data['final_data']['longitude'] + data['final_data']['latitude']



#Adding more fields for visit date, day, month and week



data['final_data']['month'] = data['final_data']['visit_date'].dt.month

data['final_data']['week'] = data['final_data']['visit_date'].dt.week

data['final_data']['day_n_of_week'] =  data['final_data']['visit_date'].dt.dayofweek + 1

data['final_data']['year'] =  data['final_data']['visit_date'].dt.year

data['final_data']['day'] =  data['final_data']['visit_date'].dt.day 



#mean reserve

data['mean_reserve'] = data['visit_reserve_final'].groupby(['air_store_id'])[['reserve_visitors']].mean().reset_index()

data['mean_reserve'].columns = ['air_store_id', 'mean_reserve_visitors']

data['final_data'] = pd.merge(data['final_data'] , data['mean_reserve'], how = "left", right_on="air_store_id", left_on="air_store_id")



data['mean_week_reserve'] = data['visit_reserve_final'].groupby(['air_store_id', 'day_n_of_week']).agg({'reserve_visitors' : [np.min,np.mean,np.median,np.max]}).reset_index()

data['mean_week_reserve'].columns = ['air_store_id', 'day_n_of_week', 'min_wk_res_visitors', 'mean_wk_res_visitors', 'median_wk_res_visitors',

                                      'max_wk_res_visitors']

data['final_data'] = pd.merge(data['final_data'] , data['mean_week_reserve'], how = "left")



data['mean_day_reserve'] = data['visit_reserve_final'].groupby(['air_store_id', 'day']).agg({'reserve_visitors' : [np.min,np.mean,np.median,np.max]}).reset_index()

data['mean_day_reserve'].columns = ['air_store_id', 'day', 'min_day_res_visitors', 'mean_day_res_visitors', 'median_day_res_visitors',

                                      'max_day_res_visitors']

data['final_data'] = pd.merge(data['final_data'] , data['mean_day_reserve'], how = "left")



#mean_visitors

data['mean_visitors'] = data['final_data'].groupby(['air_store_id'])[['visitors']].mean().reset_index()

data['mean_visitors'].columns = ['air_store_id', 'mean_visitors']

data['final_data'] = pd.merge(data['final_data'] , data['mean_visitors'], how = "left", right_on="air_store_id", left_on="air_store_id")



data['mean_week_visitors'] = data['final_data'].groupby(['air_store_id', 'day_n_of_week']).agg({'visitors' : [np.min,np.mean,np.median,np.max,np.size]}).reset_index()

data['mean_week_visitors'].columns = ['air_store_id', 'day_n_of_week', 'min_wk_visitors', 'mean_wk_visitors', 'median_wk_visitors',

                                      'max_wk_visitors','count_wk_observations']

data['final_data'] = pd.merge(data['final_data'] , data['mean_week_visitors'], how = "left")



data['mean_day_visitors'] = data['final_data'].groupby(['air_store_id', 'day']).agg({'visitors' : [np.min,np.mean,np.median,np.max,np.size]}).reset_index()

data['mean_day_visitors'].columns = ['air_store_id', 'day', 'min_day_visitors', 'mean_day_visitors', 'median_day_visitors',

                                      'max_day_visitors','count_day_observations']

data['final_data'] = pd.merge(data['final_data'] , data['mean_day_visitors'], how = "left")



#Adding 0 on missing values

data['final_data'] = data['final_data'].fillna(0)
data['test_data'] = pd.merge(data['visit_reserve_final'], data['submission_prep'], how="right")

data['test_data'] = pd.merge(data['test_data'], data['air_store_info'])

data['test_data'] = pd.merge(data['test_data'], data['date_info'], how = "left", 

                              right_on ='calendar_date', left_on='visit_date') #Merge with date information

#data['test_data'].shape



#Id Change

air_store_id_reshaped = data['test_data']['air_store_id'].values.reshape(-1, 1)

cat_encoder = CategoricalEncoder(encoding="ordinal")

data['test_data']['store_id'] = cat_encoder.fit_transform(air_store_id_reshaped)



#Area name split

data['test_data']['region_0'] = data['test_data']['air_area_name'].map(lambda x: str(x).split(' ')[0])

data['test_data']['region_1'] = data['test_data']['air_area_name'].map(lambda x: str(x).split(' ')[1])

data['test_data']['region_2'] = data['test_data']['air_area_name'].map(lambda x: str(x).split(' ')[2])

data['test_data']['region_3'] = data['test_data']['air_area_name'].map(lambda x: str(x).split(' ')[3])





#Genre name split

data['test_data']['genre_0'] = data['test_data']['air_genre_name'].map(lambda x: str(x).split(' ')[0])

data['test_data']['genre_1'] = data['test_data']['air_genre_name'].map(lambda x: str(x).split(' ')[1] 

                                                                         if len(str(x).split(' ')) > 1 else "")



data['test_data']['lat_long'] = data['test_data']['longitude'] + data['test_data']['latitude']





#Adding more fields for visit date, day, month and week



data['test_data']['month'] = data['test_data']['visit_date'].dt.month

data['test_data']['week'] = data['test_data']['visit_date'].dt.week

data['test_data']['day_n_of_week'] =  data['test_data']['visit_date'].dt.dayofweek + 1

data['test_data']['year'] =  data['test_data']['visit_date'].dt.year

data['test_data']['day'] =  data['test_data']['visit_date'].dt.day 



#mean reserve



data['test_data'] = pd.merge(data['test_data'] , data['mean_reserve'], how = "left"

                             , right_on="air_store_id", left_on="air_store_id")

data['test_data'] = pd.merge(data['test_data'] , data['mean_week_reserve'], how = "left")

data['test_data'] = pd.merge(data['test_data'] , data['mean_day_reserve'], how = "left")



#mean_visitors

data['test_data'] = pd.merge(data['test_data'] , data['mean_visitors'], how = "left"

                             , right_on="air_store_id", left_on="air_store_id")

data['test_data'] = pd.merge(data['test_data'] , data['mean_week_visitors'], how = "left")

data['test_data'] = pd.merge(data['test_data'] , data['mean_day_visitors'], how = "left")



#Adding 0 on missing values

data['test_data'] = data['test_data'].fillna(0)

data['test_data'] = data['test_data'].sort_values(by=['id'])
#Visitors based on week days

#fig, ax = plt.subplots(figsize=(10,10))

sns.barplot(x="day_n_of_week", y="visitors", data=data['final_data'])
#Visitors each day

f, ax = plt.subplots(1, 1, figsize=(15, 8))

plt1 = data['final_data'].groupby(['visit_date'], as_index=False).agg({'visitors': np.sum})

plt2 = data['final_data'].groupby(['visit_date'], as_index=False).agg({'reserve_visitors': np.sum})

plt1 = plt1.set_index('visit_date')

plt2 = plt2.set_index('visit_date')

plt1.plot(color='c', kind='area', ax=ax)

plt2.plot(color='r', kind='line', ax=ax)

plt.ylabel("Sum of Visitors")

plt.title("Visitor and Reservations")
#Visitors by Genre



plt.style.use('seaborn')

color = sns.color_palette()



f,ax=plt.subplots(1,1, figsize=(10,8))

genre=data['final_data'].groupby(['air_genre_name'],as_index=False)['visitors'].sum()

genre.sort_values(by='visitors', ascending=True, inplace=True)

genre['air_genre'] =[i for i,x in enumerate(genre['air_genre_name'])] 

genre = genre.sort_values(by='visitors', ascending=False)#.reset_index()

my_range = genre['air_genre']

plt.hlines(y=my_range, xmin=0, xmax=genre['visitors'], color='goldenrod',alpha=0.8) #[‘solid’ | ‘dashed’ | ‘dashdot’ | ‘dotted’]

plt.plot(genre['visitors'], my_range, "o",markersize=25,label='visitors',color='orangered')



# Add titles and axis names

plt.yticks(my_range, genre['air_genre_name'],fontsize=15)

plt.title("Total visitors by Air Genre", loc='center')

plt.xlabel('Score')

plt.ylabel('Features')
#Visitors by Region



plt.style.use('seaborn')

color = sns.color_palette()



f,ax=plt.subplots(1,1, figsize=(10,8))

genre=data['final_data'].groupby(['region_0'],as_index=False)['visitors'].sum()

genre.sort_values(by='visitors', ascending=True, inplace=True)

genre['region'] =[i for i,x in enumerate(genre['region_0'])] 

genre = genre.sort_values(by='visitors', ascending=False)#.reset_index()

my_range = genre['region']

plt.hlines(y=my_range, xmin=0, xmax=genre['visitors'], color='goldenrod',alpha=0.8) #[‘solid’ | ‘dashed’ | ‘dashdot’ | ‘dotted’]

plt.plot(genre['visitors'], my_range, "o",markersize=25,label='visitors',color='orangered')



# Add titles and axis names

plt.yticks(my_range, genre['region_0'],fontsize=15)

plt.title("Region 0", loc='center')

plt.xlabel('Score')

plt.ylabel('Features')
#Visitors by Region



plt.style.use('seaborn')

color = sns.color_palette()



f,ax=plt.subplots(1,1, figsize=(10,8))

genre=data['final_data'].groupby(['region_1'],as_index=False)['visitors'].sum()

genre.sort_values(by='visitors', ascending=True, inplace=True)

genre['region'] =[i for i,x in enumerate(genre['region_1'])] 

genre = genre.sort_values(by='visitors', ascending=False)#.reset_index()

my_range = genre['region']

plt.hlines(y=my_range, xmin=0, xmax=genre['visitors'], color='goldenrod',alpha=0.8) #[‘solid’ | ‘dashed’ | ‘dashdot’ | ‘dotted’]

plt.plot(genre['visitors'], my_range, "o",markersize=25,label='visitors',color='orangered')



# Add titles and axis names

plt.yticks(my_range, genre['region_1'],fontsize=15)

plt.title("Region 1", loc='center')

plt.xlabel('Score')

plt.ylabel('Features')
#Visitors by Region



plt.style.use('seaborn')

color = sns.color_palette()



f,ax=plt.subplots(1,1, figsize=(15,15))

genre=data['final_data'].groupby(['region_2'],as_index=False)['visitors'].sum()

genre.sort_values(by='visitors', ascending=True, inplace=True)

genre['region'] =[i for i,x in enumerate(genre['region_2'])] 

genre = genre.sort_values(by='visitors', ascending=False)#.reset_index()

my_range = genre['region']

plt.hlines(y=my_range, xmin=0, xmax=genre['visitors'], color='goldenrod',alpha=0.8) #[‘solid’ | ‘dashed’ | ‘dashdot’ | ‘dotted’]

plt.plot(genre['visitors'], my_range, "o",markersize=25,label='visitors',color='orangered')



# Add titles and axis names

plt.yticks(my_range, genre['region_2'],fontsize=15)

plt.title("Region 2", loc='center')

plt.xlabel('Score')

plt.ylabel('Features')
plt1=data['final_data']['visitors'].value_counts().reset_index().sort_index()

fig, ax = plt.subplots(figsize=(15, 6), nrows=1, ncols=2, sharex=False, sharey=False)

ax[0].bar(plt1['index'] ,plt1['visitors'],color='limegreen')

ax[1]= sns.boxplot(y='visitors',x='day_n_of_week', data=data['final_data'],hue='holiday_flg',palette="Set2")

ax[1].set_title('Number of daily visitors by day of the week')

ax[0].bar(plt1['index'] ,plt1['visitors'],color='limegreen')

ax[0].set_title('Frequency')

ax[0].set_xlim(0,100)

ax[1].set_ylim(0,80)

ax[1].legend(loc=1)
from sklearn.base import BaseEstimator, TransformerMixin



# column index

latitude_ix, longitude_ix = 3, 4



class CombinedAttributesAdder(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):

        return self  # nothing else to do

    def transform(self, X, y=None):

        lat_plus_long = X[:, latitude_ix] + X[:, longitude_ix]

        return np.c_[X, lat_plus_long]

    

from sklearn.base import BaseEstimator, TransformerMixin



# Create a class to select numerical or categorical columns 

# since Scikit-Learn doesn't handle DataFrames yet

class DataFrameSelector(BaseEstimator, TransformerMixin):

    def __init__(self, attribute_names):

        self.attribute_names = attribute_names

    def fit(self, X, y=None):

        return self

    def transform(self, X):

        return X[self.attribute_names].values
data['train_data'] = data['final_data'].drop('visitors', axis=1)

y_train = np.log1p(data['final_data']['visitors'].copy())



#num_attribs = list(data['train_data'].select_dtypes(include=[np.number]))

num_attribs =  [#'latitude',

                 #'longitude',

                 'store_id',

                 'reserve_visitors',

                 'day_n_of_week',

                 #'day',

                 'holiday_flg',

                 #'lat_long',

                 'month',

                 #'week',

                 'year',

                 #'mean_reserve_visitors',

                 'min_wk_res_visitors',

                 'mean_wk_res_visitors',

                 'median_wk_res_visitors',

                 'max_wk_res_visitors',

                 #'min_day_res_visitors',

                 #'mean_day_res_visitors',

                 #'median_day_res_visitors',

                 #'max_day_res_visitors',

                 #'mean_visitors',

                 'min_wk_visitors',

                 'mean_wk_visitors',

                 'median_wk_visitors',

                 'max_wk_visitors',

                 'count_wk_observations',

                # 'min_day_visitors',

                # 'mean_day_visitors',

                # 'median_day_visitors',

                # 'max_day_visitors',

                # 'count_day_observations'

                ]

#["air_store_id",

ord_cat_attribs =  ["air_store_id"]

                   #"genre_0",

                   #"genre_1",

                   #"region_0",

                   #"region_1",

                   #"region_2",

                   #"region_3"]



onehot_cat_attribs =  ["genre_0",

                   "genre_1",

                   "region_0",

                   "region_1",

                   "region_2",

                   "region_3"]



from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler, Imputer



ord_cat_pipeline = Pipeline([

        ('selector', DataFrameSelector(ord_cat_attribs)),

        ('ord_encoder', CategoricalEncoder(encoding="ordinal")),

        ('std_scaler', StandardScaler()),

    ])



onehot_cat_pipeline = Pipeline([

        ('selector', DataFrameSelector(onehot_cat_attribs)),

        ('onehot_encoder', CategoricalEncoder(encoding="onehot-dense")),

        #('std_scaler', StandardScaler()),

    ])

num_pipeline = Pipeline([

        ('selector', DataFrameSelector(num_attribs)),

        ('imputer', Imputer(strategy="mean")),

        #('attribs_adder', CombinedAttributesAdder()),

        ('std_scaler', StandardScaler()),

    ])
from sklearn.pipeline import FeatureUnion



full_pipeline = FeatureUnion(transformer_list=[

        #("ord_cat_pipeline", ord_cat_pipeline),

        ("onehot_cat_pipeline", onehot_cat_pipeline),

        ("num_pipeline", num_pipeline),

    ])



X_train = full_pipeline.fit_transform(data["train_data"])



from sklearn.pipeline import FeatureUnion



full_pipeline = FeatureUnion(transformer_list=[

        #("ord_cat_pipeline", ord_cat_pipeline),

        ("onehot_cat_pipeline", onehot_cat_pipeline),

        ("num_pipeline", num_pipeline),

    ])



X_test = full_pipeline.fit_transform(data["test_data"])
from sklearn.linear_model import LinearRegression



lin_reg = LinearRegression()

lin_reg.fit(X_train, y_train)
from sklearn.metrics import mean_squared_error



recruit_predictions = lin_reg.predict(X_train)

lin_mse = mean_squared_error(y_train, recruit_predictions)

lin_rmse = np.sqrt(lin_mse)

lin_rmse
Y_test_pred = np.expm1(lin_reg.predict(X_test))

test_submission = pd.DataFrame({"id": data["test_data"]["id"], "visitors": Y_test_pred})

#test_submission.head()

test_submission.to_csv("test_submission.csv", index=False)
test_submission.head()
from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.svm import SVR

from sklearn.linear_model import SGDRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.tree import DecisionTreeRegressor



regressors = [

    LinearRegression(),

    RandomForestRegressor(random_state=42),

    GradientBoostingRegressor(learning_rate=0.2, random_state=42),

    KNeighborsRegressor(n_neighbors=4, n_jobs=-1),

    #SGDRegressor(penalty=None, eta0=0.1),

    DecisionTreeRegressor(max_depth= 10, random_state=42)

]



log_cols = ['Regressor', 'rmse']

log = pd.DataFrame(columns=log_cols)



rmse_dict = {}



for reg in regressors:

    name = reg.__class__.__name__

    reg.fit(X_train, y_train)

    predictions = reg.predict(X_train)

    mse = mean_squared_error(y_train, predictions)

    rmse = np.sqrt(mse)

    print(rmse)

    if name in rmse_dict:

        rmse_dict[name] += rmse

    else:

         rmse_dict[name] = rmse



for reg in rmse_dict:

    log_entry = pd.DataFrame([[reg, rmse_dict[reg]]], columns=log_cols)

    log = log.append(log_entry)

    

plt.xlabel('Root Mean Square Error')

plt.title('RMSE')



#sns.set_color_codes("muted")

sns.barplot(x='rmse', y='Regressor', data=log, color="lightgreen")
def display_scores(scores):

    print("Scores:", scores)

    print("Mean:", scores.mean())

    print("Standard deviation:", scores.std())
from sklearn.model_selection import cross_val_score



lin_scores = cross_val_score(lin_reg, X_train, y_train,

                             scoring="neg_mean_squared_error", cv=5)

lin_rmse_scores = np.sqrt(-lin_scores)

display_scores(lin_rmse_scores)
from sklearn.ensemble import RandomForestRegressor



forest_reg = RandomForestRegressor(random_state=42)

forest_reg.fit(X_train, y_train)



forest_predictions = forest_reg.predict(X_train)

forest_mse = mean_squared_error(y_train, forest_predictions)

forest_rmse = np.sqrt(forest_mse)

forest_rmse
Y_forest_test_pred = np.expm1(forest_reg.predict(X_test))

test_forest_submission = pd.DataFrame({"id": data["test_data"]["id"], 

                                       "visitors": Y_forest_test_pred})

#test_submission.head()

test_forest_submission.to_csv("test_forest_submission.csv", index=False)
test_forest_submission.head()
#from sklearn.model_selection import cross_val_score



'''forest_scores = cross_val_score(forest_reg, X_train, y_train,

                             scoring="neg_mean_squared_error", cv=5)

forest_rmse_scores = np.sqrt(-forest_scores)

display_scores(forest_rmse_scores)'''
from sklearn.ensemble import GradientBoostingRegressor



gbr_reg = GradientBoostingRegressor(learning_rate=0.2, random_state=42)

gbr_reg.fit(X_train, y_train)



gbr_predictions = gbr_reg.predict(X_train)

gbr_mse = mean_squared_error(y_train, gbr_predictions)

gbr_rmse = np.sqrt(gbr_mse)

gbr_rmse
Y_gbr_test_pred = np.expm1(gbr_reg.predict(X_test))#.clip(lower=0.)

test_gbr_submission = pd.DataFrame({"id": data["test_data"]["id"], 

                                       "visitors": Y_gbr_test_pred})

#test_submission.head()

test_gbr_submission.to_csv("test_gbr_submission.csv", index=False)
'''from sklearn.model_selection import RandomizedSearchCV

from sklearn.ensemble import GradientBoostingRegressor

from scipy.stats import randint



param_distribs = {

        'n_estimators': randint(low=20, high=200),

        'max_features': randint(low=10, high=23),

        'max_depth': randint(low=5, high=20),

        'learning_rate': [0.1, 0.2, 0.3, 0.4]

    }



gbr_reg = GradientBoostingRegressor(random_state=42, subsample=0.8)

gbr_search = RandomizedSearchCV(gbr_reg, param_distributions=param_distribs,

                                n_iter=5, cv=5, scoring='neg_mean_squared_error', random_state=42, verbose=10, n_jobs=-1)

gbr_search.fit(X_train, y_train)'''
'''gbr_search.best_score_

gbr_rmse = np.sqrt(-gbr_search.best_score_)

gbr_rmse'''
'''gbr_best_model = gbr_search.best_estimator_



Y_gbr_test_pred = np.expm1(gbr_best_model.predict(X_test))

test_gbr_submission = pd.DataFrame({"id": data["test_data"]["id"], 

                                       "visitors": Y_gbr_test_pred})

#test_submission.head()

test_gbr_submission.to_csv("test_gbr_submission.csv", index=False)'''
'''gbr_scores = cross_val_score(gbr_reg, X_train, y_train,

                             scoring="neg_mean_squared_error", cv=5)

gbr_rmse_scores = np.sqrt(-gbr_scores)

display_scores(gbr_rmse_scores)'''
from sklearn.linear_model import ElasticNet



elastic_net = ElasticNet(alpha=0.01, l1_ratio=0.01, random_state=42)

elastic_net.fit(X_train, y_train)

elastic_net_predictions = elastic_net.predict(X_train)

elastic_net_mse = mean_squared_error(y_train, elastic_net_predictions)

elastic_net_rmse = np.sqrt(elastic_net_mse)

elastic_net_rmse
'''eln_scores = cross_val_score(elastic_net, X_train, y_train,

                             scoring="neg_mean_squared_error", cv=5)

eln_rmse_scores = np.sqrt(-eln_scores)

display_scores(eln_rmse_scores)'''
Y_eln_test_pred = np.expm1(elastic_net.predict(X_test))

test_eln_submission = pd.DataFrame({"id": data["test_data"]["id"], 

                                       "visitors": Y_eln_test_pred})

#test_submission.head()

test_eln_submission.to_csv("test_eln_submission.csv", index=False)
knn_reg = KNeighborsRegressor(n_neighbors=4, n_jobs=-1)

knn_reg.fit(X_train, y_train)



knn_predictions = knn_reg.predict(X_train)

knn_mse = mean_squared_error(y_train, knn_predictions)

knn_rmse = np.sqrt(knn_mse)

knn_rmse
knn_reg = KNeighborsRegressor(n_neighbors=4, n_jobs=-1)

knn_reg.fit(X_train, y_train)



'''knn_scores = cross_val_score(knn_reg, X_train, y_train,

                             scoring="neg_mean_squared_error", cv=5)

knn_rmse_scores = np.sqrt(-knn_scores)

display_scores(knn_rmse_scores)'''
Y_knn_test_pred = np.expm1(knn_reg.predict(X_test))

test_knn_submission = pd.DataFrame({"id": data["test_data"]["id"], 

                                       "visitors": Y_knn_test_pred})

#test_submission.head                                             

test_knn_submission.to_csv("test_knn_submission.csv", index=False)
from sklearn.model_selection import RandomizedSearchCV

from sklearn.tree import DecisionTreeRegressor

from scipy.stats import randint





param_distribs = {

        'max_depth': randint(low=4, high=100),

        'max_features': randint(low=1, high=23)

    }



dec_reg = DecisionTreeRegressor(criterion='mse', min_samples_split=4,random_state=42, presort=False)

dec_search = RandomizedSearchCV(dec_reg, param_distributions=param_distribs,

                                n_iter=10, cv=5, scoring='neg_mean_squared_error', random_state=42, verbose=3)

dec_search.fit(X_train, y_train)
dec_search.best_score_

rmse = np.sqrt(-dec_search.best_score_)

rmse
dec_search.best_params_
dec_best_model = dec_search.best_estimator_



Y_dec_test_pred = np.expm1(dec_best_model.predict(X_test))

test_dec_submission = pd.DataFrame({"id": data["test_data"]["id"], 

                                       "visitors": Y_dec_test_pred})

#test_submission.head()

test_dec_submission.to_csv("test_dec_submission.csv", index=False)
avg_visit = data['final_data'].groupby(['air_store_id', 'day_n_of_week'])[['visitors']].mean().reset_index()

dummy = data['submission_prep'].copy()

dummy.drop('visitors', axis=1, inplace=True)

dummy['day_n_of_week'] = dummy['visit_date'].dt.dayofweek + 1

avg_visitors = pd.merge(dummy, avg_visit, how="left")

avg_visitors = avg_visitors[['id', 'visitors']]

avg_visitors = avg_visitors.fillna(1)
data['test_data']['visitors'] = (dec_best_model.predict(X_test) + np.array(np.log1p(avg_visitors['visitors']))) / 2

data['test_data']['visitors'] = np.expm1(data['test_data']['visitors']).clip(lower=0)

recruit_predictions = data['test_data'][['id', 'visitors']]

recruit_predictions.to_csv("recruit_predictions.csv", index=False)