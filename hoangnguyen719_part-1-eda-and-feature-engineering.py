import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib as mpl

import matplotlib.style as style

import seaborn as sns

from itertools import product, combinations

import gc



import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)



from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import cross_val_score



rand_state = 719
data_path = '/kaggle/input/learn-together/'

def reload(x):

    return pd.read_csv(data_path + x, index_col = 'Id')



train = reload('train.csv')

n_train = len(train)

test = reload('test.csv')

n_test = len(test)



all_data = train.iloc[:,train.columns != 'Cover_Type'].append(test)

all_data['train'] = [1]*n_train + [0]*n_test
print('Shape of train and test sets: {0}, {1}'.format(train.shape, test.shape))

print('Number of NaN values in train: {}'.format(train.isna().sum().sum()))

print('Number of NaN values in test: {}'.format(test.isna().sum().sum()))
def data_count(df):

    sns.set_style('whitegrid')

    plt.figure(figsize = (16,8))

    count = df.nunique().sort_values(ascending=False)

    f = sns.barplot(x=count.index, y=count)

    f.set_xticklabels(labels = count.index, rotation=90)

    for i,v in enumerate(count):

        plt.text(x=i-0.2, y=v+len(str(v))*35, s=v, rotation=90)

    plt.ylabel('Count')

    plt.title('Unique data count')

    plt.show()

    

data_count(train.iloc[:,0:-1]) # except target column
numerical = ['Elevation', 'Horizontal_Distance_To_Hydrology',

             'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',

             'Horizontal_Distance_To_Fire_Points',

             'Aspect', 'Slope', 

             'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm']



categorical = ['Soil_Type{}'.format(i) for i in range(1,41)] + ['Wilderness_Area{}'.format(i) for i in range(1,5)]

print('Train data numerical features:')

train[numerical].describe().T
print('Test data numerical features:')

test[numerical].describe().T
print('Correlation matrix of continuous features in the training set:')

corr = train[numerical + ['Cover_Type']].corr()

plt.figure(figsize = (7,7))

sns.heatmap(corr, annot = True, vmin = -1, vmax = 1, annot_kws = {"fontsize":8})

plt.show()
def distplot(df, columns, colors=['red', 'green', 'blue', 'c', 'purple'], bins_num = None, hist = True, kde = False): 

    # df is either dataframe or list of ('name_df',df)

    # col is either string or list

    sns.set_style('whitegrid')

#### CONVERT INPUT DATA'S TYPE

    if type(df) != list: 

        df = [('df',df)] 

    if type(columns) == str: 

        columns = [columns]

    l_col = len(columns)

    l_df = len(df)

###### CALCULATE ROWS AND COLS OF GRAPHS

    c = min([l_col, 3]) # cols

    r = l_col//3 + sum([l_col%3!=0]) # rows

    fig = plt.figure(figsize=(c*7, r*6))

    

    for index in range(l_col):

        column = columns[index]

####### CALCULATE BINS OF HIST

        if bins_num == None: 

            combined_data = np.hstack(tuple([df[x][1][column] for x in range(l_df)])) 

            n_bins = min(50,len(np.unique(combined_data))) # number of bins: <= 50

            bins = np.histogram(combined_data, bins=n_bins)[1] # get "edge" of each bin

        bins = next(b for b in [bins_num, bins] if b is not None)

####### ADD SUBPLOT AND PLOT

        ax = fig.add_subplot(r,c,index+1) 

        for i in range(l_df):

            sns.distplot(df[i][1][column], bins=bins, hist = hist, kde=kde, color=colors[i], 

                         label=df[i][0], norm_hist=True, hist_kws={'alpha':0.4})

        plt.xlabel(column)

        if (l_df>1) & ((index+1) % c == 0): # legend at the graph on the right

            ax.legend()

    plt.tight_layout()

    plt.show() 
distplot([('train',train), ('test',test)], columns = numerical)
# count % of samples that are zero

cols_w0 = []

for col in numerical:

    if min(train[col]) <= 0:

        cols_w0.append(col)

initial_values = [0]*len(cols_w0)

zero_counts = pd.DataFrame(index = cols_w0)

for df, col in product(['train','test', 'all_data'], cols_w0):

    zero_counts.loc[col, '{}_0_count'.format(df)] = eval('len({0}[{0}.{1} == 0])'.format(df,col))

    zero_counts.loc[col, '{}_0_portion'.format(df)] = eval('sum({0}.{1}==0)/len({0}.{1})'.format(df, col))

    zero_counts.loc[col, '1/{}_nunique'.format(df)] = round(eval('{0}.{1}.nunique()'.format(df, col)) ** (-1), 6)

    

zero_counts
questionable_0 = ['Hillshade_9am', 'Hillshade_3pm'] # Hillshade_3pm visualization looks weird

distplot(all_data, questionable_0)
zero_counts.loc[questionable_0,:]
for col in questionable_0:

    all_data_0 = all_data[all_data[col] == 0].copy()

    all_data_non0 = all_data[all_data[col] != 0].copy()

    corr = all_data_non0.corr()[col]

    corr = np.abs(corr[corr.index != col]).sort_values(ascending = False)

    

    sns.set_style('whitegrid')

    plt.figure(figsize = (17,4))

    fig = sns.barplot(x=corr.index, y=corr)

    fig.set_xticklabels(labels = corr.index, rotation=90)

#     for i,v in enumerate(corr):

#         plt.text(x=i-0.2, y=v+len(str(v))*35, s=v, rotation=90)

    plt.ylabel('Correlation')

    plt.title(col)

    plt.show()
corr_cols = {'Hillshade_9am': ['Hillshade_3pm', 'Aspect', 'Slope', 'Soil_Type10', 'Wilderness_Area1',

                   'Wilderness_Area4', 'Vertical_Distance_To_Hydrology'],

             'Hillshade_3pm': ['Hillshade_9am', 'Hillshade_Noon', 'Slope', 'Aspect']

            }
rfr = RandomForestRegressor(n_estimators = 100, random_state = rand_state, verbose = 0, n_jobs = -1)



# for col in questionable_0: 

#     print('='*20)

#     scores = cross_val_score(rfr,

#                              all_data_non0[corr_cols[col]], 

#                              all_data_non0[col],

#                              n_jobs = -1)

#     print(col + ': {0:.4} (+/- {1:.4}) ## [{2}]'.format(scores.mean(), scores.std()*2, ', '.join(map(str, np.round(scores,4)))))



# ===========OUTPUT=====================

# ====================

# Hillshade_9am: 1.0 (+/- 0.00056) ## [0.9995, 0.9993, 0.9988]

# ====================

# Hillshade_3pm: 1.0 (+/- 0.0029) ## [0.9981, 0.9971, 0.9947]



## NEAR PERFECT SCORES FOR ALL => no need further feature engineering for questionable_0 predictions
summary = pd.DataFrame(index = train.describe().index)



for col in questionable_0:

    print('='*20)

    print(col, end='')

    all_data_0 = all_data[all_data[col] == 0].copy()

    all_data_non0 = all_data[all_data[col] != 0].copy()

    rfr.fit(all_data_non0[corr_cols[col]], all_data_non0[col])

    pred = rfr.predict(all_data_0[corr_cols[col]])

    pred_col = 'predicted_{}'.format(col)

    

    all_data[pred_col] = all_data[col].copy()

    all_data.loc[all_data_0.index, pred_col] = pred

    summary[pred_col] = all_data[all_data[col] == 0][[pred_col]].describe()

    print(': finished!')



summary
for col in questionable_0:

    all_data['predicted_{}'.format(col)] = all_data['predicted_{}'.format(col)].apply(int)
def scatterplot(df, x, y, title='', size=None, color = [None, None, None], cm=None, alpha=0.5):

    style.use('fivethirtyeight')

    plt.figure(figsize = (7,5))

    if color[0] != None: # color must be list [color_column, vmin, vmax] with [vmin, vmax] = range of color bar

        color_column = df[color[0]]

        if color[1] == None:

            color[1] = df[color[0]].min()

        if color[2] == None:

            color[2] = df[color[0]].max()

        if cm == None:

            cm = plt.cm.get_cmap('twilight')

    else:

        color_column = color[0]

    if size != None:

        scaler = StandardScaler()

        scaler.fit(train[[size]])

        size = scaler.transform(train[[size]])*200

    plot = plt.scatter(df[x], df[y], s=size, c=color_column, alpha = alpha, 

                       cmap = cm, vmin=color[1], vmax=color[2])

    if color[0] != None:

        bar = plt.colorbar(plot)

        bar.set_label(color[0])

    plt.xlabel(x)

    plt.ylabel(y)

    plt.title(title, fontsize = 18)

    plt.show()
for col in ['Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm']:

    scatterplot(df=train, x='Aspect', y='Slope', color=[col,0,254], cm=plt.cm.get_cmap('plasma'), alpha = 0.5)
train['Hillshade_Mean'] = train[['Hillshade_9am',

                              'Hillshade_Noon',

                              'Hillshade_3pm']].apply(np.mean, axis = 1)

scatterplot(df=train, x='Aspect', y='Slope', 

                color=['Hillshade_Mean',0,254], cm=plt.cm.get_cmap('plasma'), alpha = 0.5)
train['SlopeSin_Elevation'] = np.sin(np.radians(train.Slope)) * train.Elevation

for col in ['Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm']:

    scatterplot(df=train, x='Aspect', y='SlopeSin_Elevation', color=[col,0,254], cm=plt.cm.get_cmap('plasma'), alpha = 0.5)
def aspect_slope(df):

    df['AspectSin'] = np.sin(np.radians(df.Aspect))

    df['AspectCos'] = np.cos(np.radians(df.Aspect))

    df['AspectSin_Slope'] = df.AspectSin * df.Slope

    df['AspectCos_Slope'] = df.AspectCos * df.Slope

    df['AspectSin_Slope_Abs'] = np.abs(df.AspectSin_Slope)

    df['AspectCos_Slope_Abs'] = np.abs(df.AspectCos_Slope)

    df['Hillshade_Mean'] = df[['Hillshade_9am',

                              'Hillshade_Noon',

                              'Hillshade_3pm']].apply(np.mean, axis = 1)

    return df



train = aspect_slope(train)
for col in ['Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 'Hillshade_Mean']:

    scatterplot(df=train, x='AspectSin_Slope', y=col, alpha = 0.5)
for col in ['Aspect', 'Slope','AspectSin_Slope', 'AspectCos_Slope']:

    sns.set(style='whitegrid')

    f = plt.figure(figsize=(8,6))

    ax = sns.boxplot(x='Cover_Type', y=col, data=train)

    plt.show()
def distances(df):

    horizontal = ['Horizontal_Distance_To_Fire_Points', 

                  'Horizontal_Distance_To_Roadways',

                  'Horizontal_Distance_To_Hydrology']

    

    df['Euclidean_to_Hydrology'] = np.sqrt(df['Horizontal_Distance_To_Hydrology']**2 + df['Vertical_Distance_To_Hydrology']**2)

    df['EuclidHydro_Slope'] = df.Euclidean_to_Hydrology * df.Slope

    df['Elevation_VDH_sum'] = df.Elevation + df.Vertical_Distance_To_Hydrology

    df['Elevation_VDH_diff'] = df.Elevation - df.Vertical_Distance_To_Hydrology

    df['Elevation_2'] = df.Elevation**2

    df['Elevation_3'] = df.Elevation**3

    df['Elevation_log1p'] = np.log1p(df.Elevation) # credit: https://www.kaggle.com/evimarp/top-6-roosevelt-national-forest-competition/notebook

    

    for col1, col2 in combinations(zip(horizontal, ['HDFP', 'HDR', 'HDH']), 2):

        df['{0}_{1}_diff'.format(col1[1], col2[1])] = df[col1[0]] - df[col2[0]]

        df['{0}_{1}_sum'.format(col1[1], col2[1])] = df[col1[0]] + df[col2[0]]

    

    df['Horizontal_sum'] = df[horizontal].sum(axis = 1)

    return df



train = distances(train)
distance_cols = ['Elevation', 'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology',

            'Horizontal_Distance_To_Roadways', 'Horizontal_Distance_To_Fire_Points']



for col in ['Elevation', 'Elevation_2', 'Elevation_3']:

    f = plt.figure(figsize=(8,6))

    ax = sns.boxplot(x='Cover_Type', y=col, data=train)

    plt.show()
def OHE_to_cat(df, colname, data_range): # data_range = [min_index, max_index+1]

    df[colname] = sum([i * df[colname + '{}'.format(i)] for i in range(data_range[0], data_range[1])])

    return df



train = OHE_to_cat(train, 'Wilderness_Area',  [1,5])

test = OHE_to_cat(test, 'Wilderness_Area', [1,5])

counts = train.groupby('Cover_Type')['Wilderness_Area'].value_counts().sort_index().unstack(level=1).fillna(0)



plt.figure(figsize=(6,8))

sns.heatmap(counts/100, annot=True)
soils = [

    [7, 15, 8, 14, 16, 17,

     19, 20, 21, 23], #unknow and complex 

    [3, 4, 5, 10, 11, 13],   # rubbly

    [6, 12],    # stony

    [2, 9, 18, 26],      # very stony

    [1, 24, 25, 27, 28, 29, 30,

     31, 32, 33, 34, 36, 37, 38, 

     39, 40, 22, 35], # extremely stony and bouldery

]

soil_dict = {}

for index, soil_group in enumerate(soils):

    for soil in soil_group:

        soil_dict[soil] = index



def rocky(df):

    df['Rocky'] = sum(i * df['Soil_Type' + str(i)] for i in range(1,41))

    df['Rocky'] = df['Rocky'].map(soil_dict)

    return df
# Data setup

all_data = aspect_slope(all_data)

all_data = distances(all_data)

all_data = OHE_to_cat(all_data, 'Wilderness_Area', [1,5])

all_data = OHE_to_cat(all_data, 'Soil_Type', [1,41])

all_data = rocky(all_data)

all_data.drop(['Soil_Type7', 'Soil_Type15', 'train'] + questionable_0, axis = 1, inplace = True)
X_train = all_data.iloc[:n_train,:].copy()

y_train = train.Cover_Type.copy()

X_test = all_data.iloc[n_train:, :].copy()



def mem_reduce(df):

    # credit: https://www.kaggle.com/arateris/2-layer-k-fold-learning-forest-cover

    start_mem = df.memory_usage().sum() / 1024.0**2

    for col in df.columns:

        if df[col].dtype=='float64': 

            df[col] = df[col].astype('float32')

        if df[col].dtype=='int64': 

            if df[col].max()<1: df[col] = df[col].astype(bool)

            elif df[col].max()<128: df[col] = df[col].astype('int8')

            elif df[col].max()<32768: df[col] = df[col].astype('int16')

            else: df[col] = df[col].astype('int32')

    end_mem = df.memory_usage().sum() / 1024**2

    print('Reduce from {0:.3f} MB to {1:.3f} MB (decrease by {2:.2f}%)'.format(start_mem, end_mem, 

                                                                               (start_mem - end_mem)/start_mem*100))

    return df



X_train = mem_reduce(X_train)

print('='*10)

X_test=mem_reduce(X_test)

gc.collect()
rfc = RandomForestClassifier(n_estimators = 719,

                             max_depth = 464,

                             max_features = 0.3,

                             min_samples_split = 2,

                             min_samples_leaf = 1,

                             bootstrap = False,

                             verbose = 0,

                             random_state = rand_state,

                             n_jobs = -1)

# scores = cross_val_score(rfc

#                         , X_train

#                         , y_train

#                         , scoring = 'accuracy'

#                         , cv = 5

#                         , n_jobs = -1

#                         )



# print('scores: {0:.4} (+/- {1:.4}) ## [{2}]'.format(scores.mean(), 

#                                                     scores.std()*2, ', '.join(map(str, np.round(scores,4)))))

# # scores: 0.8097 (+/- 0.06924) ## [0.788, 0.7894, 0.7867, 0.8069, 0.8773]
rfc.fit(X_train, y_train)

# predict = rfc.predict(X_test)



# output = pd.DataFrame({'Id': test.index,

#                       'Cover_Type': predict})

# output.to_csv('Submission.csv', index=False)
importance_threshold = 0.003

# PLOT FEATURES' IMPORTANCES

importances = pd.DataFrame({'Features': X_train.columns, 

                                'Importances': rfc.feature_importances_})

mpl.rcParams.update(mpl.rcParamsDefault)

fig = plt.figure(figsize=(8,14))

sns.barplot(x='Importances', y='Features', data=importances.sort_values(by=['Importances']

                                                                       , axis = 'index'

                                                                       , ascending = False)

           , orient="h")

plt.xticks(rotation='vertical')

plt.show()



# EXTRACT IMPORTANT FEATURES

important_cols = importances[importances.Importances >= importance_threshold].Features

print('Important features:')

print([col for col in important_cols])