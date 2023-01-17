%load_ext autoreload

%autoreload 2

%matplotlib inline
from fastai.imports import *

#from fastai.structured import *

from pandas_summary import DataFrameSummary

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from IPython.display import display

from sklearn import metrics

from fastai.tabular import *

from fastai.collab import *

from IPython.display import HTML

from sklearn.preprocessing import LabelEncoder
PATH = "../input"

!ls {PATH}

path = '../working'

!ls {path}
df_raw = pd.read_csv(f'{PATH}/X_train.csv', low_memory=False)

y_raw = pd.read_csv(f'{PATH}/y_train.csv', low_memory=False)

df_test = pd.read_csv(f'{PATH}/X_test.csv', low_memory=False)

sub = pd.read_csv(f'{PATH}/sample_submission.csv')
df_raw.shape,df_test.shape,y_raw.shape
df_raw[df_raw.isnull().any(axis=1)]
y_raw[y_raw.isnull().any(axis=1)]
df_test[df_test.isnull().any(axis=1)]
sub.head()
le = LabelEncoder()

le.fit(y_raw['surface'])

y_raw['surface'] = le.transform(y_raw['surface'])
#FE from https://www.kaggle.com/vanshjatana/help-humanity-by-helping-robots-4e306b

# https://stackoverflow.com/questions/53033620/how-to-convert-euler-angles-to-quaternions-and-get-the-same-euler-angles-back-fr?rq=1

def quaternion_to_euler(x, y, z, w):

    import math

    t0 = +2.0 * (w * x + y * z)

    t1 = +1.0 - 2.0 * (x * x + y * y)

    X = math.atan2(t0, t1)



    t2 = +2.0 * (w * y - z * x)

    t2 = +1.0 if t2 > +1.0 else t2

    t2 = -1.0 if t2 < -1.0 else t2

    Y = math.asin(t2)



    t3 = +2.0 * (w * z + x * y)

    t4 = +1.0 - 2.0 * (y * y + z * z)

    Z = math.atan2(t3, t4)



    return X, Y, Z



def fe(df):

    df['total_angular_velocity'] = (df['angular_velocity_X'] ** 2 + df['angular_velocity_Y'] ** 2 + df['angular_velocity_Z'] ** 2) ** 0.5

    df['total_linear_acceleration'] = (df['linear_acceleration_X'] ** 2 + df['linear_acceleration_Y'] ** 2 + df['linear_acceleration_Z'] ** 2) ** 0.5

    df['total_xyz'] = (df['orientation_X']**2 + df['orientation_Y']**2 +df['orientation_Z'])**0.5

    

    df['acc_vs_vel'] = df['total_linear_acceleration'] / df['total_angular_velocity']

    

    

    temp_df = pd.DataFrame()

    for col in df.columns[3:]:

        temp_df[col + '_mean'] = df.groupby(['series_id'])[col].mean()

        temp_df[col + '_median'] = df.groupby(['series_id'])[col].median()

        temp_df[col + '_max'] = df.groupby(['series_id'])[col].max()

        temp_df[col + '_min'] = df.groupby(['series_id'])[col].min()

        temp_df[col + '_std'] = df.groupby(['series_id'])[col].std()

        temp_df[col + '_range'] = temp_df[col + '_max'] - temp_df[col + '_min']

        temp_df[col + '_maxtoMin'] = temp_df[col + '_max'] / temp_df[col + '_min']

        temp_df[col + '_mean_abs_chg'] = df.groupby(['series_id'])[col].apply(lambda x: np.mean(np.abs(np.diff(x))))

        temp_df[col + '_abs_min'] = df.groupby(['series_id'])[col].apply(lambda x: np.min(np.abs(x)))

        temp_df[col + '_abs_max'] = df.groupby(['series_id'])[col].apply(lambda x: np.max(np.abs(x)))

        temp_df[col + '_abs_avg'] = (temp_df[col + '_abs_min'] + temp_df[col + '_abs_max'])/2

    return temp_df
%%time

df_train = fe(df_raw)

df_test = fe(df_test)
df_train.shape, df_test.shape,y_raw.shape
df_train.head()
y_raw.head()
df_train['surface'] = y_raw['surface']
df_test.head()
#df_raw , test1 = df_raw.drop(['measurement_number','row_id'],axis=1), df_test.drop(['measurement_number','row_id'],axis=1)
def display_all(df):

    with pd.option_context("display.max_rows", 1000): 

        with pd.option_context("display.max_columns", 1000): 

            display(df)
def create_download_link(title = "Download CSV file", filename = "data.csv"):  

    html = '<a href={filename}>{title}</a>'

    html = html.format(title=title,filename=filename)

    return HTML(html)
#display_all(df_test.describe(include='all'))
# from sklearn.metrics import confusion_matrix

# cm = confusion_matrix(df_train.surface, pred)

# cm
df, y = df_train.drop('surface',axis=1), df_train.surface
# def split_vals(a,n): return a[:n].copy(), a[n:].copy()



# n_valid = 800  

# n_trn = len(df)-n_valid

# raw_train, raw_valid = split_vals(df_raw, n_trn)

# X_train, X_valid = split_vals(df, n_trn)

# y_train, y_valid = split_vals(y, n_trn)



# X_train.shape, y_train.shape, X_valid.shape
# train_series_id, valid_series_id = X_train.series_id, X_valid.series_id
# X_train, X_valid = X_train.drop(['series_id'],axis=1), X_valid.drop(['series_id'],axis=1)
# X_test = df_test.drop(['series_id'],axis=1)
# X_train.shape, y_train.shape, X_valid.shape, y_valid.shape,X_test.shape
#def rmse(x,y): return math.sqrt(((x-y)**2).mean())



def print_score(m):

    res = [ m.score(X_train, y_train), m.score(X_valid, y_valid)]

    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)

    print(res)
# train_seriesid = df_train['series_id']

# df_train = df_train.drop(['series_id'],axis=1)
# test_seriesid = df_test['series_id']

# df_test = df_test.drop(['series_id'],axis=1)
# df_train.shape,df_test.shape
dep_var = 'surface'

cont_vars = list(df_test.columns)

procs = [FillMissing]
data = (TabularList.from_df(df_train, path=path, cont_names=cont_vars, procs=procs,)

                .split_by_idx(list(range(3801,3810)))

                .label_from_df(cols=dep_var)

                .add_test(TabularList.from_df(df_test, path=path, cont_names=cont_vars))

                .databunch())
data.show_batch(rows=5)

learn = tabular_learner(data, layers=[1000,500], ps=[0.005,0.05], emb_drop=0.08, 

                         metrics=accuracy)
learn.model
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(5, 2e-3, wd=0.06)
learn.recorder.plot_losses()
learn.fit_one_cycle(6, 2e-3,wd=0.02)
learn.unfreeze()
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(10, 1e-7,wd=0.02)
test_preds = learn.get_preds(DatasetType.Test)[1]

submit1 = pd.concat([test_seriesid,pd.Series(test_preds)],axis=1)

submit1.columns =['series_id','surface']

#submit1['surface'].map(map)
learn.get_preds(DatasetType.Test)[0]
preds, _ = learn.get_preds(ds_type=DatasetType.Test)

labelled_preds = le.inverse_transform(np.array([np.argmax(preds[i]) for i in range(len(preds))]))

sub['surface'] = labelled_preds
sub.head()
sub.to_csv('submission.csv',index=False)

create_download_link(filename='submission.csv')