%load_ext autoreload
%autoreload 2
%matplotlib inline
from functools import partial

import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from matplotlib import pyplot as plt
from sklearn.pipeline import make_pipeline
plt.rcParams['figure.figsize'] = (13,4)
sns.set(
    style='whitegrid',
    color_codes=True,
    font_scale=1.5)
np.set_printoptions(
    suppress=True,
    linewidth=200)
pd.set_option(
    'display.max_rows', 1000,
    'display.max_columns', None,
)
SEED = 0
SEED_LIST = 2 ** np.array([2, 3, 5, 7, 11, 13, 17, 19, 23, 29])
VAL_SIZE = 0.3
train_csv       = '../input/titanic/train.csv'
test_csv        = '../input/titanic/test.csv'
submit_csv      = '../input/titanic/gender_submission.csv'
leaderboard_csv = '../input/titanic-public-leaderboard/titanic-publicleaderboard.csv'
from sklearn.model_selection import RepeatedStratifiedKFold

def cv(params, n=100, n_cv=5, k=5):
    cv_results = xgb.cv(
        params,
        dfull,
        num_boost_round=n,
        folds=RepeatedStratifiedKFold(n_splits=k, n_repeats=n_cv, random_state=SEED),
        seed=SEED,
    )
    plot_cv(cv_results)
    return cv_results

def holdout(params, n=100, early_stopping_rounds=None):
    evals = {}
    m = xgb.train(
        params,
        dtrain,
        num_boost_round=n,
        evals=[(dtrain, 'train'), (dval, 'val')],
        evals_result=evals,
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=None,
    )
    plot_evals(evals)
    return evals

def train(params, n):
    return xgb.train(
        params,
        dfull,
        num_boost_round=n,
        verbose_eval=None,
    )
def roll(ls, w=5):
    return pd.Series(ls).rolling(window=w).mean()

def plot(a, b, c, d):
    plt.subplot(1, 2, 1)
    plt.plot(a), plt.plot(b)
    plt.ylim(0, 0.7)

    plt.subplot(1, 2, 2)
    plt.plot(c), plt.plot(d)
    plt.ylim(0, 0.2)

def plot_cv(cv_dict, start=0, stop=None):
    keys = [
        'train-logloss-mean',
        'test-logloss-mean',
        'train-error-mean',
        'test-error-mean'
    ]
    plot(*[roll(cv_dict[k][start:stop]) for k in keys])

def plot_evals(evals, start=0, stop=None):
    eval_list = [
        roll(evals[a][b][start:stop])
        for b in ['logloss', 'error']
        for a in ['train', 'val']
    ]
    plot(*eval_list)

def plot_cv_error(cv_results, start=0, stop=None):
    plt.plot(cv_results[['train-error-mean', 'test-error-mean']][start:stop])

def plot_holdout_error(h, start=0, stop=None):
    plt.plot(
        pd.DataFrame(
            [h['train']['error'], h['val']['error']],
            index=['train', 'val'])
        .T
        [start:stop]
    )
def ensemble(params, n):
    def d(x): return dict(params, seed=x)
    return (
        np.vstack(train(d(x), n).predict(dtest) for x in SEED_LIST)
        .T
        .mean(axis=1)
    )

def submit(y_hat, name):
    df = pd.read_csv(submit_csv).assign(Survived=y_hat)
    timestamp = datetime.datetime.now().strftime('%d-%m-%Y_%H-%M')
    path = f'./{timestamp}_{name}.csv'
    df.to_csv(path, index=False)

def threshold(y_hat, pr=0.5):
    return (y_hat > pr) * 1
import datetime

def dtype_info(X):
    return pd.concat([
        X.dtypes.rename('dtypes'),
        traintest.min().astype('object').rename('min'),
        traintest.max().astype('object').rename('max'),],
        axis=1
    )

def find(col, s, df):
    if isinstance(s, str):
        pass
    else:
        s = '|'.join([f'{x}' for x in s])
    return df[(
        df
        [col]
        .str.lower()
        .str.contains(s)
    )]

def na(X):
    count = X.isna().sum()
    if len(X.shape) < 2:
        return count
    else:
        return count[lambda x: x > 0]

def perc(x):
    return np.round(x * 100, 2)

def vc(df):
    return df.value_counts(dropna=False).sort_index()
import math
from typing import Union

Numeric = Union[int, float, np.number]

def seq(
        start: Numeric,
        stop: Numeric,
        step: Numeric = None) \
        -> np.ndarray:
    """Inclusive sequence."""

    if step is None:
        if start < stop:
            step = 1
        else:
            step = -1

    if is_int(start) and is_int(step):
        dtype = 'int'
    else:
        dtype = None

    d = max(n_dec(step), n_dec(start))
    n_step = math.floor(round(round(stop - start, d + 1) / step, d + 1)) + 1
    delta = np.arange(n_step) * step
    return np.round(start + delta, decimals=d).astype(dtype)

def is_int(
        x: Numeric) \
        -> bool:
    """Whether `x` is int."""
    return isinstance(x, (int, np.integer))

def n_dec(
        x: Numeric) \
        -> int:
    """No of decimal places, using `str` conversion."""
    if x == 0:
        return 0
    _, _, dec = str(x).partition('.')
    return len(dec)
def bin_interp(X, bins, interp=None):
    """Interpolate bin values."""

    idx = X.apply(lambda x: bin_val(x, bins))

    if interp == 'median':
        v = X.groupby(idx).median()
    elif interp == 'mean':
        v = X.groupby(idx).mean()
    elif interp == 'min':
        v = X.groupby(idx).min()
    elif interp == 'max':
        v = X.groupby(idx).max()
    else:
        return seq(0, len(bins))

    v = list(v)
    bin_vals = [v[0]] + v + [v[-1]]

    return bin_vals

def bin_val(x, bins, vals=None):
    """Map `x` to bin value."""

    if vals is None:
        vals = seq(0, len(bins))

    assert len(vals) == len(bins) + 1, 'len(vals) must equal len(bins) + 1'

    if np.isnan(x):
        return np.nan
    elif x < bins[0]:
        index = 0
    elif x == bins[0]:
        index = 1
    elif x == bins[-1]:
        index = -2
    elif x > bins[-1]:
        index = -1
    else:
        index = np.searchsorted(bins, x, side='right')

    return vals[index]

def count(col, traintest):
    """Map value counts."""

    def f(x):
        if pd.notna(x) and x in vc.index:
            return vc.loc[x]
        else:
            return np.nan

    vc = traintest.value_counts()

    return (
        col
        .apply(lambda x: f(x))
        .rename(traintest.name + '_count')
    )

def eq_attr(one, attr, *rest):
    return all(all(getattr(one, attr) == getattr(x, attr)) for x in rest)

def match(X, col, with_df):
    """Yes/no inner join."""

    return (
        X[col]
        .isin(with_df[col])
        .astype(np.uint8)
        .rename(with_df.index.name)
    )

def reorder(df, order=None):
    """Sort `df` columns by dtype and name."""

    def sort(df):
        return df.dtypes.reset_index().sort_values([0, 'index'])['index']
    if order is None:
        order = [np.floating, np.integer, 'category', 'object']
    names = [sort(df.select_dtypes(s)) for s in order]
    return df[[x for ls in names for x in ls]]
from sklearn.model_selection import train_test_split

def load(csv):
    ycol = 'target'

    col_names = {
        'Survived': ycol,
        'Pclass': 'ticket_class',
        'Name': 'name',
        'Sex': 'sex',
        'Age': 'age',
        'SibSp': 'n_sib_sp',
        'Parch': 'n_par_ch',
        'Ticket': 'ticket',
        'Fare': 'fare',
        'Cabin': 'cabin',
        'Embarked': 'port',
    }

    exclude = [
        'PassengerId'
    ]

    dtype = {
        'Pclass': np.uint8,
        'Age': np.float32,
        'SibSp': np.uint8,
        'Parch': np.uint8,
        'Fare': np.float32,
    }

    df = reorder(
        pd.read_csv(
            csv,
            dtype=dtype,
            usecols=lambda x: x not in exclude,
        )
        .rename(columns=col_names)
    )

    if ycol in df.columns:
        return df.drop(columns=ycol), df[ycol]
    else:
        return df

def load_titanic():
    X, y = load(train_csv)
    test = load(test_csv)
    traintest = pd.concat([X, test])
    return X, y, test, traintest

def preprocess(pip):
    full_X, full_y, todo_test, todo_traintest = load_titanic()

    todo_X, todo_val_X, y, val_y \
        = train_test_split(
            full_X,
            full_y,
            test_size=VAL_SIZE,
            stratify=full_y,
            random_state=SEED
        )

    tr_y = full_y
    tr_X = pip.fit_transform(full_X, full_y)
    traintest = pip.transform(todo_traintest)

    X = pip.fit_transform(todo_X, y)
    val_X = pip.transform(todo_val_X)
    test = pip.transform(todo_test)

    return (
        reorder(X), y,
        reorder(val_X), val_y,
        reorder(tr_X), tr_y,
        reorder(test), reorder(traintest)
    )
from sklearn.base import TransformerMixin


class Apply(TransformerMixin):
    def __init__(self, fn):
        self.fn = fn

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.apply(self.fn)


class AsType(TransformerMixin):
    def __init__(self, t):
        self.t = t

    def fit(self, X, y=None):
        if self.t == 'category':
            self.dtype = pd.Categorical(X.unique())
        else:
            self.dtype = self.t
        return self

    def transform(self, X):
        return X.astype(self.dtype)


class ColMap(TransformerMixin):
    def __init__(self, trf):
        self.trf = trf

    def fit(self, X, y=None):
        self.trf_list = [self.trf().fit(col) for _, col in X.iteritems()]
        return self
    
    def transform(self, X):
        cols = [t.transform(X.iloc[:, i]) for i, t in enumerate(self.trf_list)]
        return pd.concat(cols, axis=1)


class ColProduct(TransformerMixin):
    def __init__(self, trf):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.product(axis=1)


class ColQuot(TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.iloc[:, 0] / X.iloc[:, 1]


class ColSum(TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.sum(axis=1)


class Cut(TransformerMixin):
    def __init__(self, bins, interp=None):
        self.bins = bins
        self.interp = interp

    def fit(self, X, y=None):
        self.name = X.name
        self.vals = bin_interp(X, self.bins, self.interp)
        return self

    def transform(self, X):
        n = len(self.vals) - 2
        return (
            X
            .apply(lambda x: bin_val(x, self.bins, self.vals))
            .rename(f'{self.name}_cut{n}')
        )


class DataFrameUnion(TransformerMixin):
    def __init__(self, trf_list):
        self.trf_list = trf_list

    def fit(self, X, y=None):
        for t in self.trf_list:
            t.fit(X, y)
        return self

    def transform(self, X):
        return pd.concat([t.transform(X) for t in self.trf_list], axis=1)


class FillNA(TransformerMixin):
    def __init__(self, val):
        self.val = val

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.fillna(self.val)


class GetDummies(TransformerMixin):
    def __init__(self, drop_first=False):
        self.drop = drop_first

    def fit(self, X, y=None):
        self.name = X.name
        self.cat = pd.Categorical(X.unique())
        return self

    def transform(self, X):
        return pd.get_dummies(X.astype(self.cat), prefix=self.name, drop_first=self.drop)


class Identity(TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X


class Map(TransformerMixin):
    def __init__(self, d):
        self.d = d

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.map(self.d)


class MeanEncode(TransformerMixin):
    def __init__(self, y):
        self.y = y

    def fit(self, X, y=None):
        m = self.y.groupby(X).mean()
        keys = m.sort_values().index.values
        vals = m.index.values
        self.encode = {k: v for (k, v) in zip(keys, vals)}
        return self

    def transform(self, X):
        return X.replace(self.encode)


class NADummies(TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.isna().astype(np.uint8).rename(X.name, + '_na')


class PdFunction(TransformerMixin):
    def __init__(self, fn):
        self.fn = fn

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return self.fn(X)


class QCut(TransformerMixin):
    def __init__(self, q, interp=None):
        self.q = q
        self.interp = interp

    def fit(self, X, y=None):
        _, self.bins = pd.qcut(X, self.q, retbins=True)
        self.bin_vals = bin_interp(X, self.bins, self.interp)
        return self

    def transform(self, X):
        return (
            X
            .apply(lambda x: bin_val(x, self.bins, self.bin_vals))
            .rename(f'{X.name}_qcut{self.q}')
        )


class Rename(TransformerMixin):
    def __init__(self, name):
        self.name = name

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.rename(self.name)


class SelectColumns(TransformerMixin):
    def __init__(self, include=None, exclude=None):
        self.include = include
        self.exclude = exclude

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.include:
            X = X[self.include]
        if self.exclude:
            return X.drop(columns=self.exclude)
        return X


class SelectDtypes(TransformerMixin):
    def __init__(self, include=None, exclude=None):
        self.include = include
        self.exclude = exclude

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.select_dtypes(include=self.include, exclude=self.exclude)


class StandardScaler(TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        self.mean = X.mean()
        self.std = X.std(ddof=0)
        return self

    def transform(self, X):
        return (X - self.mean) / self.std
def read_leaderboard():
    return (
        pd
        .read_csv(leaderboard_csv)
        .groupby('TeamId')
        .Score.max()
    )

def leaderboard_info():
    df = read_leaderboard()

    n = len(df)
    m = len(pd.read_csv(leaderboard_csv))
    print(f'{n} Teams, {m} submissions')

    mean = perc(df.mean())
    print(f'Mean: {mean}')

    std = perc(df.std())
    print(f'Stdev: {std}')

def leaderboard_percentiles(p=None):
    df = read_leaderboard()

    if p is None:
        p = seq(90, 10, step=-10)

    return pd.DataFrame({
        'Percentile': p,
        'Score': perc(np.percentile(df, p)),
    })

def plot_leaderboard(x=None):
    df = read_leaderboard()
    
    if x is None:
        x = seq(0, 100, step=0.1)
    y = np.percentile(df, q=x)

    plt.title('Leaderboard')
    plt.ylabel('Score (% Accuracy)')
    plt.xlabel('Percentile (%)')
    
    plt.plot(x, y*100)
leaderboard_info()
leaderboard_percentiles()
plot_leaderboard()
def n_fam(X):
    return (
        (X.n_sib_sp + X.n_par_ch)
        .astype(np.uint8)
        .rename('n_fam')
    )

def n_fam_2(X):
    return (
        ((X.n_sib_sp+1) * (X.n_par_ch+1))
        .astype(np.uint8)
        .rename('n_fam_2')
    )
def cabin_encode_v(X):
    return (
        (deck_encode(X) + cabin_no_encode(X) / 10)
        .astype(np.float32)
        .rename('cabin_encode_v')
    )

def cabin_encode_h(X):
    return (
        (cabin_no_encode(X) + deck_encode(X) / 10)
        .astype(np.float32)
        .rename('cabin_encode_h')
    )

def cabin_no(X):
    return (
        X
        .cabin
        .str.extract(r'(\d+)', expand=False)
        .astype(np.float32)
        .rename('cabin_no')
    )

def cabin_no_encode(X):
    def encode(x):
        if x.deck == 'T':
            return 2
        elif np.isnan(x.cabin_no):
            return np.nan
        elif x.deck == 'A':
            if x.cabin_no >= 35:
                return 4
            else:
                return 2
        elif x.deck == 'B':
            if x.cabin_no >= 51:
                return 3
            else:
                return 2
        elif x.deck == 'C':
            if x.cabin_no % 2 == 0:
                if 92 <= x.cabin_no <= 102 or 142 <= x.cabin_no <= 148:
                    return 4
                elif 62 <= x.cabin_no <= 90 or 104 <= x.cabin_no <= 140:
                    return 3
                else:
                    return 2
            else:
                if 85 <= x.cabin_no <= 93 or 123 <= x.cabin_no <= 127:
                    return 4
                elif 55 <= x.cabin_no <= 83 or 95 <= x.cabin_no <= 121:
                    return 3
                else:
                    return 2
        elif x.deck == 'D':
            if x.cabin_no >= 51:
                return 5
            else:
                return 2
        elif x.deck == 'E':
            if x.cabin_no >= 91:
                return 5
            elif x.cabin_no >= 70:
                return 4
            elif x.cabin_no >= 26:
                return 3
            else:
                return 2
        elif x.deck == 'F':
            if x.cabin_no >= 46:
                return 1
            elif x.cabin_no >= 20:
                return 5
            else:
                return 4
        elif x.deck == 'G':
            return 5
    
    df = pd.concat([X.cabin, deck(X), cabin_no(X)], axis=1)
    return (
        df
        .apply(encode, axis=1)
        .astype(np.float32)
        .rename('cabin_no_encode')
    )

def deck(X):
    return (
        X
        .cabin
        .str.extract(r'([A-Z])', expand=False)
        .rename('deck')
    )

def deck_encode(X):
    return (
        deck(X)
        .map({
            'T': 8,
            'A': 7,
            'B': 6,
            'C': 5,
            'D': 4,
            'E': 3,
            'F': 2,
            'G': 1,
        })
        .astype(np.float32)
        .rename('deck_encode')
    )

def starboard(X):
    return (
        (np.round(cabin_no(X)) % 2 == 0)
        .astype(np.uint8)
        .rename('starboard')
    )
def surname(X):
    return (
        X
        .name
        .str.lower()
        .str.extract(r'([a-z]+),', expand=False)
    )

def title(X):
    return (
        X
        .name
        .str.lower()
        .str.extract(r', (\w+)', expand=False)
        .rename('title')
    )

def title_fill(X):
    def rare(row):
        if row.title in ['miss', 'mrs', 'master', 'mr']:
            return row.title
        elif row.title in d:
            return d[row.title]
        elif row.sex == 'male':
            return 'mr'
        elif row.sex == 'female':
            return 'mrs'
        else:
            raise ValueError('row.sex is missing / not in [`male`, `female`]')

    miss = ['ms', 'mlle']
    mrs = ['mme', 'dona', 'lady', 'the']
    mr = [
        'capt',
        'col',
        'don',
        'jonkheer',
        'major',
        'rev',
        'sir',
    ]

    d = {
        **{k: 'mr' for k in mr},
        **{k: 'mrs' for k in mrs},
        **{k: 'miss' for k in miss}
    }

    return (
        X
        .assign(title=title)
        .apply(rare, axis=1)
        .rename('title')
    )
def sex(X):
    return (
        X
        .sex
        .map({'female': 1, 'male': 0})
        .astype(np.uint8)
    )
def ticket_count(X):
    _, _, _, traintest = load_titanic()
    return count(X.ticket, traintest.ticket).astype(np.uint8)
def age_mask(X, tc, sx):
    nm = f'age_tc{tc}_sex{sx}'
    return (X.age * (X.ticket_class == tc) * (sex(X) == sx)).rename(nm)

def fare_quot(X):
    return (
        (X.fare / ticket_count(X))
        .astype(np.float32)
        .rename('fare_quot')
    )

def tc_sex(X, tc, sx):
    return (
        ((X.ticket_class == tc) & (sex(X) == sx))
        .astype(np.uint8)
        .rename(f'tc{tc}_sex{sx}')
    )

def tk_fn(X, col, fn='mean'):
    _, _, _, traintest = load_titanic()
    vc = getattr(traintest[col].groupby(traintest.ticket), fn)()
    return (
        X
        .ticket
        .apply(lambda x: vc.loc[x])
        .astype(np.float32)
        .rename(f'tk_{col}_{fn}')
    )

def tk_sex(X):
    _, _, _, traintest = load_titanic()
    vc = sex(traintest).groupby(traintest.ticket).mean()
    return (
        X
        .ticket
        .apply(lambda x: vc.loc[x])
        .astype(np.float32)
        .rename('tk_sex')
    )
def surv(X):
    def encode(x):
        a = x.tk_surv_max
        b = x.sn_surv_max
        if a == 1 and b == 1:
            return 4
        elif a == 1 or b == 1:
            if a == 0 or b == 0:
                return 2
            else:
                return 3
        elif a == 0 or b == 0:
            if a == 0 and b == 0:
                return 0
            else:
                return 1
        else:
            return np.nan
    return (
        pd.concat([tk_surv(X), sn_surv(X)], axis=1)
        .apply(encode, axis=1)
        .astype(np.float32)
        .rename('surv')
    )

def sn_surv(X, fn='max'):
    tr, y, te, _ = load_titanic()
    v = getattr(y.groupby(surname(tr)), fn)()[lambda x: x.index.isin(surname(te))]
    return (
        surname(X)
        .map(v)
        .astype(np.float32)
        .rename(f'sn_surv_{fn}')
    )

def tk_surv(X, fn='max'):
    tr, y, te, _ = load_titanic()
    v = getattr(y.groupby(tr.ticket), fn)()[lambda x: x.index.isin(te.ticket)]
    return (
        X
        .ticket
        .map(v)
        .astype(np.float32)
        .rename(f'tk_surv_{fn}')
    )
X_pipeline = DataFrameUnion([
    # age
    SelectColumns('age'),

    # fare
    SelectColumns('fare'),

    # n_par_ch + n_sib_sp
    SelectColumns('n_par_ch'),
    SelectColumns('n_sib_sp'),
    PdFunction(n_fam),
    PdFunction(n_fam_2),

    # ticket_class
    make_pipeline(
        SelectColumns('ticket_class'),
        GetDummies(),
    ),

    # cabin
    PdFunction(cabin_encode_v),
    PdFunction(cabin_encode_h),
    PdFunction(cabin_no_encode),
    PdFunction(deck_encode),

    # name -> title -> dummies
    make_pipeline(
        PdFunction(title_fill),
        GetDummies(),
    ),

    # port -> 1/2/3
    make_pipeline(
        SelectColumns('port'),
        Map({'S': 1, 'Q': 2, 'C': 3}),
        AsType(np.float32)
    ),

    # sex -> 0/1
    PdFunction(sex),

    # ticket -> count
    PdFunction(ticket_count),

    #
    # interaction #
    
    # fare / ticket_count -> fare_quot
    PdFunction(fare_quot),

    # age by sex/ticket_class
    PdFunction(partial(age_mask, tc=1, sx=1)),
    PdFunction(partial(age_mask, tc=2, sx=1)),
    PdFunction(partial(age_mask, tc=3, sx=1)),
    PdFunction(partial(age_mask, tc=1, sx=0)),
    PdFunction(partial(age_mask, tc=2, sx=0)),
    PdFunction(partial(age_mask, tc=3, sx=0)),

    # 0/1 by sex/ticket_class
    PdFunction(partial(tc_sex, tc=1, sx=1)),
    PdFunction(partial(tc_sex, tc=2, sx=1)),
    PdFunction(partial(tc_sex, tc=3, sx=1)),
    PdFunction(partial(tc_sex, tc=1, sx=0)),
    PdFunction(partial(tc_sex, tc=2, sx=0)),
    PdFunction(partial(tc_sex, tc=3, sx=0)),

    # ticket grouping
    PdFunction(surv),
    PdFunction(tk_sex),
    PdFunction(partial(tk_fn, col='age')),
    PdFunction(partial(tk_fn, col='n_par_ch')),
    PdFunction(partial(tk_fn, col='n_sib_sp')),
])
X, y, val_X, val_y, tr_X, tr_y, test, traintest = preprocess(X_pipeline)
X.shape
dtype_info(X)
eq_attr(X, 'columns', val_X, tr_X, test, traintest) \
    and eq_attr(X, 'dtypes', val_X, tr_X, test, traintest)
dtrain = xgb.DMatrix(X, y)
dval = xgb.DMatrix(val_X, val_y)
dfull = xgb.DMatrix(tr_X, tr_y)
dtest = xgb.DMatrix(test)
_params = {
    'eta': 0.1,
    'gamma': 0,
    'max_depth': 3,
    'min_child_weight': 1,
    'subsample': 1,
    'colsample_bytree': 1,
    'lambda': 0,
    'eval_metric': ['error', 'logloss'],
    'objective': 'binary:logistic',
    'silent': 1,
    'seed': SEED,
}
_h = holdout(_params, n=200)
_cv = cv(_params, n=200)
_params = {
    'eta': 0.025,
    'gamma': 0,
    'max_depth': 3,
    'min_child_weight': 1,
    'subsample': 1,
    'colsample_bytree': 1,
    'lambda': 0,
    'eval_metric': ['error', 'logloss'],
    'objective': 'binary:logistic',
    'silent': 1,
    'seed': SEED,
}
_h = holdout(_params, n=200)
_cv = cv(_params, n=200)
_params = {
    'eta': 0.025,
    'gamma': 1,
    'max_depth': 3,
    'min_child_weight': 1.6,
    'subsample': 1,
    'colsample_bytree': 0.5,
    'lambda': 1,
    'eval_metric': ['error', 'logloss'],
    'objective': 'binary:logistic',
    'silent': 1,
    'seed': SEED,
}
_h = holdout(_params, n=200)
_cv = cv(_params, n=200)
params = {
    'eta': 0.005,
    'gamma': 2,
    'max_depth': 5,
    'min_child_weight': 1.6,
    'subsample': 0.9,
    'colsample_bytree': 0.5,
    'lambda': 16,
    'eval_metric': ['error', 'logloss'],
    'objective': 'binary:logistic',
    'silent': 1,
    'seed': SEED,
}
%%time
h = holdout(params, n=200)
plot_holdout_error(h, 0, 200)
%%time
cv_results = cv(params, n=200)
plot_cv_error(cv_results, 0, 200)
z = train(params, n=96)
X.columns[~X.columns.isin(z.get_fscore().keys())]
_, ax = plt.subplots(1, 1, figsize=(13, 16))
xgb.plot_importance(z, ax=ax, importance_type='weight');
_, ax = plt.subplots(1, 1, figsize=(13, 16))
xgb.plot_importance(z, ax=ax, importance_type='gain');
_, ax = plt.subplots(1, 1, figsize=(13, 16))
xgb.plot_importance(z, ax=ax, importance_type='cover');
xgb.to_graphviz(z, rankdir='LR', num_trees=0)
xgb.to_graphviz(z, rankdir='LR', num_trees=1)
xgb.to_graphviz(z, rankdir='LR', num_trees=2)
xgb.to_graphviz(z, rankdir='LR', num_trees=3)
xgb.to_graphviz(z, rankdir='LR', num_trees=4)
xgb.to_graphviz(z, rankdir='LR', num_trees=z.best_ntree_limit-1)
xgb.to_graphviz(z, rankdir='LR', num_trees=z.best_ntree_limit-2)
xgb.to_graphviz(z, rankdir='LR', num_trees=z.best_ntree_limit-3)
xgb.to_graphviz(z, rankdir='LR', num_trees=z.best_ntree_limit-4)
xgb.to_graphviz(z, rankdir='LR', num_trees=z.best_ntree_limit-5)
%%time
p = ensemble(params, n=96)
y_hat = threshold(p)
sum(y_hat)
submit(y_hat, 'xgb')
