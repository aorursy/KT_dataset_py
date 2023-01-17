#@title Base Imports



from json import load, dump



import numpy as np

import pandas as pd
#@title Base Utilities



class AttrDict(dict):

    """Convenience class"""

    __getattr__ = dict.__getitem__

    __setattr__ = dict.__setitem__

    __delattr__ = dict.__delitem__





def softmax(x):

    """Numerically stable softmax"""

    e_x = np.exp(x - x.max(1)[:, np.newaxis])

    return e_x / e_x.sum(1)[:, np.newaxis]





def mAP(label, prediction):

    """Mean Average Precision at top k = 2"""

    hit0 = np.equal(label, prediction[..., 0])

    hit1 = np.equal(label, prediction[..., 1])

    return (hit0 + hit1 * .5)[label > -1].mean()





def show_preview(data, run):

    return run(data) if data is not None else 'No data selected'
#@title Loading Utilities { display-mode: "form" }



PROJECT_DIR = '.'

DATA_INFO_DIR = '../input' #@param {type:"string"}





def avals(values):

    sorted_values = sorted((i, v) for v, i in values.items())

    for i, value in enumerate(sorted_values):

        assert i == value[0]

        yield value





def load_data(name, path='%s/%s' % (PROJECT_DIR, DATA_INFO_DIR)):

    """Store all info related to a dataset"""

    _dn = (path, name)

    train = pd.read_csv('%s/%s_data_info_train_competition.csv' % _dn)

    submit = pd.read_csv('%s/%s_data_info_val_competition.csv' % _dn)

    with open('%s/%s_profile_train.json' % _dn) as f:

        profile = load(f)



    attribute_types = list(train.columns[3:])

    attribute_values = { a: tuple(avals(v)) for a, v in profile.items() }



    labels = { a: train[a].values for a in attribute_types }

    train_size = len(train.index)

    submit_size = len(submit.index)

    return AttrDict(locals())





def get_counts(data):

    for attr in data.attribute_values:

        yield attr, data.train[attr].value_counts().drop(-1)
NDSC = AttrDict({ n: load_data(n) for n in ('beauty', 'fashion', 'mobile') })



jpg_stripped = NDSC.fashion.train.image_path.str.replace(r'\.jpg$', '')

NDSC.fashion.train.image_path = jpg_stripped + '.jpg'



for d in NDSC.values():

    d.train.fillna(-1., inplace=True, downcast='infer')

    d.counts = dict(get_counts(d))
#@title Preview { run: "auto", display-mode: "form" }



preview_category = "beauty" #@param ["Choose one...", "beauty", "fashion", "mobile"]

preview_set = "train" #@param ["train", "submit"]

show_preview(NDSC.get(preview_category), lambda d: d[preview_set].head())
#@title Default Thresholds



DEFAULT_THRESHOLDS = AttrDict()

DEFAULT_THRESHOLDS.beauty = {'Brand': 2000} #@param {type:"raw"}

DEFAULT_THRESHOLDS.mobile = {'Phone Model': 500} #@param {type:"raw"}

DEFAULT_THRESHOLDS.fashion = {} #@param {type:"raw"}
#@title `FilteredValues`



class FilteredValues(dict):

    """Filter attribute values that occur below a certain count"""



    base_default = 10



    def __init__(self, data, thresholds=None):

        super().__init__(self.get_values(data))

        self.data = data

        self.map = dict(self.get_map(data.counts))

        self.index = dict(self.get_index(data.counts))



        self.default = self.base_default

        self.thresholds = thresholds or DEFAULT_THRESHOLDS[data.name]





    @staticmethod

    def get_values(data):

        for attr, avs in data.attribute_values.items():

                yield attr, tuple(avs[i] for i in data.counts[attr].index)



    @staticmethod

    def get_index(counts):

        for attr, count in counts.items():

            yield attr, count.index.values



    @staticmethod

    def get_map(counts):

        for attr, ct in counts.items():

            index = ct.index.values

            mapper = np.full((index.max() + 1,), -1)

            mapper[index] = np.arange(len(index))

            yield attr, mapper



    def items(self, threshold=None):

        default = self.default



        if threshold is None:

            threshold = self.thresholds

        elif isinstance(threshold, int):

            default = threshold

            threshold = {}



        for attribute in self.data.attribute_types:

            ct = self.data.counts[attribute]

            keeps = ct[ct >= threshold.get(attribute, default)]

            values = self.data.attribute_values[attribute]

            yield attribute, tuple(v for v in values if v[0] in keeps)





for d in NDSC.values():

    d.filtered = FilteredValues(d)
#@title Attribute Value Occurence { run: "auto", display-mode: "form" }



preview_category = "beauty" #@param ["beauty", "fashion", "mobile"]

preview_attribute = "" #@param {type:"string"}



show = NDSC[preview_category].counts.get(preview_attribute)

show_preview(show, lambda d: d.plot.bar(figsize=(12, 6)))
#@title Scoring utilities



def blank_scores(data):

    for attribute, avs in data.attribute_values.items():

        kw = {

            'dtype': float,

            'index': data.submit.itemid,

            'columns': tuple(value for k, value in avs)

        }

        yield attribute, pd.DataFrame(**kw).fillna(0.)





def baseline_random(data, n=1):

    yield 'method', 'random'

    for attr, avs in data.attribute_values.items():

        l = len(avs)

        s = data.train_size

        labels = data.labels[attr]

        

        def pred():

            return np.array(tuple(np.random.permutation(l) for _ in range(s)))

        # Note: generating random values takes a lot of time

        # prec = np.array(tuple(mAP(labels, pred()) for i in range(n)))

        # yield attr, n / np.sum(1. / prec)

        yield attr, 0.





def baseline_frequency(data, weight=1):

    yield 'method', 'frequency'

    for attr, df in data.scores.items():

        counts = data.train[attr].replace(-1, np.nan).value_counts(True)

        precision = mAP(data.labels[attr], counts.index.values)

        for idx, frequency in counts.items():

            df.iloc[:, int(idx)] += frequency * precision * weight

        yield attr, precision
for d in NDSC.values():

    d.scores = dict(blank_scores(d))



    cols = ['method']

    cols.extend(d.attribute_types)

    rand = dict(baseline_random(d))

    freq = dict(baseline_frequency(d))

    d.mAP = pd.DataFrame.from_records((rand, freq), columns=cols)
#@title Preview mAP { run: "auto", display-mode: "form" }



preview_category = "beauty" #@param ["Choose one...", "beauty", "fashion", "mobile"]

show_preview(NDSC.get(preview_category), lambda d: d.mAP.head())
#@title Title Matching



def extract_values(column, values):

    """Match attribute value name and return length"""

    for key, name in values:

        print('\r\t%5d - %s' % (key, name), end='')

        reg = '\\b(?P<match>%s)\\b' % name.replace(' ', '.*')

        match = column.str.extract(reg, expand=False).str.len()

        yield name, match





def train_title(data):

    yield 'method', 'title regex'

    for attr, avs in data.filtered.items():

        print('\r +-- %s' % attr)

        df = pd.DataFrame(data=dict(extract_values(data.train.title, avs)))

        predicted = df.dropna(0, 'all')

        labels = data.labels[attr][predicted.index]

        pred = predicted.fillna(np.inf).values.argsort()

        prec = mAP(labels, data.filtered.index[attr][pred])

        yield attr, prec

    print('\r ')





def test_title(data, weight=1):

    title_mAP = data.mAP[data.mAP.method == 'title regex']    

    for attr, avs in data.filtered.items():

        print('\r +-- %s' % attr)

        w = title_mAP[attr].values[0] * weight

        matches = dict(extract_values(data.submit.title, avs))

        df = pd.DataFrame(data=matches).fillna(np.inf).rdiv(w)

        data.scores[attr][df.columns] += df.values

    print('\r ')
for d in NDSC.values():

    print(d.name)

    d.mAP = d.mAP.append(dict(train_title(d)), ignore_index=True).fillna(0.)
for d in NDSC.values():

    print(d.name)

    test_title(d)
#@title Preview scores { run: "auto", display-mode: "form" }



preview_category = "mobile" #@param ["beauty", "fashion", "mobile"]

preview_attribute = "Operating System" #@param {type:"string"}



show = NDSC[preview_category].scores.get(preview_attribute)

show_preview(show, lambda d: d.head())
#@title Generate submission file



submission_name = 'title_regex_only' #@param {type:"string"}

top_k = 2 #@param {type:"slider", min:0, max:5, step:1}



submission_filepath = '%s/submission_%s.csv' % (PROJECT_DIR, submission_name)

fmt = '%d_{},' + ' '.join('%d' for _ in range(top_k))



print('Saving to %s' % submission_filepath)

with open(submission_filepath, 'w') as f:

    print('id,tagging', file=f)

    for d in NDSC.values():

        print('%s ...' % d.name, end='')

        itemids = d.submit.itemid.values[:, np.newaxis]

        for attribute, df in d.scores.items():

            predictions = (-df).values.argsort()[:, :top_k]

            output = np.hstack((itemids, predictions))

            np.savetxt(f, output, fmt=fmt.format(attribute))

        print(' OK')
#@title Preview submission

!head $submission_filepath

print('...')

!tail $submission_filepath