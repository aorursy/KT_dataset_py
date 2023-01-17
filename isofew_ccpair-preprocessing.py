import pandas as pd
import numpy as np
import re
np.random.seed(2019)
class Pack():
    special = [
        '〈PAD〉',
        '〈UNK〉',
        '〈NUM〉',
        '〈STR〉',
        '〈EOS〉',
        '〈START〉',
    ]
    
    def __init__(self, filename):
        self.word_index, self.vector, self.starts, self.ends = np.load(filename)
        self.vector += len(Pack.special)
        self.word_index = dict(
            **{w: i for i, w in enumerate(Pack.special)},
            **{w: i + len(Pack.special) for w, i in self.word_index.items()}
        )
        self.index_word = list(self.word_index.keys())
    
    def __len__(self):
        return len(self.starts)
    
    def __getitem__(self, i):
        return self.vector[self.starts[i] : self.ends[i]]

    def pprint(self, item):
        print(' '.join(map(list(self.word_index).__getitem__, item)))
    
    def relabel(self, f):
        new_id = {}
        for j, vj in enumerate(self.vector):
            self.vector[j] = vj_ = f(vj)
            if vj_ not in new_id and vj_ >= len(Pack.special):
                new_id[vj_] = len(new_id) + len(Pack.special)
        for j, vj in enumerate(self.vector):
            if vj in new_id:
                self.vector[j] = -new_id[vj]
        self.vector[self.vector < 0] *= -1
        self.word_index = dict((
            *((w, i) for i, w in enumerate(Pack.special)),
            *sorted(((w, new_id[i]) for w, i in self.word_index.items() if i in new_id), key=lambda tup: tup[1])
        ))
        self.index_word = list(self.word_index.keys())
    
    def relabel_on_freq(self, f_infreq, f_freq, count_on, top_n):
        cnt = np.zeros(len(self.word_index), dtype=np.int64)
        if count_on.dtype == np.bool:
            count_on = np.where(count_on)[0]
        for item in count_on:
            for i in self[item]:
                cnt[i] += 1
        rank = sorted(range(len(self.word_index)), key=lambda i: cnt[i], reverse=True)
        freq = set(rank[:top_n])
        print('freq rate when counting', sum(cnt[i] for i in freq) / (1e-9 + cnt.sum()))
        cf = [0]
        ci = [1e-9]
        def inc(c, x):
            c[0] += 1
            return x
        self.relabel(lambda i: inc(cf, f_freq(i)) if i in freq else inc(ci, f_infreq(i)))
        print('freq rate when relabelling', cf[0] / (1e-9 + cf[0] + ci[0]))
    
    # turns a function on string/words to that on number/indices
    def conjugate(self, f):
        return lambda i: self.word_index[f(self.index_word[i])]
class Dataset():
    def __init__(self, src, dst):
        self.src = src
        self.dst = dst
        self.eos = self.dst.word_index['〈EOS〉']
        self.start = self.dst.word_index['〈START〉']

    def __len__(self):
        return len(self.src.starts)

    def __getitem__(self, i):
        return self.src[i], [self.start, *self.dst[i]], [*self.dst[i], self.eos]
    
    def pprint(self, pair):
        print('>>> src')
        self.src.pprint(pair[0])
        print()
        print('>>> dst_in')
        self.dst.pprint(pair[1])
        print()
        print('>>> dst_out')
        self.dst.pprint(pair[2])
ds = Dataset(Pack('../input/codes.npy'), Pack('../input/comms.npy'))
print('src/dst num_tokens:', len(ds.src.vector), len(ds.dst.vector))
df = pd.read_csv('../input/code_id.csv')
df['src_len'] = ds.src.ends - ds.src.starts
df['dst_len'] = ds.dst.ends - ds.dst.starts
empty = (df.src_len == 0) | (df.dst_len == 0)
print('Empty rate', np.mean(empty))
df['file_long_id'] = df.repo_id * 1000000000 + df.file_id
repos = df[~empty].repo_id.unique()
files = df[~empty].file_long_id.unique()
print('# repos', len(repos))
print('# files', len(files))
def rand(x):
    rand = np.linspace(0, 1, len(x))
    np.random.shuffle(rand)
    return rand
rand_repo = rand(repos)
rand_file = rand(files)
rand_pair = rand(df)

split = 0.05
test_repos = set(repos[(split < rand_repo) & (rand_repo < 2 * split)])
valid_repos = set(repos[rand_repo < split])
valid_files = set(files[rand_file < split])

# test set for both bleu scoring and human checking, so empty output is okay (check by hand)
test = empty | df.repo_id.map(test_repos.__contains__)
valid = ~empty & df.repo_id.map(valid_repos.__contains__)
train = ~valid & ~test

# different validation strategy
valid_same_repo = train & df.file_long_id.map(valid_files.__contains__)
valid_same_file = train & ~valid_same_repo & (rand_pair < split)
train = train & ~valid_same_repo & ~valid_same_file

print('\t'.join('train, test, valid, s_repo, s_file'.split(', ')))
print('\t'.join(map(str, map(np.sum, (train, test, valid, valid_same_repo, valid_same_file)))))
unk = Pack.special.index('〈UNK〉')
ds.dst.relabel_on_freq(
    f_infreq = lambda i: unk,
    f_freq = lambda i: i,
    count_on = train,
    top_n = 30000,
)
def chain(f, *fs):
    if len(fs) == 0:
        return f
    else:
        return lambda i: f(chain(*fs)(i))
string_re = re.compile('^".*"$')
number_re = re.compile('^([0-9]+\.?[0-9]*|[0-9]*\.[0-9]+)([eE][0-9]+)?$')
string_sub = lambda s: '〈STR〉' if string_re.match(s) else s
number_sub = lambda s: '〈NUM〉' if number_re.match(s) else s
#slash_sub = lambda s: s.split('/')[0]
to_unk_sub = lambda s: '〈UNK〉' if s not in ('〈STR〉', '〈NUM〉') else s
ds.src.relabel_on_freq(
    f_infreq = ds.src.conjugate(chain(to_unk_sub, string_sub, number_sub)),
    f_freq = ds.src.conjugate(chain(string_sub, number_sub)),
    count_on = train,
    top_n = 30000,
)
pprint_sample = lambda s: ds.pprint(ds[df[s].sample().index[0]])
pprint_sample(train & (df.src_len < 80))
print(); print('-' * 80); print()
pprint_sample(valid & (df.src_len < 80))
print(); print('-' * 80); print()
pprint_sample(test & (df.src_len < 80))
np.save('bundle.npy', (ds, df, train, test, valid, valid_same_repo, valid_same_file))
