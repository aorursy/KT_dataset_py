import os

import itertools



from collections import defaultdict, Counter

from random import Random

from functools import partial



from urllib import request, parse



import numpy as np



%matplotlib inline

from matplotlib import pyplot as plt
ATIS_BASE_URL="https://raw.githubusercontent.com/yvchen/JointSLU/master/data/"



def load_atis_ds(fname, base_url=ATIS_BASE_URL):

    res = []

    with request.urlopen(parse.urljoin(base_url,fname)) as req:

        for line in req.readlines():

            line = line.decode(req.info().get_content_charset())

            toks,si      = map(str.split, line.split("\t"))

            slots,intent = si[:-1]+['O'], si[-1]

            assert len(toks) == len(slots)

            res.append((toks, slots, intent))

    ds_name = '.'.join(fname.split('.')[:2])

    print('{:>20s}: {:4d}'.format(ds_name, len(res)))

    return res, ds_name



atis = {name: ds for ds, name in 

        map(load_atis_ds, map(lambda name: name+'.w-intent.iob',

                              ['atis.test', 'atis-2.dev','atis-2.train']))}

# a single entry looks like this

toks,slots,intent = atis['atis.test'][0]

print(' input:', ' '.join(toks))

print(' slots:', ' '.join(slots))

print('intent:',          intent)
test,dev,train = map(atis.get, ['atis.test','atis-2.dev','atis-2.train'])



def subdict(d, keys):

    return {key: d[key] for key in set(keys)}



def atis_label_counts(ds):

    tokens, slots, intents = zip(*ds)

    token_labs  = Counter(list(itertools.chain.from_iterable(tokens)))

    slot_labs   = Counter(list(itertools.chain.from_iterable(slots)))

    intent_labs = Counter(intents)

    return token_labs, slot_labs, intent_labs



def check_atis_split(train, dev, test):

    (_,ts,ti), (_,ds,di), (_,es,ei) = map(atis_label_counts,

                                        [train, dev, test])



    lens = np.array(list(map(len, [train,dev,test])))

    print("   sample count: {:5d} splitted into:".format(lens.sum()))

    for dslen, dsname in zip(lens, ['train', 'dev', 'test']):

        print("          {:>5s}: {:5d} ({:.3f})".format(dsname, dslen, dslen/lens.sum()))



    unique_entries = train+dev+test

    unique_entries = set([(tuple(toks),tuple(slots),intent) 

                        for toks, slots, intent in unique_entries])

    print("duplicate count: {:5d}".format(len(train+dev+test)-len(unique_entries)))

    

    # map slot/intent labels to usage frequency

    token_labs,slot_labs,intent_labs = atis_label_counts(train+dev+test)

    sfreqs, ifreqs = map(partial(partial,subdict), [slot_labs, intent_labs])



    print("   intent count:", len(intent_labs))

    print("     slot count:", len(slot_labs))

    print("    token count:", len(token_labs))



    print("missing data for slot/intent labels:")

    ts,ti,ds,di,es,ei = map(lambda s: set(s.keys()), [ts,ti,ds,di,es,ei])

    for dsname, mints, mslots in [("train", 

                                 ifreqs(di.union(ei).difference(ti)),

                                 sfreqs(ds.union(es).difference(ts))),

                                ("dev", 

                                 ifreqs(ti.difference(di)),

                                 sfreqs(ts.difference(ds))),

                                ("test", 

                                 ifreqs(ti.difference(ei)),

                                 sfreqs(ts.difference(es)))]:

        print("  no {:>5s} data for {:2d} intents: {}".format(dsname, len(mints), mints))

        print("  no {:>5s} data for {:2d}   slots: {}".format(dsname, len(mslots), mslots))





check_atis_split(train, dev, test)
def visualize_atis_split(train,dev,test):

    (_,ts,ti), (_,ds,di), (_,es,ei) = map(atis_label_counts,

                                        [train, dev, test])

    _,alls, alli = atis_label_counts(train+dev+test)



    tsc,dsc,esc,asc = (np.array([d[slot]   for slot,_   in alls.most_common()])

                                for d in [ts,ds,es, alls])

    tic,dic,eic,aic = (np.array([d[intent] for intent,_ in alli.most_common()]) 

                                for d in [ti,di,ei, alli])



    plt.figure(figsize=(18,4))

    plt.title('intent label distribution')

    plt.bar(x=np.arange(len(tic)), height=tic/aic,                       label='train')

    plt.bar(x=np.arange(len(dic)), height=dic/aic, bottom=tic/aic,       label='dev')

    plt.bar(x=np.arange(len(eic)), height=eic/aic, bottom=(tic+dic)/aic, label='test')

    plt.legend(loc='center left', fancybox=True, framealpha=0.5)



    plt.figure(figsize=(18,4))

    plt.title('slot labels distribution')

    plt.bar(x=np.arange(len(tsc)), height=tsc/asc,                       label='train')

    plt.bar(x=np.arange(len(dsc)), height=dsc/asc, bottom=tsc/asc,       label='dev')

    plt.bar(x=np.arange(len(esc)), height=esc/asc, bottom=(tsc+dsc)/asc, label='test')

    plt.legend(loc='center left', fancybox=True, framealpha=0.5)



visualize_atis_split(train,dev,test)
test,dev,train = map(atis.get, ['atis.test','atis-2.dev','atis-2.train'])

atis_all = train + dev + test





print("original ATIS    data count:",len(atis_all))



cleands = atis_all

# keep unique entries only

cleands = set([(tuple(toks),tuple(slots),intent) for toks, slots, intent in cleands])

print("   after duplicates removal:",len(cleands))



# count label frequencies

_, slot_labs, intent_labs = atis_label_counts(cleands)



# filter labels occuring only once

clean_slots   = dict(filter(lambda t: t[1] > 3, slot_labs.most_common()))

clean_intents = dict(filter(lambda t: t[1] > 3, intent_labs.most_common()))



# remove the corresponding data samples

cleands = list(filter(lambda t: set(t[1]).issubset(clean_slots), cleands))

cleands = list(filter(lambda t: t[2] in clean_intents,           cleands))



cleands = list(filter(lambda t: set(t[1]).issubset(slot_labs) and t[2] in intent_labs, cleands))

print("       after label cleaning:",len(cleands))



# convert back from tuples to lists

atis_clean = [(list(toks),list(slots),intent) for toks, slots, intent in sorted(list(cleands))]

def split_atis(ads, split=[0.8, 0.1, 0.1],random_state=None):

    """ Splits the ATIS dataset by starting with the least common labels."""

    split = np.array(split)

    assert split.sum() == 1



    random = None if random_state is None else Random(random_state)



    res=[[] for _ in split]

    used = set()             # used samples

    s2e = defaultdict(set)   # slot to sample

    i2e = defaultdict(set)   # intent to sample

    for ndx, (_,slots,intent) in enumerate(ads): # build label to sample maps

        i2e[intent].add(ndx)

        for slot in set(slots):

            s2e[slot].add(ndx)

    

    # sort according to usage

    s2f = Counter({slot:len(ndxs)   for slot,ndxs   in s2e.items()}).most_common()

    i2f = Counter({intent:len(ndxs) for intent,ndxs in i2e.items()}).most_common()

    

    while True:

        # select the least common intent or slot label

        if len(i2f)<1 and len(s2f)<1:            # both empty

            break

        elif len(i2f)>0 and len(s2f)>0:          # both non empty

            use_intent = i2f[-1][1] < s2f[-1][1]

        else:

            use_intent = len(i2f)>0

        

        # get the samples of least common label

        if use_intent:

            intent, _ = i2f.pop()

            ndxs = i2e.pop(intent)

        else:

            slot, _ = s2f.pop()

            ndxs = s2e.pop(slot)

        

        # shuffle the samples of the selected label

        ndxs = list(ndxs)

        if random is not None:

            random.shuffle(ndxs)

    

        splits = [[] for _ in split]

        # put each sample in a split maintaining the split ratios    

        for ndx in ndxs:

            lens = np.array([len(sp) for sp in splits])

            # fill the split with the highest frequency (reverse ratio) offset

            sndx = np.argmax(1/(lens/(lens.sum()+1e-12)+1e-12) - 1/split)

            if ndx not in used:

                used.add(ndx)

                splits[sndx].append(ndx)

        

        # add the splitted label samples to the result split

        for ndx, ndxs in enumerate(splits):

            res[ndx].extend(ndxs)

            

    return res
train,test,dev = split_atis(atis_clean,[0.8,0.1,0.1], random_state=7411)

train,test,dev = map(lambda ndxs: np.array(atis_clean)[ndxs].tolist(),

                     [train,test,dev])
check_atis_split(train,dev,test)
visualize_atis_split(train,dev,test)
# lets dump the split in a CSV file per dataset

def store_atis_csv(ds, dsname):

    fname = "atis.{}.csv".format(dsname)

    with open(fname, 'wt', encoding='UTF-8') as f:

        f.write("id,tokens,slots,intent\n") # csv header

        for ndx, (toks, slots, intent) in enumerate(ds, 1):

            uid = "{}-{:05d}".format(dsname, ndx)

            line = "{},{},{},{}\n".format(uid, 

                                         " ".join(toks), 

                                         " ".join(slots),

                                         intent)

            f.write(line)

    print("Done writting {} to {}".format(dsname,fname))

    

store_atis_csv(train, "train")

store_atis_csv(dev,   "dev")

store_atis_csv(test,  "test")
!head atis.train.csv