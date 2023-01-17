import csv
from collections import defaultdict

def topfreqs(dct, ntop=5):
    return ['%s:%d'%(v,dct[v]) for v in sorted(dct.keys(),
                                               key=lambda x:dct[x],
                                               reverse=True)[:ntop]]

def explore_csv(filename, delimiter=',', missing_value=''):
    counts = defaultdict(lambda:defaultdict(int))
    nrow=0
    with open(filename) as f:
        r=csv.DictReader(f)
        for row in r:
            nrow+=1
            for col in row:
                counts[col][row[col]] += 1
    print('%s: %d rows, %d columns'%(filename,nrow,len(counts)))
    for col in counts:
        print(' %s uniqe:%d missing:%d %s'%(col,
                                            len(counts[col]),
                                            counts[col][missing_value] if missing_value in counts[col] else 0,
                                            ' '.join(topfreqs(counts[col]))))
    print('')

explore_csv('../input/train.csv')
explore_csv('../input/test.csv')
