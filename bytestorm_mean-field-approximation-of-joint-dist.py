import pickle, csv, os
import numpy as np
from tqdm import tqdm_notebook, tqdm
import matplotlib.pyplot as plt
from collections import defaultdict
%matplotlib inline
print(os.listdir('../input/'))
row_count = 184903890
pos_count = 0
neg_count = 0

ips = defaultdict(int)
apps = defaultdict(int)
devs = defaultdict(int)
oss = defaultdict(int)
channels = defaultdict(int)

ips_attributed = defaultdict(int)
apps_attributed = defaultdict(int)
devs_attributed = defaultdict(int)
oss_attributed = defaultdict(int)
channels_attributed = defaultdict(int)

with open('../input/train.csv') as f:
    fcsv = csv.reader(f)
    header = True
    for row in tqdm_notebook(fcsv, unit='rows', total=184903890+1):
        # Skip the header row
        if header:
            header = False
            continue
        # Count positive and negative rows
        if row[-1] == '0':
            neg_count += 1
        elif row[-1] == '1':
            pos_count += 1
            
        # Maintains count for each ip, os, device etc.        
        ip, app, dev, os, channel, ct, at, is_at = row
        ips[ip] += 1
        apps[app] += 1
        devs[dev] += 1
        channels[channel] += 1
        oss[os] += 1
        
        # Also store the count how many times each ip, os... appear in
        # a positive sample
        if is_at == '1':
            ips_attributed[ip] += 1
            apps_attributed[app] += 1
            devs_attributed[dev] += 1
            channels_attributed[channel] += 1
            oss_attributed[os] += 1
print(pos_count)
print(neg_count)
print('There were {} ips, {} devices, {} apps, {} channels and {} oss'.format(
    len(ips),
    len(devs),
    len(apps),
    len(channels),
    len(oss),
))
fallback_prob = 456846/184447044
print('Fallback Prob.: ', fallback_prob)
ips_prob = defaultdict(lambda: fallback_prob, {k: ips_attributed[k]/ips[k] for k in ips.keys()})
devs_prob = defaultdict(lambda: fallback_prob, {k: devs_attributed[k]/devs[k] for k in devs.keys()})
channels_prob = defaultdict(lambda: fallback_prob, {k: channels_attributed[k]/channels[k] for k in channels.keys()})
apps_prob = defaultdict(lambda: fallback_prob, {k: apps_attributed[k]/apps[k] for k in apps.keys()})
oss_prob = defaultdict(lambda: fallback_prob, {k: oss_attributed[k]/oss[k] for k in oss.keys()})
probs = np.zeros((184903890,1))
y_true = np.zeros((184903890,1), dtype=np.uint8)

i = 0
with open('../input/train.csv') as f:
    fcsv = csv.reader(f)
    header = True
    for row in tqdm_notebook(fcsv, unit='rows', unit_divisor=1000, total=(184903890+1)):
        if header:
            header = False
            continue
            # collected_rows.append(row)
        ip, app, dev, os, channel, ct, at, is_at = row
        probs[i] = ips_prob[ip]*apps_prob[app]*devs_prob[dev]*oss_prob[os]*channels_prob[channel]
        y_true[i] = np.uint8(is_at)
        i += 1
pos_probs = probs[np.where(y_true==1)]
neg_probs = probs[np.where(y_true==0)]
safety_offset = 1e-19
def plot_hist(start = 0, bins=100):
    density_neg, bins, _ = plt.hist(np.log10(safety_offset + neg_probs[start:(start + 10000)]), bins=bins, normed=True, alpha=1.0)
    density_pos, bins, _ = plt.hist(np.log10(safety_offset + pos_probs[start:(start + 10000)]), bins=bins, normed=True, alpha=0.8)
    _ = plt.legend(['Negative', 'Positive'])
    return density_pos, density_neg, bins
density_pos, density_neg, bins = plot_hist(0)
density_pos, density_neg, bins = plot_hist(10000, bins)
density_pos, density_neg, bins = plot_hist(100000, bins)
## VERIFY: PMF sums are 1.0
print(np.sum(density_pos*(bins[1:] - bins[:-1])))
print(np.sum(density_neg*(bins[1:] - bins[:-1])))
pps = density_pos / (density_pos + density_neg)
# ranged_probs = {bins for i in range(density_pos.shape[0])}
def query_prob_to_bin_prob(query_prob):
    return pps[np.argmax((np.log10(query_prob + safety_offset) - bins[:-1]) < 0) - 1]    
query_prob_to_bin_prob(1e-14)
probs_test = np.zeros((18790469,1))
test_ids = np.zeros((18790469,1))
i = 0
with open('../input/test.csv') as f:
    fcsv = csv.reader(f)
    header = True
    for row in tqdm_notebook(fcsv, unit='rows', total=(18790469+1)):
        if header:
            header = False
            continue
            # collected_rows.append(row)
        cid, ip, app, dev, os, channel, ct = row
        probs_test[i] = query_prob_to_bin_prob(ips_prob[ip]*apps_prob[app]*devs_prob[dev]*oss_prob[os]*channels_prob[channel])
        test_ids[i] = int(cid)
        i += 1
with open('./mean_field_sub1.csv', 'w') as f:
    f.write('click_id,is_attributed\n')
    for i in tqdm(range(test_ids.shape[0])):
        f.write('%d,%0.6f\n' % (test_ids[i, 0], probs_test[i, 0]))
