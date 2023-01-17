%matplotlib inline

import numpy as np
import pandas as pd



src = pd.read_csv("../input/Salaries.csv", low_memory=False)
src.describe()
src.head(10)
from nltk.corpus import stopwords
en_stopwords = stopwords.words('english')
#stopwords quita las palabras básicas: i, as, how, etc..

src2014 = src[src["Year"]==2014]

job_titles = src2014["JobTitle"]
unique_job_titles = job_titles.unique()
#obtiene los jobs únicos, como agrupar por jobs

print("Job titles: ", job_titles.count(), " unique job titles ", len(unique_job_titles))

import re
def tokenize(title):
    return filter(lambda w: w and (w not in en_stopwords), re.split('[^a-z]*', title.lower()))


tokenized_titles = list(filter(None, [list(tokenize(title)) for title in unique_job_titles]))

lengths = list(map(len, tokenized_titles))

pd.DataFrame({"number or keywords per title": lengths}).hist(figsize=(16, 4));
#pd.DataFrame({"number or keywords per title": lengths}).plot(figsize=(16, 4));
tokenized_title = tokenized_titles[0]

print("original: ", tokenized_title)

import itertools

def keywords_for_title(tokenized_title):
    for keyword_len in range(1, len(tokenized_title)+1):
        for keyword in itertools.combinations(tokenized_title, keyword_len):
            yield keyword

print("keywords: ", list(keywords_for_title(tokenized_title)))
from collections import Counter
keywords_stats = Counter()
for title in tokenized_titles:
    for keyword in keywords_for_title(title):
        keywords_stats[keyword] += 1

print("distinct keywords: ", len(keywords_stats))
keywords_stats.most_common(1115)
salaries = Counter()
counts = Counter()
min_salaries = {}
max_salaries = {}

for index, row in src2014.iterrows():
    job_title = row["JobTitle"]
    salary_plus_benefits = row["TotalPayBenefits"]
    
    # Remove temporary jobs
    if salary_plus_benefits < 10000:
        continue
    
    tokenized_title = list(tokenize(job_title))
    if not tokenized_title:
        continue
    for keyword in keywords_for_title(tokenized_title):
        salaries[keyword] += salary_plus_benefits
        counts[keyword] += 1
        if keyword in max_salaries:
            if salary_plus_benefits < min_salaries[keyword]:
                min_salaries[keyword] = salary_plus_benefits
            if salary_plus_benefits > max_salaries[keyword]:
                max_salaries[keyword] = salary_plus_benefits
        else:
            min_salaries[keyword] = salary_plus_benefits
            max_salaries[keyword] = salary_plus_benefits
s_all = src2014["TotalPayBenefits"].sum()
n_all = src2014["TotalPayBenefits"].count()
avg_salary = s_all / n_all

shifts = Counter()
variances = Counter()

for keyword in salaries:
    s_with = salaries[keyword]
    n_with = counts[keyword]
    
    # Skip ill-cases
    if min_salaries[keyword] == 0:
        continue
    
    if n_with < 5:
        continue
    
    if len(keyword) > 2:
        continue

    avg_salary_with = s_with / n_with
        
    shifts[keyword] = (avg_salary_with - ((s_all-s_with)/(n_all-n_with))) / avg_salary
    variances[keyword] = max_salaries[keyword] / min_salaries[keyword]

print('shifts: ', shifts.most_common(15))
print()
print('variances: ', variances.most_common(15))
keys=[]
shifts_v=[]
variances_v=[]

for i in shifts.keys():
    keys.append(i)
    shifts_v.append(shifts[i])
    variances_v.append(variances[i])

stats = pd.DataFrame({"keys": keys, "shifts": shifts_v, "variances": variances_v})
stats.plot("shifts", "variances", "scatter")
# Keep only the above average salaries
keys=[]
shifts_v=[]
variances_v=[]

for i in shifts.keys():
    if shifts[i] <= 1:
        continue
    keys.append(i)
    shifts_v.append(shifts[i])
    variances_v.append(variances[i])

stats = pd.DataFrame({"keys": keys, "shifts": shifts_v, "variances": variances_v})
stats.plot("shifts", "variances", "scatter")
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
X = stats[['shifts', 'variances']].values
#X = StandardScaler().fit_transform(X)
#db = DBSCAN(eps=0.4, min_samples=10).fit(X)
db = KMeans(n_clusters=3).fit(X)

##############################################################################
# From http://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html
##############################################################################

core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
#core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

print('Estimated number of clusters: %d' % n_clusters_)

##############################################################################
# Plot result
import matplotlib.pyplot as plt

# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = 'k'

    class_member_mask = (labels == k)

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=14)

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=6)

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()
from collections import defaultdict
labels_per_class = defaultdict(list)
for label in set(labels):
    for cnt, i_label in enumerate(labels):
        if i_label == label:
            labels_per_class[label].append(cnt)

[(k, len(labels_per_class[k])) for k in labels_per_class.keys()]
for i_class in range(len(labels_per_class)):
    print("class: ", i_class)
    for ikey in labels_per_class[i_class]:
        dp = [
            ("shift", format(shifts_v[ikey], '.2f')),
            ("variance", format(variances_v[ikey], '.2f')),
            ("key", keys[ikey]),
        ]
        print("  -", dp)
