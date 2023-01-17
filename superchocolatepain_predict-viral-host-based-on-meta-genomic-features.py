# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

virus_csv_file = '../input/genome-information-for-sequenced-organisms/viruses.csv'
viruses_df = pd.read_csv(virus_csv_file)
viruses_df.head(5)


# clean up column names i.e. remove the erroneous characters from the column names (spaces, percent sign, etc.)

# clean up column names https://stackoverflow.com/a/11346337/6542644 
viruses_df.columns = ['organism_name', 'organism_groups', 'BioSample', 'Bioproject', 'Assembly', 'Level', 'size_mb', 'gc_percent', 'replicons', 'host', 'cds', 'neighbours', 'release_date', 'genbank_ftp', 'refseq_ftp', 'replicons1']

# verify column names have been changed
viruses_df.head(1)
#Explore the data

# .shape[0] gives the number of rows in the dataframe, which is the number of viral species in the dataset
print('Number of viruses: ', viruses_df.shape[0])

# .unique gives the number of unique items in a specified column, in this case the number of viral hosts
print('Number of unique viral host types: ', (len(viruses_df['host'].unique())))

viruses_df['host'].unique()
ohe_df = pd.get_dummies(viruses_df['host'], prefix='host')
ohe_df.head(5)
# drop columns we don't need 
virus_feats_only = viruses_df.drop(['organism_name', 'organism_groups', 'BioSample', 
                                    'Bioproject', 'Assembly', 'Level', 'replicons', 
                                    'neighbours', 'release_date', 'genbank_ftp', 'refseq_ftp',
                                   'replicons1'], axis=1)


# create dict to map strings to numerical values, also combines overlapping hosts: vertebrates/human and human
viruses_host_dict = {'bacteria': 0, 'fungi': 1, 'plants': 2, 'vertebrates': 3,
                    'invertebrates': 4, 'protozoa': 5, 'vertebrates, invertebrates, human': 6,
                    'invertebrates, plants': 7, 'algae': 8, 'vertebrates, invertebrates': 9,
                    'vertebrates, human': 10, 'archaea': 11, 'human': 10}

# replace method use cited from: https://stackoverflow.com/a/20250996/6542644 
virus_feats_cleanhost = virus_feats_only.replace({'host':viruses_host_dict})

first_col = virus_feats_cleanhost.pop('host')
virus_feats_cleanhost.insert(0, 'host', first_col)
virus_feats_cleanhost.head(5)
# check for NaN values in data

print("Count of NaN in host: ", virus_feats_cleanhost['host'].isnull().sum())
print("Count of NaN in Size_Mb: ", virus_feats_cleanhost['size_mb'].isnull().sum())
print("Count of NaN in GC_percent: ", virus_feats_cleanhost['gc_percent'].isnull().sum())
print("Count of NaN in cds: ", virus_feats_cleanhost['cds'].isnull().sum())

viruses_dropped_nan = virus_feats_cleanhost.dropna()

# count of NaN values in a column cited from: https://datatofish.com/check-nan-pandas-dataframe/

print("Count of NaN after dropna(): ", viruses_dropped_nan['host'].isnull().sum())

viruses_dropped_nan.head(5)
display(viruses_dropped_nan)
# Check distribution of viral hosts
counts_host2 = viruses_dropped_nan.copy()
counts_host_unique = counts_host2.groupby(['host']).size().reset_index(name='Counts')

counts_host_unique
# Plot distribution of viral hosts

import matplotlib.pyplot as plt
%matplotlib inline

group = ['host']
counts = viruses_dropped_nan.groupby(group).size().reset_index(name="Counts")

# use of plt cited from: https://python-graph-gallery.com/4-add-title-and-axis-label/
bars = ('bact', 'fungi', 'plants', 'verts', 'inverts',
        'pro', 'v/i/hum',
        'i/plants', 'algae', 'v/i',
        'v/hum', 'archaea')
y_pos = np.arange(len(bars))

#plt.title('Distribution of Unique Host Types')
plt.figure(figsize=(10,8))
plt.bar(range(len(counts)), counts['Counts'], color = 'blue')
plt.title('Distribution of Viral Hosts')
plt.xlabel('Viral Hosts')
plt.ylabel('Count')
plt.xticks(y_pos, bars)
# Get stats for viral genome size (size_mb) and plot distribution

print('Number of unique size values: ', (len(viruses_dropped_nan['size_mb'].unique())))

print(viruses_dropped_nan.size_mb.describe())

print("mode of Size_Mb is: ", viruses_dropped_nan['size_mb'].mode())

# replace values of 0 with the mean cited from: https://stackoverflow.com/a/11455375/6542644
from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values=0, strategy = 'mean')

cleaned_sizemb_df = viruses_dropped_nan['size_mb']

drop_host = viruses_dropped_nan.drop(['host'], axis=1)

imp.fit(drop_host)

cleaned_df = imp.transform(drop_host)

cleaned_df = pd.DataFrame(data=cleaned_df, columns=["size_mb", "gc_percent", "cds"])

cleaned_size = cleaned_df['size_mb']

print("count zeroes is: ", cleaned_size.isin([0]).sum())

cleaned_size = cleaned_size.sort_values()
cleaned_size.hist(bins = 100)
plt.title('Distribution of Size of Genome Mb')
plt.xlabel('Size of genome in Mb')
plt.ylabel('Count')
# Get stats for GC% (gc_percent) and plot distribution

print('Number of unique GC% values: ', (len(viruses_df['gc_percent'].unique())))

print(viruses_df.gc_percent.describe())

gc_sorted_df = viruses_df['gc_percent']
gc_sorted_df = gc_sorted_df.sort_values()

print('Most frequent value is ', gc_sorted_df.mode() )
gc_sorted_df.hist(bins = 50)
plt.title('Distribution of GC% of Viral Genomes')
plt.xlabel('GC% of Viral Genomes')
plt.ylabel('Count')
# Get stats for CDS (cds) and plot distribution

print('Number of unique CDS values: ', (len(viruses_df['cds'].unique())))

print(viruses_df.cds.describe())

cds_sorted_df = viruses_df['cds']
cds_sorted_df = cds_sorted_df.sort_values()

print('Most frequent value is ', cds_sorted_df.mode() )

print('count zeroes is ', cds_sorted_df.isin([0]).sum() )

cds_sorted_df.hist(bins = 50)
plt.title('Distribution of CDS of Viral Genomes')
plt.xlabel('CDS of Viral Genomes')
plt.ylabel('Count')
# Pre-process Data

from sklearn import preprocessing

new_viral_df = viruses_dropped_nan.copy()

targets_host = new_viral_df.pop('host')

x = new_viral_df.values

min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)

unclean_viral_df = pd.DataFrame(x_scaled)

unclean_viral_df.insert(0, 'host', targets_host)

# remove the NaN values!
new_df_clean = unclean_viral_df.dropna()

new_df_clean.columns = ['host', 'size_mb', 'gc_percent', 'cds']
new_df_clean.head(5)
# Prep the data: generate X and y

# create features and labels
y = new_df_clean['host']
X = new_df_clean.drop(['host'], axis=1)
y.columns = ['host']

# X.head(5)
# SVM usage cited from: https://towardsdatascience.com/multiclass-classification-with-support-vector-machines-svm-kernel-trick-kernel-functions-f9d5377d6f02

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import svm, datasets


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=0)

# start with linear kernel

linear = svm.SVC(kernel='linear', C=1, decision_function_shape='ovo').fit(X_train, y_train)
                                            
linear_pred = linear.predict(X_test)

# retrieve accuracy
accuracy_lin = linear.score(X_test, y_test)

print("acc linear kernel: ", accuracy_lin)

cm_lin = confusion_matrix(y_test, linear_pred)
print(cm_lin)
# now try rbf kernel

rbf = svm.SVC(kernel='rbf', C=1, decision_function_shape='ovo', probability=True).fit(X_train, y_train)
rbf_pred = rbf.predict(X_test)

accuracy_rbf = rbf.score(X_test, y_test)
print("acc rbf kernel: ", accuracy_rbf)

cm_rbf = confusion_matrix(y_test, linear_pred)
print(cm_rbf)
# now try with cross validation!

from sklearn.model_selection import cross_val_score

clf_cross_val = svm.SVC(kernel='rbf', C=1, decision_function_shape='ovo')

# use of cross_val_score cited from: https://scikit-learn.org/stable/modules/cross_validation.html

scores = cross_val_score(clf_cross_val, X, y, cv=10)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
