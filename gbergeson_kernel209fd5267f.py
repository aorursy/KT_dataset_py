# This section is auto-generated from Kaggle. I decided to do my project using 

# Kaggle's resources because my dataset is 743 MB, meaning that my computer will struggle

# to do any sort of machine learning (probably), whereas Kaggle's resources will do it a 

# lot faster.



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# imports

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
pd.set_option('display.max_columns', 500)

feature_names = [

    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 

    'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 

    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',

    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 

    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 

    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 

    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 

    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',

    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 

    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'traffic_type'

]

df = pd.read_csv('/kaggle/input/kdd-cup-1999/kddcup.data.corrected', header=None, names=feature_names)



# Trim off the last character of the traffic_type, they're all periods

df['traffic_type'] = df.traffic_type.str[:-1]
df.head()
# 1. Check for missing values

df.isna().sum()
# Nothing missing. Maybe they're coded differently? I'll look at summary statistics.

pd.set_option("display.max_columns",1000)

pd.options.display.float_format = '{:20,.2f}'.format

df.describe(percentiles=[.5])
df.describe(percentiles=[.5]).transpose()
# Nothing looks like a different encoding for missing values. The one duration is really large, so I'll check that out.

df.loc[df.duration > 58000, :]
# Looks a little odd to me for a UDP connection to be ~16 hours long, but it says it's normal traffic.

# So, no missing values in the numerical data. Now check the categorical data.



df.loc[:, df.dtypes == object].head()
for col in df.loc[:, df.dtypes == object].columns:

    print(df[col].unique())
# These are all expected values for the categorical data. Moving on to some visualization.



traffic_counts = pd.DataFrame(df.traffic_type.value_counts()).rename(columns={'traffic_type':'counts'})

traffic_counts['percent'] = df.traffic_type.value_counts() / len(df) * 100



print(traffic_counts)
sns.set(style="darkgrid")

ax = sns.barplot(y=traffic_counts.percent, x=traffic_counts.index)

ax.set_xticklabels(traffic_counts.index, rotation=80);
# Hm. Hugely imbalanced. That's going to be a problem later.



# Now generate a "traffic class" variable and we'll examine that. 

# There are four classes that all the traffic types fit into (apart from normal traffic):

#  1. DOS - denial of service

#  2. R2L - unauthorized access from remote

#  3. U2R - unauthorized access from local

#  4. Probe - port scanning or other surveillance activity



dos_class = ['back', 'land', 'neptune', 'pod', 'smurf', 'teardrop']

r2l_class = ['ftp_write', 'guess_passwd', 'imap', 'multihop',

             'phf', 'spy', 'warezclient ', 'warezmaster']

u2r_class = ['buffer_overflow', 'loadmodule', 'perl','rootkit']

probe_class = ['ipsweep', 'nmap', 'portsweep', 'satan']



df['traffic_class'] = 'Normal'

df.loc[df.traffic_type.isin(dos_class), 'traffic_class'] = 'DOS'

df.loc[df.traffic_type.isin(r2l_class), 'traffic_class'] = 'R2L'

df.loc[df.traffic_type.isin(u2r_class), 'traffic_class'] = 'U2R'

df.loc[df.traffic_type.isin(probe_class), 'traffic_class'] = 'Probe'



traffic_class_values = df.traffic_class.value_counts()

ax = sns.barplot(x=traffic_class_values.index, y=traffic_class_values)
# I'd like to understand the levels of some variables in relation to all traffic types and their various classes.

# I chose the variables I did as I looked through the data description. These looked like the most interesting

# and possibly the most helpful to distinguish between different traffic types.

# I tried this with violin plots and barplots with the standard deviation, but the plots weren't helpful.



def plot_variable(variable):

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 8))

    ax1.set_xticklabels(df.traffic_type, rotation=80)

    ax2.set_xticklabels(df.traffic_class, rotation=80)

    

    sns.barplot(x=df.traffic_type, y=variable, ax=ax1, ci=None)

    sns.barplot(x=df.traffic_class, y=variable, ax=ax2, ci=None)
# Number of "hot" indicators. This is  a composite of several of the other variables.

plot_variable(df.hot)
# Duration of the connection in seconds

plot_variable(df.duration)
# Number of wrong fragments

plot_variable(df.wrong_fragment)
# Count is the number of connections to the same host as the current 

# connection in the past two seconds

plot_variable(df['count'])
# Now to look at correlations. Rounding for display purposes. 

corrs = round(df.corr(), 2)
first_half_cols = corrs.columns[: int(len(corrs.columns) / 2)]

second_half_cols = corrs.columns[int(len(corrs.columns) / 2) :]



first_half_corrs = corrs[first_half_cols]

second_half_corrs = corrs[second_half_cols]



""" This code is for when I thought I wanted to do four heatmaps to visualize clearly. Two is enough.

halfway_point = int(len(corrs) / 2)



quad_1 = first_half_corrs[:halfway_point]

quad_2 = first_half_corrs[halfway_point:]

quad_3 = second_half_corrs[:halfway_point]

quad_4 = second_half_corrs[halfway_point:]

"""
cmap = sns.diverging_palette(10, 240, sep=20, as_cmap=True)

plt.figure(figsize=(24, 12))

sns.heatmap(first_half_corrs, annot=True, vmin=-1.0, vmax=1.0, cmap=cmap)
plt.figure(figsize=(24, 12))

sns.heatmap(second_half_corrs, annot=True, vmin=-1.0, vmax=1.0, cmap=cmap)
from sklearn.ensemble import RandomForestClassifier



rf = RandomForestClassifier()
from sklearn.preprocessing import LabelEncoder



def encode_categorical(df_, drop_columns):

    """drop_columns is list like of strings, should include target."""

    

    features = df_.drop(columns=drop_columns)

    target = df_.traffic_type



    encoder = LabelEncoder().fit(target)

    enc_target = encoder.transform(target)

    enc_features = pd.get_dummies(features)

    

    return encoder, enc_target, enc_features
from sklearn.model_selection import train_test_split



def get_train_test(df_, drop_columns):

    encoder, enc_target, enc_features = encode_categorical(df_, drop_columns)

    xtrain, xtest, ytrain, ytest = train_test_split(enc_features, enc_target, random_state=7593)

    

    return encoder, xtrain, xtest, ytrain, ytest
# Load the 10 percent data

df_10_pct = pd.read_csv('/kaggle/input/kdd-10-data/kddcup.data_10_percent_corrected', header=None, names=feature_names)



# Trim off the last character of the traffic_type, they're all periods

df_10_pct['traffic_type'] = df_10_pct.traffic_type.str[:-1]
# Encode the categorical variables from the ten percent data and split into a training and testing set



drop_columns_10 = ['traffic_type']

encoder, X_train_10, X_test_10, y_train_10, y_test_10 = get_train_test(df_10_pct, drop_columns_10)
# Small grid search for good parameters 

from sklearn.model_selection import GridSearchCV



param_grid = {

    'n_estimators': [10, 100],

    'max_depth': [2, 4, None]

}



grid_search = GridSearchCV(rf, param_grid)



grid_search.fit(X_train_10, y_train_10)

grid_search.best_estimator_
rf = grid_search.best_estimator_



# Encode the categorical variables from the full data and split into a training and testing set



drop_columns = ['traffic_type', 'traffic_class']

encoder, X_train, X_test, y_train, y_test = get_train_test(df, drop_columns)



rf.fit(X_train, y_train)

rf_score = rf.score(X_test, y_test)



print("Score for Random Forest:", rf_score)
# This is really high. I want to look at the confusion matrix to see what's going on here. 

from sklearn.metrics import confusion_matrix



y_pred = rf.predict(X_test)

conf_mat = confusion_matrix(y_test, y_pred, labels=encoder.inverse_transform(y_test))

sns.heatmap(conf_mat, annot=True)
conf_mat = confusion_matrix(y_test, y_pred, labels=encoder.inverse_transform(y_test))

plt.figure(figsize, (12,24))

sns.heatmap(conf_mat, annot=True)
# num_root and num_compromised are perfectly correlated. Num_compromised encapsulates num_root, so I'm going to remove num_root from the final calculations.

drop_features = ['num_root']



# count and srv_count are nearly perfectly correlated. I do think the extra information from srv_count (the number of connections to the same service) is

# important. So I'll leave them both. 



# all variables labeled serror_rate are perfectly correlated. I don't think the extra information from the other serror rates is useful, so I'll drop them. 

# The same is true of the rerror_rate variables.

drop_features.extend(['srv_serror_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'srv_rerror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate'])



# dst_host_srv_count and dst_host_same_srv_rate are nearly perfectly correlated. Extra information from knowing the proportions of the different services being

# connected to is important. Leaving them. 



"""# Now convert the categorical data to numeric.



features = df.drop(columns=['traffic_type'])

target = df.traffic_type



encoder = LabelEncoder().fit(target)

enc_target = encoder.transform(target)

enc_features = pd.get_dummies(features)"""
 #X_train, X_test, y_train, y_test = train_test_split(enc_features, enc_target, random_state=7593) 
"""rf = RandomForestClassifier().fit(X_train, y_train)

y_pred = rf.predict(X_test)"""