# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

print(check_output(["ls", "../input/porto-seguros-safe-driver-noisy-features"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
results = pd.read_csv("../input/porto-seguros-safe-driver-noisy-features/noisy_feature_check_results.csv")
results.sort_values(by="importance_mean", ascending=False, inplace=True)

results.dropna(axis=0, inplace=True)

results.head(10)
good_to_go = []

doubt = []

suspicious = []

rejected = []

for feature in results.feature.unique():

    sha_mean, sha_dev = results.loc[(results["feature"] == feature) 

                                    & (results["process"] == "Shadow"), ["importance_mean", "importance_std"]].values[0]

    id_mean, id_dev = results.loc[(results["feature"] == feature) 

                                    & (results["process"] == "Identity"), ["importance_mean", "importance_std"]].values[0]

    if sha_mean >= id_mean:

        rejected.append((feature, id_mean, sha_mean))

    elif sha_mean + sha_dev >= id_mean:

        suspicious.append((feature, id_mean, sha_mean))

    elif sha_mean + sha_dev >= id_mean - id_dev:

        doubt.append((feature, id_mean, sha_mean))

    else:

        good_to_go.append((feature, id_mean, sha_mean))



print("Good features (%d)" % len(good_to_go))

for f, score, sha in good_to_go:

    print("\t%-20s : %7.2f / shadow %7.2f" % (f, score, sha))

print("Doubts (%d)" % len(doubt))

for f, score, sha in doubt:

    print("\t%-20s : %7.2f / shadow %7.2f" % (f, score, sha))

print("Suspicious features (%d)" % len(suspicious))

for f, score, sha in suspicious:

    print("\t%-20s : %7.2f / shadow %7.2f" % (f, score, sha))

print("Rejected features (%d)" % len(rejected))

for f, score, sha in rejected:

    print("\t%-20s : %7.2f / shadow %7.2f" % (f, score, sha))

        