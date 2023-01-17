import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

#Input data files are available in the "../input/" directory.

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/Indian Liver Patient Dataset (ILPD).csv")

print(df.columns) # gives us the names of the features in the dataset that might help predict if patient has a disease.

df.describe()
df.info()
df['alkphos'].fillna(value=0, inplace=True)
data = df.values

feat_names = df.columns

df_neg = df.loc[df[feat_names[-1]] == 1]

df_neg.describe()
df_pos = df.loc[df[feat_names[-1]] == 2]

df_pos.describe()
sns.heatmap(df_neg.corr())

sns.heatmap(df_pos.corr())
sns.heatmap(df.corr())
print(np.array(df_neg.values)[:,2].shape)

print(np.array(df_pos.values)[:,2].shape)

for i in range(2,len(df.columns)):

    sns.distplot((np.array(df_neg.values)[:,i]),color='b')

    sns.distplot((np.array(df_pos.values)[:,i]),color='r')

    plt.figure()
neg_meas = np.array(df_neg.values)[:,2:-2].astype('float')

neg_mean = np.mean(neg_meas, axis=0)

neg_cov  = np.cov(neg_meas, rowvar=0)

neg_precision = np.linalg.inv(neg_cov)



pos_meas = np.array(df_pos.values)[:,2:-2].astype('float')

pos_mean = np.mean(pos_meas, axis=0)

pos_cov  = np.cov(pos_meas, rowvar=0)

pos_precision = np.linalg.inv(pos_cov)



#for i in range(len)

TP = 1

TN = 1

FP = 1

FN = 1

NEG = 0

POS = 0

for i in range(len(df.values)):

    meas = np.array(df.values[i])[2:-2]

    neg_diff = meas - neg_mean

    neg_dist = np.sqrt(np.dot(np.transpose(neg_diff), np.dot(neg_precision, neg_diff)))

    pos_diff = meas - pos_mean

    pos_dist = np.sqrt(np.dot(np.transpose(pos_diff), np.dot(pos_precision, pos_diff)))

    if((pos_dist/neg_dist) < 1):

        pred = 2

    else:

        pred = 1

    if(df.values[i][-1] == 1):

        NEG += 1

        if(pred == df.values[i][-1]):

            TN += 1

        else:

            FP += 1

    else:

        POS += 1

        if(pred == df.values[i][-1]):

            TP += 1

        else:

            FN += 1



conf_matrix = np.array([[TP,FP],[FN,TN]])

sns.heatmap(conf_matrix)

print(conf_matrix)

print(TP+FN+TN+FP)

precision = (TP*1.0)/(TP+FP)

recall    = (TP*1.0)/(TP+FN)

F_score  = (2.0*precision*recall)/(precision+recall)

print(precision)

print(recall)

print('F-Score : ' + str(F_score))

        
def get_fscore(cost):

    TP = 1

    TN = 1

    FP = 1

    FN = 1

    NEG = 0

    POS = 0

    for i in range(len(df.values)):

        meas = np.array(df.values[i])[2:-2]

        neg_diff = meas - neg_mean

        neg_dist = np.sqrt(np.dot(np.transpose(neg_diff), np.dot(neg_precision, neg_diff)))

        pos_diff = meas - pos_mean

        pos_dist = np.sqrt(np.dot(np.transpose(pos_diff), np.dot(pos_precision, pos_diff)))

        if((pos_dist/neg_dist) < cost):

            pred = 2

        else:

            pred = 1

        if(df.values[i][-1] == 1):

            NEG += 1

            if(pred == df.values[i][-1]):

                TN += 1

            else:

                FP += 1

        else:

            POS += 1

            if(pred == df.values[i][-1]):

                TP += 1

            else:

                FN += 1

    precision = (TP*1.0)/(TP+FP)

    recall    = (TP*1.0)/(TP+FN)

    F_score  = (2.0*precision*recall)/(precision+recall)

    return F_score

cost_vec = []

fscore_vec =[]

for i in range(50):

    cost =  i*0.2

    cost_vec.append(cost)

    fscore = get_fscore(cost)

    fscore_vec.append(fscore)

plt.plot(np.array(cost_vec), np.array(fscore_vec))

plt.xlabel('relative cost of error')

plt.ylabel('F-Score')
