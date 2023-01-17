import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import matplotlib.pylab as pylab

import seaborn as sns
def Weight(mType):

    """ From https://www.kaggle.com/mchirico/santas-uncertain-bags/santa-quick-look"""

    if mType == "horse":

        return max(0, np.random.normal(5,2,1)[0])

    elif mType == "ball":

        return max(0, 1 + np.random.normal(1,0.3,1)[0])

    elif mType == "bike":

        return max(0, np.random.normal(20,10,1)[0])

    elif mType == "train":

        return max(0, np.random.normal(10,5,1)[0])

    elif mType == "coal":

        return 47 * np.random.beta(0.5,0.5,1)[0]

    elif mType == "book":

        return np.random.chisquare(2,1)[0]

    elif mType == "doll":

        return np.random.gamma(5,1,1)[0]

    elif mType == "blocks":

        return np.random.triangular(5,10,20,1)[0]

    elif mType == "gloves":

        return 3.0 + np.random.rand(1)[0] if np.random.rand(1) < 0.3 else np.random.rand(1)[0]

    else:

        print("Wrong argument!")
n_iter = 10

presents = ['horse', 'ball', 'bike', 'train', 'coal', 'book', 'doll', 'blocks', 'gloves']

n_presents = 100

p = np.empty([len(presents), n_presents*n_iter])



pylab.rcParams['figure.figsize'] = 9, 9

c = sns.color_palette("hls", 9)

fig = plt.figure()

for i, present in enumerate(presents):

    ax = fig.add_subplot(3, 3, i+1)

    ax.set_title(present)

    p[i] = [Weight(presents[i]) for x in range(0, n_presents*n_iter)] # int(n_presents[i])*n_iter)]

    print('Average weight of ' + present + ' is ' + str(round(np.mean(p[i]),3)) + ' +/- ' \

          + str(round(np.std(p[i]),3)) + '. Min = ' + str(round(np.min(p[i]),3)) + ', max = ' \

          + str(round(np.max(p[i]),3)))

    g = sns.distplot(p[i], color=c[i]);

    g.set(xlim=(-2, 52))
n_present_counts = [1000, 1100, 500, 1000, 166, 1200, 1000, 1000, 200]

n_iter = 1000

n_present_weights = np.zeros(n_iter)



for iter_i in range(n_iter):

    for i, count in enumerate(n_present_counts):

        n_present_weights[iter_i] += np.sum([Weight(presents[i]) for x in range(count)])

        

pylab.rcParams['figure.figsize'] = 9, 9

fig = plt.figure()

g = sns.distplot(n_present_weights);

g.set(title='Distribution of summary weights', xlabel='Weight of all presents')
n_iter = 50



pylab.rcParams['figure.figsize'] = 9, 9

c = sns.color_palette("hls", 9)

fig = plt.figure()

for ex_i, present in enumerate(presents):

    ax = fig.add_subplot(3, 3, ex_i+1)

    ax.set_title(present)

    

    n_present_weights = np.zeros(n_iter)



    for iter_i in range(n_iter):

        for i, count in enumerate(n_present_counts):

            if i != ex_i:

                n_present_weights[iter_i] += np.sum([Weight(presents[i]) for x in range(count)])

        n_present_weights[iter_i] /= 10000. # Just to make plots look better

    g = sns.distplot(n_present_weights, color=c[ex_i]);

    g.set(xlim=(3.7, 5.3))
pylab.rcParams['figure.figsize'] = 9, 36

c = sns.color_palette("hls", 36)

fig = plt.figure()

subplot_idx = 1

for i, present in enumerate(presents):

    for j in range(i+1, len(presents)):

        ax = fig.add_subplot(12, 3, int(subplot_idx))

        ax.set_title(present + ' + ' + presents[j])

        p_s = p[i] + p[j]

        g = sns.distplot(p_s, color=c[subplot_idx-1]);

        g.set(xlim=(-5, 60))

        subplot_idx += 1 
df = pd.DataFrame(columns=('overweight', 'average_weight', 'std_weight', 'present 0', \

                           'present 1', 'present 2'))

idx = 0

for i, present in enumerate(presents):

    for j in range(i, len(presents)):

        for k in range(j, len(presents)):

            w = p[i] + p[j] + p[k]

            overweight = len(w[w > 50])/len(w)*100 # percents

            average_weight = np.mean(w)

            std_weight = np.std(w)

            df.loc[idx] = [overweight, average_weight, std_weight, presents[i], presents[j], \

                           presents[k]]

            idx += 1

df_sorted = df.sort_values(by=['overweight', 'average_weight'])
df_sorted = df.sort_values(by=['overweight', 'average_weight'])

print(df_sorted[df_sorted.overweight == 0][-10:])
df = pd.DataFrame(columns=('overweight', 'average_weight', 'std_weight', 'present 0', \

                           'present 1', 'present 2', 'present 3'))

idx = 0

for i, present in enumerate(presents):

    for j in range(i, len(presents)):

        for k in range(j, len(presents)):

            for l in range(k, len(presents)):

                w = p[i] + p[j] + p[k] + p[l]

                overweight = len(w[w > 50])/len(w)*100 # percents

                average_weight = np.mean(w)

                std_weight = np.std(w)

                df.loc[idx] = [overweight, average_weight, std_weight, presents[i], \

                               presents[j], presents[k], presents[l]]

                idx += 1
df_sorted = df.sort_values(by=['overweight', 'average_weight'])

print(df_sorted[df_sorted.overweight == 0][-10:])
df = pd.DataFrame(columns=('overweight', 'average_weight', 'std_weight', 'present 0', \

                           'present 1', 'present 2', 'present 3', 'present 4'))

idx = 0

for i, present in enumerate(presents):

    for j in range(i, len(presents)):

        for k in range(j, len(presents)):

            for l in range(k, len(presents)):

                for m in range(l, len(presents)):

                    w = p[i] + p[j] + p[k] + p[l] + p[m]

                    overweight = len(w[w > 50])/len(w)*100 # percents

                    average_weight = np.mean(w)

                    std_weight = np.std(w)

                    df.loc[idx] = [overweight, average_weight, std_weight, presents[i], \

                                   presents[j], presents[k], presents[l], presents[m]]

                    idx += 1
df_sorted = df.sort_values(by=['overweight', 'average_weight'])

print(df_sorted[df_sorted.overweight == 0][-10:])
df = pd.DataFrame(columns=('overweight', 'average_weight', 'std_weight', 'present 0', \

                           'present 1', 'present 2', 'present 3', 'present 4', 'present 5'))

idx = 0

for i, present in enumerate(presents):

    for j in range(i, len(presents)):

        for k in range(j, len(presents)):

            for l in range(k, len(presents)):

                for m in range(l, len(presents)):

                    for n in range(m, len(presents)):

                        w = p[i] + p[j] + p[k] + p[l] + p[m] + p[n]

                        overweight = len(w[w > 50])/len(w)*100 # percents

                        average_weight = np.mean(w)

                        std_weight = np.std(w)

                        df.loc[idx] = [overweight, average_weight, std_weight, presents[i], \

                                       presents[j], presents[k], presents[l], presents[m], \

                                       presents[n]]

                        idx += 1
df_sorted = df.sort_values(by=['overweight', 'average_weight'])

print(df_sorted[df_sorted.overweight == 0][-10:])
df = pd.DataFrame(columns=('overweight', 'average_weight', 'std_weight', 'present 0', \

                           'present 1', 'present 2', 'present 3', 'present 4', 'present 5', \

                           'present 6'))

idx = 0

for i, present in enumerate(presents):

    for j in range(i, len(presents)):

        for k in range(j, len(presents)):

            for l in range(k, len(presents)):

                for m in range(l, len(presents)):

                    for n in range(m, len(presents)):

                        for o in range(n, len(presents)):

                            w = p[i] + p[j] + p[k] + p[l] + p[m] + p[n] + p[o]

                            overweight = len(w[w > 50])/len(w)*100 # percents

                            average_weight = np.mean(w)

                            df.loc[idx] = [overweight, average_weight, std_weight, presents[i], \

                                           presents[j], presents[k], presents[l], presents[m], \

                                           presents[n], presents[o]]

                            idx += 1
df_sorted = df.sort_values(by=['overweight', 'average_weight'])

print(df_sorted[df_sorted.overweight == 0][-10:])
'''

df = pd.DataFrame(columns=('overweight', 'average_weight', 'std_weight', 'present 0', \

                           'present 1', 'present 2', 'present 3', 'present 4', 'present 5', \

                           'present 6', 'present 7'))

idx = 0

for i, present in enumerate(presents):

    for j in range(i, len(presents)):

        for k in range(j, len(presents)):

            for l in range(k, len(presents)):

                for m in range(l, len(presents)):

                    for n in range(m, len(presents)):

                        for o in range(n, len(presents)):

                            for h in range(o, len(presents)):

                                w = p[i] + p[j] + p[k] + p[l] + p[m] + p[n] + p[o] + p[h]

                                overweight = len(w[w > 50])/len(w)*100 # percents

                                average_weight = np.mean(w)

                                df.loc[idx] = [overweight, average_weight, std_weight, presents[i], \

                                               presents[j], presents[k], presents[l], presents[m], \

                                               presents[n], presents[o], presents[h]]

                                idx += 1

'''
'''

df_sorted = df.sort_values(by=['overweight', 'average_weight'])

print(df_sorted[df_sorted.overweight == 0][-10:])

'''