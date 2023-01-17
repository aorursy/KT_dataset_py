# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/diplomabigdataset/20200120_purchase_base.csv")

data.drop(['Unnamed: 0', 'creation_datetime', 'payment_type',

       'tree_level_1', 'tree_level_2', 'tree_level_3', 'tree_level_5', 'buyer_price', 'item_count', 'channel', 'location_name',

       'region_name', 'fd_name'], axis=1, inplace=True)

data.dropna(inplace=True)

#data = data.head(1000000)

diftov = data['mrid'].unique().tolist()

rantov = range(len(pd.unique(data['mrid'])))

TS1 = pd.DataFrame({'num':rantov, 'goo':diftov})

data['mrid'] = data['mrid'].map(TS1.set_index('goo')['num'])

aprorders = data.groupby('mrid')['tree_level_4'].apply(list)

#aprorders = data.groupby('tree_level_4')['mrid'].apply(list)

aprorders = aprorders.tolist()

aprorders = aprorders[:1000000]

del data

del diftov

del rantov

del TS1
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(tokenizer=lambda x: x, lowercase=False)

m = cv.fit_transform(aprorders)

goods_name = cv.get_feature_names()

del cv

m = (m > 0)

m = (m * 1)

df2 = m.todense()

del m

train_valid = []

for i in range(int(0.7 * df2.shape[0])):

    train_valid.append(df2[i].nonzero()[1].tolist()) 

goods_sum = np.asarray(df2[:int(0.7 * df2.shape[0])].sum(axis=0)).flatten()

dictionary = dict(zip(np.flip(np.argsort(goods_sum)), sorted(goods_sum, reverse=True)))

with open('sums_with_inds.txt', 'w') as f:

    print(dictionary, file=f)

sorted_goods_sum_inds = np.flip(np.argsort(goods_sum))

sorted_goods_sum_inds = sorted_goods_sum_inds.tolist()

bad_check = sorted_goods_sum_inds[:4]

sorted_goods_sum_inds = [e for e in sorted_goods_sum_inds if e not in bad_check]
from collections import Counter

from itertools import chain, combinations

set_train_valid = train_valid.copy()

for i in range(len(set_train_valid)):

    set_train_valid[i] = sorted(set_train_valid[i])

meets = Counter(chain.from_iterable(combinations(line, 2) for line in set_train_valid)) 



sum_counters = np.zeros((df2.shape[1], df2.shape[1]))

for i in range(df2.shape[1]):

    for j in range(i+1, df2.shape[1]):

        sum_counters[i][j] = meets[(i,j)]

        sum_counters[j][i] = meets[(i,j)]



def arsort(seq):

    return sorted(range(len(seq)), key=seq.__getitem__)
import random

from collections import defaultdict

# здесь и далее 0-ой метод - суммы, 2-ой - константы

answers, answers_true = [], [] # предсказания и истинное присутствие/отсутсвие для суммы

answers2, answers2_true = [], [] # предсказания и истинное присутствие/отсутсвие для кон-ты

an_0_all, an_2_all = [], [] # списки предсказаний для настоящих товаров в чеках

for j in range(int(0.7 * df2.shape[0]), int(1.0 * df2.shape[0])): # идем по последним 30% чеков

    all_ones = df2[j].nonzero()[1].tolist() # получили для чека все номера товаров, которые в нем есть 

    all_ones = [e for e in all_ones if e not in bad_check] # убрали 4 самых популярных товара

    if len(all_ones) > 4: # если в чеке есть хотя бы 5 товаров, не считая 4 самых популярных:

        an_0, an_2 = [], []

        abc = [all_ones[i] for i in arsort([sorted_goods_sum_inds.index(i) for i in range(0,len(sorted_goods_sum_inds)) if i in all_ones])[:3]] 

        # abc - три самых популярных товара, по которым идет оценка

        kicked_elems = [x for x in all_ones if x not in abc] # kicked_elems - товары, которые хотим предсказать

        all_ones = abc.copy()

        psbl_inds = np.count_nonzero(sum_counters[all_ones].sum(axis=0)) # для суммы узнаем, сколько товаров имеет 

        #ненулевую оценку, чтобы потом "обнулить" порядки для товаров с нулевой оценкой

        fq = np.argsort(sum_counters[all_ones].sum(axis=0))[::-1].tolist() #оценка по сумме

        fq = [e for e in fq if e not in bad_check] # выкинули 4 самых популярных товара

        for kicked_elem in kicked_elems:

            if kicked_elem in fq: # на самом деле бессмысленная проверка

                an_0.append(fq.index(kicked_elem))

            else:

                an_0.append(df2.shape[1])

        for kicked_elem in kicked_elems:

            if kicked_elem in sorted_goods_sum_inds: # проверка на то, встречался ли вообще товар в обучающей выборке

                an_2.append(sorted_goods_sum_inds.index(kicked_elem))

            else:

                an_2.append(df2.shape[1]) # если не встретился, то записываем худший порядок - его счетчик для кон-ты = 0

        an_0_all.append(an_0)

        an_2_all.append(an_2)

        

        #теперь выбираем товары для последующей оценки роков

        y_true = [0]*(df2.shape[1]+1)

        replacements = [1]*len(an_0)

        for (index, replacement) in zip(an_0, replacements):

            y_true[index] = replacement

        # y_true - массив, где стоят нули на всех местах, кроме тех порядков, на которых находятся предсказываемые нами т-ры 

        deck = list(range(0, len(y_true)))

        random.shuffle(deck)

        iindices = deck[:500] #случайно выбрали 500 индексов

        y_pred = list(range(len(y_true)))

        inds = list(range(psbl_inds+1, len(y_true)))

        replacements = [df2.shape[1]]*len(inds)

        for (index, replacement) in zip(inds, replacements):

            y_pred[index] = replacement

        # y_pred - массив, в котором (общее число товаров) элементов,

        # где находятся сначала числа от 0 до (сколько товаров имеет оценку по сумме), дальше - худший возможный порядок

        answers_true.append([y_true[i] for i in iindices])

        answers.append([y_pred[i] for i in iindices])

        

        # все то же самое - для констант

        y_true = [0]*(df2.shape[1]+1)

        replacements = [1]*len(an_2)

        for (index, replacement) in zip(an_2, replacements):

            y_true[index] = replacement

        deck = list(range(0, len(y_true)))

        random.shuffle(deck)

        iindices = deck[:500]

        y_pred = list(range(len(y_true)))

        #inds = list(range(psbl_inds+1, len(y_true)))

        #replacements = [psbl_inds]*len(inds)

        #for (index, replacement) in zip(inds, replacements):

        #    y_pred[index] = replacement

        answers2_true.append([y_true[i] for i in iindices])

        answers2.append([y_pred[i] for i in iindices])
sum_preds = np.array(answers)

sum_trues = np.array(answers_true)

preds_c = np.array(answers2)

trues_c = np.array(answers2_true)

import sklearn.metrics as metrics

fpr, tpr, threshold = metrics.roc_curve(sum_trues.flatten(), (-1)*sum_preds.flatten()) # не забыли умножить порядки на (-1)

roc_auc = metrics.auc(fpr, tpr)

fpr_c, tpr_c, threshold_c = metrics.roc_curve(trues_c.flatten(), (-1)*preds_c.flatten())

roc_auc_c = metrics.auc(fpr_c, tpr_c)



import matplotlib.pyplot as plt

plt.title('First Data ROC')

plt.plot(fpr, tpr, 'g', label = 'sum = %0.5f' % roc_auc)

plt.plot(fpr_c, tpr_c, 'r', label = 'const = %0.5f' % roc_auc_c)

plt.legend(loc = 'lower right')

plt.xlim([0, 1])

plt.ylim([0, 1])

#plt.ylabel('True Positive Rate')

#plt.xlabel('False Positive Rate')

plt.savefig('23032020-rocs.png')

plt.show()
with open('00_roc_preds.txt', 'w') as f:

    for item in answers:

        f.write("%s\n" % item)

with open('00_roc_trues.txt', 'w') as f:

    for item in answers_true:

        f.write("%s\n" % item)

with open('0_right_ans.txt', 'w') as f:

    for item in an_0_all:

        f.write("%s\n" % item)

with open('2_right_ans.txt', 'w') as f:

    for item in an_2_all:

        f.write("%s\n" % item)

with open('02_roc_preds.txt', 'w') as f:

    for item in answers2:

        f.write("%s\n" % item)

with open('02_roc_trues.txt', 'w') as f:

    for item in answers2_true:

        f.write("%s\n" % item)