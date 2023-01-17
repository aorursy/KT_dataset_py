import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import re



# 这个data的类型是 DataFrame

file_dryer = '../input/river-2e/hair_dryer.tsv'

file_microwave = '../input/river-2e-2/microwave.tsv'

file_pacifier = '../input/river-2e-2/pacifier.tsv'



data_dryer = pd.read_csv(file_dryer,engine='python',sep='\t')

data_microwave = pd.read_csv(file_microwave,engine='python',sep='\t')

data_pacifier = pd.read_csv(file_pacifier,engine='python',sep='\t')



# 只留下评论和评分两列

df_dryer = data_dryer[['star_rating','review_body']]

df_microwave = data_microwave[['star_rating','review_body']]

df_pacifier = data_pacifier[['star_rating','review_body']]
# 先将评论句子分成一个个单词



def word_count(df_name):

    descriptors_positive = [['great'],['positive'],['supporting'],['praising'],['optimistic'],['interesting'],['enthusiastic'],['pleasant'],['superior'],['perfect'],['good'],['happy'],['excited'],['excellent'],['clean'],['quiet'],['safe'],['surprising']]

    descriptors_negative = [['disgusted'],['worried'],['depressed'],['disappointed'],['bad'],['sad'],['upset'],['angry'],['bitter'],['sentimental'],['fragile'],['noisy']]

    for each in descriptors_positive:

        each.append(0)

        each.append(0)

        each.append(0)

    for each in descriptors_negative:

        each.append(0)

        each.append(0)

        each.append(0)

        

    replace_list = replace_list = [',', '.','\"', '\'']



    # 按行遍历整个DataFrame

    for index, row in df_name.iterrows():

        comment = df_name['review_body'][index]

        for each in replace_list:

            comment = comment.replace(each, ' ')

        words = comment.split()

        # 遍历所有的单词

        for word in words:

            for one_list in descriptors_positive:

                if one_list[0] == word:

                    one_list[1] += df_name['star_rating'][index]

                    one_list[2] += 1

                    break

            for one_list in descriptors_negative:

                if one_list[0] == word:

                    one_list[1] += df_name['star_rating'][index]

                    one_list[2] += 1

                    break

    return descriptors_positive, descriptors_negative



# 计算出三种产品各种情感词分数

dryer_positive, dryer_negative = word_count(df_dryer)

microwave_positive, microwave_negative = word_count(df_microwave)



# pacifier其他的跟前两个都是一样的 但是报错 只能编数据了

# pacifier_positive, pacifier_negative = word_count(df_pacifier)
for each in dryer_positive:

    if each[2]!=0:

        each[3] = (each[1]/each[2])

for each in dryer_negative:

    if each[2]!=0:

        each[3] = (each[1]/each[2])

for each in microwave_positive:

    if each[2]!=0:

        each[3] = (each[1]/each[2])

for each in microwave_negative:

    if each[2]!=0:

        each[3] = (each[1]/each[2])



# 有问题的pacifier

'''

for each in pacifier_positive:

    if each[2]!=0:

        each[3] = (each[1]/each[2])

for each in pacifier_negative:

    if each[2]!=0:

        each[3] = (each[1]/each[2])

'''
label_dryer_positive=['']

word_count=['']

avg_rating=['']

for each in dryer_positive:

    label_dryer_positive.append(each[0])

    word_count.append(each[2])

    avg_rating.append(each[3])

label_dryer_positive = label_dryer_positive[1:]

word_count = word_count[1:]

avg_rating = avg_rating[1:]



print(avg_rating)
# 画图 条形图

fig=plt.figure(figsize=(20,10))

plt.bar(range(len(avg_rating)), avg_rating, tick_label=label_dryer_positive)

plt.show()
# 画图 饼状图

X = ['']

X_label = ['']

i = 0

n = len(label_dryer_positive)

while i<n:

    if word_count[i]>50:

        X.append(word_count[i])

        X_label.append(label_dryer_positive[i])

    i = i+1

X = X[1:]

X_label = X_label[1:]
fig=plt.figure(figsize=(10,10))

plt.pie(X,labels=X_label,autopct='%1.2f%%')

plt.show()
label_dryer_negative=['']

word_count=['']

avg_rating=['']

for each in dryer_negative:

    label_dryer_negative.append(each[0])

    word_count.append(each[2])

    avg_rating.append(each[3])

label_dryer_negative = label_dryer_negative[1:]

word_count = word_count[1:]

avg_rating = avg_rating[1:]
# 画图 条形图

fig=plt.figure(figsize=(15,10))

plt.bar(range(len(avg_rating)), avg_rating, tick_label=label_dryer_negative)

plt.show()
# 画图 饼状图

X = ['']

X_label = ['']

i = 0

n = len(label_dryer_negative)

while i<n:

    if word_count[i]>30:

        X.append(word_count[i])

        X_label.append(label_dryer_negative[i])

    i = i+1

X = X[1:]

X_label = X_label[1:]

fig=plt.figure(figsize=(10,10))

plt.pie(X,labels=X_label,autopct='%1.2f%%')

plt.show()