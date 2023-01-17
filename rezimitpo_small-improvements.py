# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from collections import Counter

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



train['price_doc'].sample(5)
train_prices = train['price_doc'].tolist()

min_p, max_p = min(train_prices) , max(train_prices)

min_p, max_p
test['price_doc'] = pd.Series([np.random.randint(min_p,max_p) for _ in range(len(test))])

compare = test.copy()

test['price_doc'].loc[:10]
Counter(train.price_doc).most_common(250)[:10]
prices = sorted(map(lambda x:x[0],Counter(train.price_doc).most_common(250)))

prices[:10]
for i,j in test.iterrows():

    pp = j.price_doc

    pr = pp

    for p in prices:

        if pp>p:

            pass

        elif pp<p:

            if (p-pp) > (pp-past):

                if pp*0.90 > past:

                    pr = pp

                    past = p

                    break

                pr = past

                past = p

                break

            elif (p-pp) < (pp-past):

                if p*0.90 > pp:

                    pr = pp

                    past = p

                    break

                pr = p

                past = p

                break

            else:

                pr = (past+p)/2

                past = p

                break

        else:

            pr = pp

        past = p

    test.set_value(i,'price_doc',pr)
test.price_doc
compare.price_doc