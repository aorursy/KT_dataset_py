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
train = pd.read_csv(r"/kaggle/input/guess-my-price/train.tsv", sep = '\t')

test = pd.read_csv(r"/kaggle/input/guess-my-price/test.tsv", sep = '\t')
def data_prep(data, train_or_test = 'train'):

    df_final = pd.DataFrame(data.category_name.str.split('/',expand=True))

    df_final = df_final.add_prefix('category_')

    if train_or_test.lower() == 'train':

        df_final = pd.concat([df_final,data[['item_condition_id','brand_name','price','shipping']]], axis=1).fillna(0)

    else:

        df_final = pd.concat([df_final,data[['item_condition_id','brand_name','shipping']]], axis=1).fillna(0)

    return df_final



train_final = data_prep(train, train_or_test='train')

test_final = data_prep(test, train_or_test='test')
print(train_final.shape)

print(test_final.shape)
test_final[~test_final.brand_name.isin(train_final.brand_name) | 

          ~test_final.category_0.isin(train_final.category_0) | 

          ~test_final.category_1.isin(train_final.category_1) |

          ~test_final.category_2.isin(train_final.category_2) |

          ~test_final.category_3.isin(train_final.category_3) |

          ~test_final.category_4.isin(train_final.category_4) |

          ~test_final.item_condition_id.isin(train_final.item_condition_id)]
train_final_unique = train_final.groupby(['category_0','category_1','category_2','category_3','category_4','item_condition_id','brand_name','shipping']).mean()

train_final_unique = train_final_unique.reset_index()

train_final_unique.head()
train_final_unique.shape
test_final_join = pd.merge(test_final, train_final_unique, on=['category_0','category_1','category_2','category_3','category_4','item_condition_id','brand_name','shipping'], how='left')

test_final_join['price'] = test_final_join['price'].fillna(np.mean(test_final_join['price']))

test_final_join.head()
print(len(test_final_join))

print(len(test))
submission = pd.DataFrame(test['train_id'])

submission['price'] = test_final_join.price

submission.head()
submission.to_csv("submission.csv", index = False)