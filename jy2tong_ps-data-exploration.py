import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from itertools import compress



train_data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')



print('train size: {}'.format(train_data.shape))

print('test size: {}'.format(test_data.shape))



train_id = train_data['id']

test_id = test_data['id']

# check to see if there is any overlap

if (set(train_id) & set(test_id)):

    print('id Overlap')

# remove id from both train and test set

train_data.drop('id',axis=1,inplace=True)

test_data.drop('id',axis=1,inplace=True)



response = train_data['target']



train_data.drop('target',axis=1,inplace=True)

train_data = train_data.replace(-1,np.NaN)

test_data = test_data.replace(-1,np.NaN)

print('train size: {}'.format(train_data.shape))

print('test size: {}'.format(test_data.shape))

print('Count of positive: {}'.format((response==1).sum()))

print('Ratio of positive: {}'.format((response==1).sum()/response.shape[0]))
# include a count of number of NA

tmp = 57 - train_data.count(1)

train_data.insert(train_data.shape[1],column='NA_count',value = tmp)



plt.hist(tmp,bins=50)

plt.show()



tmp = 57 - test_data.count(1)

test_data.insert(test_data.shape[1],column='NA_count',value = tmp)



# include a high NA for those more than 2



tmp = (train_data['NA_count'] > 2)

train_data.insert(train_data.shape[1],column='high_NA',value = tmp)



tmp = (test_data['NA_count'] > 2)

test_data.insert(test_data.shape[1],column='high_NA',value = tmp)
# loosely count the categorical and binary variables

train_cat = []

train_bin = []

train_float = []

for eachCol in train_data.columns:

    astr = eachCol[(len(eachCol)-3):len(eachCol)]

    if (astr == 'cat'):

        train_cat.append(eachCol)

    elif (astr == 'bin'):

        train_bin.append(eachCol)

    else:

        train_float.append(eachCol)



print('There are {} categorical variables'.format(len(train_cat)))

print('There are {} binary variables'.format(len(train_bin)))

print('There are {} other variables'.format(len(train_float)))



# first look at the binary data as ratio

one_count = []

zero_count = []

for each in train_bin:

    one_count.append((train_data[each] == 1).sum())

    zero_count.append((train_data[each] == 0).sum())

    

plt.rcParams["figure.figsize"] = [12,12]

p1 = plt.bar(range(0,len(train_bin)), one_count, color='#d62728')

p2 = plt.bar(range(0,len(train_bin)), zero_count,

             bottom=one_count)



plt.xticks(range(0,len(train_bin)),train_bin,rotation='vertical')

plt.legend((p1,p2),('One','Zero'))

plt.title('Count comparison for binary variables.')

plt.show()
# notice 4 variables with majority 0, we can discard that

train_data.drop(train_bin[4:8],axis=1,inplace=True)

test_data.drop(train_bin[4:8],axis=1,inplace=True)

del train_bin[4:8]

print('Train data size after dropping: {}'.format(train_data.shape))

print('Test data size after dropping: {}'.format(test_data.shape))



# remove 'ps_car_05_cat'

# remove 'ps_car_08_cat'

# remove 'ps_car_10_cat

tmp = ['ps_car_05_cat','ps_car_08_cat','ps_car_10_cat']

train_data.drop(tmp,axis=1,inplace=True)

test_data.drop(tmp,axis=1,inplace=True)

print('Train data size before dropping: {}'.format(train_data.shape))

print('Test data size before dropping: {}'.format(test_data.shape))
# Hypothesize that ps_car_15 is the age of the car

tmp = pd.Series(train_data['ps_car_15']**(2.0))



#plt.hist(tmp,bins=50)

#plt.show()

tmp = np.int_(tmp)

acount = [0] * 15

for nn in range(0,15):

    acount[nn] = sum(list(compress(response,(tmp==nn))))

    

#plt.bar(range(-1,14),acount)

#plt.show()



# age >= 6

old_car = tmp >= 6

train_data.insert(train_data.columns.get_loc('ps_car_15'),column = 'old_car',value = old_car)



tmp = pd.Series(test_data['ps_car_15']**(2.0))

tmp = np.int_(tmp)

old_car = tmp >= 6

test_data.insert(test_data.columns.get_loc('ps_car_15'),column = 'old_car',value = old_car)
miss_data = []

for each in train_data.columns:

    if(train_data[each].isnull().sum()>0):

        miss_data.append(each)



print('There are total {} columns with NA in train_data.'.format(len(miss_data)))

train_missInv = {}

for each in miss_data:

    train_missInv[each] = (train_data[each].isnull().sum())/train_data.shape[0]

    

for each in miss_data:

    print(each," ",train_missInv[each])

    

# For categorical variable

# Those with less than 1% missing, replace with mode



# missing 'ps_ind_02_cat' is small percentage of the entire data, so replace with mode

# train_cat[0] = 'ps_ind_02_cat'

tmp = train_data['ps_ind_02_cat'].isnull()

train_data.insert(train_data.columns.get_loc('ps_ind_02_cat'),column='ind_02_cat_NA',value=tmp)

train_data['ps_ind_02_cat'].fillna(train_data['ps_ind_02_cat'].mode()[0],inplace=True)



# 'ps_ind_04_cat'

tmp = train_data['ps_ind_04_cat'].isnull()

train_data.insert(train_data.columns.get_loc('ps_ind_04_cat'),column='ind_04_cat_NA',value=tmp)

train_data['ps_ind_04_cat'].fillna(train_data['ps_ind_04_cat'].mode()[0],inplace=True)



# 'ps_ind_05_cat'

tmp = train_data['ps_ind_05_cat'].isnull()

train_data.insert(train_data.columns.get_loc('ps_ind_05_cat'),column='ind_05_cat_NA',value=tmp)

train_data['ps_ind_05_cat'].fillna(train_data['ps_ind_05_cat'].mode()[0],inplace=True)



# 'ps_car_09_cat'

tmp = train_data['ps_car_09_cat'].isnull()

train_data.insert(train_data.columns.get_loc('ps_car_09_cat'),column='car_09_cat_NA',value=tmp)

train_data['ps_car_09_cat'].fillna(train_data['ps_car_09_cat'].mode()[0],inplace=True)



# 'ps_car_01_cat'

tmp = train_data['ps_car_01_cat'].isnull()

train_data.insert(train_data.columns.get_loc('ps_car_01_cat'),column='car_01_cat_NA',value=tmp)

train_data['ps_car_01_cat'].fillna(train_data['ps_car_01_cat'].mode()[0],inplace=True)



# 'ps_car_02_cat

tmp = train_data['ps_car_02_cat'].isnull()

train_data.insert(train_data.columns.get_loc('ps_car_02_cat'),column='car_02_cat_NA',value=tmp)

train_data['ps_car_02_cat'].fillna(train_data['ps_car_02_cat'].mode()[0],inplace=True)



miss_data = []

for each in train_data.columns:

    if(train_data[each].isnull().sum()>0):

        miss_data.append(each)



print('There are total {} columns with NA in train_data.'.format(len(miss_data)))

miss_data = []

for each in test_data.columns:

    if(test_data[each].isnull().sum()>0):

        miss_data.append(each)



print('There are total {} columns with NA in test_data.'.format(len(miss_data)))



train_missInv = {}

for each in miss_data:

    train_missInv[each] = (test_data[each].isnull().sum())/test_data.shape[0]

    

for each in miss_data:

    print(each," ",train_missInv[each])



# Categorical variables with less than 1% missing, fill with mode

tmp = test_data['ps_ind_02_cat'].isnull()

test_data.insert(test_data.columns.get_loc('ps_ind_02_cat'),column='ind_02_cat_NA',value=tmp)

test_data['ps_ind_02_cat'].fillna(test_data['ps_ind_02_cat'].mode()[0],inplace=True)



tmp = test_data['ps_ind_04_cat'].isnull()

test_data.insert(test_data.columns.get_loc('ps_ind_04_cat'),column='ind_04_cat_NA',value=tmp)

test_data['ps_ind_04_cat'].fillna(test_data['ps_ind_04_cat'].mode()[0],inplace=True)



tmp = test_data['ps_ind_05_cat'].isnull()

test_data.insert(test_data.columns.get_loc('ps_ind_05_cat'),column='ind_05_cat_NA',value=tmp)

test_data['ps_ind_05_cat'].fillna(test_data['ps_ind_05_cat'].mode()[0],inplace=True)



tmp = test_data['ps_car_01_cat'].isnull()

test_data.insert(test_data.columns.get_loc('ps_car_01_cat'),column='car_01_cat_NA',value=tmp)

test_data['ps_car_01_cat'].fillna(test_data['ps_car_01_cat'].mode()[0],inplace=True)



tmp = test_data['ps_car_02_cat'].isnull()

test_data.insert(test_data.columns.get_loc('ps_car_02_cat'),column='car_02_cat_NA',value=tmp)

test_data['ps_car_02_cat'].fillna(test_data['ps_ind_02_cat'].mode()[0],inplace=True)



tmp = test_data['ps_car_09_cat'].isnull()

test_data.insert(test_data.columns.get_loc('ps_car_09_cat'),column='car_09_cat_NA',value=tmp)

test_data['ps_car_09_cat'].fillna(test_data['ps_car_09_cat'].mode()[0],inplace=True)



miss_data = []

for each in test_data.columns:

    if(test_data[each].isnull().sum()>0):

        miss_data.append(each)



print('There are total {} columns with NA in test_data.'.format(len(miss_data)))

# now we deal with 'ps_car_03_cat','ps_car_05_cat','ps_car_07_cat'



# insert dummy variable so to indicate the missing

tmp = train_data['ps_car_03_cat'].isnull()

train_data.insert(train_data.columns.get_loc('ps_car_03_cat'),column = 'car_03_cat_NA', value = tmp)



# insert dummy variable so to indicate the missing

tmp = train_data['ps_car_07_cat'].isnull()

train_data.insert(train_data.columns.get_loc('ps_car_07_cat'),column = 'car_07_cat_NA', value = tmp)



# for test data

# insert dummy variable so to indicate the missing

tmp = test_data['ps_car_03_cat'].isnull()

test_data.insert(test_data.columns.get_loc('ps_car_03_cat'),column = 'car_03_cat_NA', value = tmp)



# insert dummy variable so to indicate the missing

tmp = test_data['ps_car_07_cat'].isnull()

test_data.insert(test_data.columns.get_loc('ps_car_07_cat'),column = 'car_07_cat_NA', value = tmp)



# filling in the missing data for both test and train

prob = [0,0]

tmp = pd.concat([train_data['ps_car_03_cat'],test_data['ps_car_03_cat']],ignore_index=True)

tmp1 = pd.Series(list(compress(tmp,(tmp.isnull() == False))))

prob = [(tmp1==0).sum()/len(tmp1),(tmp1==1).sum()/len(tmp1)]    

tmp = pd.Series(np.random.choice([0,1],size = len(train_data['ps_car_03_cat']),p=prob))

train_data['ps_car_03_cat'].fillna(tmp,inplace=True)

tmp = pd.Series(np.random.choice([0,1],size = len(test_data['ps_car_03_cat']),p=prob))

test_data['ps_car_03_cat'].fillna(tmp,inplace=True)



# filling in the missing data for both test and train

tmp = pd.concat([train_data['ps_car_07_cat'],test_data['ps_car_07_cat']],ignore_index=True)

tmp1 = pd.Series(list(compress(tmp,(tmp.isnull() == False))))

prob = [(tmp1==0).sum()/len(tmp1),(tmp1==1).sum()/len(tmp1)]    

tmp = pd.Series(np.random.choice([0,1],size = len(train_data['ps_car_07_cat']),p=prob))

train_data['ps_car_07_cat'].fillna(tmp,inplace=True)

tmp = pd.Series(np.random.choice([0,1],size = len(test_data['ps_car_07_cat']),p=prob))

test_data['ps_car_07_cat'].fillna(tmp,inplace=True)



print('Done with the missing data treatment for categorical feature')



miss_data = []

for each in test_data.columns:

    if(test_data[each].isnull().sum()>0):

        miss_data.append(each)



print('There are total {} columns with NA in test_data.'.format(len(miss_data)))



miss_data = []

for each in train_data.columns:

    if(train_data[each].isnull().sum()>0):

        miss_data.append(each)



print('There are total {} columns with NA in train_data.'.format(len(miss_data)))

# car_11 and car_12 has small number of missing-ness



# car_11

tmp = train_data['ps_car_11'].isnull()

train_data.insert(train_data.columns.get_loc('ps_car_11'),column = 'car_11_NA',value = tmp)

train_data['ps_car_11'].fillna(train_data['ps_car_11'].mode()[0],inplace=True)



tmp = test_data['ps_car_11'].isnull()

test_data.insert(test_data.columns.get_loc('ps_car_11'),column = 'car_11_NA',value = tmp)

test_data['ps_car_11'].fillna(test_data['ps_car_11'].mode()[0],inplace=True)



# car_12

train_data['ps_car_12'].fillna(train_data['ps_car_12'].mean(),inplace=True)



miss_data = []

for each in test_data.columns:

    if(test_data[each].isnull().sum()>0):

        miss_data.append(each)



print('There are total {} columns with NA in test_data.'.format(len(miss_data)))



for each in miss_data:

    print(each,': ',test_data[each].isnull().sum()/test_data.shape[0])



    

miss_data = []

for each in train_data.columns:

    if(train_data[each].isnull().sum()>0):

        miss_data.append(each)



print('There are total {} columns with NA in train_data.'.format(len(miss_data)))



for each in miss_data:

    print(each,': ',train_data[each].isnull().sum()/train_data.shape[0])

tmp = train_data['ps_car_14'].isnull()

train_data.insert(train_data.columns.get_loc('ps_car_14'),column = 'car_14_NA', value = tmp)



tmp = test_data['ps_car_14'].isnull()

test_data.insert(test_data.columns.get_loc('ps_car_14'),column = 'car_14_NA', value = tmp)



tmp = pd.concat([train_data['ps_car_14'],test_data['ps_car_14']],ignore_index=True)

tmp1 = pd.DataFrame(list(compress(tmp,(tmp.isnull() == False))))

prob[0] = tmp1.mean()

prob[1] = tmp1.var()**(0.5)

tmp = pd.Series(np.random.normal(prob[0],prob[1],size = len(train_data['ps_car_14'])))

train_data['ps_car_14'].fillna(tmp,inplace=True)

tmp = pd.Series(np.random.normal(prob[0],prob[1],size = len(test_data['ps_car_14'])))

test_data['ps_car_14'].fillna(tmp,inplace=True)



# This is a temporary solution, may consider:

# removing outlier
tmp = train_data['ps_reg_03'].isnull()

train_data.insert(train_data.columns.get_loc('ps_reg_03'),column = 'reg_03_NA', value = tmp)



tmp = test_data['ps_reg_03'].isnull()

test_data.insert(test_data.columns.get_loc('ps_reg_03'),column = 'reg_03_NA', value = tmp)



tmp = pd.concat([train_data['ps_reg_03'],test_data['ps_reg_03']],ignore_index=True)

tmp1 = pd.DataFrame(list(compress(tmp,(tmp.isnull() == False))))

prob = [0,0]

prob[0] = tmp1.mean()

prob[1] = tmp1.var()**(0.5)

tmp = pd.Series(np.random.normal(prob[0],prob[1],size = len(train_data['ps_reg_03'])))

train_data['ps_reg_03'].fillna(tmp,inplace=True)

tmp = pd.Series(np.random.normal(prob[0],prob[1],size = len(test_data['ps_reg_03'])))

test_data['ps_reg_03'].fillna(tmp,inplace=True)
train_calc = []

for each in train_data.columns:

    astr = each[0:7]

    if (astr == 'ps_calc'):

        train_calc.append(each)



tmp = (train_data['ps_ind_03'])**0.5

plt.hist(tmp,bins=100)

plt.show()



plotNum = 1

for each in train_calc:

    plt.subplot(5,4,plotNum)

    tmp = ((train_data['ps_ind_03'])**0.5) * train_data[each]

    plt.hist(tmp,bins=100)

    plt.title(each)

    plotNum += 1

    

plt.show()
# this is me attempting converting the actual features into something 

# that resembles an actual density



# ('ps_car_14' x 'ps_calc_10)**0.5

tmp = (train_data['ps_car_14'] * train_data['ps_calc_10'])**0.5

train_data.insert(train_data.columns.get_loc('ps_car_14'),column = 'car_14_feat',value = tmp)

tmp = test_data['ps_car_14'] * test_data['ps_calc_10']

test_data.insert(test_data.columns.get_loc('ps_car_14'),column = 'car_14_feat',value = tmp)



# 'ps_ind_15'

# 'ps_ind_01'

# (ind_03 x car_13)

tmp = train_data['ps_ind_03'] * train_data['ps_car_13']

train_data.insert(train_data.columns.get_loc('ps_car_13'),column = 'car_13_featx',value = tmp)

tmp = test_data['ps_ind_03'] * test_data['ps_car_13']

test_data.insert(test_data.columns.get_loc('ps_car_13'),column = 'car_13_featx',value = tmp)



# 'ps_ind_03' x 'ps_calc_03'

#tmp = ((train_data['ps_ind_03'])**0.5) * train_data['ps_calc_04']

#train_data.insert(train_data.columns.get_loc('ps_ind_03'),column = 'ind_03_feat',value = tmp)

#tmp = ((test_data['ps_ind_03'])**0.5) * test_data['ps_calc_04']

#test_data.insert(test_data.columns.get_loc('ps_ind_03'),column = 'ind_03_feat',value = tmp)



# ('ps_car_13' x 'ps_calc_06)**0.5

tmp = (train_data['ps_car_13'] * train_data['ps_calc_06'])**0.5

train_data.insert(train_data.columns.get_loc('ps_car_13'),column = 'car_13_feat',value = tmp)

tmp = (test_data['ps_car_13'] * test_data['ps_calc_06'])**0.5

test_data.insert(test_data.columns.get_loc('ps_car_13'),column = 'car_13_feat',value = tmp)



# log ('ps_reg_03 x 'ps_calc_11)

tmp = train_data['ps_reg_03'] * train_data['ps_calc_11']

train_data.insert(train_data.columns.get_loc('ps_reg_03'),column = 'reg_03_feat',value = tmp)

tmp = test_data['ps_reg_03'] * test_data['ps_calc_11']

test_data.insert(test_data.columns.get_loc('ps_reg_03'),column = 'reg_03_feat',value = tmp)



print('features done')
train_calc = []

for each in train_data.columns:

    astr = each[0:7]

    if (astr == 'ps_calc'):

        train_calc.append(each)



train_data.drop(train_calc,axis=1,inplace=True)

test_data.drop(train_calc,axis=1,inplace=True)

print('Dropped all calc.')

print(train_data.shape)

print(train_data.shape)



plt.imshow(train_data.corr())

plt.yticks(range(0,len(train_data.columns)),train_data.columns)

plt.xticks(range(0,len(train_data.columns)),train_data.columns,rotation = 'vertical')

plt.colorbar()

plt.show()
train_data.insert(0,column='target',value = response)

train_data.insert(0,column = 'id',value = train_id)

test_data.insert(0,column = 'id',value =test_id)



train_data.to_csv('train_clean.csv',index=False)

test_data.to_csv('test_clean.csv',index=False)