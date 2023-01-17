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
# import libraries

import numpy as np 

import pandas as pd 

from scipy import stats

from scipy.stats import norm, skew

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

sns.set_style('darkgrid')

pd.set_option('display.float_format', lambda x: '{:.2f}'.format(x)) #Limiting floats output to 2 decimal points



from fastai.tabular import *
test = pd.read_csv('../input/TTiDS20/test_no_target.csv')

train = pd.read_csv('../input/TTiDS20/train.csv')

#Save the 'Id' column

train_ID = train['Unnamed: 0']

test_ID = test['Unnamed: 0']



#Now drop the  'Id' colum since it's unnecessary for  the prediction process.

train.drop('Unnamed: 0', axis = 1, inplace = True)

test.drop('Unnamed: 0', axis = 1, inplace = True)
sns.distplot(train['price']) ;



# Get the fitted parameters 

(mu, sigma) = norm.fit(train['price'])



#Now plot the distribution

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')

plt.ylabel('Frequency')

plt.title('price distribution');
# log-transform the price

train['price'] = train['price'].apply(np.log)



sns.distplot(train['price'])

plt.show();
ntrain = train.shape[0]

ntest = test.shape[0]



y_train = train.price.values.copy()

all_data = pd.concat((train, test), sort = 'True').reset_index(drop=True)

all_data.drop(['price'], axis=1, inplace=True)

all_data.drop(['engine_capacity'], axis=1, inplace=True)



all_data.drop(['zipcode'], axis=1, inplace=True)



print("all_data size is : {}".format(all_data.shape))
missing_data = (all_data.isnull().sum() / len(all_data)).sort_values(ascending = False)*100



missing_data = pd.DataFrame(missing_data)



plt.figure(figsize = (10,7))

missing_data.head(20).plot(kind = 'bar')

plt.title('Percent of missing data');
col = train.corr().nlargest(10, 'price')['price'].index

corr_matrix = np.corrcoef(train[col].values.T)
plt.figure(figsize = (10,8))

sns.heatmap(corr_matrix, cmap = 'coolwarm', annot = True, xticklabels= col.values, yticklabels= col.values);
ntrain = train.shape[0]

ntest = test.shape[0]



y_train = train.price.values.copy()

all_data = pd.concat((train, test), sort = 'True').reset_index(drop=True)

all_data.drop(['price'], axis=1, inplace=True)

all_data.drop(['model'], axis=1, inplace=True)

all_data.drop(['gearbox'], axis=1, inplace=True)

all_data.drop(['fuel'], axis=1, inplace=True)

all_data.drop(['mileage'], axis=1, inplace=True)









print("all_data size is : {}".format(all_data.shape))
missing_data = (all_data.isnull().sum() / len(all_data)).sort_values(ascending = False)*100



missing_data = pd.DataFrame(missing_data)

missing_data.head(5)
all_data
all_data['registration_year'] = all_data['registration_year'].astype(str) 

all_data['type'] = all_data['type'].astype(str) 

all_data['brand'] = all_data['brand'].astype(str) 

numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index



# Check the skew of all numerical features

skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

print("\nSkew in numerical features: \n")

skewness = pd.DataFrame({'Skew' :skewed_feats})

skewness.head(10)
skewness = skewness[abs(skewness) > 0.75]

print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))



from scipy.special import boxcox1p

skewed_features = skewness.index

lam = 0.15

for feat in skewed_features:

    all_data[feat] = boxcox1p(all_data[feat], lam)
train = all_data[:ntrain]

test = all_data[ntrain:] 
TARGET = 'price'
cat_names = list(train.select_dtypes(include = ['object', 'bool']).columns)
cont_names = list(train.select_dtypes(exclude = ['object', 'bool']).columns)
# Add back sale prices

train['price'] = y_train

# defining steps to process the input data

procs = [FillMissing, Categorify, Normalize]



# Test tabularlist

test = TabularList.from_df(test, cat_names=cat_names, cont_names=cont_names, procs=procs)
data = (TabularList.from_df(train, path='.', cat_names=cat_names, cont_names=cont_names, procs=procs)

                        .split_by_rand_pct(valid_pct = 0.2)

                        .label_from_df(cols = TARGET, label_cls = FloatList, log = False )

                        .add_test(test)

                        .databunch(bs = 128))
data.show_batch(rows=5, ds_type=DatasetType.Valid)
max_log_y = (np.max(train['price'])*1.2)

y_range = torch.tensor([0, max_log_y], device=defaults.device)
def mean_absolute_percentage_error(y_true:Tensor, y_pred:Tensor):

    y_true, y_pred = torch.from_numpy(np.array(y_true)), torch.from_numpy(np.array(y_pred))

    return (torch.abs((y_true - y_pred) / y_true) * 100).mean()
# create the model

learn = tabular_learner(data, layers=[800,200], ps=[0.001,0.01], y_range = y_range, emb_drop=0.04, metrics=mean_absolute_percentage_error)



learn.model



# select the appropriate learning rate

learn.lr_find()



# we typically find the point where the slope is steepest

learn.recorder.plot()
# Fit the model based on selected learning rate

learn.fit_one_cycle(30, max_lr =1e-03)
#Plotting The losses for training and validation

learn.recorder.plot_losses(skip_start = 500)
# get predictions

preds, targets = learn.get_preds(DatasetType.Test)

labels = [np.exp(p[0].data.item()) for p in preds]



# create submission file to submit in Kaggle competition

submission = pd.DataFrame({'Id': test_ID, 'Predicted': labels})

submission.to_csv('submission.csv', index=False)



submission.describe()
submission.head()