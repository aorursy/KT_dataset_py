# Import packages

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

plt.style.use('ggplot')

%matplotlib inline

import seaborn as sns
# Load train and test sets

train = pd.read_csv("/kaggle/input/learn-together/train.csv")

test = pd.read_csv("/kaggle/input/learn-together/test.csv")



# Check that Ids are unique and distinct between train and test set

print('Id in train set is unique.') if train.Id.nunique() == train.shape[0] else print('Id is not unique in train set')

print('Train and test sets are distinct.') if len(np.intersect1d(train.Id.values, test.Id.values))== 0 else print('Ids in train and test overlap')
def compare_datasets(feature):

    fig, ax = plt.subplots(figsize=(6,4))

    sns.distplot(test[feature], ax=ax, kde=True, hist=False, label='test', kde_kws={'color': 'b', 'lw': 2})

    sns.distplot(train[feature], ax=ax, kde=True, hist=False, label='train', kde_kws={'color': 'g', 'lw': 2})

    plt.title('Comparison of ' + feature + ' in train and test set')

    plt.legend();



interval_cols = ['Elevation',

 'Aspect',

 'Slope',

 'Horizontal_Distance_To_Hydrology',

 'Vertical_Distance_To_Hydrology',

 'Horizontal_Distance_To_Roadways',

 'Hillshade_9am',

 'Hillshade_Noon',

 'Hillshade_3pm',

 'Horizontal_Distance_To_Fire_Points']



for c in interval_cols:

    compare_datasets(c)
def compare_bools_traintest(feature_list):

    train_feat = train[feature_list].mean()

    train_feat.name = 'Train'

    test_feat = test[feature_list].mean()

    test_feat.name = 'Test'

    print(pd.concat([train_feat, test_feat], axis=1))



wilderness_cols = [c for c in train.columns if c.startswith('Wilderness')]

soil_cols = [c for c in train.columns if c.startswith('Soil')]



compare_bools_traintest(wilderness_cols)

compare_bools_traintest(soil_cols)