import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from transliterate import translit, get_available_language_codes
import os


train_data = pd.read_csv("../input/train.csv")
test_data = pd.read_csv("../input/test.csv")

#train_data.head(n=5)
#print(os.listdir("../input/"))

region = (train_data['region']).apply(translit, 'ru', reversed=True)
city = (train_data['city']).apply(translit, 'ru', reversed=True)
parent_category_name = (train_data['parent_category_name']).apply(translit, 'ru', reversed=True)
category_name = (train_data['category_name']).apply(translit, 'ru', reversed=True)
param_1 = train_data['param_1'].apply(lambda row:row if pd.isnull(row) else translit(row, 'ru', reversed=True))
param_2 = train_data['param_2'].apply(lambda row:row if pd.isnull(row) else translit(row, 'ru', reversed=True))
param_3 = train_data['param_3'].apply(lambda row:row if pd.isnull(row) else translit(row, 'ru', reversed=True))
title = train_data['title'].apply(lambda row:row if pd.isnull(row) else translit(row, 'ru', reversed=True))
description = train_data['description'].apply(lambda row:row if pd.isnull(row) else translit(row, 'ru', reversed=True))


region_test = (test_data['region']).apply(translit, 'ru', reversed=True)
city_test = (test_data['city']).apply(translit, 'ru', reversed=True)
parent_category_name_test = (test_data['parent_category_name']).apply(translit, 'ru', reversed=True)
category_name_test = (test_data['category_name']).apply(translit, 'ru', reversed=True)
param_1_test = test_data['param_1'].apply(lambda row:row if pd.isnull(row) else translit(row, 'ru', reversed=True))
param_2_test = test_data['param_2'].apply(lambda row:row if pd.isnull(row) else translit(row, 'ru', reversed=True))
param_3_test = test_data['param_3'].apply(lambda row:row if pd.isnull(row) else translit(row, 'ru', reversed=True))
title_test = test_data['title'].apply(lambda row:row if pd.isnull(row) else translit(row, 'ru', reversed=True))
description_test = test_data['description'].apply(lambda row:row if pd.isnull(row) else translit(row, 'ru', reversed=True))

train_data_translated = train_data
train_data_translated['region'] = region
train_data_translated['city'] = city
train_data_translated['parent_category_name'] = parent_category_name
train_data_translated['category_name'] = category_name
train_data_translated['param_1'] = param_1
train_data_translated['param_2'] = param_2
train_data_translated['param_3'] = param_3
train_data_translated['title'] = title
train_data_translated['description'] = description

test_data_translated = test_data
test_data_translated['region'] = region_test
test_data_translated['city'] = city_test
test_data_translated['parent_category_name'] = parent_category_name_test
test_data_translated['category_name'] = category_name_test
test_data_translated['param_1'] = param_1_test
test_data_translated['param_2'] = param_2_test
test_data_translated['param_3'] = param_3_test
test_data_translated['title'] = title_test
test_data_translated['description'] = description_test
## @hidden_cell
#train_data_translated.to_csv("train_translated.csv")
#train_data_translated = pd.read_csv("../input/translated/train_translated.csv")
#train_data_translated.head(n=5)

train_data_transformed_feature = train_data_translated
sliced_data_for_feature_engg = train_data_translated.iloc[:, [2,3,4,5,6,7,8,13,14]]
sliced_data_for_feature_engg = sliced_data_for_feature_engg.apply(lambda s: s.map({k:i for i,k in enumerate(s.unique())}))
train_data_transformed_feature.iloc[:, [2,3,4,5,6,7,8,13,14]] = sliced_data_for_feature_engg
#train_data_transformed_feature.head(5)

test_data_transformed_feature = test_data_translated
sliced_data_for_feature_engg = test_data_translated.iloc[:, [2,3,4,5,6,7,8,13,14]]
sliced_data_for_feature_engg = sliced_data_for_feature_engg.apply(lambda s: s.map({k:i for i,k in enumerate(s.unique())}))
test_data_transformed_feature.iloc[:, [2,3,4,5,6,7,8,13,14]] = sliced_data_for_feature_engg
test_data_transformed_feature.head(5)
from string import ascii_letters
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="white")
corr = train_data_transformed_feature.iloc[:, [3,4,5,6,7,8,11,12,13,14,16,17]].corr()

mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
sns.pairplot(train_data_transformed_feature.iloc[:, [3,4,5,6,7,8,13,14]])
