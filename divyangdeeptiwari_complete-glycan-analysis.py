import pandas as pd

import numpy as np



import seaborn as sns



from matplotlib import pyplot as plt



from sklearn.ensemble import GradientBoostingRegressor

from sklearn.model_selection import train_test_split



import eli5

from eli5.sklearn import PermutationImportance
glycan = pd.read_excel('../input/Glycan data_updated_formatted.xlsx')

glycan_fillna = glycan.fillna(0)

glycan
glycan['GroupM']= glycan_fillna['Man5']+glycan_fillna['Man6']+glycan_fillna['Man7']

glycan['GroupB']= glycan_fillna['G0F-N']+glycan_fillna['G1F-N']+glycan_fillna['G2FB']

glycan['GroupF']= glycan_fillna['G0F-N']+glycan_fillna['G0F']+glycan_fillna['G1F-N']+glycan_fillna['G1F']+glycan_fillna['G2F']+glycan_fillna['G2FB']+glycan_fillna['[G1F]SA']+glycan_fillna['[G2F]2SA']+glycan_fillna['[G2F]SA']

glycan['GroupF0']= glycan_fillna['G0']+glycan_fillna['G1']+glycan_fillna['Man5']+glycan_fillna['Man6']+glycan_fillna['Man7']

glycan['GroupG']= glycan_fillna['G1']+glycan_fillna['G1F-N']+glycan_fillna['G1F']+glycan_fillna['G2F']+glycan_fillna['G2FB']

glycan['GroupG0']= glycan_fillna['G0']+glycan_fillna['G0F-N']+glycan_fillna['G0F']+glycan_fillna['Man5']+glycan_fillna['Man6']+glycan_fillna['Man7']+glycan_fillna['[G1F]SA']+glycan_fillna['[G2F]2SA']+glycan_fillna['[G2F]SA']

glycan['GroupSA']= glycan_fillna['[G1F]SA']+glycan_fillna['[G2F]2SA']+glycan_fillna['[G2F]SA']

glycan
#For comparing Mains with different groups 

main_groups_sliced = glycan.iloc[[2,7,14,21,28,34],15:]

ind = np.arange(len(main_groups_sliced.columns)-1)  # the x locations for the groups

width = 0.36  # the width of the bars



fig, axs = plt.subplots()

fig = plt.figure(figsize=(20,20))

rects1 = axs.bar(ind - 3*width/3, list(main_groups_sliced.iloc[0,1:]), width/3,

                label='Ristova M')

rects2 = axs.bar(ind - 2*width/3, list(main_groups_sliced.iloc[1,1:]), width/3, 

                label='Biosim1 M')

rects3 = axs.bar(ind - width/3, list(main_groups_sliced.iloc[2,1:]), width/3, 

                label='Biosim2 M')

rects4 = axs.bar(ind , list(main_groups_sliced.iloc[3,1:]), width/3, 

                label='Biosim3 M')

rects5 = axs.bar(ind + width/3, list(main_groups_sliced.iloc[4,1:]), width/3, 

                label='Biosim4 M')

rects6 = axs.bar(ind + 2*width/3, list(main_groups_sliced.iloc[5,1:]), width/3, 

                label='Biosim5 M')



# Add some text for labels, title and custom x-axis tick labels, etc.

axs.set_title('Glycoforms % by groups for main samples')

axs.set_ylabel('Glycoforms %')

axs.set_xticks(ind)

axs.set_xticklabels(('GroupM','GroupB','GroupF','GroupF0','GroupG','GroupG0','GroupSA'))

axs.legend()
#For comparing Acidic samples with different groups

acid_groups_sliced = glycan.iloc[[0,1,5,6,12,13,19,20,26,27,32,33],15:]



fig, axs = plt.subplots(1,2,figsize=(13,5))

rects1 = axs[1].bar(ind - 3*width/3, list(acid_groups_sliced.iloc[0,1:]), width/3,

                label='Ristova A2')

rects2 = axs[1].bar(ind - 2*width/3, list(acid_groups_sliced.iloc[2,1:]), width/3, 

                label='Biosim1 A2')

rects3 = axs[1].bar(ind - width/3, list(acid_groups_sliced.iloc[4,1:]), width/3, 

                label='Biosim2 A2')

rects4 = axs[1].bar(ind , list(acid_groups_sliced.iloc[6,1:]), width/3, 

                label='Biosim3 A2')

rects5 = axs[1].bar(ind + width/3, list(acid_groups_sliced.iloc[8,1:]), width/3, 

                label='Biosim4 A2')

rects6 = axs[1].bar(ind + 2*width/3, list(acid_groups_sliced.iloc[10,1:]), width/3, 

                label='Biosim5 A2')



rects1 = axs[0].bar(ind - 3*width/3, list(acid_groups_sliced.iloc[1,1:]), width/3,

                label='Ristova A1')

rects2 = axs[0].bar(ind - 2*width/3, list(acid_groups_sliced.iloc[3,1:]), width/3, 

                label='Biosim1 A1')

rects3 = axs[0].bar(ind - width/3, list(acid_groups_sliced.iloc[5,1:]), width/3, 

                label='Biosim2 A1')

rects4 = axs[0].bar(ind , list(acid_groups_sliced.iloc[7,1:]), width/3, 

                label='Biosim3 A1')

rects5 = axs[0].bar(ind + width/3, list(acid_groups_sliced.iloc[9,1:]), width/3, 

                label='Biosim4 A1')

rects6 = axs[0].bar(ind + 2*width/3, list(acid_groups_sliced.iloc[11,1:]), width/3, 

                label='Biosim5 A1')



fig.suptitle('Glycoforms % by groups for A1 & A2 samples respectively')

for i in range(2):

    axs[i].set_ylabel('Glycoforms %')

    axs[i].set_xticks(ind)

    axs[i].set_xticklabels(('GroupM','GroupB','GroupF','GroupF0','GroupG','GroupG0','GroupSA'))

    axs[i].legend()
#For comparing Basic samples with different groups

base_groups_sliced = glycan.iloc[[3,4,8,9,15,16,22,23,29,30,35,36],15:]



fig, axs = plt.subplots(1,2,figsize=(13,5))

rects1 = axs[0].bar(ind - 3*width/3, list(base_groups_sliced.iloc[0,1:]), width/3,

                label='Ristova B1')

rects2 = axs[0].bar(ind - 2*width/3, list(base_groups_sliced.iloc[2,1:]), width/3, 

                label='Biosim1 B1')

rects3 = axs[0].bar(ind - width/3, list(base_groups_sliced.iloc[4,1:]), width/3, 

                label='Biosim2 B1')

rects4 = axs[0].bar(ind , list(base_groups_sliced.iloc[6,1:]), width/3, 

                label='Biosim3 B1')

rects5 = axs[0].bar(ind + width/3, list(base_groups_sliced.iloc[8,1:]), width/3, 

                label='Biosim4 B1')

rects6 = axs[0].bar(ind + 2*width/3, list(base_groups_sliced.iloc[10,1:]), width/3, 

                label='Biosim5 B1')



rects1 = axs[1].bar(ind - 3*width/3, list(base_groups_sliced.iloc[1,1:]), width/3,

                label='Ristova B2')

rects2 = axs[1].bar(ind - 2*width/3, list(base_groups_sliced.iloc[3,1:]), width/3, 

                label='Biosim1 B2')

rects3 = axs[1].bar(ind - width/3, list(base_groups_sliced.iloc[5,1:]), width/3, 

                label='Biosim2 B2')

rects4 = axs[1].bar(ind , list(base_groups_sliced.iloc[7,1:]), width/3, 

                label='Biosim3 B2')

rects5 = axs[1].bar(ind + width/3, list(base_groups_sliced.iloc[9,1:]), width/3, 

                label='Biosim4 B2')

rects6 = axs[1].bar(ind + 2*width/3, list(base_groups_sliced.iloc[11,1:]), width/3, 

                label='Biosim5 B2')



fig.suptitle('Glycoforms % by groups for B1 & B2 samples respectively')

for i in range(2):

    axs[i].set_ylabel('Glycoforms %')

    axs[i].set_xticks(ind)

    axs[i].set_xticklabels(('GroupM','GroupB','GroupF','GroupF0','GroupG','GroupG0','GroupSA'))

    axs[i].legend()
B3_groups_sliced = glycan.iloc[[10,17,24,31,37],15:]

B4_groups_sliced = glycan.iloc[[11,25],15:]

B4_groups_sliced
features = glycan.columns[1:15]

target = 'Charge Variant'

X = glycan[features].fillna(0)

y = glycan[target]

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.00000000000001, random_state=324)

gbr_model = GradientBoostingRegressor().fit(X,y)

perm_gbr = PermutationImportance(gbr_model, random_state=0).fit(X, y)

eli5.show_weights(perm_gbr, feature_names = X.columns.tolist())

features = glycan.columns[16:23]

target = 'Charge Variant'

X = glycan[features]

y = glycan[target]

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.00000000000001, random_state=324)

gbr_model = GradientBoostingRegressor().fit(X,y)

perm_gbr = PermutationImportance(gbr_model, random_state=0).fit(X, y)

eli5.show_weights(perm_gbr, feature_names = X.columns.tolist())
glycan['Potency'] = [1.2,1.2,1.8,1.1,1.08,1.29,1.33,2.09,1.66,1.54,1.33,1.1,0.94,0.86,1.4,1.01,0.89,0.75,1,1.01,1.55,1.7,1.49,1.48,0.98,0.89,1.3,1.2,1.53,1.3,1.3,0.99,1.56,1.81,2.31,2.2,1.95,0.97]

glycan
corr = glycan[glycan.columns[:]].corr()['Charge Variant'][:]

corr = corr.sort_values(ascending=False)

corr = corr.drop(labels=['Charge Variant'])

corr



sns.set(font_scale=1.5)  

a4_dims = (18, 10)

fig, axs = plt.subplots(figsize=a4_dims)

ax = sns.barplot(corr.index, corr[:], palette="Blues_d",ax=axs)

ax.set_xticklabels(labels=corr.index,rotation=30)
features = glycan.columns[1:16]

target = 'Potency'

X = glycan[features].fillna(0)

y = glycan[target]

potency_gbr_model = GradientBoostingRegressor().fit(X,y)

#prediction = gbr_model.predict(X_train) 
