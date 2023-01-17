import os
PATH="/Users/lixi/Desktop/CMPS242"
print(os.listdir(PATH))
import pandas as pd 
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline 
application_train = pd.read_csv(PATH+"/application_train.csv")
application_test = pd.read_csv(PATH+"/application_test.csv")
print("application_train -  rows:",application_train.shape[0]," columns:", application_train.shape[1])
print("application_test -  rows:",application_test.shape[0]," columns:", application_test.shape[1])
application_train.head()
application_train.columns.values
application_test.head()
application_test.columns.values
def tranform_bin(dataset, feature, start, end, number, target, datetoyear,nan):
    plt.scatter(dataset['SK_ID_CURR'],dataset[feature])
    plt.figure(figsize = (10, 8))
    if (target):
        # KDE plot of loans that were repaid on time
        sns.kdeplot(dataset.loc[dataset['TARGET'] == 0, feature] , label = 'target == 0')

        # KDE plot of loans which were not repaid on time
        sns.kdeplot(dataset.loc[dataset['TARGET'] == 1, feature] , label = 'target == 1')

        # Labeling of plot
        plt.xlabel(feature); plt.ylabel('Density'); plt.title('Distribution');
        plt.show()
        if (nan):
            dataset[feature]=dataset[feature].fillna(0)
    
         # information into a separate dataframe
        tran_data = dataset[['TARGET', feature]]
        # Bin the  data
        if (datetoyear):
            tran_data[feature] = tran_data[feature] / 365
        tran_data['BINNED'] = pd.cut(tran_data[feature], bins = np.linspace(start,end, num=number ))
        print(tran_data)
        # Group by the bin and calculate averages
        tran_groups_new0 = tran_data['BINNED'].to_frame()
        tran_groups_new0.columns = ['range']
        #concatenate age and its bin
        tran_groups_new = pd.concat([tran_data[feature],tran_groups_new0],axis = 1)
        sns.set(style="white")
        sns.set(style="whitegrid", color_codes=True)
 
        #draw histogram plot
        sns.countplot(x = 'range', data = tran_groups_new, palette = 'hls')
        plt.show()
        tran_groups  = tran_data.groupby('BINNED').mean()
        print(tran_groups)
        plt.figure(figsize = (8, 8))
        # Graph the bins and the average of the target as a bar plot
        plt.xticks(range(len(tran_groups.index.astype(str))), tran_groups.index.astype(str))
        plt.bar(range(len(tran_groups.index.astype(str))), 100 * tran_groups['TARGET'])
        # Plot labeling
        plt.xticks(rotation = 75); plt.xlabel('Range'); plt.ylabel('Failure to Repay (%)')
        plt.title('Failure to Repay by Group');
        return tran_groups_new['range']
    else:
        if (nan):
            dataset[feature]=dataset[feature].fillna(0)
    
         # information into a separate dataframe
        tran_data = dataset[[feature]]
        # Bin the  data
        if (datetoyear):
            tran_data[feature] = tran_data[feature] / 365
        tran_data['BINNED'] = pd.cut(tran_data[feature], bins = np.linspace(start,end, num=number ))
        print(tran_data)
        # Group by the bin and calculate averages
        tran_groups_new0 = tran_data['BINNED'].to_frame()
        tran_groups_new0.columns = ['range']
        #concatenate age and its bin
        tran_groups_new = pd.concat([tran_data[feature],tran_groups_new0],axis = 1)
        sns.set(style="white")
        sns.set(style="whitegrid", color_codes=True)
        #draw histogram plot
        sns.countplot(x = 'range', data = tran_groups_new, palette = 'hls')
        plt.show()
        tran_groups  = tran_data.groupby('BINNED').mean()
        print(tran_groups)
        return tran_groups_new['range']
        
application_train['OWN_CAR_AGE']=tranform_bin(application_train, 'OWN_CAR_AGE', -1, 71, 5, target=True, datetoyear=False, nan=True)
application_train['OBS_30_CNT_SOCIAL_CIRCLE'] = (application_train['OBS_30_CNT_SOCIAL_CIRCLE']+application_train['DEF_30_CNT_SOCIAL_CIRCLE']+application_train['OBS_60_CNT_SOCIAL_CIRCLE']+application_train['DEF_60_CNT_SOCIAL_CIRCLE'])/4
application_train['OBS_30_CNT_SOCIAL_CIRCLE']=tranform_bin(application_train, 'OBS_30_CNT_SOCIAL_CIRCLE', -1, 25, 6, target=True, datetoyear=False, nan=False)
application_train['DAYS_LAST_PHONE_CHANGE']=tranform_bin(application_train, 'DAYS_LAST_PHONE_CHANGE', -10, 0, 6, target=True, datetoyear=True,nan=False)
application_test['OWN_CAR_AGE']=tranform_bin(application_test, 'OWN_CAR_AGE', -1, 71, 5, target=False, datetoyear=False, nan=True)
application_test['DAYS_LAST_PHONE_CHANGE']=tranform_bin(application_test, 'DAYS_LAST_PHONE_CHANGE', -10, 0, 6, target=False, datetoyear=True,nan=False)
application_test['OBS_30_CNT_SOCIAL_CIRCLE'] = (application_test['OBS_30_CNT_SOCIAL_CIRCLE']+application_test['DEF_30_CNT_SOCIAL_CIRCLE']+application_test['OBS_60_CNT_SOCIAL_CIRCLE']+application_test['DEF_60_CNT_SOCIAL_CIRCLE'])/4
application_test['OBS_30_CNT_SOCIAL_CIRCLE']=tranform_bin(application_test, 'OBS_30_CNT_SOCIAL_CIRCLE', -1, 25, 6, target=False, datetoyear=False, nan=False)
# one-hot encoding of categorical variables
application_train = pd.get_dummies(application_train)
application_test = pd.get_dummies(application_test)

print('Training Features shape: ', application_train.shape)
print('Testing Features shape: ', application_test.shape)
