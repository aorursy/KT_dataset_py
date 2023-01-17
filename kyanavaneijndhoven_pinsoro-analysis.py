##In anaconda prompt pip install neupy 

import glob

import pandas as pd

import numpy as np

import skmultilearn

from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn import preprocessing

from scipy import sparse

import matplotlib.pyplot as plt

import matplotlib as mpl

import keras

from keras import models

from keras import layers

from keras.utils.np_utils import to_categorical

from keras.callbacks import EarlyStopping

from neupy import algorithms

import random as rand

rand.seed(1)
##Load in child-robot csv files

df_robot = pd.concat((pd.read_csv(filename, index_col=None, na_values= 'NaN') for filename in glob.glob('data/Annotated/Child-Robot/**/*.csv', recursive=True)))

df_robot.shape

df_robot.head()
##Create child-robot df for further analysis

delete_rows1 = df_robot.columns[3:195]

delete_rows2 = df_robot.columns[213:404]

delete_rows3 = df_robot.columns[404:443]

delete_rows4 = df_robot.columns[446:]

delete_rows = delete_rows1.append(delete_rows2)

delete_rows = delete_rows.append(delete_rows3)

delete_rows = delete_rows.append(delete_rows4)



child_robot = df_robot.drop(delete_rows, axis=1)

print(child_robot.head())

print(child_robot.shape)

print(child_robot.columns)
##Check conditions

print(pd.unique(child_robot['condition']))
##Input variables NaN count

print(child_robot['purple_child_au01'].isna().sum())

print(child_robot['purple_child_au02'].isna().sum())

print(child_robot['purple_child_au04'].isna().sum())

print(child_robot['purple_child_au05'].isna().sum())

print(child_robot['purple_child_au06'].isna().sum())

print(child_robot['purple_child_au07'].isna().sum())

print(child_robot['purple_child_au09'].isna().sum())

print(child_robot['purple_child_au10'].isna().sum())

print(child_robot['purple_child_au12'].isna().sum())

print(child_robot['purple_child_au14'].isna().sum())

print(child_robot['purple_child_au15'].isna().sum())

print(child_robot['purple_child_au17'].isna().sum())

print(child_robot['purple_child_au20'].isna().sum())

print(child_robot['purple_child_au23'].isna().sum())

print(child_robot['purple_child_au25'].isna().sum())

print(child_robot['purple_child_au26'].isna().sum())

print(child_robot['purple_child_au28'].isna().sum())

print(child_robot['purple_child_au45'].isna().sum())
##Copy df, delete completely empty column

robot = child_robot.dropna(axis=1, how='all')

print(child_robot.shape)

print(robot.shape)

print(child_robot.columns)

print(robot.columns)



##Delete completely empty input rows

robot = robot.dropna(axis=0, how='all', subset=['purple_child_au01',

       'purple_child_au02', 'purple_child_au04', 'purple_child_au05',

       'purple_child_au06', 'purple_child_au07', 'purple_child_au09',

       'purple_child_au10', 'purple_child_au12', 'purple_child_au14',

       'purple_child_au15', 'purple_child_au17', 'purple_child_au20',

       'purple_child_au23', 'purple_child_au25', 'purple_child_au26',

       'purple_child_au45'])



print(robot.shape)



##Check if number of deleted rows is equal to 139715

print(877558-737843==139715)



##Double check for NaN

print(robot['purple_child_au01'].isna().sum())

print(robot['purple_child_au02'].isna().sum())

print(robot['purple_child_au04'].isna().sum())

print(robot['purple_child_au05'].isna().sum())

print(robot['purple_child_au06'].isna().sum())

print(robot['purple_child_au07'].isna().sum())

print(robot['purple_child_au09'].isna().sum())

print(robot['purple_child_au10'].isna().sum())

print(robot['purple_child_au12'].isna().sum())

print(robot['purple_child_au14'].isna().sum())

print(robot['purple_child_au15'].isna().sum())

print(robot['purple_child_au17'].isna().sum())

print(robot['purple_child_au20'].isna().sum())

print(robot['purple_child_au23'].isna().sum())

print(robot['purple_child_au25'].isna().sum())

print(robot['purple_child_au26'].isna().sum())

print(robot['purple_child_au45'].isna().sum())
##Output variables NaN count

print(robot['purple_child_task_engagement'].isna().sum())

print(robot['purple_child_social_engagement'].isna().sum())

print(robot['purple_child_social_attitude'].isna().sum())



##Delete completely empty output rows

robot = robot.dropna(axis=0, how='all', subset=['purple_child_task_engagement',

       'purple_child_social_engagement', 'purple_child_social_attitude'])



print(robot.shape)



##Check if number of deleted rows is equal to 42489

print(737843-695354==42489)



##Double check for NaN

print(robot['purple_child_task_engagement'].isna().sum())

print(robot['purple_child_social_engagement'].isna().sum())

print(robot['purple_child_social_attitude'].isna().sum())
##Check representation of labels in the dataset: balanced/unbalanced? 

count_taskeng = pd.crosstab(index=robot['purple_child_task_engagement'],columns="count")      



count_soceng = pd.crosstab(index=robot['purple_child_social_engagement'],columns="count")



count_socatt = pd.crosstab(index=robot['purple_child_social_attitude'],columns="count")



print(count_taskeng)

print(count_soceng)

print(count_socatt)
##Delete label cooperative since there are no independent observations

robot = robot[robot['purple_child_social_engagement'].str.contains('cooperative', regex=False) == False]
##Create a smaller sample to work with, so running code goes faster.

##robot_sample = robot.sample(frac=0.1, random_state=1)

##print(robot_sample.shape)

robot_sample = robot

robot_sample.shape
##Split up double annotated target variables

robot_sample['purple_child_task_engagement'] = [[x.split('+')[0]] for x in robot_sample['purple_child_task_engagement']]

robot_sample['purple_child_social_engagement'] = [[x.split('+')[0]] for x in robot_sample['purple_child_social_engagement']]

robot_sample['purple_child_social_attitude'] = [[x.split('+')[0]] for x in robot_sample['purple_child_social_attitude']]
##Check annotations

print(np.unique(robot_sample['purple_child_task_engagement']))

print(np.unique(robot_sample['purple_child_social_engagement']))

print(np.unique(robot_sample['purple_child_social_attitude']))
##Create dataframes with mean FAU intensity per label for each target variable

au_input = ['purple_child_au01', 'purple_child_au02', 'purple_child_au04', 'purple_child_au05', 'purple_child_au06', 

            'purple_child_au07', 'purple_child_au09', 'purple_child_au10', 'purple_child_au12', 'purple_child_au14',

            'purple_child_au15', 'purple_child_au17', 'purple_child_au20', 'purple_child_au23', 'purple_child_au25', 

            'purple_child_au26', 'purple_child_au45']

te_labels = ['adultseeking', 'aimless', 'goaloriented', 'noplay']

se_labels = ['associative', 'onlooker', 'parallel', 'solitary']

sa_labels = ['assertive', 'passive', 'prosocial']



def mean_bylabel(df, target, labels, au_input):

    mean_list = []

    for label in labels:

        for au in au_input:

            mean = df.loc[df[target].str.contains(label, regex=False) == True, au].mean()

            mean_list.append(mean)

    return(pd.DataFrame(np.reshape(mean_list, (len(labels), len(au_input))).T))



te_mean_au = mean_bylabel(robot_sample, 'purple_child_task_engagement', te_labels, au_input)

te_mean_au.rename(columns={0:'adultseeking', 1:'aimless', 2:'goaloriented', 3:'noplay'}, 

                 index={0:'purple_child_au01', 1:'purple_child_au02', 2:'purple_child_au04', 3:'purple_child_au05', 

                           4:'purple_child_au06', 5:'purple_child_au07', 6:'purple_child_au09', 7:'purple_child_au10', 

                           8:'purple_child_au12', 9:'purple_child_au14', 10:'purple_child_au15', 11:'purple_child_au17', 

                           12:'purple_child_au20', 13:'purple_child_au23', 14:'purple_child_au25', 15:'purple_child_au26', 

                           16:'purple_child_au45'}, inplace=True)



se_mean_au = mean_bylabel(robot_sample, 'purple_child_social_engagement', se_labels, au_input)

se_mean_au.rename(columns={0:'associative', 1:'onlooker', 2:'parallel', 3:'solitary'}, 

                 index={0:'purple_child_au01', 1:'purple_child_au02', 2:'purple_child_au04', 3:'purple_child_au05', 

                           4:'purple_child_au06', 5:'purple_child_au07', 6:'purple_child_au09', 7:'purple_child_au10', 

                           8:'purple_child_au12', 9:'purple_child_au14', 10:'purple_child_au15', 11:'purple_child_au17', 

                           12:'purple_child_au20', 13:'purple_child_au23', 14:'purple_child_au25', 15:'purple_child_au26', 

                           16:'purple_child_au45'}, inplace=True)



sa_mean_au = mean_bylabel(robot_sample, 'purple_child_social_attitude', sa_labels, au_input)

sa_mean_au.rename(columns={0:'assertive', 1:'passive', 2:'prosocial'}, 

                 index={0:'purple_child_au01', 1:'purple_child_au02', 2:'purple_child_au04', 3:'purple_child_au05', 

                           4:'purple_child_au06', 5:'purple_child_au07', 6:'purple_child_au09', 7:'purple_child_au10', 

                           8:'purple_child_au12', 9:'purple_child_au14', 10:'purple_child_au15', 11:'purple_child_au17', 

                           12:'purple_child_au20', 13:'purple_child_au23', 14:'purple_child_au25', 15:'purple_child_au26', 

                           16:'purple_child_au45'}, inplace=True)
##Visualise FAU intensity mean per FAU and per label for task engagement

au_plotlabels = ['fau01', 'fau02', 'fau04', 'fau05', 'fau06', 'fau07', 'fau09', 'fau10', 'fau12', 'fau14',

                 'fau15', 'fau17', 'fau20', 'fau23', 'fau25', 'fau26', 'fau45']

y1 = te_mean_au['adultseeking']

y2 = te_mean_au['aimless']

y3 = te_mean_au['goaloriented']

y4 = te_mean_au['noplay']

x = np.linspace(0, 16, 17, endpoint=True)

plt.scatter(x, y1)

plt.scatter(x, y2, marker=">")

plt.scatter(x, y3, marker="+")

plt.scatter(x, y4, marker="*")

plt.ylim([0, 1])

plt.legend(loc='upper center', bbox_to_anchor=(0.4, 0.98))

plt.xticks(np.arange(0, 17, step=1))

plt.xticks(np.arange(17), au_plotlabels, rotation=90)

plt.show()
##Visualise FAU intensity mean per FAU and per label for social engagement

y1 = se_mean_au['associative']

y2 = se_mean_au['onlooker']

y3 = se_mean_au['parallel']

y4 = se_mean_au['solitary']

x = np.linspace(0, 16, 17, endpoint=True)

plt.scatter(x, y1)

plt.scatter(x, y2, marker=">")

plt.scatter(x, y3, marker="+")

plt.scatter(x, y4, marker="*")

plt.ylim([0.02, 0.95])

plt.legend(loc='upper center', bbox_to_anchor=(1.15, 1.03))

plt.xticks(np.arange(0, 17, step=1))

plt.xticks(np.arange(17), au_plotlabels, rotation=90)

plt.show()
##Visualise FAU intensity mean per FAU and per label for social attitude

y1 = sa_mean_au['assertive']

y2 = sa_mean_au['passive']

y3 = sa_mean_au['prosocial']

x = np.linspace(0, 16, 17, endpoint=True)

plt.scatter(x, y1)

plt.scatter(x, y2, marker=">")

plt.scatter(x, y3, marker="+")

plt.ylim([0, 1.2])

plt.legend()

plt.xticks(np.arange(0, 17, step=1))

plt.xticks(np.arange(17), au_plotlabels, rotation=90)

plt.show()
##Check representation of labels in the dataset: balanced/unbalanced?



def count_bylabel(df, colname):

    unique_elements, counts_elements = np.unique(df[colname], return_counts=True)

    print("Frequency of levels of", colname, ':')

    for x in range(len(unique_elements)):

        print(unique_elements[x], counts_elements[x])



##Task engagement

count_bylabel(robot_sample, 'purple_child_task_engagement')



##Social engagement

count_bylabel(robot_sample, 'purple_child_social_engagement')



##Social attitude

count_bylabel(robot_sample, 'purple_child_social_attitude')
##Divide by label



def df_bylabel(df, colname, labelname):

    df_bylabel = df[df[colname].str.contains(labelname, regex=False) == True]

    print(colname, ':', labelname, ', shape =', df_bylabel.shape)

    return df_bylabel



##Task engagement

te_adultseeking = df_bylabel(robot_sample, 'purple_child_task_engagement', 'adultseeking')

te_aimless = df_bylabel(robot_sample, 'purple_child_task_engagement', 'aimless')

te_goaloriented = df_bylabel(robot_sample, 'purple_child_task_engagement', 'goaloriented')

te_noplay = df_bylabel(robot_sample, 'purple_child_task_engagement', 'noplay')



##Social engagement

se_associative = df_bylabel(robot_sample, 'purple_child_social_engagement', 'associative')

se_onlooker = df_bylabel(robot_sample, 'purple_child_social_engagement', 'onlooker')

se_parallel = df_bylabel(robot_sample, 'purple_child_social_engagement', 'parallel')

se_solitary = df_bylabel(robot_sample, 'purple_child_social_engagement', 'solitary')



##Social attitude

sa_assertive = df_bylabel(robot_sample, 'purple_child_social_attitude', 'assertive')

sa_passive = df_bylabel(robot_sample, 'purple_child_social_attitude', 'passive')

sa_prosocial = df_bylabel(robot_sample, 'purple_child_social_attitude', 'prosocial')

##Random under-sampling



##Task engagement

te_aimless_under = te_aimless.sample(19632)

te_goaloriented_under = te_goaloriented.sample(19632)

te_noplay_under = te_noplay.sample(19632)

te_sample_under = pd.concat([te_adultseeking, te_aimless_under, te_goaloriented_under, te_noplay_under], axis=0)



##Frequency of levels of task engagement after under-sampling

count_bylabel(te_sample_under, 'purple_child_task_engagement')



##Social engagement

se_onlooker_under = se_onlooker.sample(13540)

se_parallel_under = se_parallel.sample(13540)

se_solitary_under = se_solitary.sample(13540)

se_sample_under = pd.concat([se_associative, se_onlooker_under, se_parallel_under, 

                             se_solitary_under], axis=0)



##Frequency of levels of social engagement after under-sampling

count_bylabel(se_sample_under, 'purple_child_social_engagement')



##Social attitude

sa_assertive_under = sa_assertive.sample(60090)

sa_passive_under = sa_passive.sample(60090)

sa_sample_under = pd.concat([sa_prosocial, sa_assertive_under, sa_passive_under], axis=0)



##Frequency of levels of social attitude after under-sampling

count_bylabel(sa_sample_under, 'purple_child_social_attitude')
##Scatter plot au06 vs au14 by label

lab = {0:'purple_child_au01', 1:'purple_child_au02', 2:'purple_child_au04', 3:'purple_child_au05', 

                4:'purple_child_au06', 5:'purple_child_au07', 6:'purple_child_au09', 7:'purple_child_au10', 

                8:'purple_child_au12', 9:'purple_child_au14', 10:'purple_child_au15', 11:'purple_child_au17', 

                12:'purple_child_au20', 13:'purple_child_au23', 14:'purple_child_au25', 15:'purple_child_au26', 

                16:'purple_child_au45'} 

N = 60

x1 = se_onlooker_under['purple_child_au09']

y1 = se_parallel_under['purple_child_au09']

z1 = se_solitary_under['purple_child_au09']

a1 = se_associative['purple_child_au09']

x2 = se_onlooker_under['purple_child_au14']

y2 = se_parallel_under['purple_child_au14']

z2 = se_solitary_under['purple_child_au14']

a2 = se_associative['purple_child_au14']

 

one =np.concatenate((x1,y1,z1,a1))

two =np.concatenate((x2,y2,z2,a2))

 

color_array = ['b'] * 15 + ['g'] * 17 + ['r']*15 + ['y']*15

 

plt.scatter(one, two, c=color_array)

 

plt.xlabel('FAU 09', fontsize=16)

plt.ylabel('FAU 14', fontsize=16)

plt.title('grouped scatter plot - FAU 06 vs FAU 14',fontsize=20)

plt.show()
##Visualisation of FAU data per label for the three target variables (histogram), before and after under-sampling



FAU_cols = ['purple_child_au01', 'purple_child_au02', 'purple_child_au04', 

            'purple_child_au05', 'purple_child_au06', 'purple_child_au07', 

            'purple_child_au09', 'purple_child_au10', 'purple_child_au12', 

            'purple_child_au14', 'purple_child_au15', 'purple_child_au17', 

            'purple_child_au20', 'purple_child_au23', 'purple_child_au25', 

            'purple_child_au26', 'purple_child_au45']



def FAUs_bylabel(df, FAU_cols, df_name):

    df[FAU_cols].hist(bins=17, color='steelblue', edgecolor='black', linewidth=1.0,

           xlabelsize=8, ylabelsize=8, grid=False)   

    plt.suptitle(df_name,  y=1.3, fontsize=17)

    plt.tight_layout(rect=(0, 0, 1.2, 1.2)) 



##Task engagement

##te_adultseeking  

FAUs_bylabel(te_adultseeking, FAU_cols, 'te_adultseeking')



##te_aimless vs. te_aimless_under

FAUs_bylabel(te_aimless, FAU_cols, 'te_aimless')

FAUs_bylabel(te_aimless_under, FAU_cols, 'te_aimless_under')



##te_goaloriented vs. te_goaloriented_under 

FAUs_bylabel(te_goaloriented, FAU_cols, 'te_goaloriented')

FAUs_bylabel(te_goaloriented_under, FAU_cols, 'te_goaloriented_under')



##te_noplay vs. te_noplay_under

FAUs_bylabel(te_noplay, FAU_cols, 'te_noplay')

FAUs_bylabel(te_noplay_under, FAU_cols, 'te_noplay_under')
##Social engagement

##se_associative vs. se_associative_under

FAUs_bylabel(se_associative, FAU_cols, 'se_associative')



##se_onlooker vs. se_onlooker_under

FAUs_bylabel(se_onlooker, FAU_cols, 'se_onlooker')

FAUs_bylabel(se_onlooker_under, FAU_cols, 'se_onlooker_under')



##se_parallel vs. se_parallel_under

FAUs_bylabel(se_parallel, FAU_cols, 'se_parallel')

FAUs_bylabel(se_parallel_under, FAU_cols, 'se_parallel_under')



##se_solitary vs. se_solitary_under

FAUs_bylabel(se_solitary, FAU_cols, 'se_solitary')

FAUs_bylabel(se_solitary_under, FAU_cols, 'se_solitary_under')
##Social attitude



##sa_assertive vs. sa_assertive_under

FAUs_bylabel(sa_assertive, FAU_cols, 'sa_assertive')

FAUs_bylabel(sa_assertive_under, FAU_cols, 'sa_assertive_under')



##sa_passive vs. sa_passive_under

FAUs_bylabel(sa_passive, FAU_cols, 'sa_passive')

FAUs_bylabel(sa_passive_under, FAU_cols, 'sa_passive')



##sa_prosocial 

FAUs_bylabel(sa_prosocial, FAU_cols, 'sa_prosocial')
##Binarize the target variables



##Random under-sampled task engagement

##Binarize

lb = preprocessing.MultiLabelBinarizer()

te_binarized = lb.fit_transform(te_sample_under['purple_child_task_engagement'])



te_sample_under['te_adultseeking'] = te_binarized[:, 0]

te_sample_under['te_aimless'] = te_binarized[:, 1]

te_sample_under['te_goaloriented'] = te_binarized[:, 2]

te_sample_under['te_noplay'] = te_binarized[:, 3]



##Drop other target variables

te_sample_under = te_sample_under.drop(['purple_child_task_engagement', 'purple_child_social_engagement', 

                                          'purple_child_social_attitude'], 1)



print(te_sample_under.columns)



##Random under-sampled social engagement

##Binarize

se_binarized = lb.fit_transform(se_sample_under['purple_child_social_engagement'])



se_sample_under['se_associative'] = se_binarized[:, 0]

se_sample_under['se_onlooker'] = se_binarized[:, 1]

se_sample_under['se_parallel'] = se_binarized[:, 2]

se_sample_under['se_solitary'] = se_binarized[:, 3]



##Drop other target variables

se_sample_under = se_sample_under.drop(['purple_child_task_engagement', 'purple_child_social_engagement', 

                                          'purple_child_social_attitude'], 1)



print(se_sample_under.columns)



##Random under-sampled social attitude

##Binarize

sa_binarized = lb.fit_transform(sa_sample_under['purple_child_social_attitude'])



sa_sample_under['sa_assertive'] = sa_binarized[:, 0]

sa_sample_under['sa_passive'] = sa_binarized[:, 1]

sa_sample_under['sa_prosocial'] = sa_binarized[:, 2]



##Drop other target variables

sa_sample_under = sa_sample_under.drop(['purple_child_task_engagement', 'purple_child_social_engagement', 

                                          'purple_child_social_attitude'], 1)



print(sa_sample_under.columns)
##Create train/test split for te_sample_under

te_X_train, te_X_test, te_y_train, te_y_test = train_test_split(te_sample_under[['purple_child_au01',

       'purple_child_au02', 'purple_child_au04', 'purple_child_au05',

       'purple_child_au06', 'purple_child_au07', 'purple_child_au09',

       'purple_child_au10', 'purple_child_au12', 'purple_child_au14',

       'purple_child_au15', 'purple_child_au17', 'purple_child_au20',

       'purple_child_au23', 'purple_child_au25', 'purple_child_au26',

       'purple_child_au45']], te_sample_under[['te_adultseeking', 'te_aimless', 'te_goaloriented',

       'te_noplay']], test_size=0.2, random_state=1)



print('te_X_train', te_X_train.shape)

print('te_y_train', te_y_train.shape)

print('te_X_test', te_X_test.shape)

print('te_y_test', te_y_test.shape)



##Create train/test split for se_sample_under

se_X_train, se_X_test, se_y_train, se_y_test = train_test_split(se_sample_under[['purple_child_au01',

       'purple_child_au02', 'purple_child_au04', 'purple_child_au05',

       'purple_child_au06', 'purple_child_au07', 'purple_child_au09',

       'purple_child_au10', 'purple_child_au12', 'purple_child_au14',

       'purple_child_au15', 'purple_child_au17', 'purple_child_au20',

       'purple_child_au23', 'purple_child_au25', 'purple_child_au26',

       'purple_child_au45']], se_sample_under[['se_associative',  'se_onlooker', 

       'se_parallel', 'se_solitary']], test_size=0.2, random_state=1)



print('se_X_train', se_X_train.shape)

print('se_y_train', se_y_train.shape)

print('se_X_test', se_X_test.shape)

print('se_y_test', se_y_test.shape)



##Create train/test split for sa_sample_under

sa_X_train, sa_X_test, sa_y_train, sa_y_test = train_test_split(sa_sample_under[['purple_child_au01',

       'purple_child_au02', 'purple_child_au04', 'purple_child_au05',

       'purple_child_au06', 'purple_child_au07', 'purple_child_au09',

       'purple_child_au10', 'purple_child_au12', 'purple_child_au14',

       'purple_child_au15', 'purple_child_au17', 'purple_child_au20',

       'purple_child_au23', 'purple_child_au25', 'purple_child_au26',

       'purple_child_au45']], sa_sample_under[['sa_assertive', 'sa_passive', 

       'sa_prosocial']], test_size=0.2, random_state=1)



print('sa_X_train', sa_X_train.shape)

print('sa_y_train', sa_y_train.shape)

print('sa_X_test', sa_X_test.shape)

print('sa_y_test', sa_y_test.shape)
##Create suitable input for the label-powerset approach

te_X_train = sparse.csr_matrix(te_X_train.values).toarray()

te_y_train = sparse.csr_matrix(te_y_train.values).toarray()

te_X_test = sparse.csr_matrix(te_X_test.values).toarray()

te_y_test = sparse.csr_matrix(te_y_test.values).toarray()



se_X_train = sparse.csr_matrix(se_X_train.values).toarray()

se_y_train = sparse.csr_matrix(se_y_train.values).toarray()

se_X_test = sparse.csr_matrix(se_X_test.values).toarray()

se_y_test = sparse.csr_matrix(se_y_test.values).toarray()



sa_X_train = sparse.csr_matrix(sa_X_train.values).toarray()

sa_y_train = sparse.csr_matrix(sa_y_train.values).toarray()

sa_X_test = sparse.csr_matrix(sa_X_test.values).toarray()

sa_y_test = sparse.csr_matrix(sa_y_test.values).toarray()

##Check class balance

print(np.unique(te_y_train[:, 0], return_counts = True))

print(np.unique(te_y_test[:, 0], return_counts = True))

print(np.unique(te_y_train[:, 1], return_counts = True))

print(np.unique(te_y_test[:, 1], return_counts = True))

print(np.unique(te_y_train[:, 2], return_counts = True))

print(np.unique(te_y_test[:, 2], return_counts = True))

print(np.unique(te_y_train[:, 3], return_counts = True))

print(np.unique(te_y_test[:, 3], return_counts = True))



##Define classifier function



def clf(X_train, y_train, X_test, y_test, col, classifier):

    classifier = classifier

    classifier.fit(X_train, y_train[:, col])

    preds = classifier.predict(X_test)

    tn, fp, fn, tp = metrics.confusion_matrix(y_test[:, col], preds).ravel()

    print(((tn/(tn+fp)) + (tp/(tp+fn))) /2 )
##Logistic regression for task engagement labels

clf(te_X_train, te_y_train, te_X_test, te_y_test, 0, LogisticRegression(solver = 'lbfgs'))

clf(te_X_train, te_y_train, te_X_test, te_y_test, 1, LogisticRegression(solver = 'lbfgs'))

clf(te_X_train, te_y_train, te_X_test, te_y_test, 2, LogisticRegression(solver = 'lbfgs'))

clf(te_X_train, te_y_train, te_X_test, te_y_test, 3, LogisticRegression(solver = 'lbfgs'))
##Logistic regression for social engagement labels

clf(se_X_train, se_y_train, se_X_test, se_y_test, 0, LogisticRegression(solver = 'lbfgs'))

clf(se_X_train, se_y_train, se_X_test, se_y_test, 1, LogisticRegression(solver = 'lbfgs'))

clf(se_X_train, se_y_train, se_X_test, se_y_test, 2, LogisticRegression(solver = 'lbfgs'))

clf(se_X_train, se_y_train, se_X_test, se_y_test, 3, LogisticRegression(solver = 'lbfgs'))
##Logistic regression for social attitude labels

clf(sa_X_train, sa_y_train, sa_X_test, sa_y_test, 0, LogisticRegression(solver = 'lbfgs'))

clf(sa_X_train, sa_y_train, sa_X_test, sa_y_test, 1, LogisticRegression(solver = 'lbfgs'))

clf(sa_X_train, sa_y_train, sa_X_test, sa_y_test, 2, LogisticRegression(solver = 'lbfgs'))
##Naive bayes for task engagement labels

clf(te_X_train, te_y_train, te_X_test, te_y_test, 0, GaussianNB())

clf(te_X_train, te_y_train, te_X_test, te_y_test, 1, GaussianNB())

clf(te_X_train, te_y_train, te_X_test, te_y_test, 2, GaussianNB())

clf(te_X_train, te_y_train, te_X_test, te_y_test, 3, GaussianNB())
##Naive bayes for social engagement labels

clf(se_X_train, se_y_train, se_X_test, se_y_test, 0, GaussianNB())

clf(se_X_train, se_y_train, se_X_test, se_y_test, 1, GaussianNB())

clf(se_X_train, se_y_train, se_X_test, se_y_test, 2, GaussianNB())

clf(se_X_train, se_y_train, se_X_test, se_y_test, 3, GaussianNB())
##Naive bayes for social attitude labels

clf(sa_X_train, sa_y_train, sa_X_test, sa_y_test, 0, GaussianNB())

clf(sa_X_train, sa_y_train, sa_X_test, sa_y_test, 1, GaussianNB())

clf(sa_X_train, sa_y_train, sa_X_test, sa_y_test, 2, GaussianNB())
##Create smaller sample of data for faster computations

##Stack data

#te_train = np.hstack((te_X_train,te_y_train))

#print(te_train.shape)

#te_test = np.hstack((te_X_test,te_y_test))

#print(te_test.shape)



#se_train = np.hstack((se_X_train,se_y_train))

#print(se_train.shape)

#se_test = np.hstack((se_X_test,se_y_test))

#print(se_test.shape)



#sa_train = np.hstack((sa_X_train,sa_y_train))

#print(sa_train.shape)

#sa_test = np.hstack((sa_X_test,sa_y_test))

#print(sa_test.shape)



##Sample data

#te_X_train = te_train[np.random.choice(te_X_train.shape[0], 3000, replace=False), :]

#print(te_X_train.shape)

#te_y_train = te_X_train[:, 17:]

#print(te_y_train.shape)

#te_X_train = te_X_train[:, 0:17]

#print(te_X_train.shape)



#te_X_test = te_test[np.random.choice(te_X_test.shape[0], 1000, replace=False), :]

#print(te_X_test.shape)

#te_y_test = te_X_test[:, 17:]

#print(te_y_test.shape)

#te_X_test = te_X_test[:, 0:17]

#print(te_X_test.shape)



#se_X_train = se_train[np.random.choice(se_X_train.shape[0], 3000, replace=False), :]

#print(se_X_train.shape)

#se_y_train = se_X_train[:, 17:]

#print(se_y_train.shape)

#se_X_train = se_X_train[:, 0:17]

#print(se_X_train.shape)



#se_X_test = se_test[np.random.choice(se_X_test.shape[0], 1000, replace=False), :]

#print(se_X_test.shape)

#se_y_test = se_X_test[:, 17:]

#print(se_y_test.shape)

#se_X_test = se_X_test[:, 0:17]

#print(se_X_test.shape)



#sa_X_train = sa_train[np.random.choice(sa_X_train.shape[0], 3000, replace=False), :]

#print(sa_X_train.shape)

#sa_y_train = sa_X_train[:, 17:]

#print(sa_y_train.shape)

#sa_X_train = sa_X_train[:, 0:17]

#print(sa_X_train.shape)



#sa_X_test = sa_test[np.random.choice(sa_X_test.shape[0], 1000, replace=False), :]

#print(sa_X_test.shape)

#sa_y_test = sa_X_test[:, 17:]

#print(sa_y_test.shape)

#sa_X_test = sa_X_test[:, 0:17]

#print(sa_X_test.shape)
##In Ananconda prompt: pip install neupy

##PNN 

def PNN(X_train, y_train, X_test, y_test, col):

    pnn = algorithms.PNN(std=0.5, verbose=True)

    pnn.train(X_train, y_train[:,col])

    preds = pnn.predict(X_test)

    tn, fp, fn, tp = metrics.confusion_matrix(y_test[:, col], preds).ravel()

    print(((tn/(tn+fp)) + (tp/(tp+fn))) /2 )

    
##Task engagement --> 'adultseeking'

PNN(te_X_train, te_y_train, te_X_test, te_y_test, 0)
##Task engagement --> 'aimless'

PNN(te_X_train, te_y_train, te_X_test, te_y_test, 1)
##Task engagement --> 'goaloriented'

PNN(te_X_train, te_y_train, te_X_test, te_y_test, 2)
##Task engagement --> 'noplay'

PNN(te_X_train, te_y_train, te_X_test, te_y_test, 3)
##Social engagement --> 'associative'

PNN(se_X_train, se_y_train, se_X_test, se_y_test, 0)
##Social engagement --> 'onlooker'

PNN(se_X_train, se_y_train, se_X_test, se_y_test, 1)
##Social engagement --> 'parallel'

PNN(se_X_train, se_y_train, se_X_test, se_y_test, 2)
##Social engagement --> 'solitary'

PNN(se_X_train, se_y_train, se_X_test, se_y_test, 3)
##Social attitude --> 'assertive'

PNN(sa_X_train, sa_y_train, sa_X_test, sa_y_test, 0)
##Social attitude --> 'passive'

PNN(sa_X_train, sa_y_train, sa_X_test, sa_y_test, 1)
##Social attitude --> 'prosocial'

PNN(sa_X_train, sa_y_train, sa_X_test, sa_y_test, 2)
##Data to plot Task Engagement per classifier

n_groups = 4

acc_LR = (51.3,50.0,50.2,50.0)

acc_NB = (56.2,53.0,55.1,53.3)

acc_PNN = (87.3,78.4,75.2,78.2)



##Create plot



mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["#3B5998", "c", "grey"])

classifiers = ['Logistic Regression', 'Naive Bayes', 'Probabilistic Neural Network']



def bar_group(classes, values, width=0.8):

    total_data = len(values)

    classes_num = np.arange(len(classes))

    for i in range(total_data):

        bars = plt.bar(classes_num - width / 2. + i / total_data * width, values[i], 

                width=width / total_data, align="edge", animated=0.4, label = classifiers[i])

        for rect in bars:

            height = rect.get_height()

            plt.text(rect.get_x() + rect.get_width()/2.0, height, '%d' % int(height), ha='center', va='bottom')

        



group = ['Adultseeking', 'Aimless', 'Goaloriented', 'Noplay']

bar_group(group, [acc_LR, acc_NB, acc_PNN])

index = np.arange(n_groups)   



plt.xlabel('Label', fontsize = 20)

plt.ylabel('Balanced accuracy', fontsize = 20)

plt.tick_params(labelsize=18)

plt.title('Task engagement scores', fontsize = 20)

plt.xticks(index, ('Adultseeking', 'Aimless', 'Goaloriented', 'Noplay'), fontsize = 13)

plt.ylim([0, 100])

plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.15),

          fancybox=True, ncol=4)



plt.savefig('Ex1_TE.jpg', bbox_inches='tight')

 

plt.tight_layout()

plt.show()
##Data to plot Social Engagement per classifier

n_groups = 4

acc_LR = (54.2,50.9,50.1,50.0)

acc_NB = (56.7,62.1,52.5,54.8)

acc_PNN = (89.8,82.5,76.7,78.7)

 

##Create plot



mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["#3B5998", "c", "grey"])

classifiers = ['Logistic Regression', 'Naive Bayes', 'Probabilistic Neural Network']



def bar_group(classes, values, width=0.8):

    total_data = len(values)

    classes_num = np.arange(len(classes))

    for i in range(total_data):

        bars = plt.bar(classes_num - width / 2. + i / total_data * width, values[i], 

                width=width / total_data, align="edge", animated=0.4, label = classifiers[i])

        for rect in bars:

            height = rect.get_height()

            plt.text(rect.get_x() + rect.get_width()/2.0, height, '%d' % int(height), ha='center', va='bottom')

        



group = ['Associative', 'Onlooker', 'Parallel', 'Solitary']

bar_group(group, [acc_LR, acc_NB, acc_PNN])

index = np.arange(n_groups)



plt.xlabel('Label', fontsize = 20)

plt.ylabel('Balanced accuracy', fontsize = 20)

plt.tick_params(labelsize=18)

plt.title('Social engagement scores', fontsize = 20)

plt.xticks(index, ('Associative', 'Onlooker', 'Parallel', 'Solitary'), fontsize = 13)

plt.ylim([0, 100])

plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.15),

          fancybox=True, ncol=4)



plt.savefig('Ex1_SE.jpg', bbox_inches='tight')

plt.tight_layout()

plt.show()
##Data to plot Social Attitude per classifier

n_groups = 3

acc_LR = (53.6,50.2,55.1)

acc_NB = (55.0,55.8,55.9)

acc_PNN = (86.3,80.5,85.6)

 

##Create plot



mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["#3B5998", "c", "grey"])

classifiers = ['Logistic Regression', 'Naive Bayes', 'Probabilistic Neural Network']



def bar_group(classes, values, width=0.8):

    total_data = len(values)

    classes_num = np.arange(len(classes))

    for i in range(total_data):

        bars = plt.bar(classes_num - width / 2. + i / total_data * width, values[i], 

                width=width / total_data, align="edge", animated=0.4, label = classifiers[i])

        for rect in bars:

            height = rect.get_height()

            plt.text(rect.get_x() + rect.get_width()/2.0, height, '%d' % int(height), ha='center', va='bottom')

        



group = ['Assertive', 'Passive', 'Prosocial']

bar_group(group, [acc_LR, acc_NB, acc_PNN])

index = np.arange(n_groups)

 

plt.xlabel('Label', fontsize = 20)

plt.ylabel('Balanced accuracy', fontsize = 20)

plt.tick_params(labelsize=18)

plt.title('Social attitude scores', fontsize = 20)

plt.xticks(index, ('Assertive', 'Passive', 'Prosocial'), fontsize = 13)

plt.ylim([0, 100])

plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.15),

          fancybox=True, ncol=3)



plt.savefig('Ex1_SA.jpg', bbox_inches='tight')

 

plt.tight_layout()

plt.show()
##Load in child-child1 csv files

df_child1 = pd.concat((pd.read_csv(filename, index_col=None, na_values= 'NaN') for filename in glob.glob('data/Annotated/Child-Child1/**/*.csv', recursive=True)))

print(df_child1.shape)

print(df_child1.columns)
##Load in child-child2 csv files

df_child2 = pd.concat((pd.read_csv(filename, index_col=None, na_values= 'NaN') for filename in glob.glob('data/Annotated/Child-Child2/**/*.csv', recursive=True)))

print(df_child2.shape)

print(df_child2.columns)
##Create child-robot df for further analysis

delete_rows1 = df_child1.columns[3:195]

delete_rows2 = df_child1.columns[213:404]

delete_rows3 = df_child1.columns[422:443]

delete_rows = delete_rows1.append(delete_rows2)

delete_rows = delete_rows.append(delete_rows3)

child_child1 = df_child1.drop(delete_rows, axis=1)

child_child2 = df_child2.drop(delete_rows, axis=1)

##Create complete df_child

child_child = pd.concat([child_child1, child_child2], axis=0)

print(child_child.head())

print(child_child.shape)

print(child_child.columns)
##Check conditions

print(pd.unique(child_child['condition']))
##Creating child-child subset where 1 child behaves similar to the robot.

unique_robot = df_robot[['yellow_child_task_engagement','yellow_child_social_engagement', 

                      'yellow_child_social_attitude']].drop_duplicates()

#print(unique_robot)



##Drop row 28532 since NaN are deleted

unique_robot = unique_robot.drop([28532], axis = 0)

#print(unique_robot)



##Change double annotations to single annotations keeping each unique combination

unique_robot['yellow_child_social_engagement'][9:10] = 'parallel'

unique_robot['yellow_child_social_engagement'][10:11] = 'solitary'



print(unique_robot)
##First distinguish all rows where social attitude is passive

child_y1 = child_child[child_child['yellow_child_social_attitude'].str.contains('passive', regex=False) == True]

print(child_y1.shape)

child_p1 = child_child[child_child['purple_child_social_attitude'].str.contains('passive', regex=False) == True]

print(child_p1.shape)
##Second define a function that subsets unique combinations of task engagement and social engagement 

def child_subset(df, var1, var2, label1, label2):

    subset = df[(df[var1].str.contains(label1, regex=False) == True) & (df[var2].str.contains(label2, regex=False) == True)]

    return subset
##Create subset for each of the unique combination in unique_robot for both child observations (yellow and purple)

child_y2 = child_subset(child_y1, 'yellow_child_task_engagement', 'yellow_child_social_engagement',

                        unique_robot.values[0][0], unique_robot.values[0][1])

child_y3 = child_subset(child_y1, 'yellow_child_task_engagement', 'yellow_child_social_engagement',

                        unique_robot.values[1][0], unique_robot.values[1][1])

child_y4 = child_subset(child_y1, 'yellow_child_task_engagement', 'yellow_child_social_engagement',

                        unique_robot.values[2][0], unique_robot.values[2][1])

child_y5 = child_subset(child_y1, 'yellow_child_task_engagement', 'yellow_child_social_engagement',

                        unique_robot.values[3][0], unique_robot.values[3][1])

child_y6 = child_subset(child_y1, 'yellow_child_task_engagement', 'yellow_child_social_engagement',

                        unique_robot.values[4][0], unique_robot.values[4][1])

child_y7 = child_subset(child_y1, 'yellow_child_task_engagement', 'yellow_child_social_engagement',

                        unique_robot.values[5][0], unique_robot.values[5][1])

child_y8 = child_subset(child_y1, 'yellow_child_task_engagement', 'yellow_child_social_engagement',

                        unique_robot.values[6][0], unique_robot.values[6][1])

child_y9 = child_subset(child_y1, 'yellow_child_task_engagement', 'yellow_child_social_engagement',

                        unique_robot.values[7][0], unique_robot.values[7][1])

child_y10 = child_subset(child_y1, 'yellow_child_task_engagement', 'yellow_child_social_engagement',

                        unique_robot.values[8][0], unique_robot.values[8][1])

child_y11 = child_subset(child_y1, 'yellow_child_task_engagement', 'yellow_child_social_engagement',

                        unique_robot.values[9][0], unique_robot.values[9][1])

child_y12 = child_subset(child_y1, 'yellow_child_task_engagement', 'yellow_child_social_engagement',

                        unique_robot.values[10][0], unique_robot.values[10][1])



child_p2 = child_subset(child_p1, 'purple_child_task_engagement', 'purple_child_social_engagement',

                        unique_robot.values[0][0], unique_robot.values[0][1])

child_p3 = child_subset(child_p1, 'purple_child_task_engagement', 'purple_child_social_engagement',

                        unique_robot.values[1][0], unique_robot.values[1][1])

child_p4 = child_subset(child_p1, 'purple_child_task_engagement', 'purple_child_social_engagement',

                        unique_robot.values[2][0], unique_robot.values[2][1])

child_p5 = child_subset(child_p1, 'purple_child_task_engagement', 'purple_child_social_engagement',

                        unique_robot.values[3][0], unique_robot.values[3][1])

child_p6 = child_subset(child_p1, 'purple_child_task_engagement', 'purple_child_social_engagement',

                        unique_robot.values[4][0], unique_robot.values[4][1])

child_p7 = child_subset(child_p1, 'purple_child_task_engagement', 'purple_child_social_engagement',

                        unique_robot.values[5][0], unique_robot.values[5][1])

child_p8 = child_subset(child_p1, 'purple_child_task_engagement', 'purple_child_social_engagement',

                        unique_robot.values[6][0], unique_robot.values[6][1])

child_p9 = child_subset(child_p1, 'purple_child_task_engagement', 'purple_child_social_engagement',

                        unique_robot.values[7][0], unique_robot.values[7][1])

child_p10 = child_subset(child_p1, 'purple_child_task_engagement', 'purple_child_social_engagement',

                        unique_robot.values[8][0], unique_robot.values[8][1])

child_p11 = child_subset(child_p1, 'purple_child_task_engagement', 'purple_child_social_engagement',

                        unique_robot.values[9][0], unique_robot.values[9][1])

child_p12 = child_subset(child_p1, 'purple_child_task_engagement', 'purple_child_social_engagement',

                        unique_robot.values[10][0], unique_robot.values[10][1])
##Merge subsets into dataframe

child_child1 = pd.concat([child_y2, child_y3, child_y4, child_y5, child_y6, child_y7, child_y8, child_y9, 

                          child_y10, child_y11, child_y12], axis=0)



child_child2 = pd.concat([child_p2, child_p3, child_p4, child_p5, child_p6, child_p7, child_p8, child_p9, 

                          child_p10, child_p11, child_p12], axis=0)

print(child_child1.shape)

print(child_child2.shape)
##Copy df,create 1 column per target variable instead of 2

##If yellow child social dynamics equal to robot, delete yellow keep purple observations --> child_child1

##If purple child social dynamics equal to robot, delete purple keep yellow observations --> child_child2



print('before', child_child1.shape, child_child2.shape)

print(child_child1.columns)

print(child_child2.columns)

child_child1 = child_child1.drop(['yellow_child_au01', 'yellow_child_au02', 'yellow_child_au04', 'yellow_child_au05',

                                 'yellow_child_au06', 'yellow_child_au07', 'yellow_child_au09', 'yellow_child_au10', 

                                 'yellow_child_au12', 'yellow_child_au14', 'yellow_child_au15', 'yellow_child_au17', 

                                 'yellow_child_au20', 'yellow_child_au23', 'yellow_child_au25', 'yellow_child_au26',

                                 'yellow_child_au28', 'yellow_child_au45','yellow_child_task_engagement', 

                                 'yellow_child_social_engagement', 'yellow_child_social_attitude'], axis=1)



child_child2 = child_child2.drop(['purple_child_au01', 'purple_child_au02', 'purple_child_au04', 'purple_child_au05',

                                 'purple_child_au06', 'purple_child_au07', 'purple_child_au09', 'purple_child_au10', 

                                 'purple_child_au12', 'purple_child_au14', 'purple_child_au15', 'purple_child_au17', 

                                 'purple_child_au20', 'purple_child_au23', 'purple_child_au25', 'purple_child_au26',

                                 'purple_child_au28', 'purple_child_au45', 'purple_child_task_engagement', 

                                 'purple_child_social_engagement', 'purple_child_social_attitude'], axis=1)



print(child_child1.shape)

print(child_child2.shape)
child_child2.rename(columns={'yellow_child_au01': 'purple_child_au01', 'yellow_child_au02': 'purple_child_au02', 

                             'yellow_child_au04': 'purple_child_au04', 'yellow_child_au05': 'purple_child_au05',

                             'yellow_child_au06': 'purple_child_au06', 'yellow_child_au07': 'purple_child_au07', 

                             'yellow_child_au09': 'purple_child_au09', 'yellow_child_au10': 'purple_child_au10', 

                             'yellow_child_au12': 'purple_child_au12', 'yellow_child_au14': 'purple_child_au14', 

                             'yellow_child_au15': 'purple_child_au15', 'yellow_child_au17': 'purple_child_au17', 

                             'yellow_child_au20': 'purple_child_au20', 'yellow_child_au23': 'purple_child_au23', 

                             'yellow_child_au25': 'purple_child_au25', 'yellow_child_au26': 'purple_child_au26',

                             'yellow_child_au28': 'purple_child_au28', 'yellow_child_au45': 'purple_child_au45', 

                             'yellow_child_task_engagement': 'purple_child_task_engagement', 

                             'yellow_child_social_engagement': 'purple_child_social_engagement',

                             'yellow_child_social_attitude': 'purple_child_social_attitude'}, inplace=True)



print(child_child2.columns)

child_child = pd.concat([child_child1, child_child2], axis=0)

print('after', child_child.shape)

print(child_child.columns)
##Input variables NaN count

print('purple_child')

print(child_child['purple_child_au01'].isna().sum())

print(child_child['purple_child_au02'].isna().sum())

print(child_child['purple_child_au04'].isna().sum())

print(child_child['purple_child_au05'].isna().sum())

print(child_child['purple_child_au06'].isna().sum())

print(child_child['purple_child_au07'].isna().sum())

print(child_child['purple_child_au09'].isna().sum())

print(child_child['purple_child_au10'].isna().sum())

print(child_child['purple_child_au12'].isna().sum())

print(child_child['purple_child_au14'].isna().sum())

print(child_child['purple_child_au15'].isna().sum())

print(child_child['purple_child_au17'].isna().sum())

print(child_child['purple_child_au20'].isna().sum())

print(child_child['purple_child_au23'].isna().sum())

print(child_child['purple_child_au25'].isna().sum())

print(child_child['purple_child_au26'].isna().sum())

print(child_child['purple_child_au28'].isna().sum())

print(child_child['purple_child_au45'].isna().sum())
##Copy df, delete completely empty column

child = child_child.dropna(axis=1, how='all')

print(child_child.shape)

print(child.shape)

print(child_child.columns)

print(child.columns)



##Delete completely empty input rows

child = child.dropna(axis=0, how='all', subset=['purple_child_au01',

       'purple_child_au02', 'purple_child_au04', 'purple_child_au05',

       'purple_child_au06', 'purple_child_au07', 'purple_child_au09',

       'purple_child_au10', 'purple_child_au12', 'purple_child_au14',

       'purple_child_au15', 'purple_child_au17', 'purple_child_au20',

       'purple_child_au23', 'purple_child_au25', 'purple_child_au26',

       'purple_child_au45'])



print(child.shape)



##Double check for NaN

print(child['purple_child_au01'].isna().sum())

print(child['purple_child_au02'].isna().sum())

print(child['purple_child_au04'].isna().sum())

print(child['purple_child_au05'].isna().sum())

print(child['purple_child_au06'].isna().sum())

print(child['purple_child_au07'].isna().sum())

print(child['purple_child_au09'].isna().sum())

print(child['purple_child_au10'].isna().sum())

print(child['purple_child_au12'].isna().sum())

print(child['purple_child_au14'].isna().sum())

print(child['purple_child_au15'].isna().sum())

print(child['purple_child_au17'].isna().sum())

print(child['purple_child_au20'].isna().sum())

print(child['purple_child_au23'].isna().sum())

print(child['purple_child_au25'].isna().sum())

print(child['purple_child_au26'].isna().sum())

print(child['purple_child_au45'].isna().sum())
##Double check for NaN, previously filtered out by child_subset() function

print(child['purple_child_task_engagement'].isna().sum())

print(child['purple_child_social_engagement'].isna().sum())

print(child['purple_child_social_attitude'].isna().sum())
##Check representation of labels in the dataset: balanced/unbalanced? 

count_taskeng1 = pd.crosstab(index=child['purple_child_task_engagement'],columns="count")      



count_soceng1 = pd.crosstab(index=child['purple_child_social_engagement'],columns="count")



count_socatt1 = pd.crosstab(index=child['purple_child_social_attitude'],columns="count")



print('\npurple_child\n')

print(count_taskeng1)

print(count_soceng1)

print(count_socatt1)
##Delete labels to make child data similar to robot data

child = child[child['purple_child_social_engagement'].str.contains('cooperative', regex=False) == False]

child = child[child['purple_child_social_attitude'].str.contains('adversarial', regex=False) == False]

child = child[child['purple_child_social_attitude'].str.contains('frustrated', regex=False) == False]
##Create a smaller sample to work with, so running code goes faster.

##child_sample = child.sample(frac=0.1, random_state=1)

##print(child_sample.shape)

child_sample = child

child_sample.shape
##Split up double annotated target variables

child_sample['purple_child_task_engagement'] = [[x.split('+')[0]] for x in child_sample['purple_child_task_engagement']]

child_sample['purple_child_social_engagement'] = [[x.split('+')[0]] for x in child_sample['purple_child_social_engagement']]

child_sample['purple_child_social_attitude'] = [[x.split('+')[0]] for x in child_sample['purple_child_social_attitude']]
##Check annotations

print(np.unique(child_sample['purple_child_task_engagement']))

print(np.unique(child_sample['purple_child_social_engagement']))

print(np.unique(child_sample['purple_child_social_attitude']))
##Create dataframes with mean FAU intensity per label for each target variable

te_mean_au1 = mean_bylabel(child_sample, 'purple_child_task_engagement', te_labels, au_input)

te_mean_au1.rename(columns={0:'adultseeking', 1:'aimless', 2:'goaloriented', 3:'noplay'}, 

                 index={0:'purple_child_au01', 1:'purple_child_au02', 2:'purple_child_au04', 3:'purple_child_au05', 

                           4:'purple_child_au06', 5:'purple_child_au07', 6:'purple_child_au09', 7:'purple_child_au10', 

                           8:'purple_child_au12', 9:'purple_child_au14', 10:'purple_child_au15', 11:'purple_child_au17', 

                           12:'purple_child_au20', 13:'purple_child_au23', 14:'purple_child_au25', 15:'purple_child_au26', 

                           16:'purple_child_au45'}, inplace=True)



se_mean_au1 = mean_bylabel(child_sample, 'purple_child_social_engagement', se_labels, au_input)

se_mean_au1.rename(columns={0:'associative', 1:'onlooker', 2:'parallel', 3:'solitary'}, 

                 index={0:'purple_child_au01', 1:'purple_child_au02', 2:'purple_child_au04', 3:'purple_child_au05', 

                           4:'purple_child_au06', 5:'purple_child_au07', 6:'purple_child_au09', 7:'purple_child_au10', 

                           8:'purple_child_au12', 9:'purple_child_au14', 10:'purple_child_au15', 11:'purple_child_au17', 

                           12:'purple_child_au20', 13:'purple_child_au23', 14:'purple_child_au25', 15:'purple_child_au26', 

                           16:'purple_child_au45'}, inplace=True)



sa_mean_au1 = mean_bylabel(child_sample, 'purple_child_social_attitude', sa_labels, au_input)

sa_mean_au1.rename(columns={0:'assertive', 1:'passive', 2:'prosocial'}, 

                 index={0:'purple_child_au01', 1:'purple_child_au02', 2:'purple_child_au04', 3:'purple_child_au05', 

                           4:'purple_child_au06', 5:'purple_child_au07', 6:'purple_child_au09', 7:'purple_child_au10', 

                           8:'purple_child_au12', 9:'purple_child_au14', 10:'purple_child_au15', 11:'purple_child_au17', 

                           12:'purple_child_au20', 13:'purple_child_au23', 14:'purple_child_au25', 15:'purple_child_au26', 

                           16:'purple_child_au45'}, inplace=True)
##Visualise FAU intensity mean per FAU and per label for task engagement

au_plotlabels = ['fau01', 'fau02', 'fau04', 'fau05', 'fau06', 'fau07', 'fau09', 'fau10', 'fau12', 'fau14',

                 'fau15', 'fau17', 'fau20', 'fau23', 'fau25', 'fau26', 'fau45']

y1 = te_mean_au1['adultseeking']

y2 = te_mean_au1['aimless']

y3 = te_mean_au1['goaloriented']

y4 = te_mean_au1['noplay']

x = np.linspace(0, 16, 17, endpoint=True)

plt.scatter(x, y1)

plt.scatter(x, y2, marker=">")

plt.scatter(x, y3, marker="+")

plt.scatter(x, y4, marker="*")

plt.ylim([0.07, 1])

plt.legend(loc='upper center', bbox_to_anchor=(0.2, 1.05))

plt.xticks(np.arange(0, 17, step=1))

plt.xticks(np.arange(17), au_plotlabels, rotation=90)

plt.show()
##Visualise FAU intensity mean per FAU and per label for social engagement

y1 = se_mean_au1['associative']

y2 = se_mean_au1['onlooker']

y3 = se_mean_au1['parallel']

y4 = se_mean_au1['solitary']

x = np.linspace(0, 16, 17, endpoint=True)

plt.scatter(x, y1)

plt.scatter(x, y2, marker=">")

plt.scatter(x, y3, marker="+")

plt.scatter(x, y4, marker="*")

plt.ylim([0.08, 1])

plt.legend(loc='upper center', bbox_to_anchor=(0.28, 1.07))

plt.xticks(np.arange(0, 17, step=1))

plt.xticks(np.arange(17), au_plotlabels, rotation=90)

plt.show()
##Visualise FAU intensity mean per FAU and per label for social attitude

y1 = sa_mean_au1['assertive']

y2 = sa_mean_au1['passive']

y3 = sa_mean_au1['prosocial']

x = np.linspace(0, 16, 17, endpoint=True)

plt.scatter(x, y1)

plt.scatter(x, y2, marker=">")

plt.scatter(x, y3, marker="+")

plt.ylim([0, 1.2])

plt.legend()

plt.xticks(np.arange(0, 17, step=1))

plt.xticks(np.arange(17), au_plotlabels, rotation=90)

plt.show()
##Check representation of labels in the dataset: balanced/unbalanced?



##Task engagement

count_bylabel(child_sample, 'purple_child_task_engagement')



##Social engagement

count_bylabel(child_sample, 'purple_child_social_engagement')



##Social attitude

count_bylabel(child_sample, 'purple_child_social_attitude')
##Divide by label



##Task engagement

te_adultseeking1 = df_bylabel(child_sample, 'purple_child_task_engagement', 'adultseeking')

te_aimless1 = df_bylabel(child_sample, 'purple_child_task_engagement', 'aimless')

te_goaloriented1 = df_bylabel(child_sample, 'purple_child_task_engagement', 'goaloriented')

te_noplay1 = df_bylabel(child_sample, 'purple_child_task_engagement', 'noplay')



##Social engagement

se_associative1 = df_bylabel(child_sample, 'purple_child_social_engagement', 'associative')

se_onlooker1 = df_bylabel(child_sample, 'purple_child_social_engagement', 'onlooker')

se_parallel1 = df_bylabel(child_sample, 'purple_child_social_engagement', 'parallel')

se_solitary1 = df_bylabel(child_sample, 'purple_child_social_engagement', 'solitary')



##Social attitude

sa_assertive1 = df_bylabel(child_sample, 'purple_child_social_attitude', 'assertive')

sa_passive1 = df_bylabel(child_sample, 'purple_child_social_attitude', 'passive')

sa_prosocial1 = df_bylabel(child_sample, 'purple_child_social_attitude', 'prosocial')
##Random under-sampling



##Task engagement

te_aimless_under1 = te_aimless1.sample(29876)

te_goaloriented_under1 = te_goaloriented1.sample(29876)

te_noplay_under1 = te_noplay1.sample(29876)

te_sample_under1 = pd.concat([te_adultseeking1, te_aimless_under1, te_goaloriented_under1, te_noplay_under1], axis=0)



##Frequency of levels of task engagement after under-sampling

count_bylabel(te_sample_under1, 'purple_child_task_engagement')



##Social engagement

se_onlooker_under1 = se_onlooker1.sample(8090)

se_parallel_under1 = se_parallel1.sample(8090)

se_solitary_under1 = se_solitary1.sample(8090)

se_sample_under1 = pd.concat([se_associative1, se_onlooker_under1, se_parallel_under1, 

                             se_solitary_under1], axis=0)



##Frequency of levels of social engagement after under-sampling

count_bylabel(se_sample_under1, 'purple_child_social_engagement')



##Social attitude

sa_assertive_under1 = sa_assertive1.sample(33943)

sa_passive_under1 = sa_passive1.sample(33943)

sa_sample_under1 = pd.concat([sa_prosocial1, sa_assertive_under1, sa_passive_under1], axis=0)



##Frequency of levels of social attitude after under-sampling

count_bylabel(sa_sample_under1, 'purple_child_social_attitude')
##Scatter plot au06 vs au14 by label

lab = {0:'purple_child_au01', 1:'purple_child_au02', 2:'purple_child_au04', 3:'purple_child_au05', 

                4:'purple_child_au06', 5:'purple_child_au07', 6:'purple_child_au09', 7:'purple_child_au10', 

                8:'purple_child_au12', 9:'purple_child_au14', 10:'purple_child_au15', 11:'purple_child_au17', 

                12:'purple_child_au20', 13:'purple_child_au23', 14:'purple_child_au25', 15:'purple_child_au26', 

                16:'purple_child_au45'} 

N = 60

x1 = se_onlooker_under1['purple_child_au09']

y1 = se_parallel_under1['purple_child_au09']

z1 = se_solitary_under1['purple_child_au09']

a1 = se_associative1['purple_child_au09']

x2 = se_onlooker_under1['purple_child_au14']

y2 = se_parallel_under1['purple_child_au14']

z2 = se_solitary_under1['purple_child_au14']

a2 = se_associative1['purple_child_au14']

 

one =np.concatenate((x1,y1,z1,a1))

two =np.concatenate((x2,y2,z2,a2))

 

color_array = ['b'] * 15 + ['g'] * 17 + ['r']*15 + ['y']*15

 

plt.scatter(one, two, c=color_array)

 

plt.xlabel('FAU 09', fontsize=16)

plt.ylabel('FAU 14', fontsize=16)

plt.title('grouped scatter plot - FAU 06 vs FAU 14',fontsize=20)

plt.show()
##Binarize the target variables



##Random under-sampled task engagement

##Binarize

lb = preprocessing.MultiLabelBinarizer()

te_binarized = lb.fit_transform(te_sample_under1['purple_child_task_engagement'])



te_sample_under1['te_adultseeking'] = te_binarized[:, 0]

te_sample_under1['te_aimless'] = te_binarized[:, 1]

te_sample_under1['te_goaloriented'] = te_binarized[:, 2]

te_sample_under1['te_noplay'] = te_binarized[:, 3]



##Drop other target variables

te_sample_under1 = te_sample_under1.drop(['purple_child_task_engagement', 'purple_child_social_engagement', 

                                          'purple_child_social_attitude'], 1)



print(te_sample_under1.columns)



##Random under-sampled social engagement

##Binarize

se_binarized = lb.fit_transform(se_sample_under1['purple_child_social_engagement'])



se_sample_under1['se_associative'] = se_binarized[:, 0]

se_sample_under1['se_onlooker'] = se_binarized[:, 1]

se_sample_under1['se_parallel'] = se_binarized[:, 2]

se_sample_under1['se_solitary'] = se_binarized[:, 3]



##Drop other target variables

se_sample_under1 = se_sample_under1.drop(['purple_child_task_engagement', 'purple_child_social_engagement', 

                                          'purple_child_social_attitude'], 1)



print(se_sample_under1.columns)
##Random under-sampled social attitude

##Binarize

sa_binarized = lb.fit_transform(sa_sample_under1['purple_child_social_attitude'])



sa_sample_under1['sa_assertive'] = sa_binarized[:, 0]

sa_sample_under1['sa_passive'] = sa_binarized[:, 1]

sa_sample_under1['sa_prosocial'] = sa_binarized[:, 2]



##Drop other target variables

sa_sample_under1 = sa_sample_under1.drop(['purple_child_task_engagement', 'purple_child_social_engagement', 

                                          'purple_child_social_attitude'], 1)



print(sa_sample_under1.columns)
##Create train/test split for te_sample_under

te_X_train1, te_X_test1, te_y_train1, te_y_test1 = train_test_split(te_sample_under1[['purple_child_au01',

       'purple_child_au02', 'purple_child_au04', 'purple_child_au05',

       'purple_child_au06', 'purple_child_au07', 'purple_child_au09',

       'purple_child_au10', 'purple_child_au12', 'purple_child_au14',

       'purple_child_au15', 'purple_child_au17', 'purple_child_au20',

       'purple_child_au23', 'purple_child_au25', 'purple_child_au26',

       'purple_child_au45']], te_sample_under1[['te_adultseeking', 'te_aimless', 'te_goaloriented',

       'te_noplay']], test_size=0.2, random_state=1)



print('te_X_train', te_X_train1.shape)

print('te_y_train', te_y_train1.shape)

print('te_X_test', te_X_test1.shape)

print('te_y_test', te_y_test1.shape)



##Create train/test split for se_sample_under

se_X_train1, se_X_test1, se_y_train1, se_y_test1 = train_test_split(se_sample_under1[['purple_child_au01',

       'purple_child_au02', 'purple_child_au04', 'purple_child_au05',

       'purple_child_au06', 'purple_child_au07', 'purple_child_au09',

       'purple_child_au10', 'purple_child_au12', 'purple_child_au14',

       'purple_child_au15', 'purple_child_au17', 'purple_child_au20',

       'purple_child_au23', 'purple_child_au25', 'purple_child_au26',

       'purple_child_au45']], se_sample_under1[['se_associative',  'se_onlooker', 

       'se_parallel', 'se_solitary']], test_size=0.2, random_state=1)



print('se_X_train', se_X_train1.shape)

print('se_y_train', se_y_train1.shape)

print('se_X_test', se_X_test1.shape)

print('se_y_test', se_y_test1.shape)



##Create train/test split for sa_sample_under

sa_X_train1, sa_X_test1, sa_y_train1, sa_y_test1 = train_test_split(sa_sample_under1[['purple_child_au01',

       'purple_child_au02', 'purple_child_au04', 'purple_child_au05',

       'purple_child_au06', 'purple_child_au07', 'purple_child_au09',

       'purple_child_au10', 'purple_child_au12', 'purple_child_au14',

       'purple_child_au15', 'purple_child_au17', 'purple_child_au20',

       'purple_child_au23', 'purple_child_au25', 'purple_child_au26',

       'purple_child_au45']], sa_sample_under1[['sa_assertive', 'sa_passive', 

       'sa_prosocial']], test_size=0.2, random_state=1)



print('sa_X_train', sa_X_train1.shape)

print('sa_y_train', sa_y_train1.shape)

print('sa_X_test', sa_X_test1.shape)

print('sa_y_test', sa_y_test1.shape)
##Create suitable input for the label-powerset approach

te_X_train1 = sparse.csr_matrix(te_X_train1.values).toarray()

te_y_train1 = sparse.csr_matrix(te_y_train1.values).toarray()

te_X_test1 = sparse.csr_matrix(te_X_test1.values).toarray()

te_y_test1 = sparse.csr_matrix(te_y_test1.values).toarray()



se_X_train1 = sparse.csr_matrix(se_X_train1.values).toarray()

se_y_train1 = sparse.csr_matrix(se_y_train1.values).toarray()

se_X_test1 = sparse.csr_matrix(se_X_test1.values).toarray()

se_y_test1 = sparse.csr_matrix(se_y_test1.values).toarray()



sa_X_train1 = sparse.csr_matrix(sa_X_train1.values).toarray()

sa_y_train1 = sparse.csr_matrix(sa_y_train1.values).toarray()

sa_X_test1 = sparse.csr_matrix(sa_X_test1.values).toarray()

sa_y_test1 = sparse.csr_matrix(sa_y_test1.values).toarray()
##Check class balance

print(np.unique(te_y_train1[:, 0], return_counts = True))

print(np.unique(te_y_test1[:, 0], return_counts = True))

print(np.unique(te_y_train1[:, 1], return_counts = True))

print(np.unique(te_y_test1[:, 1], return_counts = True))

print(np.unique(te_y_train1[:, 2], return_counts = True))

print(np.unique(te_y_test1[:, 2], return_counts = True))

print(np.unique(te_y_train1[:, 3], return_counts = True))

print(np.unique(te_y_test1[:, 3], return_counts = True))
##Logistic regression for task engagement labels

clf(te_X_train1, te_y_train1, te_X_test1, te_y_test1, 0, LogisticRegression(solver = 'lbfgs'))

clf(te_X_train1, te_y_train1, te_X_test1, te_y_test1, 1, LogisticRegression(solver = 'lbfgs'))

clf(te_X_train1, te_y_train1, te_X_test1, te_y_test1, 2, LogisticRegression(solver = 'lbfgs'))

clf(te_X_train1, te_y_train1, te_X_test1, te_y_test1, 3, LogisticRegression(solver = 'lbfgs'))
##Logistic regression for social engagement labels

clf(se_X_train1, se_y_train1, se_X_test1, se_y_test1, 0, LogisticRegression(solver = 'lbfgs'))

clf(se_X_train1, se_y_train1, se_X_test1, se_y_test1, 1, LogisticRegression(solver = 'lbfgs'))

clf(se_X_train1, se_y_train1, se_X_test1, se_y_test1, 2, LogisticRegression(solver = 'lbfgs'))

clf(se_X_train1, se_y_train1, se_X_test1, se_y_test1, 3, LogisticRegression(solver = 'lbfgs'))
##Logistic regression for social attitude labels

clf(sa_X_train1, sa_y_train1, sa_X_test1, sa_y_test1, 0, LogisticRegression(solver = 'lbfgs'))

clf(sa_X_train1, sa_y_train1, sa_X_test1, sa_y_test1, 1, LogisticRegression(solver = 'lbfgs'))

clf(sa_X_train1, sa_y_train1, sa_X_test1, sa_y_test1, 2, LogisticRegression(solver = 'lbfgs'))
## Naive bayes for task engagement labels

clf(te_X_train1, te_y_train1, te_X_test1, te_y_test1, 0, GaussianNB())

clf(te_X_train1, te_y_train1, te_X_test1, te_y_test1, 1, GaussianNB())

clf(te_X_train1, te_y_train1, te_X_test1, te_y_test1, 2, GaussianNB())

clf(te_X_train1, te_y_train1, te_X_test1, te_y_test1, 3, GaussianNB())
##Logistic regression for social engagement labels

clf(se_X_train1, se_y_train1, se_X_test1, se_y_test1, 0, GaussianNB())

clf(se_X_train1, se_y_train1, se_X_test1, se_y_test1, 1, GaussianNB())

clf(se_X_train1, se_y_train1, se_X_test1, se_y_test1, 2, GaussianNB())

clf(se_X_train1, se_y_train1, se_X_test1, se_y_test1, 3, GaussianNB())
##Logistic regression for social attitude labels

clf(sa_X_train1, sa_y_train1, sa_X_test1, sa_y_test1, 0, GaussianNB())

clf(sa_X_train1, sa_y_train1, sa_X_test1, sa_y_test1, 1, GaussianNB())

clf(sa_X_train1, sa_y_train1, sa_X_test1, sa_y_test1, 2, GaussianNB())
##Create smaller sample of data for faster computations

##Stack data

#te_train1 = np.hstack((te_X_train1,te_y_train1))

#print(te_train1.shape)

#te_test1 = np.hstack((te_X_test1,te_y_test1))

#print(te_test1.shape)



#se_train1 = np.hstack((se_X_train1,se_y_train1))

#print(se_train1.shape)

#se_test1 = np.hstack((se_X_test1,se_y_test1))

#print(se_test1.shape)



#sa_train1 = np.hstack((sa_X_train1,sa_y_train1))

#print(sa_train1.shape)

#sa_test1 = np.hstack((sa_X_test1,sa_y_test1))

#print(sa_test1.shape)



##Sample data

#te_X_train1 = te_train1[np.random.choice(te_X_train1.shape[0], 6000, replace=False), :]

#print(te_X_train1.shape)

#te_y_train1 = te_X_train1[:, 17:]

#print(te_y_train1.shape)

#te_X_train1 = te_X_train1[:, 0:17]

#print(te_X_train1.shape)



#te_X_test1 = te_test1[np.random.choice(te_X_test1.shape[0], 2000, replace=False), :]

#print(te_X_test1.shape)

#te_y_test1 = te_X_test1[:, 17:]

#print(te_y_test1.shape)

#te_X_test1 = te_X_test1[:, 0:17]

#print(te_X_test1.shape)



#se_X_train1 = se_train1[np.random.choice(se_X_train1.shape[0], 6000, replace=False), :]

#print(se_X_train1.shape)

#se_y_train1 = se_X_train1[:, 17:]

#print(se_y_train1.shape)

#se_X_train1 = se_X_train1[:, 0:17]

#print(se_X_train1.shape)



#se_X_test1 = se_test1[np.random.choice(se_X_test1.shape[0], 2000, replace=False), :]

#print(se_X_test1.shape)

#se_y_test1 = se_X_test1[:, 17:]

#print(se_y_test1.shape)

#se_X_test1 = se_X_test1[:, 0:17]

#print(se_X_test1.shape)



#sa_X_train1 = sa_train1[np.random.choice(sa_X_train1.shape[0], 6000, replace=False), :]

#print(sa_X_train1.shape)

#sa_y_train1 = sa_X_train1[:, 17:]

#print(sa_y_train1.shape)

#sa_X_train1 = sa_X_train1[:, 0:17]

#print(sa_X_train1.shape)



#sa_X_test1 = sa_test1[np.random.choice(sa_X_test1.shape[0], 2000, replace=False), :]

#print(sa_X_test1.shape)

#sa_y_test1 = sa_X_test1[:, 17:]

#print(sa_y_test1.shape)

#sa_X_test1 = sa_X_test1[:, 0:17]

#print(sa_X_test1.shape)
##Task engagement --> 'adultseeking'

PNN(te_X_train1, te_y_train1, te_X_test1, te_y_test1, 0)
##Task engagement --> 'adultseeking'

##Model trained on child-child data, tested on child-robot data

PNN(te_X_train1, te_y_train1, te_X_test, te_y_test, 0)
##Task engagement --> 'aimless'

PNN(te_X_train1, te_y_train1, te_X_test1, te_y_test1, 1)
##Task engagement --> 'aimless'

##Model trained on child-child data, tested on child-robot data

PNN(te_X_train1, te_y_train1, te_X_test, te_y_test, 1)
##Task engagement --> 'goaloriented'

PNN(te_X_train1, te_y_train1, te_X_test1, te_y_test1, 2)
##Task engagement --> 'goaloriented'

##Model trained on child-child data, tested on child-robot data

PNN(te_X_train1, te_y_train1, te_X_test, te_y_test, 2)
##Task engagement --> 'noplay'

PNN(te_X_train1, te_y_train1, te_X_test1, te_y_test1, 3)
##Task engagement --> 'noplay'

##Model trained on child-child data, tested on child-robot data

PNN(te_X_train1, te_y_train1, te_X_test, te_y_test, 3)
##Social engagement --> 'associative'

PNN(se_X_train1, se_y_train1, se_X_test1, se_y_test1, 0)
##Social engagement --> 'associative'

##Model trained on child-child data, tested on child-robot data

PNN(se_X_train1, se_y_train1, se_X_test, se_y_test, 0)
##Social engagement --> 'onlooker'

PNN(se_X_train1, se_y_train1, se_X_test1, se_y_test1, 1)
##Social engagement --> 'onlooker'

##Model trained on child-child data, tested on child-robot data

PNN(se_X_train1, se_y_train1, se_X_test, se_y_test, 1)
##Social engagement --> 'parallel'

PNN(se_X_train1, se_y_train1, se_X_test1, se_y_test1, 2)
##Social engagement --> 'parallel'

##Model trained on child-child data, tested on child-robot data

PNN(se_X_train1, se_y_train1, se_X_test, se_y_test, 2)
##Social engagement --> 'solitary'

PNN(se_X_train1, se_y_train1, se_X_test1, se_y_test1, 3)
##Social engagement --> 'solitary'

##Model trained on child-child data, tested on child-robot data

PNN(se_X_train1, se_y_train1, se_X_test, se_y_test, 3)
##Social attitude --> 'assertive'

PNN(sa_X_train1, sa_y_train1, sa_X_test1, sa_y_test1, 0)
##Social attitude --> 'assertive'

##Model trained on child-child data, tested on child-robot data

PNN(sa_X_train1, sa_y_train1, sa_X_test, sa_y_test, 0)
##Social attitude --> 'passive'

PNN(sa_X_train1, sa_y_train1, sa_X_test1, sa_y_test1, 1)
##Social attitude --> 'passive'

##Model trained on child-child data, tested on child-robot data

PNN(sa_X_train1, sa_y_train1, sa_X_test, sa_y_test, 1)
##Social attitude --> 'prosocial'

PNN(sa_X_train1, sa_y_train1, sa_X_test1, sa_y_test1, 2)
##Social attitude --> 'prosocial'

##Model trained on child-child data, tested on child-robot data

PNN(sa_X_train1, sa_y_train1, sa_X_test, sa_y_test, 2)
##Data to plot Task Engagement per experiment

n_groups = 4

acc_CR = (87.3,78.4,75.2,78.2)

acc_CC = (88.4,84.2,85.1,84.4)

acc_CCCR = (50.9,50.4,50.8,50.5)



##Create plot



mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["#3B5998", "c", "grey"])

classifiers = ['Child-Child', 'Child-Child - Child-Robot', 'Child-Robot']



def bar_group(classes, values, width=0.8):

    total_data = len(values)

    classes_num = np.arange(len(classes))

    for i in range(total_data):

        bars = plt.bar(classes_num - width / 2. + i / total_data * width, values[i], 

                width=width / total_data, align="edge", animated=0.4, label = classifiers[i])

        for rect in bars:

            height = rect.get_height()

            plt.text(rect.get_x() + rect.get_width()/2.0, height, '%d' % int(height), ha='center', va='bottom')

        



group = ['Adultseeking', 'Aimless', 'Goaloriented', 'Noplay']

bar_group(group, [acc_CC, acc_CCCR, acc_CR])

index = np.arange(n_groups)



 

plt.xlabel('Label', fontsize = 20)

plt.ylabel('Balanced accuracy', fontsize = 20)

plt.tick_params(labelsize=18)

plt.title('Task engagement scores', fontsize = 20)

plt.xticks(index, ('Adultseeking', 'Aimless', 'Goaloriented', 'Noplay'), fontsize = 13)

plt.ylim([0,100])

plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.27),

          fancybox=True, ncol=4)



plt.savefig('Ex2_TE.jpg', bbox_inches='tight')



plt.tight_layout()

plt.show()
##Data to plot Social Engagement per experiment

n_groups = 4

acc_CR = (89.8,82.5,76.7,78.7)

acc_CC = (93.6,85.6,81.0,80.7)

acc_CCCR = (54.7,56.0,50.9,50.3)



##Create plot



mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["#3B5998", "c", "grey"])

classifiers = ['Child-Child', 'Child-Child - Child-Robot', 'Child-Robot']



def bar_group(classes, values, width=0.8):

    total_data = len(values)

    classes_num = np.arange(len(classes))

    for i in range(total_data):

        bars = plt.bar(classes_num - width / 2. + i / total_data * width, values[i], 

                width=width / total_data, align="edge", animated=0.4, label = classifiers[i])

        for rect in bars:

            height = rect.get_height()

            plt.text(rect.get_x() + rect.get_width()/2.0, height, '%d' % int(height), ha='center', va='bottom')

        



group = ['Associative', 'Onlooker', 'Parallel', 'Solitary']

bar_group(group, [acc_CC, acc_CCCR, acc_CR])

index = np.arange(n_groups)

 

plt.xlabel('Label', fontsize = 20)

plt.ylabel('Balanced accuracy', fontsize = 20)

plt.tick_params(labelsize=18)

plt.title('Social engagement scores', fontsize = 20)

plt.xticks(index, ('Associative', 'Onlooker', 'Parallel', 'Solitary'), fontsize = 13)

plt.ylim([0,100])

plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.27),

          fancybox=True, ncol=4)



plt.savefig('Ex2_SE.jpg', bbox_inches='tight')



plt.tight_layout()

plt.show()
##Data to plot Social Attitude per experiment

n_groups = 3

acc_CR = (86.3,80.5,85.6)

acc_CC = (89.9,87.0,90.7)

acc_CCCR = (51.3,51.7,49.2)



##Create plot



mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["#3B5998", "c", "grey"])

classifiers = ['Child-Child', 'Child-Child - Child-Robot', 'Child-Robot']



def bar_group(classes, values, width=0.8):

    total_data = len(values)

    classes_num = np.arange(len(classes))

    for i in range(total_data):

        bars = plt.bar(classes_num - width / 2. + i / total_data * width, values[i], 

                width=width / total_data, align="edge", animated=0.4, label = classifiers[i])

        for rect in bars:

            height = rect.get_height()

            plt.text(rect.get_x() + rect.get_width()/2.0, height, '%d' % int(height), ha='center', va='bottom')

        



group = ['Assertive', 'Passive', 'Prosocial']

bar_group(group, [acc_CC, acc_CCCR, acc_CR])

index = np.arange(n_groups)

 

plt.xlabel('Label', fontsize = 20)

plt.ylabel('Balanced accuracy', fontsize = 20)

plt.tick_params(labelsize=18)

plt.title('Social attitude scores', fontsize = 20)

plt.xticks(index, ('Assertive', 'Passive', 'Prosocial'), fontsize = 13)

plt.ylim([0,100])

plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.27),

          fancybox=True, ncol=3)



plt.savefig('Ex2_SA.jpg', bbox_inches='tight')



plt.tight_layout()

plt.show()