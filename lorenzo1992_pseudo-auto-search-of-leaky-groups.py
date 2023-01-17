import numpy as np # math
import pandas as pd # data
pd.options.display.max_columns = 999
pd.options.display.max_rows = 999
from itertools import compress # for list

import os
print(os.listdir("../input"))
# import train
train = pd.read_csv('../input/train.csv', index_col=0)
cols = list(train.columns)[1:]

# matrix of data values
matrix_train = train[cols].values
# new df
df_train = pd.DataFrame(index=train.index)
df_train['target'] = train.target.values
df_train['in_df'] = 0
%%time

if 1==1:
    # count the number of columns for each target value
    for enum, idx in enumerate(df_train.index):
        df_train.loc[idx, 'in_df'] = sum((matrix_train == df_train.target[enum]).sum(axis=0) > 0)

    df_train.to_csv('df_train.csv')

df_train = pd.read_csv('df_train.csv', index_col=0)
df_train.head()
df_train.sort_values(by = 'in_df', inplace = True)
df_train.hist(column='in_df',bins=50, figsize = (15,4))
giba_rows = ['7862786dc','c95732596','16a02e67a','ad960f947','8adafbb52','fd0c7cfc2','a36b78ff7','e42aae1b8','0b132f2c6',
             '448efbb28','ca98b17ca','2e57ec99f','fef33cb02']

df_train.loc[giba_rows,:]
%%time

if 1==1:
    df_analysis = pd.DataFrame(index = df_train.in_df.unique())
    df_analysis['sum_in_sub_df'] = 0

    for idx in df_analysis.index:

        # print('analysing index:', idx, ' ...')
        # choose rows
        df_train_sub = df_train[df_train.in_df == idx]

        # select rows & associated columns
        train_sub = train.loc[list(df_train_sub.index),:]
        train_sub['in_sub_df'] = 0

        # matrix of sub df
        matrix_sub_train = train_sub[cols].values

        # count the number of columns for each target value in the train_sub
        for enum, idx_2 in enumerate(train_sub.index):
            train_sub.loc[idx_2, 'in_sub_df'] = sum((matrix_sub_train == train_sub.target[enum]).sum(axis=0) > 0)

        df_analysis.loc[idx, 'sum_in_sub_df'] = train_sub.in_sub_df.values.sum()
        # print('sum is: ', df_analysis.loc[idx, 'sum_in_sub_df'])

    df_analysis.to_csv('df_analysis.csv')

df_analysis = pd.read_csv('df_analysis.csv', index_col=0).sort_index()
df_analysis
# choose all possible first rows
df_train_sub = df_train[df_train.in_df == 37]
# df_train_sub = df_train[(df_train.in_df == 35) | (df_train.in_df == 36) | (df_train.in_df == 37)]

# select rows and columns
train_sub = train.loc[list(df_train_sub.index),:]
print(train_sub.shape) # 27 possible first rows
# initialize dataframe in which we are going to save the found leaky groups
df_matches = pd.DataFrame(columns=['in_df', 
                                   'first_row', 'first_row_pt2', 
                                   'second_row', 'second_row_pt2', 'third_row',
                                   'first_column', 'second_column', 'third_column'])
%%time

#for index_row_0, row_target_to_find in enumerate(list(train_sub.index)):
for index_row_0, row_target_to_find in enumerate(list(train_sub.index)[9:10]):

    target_to_find = train.loc[row_target_to_find, 'target']
    rows_found = train.index[np.sum((matrix_train == target_to_find), axis=1)>0]
    
    print('\n')
    print('*'*110)
    print('first_row number {}: {}'.format(index_row_0, row_target_to_find))
    print('num possible next rows:', len(rows_found))
    
    for idx_row, row in enumerate(rows_found):

        print('\n')
        print('*'*30)
        print('searcing for second possible row number {}: {}'.format(idx_row,row), '\n')
        # select row:
        row_selected = row

        # columns of first row found
        cols_found_of_row_temp = list(compress(cols, train.loc[row_selected, cols].values == target_to_find))
        target_to_search_second_row = train.loc[row_selected, 'target']
        print('target_to_search_second_row:', target_to_search_second_row)
        cols_found_of_row = []
        for col in cols_found_of_row_temp:
            if sum(train[col].values == target_to_search_second_row) >= 1:
                cols_found_of_row.append((col, sum(train[col].values == target_to_search_second_row)))
        print('total columns found:', len(cols_found_of_row))
        print('that is/are:', cols_found_of_row)

        if len(cols_found_of_row) > 0:
            for first_column, _ in cols_found_of_row:
                all_possible_third_rows = list(train.index[train[first_column].values == target_to_search_second_row])
                print('all_possible_third_rows:', all_possible_third_rows)
                
                value_to_search = train.loc[row_target_to_find, first_column]
                print('value_to_search:', value_to_search)
                
                # first set of admissible second_columns:
                cols_both_values_temp = []
                for idx, col in enumerate(cols):
                    temp_1 = 1*(sum(matrix_train[:,idx] == value_to_search)>=1)
                    if temp_1 > 0:
                        temp_2 = 1*(sum(matrix_train[:,idx] == target_to_find)>=1)
                        if temp_2 > 0:
                            temp_3 = 1*(sum(matrix_train[:,idx] == target_to_search_second_row)>=1)
                            if temp_3 > 0:
                                cols_both_values_temp.append(col)
                # print('possible second columns temp:', cols_both_values_temp)
                
                for index_third_row, third_row in enumerate(all_possible_third_rows):
                    print('third row n {} is: {}'.format(index_third_row, third_row))
                    target_to_search_third_row = train.loc[third_row, 'target']
                    print('target_to_search_third_row:', target_to_search_third_row)

                    # now i search for other columns containing value_to_search, target_to_find e the target of the new row!!
                    cols_both_values = []
                    for idx, col in enumerate(cols_both_values_temp):
                        occurrences = 1*(sum(train[col] == target_to_search_third_row)>=1)
                        if occurrences >= 1:
                            cols_both_values.append(col)
                    
                    print('All possible second columns are:', cols_both_values)
                    for second_column in cols_both_values:
                        if second_column != first_column:

                            # check if E4 is in previous column (D3)
                            coeff_e4 = train.loc[row_selected, second_column]
                            first_row_pt2_temp_0 = list(train.index[train[first_column].values == coeff_e4])
                            first_row_pt2_temp_1 = []
                            for temp_first_row_pt2 in first_row_pt2_temp_0:
                                if train.loc[row_target_to_find, first_column] == train.loc[temp_first_row_pt2,second_column]:
                                    first_row_pt2_temp_1.append(temp_first_row_pt2)
                            
                            #print('all possible first_row_pt2_temp_1:', first_row_pt2_temp_1)
                            # for all possible first_row_pt2_temp_1 check wheter its target is in first and second column
                            first_row_pt2 = []
                            possible_second_row_pt2 = []
                            for temp_first_row_pt2 in first_row_pt2_temp_1:
                                target_temp = train.loc[temp_first_row_pt2, 'target']
                                # target_temp in third row & second_column?
                                if train.loc[third_row, second_column] == target_temp:
                                    # target_temp in any second row pt2 & first column?
                                    possible_second_row_pt2_temp = list(compress(list(train.index), train[first_column] == target_temp))
                                    if len(possible_second_row_pt2_temp) >= 1:
                                        target_temp_2 = train.loc[row_selected, first_column]
                                        # print('target_temp_2:', target_temp_2)
                                        possible_second_row_pt2_temp_2 = list(compress(list(train.index), train[second_column] == target_temp_2))
                                        # intersection:
                                        possible_second_row_pt2_per_first_row_pt2 = list(set(possible_second_row_pt2_temp).intersection(set(possible_second_row_pt2_temp_2)))
                                        if len(possible_second_row_pt2_per_first_row_pt2) >= 1:
                                            first_row_pt2.append(temp_first_row_pt2)
                                            possible_second_row_pt2.append(possible_second_row_pt2_per_first_row_pt2)
                            #print('all possible possible_second_row_pt2:', possible_second_row_pt2)

                            # save all possible matches
                            for idx_temp_first_row_pt2 ,temp_first_row_pt2 in enumerate(first_row_pt2):
                                for temp_second_row_pt2 in list(possible_second_row_pt2[idx_temp_first_row_pt2]):
                                    # now i search a third column!!
                                    cols_1 = list(compress(train.columns, 
                                                           train.loc[temp_first_row_pt2,:] == train.loc[row_target_to_find, second_column]))
                                    cols_2 = list(compress(train.columns, 
                                                           train.loc[row_selected,:] == train.loc[temp_first_row_pt2,second_column]))
                                    cols_3 = list(compress(train.columns,
                                                           train.loc[temp_second_row_pt2,:] == train.loc[row_selected,second_column]))
                                    cols_4 = list(compress(train.columns,
                                                           train.loc[third_row,:] == train.loc[temp_second_row_pt2,second_column]))
                                    # print(cols_1,cols_2,cols_3,cols_4)
                                    possible_third_columns = list(set(cols_1).intersection(set(cols_2)).intersection(set(cols_3)).intersection(set(cols_4)))
                                    # print('CALCOLATO THIRD COLUMNS:',possible_third_columns)
                                    for third_column in possible_third_columns:
                                        # print(third_column)
                                        df_matches.loc[df_matches.shape[0]] = [35, 
                                                                               row_target_to_find,
                                                                               temp_first_row_pt2,
                                                                               row_selected, 
                                                                               temp_second_row_pt2, 
                                                                               third_row, 
                                                                               first_column, 
                                                                               second_column,
                                                                               third_column]
                                        print('FOUND {} SOLUTION: {}'.format(df_matches.shape[0], df_matches.loc[df_matches.shape[0]-1,:].values))

# save
df_matches.to_csv('df_matches.csv')
for idx in range(df_matches.shape[0]):
    rows_found = list(df_matches.loc[idx,['first_row','first_row_pt2','second_row','second_row_pt2','third_row']])
    cols_found = list(['target'] + list(df_matches.loc[idx,['first_column', 'second_column', 'third_column']]))
    print('*'*50)
    print(train.loc[rows_found,cols_found])
    print('*'*50)
