import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
raw_file_test=pd.read_csv('../input/Test.csv')
raw_file_test
for i in raw_file_test.index:
    if raw_file_test.loc[i,'WS'] == 'J':
        raw_file_test.loc[i,'WS'] = 1
    if raw_file_test.loc[i,'WS'] == 'K':
        raw_file_test.loc[i,'WS'] = 2
    if raw_file_test.loc[i,'WS'] == 'L':
        raw_file_test.loc[i,'WS'] = 3
    if raw_file_test.loc[i,'WS'] == 'M':
        raw_file_test.loc[i,'WS'] = 4
    if raw_file_test.loc[i,'WS'] == 'N':
        raw_file_test.loc[i,'WS'] = 5
    if raw_file_test.loc[i,'WS'] == 'O':
        raw_file_test.loc[i,'WS'] = 6
    if raw_file_test.loc[i,'WS'] == 'P':
        raw_file_test.loc[i,'WS'] = 7
    if raw_file_test.loc[i,'WS'] == 'Q':
        raw_file_test.loc[i,'WS'] = 8
    if raw_file_test.loc[i,'WS'] == 'R':
        raw_file_test.loc[i,'WS'] = 9
    if raw_file_test.loc[i,'WS'] == 'S':
        raw_file_test.loc[i,'WS'] = 10
    if raw_file_test.loc[i,'WS'] == 'T':
        raw_file_test.loc[i,'WS'] = 11
raw_file_test
processed_test=raw_file_test.drop(['MCAT Total','Probablitiy of acceptance'], axis=1)
processed_test
for i in processed_test.index:
    if 'Bio' in processed_test.loc[i,'Undergrad Major'] or 'Chem' in processed_test.loc[i, 'Undergrad Major'] or 'Psy' in processed_test.loc[i, 'Undergrad Major']:
        processed_test.loc[i,'Undergrad Major Classification'] = 'Bio/Chem/Pys'
    else:
        processed_test.loc[i,'Undergrad Major Classification'] = 'Other'
processed_test
processed_test=processed_test.drop(['Undergrad Major'],axis=1)
category_variables=['Undergrad Major Classification']
for var in category_variables:
    cat_list='var'+'_'+var
    cat_list=pd.get_dummies(processed_test[var],prefix=var)
    raw_data1=processed_test.join(cat_list)
    processed_test=raw_data1
processed_test=processed_test.drop(['Undergrad Major Classification'],axis=1)
processed_test_x=pd.DataFrame(processed_test,columns=['CU BCPM GPA','CU AO GPA','CU CUM GPA','VR','PS','WS','BS','Undergrad Major Classification_Bio/Chem/Pys','Undergrad Major Classification_Other'])
processed_test
