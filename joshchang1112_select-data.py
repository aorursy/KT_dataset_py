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
data = pd.read_excel('../input/adni-excel/DXSUM_PDXCONV_ADNIALL.xlsx')
data
# read data

data = pd.read_excel('../input/adni-excel/DXSUM_PDXCONV_ADNIALL.xlsx', names=data.iloc[1, :])
data
data = data[2:]
print(data.iloc[2,0])
data
# 找出有多少個受試者(看RID的數量)

RID_dict = {}

print(RID_dict)

for i in range(11931):

    if data.iloc[i, 2] not in RID_dict:

        RID_dict[data.iloc[i, 2]] = data.iloc[i, :]

      

    else:

        RID_dict[data.iloc[i, 2]] = pd.concat([RID_dict[data.iloc[i, 2]], data.iloc[i, :]])

print(len(RID_dict))
RID_dict[44]['VISCODE2']
# 挑選出是baseline且是MCI的受試者

baseline_mci = data.loc[(data['VISCODE2'] == 'bl') & ( (data['DXCURREN'] == 2) |  (data['DXCHANGE'] == 2)  | (data['DIAGNOSIS'] == 2)) & (data['DXMDUE'] == 1) ]['RID']
baseline_mci
# 去檢查到底有沒有變AD 還是持續MCI 還是有轉變

count = 0

change = 0

none = 0

MCI_AD = []



for i in range(959):

    #print(type(RID_dict[baseline_mci.iloc[i]]['DXCURREN']))

    if isinstance(RID_dict[baseline_mci.iloc[i]]['DXCURREN'], int):

        #print(RID_dict[baseline_mci.iloc[i]]['DXCURREN'])

        continue

    if isinstance(RID_dict[baseline_mci.iloc[i]]['DXCURREN'], float):

        continue

    change = 0

    none = 0

    #print(RID_dict[baseline_mci.iloc[i]]['DXCURREN'])

    #print(RID_dict[baseline_mci.iloc[i]]['DXCHANGE'])

    #print(RID_dict[baseline_mci.iloc[i]]['DIAGNOSIS'])

    

    for j in range(len(RID_dict[baseline_mci.iloc[i]]['DXCURREN'])):

        

        if j != len(RID_dict[baseline_mci.iloc[i]]['DXCURREN']) - 1:

            if change == 0 and RID_dict[baseline_mci.iloc[i]]['DXCURREN'].iloc[j] == 2:

                continue

            elif change == 0 and RID_dict[baseline_mci.iloc[i]]['DXCURREN'].iloc[j] == 3:

                change = 1

                continue

            elif change == 1 and RID_dict[baseline_mci.iloc[i]]['DXCURREN'].iloc[j] == 3:

                continue

            elif pd.isna(RID_dict[baseline_mci.iloc[i]]['DXCURREN'].iloc[j]):

                for k in range(j, len(RID_dict[baseline_mci.iloc[i]]['DXCHANGE'])):

                    if change == 0 and RID_dict[baseline_mci.iloc[i]]['DXCHANGE'].iloc[k] == 2:

                        continue

                    elif change == 0 and RID_dict[baseline_mci.iloc[i]]['DXCHANGE'].iloc[k] == 5:

                        change = 1

                        continue

                    elif change == 1 and RID_dict[baseline_mci.iloc[i]]['DXCHANGE'].iloc[k] == 3:

                        continue

                    elif pd.isna(RID_dict[baseline_mci.iloc[i]]['DXCHANGE'].iloc[k]):

                        for m in range(k, len(RID_dict[baseline_mci.iloc[i]]['DIAGNOSIS'])):

                            if change == 0 and RID_dict[baseline_mci.iloc[i]]['DIAGNOSIS'].iloc[m] == 2:

                                continue

                            elif change == 0 and RID_dict[baseline_mci.iloc[i]]['DIAGNOSIS'].iloc[m] == 3:

                                change = 1

                                continue

                            elif change == 1 and RID_dict[baseline_mci.iloc[i]]['DIAGNOSIS'].iloc[m] == 3:

                                continue

                            else:

                                none = 1

                                break

                    else:

                        none = 1

                        break

            elif none == 1:

                break

            else:

                #print(RID_dict[baseline_mci.iloc[i]]['DXCURREN'])

                break

                

        if none == 0 and change == 0 and j == len(RID_dict[baseline_mci.iloc[i]]['DXCURREN']) - 1:

            count += 1

            MCI_AD.append(baseline_mci.iloc[i])

            #print(count)

        

print(count)
MCI_AD