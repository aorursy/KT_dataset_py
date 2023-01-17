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
def smooze_output(df,div_num):

    # df にデータフレーム

    #div_numは割る数

    for label, items in df.iteritems():    

        for j in range(len(items)):

            x=items[j]

            nums=[]

            one_cut=1/div_num

            z=0

            while z<1:

                nums.append(z)

                z=z+one_cut

            nums.append(1)    



            for i in range(len(nums)):

                if x<=nums[i]:

                    if i==0:

                        x=0

                        break

                    diff_first=abs(x-nums[i])  #i and i-1 is compared(i is bigger)

                    diff_second=abs(x-nums[i-1])    

                    if diff_first>=diff_second:

                        x=nums[i-1]

                        break

                    if diff_second>=diff_first:

                        x=nums[i]

                        break

                #print(df[label][j]) 

                #print(x)

            df[label][j]=float(x)

    return df
#print(smooze_output(0.234))

#print(smooze_output(0))
df=pd.DataFrame({'a' : [ 1,0.2233233,0.3],'b' : [0.5,0.3423,0.15],'c' : [0.545,0.3434,0.3423]},index=['A1','B1','C1'])

#df.applymap(smooze_output)

df=smooze_output(df,90)

print(df['a'])
for label, items in df.iteritems():

    print("label : ", label)

    print("items :\n", items[1])

    print("-----------\n")