# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#load traindata and testdata

test_data = pd.read_csv('../input/test.csv',index_col=False)

train_data = pd.read_csv ('../input/train.csv', index_col=False)



print(test_data.head())

print(train_data.head())
#filter function one



def drop_and_purge(table,colums_to_drop,purge=True):

    for col in colums_to_drop:

        try:

            del table[col]

        except KeyError as exp:

            print('colum',exp,'not found in table')

    if purge:

        table.dropna()
#filter funtion two

def quantise(table,dict_to_quantize):

    '''takes table and dict {column_name:[{value_in_table:value_to_replace_with}]}

    example:- quantise(table,{Sex:[{male:1},{female:0}]}) will replace male and female in Sex column of table with 1 and 0'''

    for val in dict_to_quantize:

        for string_to_replace in dict_to_quantize[val]:

            value_to_replace_with = dict_to_quantize[val][string_to_replace]

            table[val].replace(string_to_replace,value_to_replace_with,inplace=True)
#test drop_and_purge functions

coloums_to_drop = ['Name','Ticket','Cabin','Fare']

drop_and_purge(test_data,coloums_to_drop,purge=False)

drop_and_purge(train_data,coloums_to_drop)
#use quantise function



quantise(test_data,{'Sex':{'male':1,'female':0}})

quantise(test_data,{'Embarked':{'Q':0,'C':1,'S':2}})

quantise(train_data,{'Sex':{'male':1,'female':0}})

quantise(train_data,{'Embarked':{'Q':0,'C':1,'S':2}})

print(test_data.head())

print(train_data.head())
#function for visualisation of data

def make_pivot (param1, param2):

    df_slice = train_data[[param1, param2, 'PassengerId']]

    slice_pivot = df_slice.pivot_table(index=[param1], columns=[param2],aggfunc=np.size, fill_value=0)

    print('correlation:',np.corrcoef(train_data[param1],train_data[param2])[0,1])

    p_chart = slice_pivot.plot.bar()

    for p in p_chart.patches:

        p_chart.annotate(str(p.get_height()), (p.get_x() * 1.05, p.get_height() * 1.01))

    

    return slice_pivot

    return p_chart
make_pivot ('Survived','Pclass')

make_pivot('Survived','Sex')
#as we can see the high corelation between sex and survived lets create a model based on following data

data_dist_sex = train_data['Sex'].value_counts()

data_dist_pclass = train_data['Pclass'].value_counts()

#print(data_dist_sex)

#print(data_dist_pclass)





male_survived = 0

female_survived = 0

pclass_1_survived = 0

pclass_2_survived = 0

pclass_3_survived = 0

male_pclass_1_survived = 0

female_pclass_1_survived = 0

male_pclass_2_survived = 0

female_pclass_2_survived = 0

male_pclass_3_survived = 0

female_pclass_3_survived = 0

for index, row in train_data.iterrows():

    if row['Survived'] == 1:

        if row['Sex'] == 1:

            male_survived = male_survived + 1

            if row['Pclass'] == 1:

                male_pclass_1_survived = male_pclass_1_survived + 1

                pclass_1_survived = pclass_1_survived + 1

            elif row['Pclass'] == 2:

                male_pclass_2_survived = male_pclass_2_survived + 1

                pclass_2_survived = pclass_2_survived + 1

            elif row['Pclass'] == 3:

                male_pclass_3_survived = male_pclass_3_survived + 1

                pclass_3_survived = pclass_3_survived + 1

            else:

                "pclass not defined"

        else:

            female_survived = female_survived + 1

            if row['Pclass'] == 1:

                female_pclass_1_survived = female_pclass_1_survived + 1

                pclass_1_survived = pclass_1_survived + 1

            elif row['Pclass'] == 2:

                female_pclass_2_survived = female_pclass_2_survived + 1

                pclass_2_survived = pclass_2_survived + 1

            elif row['Pclass'] == 3:

                female_pclass_3_survived = female_pclass_3_survived + 1

                pclass_3_survived = pclass_3_survived + 1

            else:

                "pclass not defined"



male_pclass_1 = len(train_data[(train_data['Pclass']==1) & (train_data['Sex']==1)])

male_pclass_2 = len(train_data[(train_data['Pclass']==2) & (train_data['Sex']==1)])

male_pclass_3 = len(train_data[(train_data['Pclass']==3) & (train_data['Sex']==1)])

female_pclass_1 = len(train_data[(train_data['Pclass']==1) & (train_data['Sex']==0)])

female_pclass_2 = len(train_data[(train_data['Pclass']==2) & (train_data['Sex']==0)])

female_pclass_3 = len(train_data[(train_data['Pclass']==3) & (train_data['Sex']==0)])



def survivedCalc(prob):

    random = np.random.choice([0, 1], p=[1-prob, prob])

    return random



perct_male_survived = male_survived/data_dist_sex[1]

perct_female_survived = female_survived/data_dist_sex[0]

perct_pclass_1_survived = pclass_1_survived/data_dist_pclass[1]

perct_pclass_2_survived = pclass_2_survived/data_dist_pclass[2]

perct_pclass_3_survived = pclass_3_survived/data_dist_pclass[3]

perct_male_pclass_1_survived = male_pclass_1_survived/male_pclass_1

perct_male_pclass_2_survived = male_pclass_2_survived/male_pclass_2

perct_male_pclass_3_survived = male_pclass_3_survived/male_pclass_3

perct_female_pclass_1_survived = female_pclass_1_survived/female_pclass_1

perct_female_pclass_2_survived = female_pclass_2_survived/female_pclass_2

perct_female_pclass_3_survived = female_pclass_3_survived/female_pclass_3



print('probability:','prob_male_survived','prob_female_survived','prob_pclass_1_survived','prob_pclass_2_survived','prob_pclass_3_survived','prob_male_pclass_1_survived','prob_male_pclass_2_survived','prob_male_pclass_3_survived','prob_female_pclass_1_survived','prob_female_pclass_2_survived','prob_female_pclass_3_survived','\n',perct_male_survived,perct_female_survived,perct_pclass_1_survived,perct_pclass_2_survived,perct_pclass_3_survived,perct_male_pclass_1_survived,perct_male_pclass_2_survived,perct_male_pclass_3_survived,perct_female_pclass_1_survived,perct_female_pclass_2_survived,perct_female_pclass_3_survived)
def calculate_probability(passanger):

    cummalitive_prob = None

    if row['Sex'] == 1:

        if row['Pclass'] == 1:

            cummalitive_prob = perct_male_pclass_1_survived

        elif row['Pclass'] == 2:

            cummalitive_prob = perct_male_pclass_2_survived

        else:

            cummalitive_prob = perct_male_pclass_3_survived

    else:

        if row['Pclass'] == 1:

            cummalitive_prob = perct_female_pclass_1_survived

        elif row['Pclass'] == 2:

            cummalitive_prob = perct_female_pclass_2_survived

        else:

            cummalitive_prob = perct_female_pclass_3_survived

    return cummalitive_prob
for index, row in test_data.iterrows():

    prob = calculate_probability(row)

    val = survivedCalc(prob)

    test_data = test_data.set_value(index,'Survived',int(val))

    

print(test_data.head(20))
#this submission has an accuracy of 0.667

gensubmission = test_data[['PassengerId', 'Survived']].copy()

gensubmission.Survived = gensubmission.Survived.astype(int)

gensubmission.to_csv('./DSsubmission.csv',index=False)