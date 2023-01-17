# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/Iris.csv")

data.head()
import operator

import itertools



def add_interactions(dataframe, uptowhichdegree):

# seriously not perfect, ignores chain rules and equality by variable exchange

    if type(dataframe) != type(pd.DataFrame()):

        print("Data frame required")

        return

    df_new = dataframe._get_numeric_data()

    df_non_numeric = np.setdiff1d(dataframe.columns, df_new.columns, assume_unique=True)

    

    operators = [operator.add, operator.sub, operator.mul, operator.truediv]

    

    original_cols = df_new.columns

    

    for i in range(2,uptowhichdegree+1):

        col_comb = itertools.combinations(original_cols, i)

        for item in col_comb:

            for col in item:

                if col not in original_cols:

                    print(item)

            op_comb = itertools.combinations(operators, len(item)-1)

            for oper_tuple in op_comb:

                #print(oper_tuple)

                temp = 0

                new_series = pd.Series()

                new_col = ""

                for i in range(0,len(item)-1):

                    #print(i)

                    if(temp == 0):

                        new_series = oper_tuple[i](df_new[item[i]], df_new[item[i+1]])

                        #print(oper_tuple[i].__name__)

                        #print(item[i])

                        #print(item[i+1])

                        new_col += item[i] + "_" + oper_tuple[i].__name__ + "_" + item[i+1]

                        temp = 1

                    else:

                        new_series = oper_tuple[i](new_series, df_new[item[i+1]])

                        #print(oper_tuple[i].__name__)

                        #print(new_series.name)

                        #print(item[i+1])

                        new_col += "_" + oper_tuple[i].__name__ + "_" + item[i+1]

                #print("Neue Spalte: " + new_col)

                if new_col not in df_new:

                    df_new[new_col] = new_series

                    

    for col in df_non_numeric:

        df_new[col] = dataframe[col]

    

    return df_new
data = add_interactions(data, 4)

data