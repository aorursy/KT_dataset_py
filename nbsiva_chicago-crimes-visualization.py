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
def create_data_and(data1, data2, compare_param):

    ind = np.arange(len(data1))



    result = []

    for i in ind:

        if(data2[i] == compare_param):

            result.append(data1[i])

            

    return result
def plot_bar_chart(data, title):



    dict = {}



    for row in data:

        if(row in dict.keys()):

            dict[row] = dict[row] + 1

        else:

            dict[row] = 1



    X_VAL = [];

    Y_VAL = [];







    for key in dict.keys():

        X_VAL.append(key);

        Y_VAL.append(dict[key])



    ind = np.arange(len(X_VAL))



    plt.barh(ind, Y_VAL, align='center', alpha=0.5)

    plt.yticks(ind, X_VAL)

    plt.tight_layout()

    plt.title(title)

    plt.xlim((0, max(Y_VAL)+10))



    plt.show()
X_VAL = [1,2,3,4,5]

Y_VAL = [1,2,3,4,5]



ind = np.arange(len(X_VAL))



plt.barh(ind, Y_VAL, align='center', alpha=0.5)

plt.yticks(ind, X_VAL)

plt.tight_layout()

plt.title("title")

plt.xlim((0, max(Y_VAL)+10))



plt.show()
data_frame = pd.read_csv("../input/Chicago_Crimes_2001_to_2004.csv", header=None, error_bad_lines=False);

offense_type = data_frame.values[:,6]

plot_bar_chart(offense_type, "Distribution of Crime Type")
offense_location = data_frame.values[:,8]

width = 6

height = 22

plt.figure(figsize=(width, height))



plot_bar_chart(offense_location, "Distribution of Location")
offense_arrest = data_frame.values[:,9]

plot_bar_chart(offense_arrest, "Distribution of Arrest")

offense_year = data_frame.values[:,18]

plot_bar_chart(offense_year, "Crime Distribution based on Year")

type_arrest_true = create_data_and(offense_type, offense_arrest, True)

plot_bar_chart(type_arrest_true, "Crime Type and Arrested")

type_arrest_false = create_data_and(offense_type, offense_arrest, False)

plot_bar_chart(type_arrest_false, "Crime Type and Not Arrested")

offense_location_arrest_true = create_data_and(offense_location, offense_arrest, True)



width = 6

height = 22

plt.figure(figsize=(width, height))





plot_bar_chart(offense_location_arrest_true, "Crime Location and Arrested")

offense_location_arrest_true = create_data_and(offense_location, offense_arrest, False)



width = 6

height = 22

plt.figure(figsize=(width, height))





plot_bar_chart(offense_location_arrest_true, "Crime Location and Not Arrested")
