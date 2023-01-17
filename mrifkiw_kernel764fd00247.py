import pandas as pd

import math

import os
data = pd.read_excel ('../input/datasoleh/TUBES SHOLEH.xlsx')
data.head()
def main():

    data_H = data['emazeromean']

    data_float = []

    for item in data_H:

        # print("INI ITEM DONG: "+item)

        data_float.append(float(item))

    negative = []

    positive = []

    data_float.pop(0)

    min_val = True

   

    for item in data_float:

        if min_val == True:

            if item < 0:

                negative.append(item)

            else:

                min_val = False

                positive.append(item)

        else:

            if item > 0:

                positive.append(item)

            else:

                min_val = True

                positive.sort(reverse = True)

                negative.sort()

                jumlah = abs(negative[0]) + positive[0]

                print("Puncak = " + str(positive[0]) + " cm")

                print("Lembah = " + str(negative[0]) + " cm")

                print("H = " + str(jumlah) + " cm")

                print("--------------------------------------")

                negative = []

                positive = []

                negative.append(item)
main()