import csv
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn import svm
import sqlite3
import sys

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
# Define functions
def read_csv_data_file():
    f = open("../input/data.csv", "r")
    csv_data = csv.reader(f, delimiter=",")
    dataset = []
    data = []
    target = []
    i = 0
    for row in csv_data:
        # Skip first row with column names
        # Skip rows with missing shot_made_flag
        if i != 0 and row[14] != '':
            f0 = row[0]         # action_type
            f1 = row[1]         # combined_shot_type
            f2 = int(row[2])    # game_event_id
            f3 = int(row[3])    # game_id
            f4 = float(row[4])  # lat
            f5 = int(row[5])    # loc_x
            f6 = int(row[6])    # loc_y
            f7 = float(row[7])  # lon
            f8 = int(row[8])    # minutes_remaining
            f9 = int(row[9])    # period
            f10 = int(row[10])  # playoffs
            f11 = row[11]       # season
            f12 = int(row[12])  # seconds_remaining
            f13 = int(row[13])  # shot_distance
            f14 = int(row[14])  # shot_made_flag (predicting this)
            f15 = row[15]       # shot_type
            f16 = row[16]       # shot_zone_area
            f17 = row[17]       # shot_zone_basic
            f18 = row[18]       # shot_zone_range
            f19 = int(row[19])  # team_id
            f20 = row[20]       # team_name
            f21 = row[21]       # game_date
            f22 = row[22]       # matchup
            f23 = row[23]       # opponent
            f24 = int(row[24])  # shot_id
            #t = (f0, f1, f2, f3, f4, f5, f6, f7, f8, f9,
            #     f10, f11, f12, f13, f14, f15, f16, f17, f18, f19,
            #     f20, f21, f22, f23, f24)
            t = (f5, f6)
            data.append(t)
            target.append(f14)
        i += 1
    dataset.append([data, target])
    f.close()
    return dataset
    
def create_plot(xdata, ydata, sdata):
    plt.figure(num=1, figsize=(12, 12))
    plt.axis([-250, 250, -50, 450])
    for i in range(len(sdata)):
        if sdata[i] == 1:
            color_val = 'g'
        else:
            color_val = 'r'
        plt.plot(xdata[i], ydata[i], marker='o', color=color_val, linestyle='None')
    plt.show()

d = read_csv_data_file()
print(d[0][0][0], '\t', d[0][1][0])
print(d[0][0][1], '\t', d[0][1][1])
print(d[0][0][25696], '\t', d[0][1][25696])
clf = svm.SVC(gamma=0.001, C=100.0)
clf.fit(d[0][0][0:10], d[0][1][0:10])
result = clf.predict(np.array([[70, 70]]))
print(result)
#create_plot(d[0], d[1], d[2])
