# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

import numpy as np
import glob
import matplotlib.pyplot as plt
from pylab import rcParams

rcParams['figure.figsize'] = 10, 15
font = {'family': 'serif',
        'color': 'darkred',
        'weight': 'normal',
        'size': 16,
        }


def add_node(file, name, x, y, label):
    line = f'\\node ({name}) at ({x}mm,{y}mm) {{{label}}};\n'
    file.write(line)


def add_arc(file, name_1, name_2, in_d, out_d, col):
    line = f'\\draw[-,> = latex,semithick,{col}] ({name_1}) to ({name_2});\n'
    # line = f'\\draw[-,> = latex,semithick] ({name_1}) to [out={out_d},in={in_d}]({name_2});\n'
    file.write(line)


def get_x(time_diff):
    return time_diff.days * 24 * 60 + time_diff.seconds / 3600


def get_y(station, y_max, y_fac):
    if station in station2y.keys():
        return station2y[station], y_max
    else:
        y_max += y_fac
        station2y[station] = round(y_max, 2)
        return round(y_max, 2), y_max


def add_duty(duty, y_max):
    x1 = round(get_x(duty.DutyStartTimeUTC - ref_time) * x_fac, 2)
    y1, y_max = get_y(duty.StartAirport, y_max, y_fac)
    name_1 = f'{duty.DutyID}_1'
    add_node(file, name_1, x1, y1, '')

    x2 = round(get_x(duty.DutyEndTimeUTC - ref_time) * x_fac, 2)
    y2, y_max = get_y(duty.EndAirport, y_max, y_fac)
    name_2 = f'{duty.DutyID}_2'
    add_node(file, name_2, x2, y2, '')

    add_arc(file, name_1, name_2, 180, 0, 'black')
    return y_max


if __name__ == '__main__':
    df_crew_base_stat = pd.read_csv(glob.glob('./input/CrewBaseStatistics_integrated.csv')[0])
    # df_crew_base_stat = df_crew_base_stat[
    #     (~df_crew_base_stat.PrevCrewDutyID.isnull()) | (~df_crew_base_stat.NextCrewDutyID.isnull())]
    df_crew_base_stat.PrevCrewDutyID = df_crew_base_stat.PrevCrewDutyID.astype('Int64')
    df_crew_base_stat.NextCrewDutyID = df_crew_base_stat.NextCrewDutyID.astype('Int64')

    df_crew_base_stat.DutyStartTimeUTC = pd.to_datetime(df_crew_base_stat.DutyStartTimeUTC)
    df_crew_base_stat.DutyEndTimeUTC = pd.to_datetime(df_crew_base_stat.DutyEndTimeUTC)

    df_crew_base_stat = df_crew_base_stat.sort_values(by=['DutyStartTimeUTC', 'DutyEndTimeUTC']).reset_index(drop=True)

    x_fac = 297 / get_x(df_crew_base_stat.DutyEndTimeUTC.max() - df_crew_base_stat.DutyStartTimeUTC.min())
    y_fac = 210 / len(df_crew_base_stat.StartAirport.unique())
    ref_time = df_crew_base_stat.DutyStartTimeUTC.min()

    station2y = {}
    y_max = 0
    visited_duties = set()
    counter = 0
    with open('crew_duty.tex', 'w') as file:
        file.write('\\begin{tikzpicture}\n')

        for indx, duty in df_crew_base_stat.iterrows():
            if duty.DutyID in visited_duties or \
                    not (isinstance(duty.PrevCrewDutyID, int) or isinstance(duty.NextCrewDutyID, int)):
                continue
            counter += 1
            y_max = add_duty(duty, y_max)

            # if isinstance(duty.PrevCrewDutyID, int):
            #     prev_duty = df_crew_base_stat[df_crew_base_stat.DutyID == duty.PrevCrewDutyID].iloc[0]
            #     add_duty(prev_duty, y_max)
            #     visited_duties.add(duty.PrevCrewDutyID)
            #
            #     add_arc(file, f'{duty.PrevCrewDutyID}_2', f'{duty.DutyID}_1', 180, 0, 'blue')
            #
            #
            # elif isinstance(duty.NextCrewDutyID, int):
            #     next_duty = df_crew_base_stat[df_crew_base_stat.DutyID == duty.NextCrewDutyID].iloc[0]
            #     add_duty(next_duty, y_max)
            #     visited_duties.add(duty.NextCrewDutyID)
            #
            #     add_arc(file, f'{duty.DutyID}_2', f'{duty.NextCrewDutyID}_1', 180, 0, 'blue')

        file.write('\\end{tikzpicture}\n')
