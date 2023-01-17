import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
file_path = "../input/bus-breakdown-and-delays.csv"

df = pd.read_csv(file_path)
df.head()
N_OF_DAYS = 20

today = datetime.date.today()
first_day = today - datetime.timedelta(N_OF_DAYS)

df['datetime'] = pd.to_datetime(df['Occurred_On'])
recent_data = df[(df.datetime >= pd.Timestamp(first_day)) & (df.datetime <= pd.Timestamp(today))]
def compute_lost_minutes(row) -> int:
    try:
        n_students = int(row['Number_Of_Students_On_The_Bus'])
    except (ValueError, TypeError):
        print('n_students', row['Number_Of_Students_On_The_Bus'])
        n_students = 0
    try :
        if isinstance(row['How_Long_Delayed'], str):
            n_delay = int(''.join([c for c in row['How_Long_Delayed'] if c in '0123456789']))
        else :
            n_delay = row['How_Long_Delayed']
    except (ValueError, TypeError):
        print('n_delay', row['How_Long_Delayed'])
        n_delay = 0
    return n_students * n_delay

recent_data['lost_minutes'] = recent_data.apply(compute_lost_minutes, axis = 1)

minutes_per_day = []
for n_day in range(1, N_OF_DAYS):
    sub_df = recent_data[
                (recent_data.datetime >= pd.Timestamp(today - datetime.timedelta(n_day))) &
                (recent_data.datetime <= pd.Timestamp(today - datetime.timedelta(n_day - 1)))
            ]
    minutes_per_day.append([str(today - datetime.timedelta(n_day)), sub_df['lost_minutes'].sum()])

minutes_per_day = list(reversed([m for m in minutes_per_day if m[1]]))
    
plt.bar([m[0] for m in minutes_per_day], [m[1] for m in minutes_per_day])
plt.xticks([m[0] for m in minutes_per_day], rotation='vertical')
plt.show()
borough = recent_data[recent_data.Breakdown_or_Running_Late == 'Breakdown']['Boro'].value_counts()

borough.plot(kind = 'bar')
