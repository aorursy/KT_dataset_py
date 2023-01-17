import glob
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import math
base_time = pd.datetime(year=2018, month=12, day=10)

df_flight_stat = pd.read_csv(glob.glob('./data/FlightStatistics*.csv')[0],
                             usecols=['FlightID', 'DayNum', 'FlightNum', 'Orig', 'Dest', 'ScDepTimeUTC', 'ScArrTimeUTC',
                                      'SeqNum', 'RouteCode'])

df_flight_stat = df_flight_stat[df_flight_stat.DayNum == 1]  # filter for one day
df_flight_stat['ScDepTimeUTC'] = pd.to_datetime(df_flight_stat['ScDepTimeUTC'])
df_flight_stat['ScArrTimeUTC'] = pd.to_datetime(df_flight_stat['ScArrTimeUTC'])

df_crew_rule = pd.read_csv(glob.glob('./data/CrewDutyTime*.csv')[0], skiprows=2)
df_crew_rule = df_crew_rule[df_crew_rule.CrewType == 'Cockpit']  # cockpit crew for SWA
df_crew_rule.CheckInWindowStartTime = pd.to_timedelta(df_crew_rule.CheckInWindowStartTime + ':00')
df_crew_rule.CheckInWindowEndTime = pd.to_timedelta(df_crew_rule.CheckInWindowEndTime + ':00')
indx = df_crew_rule.CheckInWindowEndTime < df_crew_rule.CheckInWindowStartTime
df_crew_rule.CheckInWindowEndTime[indx] += pd.Timedelta(hours=24)
df_crew_rule.CrewDutyTime = pd.to_timedelta(df_crew_rule.CrewDutyTime + ':00')

df_crew_stat = pd.read_csv(glob.glob('./data/CrewBaseStatistics*.csv')[0])
df_crew_stat = df_crew_stat[df_crew_stat.DayNum == 1]  # filter for one day
df_crew_stat.DutyStartTimeLocal = pd.to_datetime(df_crew_stat.DutyStartTimeLocal)
df_crew_stat.DutyEndTimeLocal = pd.to_datetime(df_crew_stat.DutyEndTimeLocal)
df_crew_stat.DutyStartTimeUTC = pd.to_datetime(df_crew_stat.DutyStartTimeUTC)
df_crew_stat.DutyEndTimeUTC = pd.to_datetime(df_crew_stat.DutyEndTimeUTC)
for index, plan in df_crew_stat.iterrows():
    flight_seq = plan.FlightNum.split('>')

    flight_seq_from_stat = []
    for i, flight_name in enumerate(flight_seq):
        flight_num = int(flight_name.replace('WN', ''))
        flight_indx = (df_flight_stat.FlightNum == flight_num)
        if i == 0:
            flight_indx &= df_flight_stat.Orig == plan.StartAirport
        else:
            flight_indx &= df_flight_stat.Orig == flight_seq_from_stat[i - 1].Dest

        if flight_indx.sum() != 1:
            print('couldn\'t find a unique flight:', plan.FlightNum)
            continue

        flight_seq_from_stat.append(df_flight_stat[flight_indx].iloc[0])

        if len(flight_seq_from_stat) > 1 and flight_seq_from_stat[-1].SeqNum != flight_seq_from_stat[-2].SeqNum + 1:
            print('seq number issue:', plan.FlightNum)

    if flight_seq_from_stat[-1].Dest != plan.EndAirport:
        print('end airport mismatch:', plan.FlightNum)

    routeCode = set([f.RouteCode for f in flight_seq_from_stat])
    if len(routeCode) != 1:
        print('different routes:', plan.FlightNum)

    # Depending on if crew duty times are in UTC or any other time zone this can change
    duty_start_time = pd.Timedelta(hours=plan.DutyStartTimeUTC.time().hour + plan.DutyStartTimeUTC.time().minute / 60)
    rule_indx = (df_crew_rule.CheckInWindowStartTime <= duty_start_time) & \
                (df_crew_rule.CheckInWindowEndTime >= duty_start_time) & \
                (df_crew_rule.MinSectorCount <= plan.NumLegs) & \
                (df_crew_rule.MaxSectorCount >= plan.NumLegs)

    if rule_indx.sum() != 1:
        print('rule not found', plan.FlightNum)
        continue

    rule = df_crew_rule[rule_indx].iloc[0]

    time_diff = flight_seq_from_stat[-1].ScArrTimeUTC - flight_seq_from_stat[
        0].ScDepTimeUTC - rule.CrewDutyTime + pd.Timedelta(hours=0.5)
    if time_diff > pd.Timedelta(hours=0):
        print('duty max duration violated', plan.FlightNum)


