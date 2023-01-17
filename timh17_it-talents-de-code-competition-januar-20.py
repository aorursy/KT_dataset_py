# import required python packages

import pandas as pd

import numpy as np

import datetime

import re

from scipy.stats import norm

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns 
races = pd.read_csv("../input/ittalentsraces/races.csv", delimiter=";") # load data set 
print('Enthält die id Spalte null values? Antwort: ', 'Ja' if races.id.isnull().sum() > 0 else 'Nein')
print('Ist die id für jedes Rennen einzigartig? Antwort: ', 'Ja' if races.id.is_unique > 0 else 'Nein')
sum_auto = races.id.sum() # auto sum of the column 

sum = 0

for x in range(1, len(races["id"])+1):

    sum += x

print('Werden die IDs inkrementell erzeugt? Antwort: ', 'Ja' if sum_auto-sum == 0 else 'Nein')
print('Enthält die race_created Spalte null values? Antwort: ', 'Ja' if races.race_created.isnull().sum() > 0 else 'Nein')
races["race_created"].apply(lambda x: datetime.datetime.strptime(x, "%d.%m.%Y"))
races.race_driven.value_counts() # get the distribution of the present date values 
races.status.value_counts().plot.pie(subplots=True, figsize=(10,5), autopct='%1.1f%%', title="Distribution of race status column")

races.status.loc[races.race_driven == '0000-00-00 00:00:00'].value_counts() # get distribution when race_driven equals == 0000-00-00 00:00:00'
print('Enthält die track_id Spalte null values? Antwort: ', 'Ja' if races.track_id.isnull().sum() > 0 else 'Nein')
races.track_id.value_counts()
print('Enthält die challenger Spalte null values? Antwort: ', 'Ja' if races.challenger.isnull().sum() > 0 else 'Nein')

print('Enthält die opponent Spalte null values? Antwort: ', 'Ja' if races.opponent.isnull().sum() > 0 else 'Nein')

print('Enthält die winner Spalte null values? Antwort: ', 'Ja' if races.winner.isnull().sum() > 0 else 'Nein')
print('Enthält die challenger Spalte einen Fahrer mit id==0? Antwort: ', 'Nein' if len(races.challenger.loc[races.challenger == 0]) == 0 else 'Ja'  )

print('Enthält die opponent Spalte einen Fahrer mit id==0? Antwort: ', 'Nein' if len(races.opponent.loc[races.opponent == 0]) == 0 else 'Ja')

print('Enthält die winner Spalte neinen Fahrer mit id==0? Antwort: ', 'Nein' if len(races.winner.loc[races.winner == 0]) == 0 else 'Ja')
print('Anzahl der Einträge mit id==0 in der opponent Spalte: ', len(races.opponent.loc[races.opponent==0]))

print('Anzahl der Einträge mit id==0 in der winner Spalte: ', len(races.winner.loc[races.winner==0]))
races.status.loc[(races.winner == 0)].value_counts() 
races.status.loc[races.opponent == 0].value_counts()
print('Anzahl der Rennen mit Rennstatus `abesagt` und Gegner id==0: ', len(races.status.loc[((races.status=="retired") & (races.opponent == 0))]))

print('Anzahl der Rennen mit Rennstatus `abesagt`: ', len(races.status.loc[(races.status=="retired")]))
print("Anzahl der abgesagten Rennen mit Gegner id==0: ", len(races.status.loc[(races.status == "retired") & (races.opponent == 0)]), " => Entpricht im Schaubild 2.") # Challenger retired race before an inviation to a possible opponent was sent (node 2)

print("Anzahl der abgesagten Rennen mit Gegner id!=0: ", len(races.status.loc[(races.status == "retired") & (races.opponent != 0)]), " =>Entpricht im Schaubild 3.") # Challenger retired race after an inviation to a possible opponent was sent  (node 3)

print("Anzahl der wartenden Rennen mit Gegner id==0: ", len(races.status.loc[(races.status == "waiting") & (races.opponent == 0)]), " =>Entspricht im Schaubild 1.") # Challenger created new race, equals first state in flow chart. Seems valid because its the second last entry in the csv

print("Anzahl der wartenden Rennen mit Gegner id!=0: ", len(races.status.loc[(races.status == "waiting") & (races.opponent != 0)]), " =>Entspricht im Schaubild 3. oder 6.") # Challenger created new race, invitation was either accepted or is still pending. (equals 3 or 6)
races.winner.loc[(races.status != "finished") & (races.winner == 0)] = -1 # replace 0 with -1 if race was not finished 

races.opponent.loc[races.opponent==0] = -1
# get all races where the status is finsihed and the winner column is consistent 

races[["opponent", "challenger", "winner", "status"]].loc[(races["status"] == "finished") & ((races["opponent"] == races["winner"]) | (races["challenger"] == races["winner"]))]
# get all races where status is finsihed

races[["opponent", "challenger", "winner", "status"]].loc[races["status"] == "finished"]
print('Enthält die money Spalte null values? Antwort: ', 'Ja' if races.money.isnull().sum() > 0 else 'Nein')

print('Enthält die money Spalte `0`? Antwort: ', 'Ja' if len(races.money.loc[races.money==0]) > 0 else 'Nein')

print('Enthält die money Spalte negative Werte? Antwort: ', 'Ja' if len(races.money.loc[races.money<0]) > 0 else 'Nein')

print('Durchschnittlicher/s Einsatz/Preisgeld: ', races.money.mean())

print('Minimaler/s Einsatz/Preisgeld: ', races.money.min())

print('Maximaler/s Einsatz/Preisgeld: ', races.money.max())
def clean_fuel_consumption_column(txt):

    try:

        return float(txt)

    except ValueError:

        return np.nan

    

# replace all non-float values with non and save them in a new column 

races["fuel_with_nan"] = races.fuel_consumption.apply(clean_fuel_consumption_column)
def clean_fuel_with_regex(txt):

    try:

        return float(txt)

    except ValueError:

        txt =txt.replace(" ", "")

        if re.search("^Jan", txt):

            txt = txt.replace("Jan", "1.")

        if re.search("^Feb", txt):

            txt = txt.replace("Feb", "2.")

        if re.search("^Mrz", txt):

            txt = txt.replace("Mrz", "3.")

        if re.search("^Apr", txt):

            txt = txt.replace("Apr", "4.")

        if re.search("^Mai", txt):

            txt = txt.replace("Mai", "5.")

        if re.search("^Jun", txt):

            txt = txt.replace("Jun", "6.")

        if re.search("^Jul", txt):

            txt = txt.replace("Jul", "7.")

        if re.search("^Aug", txt):

            txt = txt.replace("Aug", "8.")

        if re.search("^Sep", txt):

            txt = txt.replace("Sep", "9.")

        if re.search("^Okt", txt):

            txt = txt.replace("Okt", "10.")

        if re.search("^Nov", txt):

            txt = txt.replace("Nov", "11.")

        if re.search("^Dez", txt):

            txt = txt.replace("Dez", "12.")

        

        if re.search("Jan$", txt):

            txt = txt.replace("Jan", "1")

        if re.search("Feb$", txt):

            txt = txt.replace("Feb", "2")

        if re.search("Mrz$", txt):

            txt = txt.replace("Mrz", "3")

        if re.search("Apr$", txt):

            txt = txt.replace("Apr", "4")

        if re.search("Mai$", txt):

            txt = txt.replace("Mai", "5")

        if re.search("Jun$", txt):

            txt = txt.replace("Jun", "6")

        if re.search("Jul$", txt):

            txt = txt.replace("Jul", "7")

        if re.search("Aug$", txt):

            txt = txt.replace("Aug", "8")

        if re.search("Sep$", txt):

            txt = txt.replace("Sep", "9")

        if re.search("Okt$", txt):

            txt = txt.replace("Okt", "10")

        if re.search("Nov$", txt):

            txt = txt.replace("Nov", "11")

        if re.search("Dez$", txt):

            txt = txt.replace("Dez", "12")

        try:

            return float(txt)

        except ValueError:

            print ('VE', txt)

races["fuel_with_replaced_values"] = races.fuel_consumption.apply(clean_fuel_with_regex)
df = pd.DataFrame({

    '1. Möglichkeit': races["fuel_with_nan"],

    '2. Möglichkeit': races["fuel_with_replaced_values"],

})

ax = df.plot.kde()

print('1. Möglichkeit durchschnittlicher Spritverbrauch', races["fuel_with_nan"].mean())

print('2. Möglichkeit durchschnittlicher Spritverbrauch', races["fuel_with_replaced_values"].mean())

print('1. Möglichkeit Anzahl von validen Verbräuchen', len(races["fuel_with_nan"].loc[pd.notna(races["fuel_with_nan"])]))

print('2. Möglichkeit Anzahl von validen Verbräuchen', len(races["fuel_with_replaced_values"]))
races.fuel_consumption = races.fuel_with_replaced_values # replace old values

races = races.drop(["fuel_with_nan", "fuel_with_replaced_values"], axis=1) # remove old columns 
ax = sns.barplot(data=races, x="status", y="fuel_consumption") ## Spritverbrauch 
ax = sns.countplot(data=races, x="status") # Anzahl Rennen 
# Extract weather condition and possibility. Create new column and store possibility

races[["sunny", "rainy", "thundery", "snowy"]] =races.forecast.str.extractall('"(?P<column>.*?)";i:(?P<value>\d+)').reset_index(level=0).pivot('level_0','column','value')

# delete old forecast column

races = races.drop(["forecast"], axis=1) 
# For each row, check if the sum of the weather forecast probabilties is 100

races[["sunny", "rainy", "thundery", "snowy"]].astype(int).sum(axis=1).value_counts()
print('Enthält die weather Spalte NaN Werte? Antwort: ', 'Ja' if races.weather.isnull().sum()>0 else 'Nein')
races.weather.value_counts( dropna=False).plot.pie(subplots=True, autopct='%1.1f%%')
races["forecasted_weather"] = races[["sunny", "rainy", "thundery", "snowy"]].astype(int).idxmax(axis=1) # return column name with highest value 

false_forecasts = races[["weather", "forecasted_weather"]].loc[(pd.notnull(races.weather) & (races.weather != races.forecasted_weather))] # return all rows where forecast != actual weather 

correct_forecasts = races[["weather", "forecasted_weather"]].loc[(pd.notnull(races.weather) & (races.weather == races.forecasted_weather))] # return all rows where forecast == actual weather 

print('Anzahl korrekter Wettervorhersagen', len(correct_forecasts))

print('Anzahl falscher Wettervorhersagen', len(false_forecasts))

print('Anteil korrekter Wettervorhersagen', (len(correct_forecasts) / (len(correct_forecasts)+len(false_forecasts))))
import random

def replace_int_with_weather(weather, state):

    if pd.isna(weather):

        # no real weather given, choose random weather 

        if state == 0:

            return 'sunny'

        if state == 1:

            return 'rainy'

        if state == 2:

            return 'thundery'

        if state == 3:

            return 'snowy'

    else:

        return weather

races['weather'] = races['weather'].apply(lambda x: replace_int_with_weather(x, random.randint(0,3)))
races.weather.value_counts( dropna=False).plot.pie(subplots=True, autopct='%1.1f%%')
# On which day of the week are the most races driven? 0: Monday, 1: Tuesday .... 6: Sunday 

races["race_day"] = races.race_driven.loc[races.status == "finished"].apply(lambda x: datetime.datetime.strptime(x, "%d.%m.%Y %H:%M").weekday())
# for convenience we will also save a string of the raceday 

def replace_int(day):

    if day == 0:

        return "Monday"

    if day == 1:

        return "Tuesday"

    if day == 2: 

        return "Wednessday"

    if day == 3:

        return "Thursday"

    if day == 4: 

        return "Friday"

    if day == 5:

        return "Saturday"

    if day == 6:

        return "Sunday"

races["race_day_st"] = races["race_day"].apply(replace_int)
# Is there a offset between race created and race driven?

races.race_driven.loc[races.status == "finished"].apply(lambda x: int(datetime.datetime.strptime(x, "%d.%m.%Y %H:%M").strftime("%s")))

def f(created, driven):

    created_ts = int(datetime.datetime.strptime(created, "%d.%m.%Y").strftime("%s"))

    driven_ts = int(datetime.datetime.strptime(driven, "%d.%m.%Y %H:%M").strftime("%s"))

    return int((driven_ts-created_ts)/86400)

races["start_offset"] = races.loc[races.status == "finished"].apply(lambda x: f(x.race_created, x.race_driven), axis=1)
races["start_offset"]
races["start_offset"] = races["start_offset"].loc[pd.notna(races.start_offset)].apply(lambda x: 0 if x < 1 else int(x)).value_counts()
print('Insgesamt wurden ', len(races), ' rennen aufgezeichnet.')
ax = races.status.value_counts().plot.pie(autopct='%1.1f%%')

ax.set_title('Rennstatus Verteilung')

print('Erfolgreich beendete Rennen:', len(races.loc[races.status=="finished"]))

print('Abgesagte Rennen:', len(races.loc[races.status=="retired"]))

print('Abgelehnte Rennen:', len(races.loc[races.status=="declined"]))

print('Wartende Rennen:', len(races.loc[races.status=="waiting"]))
distinct_drivers = pd.concat([races.challenger, races.opponent]).loc[races.opponent != -1].nunique()

print('Insgesamt haben', distinct_drivers, 'unterschiedliche Fahrer an den Rennen teilgenommen.')
print ('Insgesamt wurden', races.money.sum(), 'erspielt')
p = races.winner.loc[races.winner!=-1].value_counts().nlargest(10).plot.bar()

p.set_title("Top 10 der Fahrer mit den meisten Rennsiegen")

p.set_xlabel('Fahrer ID')

p.set_ylabel('Rennsiege');
#First we get the the amount of finished races for the challenger and opponent column

a = races.challenger.loc[races.winner!=-1].value_counts()

b = races.opponent.loc[races.winner!=-1].value_counts()

# now we combine those columns 

finished_races = a.add(b, fill_value=0)

# get wins for each driver 

wins = races.winner.loc[races.winner!=-1].value_counts()

# calculate ratio 

ax1 = wins.divide(finished_races).loc[finished_races > 50].nlargest(10).to_frame().plot.bar()

ax1.get_legend().remove()

ax1.set_ylim([0.8,1.0])

ax1.set_ylabel('Siegquote')

ax1.set_xlabel('Fahrer ID')

ax1.set_title('Top 10 Fahrer mit höchsten Siegquote');
#Q3: First we sum up the fuel consumption for each driver in the challenger and opponent column

challenger_fuel = races.loc[races.winner!=-1].groupby(['challenger'])['fuel_consumption'].sum()

opponent_fuel = races.loc[races.winner!=-1].groupby(['opponent'])['fuel_consumption'].sum()

## now we can combine the challenger and opponent fuel 

combined_fuel = challenger_fuel.add(opponent_fuel)

## create dataframe with finished races 

finished_races = pd.DataFrame(finished_races, columns=['finished_races'])

## concat combined fuel with finished races and rename the column 

df = pd.concat([combined_fuel, finished_races], axis=1).rename(columns={"fuel_consumption": "combined_fuel"})

## divide fuel_consumption by 2, because we have a summed up value of two drivers

df['combined_fuel'] = df.combined_fuel.divide(2) 

## get the 10 largest fuel consumes with at least 100 races

pp = df.loc[df.finished_races > 50].nlargest(10, 'combined_fuel').plot.bar()

pp.set_ylabel("Spritverbrauch in l / Beendete Rennen")

pp.set_xlabel("Fahrer ID");
axx = (df.loc[df.finished_races > 50].apply(lambda x: x.combined_fuel / x.finished_races, axis=1)).nlargest(10).to_frame().plot.bar()

axx.get_legend().remove()

axx.set_ylim([20,40])

axx.set_ylabel('Durchschnittlicher Spritverbrauch in l')

axx.set_xlabel('Fahrer ID')

axx.set_title('Top 10 Fahrer mit höchsten durchschnitlichen Verbauch');
axx = (df.loc[df.finished_races > 50].apply(lambda x: x.combined_fuel / x.finished_races, axis=1)).nsmallest(10).to_frame().plot.bar()

axx.get_legend().remove()

#axx.set_ylim([20,40])

axx.set_ylabel('Durchschnittlicher Spritverbrauch in l')

axx.set_xlabel('Fahrer ID')

axx.set_title('Top 10 Fahrer mit dem niedrigsten durchschnitlichen Verbauch');
finished_races.nlargest(10, 'finished_races').plot.bar();
top10amount = finished_races.nlargest(10, 'finished_races').sum()

allamount = finished_races.sum()/2

ratio = top10amount/allamount

ratio

print('Anteil der Top 10 Fahrer im Vergleich zu allen beendeten Rennen: ', ratio[0])
# challengers with most retired races

retired_races = races.challenger.loc[races.status=="retired"]

ax = retired_races.value_counts().nlargest(10).to_frame().plot.bar()

ax.get_legend().remove()

ax.set_ylabel('Abegsagte Rennen')

ax.set_xlabel('Fahrer ID')

ax.set_title('Top 10 Fahrer mit den meisten abgesagten Rennen');
# opponents with most declined races

declined_races = races.opponent.loc[races.status=="declined"]

ax = declined_races.value_counts().nlargest(10).to_frame().plot.bar()

ax.get_legend().remove()

ax.set_ylabel('Abgelehnte Rennen')

ax.set_xlabel('Fahrer ID')

ax.set_title('Top 10 Fahrer mit den meisten abgelehnten Rennen');
data_array = []

for val, cnt in declined_races.value_counts().iteritems():

    if val in finished_races.index:

        fr = finished_races.loc[val].values[0]

        data_array.append([val, cnt, fr, (cnt/fr)])

        

pd.DataFrame(data_array, columns=['driver id', 'declined races', 'finished_races', 'ratio']).nlargest(10, 'ratio')
data_array = []

for val, cnt in retired_races.value_counts().iteritems():

    if val in finished_races.index:

        fr = finished_races.loc[val].values[0]

        data_array.append([val, cnt, fr, (cnt/fr)])

        

pd.DataFrame(data_array, columns=['driver id', 'retired races', 'finished_races', 'ratio']).nlargest(10, 'ratio')
#Q8: highest carrer earnings 

hce = races[["winner", "money"]].loc[races.winner != -1].groupby(["winner"])["money"].sum().nlargest(10).to_frame().plot.bar()

hce.get_legend().remove()

hce.set_ylabel('Einnahmen in 10 Millionen')

hce.set_xlabel('Fahrer ID')

hce.set_title('Top 10 der Fahrer mit den höchsten Gesamteinnahmen');
earnings = races[["winner", "money"]].loc[races.winner != -1].groupby(["winner"], as_index=False)["money"].sum()

earnings = earnings.set_index("winner")

dff = pd.concat([earnings, finished_races], axis=1)

dff['average_income'] = dff["money"]/dff["finished_races"] 

dff.nlargest(10, "average_income") ## top 10  average income 

#Q10

ttt = dff.nlargest(10, "average_income") ## top 10  average income with at least 10 races

ttt

ax = ttt[["average_income"]].plot.bar()

ax.get_legend().remove()

ax.set_ylabel('Durchschnittliche Einnahmen')

ax.set_xlabel('Fahrer ID')

ax.set_title('Höchste Durchschnittseinnahmen');
earnings = races[["winner", "money"]].loc[races.winner != -1].groupby(["winner"], as_index=False)["money"].sum()

earnings = earnings.set_index("winner")

dff = pd.concat([earnings, finished_races], axis=1)

dff['average_income'] = dff["money"]/dff["finished_races"] 

dff.nlargest(10, "average_income") ## top 10  average income 

#Q10

ttt = dff.loc[dff.finished_races > 10].nlargest(10, "average_income") ## top 10  average income with at least 10 races

ttt
ax1 = ttt[["average_income"]].plot.bar()

ax1.get_legend().remove()

ax1.set_ylabel('Durchschnittliche Einnahmen')

ax1.set_xlabel('Fahrer ID')

ax1.set_title('Höchste Durchschnittseinnahmen mindestens 10 beendete Rennen');
#Q10: highest career loss 

def get_id_of_loser(challenger, opponent, winner):

    if challenger == winner:

        return opponent 

    else:

        return challenger

races['loser'] = races.loc[(races.status == "finished")].apply(lambda x: get_id_of_loser(x.challenger, x.opponent, x.winner), axis=1)

losses = races[["loser", "money"]].loc[(races.status == "finished")].groupby(["loser"], as_index=True)["money"].sum().to_frame()

ax2 = losses.nlargest(10, 'money').plot.bar()

ax2.get_legend().remove()

ax2.set_ylabel("Verluste in Millionen")

ax2.set_xlabel("Fahrer ID")

ax2.set_title("Top 10 der Fahrer mit den höchsten Verlusten");
balance = earnings.subtract(losses, fill_value=0)

tp = balance.nlargest(10, 'money').plot.bar()

tp.get_legend().remove()

tp.set_ylabel("Reingewinn in Millionen")

tp.set_xlabel("Fahrer ID")

tp.set_title("Top 10 der Fahrer mit den höchsten Reingewinn");
tp2 = balance.nsmallest(10, 'money').plot.bar()

tp2.get_legend().remove()

tp2.set_ylabel("Reinverlust in Millionen")

tp2.set_xlabel("Fahrer ID")

tp2.set_title("Top 10 der Fahrer mit den höchsten Reinverlust");
t1 = races[["challenger", "opponent"]].loc[races.status=="finished"].groupby(races[["challenger", "opponent"]].columns.tolist(),as_index=False).size()

t2 = races[["challenger", "opponent"]].loc[races.status=="finished"].groupby(races[["opponent", "challenger"]].columns.tolist(),as_index=False).size()

dt1 = t1.to_frame(name="count")

dt2 = t2.to_frame(name="count")

dt2.index.names = ["challenger", "opponent"]

test = dt1.add(dt2, fill_value=0)

t = test.nlargest(20, 'count').reset_index().iloc[::2,:]

t = t.rename(columns={"challenger": "Fahrer 1", "opponent": "Fahrer 2", "count": "Anzahl Duellbegegnungen"}).reset_index(drop=True)

t
driver1_array = []

driver2_array = []

for index, row in t.iterrows():

    driver1 = row[0]

    driver2 = row[1]

    res = races.winner.loc[(races.status=="finished") & (((races.challenger == driver1) & (races.opponent == driver2)) | ((races.challenger == driver2) & (races.opponent == driver1)))].value_counts()

    if res.index[0] == driver1:

        driver1_array.append(res.values[0])

        driver2_array.append(res.values[1])

    else:

        driver1_array.append(res.values[1])

        driver2_array.append(res.values[0])

t['Siege Fahrer 1'] = driver1_array

t['Siege Fahrer 2'] = driver2_array

t
weather_frame = races[["winner", "weather"]].loc[((races.winner != -1 ) & (races.weather.notnull()) & (races.weather == "rainy"))].groupby(["winner", "weather"]).size().to_frame()

weather_frame.columns =  ["count"]

tx = weather_frame.nlargest(10, "count").reset_index()[["count"]].plot.bar()

x = [t for t in range(0,10)]

labels = [weather_frame.nlargest(10, "count").reset_index()[["winner"]].values[t][0] for t in range(0,10)]

#print ('labels', weather_frame.nlargest(10, "count").reset_index()[["winner"]].values)

tx.set_xticks(x)

tx.set_xticklabels(labels)

tx.get_legend().remove()

tx.set_ylabel("Regen - Rennsiege")

tx.set_title("Welche Fahrer gewinnen bei Regen am häufigsten? ");
weather_frame = races[["winner", "weather"]].loc[((races.winner != -1 ) & (races.weather.notnull()) & (races.weather == "sunny"))].groupby(["winner", "weather"]).size().to_frame()

weather_frame.columns =  ["count"]

x = [t for t in range(0,10)]

labels = [weather_frame.nlargest(10, "count").reset_index()[["winner"]].values[t][0] for t in range(0,10)]

txt = weather_frame.nlargest(10, "count").reset_index()[["count"]].plot.bar()

txt.set_xticks(x)

txt.set_xticklabels(labels)

txt.get_legend().remove()

txt.set_ylabel("Sonnenschein - Rennsiege")

txt.set_title("Welche Fahrer gewinnen bei Sonnenschein am häufigsten? ");
weather_frame = races[["winner", "weather"]].loc[((races.winner != -1 ) & (races.weather.notnull()) & (races.weather == "thundery"))].groupby(["winner", "weather"]).size().to_frame()

weather_frame.columns =  ["count"]

x = [t for t in range(0,10)]

labels = [weather_frame.nlargest(10, "count").reset_index()[["winner"]].values[t][0] for t in range(0,10)]

txt = weather_frame.nlargest(10, "count").reset_index()[["count"]].plot.bar()

txt.set_xticks(x)

txt.set_xticklabels(labels)

txt.get_legend().remove()

txt.set_ylabel("Gewitter - Rennsiege")

txt.set_title("Welche Fahrer gewinnen bei Gewitter am häufigsten? ");
weather_frame = races[["winner", "weather"]].loc[((races.winner != -1 ) & (races.weather.notnull()) & (races.weather == "snowy"))].groupby(["winner", "weather"]).size().to_frame()

weather_frame.columns =  ["count"]

x = [t for t in range(0,10)]

labels = [weather_frame.nlargest(10, "count").reset_index()[["winner"]].values[t][0] for t in range(0,10)]

txt = weather_frame.nlargest(10, "count").reset_index()[["count"]].plot.bar()

txt.set_xticks(x)

txt.set_xticklabels(labels)

txt.get_legend().remove()

txt.set_ylabel("Schneefall - Rennsiege")

txt.set_title("Welche Fahrer gewinnen bei Schneefall am häufigsten? ");
print('Durchschnittsverbrauch bei Sonnenschein:', races.fuel_consumption.loc[(races.status == "finished")& (races.weather == "sunny")].mean())

print('Durchschnittsverbrauch bei Regen:', races.fuel_consumption.loc[(races.status == "finished")& (races.weather == "rainy")].mean())

print('Durchschnittsverbrauch bei Gewitter:', races.fuel_consumption.loc[(races.status == "finished")& (races.weather == "thundery")].mean())

print('Durchschnittsverbrauch bei Schnee:', races.fuel_consumption.loc[(races.status == "finished")& (races.weather == "snowy")].mean())
weather_frame = races[["track_id", "weather"]].groupby(["track_id", "weather"]).size().to_frame()

weather_frame.columns =  ["count"]

fig, axes = plt.subplots(nrows=2, ncols=6, figsize=(20,10))

for y in range(0,2):

    for x in range(0,6):

        index = (x+3)+(y*6)

        weather_frame.loc[(index)].plot.pie(subplots=True, autopct='%1.1f%%', ax=axes[y,x])

        axes[y,x].set_title("Rennstrecke "+str(index))

        axes[y,x].get_legend().remove()
tid = races.track_id.loc[races.status=="finished"].value_counts().to_frame().plot.bar()

tid.get_legend().remove()

tid.set_ylabel("Anzahl von Rennen")

tid.set_xlabel("Rennstrecken ID")

tid.set_title("Rennstrecken sortiert nach meistbefahren");
indices = [x for x in range(3,15)]

track_by_retired_races = []

track_by_declined_races = []

for id in range(3,15):

    retired = races.track_id.loc[(races.status=="retired") & (races.track_id == id)].value_counts().values[0]

    track_by_retired_races.append(retired)

    declined = races.track_id.loc[(races.status=="declined") & (races.track_id == id)].value_counts().values[0]

    track_by_declined_races.append(declined)

xt0 =  pd.DataFrame({'retired': track_by_retired_races, 'declined': track_by_declined_races }, index=indices).plot.bar(figsize=(10,5))

xt0.set_title("Abbrüche und Ablehnungen je Rennstrecke")

xt0.set_ylabel("Anzahl Rennen")

xt0.set_xlabel("Strecken ID");
top10drivers = races.winner.loc[races.winner!=-1].value_counts().nlargest(10)

driver_ids = top10drivers.index

driver_wins = top10drivers.values



counter = 0

fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(20,10))        

for y in range(0,2):

    for x in range(0,5):

        id = driver_ids[counter]

        tracks = races.track_id.loc[((races.winner!=-1) & ((races.challenger == id) | (races.opponent == id)))].value_counts().plot.bar(ax=axes[y,x])

        axes[y,x].set_title("FahrerID : "+str(id))

        axes[y,x].set_xlabel("Rennstrecken ID")

        axes[y,x].set_ylabel("Anzahl der Rennen")

        #axes[y,x].get_legend().remove()

        counter += 1

    
# calculate mean fuel consumption for each track 

tracks_mean_fuel = races[["fuel_consumption","track_id"]].loc[(races.status=="finished")].groupby(['track_id']).mean()

# calculate mean fuel consumption for all tracks 

overall_mean = races.fuel_consumption.loc[races.status=="finished"].mean()



indices = tracks_mean_fuel.index

index = [str(x) for x in indices]

mean = tracks_mean_fuel.values

mean = [x[0] for x in mean]

std = races[["fuel_consumption","track_id"]].loc[(races.status=="finished")].groupby(['track_id']).std().values

std = [x[0] for x in std]

df = pd.DataFrame({'mean': mean, 'std': std }, index=index)

ax = df.plot.bar(figsize=(10,5))

ax.set_ylabel("Spritverbrauch in l")

ax.set_xlabel("Rennstrecken ID")

ax.set_title('Durchschnittlicher Spritverbrauch je Strecke')

plt.axhline(y=overall_mean);
races.race_day_st.value_counts().plot.pie(autopct='%1.1f%%');
races.race_day_st.value_counts().plot.bar();
top10_drivers = races.winner.loc[races.winner!=-1].value_counts().nlargest(10).to_frame().index

fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(20,10))

for y in range(0,2):

    for x in range(0,5):

        index = x+(y*5)

        id = top10_drivers[index]

        races.race_day_st.loc[((races.status=="finished") & ((races.challenger == id) | (races.opponent == id)))].value_counts().plot.bar(subplots=True, ax=axes[y,x])

        axes[y,x].set_title("Renntag-Verteilung - Fahrer ID:"+str(id))

        axes[y,x].set_ylabel("Anzahl Rennen")

plt.subplots_adjust(top = 0.99, bottom=0.01, hspace=0.5, wspace=0.4)
pdtmp = races[["opponent", "start_offset"]].nlargest(10, 'start_offset').rename(columns={"start_offset": "Verzögerung in Tagen"})

pdtmp["Verzögerung in Jahren"] = pdtmp["Verzögerung in Tagen"].divide(365)

pdtmp
print('Die durchschnittliche Verzögerung beträgt', round(races.start_offset.mean(),2) , 'Tage')