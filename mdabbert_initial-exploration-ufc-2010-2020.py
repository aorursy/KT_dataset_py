import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
df = pd.read_csv('/kaggle/input/ufc-fights-2010-2020-with-betting-odds/data.csv')
df.info(verbose=True)
df['date'] = pd.to_datetime(df['date'])
df = df.dropna()
df.info(verbose=True)
df[['R_fighter', 'B_fighter']].describe()
df[['date']].describe()
df[['R_odds', 'B_odds']].describe()
df[['location']].describe()
#There is a blank space problem that causes 2 countries to be counted twice....

df['country'] = df['country'].str.strip()

display(df[['country']].describe())

display(df['country'].unique())
print(df['Winner'].describe())

print()

print(df['Winner'].unique())
print(df['title_bout'].describe())
print(df['weight_class'].describe())

print()

print(df['weight_class'].unique())
print(df['gender'].describe())
year_labels = []

for z in range(2010, 2021):

    year_labels.append(z)

    

fight_counts = []

for z in (year_labels):

    fight_counts.append(len(df[df['date'].dt.year==z]))
plt.figure(figsize=(9,5))

plt.plot(year_labels, fight_counts)

plt.xlabel('Year', fontsize=16)

plt.ylabel('# of Fights', fontsize=16)

plt.title('Fights Per Year', fontweight='bold', fontsize=16)

plt.show()
female_fight_counts = []

for z in (year_labels):

    female_fight_counts.append(len(df[(df['date'].dt.year==z) & (df['gender']=='FEMALE')])) 

#print(female_fight_counts)



plt.figure(figsize=(9,5))

plt.plot(year_labels, female_fight_counts)

plt.xlabel('Year', fontsize=16)

plt.ylabel('# of Fights', fontsize=16)

plt.title('Female Fights Per Year', fontweight='bold', fontsize=16)

plt.show()
df['underdog'] = ''



red_underdog_mask = df['R_odds'] > df['B_odds']

#print(red_underdog_mask)

#print()



blue_underdog_mask = df['B_odds'] > df['R_odds']

#print(blue_underdog_mask)

#print()



even_mask = (df['B_odds'] == df['R_odds'])

#print(even_mask)

#print()



df['underdog'][red_underdog_mask] = 'Red'

df['underdog'][blue_underdog_mask] = 'Blue'

df['underdog'][even_mask] = 'Even'

df_no_even = df[df['underdog'] != 'Even']

df_no_even = df_no_even[df_no_even['Winner'] != 'Draw']

print(f"Number of fights including even fights and draws: {len(df)}")

print(f"Number of fights with even fights and draws removed: {len(df_no_even)}")
number_of_fights = len(df_no_even)

number_of_upsets = len(df_no_even[df_no_even['Winner'] == df_no_even['underdog']])

number_of_favorites = len(df_no_even[df_no_even['Winner'] != df_no_even['underdog']])

#print(number_of_upsets)

#print(number_of_fights)

#print(number_of_favorites)

upset_percent = (number_of_upsets / number_of_fights) * 100

favorite_percent = (number_of_favorites / number_of_fights) * 100

#print(upset_percent)

#print(favorite_percent)

labels = 'Favorites', 'Underdogs'

sizes = [favorite_percent, upset_percent]

fig1, ax1 = plt.subplots(figsize=(9,9))

ax1.pie(sizes, labels=labels, autopct='%1.1f%%', textprops={'fontsize': 14})

year_labels

year_fight_counts = []

year_upset_counts = []

year_upset_percent = []



for y in year_labels:

    temp_fights = df_no_even[df_no_even['date'].dt.year==y]

    temp_upsets = temp_fights[temp_fights['Winner'] == temp_fights['underdog']]

    year_fight_counts.append(len(temp_fights))

    year_upset_counts.append(len(temp_upsets))

    year_upset_percent.append(len(temp_upsets)/len(temp_fights))

    

#print(year_fight_counts)

#print()

#print(year_upset_counts)

#print()

#print(year_upset_percent)



year_upset_percent = [x*100 for x in year_upset_percent]



plt.figure(figsize=(9,5))

barlist = plt.bar(year_labels, year_upset_percent)

plt.xlabel("Year", fontsize=16)

plt.ylabel("Percent of Upset Winners", fontsize=16)

plt.xticks(year_labels, rotation=90)

plt.title('Upset Percentage By Year', fontweight='bold', fontsize=16)

barlist[10].set_color('black')

barlist[3].set_color('grey')
temp_df = pd.DataFrame({"Percent of Underdog Winners": year_upset_percent},

                      index=year_labels)



fig, ax = plt.subplots(figsize=(4,8))

sns.heatmap(temp_df, annot=True, fmt=".4g", cmap='binary', ax=ax)

plt.yticks(rotation=0)

plt.title("Upset Percentage by Year", fontsize=16, fontweight='bold')
#weight_class_list = df['weight_class'].unique()

#We are manually going to enter the weight class list so we can enter it in order of lightest to heaviest.

weight_class_list = ['Flyweight', 'Bantamweight', 'Featherweight', 'Lightweight', 'Welterweight', 

                     'Middleweight', 'Light Heavyweight', 'Heavyweight', "Women's Strawweight", 

                    "Women's Flyweight", "Women's Bantamweight", "Women's Featherweight", "Catch Weight"]

wc_fight_counts = []

wc_upset_counts = []

wc_upset_percent = []



for wc in weight_class_list:

    temp_fights = df_no_even[df_no_even['weight_class']==wc]

    temp_upsets = temp_fights[temp_fights['Winner'] == temp_fights['underdog']]

    wc_fight_counts.append(len(temp_fights))

    wc_upset_counts.append(len(temp_upsets))

    wc_upset_percent.append(len(temp_upsets)/len(temp_fights))



#print(weight_class_list)

#print()

#print(wc_fight_counts)

#print()

#print(wc_upset_counts)

#print()

wc_upset_percent = [x*100 for x in wc_upset_percent]    

#print(wc_upset_percent)

plt.figure(figsize=(9,5))

barlist = plt.bar(weight_class_list, wc_upset_percent)

plt.xlabel("Weight Class", fontsize=16)

plt.ylabel("Percent of Upset Winners", fontsize=16)

plt.xticks(weight_class_list, rotation=90)

plt.title('Upset Percentage By Weight Class', fontweight='bold', fontsize=16)

barlist[9].set_color('black')

barlist[11].set_color('grey')
temp_df = pd.DataFrame({"Percent of Underdog Winners": wc_upset_percent},

                      index=weight_class_list)



fig, ax = plt.subplots(figsize=(4,8))

sns.heatmap(temp_df, annot=True, fmt=".4g", cmap='binary', ax=ax)

plt.yticks(rotation=0)

plt.title("Upset Percentage by Weight Class", fontsize=16, fontweight='bold')
gender_list = df['gender'].unique()

gender_fight_counts = []

gender_upset_counts = []

gender_upset_percent = []



for g in gender_list:

    temp_fights = df_no_even[df_no_even['gender']==g]

    temp_upsets = temp_fights[temp_fights['Winner'] == temp_fights['underdog']]

    gender_fight_counts.append(len(temp_fights))

    gender_upset_counts.append(len(temp_upsets))

    gender_upset_percent.append(len(temp_upsets)/len(temp_fights))

    

plt.figure(figsize=(9,5))

barlist = plt.bar(gender_list, gender_upset_percent)

plt.xlabel("Gender", fontsize=16)

plt.ylabel("Percent of Upset Winners", fontsize=16)

plt.xticks(gender_list, rotation=90)

plt.title('Upset Percentage By Gender', fontweight='bold', fontsize=16)
title_list = df['title_bout'].unique()

title_fight_counts = []

title_upset_counts = []

title_upset_percent = []



for t in title_list:

    temp_fights = df_no_even[df_no_even['title_bout']==t]

    temp_upsets = temp_fights[temp_fights['Winner'] == temp_fights['underdog']]

    title_fight_counts.append(len(temp_fights))

    title_upset_counts.append(len(temp_upsets))

    title_upset_percent.append(len(temp_upsets)/len(temp_fights))

    

#print(title_list)

#print()

#print(title_fight_counts)

#print()

#print(title_upset_counts)

#print()

#title_upset_percent = [x*100 for x in title_upset_percent]    

#print(title_upset_percent)    



plt.figure(figsize=(9,5))

barlist = plt.bar(['Non-Title', 'Title'], title_upset_percent)

plt.xlabel("Bout Status", fontsize=16)

plt.ylabel("Percent of Upset Winners", fontsize=16)

plt.xticks(['Non-Title', 'Title'])

plt.title('Upset Percentage By Title Bout', fontweight='bold', fontsize=16)
df_title = df_no_even[df_no_even['title_bout']==True]

weight_class_list = ['Flyweight', 'Bantamweight', 'Featherweight', 'Lightweight', 'Welterweight', 

                     'Middleweight', 'Light Heavyweight', 'Heavyweight', "Women's Strawweight", 

                    "Women's Flyweight", "Women's Bantamweight", "Women's Featherweight"]



wc_fight_counts = []

wc_upset_counts = []

wc_upset_percent = []



for wc in weight_class_list:

    temp_fights = df_title[df_title['weight_class']==wc]

    temp_upsets = temp_fights[temp_fights['Winner'] == temp_fights['underdog']]

    wc_fight_counts.append(len(temp_fights))

    wc_upset_counts.append(len(temp_upsets))

    wc_upset_percent.append(len(temp_upsets)/len(temp_fights))



#print(weight_class_list)

#print()

#print(wc_fight_counts)

#print()

#print(wc_upset_counts)

#print()

wc_upset_percent = [x*100 for x in wc_upset_percent]    

#print(wc_upset_percent)
plt.figure(figsize=(9,5))

barlist = plt.bar(weight_class_list, wc_upset_percent)

plt.xlabel("Weight Class", fontsize=16)

plt.ylabel("Percent of Upset Winners", fontsize=16)

plt.xticks(weight_class_list, rotation=90)

plt.title('Title Fight Upset Percentage By Weight Class', fontweight='bold', fontsize=16)

barlist[7].set_color('black')
temp_df = pd.DataFrame({"Percent of Underdog Winners": wc_upset_percent, 

                        "Number of Fights": wc_fight_counts},

                      index=weight_class_list)



fig, ax = plt.subplots(figsize=(8,8))

sns.heatmap(temp_df, annot=True, fmt=".4g", cmap='binary', ax=ax)

plt.yticks(rotation=0, fontsize=12)

plt.title("Title Fight Upset Percentage by Weight Class", fontsize=16, fontweight='bold')

plt.xticks(fontsize=12)
red_fighter_list = df_no_even['R_fighter'].unique()

blue_fighter_list = df_no_even['B_fighter'].unique()

fighter_list = list(set(red_fighter_list) | set(blue_fighter_list))

upset_list = []



for f in fighter_list:

    temp_fights = df_no_even[(df_no_even['R_fighter']==f) | (df_no_even["B_fighter"]==f)]



    #Filter out fights where the fighter is not the winner.

    temp_fights = temp_fights[((temp_fights['R_fighter']==f) & (temp_fights['Winner']=='Red')) |

                             ((temp_fights['B_fighter']==f) & (temp_fights['Winner']=='Blue'))]

    

    

    #Filter out the fights where our hero is not the underdog.

    temp_fights = temp_fights[((temp_fights['R_fighter']==f) & (temp_fights['underdog']=='Red')) |

                             ((temp_fights['B_fighter']==f) & (temp_fights['underdog']=='Blue'))]

    

    

    upset_list.append(len(temp_fights)) 

    

    #print(temp_upset_count)

    #print(temp_fights)

    #print(f"{f}: {len(temp_fights)}")

    #temp_upsets = temp_fights[temp_fights['Winner'] == temp_fights['underdog']]

    #wc_fight_counts.append(len(temp_fights))

    #wc_upset_counts.append(len(temp_upsets))

    #wc_upset_percent.append(len(temp_upsets)/len(temp_fights))



#Zip the two lists into a dataframe

upset_tuples = list(zip(fighter_list, upset_list))

upset_df = pd.DataFrame(upset_tuples, columns=['fighter', 'upset_count'])

upset_df = upset_df.sort_values(by=['upset_count'], ascending=False)

display(upset_df.head(8))
#It is possible that there are 2 events that occur in the same day, but there would not be two events on the same day 

#in the same location.



#event_list = df_no_even['date'].unique()



event_df = df_no_even[['date', 'location']]

event_df = event_df.drop_duplicates()



event_array = event_df.values

upset_list = []

date_list = []

location_list = []

for e in event_array:

    temp_event = df_no_even[(df_no_even['date']==e[0]) & (df_no_even["location"]==e[1])]

    #Temp event now has all of the fights in the array

    underdog_df = temp_event[((temp_event['Winner'] == temp_event['underdog']))]

    #print(len(temp_fights))

    upset_list.append(len(underdog_df)) 

    date_list.append(e[0])

    location_list.append(e[1])

    

#print(len(upset_list))

#print(len(event_array))

upset_tuples = list(zip(location_list, date_list, upset_list))

upset_df = pd.DataFrame(upset_tuples, columns = ['location', 'date', 'upset_count'])

upset_df = upset_df.sort_values(by=['upset_count'], ascending=False)

display(upset_df.head(9))
country_list = df_no_even['country'].unique()

#print(country_list)

upset_list = []

upset_per_list = []

for c in country_list:

    temp_event = df_no_even[(df_no_even['country']==c)]

    #Temp event now has all of the fights in the array

    underdog_df = temp_event[((temp_event['Winner'] == temp_event['underdog']))]

    #print(len(temp_fights))

    underdog_count = len(underdog_df)

    fight_count = len(temp_event)

    upset_per_list.append((underdog_count) / (fight_count) * 100)     

    upset_list.append(underdog_count) 

upset_tuples = list(zip(country_list, upset_list, upset_per_list))

upset_df = pd.DataFrame(upset_tuples, columns=['country', 'upset_count', 'upset_per'])

upset_df = upset_df.sort_values(by=['upset_count'], ascending=False)

#print(upset_list)

#print(country_list)
plt.figure(figsize=(15,6))

barlist = plt.bar(upset_df['country'], upset_df['upset_count'])

plt.xlabel("Country", fontsize=16)

plt.ylabel("Number of Upsets", fontsize=16)

plt.xticks(upset_df['country'], rotation=90)

plt.title('Upset Count by Country', fontweight='bold', fontsize=16)

barlist[0].set_color('black')

barlist[24].set_color('grey')
temp_df = upset_df.set_index('country')



fig, ax = plt.subplots(figsize=(4,8))

sns.heatmap(temp_df, annot=True, fmt=".4g", cmap='binary', ax=ax)

plt.yticks(rotation=0)

plt.title("Upset Count By Country", fontsize=16, fontweight='bold')
plt.figure(figsize=(15,6))

barlist = plt.bar(upset_df['country'], upset_df['upset_per'])

plt.xlabel("Country", fontsize=16)

plt.ylabel("Upset Percentage", fontsize=16)

plt.xticks(upset_df['country'], rotation=90)

plt.title('Upset Percentage by Country', fontweight='bold', fontsize=16)

barlist[16].set_color('black')

barlist[23].set_color('grey')
temp_df = upset_df[['country', 'upset_per']]

temp_df = temp_df.set_index('country')



fig, ax = plt.subplots(figsize=(4,8))

sns.heatmap(temp_df, annot=True, fmt=".4g", cmap='binary', ax=ax)

plt.yticks(rotation=0)

plt.title("Upset Percentage By Country", fontsize=16, fontweight='bold')
location_list = df_no_even['location'].unique()

#print(len(location_list))

upset_list = []

upset_per_list = []

for l in location_list:

    temp_event = df_no_even[(df_no_even['location']==l)]

    #Temp event now has all of the fights in the array

    underdog_df = temp_event[((temp_event['Winner'] == temp_event['underdog']))]

    #print(len(temp_fights))

    underdog_count = len(underdog_df)

    fight_count = len(temp_event)

    upset_per_list.append((underdog_count) / (fight_count) * 100)     

    upset_list.append(underdog_count) 

upset_tuples = list(zip(location_list, upset_list, upset_per_list))

upset_df = pd.DataFrame(upset_tuples, columns=['location', 'upset_count', 'upset_per'])

upset_df = upset_df.sort_values(by=['upset_count'], ascending=False)

#print(upset_list)

#print(country_list)

#display(upset_df)
plt.figure(figsize=(10,30))

plt.grid(axis='x')

barlist = plt.barh(upset_df['location'], upset_df['upset_count'])

plt.xlabel("Number of Upsets", fontsize=16)

plt.ylabel("Location", fontsize=16)

plt.yticks(upset_df['location'])

plt.title('Upset Count by Location', fontweight='bold', fontsize=16)

barlist[0].set_color('black')

barlist[-1].set_color('grey')
plt.figure(figsize=(10,30))

plt.grid(axis='x')

barlist = plt.barh(upset_df['location'], upset_df['upset_per'])

plt.xlabel("Upset Percentage", fontsize=16)

plt.ylabel("Location", fontsize=16)

plt.yticks(upset_df['location'])

plt.title('Upset Count by Location', fontweight='bold', fontsize=16)

barlist[33].set_color('black')

barlist[-1].set_color('grey')
temp_df = upset_df[['location', 'upset_per']]

temp_df = temp_df.set_index('location')



fig, ax = plt.subplots(figsize=(8,30))

sns.heatmap(temp_df, annot=True, fmt=".4g", cmap='binary', ax=ax)

plt.yticks(rotation=0)

plt.title("Upset Percentage By Location", fontsize=16, fontweight='bold')
underdog_win_df = df_no_even[(df_no_even['Winner'] == df_no_even['underdog'])].copy()

underdog_win_df['winner_odds'] = underdog_win_df[['B_odds', 'R_odds']].values.max(1)

underdog_win_df = underdog_win_df.sort_values(by=['winner_odds'], ascending=False)

underdog_display = underdog_win_df[['R_fighter', 'B_fighter', 'weight_class', 'date', 'Winner', 'winner_odds']]



display(underdog_display.head(10))