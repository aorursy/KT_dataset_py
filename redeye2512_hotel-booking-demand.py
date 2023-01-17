# load libraries
import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
import matplotlib
import datetime
%matplotlib inline 
import os 
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

font = {
    'family' : 'normal',
        'weight' : 'normal',
    'size'   : 12
}

matplotlib.rc('font', **font)
from IPython.core.display import HTML
HTML("""
<style>
.output_png {
    display: table-cell;
    text-align: center;
    vertical-align: middle;
}
</style>
""")
df = pd.read_csv('/kaggle/input/hotel-booking-demand/hotel_bookings.csv')
print('Size:', len(df))
df.head()
df.isnull().sum()
temp_df = df[['country', 'is_canceled']]
number_of_guests_from_each_country_df = temp_df['country'].value_counts().reset_index().rename(
    columns={'country':'number_of_guests','index':'country'}
)

unknown_countries = []
def convert_small_country_name(name, number_of_guests):
    if number_of_guests <=3000:
        unknown_countries.append(name)
        return 'Unk'
    return name

number_of_guests_from_each_country_df['new_country_name'] = number_of_guests_from_each_country_df.apply(lambda x: convert_small_country_name(x['country'], x['number_of_guests']),
                                            axis=1)
number_of_guests_from_each_country_df = number_of_guests_from_each_country_df.groupby(['new_country_name']).agg({
    'number_of_guests':'sum'
}).reset_index().sort_values(['number_of_guests'])


fig = plt.figure(figsize=(6,6))
plt.pie(number_of_guests_from_each_country_df['number_of_guests'], 
        labels=number_of_guests_from_each_country_df['new_country_name'], autopct='%1.1f%%')
plt.title("Ratio of country where guest come from")
# plt.legend(fontsize=10)
plt.show()
unknown_countries = list(set(unknown_countries))
temp_df = df['country'].value_counts().reset_index().rename(
    columns={'country':'number_of_guests','index':'country'}
)
temp_canceled_df = df[df['is_canceled']==1]['country'].value_counts().reset_index().rename(
    columns={'country':'number_of_guests_canceled','index':'country'}
)

temp_df = pd.merge(temp_df, temp_canceled_df, how='left', on=['country'])
temp_df['number_of_guests_canceled'].fillna(0, inplace=True)
# temp_df['cancaled_rate'] = temp_df['number_of_guests_canceled']/temp_df['number_of_guests']
temp_df['new_country_name'] = temp_df['country'].apply(
    lambda x: x if x not in unknown_countries else 'UNK'                                       
)
temp_df = temp_df.groupby(['new_country_name']).agg({
    'number_of_guests_canceled':'sum',
    'number_of_guests':'sum'
}).reset_index()
temp_df['cancaled_rate'] = temp_df['number_of_guests_canceled']/temp_df['number_of_guests']
temp_df = temp_df.sort_values(['cancaled_rate'])

fig, ax = plt.subplots(1,1,figsize=(10,7))
rect1 = ax.bar(height=temp_df['cancaled_rate'], 
        x=temp_df['new_country_name'])
def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('%.2f'%(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
autolabel(rect1)
# ax.set_xticklabels()
plt.title("Cancellation rate of each country")
# plt.legend(fontsize=10)
plt.show()

temp_not_cancel_df = df[df['is_canceled']==0]['country'].value_counts().reset_index().rename(
    columns={'country':'number_of_guests','index':'country'}
)
temp_not_cancel_df['new_country_name'] = temp_not_cancel_df['country'].apply(
     lambda x: x if x not in unknown_countries else 'UNK'        
)
temp_not_cancel_df = temp_not_cancel_df.groupby(['new_country_name']).agg({
    'number_of_guests':'sum'
}).reset_index().sort_values(['number_of_guests'])

fig, ax = plt.subplots(1,1,figsize=(10,7))
rect1 = ax.bar(height=temp_not_cancel_df['number_of_guests'], 
        x=temp_not_cancel_df['new_country_name'])
def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('%d'%(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
autolabel(rect1)
# ax.set_xticklabels()
plt.title("Number of real guests of each country")
# plt.legend(fontsize=10)
plt.show()
temp_df = df.groupby(['hotel', 'arrival_date_month']).agg({
    'adr':'sum',
    'adults':'sum',
    'children':'sum'
}).reset_index()
temp_df['adr_ppn'] = temp_df['adr'] / (temp_df['adults'] + temp_df['children'])

ordered_months = ["January", "February", "March", "April", "May", "June", 
          "July", "August", "September", "October", "November", "December"]
temp_df["arrival_date_month"] = pd.Categorical(temp_df["arrival_date_month"], 
                                                          categories=ordered_months, ordered=True)
plt.figure(figsize=(12, 8))
sns.lineplot(x = "arrival_date_month", y="adr_ppn", hue="hotel", data=temp_df, 
            hue_order = ["City Hotel", "Resort Hotel"], ci="sd", size="hotel", sizes=(2.5, 2.5))
plt.title("Room price per night and person over the year", fontsize=16)
plt.xlabel("Month", fontsize=16)
plt.xticks(rotation=45)
plt.ylabel("Price [EUR]", fontsize=16)
plt.show()
temp_df = df.groupby(['hotel', 'arrival_date_month']).agg({
    'is_canceled':'sum'
}).reset_index()

ordered_months = ["January", "February", "March", "April", "May", "June", 
          "July", "August", "September", "October", "November", "December"]
temp_df["arrival_date_month"] = pd.Categorical(temp_df["arrival_date_month"], 
                                                          categories=ordered_months, ordered=True)
plt.figure(figsize=(12, 8))
sns.lineplot(x = "arrival_date_month", y="is_canceled", hue="hotel", data=temp_df, 
            hue_order = ["City Hotel", "Resort Hotel"], ci="sd", size="hotel", sizes=(2.5, 2.5))
plt.title("Cancellation number over the year", fontsize=16)
plt.xlabel("Month", fontsize=16)
plt.xticks(rotation=45)
plt.ylabel("Number of cancellations", fontsize=16)
plt.show()
fig, ax = plt.subplots(1,1, figsize=(8,5))
sns.boxplot(x='is_canceled', y='lead_time', hue='hotel',data=df,
            hue_order=["City Hotel", "Resort Hotel"],
            fliersize=0)
plt.title('Relationship between Leading time and Cancellation ratio')
plt.show()

fig, ax = plt.subplots(1,1, figsize=(8,5))
sns.boxplot(hue='is_canceled', y='adr', x='hotel',data=df,
            hue_order=[0, 1],
            fliersize=0)
ax.set_ylim(0,400)
plt.title('Relationship between ADR and Cancellation ratio')
plt.legend(loc='upper right')
plt.show()

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, precision_score, accuracy_score, recall_score
import random
pre_df = df.copy()
pre_df['country'] = pre_df['country'].apply(
    lambda x: x if x not in unknown_countries else 'UNK'                                       
)
pre_df.drop(columns=['reservation_status', 'reservation_status_date', 'company', 'days_in_waiting_list'], 
            axis=1, inplace=True)
pre_df['agent'].fillna('UNK', inplace=True)
pre_df['agent'] = pre_df['agent'].astype('str')
pre_df['country'].fillna('UNK', inplace=True)
month_dict = {v:k+1 for k,v in dict(enumerate(ordered_months)).items()}

pre_df['arrival_date_month'] = pre_df['arrival_date_month'].apply(lambda x: month_dict[x])
pre_df['arrival_weekday'] = pre_df.apply(lambda x: datetime.datetime(x['arrival_date_year'],
                                                           x['arrival_date_month'],
                                                           x['arrival_date_day_of_month']
                                                           ).weekday(), axis=1)
pre_df.drop(columns=['arrival_date_year', 'arrival_date_day_of_month', 
                     'arrival_date_week_number', 'assigned_room_type'], 
            axis=1, inplace=True)
pre_df['reserved_room_type'] = pre_df.apply(lambda x: x['hotel'] + x['reserved_room_type'], axis=1)
categorical_columns = [
    'hotel', 'arrival_date_month', 'meal', 'country', 'market_segment', 
    'distribution_channel', 'is_repeated_guest', 'reserved_room_type',                   
    'deposit_type', 'customer_type', 'arrival_weekday', 'agent'
]
numeric_columns = [
    'lead_time', 'stays_in_weekend_nights', 'stays_in_week_nights', 'adults', 'children', 'babies',
    'previous_cancellations', 'previous_bookings_not_canceled', 'booking_changes', 'adr',
    'required_car_parking_spaces', 'total_of_special_requests'
]
label_encoder = LabelEncoder()
for col in categorical_columns:
    try:
        pre_df[col] = label_encoder.fit_transform(pre_df[col])
    except Exception as e:
        print(col)
        break
X, Y = pre_df[categorical_columns + numeric_columns], pre_df['is_canceled']
train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.2, random_state=10)
print(train_x.shape)
print(test_x.shape)
