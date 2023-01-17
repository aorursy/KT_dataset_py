import pandas as pd    ## library for playing with dataframe

import numpy as np    ## library for dealing with array & numeric value

import matplotlib.pyplot as plt   ## Visualize library

import seaborn as sns 



## make notebook more clean by not show the warning

import warnings

warnings.filterwarnings("ignore")



## make dataframe show only 2 digits float 

pd.options.display.float_format = '{:.2f}'.format





%matplotlib inline
data = pd.read_csv('../input/hotel-booking-demand/hotel_bookings.csv')
data.head()
data.columns
plt.rcParams['figure.figsize'] = 10,10

labels = data['hotel'].value_counts().index.tolist()

sizes = data['hotel'].value_counts().tolist()

explode = (0, 0.2)

colors = ['indianred','khaki']



plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',

        shadow=False, startangle=30)

plt.axis('equal')

plt.tight_layout()

plt.title("How many type of hotel in this dataset", fontdict=None, position= [0.48,1], size = 'xx-large')

plt.show()
plt.rcParams['figure.figsize'] = 15,6

plt.hist(data['lead_time'].dropna(), bins=30,color = 'paleturquoise' )



plt.ylabel('Count')

plt.xlabel('Time (days)')

plt.title("Lead time distribution ", fontdict=None, position= [0.48,1.05], size = 'xx-large')

plt.show()
plt.rcParams['figure.figsize'] = 15,8



height = data['is_canceled'].value_counts().tolist()

bars =  ['Not Cancel','Cancel']

y_pos = np.arange(len(bars))

color = ['lightgreen','salmon']

plt.bar(y_pos, height , width=0.7 ,color= color)

plt.xticks(y_pos, bars)

plt.xticks(rotation=90)

plt.title("How many booking was cancel", fontdict=None, position= [0.48,1.05], size = 'xx-large')

plt.show()

plt.rcParams['figure.figsize'] = 15,6



plt.hist(data['stays_in_week_nights'][data['stays_in_week_nights'] < 10].dropna(), 

         bins=8,alpha = 1,color = 'lemonchiffon',label='Stays in week night' )



plt.hist(data['stays_in_weekend_nights'][data['stays_in_weekend_nights'] < 10].dropna(),

         bins=8, alpha = 0.5,color = 'blueviolet',label='Stays in weekend night' )



plt.ylabel('Count')

plt.xlabel('Time (days)')

plt.title("Stays in Week Night vs Weekend Night ", fontdict=None, position= [0.48,1.05], size = 'xx-large')

plt.legend(loc='upper right')

plt.show()


plt.rcParams['figure.figsize'] =10,10

sizes = data['agent'].value_counts()[:8].tolist() + [len(data) - sum(data['agent'].value_counts()[:8].tolist())]

labels = ["Agent " + str(string) for string in data['agent'].value_counts()[:8].index.tolist()] + ["Other"]



explode = (0.18,0.11,0.12,0,0,0,0,0,0,0,0)

colors =  ['royalblue','mediumaquamarine','moccasin'] +['linen']*7 + ['oldlace']



plt.pie(sizes, explode = explode, colors = colors ,labels=labels, autopct='%1.1f%%',

        shadow=False, startangle=96)

plt.axis('equal')

plt.tight_layout()

plt.title("Who is the best agent", fontdict=None, position= [0.5,1], size = 'xx-large')



plt.show()
print("The highest value is : ", data['adr'].max())

print("The lowest value is : ", data['adr'].min())
Q1 = data['adr'].quantile(0.25)

Q3 = data['adr'].quantile(0.75)

IQR = Q3 - Q1



lower_bound = (Q1 - 1.5 * IQR)

upper_bound = (Q3 + 1.5 * IQR)
without_outlier = data[(data['adr'] > lower_bound ) & (data['adr'] < upper_bound)]
plt.boxplot(without_outlier['adr'],  notch=True,  # notch shape

                         patch_artist=True,

                   boxprops=dict(facecolor="sandybrown", color="black"),)

plt.ylabel('ADR')

plt.title("Box plot for Average Daily Rate ", fontdict=None, position= [0.48,1.05], size = 'xx-large')



plt.show()
plt.rcParams['figure.figsize'] = 15,8



height = data['reserved_room_type'].value_counts().tolist()

bars =  data['reserved_room_type'].value_counts().index.tolist()

y_pos = np.arange(len(bars))

color= ['c']+['paleturquoise']*10

plt.bar(y_pos, height , width=0.7 ,color= color)

plt.xticks(y_pos, bars)

plt.ylabel('Count')

plt.xlabel('Roomtype')

plt.title("How many reserves in each type of room", fontdict=None, position= [0.48,1.05], size = 'xx-large')

plt.show()

data.is_canceled.value_counts()
plt.rcParams['figure.figsize'] = 10,10

labels = ['Not','Cancel']

sizes = data['is_canceled'].value_counts().tolist()

explode = (0, 0.2)

colors = ['dodgerblue','tomato']



plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',

        shadow=False, startangle=190)

plt.axis('equal')

plt.tight_layout()

plt.title("How many bookings were cancel", fontdict=None, position= [0.48,1], size = 'xx-large')

plt.show()
input_information = data[['hotel','lead_time','stays_in_week_nights','stays_in_weekend_nights','adults','reserved_room_type','adr'

                          ,'is_canceled']]
input_information.shape
## Binary encoding the categorical data



input_information = pd.get_dummies(data=input_information)
input_information.shape
Y_train = input_information["is_canceled"]

X_train = input_information.drop(labels = ["is_canceled"],axis = 1)
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import  cross_val_score,GridSearchCV



Rfclf = RandomForestClassifier(random_state=15)

Rfclf.fit(X_train, Y_train)
clf_score = cross_val_score(Rfclf, X_train, Y_train, cv=10)

print(clf_score)
clf_score.mean()
Rfclf_fea = pd.DataFrame(Rfclf.feature_importances_)

Rfclf_fea["Feature"] = list(X_train) 

Rfclf_fea.sort_values(by=0, ascending=False).head()
g = sns.barplot(0,"Feature",data = Rfclf_fea.sort_values(by=0, ascending=False)[0:5], palette="Pastel1",orient = "h")

g.set_xlabel("Weight")

g = g.set_title("Random Forest")