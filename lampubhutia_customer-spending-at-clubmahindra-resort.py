import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns

import os

print(os.listdir("../input"))
# reading the data



train = pd.read_csv('../input/train_clubmahindra.csv')

test = pd.read_csv('../input/test_clubmahindra.csv')



# getting the shapes of the datasets

print("Shape of Train :", train.shape)

print("Shape of Test :", test.shape)
train_or= train.copy()

test_or=  test.copy()
# Setting up time marker



train['booking_date'] = pd.to_datetime(train.booking_date,format='%d-%m-%Y',infer_datetime_format=True) 

train['checkin_date'] = pd.to_datetime(train.checkin_date,format='%d-%m-%Y',infer_datetime_format=True) 

train['checkout_date'] = pd.to_datetime(train.checkout_date,format='%d-%m-%Y',infer_datetime_format=True) 





test['booking_date'] = pd.to_datetime(test.booking_date,format='%d-%m-%Y',infer_datetime_format=True) 

test['checkin_date'] = pd.to_datetime(test.checkin_date,format='%d-%m-%Y',infer_datetime_format=True) 

test['checkout_date'] = pd.to_datetime(test.checkout_date,format='%d-%m-%Y',infer_datetime_format=True) 
train['Total_traveller'] = (train['numberofadults']+train['numberofchildren'])



test['Total_traveller'] = (test['numberofadults']+test['numberofchildren'])
train.head()
train.describe()
train.info()
# replacing the values in sequential manner.



train['persontravellingid'] = train['persontravellingid'].replace([45, 47, 46, 4752, 4753, 4995], [0, 1, 2, 3, 4, 5])

test['persontravellingid'] = test['persontravellingid'].replace([45, 47, 46, 4752, 4753, 4995], [0, 1, 2, 3, 4, 5])



train['main_product_code'] = train['main_product_code'].replace(7, 0)

test['main_product_code'] = test['main_product_code'].replace(7, 0)

# filling the missing values in the season_holidayed_code.Type attribute of train and test sets

# filling the missing values in the state_code_residence.Type attribute of train and test sets



# season_holidayed_code Type has four types of codes i.e., 1,2,3,4

# but the empty values must be for a different season all that's why it is empty.So lets fill this with 5 in place of null value

# Similarly,state_code_residence is order from 1 to 38, except 17 is missing. So fill this with 17 in place of null value



train['season_holidayed_code'].fillna('0', inplace = True)

test['season_holidayed_code'].fillna('0', inplace = True)

train['state_code_residence'].fillna('17', inplace = True)

test['state_code_residence'].fillna('17', inplace = True)



# let's check if there is any null values still left or not

print("Null values left in the train set:", train.isnull().sum().sum())

print("Null values left in the test set:", test.isnull().sum().sum())



reservation_id = test['reservation_id']

y_train = train.iloc[:, -1]



# let's delete the last column from the dataset to  concat train and test

train = train.drop(['amount_spent_per_room_night_scaled'], axis = 1)



# shape of train

train.shape
# lets concat the train and test sets for preprocessing and visualizations



data = pd.concat([train, test], axis = 0)



# let's check the shape

data.shape
data['season_holidayed_code']=data['season_holidayed_code'].astype(np.int64)

data['state_code_residence']=data['state_code_residence'].astype(np.int64)
# let's check the seasoned holiday code

# 75% customers having booking with holiday code 2 & 3. 

import seaborn as sns

import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (15, 6)

plt.rcParams['font.size'] = 12



plt.subplot(1, 2, 1)

sns.countplot(data['season_holidayed_code'], color = 'blue')

plt.title('Season Code of Holdiay')



plt.subplot(1, 2, 2)

sns.boxplot(x=train['season_holidayed_code'], y=train_or['amount_spent_per_room_night_scaled'])

plt.title('Amount spent for a room per night')



#sns.countplot(data['season_holidayed_code'],  palette = 'PiYG')

data['season_holidayed_code'].value_counts()
plt.rcParams['figure.figsize'] = (18, 6)

plt.rcParams['font.size'] = 12



plt.subplot(1, 2, 1)

sns.countplot(data['state_code_residence'], color = 'blue')

plt.title('Residence state_code')



plt.subplot(1, 2, 2)

sns.scatterplot(x=train['state_code_residence'], y=train_or['amount_spent_per_room_night_scaled'])

plt.title('Amount spent for a room per night')



data['state_code_residence'].value_counts()
# Classifying state code residence into three categories

# Grouping the 1to 9 and 10 to 19 based on residence state code numbers. As majority of residece coming from 1 to 9. 



def groups(state_code_residence):

    if state_code_residence <= 10:

        return 1

    if state_code_residence <= 20 and state_code_residence > 10:

        return 2

    else:

        return 3

data['state_code_residence'] = data.apply(lambda x: groups(x['state_code_residence']), axis = 1)



# checking the values

data['state_code_residence'].value_counts()
data.columns
plt.rcParams['figure.figsize'] = (15, 6)

plt.rcParams['font.size'] = 12



plt.subplot(1, 2, 1)

sns.countplot(data['member_age_buckets'], color = 'green')

plt.title('Distribution of member_age_buckets')



plt.subplot(1, 2, 2)

sns.lineplot(x=train['member_age_buckets'], y=train_or['amount_spent_per_room_night_scaled'])

plt.title('Amount spent for a room per night')
# organizing the value in sequential manner for visualiztion trend. 



data['member_age_buckets'] = data['member_age_buckets'].replace(['D', 'E', 'F', 'C', 'H', 'G', 'B', 'I', 'A', 'J'], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

plt.rcParams['figure.figsize'] = (10, 6)

plt.rcParams['font.size'] = 12



sns.countplot(x="member_age_buckets", data=data)

plt.title('Age Buckets of Members')



data['member_age_buckets'].value_counts()
# Classifying member_age_buckets into three categories



def groups(member_age_buckets):

    if member_age_buckets <= 3:

        return 1

    if member_age_buckets <= 6 and member_age_buckets > 3:

        return 2

    else:

        return 3

data['member_age_buckets'] = data.apply(lambda x: groups(x['member_age_buckets']), axis = 1)



# checking the values

data['member_age_buckets'].value_counts()
data['cluster_code'] = data['cluster_code'].replace('A', 0)

data['cluster_code'] = data['cluster_code'].replace('B', 1)

data['cluster_code'] = data['cluster_code'].replace('C', 2)

data['cluster_code'] = data['cluster_code'].replace('D', 3)

data['cluster_code'] = data['cluster_code'].replace('E', 4)

data['cluster_code'] = data['cluster_code'].replace('F', 5)

plt.rcParams['figure.figsize'] = (15, 6)

plt.rcParams['font.size'] = 12



plt.subplot(1, 2, 1)

sns.countplot(x="cluster_code", data=data)

plt.title('Resort Cluster Code')



plt.subplot(1, 2, 2)

sns.lineplot(x=train_or['cluster_code'], y=train_or['amount_spent_per_room_night_scaled'])

plt.title('Amount spent for a room per night')



data['cluster_code'].value_counts()
data['reservationstatusid_code'] = data['reservationstatusid_code'].replace('A', 0)

data['reservationstatusid_code'] = data['reservationstatusid_code'].replace('B', 1)

data['reservationstatusid_code'] = data['reservationstatusid_code'].replace('C', 2)

data['reservationstatusid_code'] = data['reservationstatusid_code'].replace('D', 3)
plt.rcParams['figure.figsize'] = (15, 6)

plt.rcParams['font.size'] = 12



plt.subplot(1, 2, 1)

sns.countplot(x="reservationstatusid_code", data=data)

plt.title('Reservation Status')



plt.subplot(1, 2, 2)

sns.lineplot(x=train['reservationstatusid_code'], y=train_or['amount_spent_per_room_night_scaled'])

plt.title('Amount spent for a room per night')



data['reservationstatusid_code'].value_counts()

plt.rcParams['figure.figsize'] = (18, 6)

plt.rcParams['font.size'] = 12



# for Booking

data['day of week_booking']=data['booking_date'].dt.dayofweek 

temp = data['day of week_booking']

plt.subplot(1, 3, 1)

sns.distplot(data['day of week_booking'], color = 'blue')

plt.title('Distribution of day of week_booking')



# for Checkin

data['day_of_week_checkin']=data['checkin_date'].dt.dayofweek 

temp1 = data['day_of_week_checkin']

plt.subplot(1, 3, 2)

sns.distplot(data['day_of_week_checkin'], color = 'green')

plt.title('Distribution of day_of_week_checkin')

          

# for Checkout

data['day_of_week_checkout']=data['checkout_date'].dt.dayofweek 

temp1 = data['day_of_week_checkout']

plt.subplot(1, 3, 3)

sns.distplot(data['day_of_week_checkout'], color = 'red')

plt.title('Distribution of day_of_week_checkout')

          
# going for weekend and weekday 



def applyer(row):

    if row.dayofweek == 5 or row.dayofweek == 6:

        return 1

    else:

        return 0 

temp2 = data['booking_date'].apply(applyer) 

temp3 = data['checkin_date'].apply(applyer) 

temp4 = data['checkout_date'].apply(applyer) 

data['weekend_booking']=temp2

data['weekend_checkin']=temp3

data['weekend_checkout']=temp4
# Lets see the weekend and weekday pattern



plt.subplot(1, 3, 1)

sns.distplot(data['weekend_booking'], color = 'blue')

plt.title('Booking distribution of weekend & weekday')





plt.subplot(1, 3, 2)

sns.distplot(data['weekend_checkin'], color = 'green')

plt.title('Checkin distribution of weekend & weekday')





plt.subplot(1, 3, 3)

sns.distplot(data['weekend_checkout'], color = 'red')

plt.title('Checkout distribution of weekend & weekday')

          

# plotting yearly and monthly booking pattern 



plt.rcParams['figure.figsize'] = (15, 6)

plt.rcParams['font.size'] = 14

data['booking_date'] = pd.to_datetime(data['booking_date'], errors = 'coerce')



# extracting the year of booking of the customers

plt.subplot(1, 2, 1)

data['Year_of_booking'] = data['booking_date'].dt.year

sns.distplot(data['Year_of_booking'], color = 'blue')

plt.title('Distribution of Year of booking')



# extracting the month of checkin of the customers

plt.subplot(1, 2, 2)

data['Month_of_booking'] = data['booking_date'].dt.month



sns.distplot(data['Month_of_booking'], color = 'blue')

plt.title('Distribution of Month of Booking')

          
plt.rcParams['figure.figsize'] = (15, 6)

plt.rcParams['font.size'] = 14



data['checkin_date'] = pd.to_datetime(data['checkin_date'], errors = 'coerce')



# extracting the year of checkin of the customers

plt.subplot(1, 2, 1)

data['Year_of_checkin'] = data['checkin_date'].dt.year



sns.distplot(data['Year_of_checkin'], color = 'blue')

plt.title('Distribution of Year of checkin')



# extracting the month of checkin of the customers

plt.subplot(1, 2, 2)

data['Month_of_checkin'] = data['checkin_date'].dt.month



sns.distplot(data['Month_of_checkin'], color = 'blue')

plt.title('Distribution of Month of checkin')

plt.rcParams['figure.figsize'] = (15, 6)

plt.rcParams['font.size'] = 14



data['checkout_date'] = pd.to_datetime(data['checkout_date'], errors = 'coerce')



plt.subplot(1, 2, 1)

# extracting the year of checkout of the customers

data['Year_of_checkout'] = data['checkout_date'].dt.month



# checking the values inside date of year

sns.distplot(data['Year_of_checkout'], color = 'blue')

plt.title('Distribution of Year of checkout')





# extracting the month of checkin of the customers

plt.subplot(1, 2, 2)

data['Month_of_checkout'] = data['checkout_date'].dt.month



sns.distplot(data['Month_of_checkout'], color = 'blue')

plt.title('Distribution of Month of checkout')
print("Total no. of member Ids :", train['memberid'].nunique())

print("Total no. of resort Ids :", train['resort_id'].nunique())

data['resort_id'] = data['resort_id'].replace('4e07408562bedb8b60ce05c1decfe3ad16b72230967de01f640b7e4729b49fce', 1)

data['resort_id'] = data['resort_id'].replace('39fa9ec190eee7b6f4dff1100d6343e10918d044c75eac8f9e9a2596173f80c9', 2)

data['resort_id'] = data['resort_id'].replace('535fa30d7e25dd8a49f1536779734ec8286108d115da5045d77f3b4185d8f790', 3)

data['resort_id'] = data['resort_id'].replace('d4735e3a265e16eee03f59718b9b5d03019c07d8b6c51f90da3a666eec13ab35', 4)

data['resort_id'] = data['resort_id'].replace('b17ef6d19c7a5b1ee83b907c595526dcb1eb06db8227d650d5dda0a9f4ce8cd9', 5)

data['resort_id'] = data['resort_id'].replace('ff5a1ae012afa5d4c889c50ad427aaf545d31a4fac04ffc1c4d03d403ba4250a', 6)

data['resort_id'] = data['resort_id'].replace('0b918943df0962bc7a1824c0555a389347b4febdc7cf9d1254406d80ce44e3f9', 7)

data['resort_id'] = data['resort_id'].replace('a68b412c4282555f15546cf6e1fc42893b7e07f271557ceb021821098dd66c1b', 8)

data['resort_id'] = data['resort_id'].replace('7f2253d7e228b22a08bda1f09c516f6fead81df6536eb02fa991a34bb38d9be8', 9)

data['resort_id'] = data['resort_id'].replace('4ec9599fc203d176a301536c2e091a19bc852759b255bd6818810a42c5fed14a', 10)

data['resort_id'] = data['resort_id'].replace('49d180ecf56132819571bf39d9b7b342522a2ac6d23c1418d3338251bfe469c8', 11)

data['resort_id'] = data['resort_id'].replace('e7f6c011776e8db7cd330b54174fd76f7d0216b612387a5ffcfb81e6f0919683', 12)

data['resort_id'] = data['resort_id'].replace('624b60c58c9d8bfb6ff1886c2fd605d2adeb6ea4da576068201b6c6958ce93f4', 13)

data['resort_id'] = data['resort_id'].replace('3e1e967e9b793e908f8eae83c74dba9bcccce6a5535b4b462bd9994537bfe15c', 14)

data['resort_id'] = data['resort_id'].replace('9f14025af0065b30e47e23ebb3b491d39ae8ed17d33739e5ff3827ffb3634953', 15)

data['resort_id'] = data['resort_id'].replace('e29c9c180c6279b0b02abd6a1801c7c04082cf486ec027aa13515e4f3884bb6b', 16)

data['resort_id'] = data['resort_id'].replace('da4ea2a5506f2693eae190d9360a1f31793c98a1adade51d93533a6f520ace1c', 17)

data['resort_id'] = data['resort_id'].replace('9400f1b21cb527d7fa3d3eabba93557a18ebe7a2ca4e471cfe5e4c5b4ca7f767', 18)

data['resort_id'] = data['resort_id'].replace('48449a14a4ff7d79bb7a1b6f3d488eba397c36ef25634c111b49baf362511afc', 19)

data['resort_id'] = data['resort_id'].replace('6b86b273ff34fce19d6b804eff5a3f5747ada4eaa22f1d49c01e52ddb7875b4b', 20)

data['resort_id'] = data['resort_id'].replace('670671cd97404156226e507973f2ab8330d3022ca96e0c93bdbdb320c41adcaf', 21)

data['resort_id'] = data['resort_id'].replace('f5ca38f748a1d6eaf726b8a42fb575c3c71f1864a8143301782de13da2d9202b', 22)

data['resort_id'] = data['resort_id'].replace('c6f3ac57944a531490cd39902d0f777715fd005efac9a30622d5f5205e7f6894', 23)

data['resort_id'] = data['resort_id'].replace('81b8a03f97e8787c53fe1a86bda042b6f0de9b0ec9c09357e107c99ba4d6948a', 24)

data['resort_id'] = data['resort_id'].replace('c75cb66ae28d8ebc6eded002c28a8ba0d06d3a78c6b5cbf9b2ade051f0775ac4', 25)

data['resort_id'] = data['resort_id'].replace('7902699be42c8a8e46fbbb4501726517e86b22c56a189f7625a6da49081b2451', 26)

data['resort_id'] = data['resort_id'].replace('6208ef0f7750c111548cf90b6ea1d0d0a66f6bff40dbef07cb45ec436263c7d6', 27)

data['resort_id'] = data['resort_id'].replace('ef2d127de37b942baad06145e54b0c619a1f22327b2ebbcfbec78f5564afe39d', 28)

data['resort_id'] = data['resort_id'].replace('8722616204217eddb39e7df969e0698aed8e599ba62ed2de1ce49b03ade0fede', 29)

data['resort_id'] = data['resort_id'].replace('3fdba35f04dc8c462986c992bcf875546257113072a909c162f7e470e581e278', 30)

data['resort_id'] = data['resort_id'].replace('4b227777d4dd1fc61c6f884f48641d02b4d121d3fd328cb08b5531fcacdabf8a', 31)

data['resort_id'] = data['resort_id'].replace('98a3ab7c340e8a033e7b37b6ef9428751581760af67bbab2b9e05d4964a8874a', 32)
plt.rcParams['figure.figsize'] = (10, 6)

#sns.distplot(data['resort_id'], color = 'violet')



plt.subplot(1, 2, 1)

sns.boxplot(x=data['resort_id'], orient='h')

plt.title('Unique Resort ID')

plt.xticks(rotation = 90)



data['resort_id'].value_counts()
plt.rcParams['figure.figsize'] = (15, 15) 

plt.subplot(2, 2, 1)

sns.countplot(data['channel_code'], color = 'violet')

plt.title('Channels of booking')

plt.xticks(rotation = 45)



plt.subplot(2, 2, 2)

sns.countplot(data['main_product_code'], color = 'blue')

plt.title('Type of product a member has purchased')

plt.xticks(rotation = 45)



plt.subplot(2, 2, 3)

sns.countplot(data['numberofadults'], color = 'green')

plt.title('Number of adults travelling')

plt.xticks(rotation = 90)



plt.subplot(2, 2, 4)

sns.countplot(data['numberofchildren'], color = 'green')

plt.title('Number of children travelling')

plt.xticks(rotation = 45)
plt.rcParams['figure.figsize'] = (18, 6) 

plt.subplot(1, 2, 1)

sns.distplot(data['Total_traveller'], color = 'green')

plt.title('Total Number of Travelers')



plt.subplot(1, 2, 2)

sns.scatterplot(x=train['Total_traveller'], y=train_or['amount_spent_per_room_night_scaled'])

plt.title('Amount spent for a room per night')
def groups(Total_traveller):

  if Total_traveller <= 5:

    return 1

  if Total_traveller <= 10 and Total_traveller >5:

    return 2

  else:

    return 3





data['Total_traveller'] = data.apply(lambda x: groups(x['Total_traveller']), axis = 1)

data['Total_traveller'].value_counts()

plt.rcParams['figure.figsize'] = (15, 15)    

plt.subplot(2, 2, 1)

sns.countplot(data['persontravellingid'], color = 'violet')

plt.title('Type of person travelling')

plt.xticks(rotation = 10)



plt.subplot(2, 2, 2)

sns.countplot(data['resort_region_code'], color = 'blue')

plt.title('Resort Region')

plt.xticks(rotation = 45)



plt.subplot(2, 2, 3)

sns.countplot(data['resort_type_code'], color = 'green')

plt.title('Resort Type')

plt.xticks(rotation = 45)



plt.subplot(2, 2, 4)

sns.countplot(data['room_type_booked_code'], color = 'green')

plt.title('Room Type')

plt.xticks(rotation = 45)
plt.rcParams['figure.figsize'] = (15, 15)    

plt.subplot(2, 2, 1)

sns.countplot(data['roomnights'], color = 'violet')

plt.title('Number of roomnights booked')

plt.xticks(rotation = 90)



plt.subplot(2, 2, 2)

sns.countplot(data['season_holidayed_code'], color = 'blue')

plt.title('Season_holidayed')

plt.xticks(rotation = 90)



plt.subplot(2, 2, 3)

sns.countplot(data['state_code_residence'], color = 'green')

plt.title('Residence State of Member')

plt.xticks(rotation = 90)



plt.subplot(2, 2, 4)

sns.countplot(data['state_code_resort'], color = 'green')

plt.title('State in which resort is located')

plt.xticks(rotation = 90)
plt.rcParams['figure.figsize'] = (15, 6)    



plt.subplot(1, 2, 1)

sns.countplot(data['booking_type_code'], color = 'blue')

plt.title('Type of booking')

plt.xticks(rotation = 45)





plt.subplot(1, 2, 2)

sns.boxplot(train['booking_type_code'], y=train_or['amount_spent_per_room_night_scaled'])

plt.title('Type of Booking')





data['booking_type_code'].value_counts()


from sklearn.preprocessing import LabelEncoder



le = LabelEncoder()

data['reservation_id'] = le.fit_transform(data['reservation_id'])

data['memberid'] = le.fit_transform(data['memberid'])
data.dtypes
data=data.drop(['booking_date','checkin_date','checkout_date','reservation_id','persontravellingid','room_type_booked_code', 'season_holidayed_code','state_code_residence', 'state_code_resort','member_age_buckets', 'booking_type_code', 'cluster_code','memberid','reservationstatusid_code', 'resort_id','weekend_booking','weekend_checkin', 'weekend_checkout','day of week_booking','day_of_week_checkin', 'day_of_week_checkout','Year_of_booking','Month_of_booking','channel_code','main_product_code','resort_type_code','resort_region_code','Year_of_booking','Month_of_booking', 'Year_of_checkin', 'Month_of_checkin', 'Year_of_checkout', 'Month_of_checkout'], axis=1)

data.columns
data.head()

#from sklearn import preprocessing

#from sklearn import utils

#print(y_train)

#print(utils.multiclass.type_of_target(y_train))

#print(utils.multiclass.type_of_target(y_train.astype('int'))) 
# separating train and test datasets from data



x_train = data.iloc[:341424,:]

x_test = data.iloc[341424:,:]



# checking the shape of train and test

print("Shape of train :", x_train.shape)

print("Shape of test :", x_test.shape)


from sklearn.model_selection import train_test_split



x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size = 0.2, random_state = 0)
print(x_train.shape)

print(y_train.shape)

print(x_valid.shape)

print(y_valid.shape)
from sklearn.preprocessing import StandardScaler



sc = StandardScaler()

x_train = sc.fit_transform(x_train)

x_valid = sc.transform(x_valid)

x_test = sc.transform(x_test)
# Different parameters we want to test

max_depth = [5,10,15] 

criterion = ['mse']

min_samples_split = [5,10,15]
# Importing GridSearch



from sklearn.model_selection import GridSearchCV
# Building the model



from sklearn import tree





my_tree = tree.DecisionTreeRegressor(random_state = 101)



# Cross-validation tells how well a model performs on a dataset using multiple samples of train data

grid = GridSearchCV(estimator = my_tree, cv=3, 

                    param_grid = dict(max_depth = max_depth, criterion = criterion, min_samples_split=min_samples_split), verbose=2)
grid.fit(x_train,y_train)
# Best accuracy score



print('Avg accuracy score across 27 models:', grid.best_score_)
# Best parameters for the model



grid.best_params_
# Building and fitting Random Forest



from sklearn.ensemble import RandomForestRegressor



forest = RandomForestRegressor(criterion ='mse', n_estimators = 100, max_depth = 10, min_samples_split=5, random_state = 101)
rf_forest = forest.fit(x_train, y_train)
# Print the accuracy score of the fitted random forest



print("RF Accuracy Train:", rf_forest.score(x_train, y_train))

print("RF Accuracy Test:", rf_forest.score(x_valid, y_valid))
from sklearn.metrics import mean_squared_error

y_pred_rf = rf_forest.predict(x_valid)

forest_mse = mean_squared_error(y_valid, y_pred_rf)

forest_rmse = np.sqrt(forest_mse)

print('Random Forest RMSE: %.4f' % forest_rmse)
pred_rf = rf_forest.predict(x_test)
submission = pd.read_csv('../input/submission_CM.csv')

submission['amount_spent_per_room_night_scaled']=pred_rf

submission.to_csv("submission_rf.csv", index=False)

submission.head()
from sklearn import ensemble

from sklearn.ensemble import GradientBoostingRegressor

model = ensemble.GradientBoostingRegressor()

model.fit(x_train, y_train)
print("XGB Accuracy Train:", model.score(x_train, y_train))

print("XGB Accuracy Test:", model.score(x_valid, y_valid))
y_pred_xgb = model.predict(x_valid)

model_mse = mean_squared_error(y_valid,y_pred_xgb)

model_rmse = np.sqrt(model_mse)

print('Gradient Boosting RMSE: %.4f' % model_rmse)
pred_xg = model.predict(x_test)
submission['amount_spent_per_room_night_scaled']=pred_xg

submission.to_csv("submission_xg.csv",index=False)

submission.head()


list(zip(data.columns,model.feature_importances_))