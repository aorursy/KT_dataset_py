# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.

res_df = pd.read_excel("../input/Data_Train.xlsx")
res_df1 = res_df.copy()

res_df1.head()
res_df1.info()
#city_count = res_df1.groupby('CITY').count()

#city_count[city_count['LOCALITY'] > 3]

res_df1.replace(to_replace = ['Bangalor','Bangalore-560066','Bengalore','Bengaluru','Banglore',

                              '5th Main Teachers Colony Koramangala Block 1 Bangalore 560034',

                              'BTM Layout','JP Nagar Bangalore','Whitefield Bangalore',

                              'Bangalore - 560076','Bangalore.',

                              'Banaswadi (Next to Indian Bank) Bangalore','BTM Bangalore',

                              'Bangalore Koramangala 7th Block','bangalore : 560085',

                              'CPR layout harlur main road opposite to ozone ever green apartment Bangalore -',

                              'Karnataka 560043','Bangalore - 560103','Banglaore',

                              'Kanakapura Road Banglore','BTM 1st Stage','Marathahalli',

                              'Bangalore land mark above mahaveer hard ware','Phase 1 Bangalore',

                              'Bangalore 560076','Bangalore Road','Karnataka','Karnataka 560103',

                              'SG Palya','Karnataka 560102','Karnataka 560037', 'B.B.M.P East (Karnataka) - 560049',

                              'Malleshwaram Bangalore', 'Kadubesanahalli Bangalore',

                             'Mahadevpura','NEW BEL ROAD 560054','JP Nagar','Rajarajeshwari Nagar bangalore',

                              'Ulsoo','1st Tavarekere','Thanisandra','Indiranagar','HSR Layout'], 

                                value='Bangalore',inplace=True)



res_df1.replace(to_replace = ['Chennai Teynampet','Tamil Nadu','Besant Nagar','Arumbakkam chennai-600106.',

                              'Avadi', 'Velachery','Pallavaram','Chennai 600034.','Chennai - 600040', 'Perungudi',

                              'Chennai Padur', 'Medavakkam', 'Chennai Kovalam', 'Chennai opp: Vasanth & co',  

                              'Mogappair. Chennai', 'Chennai Perungudi', 'Chennai Thuraipakkam', 'OMR Karapakkam',

                              'Chennai Thousand Lights',  'Chennai- 600107', 'ECR NEELANKARAI Chennai 600115',

                              'Chennai.', 'Chennai (Bang Opposite Indian Bank)','Chennai Opposite 5C Bus stand', 

                              'Mahabalipuram', 'Chennai Mahabalipuram', 'Nungambakkam','Chennai-40',

                              'East Coast Road (ECR)', 'Ramapuram', 'Chennai Chrompet','Nandanam','Thiruvanmiyur', 

                              'Ambattur','Chennai - 34 Landmark - Near Loyola College', 'Anna Nagar West', 

                              'Anna Salai','Chenn ai', 'Perambur', 'Vadapalani','Palavakkam','Sholinganallur',

                              'Mogappair', 'Ashok Nagar', 'Chennai. (Near Hotel Matshya)',  'Chromepet',

                              'chennai','Tambaram','Vadapalani', 'Anna Nagar East','Kilpauk','Potheri',

                              'GST Road', 'Vadapalani.','Semmancheri','Dewan Rama Road','Navallur','Kolathur'],

                                value='Chennai',inplace=True)



res_df1.replace(to_replace=['Telagana Land Line:040-48507016', 'Telangana','Gachibowli','Kondapur', 'Madhapur',

                            'Hyderabad Behind Vacs Pastries','Hyderabad neerus emporium.', 'Hitech City',

                            'Telangana 500003','Hyderabad.', 'Telangana 500034','Telangana 500032',

                            'Hyderabad-500032','Near Santosh Banjara Hyderabad','Gachibowli Hyderabad',

                            'hyderabad', 'Telengana','Telangana 500081','Kondapur.',  'Telangana 500070',

                            'Begumpet Hyderabad','Hyderabad Manikonda','Jubilee Hills'],value='Hyderabad',inplace=True)



res_df1.replace(to_replace = ['Kerala', 'Edappally','Ernakulam Circle kochi','Kochi Elamkulam','Ponnuruni Kochi', 

                              'Kochi Vyttila', 'Kochi Palarivattom','Ernakulam', 'Kochi Chullickal',

                              'Kochi-683101','p.o Kochi','Kerala 683104','Kerala 682013',

                              'MALAPALLIPURAM P .O THRISSUR','Kochi Ravipuram', 'Kerala 690525','Kochi-18',

                              'MG Road Ernakulam','Kochi Kakkanad', 'Kochi International Airport VIP Road', 

                              'Kerala 682001 India', 'Kerala 683585', 'Kerala 682304','Fort Kochi',

                              'Kochi Aluva 102', 'Kerala 682024', 'Kochi','Kakkanad','Kaloor','Palarivattom',

                             'Kochi-16'], 

                                value='Kochi',inplace=True)



res_df1.replace(to_replace = [ 'Secunderabad', 'Secunderabad main road near signal NMREC COLLEGE',

                              'Secunderabad ECIL','Secunderabad. WE HAVE NO BRANCHES.','Secunderabad.'],

                                value='Secunderabad',inplace=True)



res_df1.replace(to_replace = ['Mumbai','Navi Mumbai','Mumbai Mahim','Navi Mumbai.', 'Mumbai - 400007',

                              'Mumbai.','Mumbai Andheri East','Mumbai Dombivali East','navi mumbai',

                              'Mumbai - 400013','West Mumbai','Andheri west Mumbai','Mumbai Chakala',

                              'BK Guda', 'Jogeshwari (w) Mumbai','Mumbai - 80','East Mumbai','Navi-Mumbai',

                              'Mumbai This is a Delivery & Take-away Restaurant only.','Thane Mumbai',

                              'Khar Mumbai','Andheri Lokhandwala','Andheri West','Andheri Lokhandwala.',

                              'Thane','Thane West','Thane (W)','Maharashtra','Maharashtra 400102','Bandra West',

                             'Powai'],

                               value='Mumbai',inplace=True)



res_df1.replace(to_replace=[ 'New Delhi','Delhi','Delhi NCR','Chander Nagar New Delhi','New Delhi.',

                            'New Delhi-110024','Greater Kailash 2 New Delhi',

                            'Amrit kaur market opposite new delhi railway station paharganj',

                            'Delhi 110085','Old Delhi','New Delhi..Near by SBI bank','New Delhi 110075',

                            'Janakpuri', 'Dist. Center New Delhi','Gurugram'],value='New Delhi',inplace=True)



res_df1.replace(to_replace=['Noida','Sector 51 Noida','Greater Noida','Near Sector 110 Noida'],value='Noida',

                            inplace=True)



res_df1.replace(to_replace=['Gurgaon','Gurgoan','Gurgaon Haryana India','Gurgaon Haryana'],value='Gurgaon',inplace=True)













#res_df1['RESTAURANT_ID'].value_counts()

#res_df1.info()

#res_df1[res_df1['RESTAURANT_ID'] == 11204]

#res_df1['TIME'].value_counts()

#res.shape

res_df1.drop(['TIME'],axis=1,inplace=True)

res_df1.dropna(axis=0,inplace=True)

res_df1.head(15)





res_df1['VOTE'] = res_df1['VOTES'].apply(lambda x: str(x)[0:-5])

res_df1['RATE_VOTE'] = np.multiply(pd.to_numeric(res_df1['VOTE']), pd.to_numeric(res_df1['RATING']))

res_df1.drop(['RATING','VOTES'],axis=1,inplace=True)

res_df1.head()
res_df2 = res_df1.drop(['TITLE','RESTAURANT_ID'],axis=1)

res_df2.head()



res_df2 = pd.get_dummies(res_df2)

x = res_df2.drop(['COST'],axis=1)

y = res_df2['COST']

X_train,X_test, Y_train, Y_test = train_test_split(x,y,test_size=0.3,random_state=12)

model = LinearRegression()

model.fit(X_train,Y_train)

print("Coeff:", model.coef_)

print("Intercept:", model.intercept_)

Y_train_predict = model.predict(X_train)

Y_test_predict = model.predict(X_test)

print("Train MSE:", mean_squared_error(Y_train,Y_train_predict))