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
import numpy as np
from datetime import datetime, timedelta
from collections import Counter

# Import data
df = pd.read_csv("/kaggle/input/order-brush-order/order_brush_order.csv")

# Drop orderid (might as well)
df = df.drop(['orderid'], axis=1)

# Sort df by shopid
df = df.sort_values('shopid', ascending=True)
df = df.reset_index(drop='True')
# print(df.head())

# Create new column called one hour later
df['event_time'] = pd.to_datetime(df['event_time'])
df['one_hour_later'] = df['event_time'] + timedelta(hours=1)
# print(df.head())

# Get list of unique shop id
unique_shop_id = df['shopid'].unique()
# print(unique_shop_id)
# print(len(unique_shop_id))  # 18770

# answer_dict stores shopid: userid
answer_dict = {}

print('Ok!')


# Define the functions which I am going to use

# unique_only takes a list and return only the unique values
def unique_only(buyer_list):
    return_list = []
    buyer_list.sort()
    for x in buyer_list:
        if x not in return_list:
            return_list.append(x)
    return return_list

# function to convert dictionary into output.csv
def get_output():
    suspicious_shops = 0
    # Reformat the answer dict using this
    for k, v in answer_dict.items():
        if len(v) == 0:
            answer_dict[k] = 0
        elif len(v) == 1:
            answer_dict[k] = v[0]
            suspicious_shops += 1
        else:
            suspicious_shops += 1
            new_str = ''
            unique_user_id = unique_only(v)
            if len(unique_user_id) > 1:
                for each_id in unique_user_id:
                    new_str += str(each_id) + '&'
                new_str = new_str[:-1]
                answer_dict[k] = new_str
            else:
                answer_dict[k] = v[0]

    # Convert dictionary into dataframe
    answer_df = pd.DataFrame(list(answer_dict.items()), columns=['shopid', 'userid'])
    answer_df.shopid = answer_df.shopid.astype(np.int32)

    # Output answer as output.csv
    answer_df.to_csv("submission.csv", index=False)
    print(suspicious_shops)

# Main program will attempt to window the dataframe in order to detect order brushing
# self.pointer will track the current row, which forms the top window
# self.selector will track the row which has event_time within an hour of the top window
# this assumes that the event_time is the start of order brushing window
# then, the program will check if order concentration >= 3
# and store the userid responsible in self.full_buyer_list if there is order concentration
# self.full_buyer_list > stores userid by the number of times the userid appears
# if there's no order concentration, self.selector will do a decrement
# this narrows the window. 

class MainApp:
    def __init__(self, input_df):
        # df initialization
        self.df = input_df
        self.temp_df = None
        
        # create suspicious_user_list to reduce runtime
#         self.suspicious_users_list = []
        
        # windowing pointers
        self.pointer = 0
        self.selector = 0
        self.checked_window = [0, 0]

        # relevant values for calculations
        self.no_of_unique_users = 0
        self.total_orders = 0
        self.order_volume = 0
        self.full_buyer_list = []
        self.buyer_dict = {}
        self.order_proportion_dict = {}

    def check_unique_users(self, temp_df):
        grouped_df = temp_df.groupby('userid').count()
        grouped_df = grouped_df.sort_values('event_time', ascending=False)
        if grouped_df.event_time.iloc[0] > 2:
#             suspicious_users_df = grouped_df[grouped_df['event_time']>2]
#             suspicious_users_df = suspicious_users_df.reset_index()
#             self.suspicious_users_list = suspicious_users_df.userid.tolist()
            return True
        else:
            return False

    def check_order_brush(self):
        order_brushing = False  # boolean value to return
        self.selector = self.pointer # reset self.selector

        # Iterate down self.df to check the columns that are within one hour period
        window_end_time = self.df.one_hour_later.iloc[self.pointer]
        for event_time in self.df['event_time'][self.pointer+1:]:
            if event_time <= window_end_time:
                self.selector += 1
            else:
                break

        if self.selector - self.pointer < 2:
            return order_brushing

        self.temp_df = self.df.iloc[self.pointer:self.selector+1]
        
        repeated = False
        while self.selector - self.pointer > 1:
            # temp_df is the current window 
            self.temp_df = self.df.iloc[self.pointer:self.selector+1]
                
            if self.check_unique_users(self.temp_df):
#                 while loop below doesn't work 
#                 while self.df.userid.iloc[self.selector] not in self.suspicious_users_list:
#                     # if the last userid in self.tempdf is not a suspicious user
#                     # we will skip that userid
#                     self.selector -= 1
#                     continue
#                 self.temp_df = self.df.iloc[self.pointer:self.selector+1]  # redefine self.temp_df
                
                if repeated:
                    # check for other possible order brushing start times
                    if self.pointer > 0:
                        new_start_time_limit = self.df.event_time.iloc[self.pointer - 1]
                    else:
                        new_start_time_limit = self.df.event_time.iloc[0] - timedelta(hours=1)
                    
                    # Assume that the last item in the current window is the end time
                    check_start_window = self.temp_df.event_time.iloc[-1] - timedelta(hours=1)
                    should_break = False
                    if check_start_window < new_start_time_limit:
                        should_break = True
                    
                    if should_break:
                        # Else, assume that the row after the last item is the end time
                        check_start_window = self.df.event_time.iloc[self.selector + 1] - timedelta(hours=1)
                        if check_start_window > new_start_time_limit:
                            should_break = False
                    
                    if should_break:
                        break
                        

                if self.check_order_concentrate():
                    if self.selector > self.checked_window[1]:
                        if self.pointer > self.checked_window[0] and self.pointer < self.checked_window[1]:
                            break
                        else:
                            self.checked_window = [self.pointer, self.selector]
                            order_brushing = True
                            break
                    else:
                        break
            else:
                break
            
            repeated = True
            self.selector -= 1
            continue

        return order_brushing

    def check_order_concentrate(self):
        # order_concentration = order_volume / no_of_unique_users
        self.order_volume = len(self.temp_df)
        self.full_buyer_list = []
        for each_user in self.temp_df['userid']:
            self.full_buyer_list.append(each_user)

        unique_users_list = unique_only(self.full_buyer_list)
        self.no_of_unique_users = len(unique_users_list)

        order_concentration = self.order_volume / self.no_of_unique_users
        if order_concentration >= 3:
            return True
        else:
            return False

    def store_userid(self):
        # store userid and frequency of orders into self.buyer_dict
        c = Counter(self.full_buyer_list)
        freq_list = c.most_common(self.no_of_unique_users)
        if self.no_of_unique_users > 1:
            max_freq = 0
            for number in range(self.no_of_unique_users):
                if freq_list[number][1] > max_freq:
                    max_freq = freq_list[number][1]
                    most_freq_users = []
                    for times in range(max_freq):
                        most_freq_users.append(freq_list[number][0])
                elif freq_list[number][1] == max_freq:
                    for times in range(max_freq):
                        most_freq_users.append(freq_list[number][0])
                else:
                    break
        else:
            most_freq_users = self.full_buyer_list

        # self.buyer_dict = {userid: frequency}
        self.total_orders += self.order_volume
        for ee in most_freq_users:
            try:
                self.buyer_dict[ee] += 1
            except KeyError:
                self.buyer_dict[ee] = 1
        return

    def check_order_proportion(self):
        # Once all the rows in self.df has been checked,
        # calculate order proportion of each userid identified
        for key, value in self.buyer_dict.items():
            self.order_proportion_dict[key] = value / self.total_orders

        # Do a loop to keep only user(s) with the highest order proportion
        # Store those users into user_id_answer
        highest_order_proportion = 0
        user_id_answer = []
        if len(self.order_proportion_dict.keys()) > 0:
            for key, value in self.order_proportion_dict.items():
                if value > highest_order_proportion:
                    highest_order_proportion = value
                    user_id_answer = [key]
                elif value == highest_order_proportion:
                    user_id_answer.append(key)

        # Store the answers into answerid, if any
        if len(user_id_answer) > 0:
            for ele in user_id_answer:
                answer_dict[self.df.shopid.iloc[0]].append(ele)
        return

    def run(self):
        # combines all the above functions 
        while len(self.df) - self.pointer > 2:
            if self.check_order_brush():
                self.store_userid()
            self.pointer += 1

        if self.buyer_dict != {}:
            self.check_order_proportion()
        return
    
print('Ok!')
startTime = datetime.now()

itera = 0  # track progress
for each in unique_shop_id:
    itera += 1
    print("Iteration: ", itera)
    answer_dict[each] = []
    cond = df['shopid'] == each
    df_by_shop = df[cond]
    df_by_shop = df_by_shop.sort_values('event_time', ascending=True).reset_index(drop='True')
    if len(df_by_shop) > 2:
        temp_grouped_df = df_by_shop.groupby('userid').count()
        temp_grouped_df = temp_grouped_df.sort_values('event_time', ascending=False)
        if temp_grouped_df.event_time.iloc[0] > 2:
#             print(df_by_shop.to_string())
            app = MainApp(df_by_shop)
            app.run()

get_output()

# Check run time
print(datetime.now() - startTime)