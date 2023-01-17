import pandas as pd

import numpy as np



import random



from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score



lr_model = LinearRegression()
# Dataframe Maker

# Declare a string of column names and range of values with which the dataframe is to be created



content = """

Cols:

A. Hours_Spent_in_Reading     		(1-10)

B. Hours_Spent_in_Games        		(1-20)

C. Hours_Spent_in_TV				(1-10)

D. Hours_Spent_in_Assignments 		(1-30)

E. Score 							(1-100)

-------

Calculation:

E = (A * 2.2) - (B * 1.2) + (C * 1.3) + (D * 1.4)

"""
class DFMaker:

    def __init__(self, content, n_rows):

        self.content = content

        self.n_rows = n_rows

        self.error_value = 0 #range 1 - 5

        

        

    def get_dataframe(self, min_acc, max_acc):

        

        while True:



            def get_column_conf(_content):



              # Get columns

                content_parts = _content.split("-------") 



                col_content = content_parts[0]

                col_content = col_content.replace('Cols:', '')



                col_content_parts = col_content.splitlines(True)



                col_list_length = len(col_content_parts)



                col_list = []

                col_conf_map = {}

                for col_index, col in enumerate(col_content_parts):



                # if the content doesn't have (), skip it

                    if('(' not in col):

                        continue



                    col = col.replace('\n', '')



                    col_parts = col.split('(')



                    col_name = col_parts[0]

                    col_range = col_parts[1]



                    col_range = col_range.replace(')', '')



                    # print('col_range : ', col_range)



                    col_ranges = col_range.split('-')



                    col_range_min = int(col_ranges[0])

                    col_range_max = int(col_ranges[1])



                    # print('s : [', col, ']')



                    col_code = col_name.split('. ')[0].strip()

                    col_name = col_name.split('. ')[1].strip()



                    # print(f'col_code : {col_code}, col : {col_name}, min : {col_range_min}, max : {col_range_max}')



                    # Choose whether this is independent or target variable

                    col_type = 'independent'



                    if(col_index == (col_list_length-1)):

                        col_type = 'target'



                    current_col_map = {

                        'code' : col_code,

                        'name' : col_name,

                        'range_min' : col_range_min,

                        'range_max' : col_range_max,

                        'col_type' : col_type

                    }



                    col_conf_map[col_code] = {

                        'name' : col_name,

                        'range_min' : col_range_min,

                        'range_max' : col_range_max,

                        'col_type' : col_type

                    }



                    col_list.append(current_col_map)



                return col_list, col_conf_map



            def get_col_code_by_name(col_name):



                for col in col_conf_list:

                    if(col['name'] == col_name):

                        return col['code']



                return None

            def get_col_and_factor(content):



               #print('current content : ', content)



                contents = content.split('*')

                contents[0] = contents[0].strip()

                contents[1] = contents[1].strip()



                col_positive = True



                if('-' in content):

                    col_positive = False



                # print(contents[0])

                # print(contents[1])



                #   print('------------------')



                col_code, col_factor = 1, 2



                col_code = contents[0].strip()

                col_code = col_code.replace('-', '').strip()

                col_code = col_code.replace('+', '').strip()



                col_factor = contents[1]

                col_factor = float(col_factor)



                return col_code, col_factor, col_positive



            def get_calculation_conf(_content):



                # Get columns

                content_parts = _content.split("-------") 



                col_content = content_parts[1]

                col_content = col_content.replace('Calculation:', '')



                col_content_parts = col_content.splitlines(True)



                col_list = []

                for col in col_content_parts:



                # if the content doesn't have (), skip it

                    if('(' not in col):

                        continue



                    #     print(col)



                    col_parts = col.split('=')

                    col_left = col_parts[0].strip()

                    col_right = col_parts[1].strip()



                    # calc_list = []

                    calc_conf_map = {}



                    for cal in col_right.split(')'):

                        cal = cal.strip()



                        if(len(cal) == 0):

                            continue



                        cal = cal.replace('(', '')



                        # print('cal : [', cal, ']', len(cal))



                        col_code, col_factor, col_positive = get_col_and_factor(cal)



                        calc_conf_map[col_code] = {

                            'col_factor' : col_factor,

                            'col_positive' : col_positive

                        }



                        # calc_list.append(col_map)



                        target_col = None

                        inde_cols = None



                return calc_conf_map



            def get_random_numbers(min, max, size, fill_none = False):



                number_list = []



                for i in range(size):



                    if(fill_none):

                        number_list.append(None)

                    else:

                        number_list.append(random.randint(min, max))



                return number_list



            col_conf_list, col_conf_map = get_column_conf(content)

            calc_configuration_map = get_calculation_conf(content)



            users = pd.DataFrame({}) 



            def fill_dataframe(df):

                for col in col_conf_list:



                    range_min = col['range_min']

                    range_max = col['range_max']



                    if(col['col_type'] == 'target'):

                        temp_list = get_random_numbers(range_min, range_max, self.n_rows, True)

                    else:

                        temp_list = get_random_numbers(range_min, range_max, self.n_rows)



                    df[col['name']] = pd.Series(temp_list)



            fill_dataframe(users)



            def get_target_value(x):



                final_value = 0

                for i, v in x.items():

                # print('index: ', i, 'value: ', v)

                    col_name = i

                    col_value = v



                    col_code = get_col_code_by_name(col_name)

                    col_type = col_conf_map[col_code]['col_type']



                    if(col_type == 'target'):

                        continue



                    cal_conf_map = calc_configuration_map[col_code]

                    col_factor = cal_conf_map['col_factor']

                    col_positive = cal_conf_map['col_positive']



                    # print(x['Hours_Spent_in_Reading'])



                    #     print(f'col_name: {col_name}, col_value : {col_value}, col_code : {col_code}, col_factor : {col_factor}')



                    current_value = (col_value) * (col_factor)

                    

                    self.error_value = random.randint(1, 5)

                    if(col_positive):

                        final_value = final_value + current_value + self.error_value

                    else:

                        final_value = final_value - current_value - self.error_value



                return final_value



            users['Score'] = users.apply(get_target_value, axis = 1)



            X = users.iloc[:, :-1]

            y = users.iloc[:, -1]



            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)



            lr_model.fit(X_train, y_train)



            y_pred = lr_model.predict(X_test)



            acc = r2_score(y_test, y_pred)

            

            print("Accuracy for this dataframe is ",acc)



            if(~(acc >= min_acc and acc <= max_acc)):

                break

            

            users['Score'] = None

            

        return users
# Declare the min, max accuracy and number of rows



NUMBER_OF_ROWS = 500



min_accuracy = 80



max_accuracy = 98
dfmaker = DFMaker(content, NUMBER_OF_ROWS)



df = dfmaker.get_dataframe(min_accuracy, max_accuracy)
df