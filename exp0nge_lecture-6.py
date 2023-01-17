import numpy as np
my_list = ["1"]

my_list.append(
    # float(): tries to turn a variable into a floating point number (1.333)
    # startswith: look at the beginning of a string, eg "my_string", and tell you true or false if it starts with the input
    #      x.startswith('N/A') :: if x (string) starts with 'N/A' then true else false.
    #      "Hello World" -> false, "N/A item" -> true
    # np.nan: special value for representing things that are not numbers
    float(variable) if not(variable).startswith('N/A') else np.nan
    # ___true___ if __conditional__ else __false___
    # if your if is evaulated to True, than the first set of ___true___ is run
    # otherwise, run the last set of __false___
)

if variable.startswith('N/A'):
    return np.nan
else:
    return float(variable)

!ls ../input
path = "../input/employee_birthday.csv"
import csv
with open(path) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    print(csv_reader)
line0 = "name, department, birthday month"
line0
line0.split(",")
with open(path) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            columns = ", ".join(row)
            print(f'Column names are', columns)
            line_count += 1
        else:
            print(row)
            # print(f'\t{row[0]} works in the {row[1]} department, and was born in {row[2]}.')
            line_count += 1
    print(f'Processed {line_count} lines.')
with open(path, mode='r') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            print(f'Column names are {", ".join(row)}')
            line_count += 1
        print(row)
        print(row["department"])
#         print(f'\t{row["name"]} works in the {row["department"]} department, and was born in {row["birthday month"]}.')
        line_count += 1
    print(f'Processed {line_count} lines.')
"john smith,1132 Anywhere Lane Hoboken NJ, 07030,Jan 4".split(",")
!cat "../input/employee_addresses.csv"
with open('../input/employee_addresses.csv', mode='r') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            print(row)
            line_count += 1
        print(row)
        line_count += 1
    print(f'Processed {line_count} lines.')

!cat "../input/employee_addresses_quoted.csv"
with open('../input/employee_addresses_quoted.csv', mode='r') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            print(f'Column names are {", ".join(row)}')
            line_count += 1
        print(row)
        line_count += 1
    print(f'Processed {line_count} lines.')

import pandas as pd
df = pd.read_csv('../input/hrdata.csv')
df
df.dtypes
with open('../input/employee_file.csv', mode='w') as employee_file:
    employee_writer = csv.writer(employee_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    employee_writer.writerow(['John Smith', 'Accounting', 'November'])
    employee_writer.writerow(['Erica Meyers', 'IT', 'March'])
from io import StringIO
employee_file = StringIO()
employee_writer = csv.writer(employee_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
employee_writer.writerow(['John Smith', 'Accounting', 'November'])
employee_writer.writerow(['Erica Meyers', 'IT', 'March'])
employee_file.getvalue()
my_dict = {}
my_dict
my_dict['alice'] = 'person'
my_dict
my_dict['alice']
my_dict[1] = 'age'
my_dict
my_dict[1]
my_dict['colors'] = ['red', 'green']
my_dict
my_dict['colors']
type(my_dict['colors'])
my_dict['colors'][0]
len(my_dict['colors'])
my_dict['second_dict'] = {
    'book': 'Red Rising'
}
my_dict
type(my_dict['second_dict'])
my_dict['second_dict']
my_dict['second_dict']['book']
my_dict['second_dict'][0]
my_dict['second_dict']['another_book']
my_dict['second_dict']
my_dict['second_dict']['book2'] = 'Harry Potter'
my_dict
my_dict['second_dict']
my_dict['colors'] = 'green'
my_dict
type(my_dict['colors'])
my_dict['colors'][0]
'green'[0]
import json
json_str = """
{
	"business_id": "qYuSoVBS_1i_btLzFefDeg",
	"full_address": "4520 Potters Rd\nMatthews, NC 28104",
	"hours": {
		"Monday": {
			"close": "22:00",
			"open": "11:00"
		},
		"Tuesday": {
			"close": "22:00",
			"open": "11:00"
		},
		"Friday": {
			"close": "22:00",
			"open": "11:00"
		},
		"Wednesday": {
			"close": "22:00",
			"open": "11:00"
		},
		"Thursday": {
			"close": "22:00",
			"open": "11:00"
		},
		"Sunday": {
			"close": "21:00",
			"open": "13:00"
		},
		"Saturday": {
			"close": "22:00",
			"open": "11:00"
		}
	},
	"open": true,
	"categories": ["Pizza", "Restaurants"],
	"city": "Matthews",
	"review_count": 14,
	"name": "New York Pizza Express",
	"neighborhoods": [],
	"longitude": -80.697074000000001,
	"state": "NC",
	"stars": 4.0,
	"latitude": 35.084142999999997,
	"attributes": {
		"Take-out": true,
		"Wi-Fi": "no",
		"Good For": {
			"dessert": false,
			"latenight": false,
			"lunch": false,
			"dinner": false,
			"brunch": false,
			"breakfast": false
		},
		"Caters": true,
		"Noise Level": "quiet",
		"Takes Reservations": false,
		"Ambience": {
			"romantic": false,
			"intimate": false,
			"touristy": false,
			"hipster": false,
			"divey": false,
			"classy": false,
			"trendy": false,
			"upscale": false,
			"casual": false
		},
		"Has TV": false,
		"Delivery": true,
		"Dogs Allowed": false,
		"Parking": {
			"garage": false,
			"street": false,
			"validated": false,
			"lot": true,
			"valet": false
		},
		"Wheelchair Accessible": true,
		"Outdoor Seating": false,
		"Attire": "casual",
		"Alcohol": "none",
		"Waiter Service": false,
		"Accepts Credit Cards": true,
		"Good for Kids": true,
		"Good For Groups": true,
		"Price Range": 2
	},
	"type": "business"
}
"""
type(json_str)
json_str['categories']
import json
my_json_obj = json.loads(json_str, strict=False)
my_json_obj
type(my_json_obj)
my_json_obj["name"]
my_json_obj["business_name"]
print(my_json_obj.get("business_name"))
my_json_obj.get("business_name", [1, 2])
my_json_obj.get('name')
my_json_obj["attributes"]
my_json_obj["attributes"]["Parking"]
my_json_obj["attributes"]["Parking"]['garage']
my_json_obj["attributes"]["Parking"]['garage']['fee']
my_json_obj["categories"]
my_json_obj["categories"]['Pizza']
my_json_obj["categories"][1]
!ls ../input
json_dataset = []
with open("../input/yelp_academic_dataset_business.json") as f:
    for line in f.readlines():
        
        json_line = json.loads(line, strict=False)
        json_dataset.append(json_line)
len(json_dataset)
json_dataset[:5]
import pandas as pd
df = pd.DataFrame(json_dataset)
df
df.dtypes
df['type'].unique()
df['city'].unique()
def doctors_func(column):
    return 'Doctors' in column
df['categories'].apply(doctors_func)
doctor_businesses =  df[df['categories'].apply(doctors_func)]
doctor_businesses
def appt_func(row):
    # If the key exists: row['By Appointment Only'] == False
    # else: None == False
    return row.get('By Appointment Only') == False
doctor_businesses['attributes'].apply(appt_func)
no_appt_df = doctor_businesses[doctor_businesses['attributes'].apply(appt_func)]
no_appt_df
def phoenix_func(column):
    return 'Phoenix' in column
no_appt_df[no_appt_df.city.apply(phoenix_func)]
no_appt_df.city.unique()
phoenix_df = no_appt_df[no_appt_df.city.apply(phoenix_func)]
phoenix_df.city.unique()
phoenix_df.sort_values('stars')
phoenix_df.sort_values('stars').plot.hist(x = 'name', y = 'stars')
gt_4_df = phoenix_df.query('stars > 4.0').sort_values('stars')
gt_4_df.plot(x = 'name', y = 'stars')
gt_4_df