array = ['a', 'b', 'c', 'd', 'e', 'a', 'a', 'b', 'c']
letter_dict = {}

for i in range(len(array)):
    if letter_dict.get(array[i]) == None:
        letter_dict[array[i]] = [i]
    else:
        letter_dict[array[i]].append(i)
print(letter_dict)
array_to_dict = {element: list(filter(lambda i: array[i] == element, range(len(array)))) for element in array}
print(array_to_dict)
set_a = {'Michelle', 'Nicholas', 'John', 'Mercy'}
set_b = {2.0, 'Nicholas', (1, 2, 3)}

intersection = 0

for i in set_a:
    if i in set_b:
        intersection += 1
        
print(intersection / (len(set_a) + len(set_b) - intersection))
import json
import csv

with open("sales.json") as json_obj, open("result.csv", 'w', newline='') as csv_obj:

    json_data = json.loads(json_obj.read())
    writer = csv.writer(csv_obj)

    first_row = ['item', 'country', 'year', 'sales']
    writer.writerow(first_row)

    for item in json_data:
        for country in item['sales_by_country']:
            for year in item['sales_by_country'][country]:
                row = [item['item'], country, year, item['sales_by_country'][country][year]]
                writer.writerow(row)
import csv
import requests
import xml.etree.ElementTree as ET

currencies = ["R01235", "R01239", "R01270", "R01720"]


def xml_to_csv():
    csv_obj = ["Date", "Dollar", "Euro", "Rupee", "Hryvna"]

    with open("result.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(csv_obj)
        csv_list = []

        for cur in currencies:
            result = requests.get(
                "http://www.cbr.ru/scripts/XML_dynamic.asp?"
                "date_req1=01/03/2020&date_req2=01/07/2020&VAL_NM_RQ={}".format(cur))

            request = ET.fromstring(result.text)
            cur_list = []
            dates = []

            for i in request:
                cur_list.append((round((float(i[1].text.replace(',', '.')) / float(i[0].text)), 3)))
                dates.append(i.attrib['Date'])

            if dates not in csv_list:
                csv_list.append(list(dates))

            csv_list.append(list(cur_list))

        for i in range(len(csv_list[0])):
            help_list = list()
            help_list.append(csv_list[0][i])
            help_list.append(csv_list[1][i])
            help_list.append(csv_list[2][i])
            help_list.append(csv_list[3][i])
            help_list.append(csv_list[4][i])
            writer.writerow(help_list)


xml_to_csv()
array = ['a', 'b', 'c', 'd', 'e', 'a', 'a', 'b', 'c']
letter_dict = {}

for i in range(len(array)):
    letter_dict[array[i]] = [j for j in range(len(array)) if array[j] == array[i]]
print(letter_dict)

array_to_dict = {k: [x for x in range(len(array)) if array[x] == k] for k in array for x in range(len(array)) if array[x] == k}
print(array_to_dict)