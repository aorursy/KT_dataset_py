array = ['a', 'b', 'c', 'd', 'e', 'a', 'a', 'b', 'c']
print(dict({item: list(filter((lambda x: array[x] == item), range(len(array)))) for item in array}))
my_set = {"A", "B", "C", "D"}
my_set2 = {"A", "E"}

def len_inter(my_set, my_set2):
    count_res = 0
    for item in my_set:
        for nes_item in my_set2:
            if item == nes_item:
                count_res += 1
    return count_res

def len_union(my_set, my_set2):
    return len(my_set) + len(my_set2) - len_inter(my_set, my_set2)

def my_jaccard(my_set, my_set2):
    return len_inter(my_set, my_set2) / len_union(my_set, my_set2)

print(my_jaccard(my_set, my_set2))
import json
import csv

with open("sales.json", "r") as file:
    json_file = file.read()

des_json = json.loads(json_file)
first_line = ["item, country, year, sales"]
res_csv = [first_line]

for index in range(len(des_json)):
    item = des_json[index]["item"]
    map_sales = des_json[index]["sales_by_country"]
    for country in map_sales.keys():
        country_count = map_sales[country]
        for year in country_count:
            res_csv.append(["{}, {}, {}, {}".format(item, country, year, country_count[year])])

with open("result.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerows(res_csv)