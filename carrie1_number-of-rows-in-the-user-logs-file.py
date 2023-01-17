import csv

file = '../input/user_logs.csv'

print("Number of rows in user logs file: ")

count = 0

with open(file, 'r') as count_file:

    csv_reader = csv.reader(count_file)

    for row in csv_reader:

        count += 1

print(count)