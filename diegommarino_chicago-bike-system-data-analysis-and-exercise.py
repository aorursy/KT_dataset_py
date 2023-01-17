# coding: utf-8

# Here goes the imports
import csv
import matplotlib.pyplot as plt

# Let's read the data as a list
print("Reading the document...")
with open("../input/chicago.csv", "r") as file_read:
    reader = csv.reader(file_read)
    data_list = list(reader)
print("Ok!")

# Let's check how many rows do we have
print("Number of rows:")
print(len(data_list))

# Printing the first row of data_list to check if it worked.
print("Row 0: ")
print(data_list[0])
# It's the data header, so we can identify the columns.

# Printing the second row of data_list, it should contain some data
print("Row 1: ")
print(data_list[1])
data_list = data_list[1:]

for i in range(20):
    print(data_list[i])
print("\nTASK 2: Printing the genders of the first 20 samples")
for i in range(20):
    print(data_list[i][6])
def column_to_list(data, index):
    '''Get a dataset and get all the values of a column and return as a list.

    Args:
        data (list(list)): The dataset
        index (int): The column index

    Returns:
        list: List with all values of a column
    '''
    # Tip: You can use a for to iterate over the samples, get the feature by index and append into a list
    column_list = [sample[index] for sample in data]
    return column_list


# Let's check with the genders if it's working (only the first 20)
print("\nTASK 3: Printing the list of genders of the first 20 samples")
print(column_to_list(data_list, -2)[:20])

# ------------ DO NOT CHANGE ANY CODE HERE ------------
assert type(column_to_list(data_list, -2)) is list, "TASK 3: Wrong type returned. It should return a list."
assert len(column_to_list(data_list, -2)) == 1551505, "TASK 3: Wrong lenght returned."
assert column_to_list(data_list, -2)[0] == "" and column_to_list(data_list, -2)[1] == "Male", "TASK 3: The list doesn't match."
# -----------------------------------------------------
male = len([sample[-2] for sample in data_list if sample[-2] == "Male"])
female = len([sample[-2] for sample in data_list if sample[-2] == "Female"])


# Checking the result
print("\nTASK 4: Printing how many males and females we found")
print("Male: ", male, "\nFemale: ", female)

# ------------ DO NOT CHANGE ANY CODE HERE ------------
assert male == 935854 and female == 298784, "TASK 4: Count doesn't match."
# -----------------------------------------------------
def count_gender(data_list):
    '''Count how many males and females have on data_list.

    Args:
        data_list (list): The list with genders

    Returns:
        [int, int]: List with all quanity of males and quantity of females [males_qty, females_qty].
    '''
    male = 0
    female = 0
    for sample in data_list:
        if sample[-2] == "Male":
            male += 1
        elif sample[-2] == "Female":
            female += 1
    return [male, female]


print("\nTASK 5: Printing result of count_gender")
print(count_gender(data_list))

# ------------ DO NOT CHANGE ANY CODE HERE ------------
assert type(count_gender(data_list)) is list, "TASK 5: Wrong type returned. It should return a list."
assert len(count_gender(data_list)) == 2, "TASK 5: Wrong lenght returned."
assert count_gender(data_list)[0] == 935854 and count_gender(data_list)[1] == 298784, "TASK 5: Returning wrong result!"
# -----------------------------------------------------
def most_popular_gender(data_list):
    '''Return the most popular genre name (Male, Female or Equal)

    Args:
        data_list (list): A list with all genres from a data set.

    Returns:
        str: Most repeated genre name.
    '''
    male, female = count_gender(data_list)
    if male > female:
        return "Male"
    elif male < female:
        return "Female"
    else:
        return "Equal"

print("\nTASK 6: Which one is the most popular gender?")
print("Most popular gender is: ", most_popular_gender(data_list))

# ------------ DO NOT CHANGE ANY CODE HERE ------------
assert type(most_popular_gender(data_list)) is str, "TASK 6: Wrong type returned. It should return a string."
assert most_popular_gender(data_list) == "Male", "TASK 6: Returning wrong result!"
# -----------------------------------------------------

# If it's everything running as expected, check this graph!
gender_list = column_to_list(data_list, -2)
types = ["Male", "Female"]
quantity = count_gender(data_list)
y_pos = list(range(len(types)))
plt.bar(y_pos, quantity)
plt.ylabel('Quantity')
plt.xlabel('Gender')
plt.xticks(y_pos, types)
plt.title('Quantity by Gender')
plt.show(block=True)
print("\nTASK 7: Check the chart!")

user_type_list = column_to_list(data_list, -3)
types = ["Subscriber", "Customer"]
quantity = [user_type_list.count("Subscriber"), user_type_list.count("Customer")]
y_pos = list(range(len(types)))
plt.bar(y_pos, quantity)
plt.ylabel('Quantity')
plt.xlabel('User Type')
plt.xticks(y_pos, types)
plt.title('Quantity by User Type')
plt.show(block=True)
male, female = count_gender(data_list)
print("\nTASK 8: Why the following condition is False?")
print("male + female == len(data_list):", male + female == len(data_list))
print("Write your answer: ")
answer = input()
print("Answer:", answer)

# ------------ DO NOT CHANGE ANY CODE HERE ------------
assert answer != "Type your answer here.", "TASK 8: Write your own answer!"
# -----------------------------------------------------

trip_duration_list = list(map(int, column_to_list(data_list, 2)))
list_size = len(trip_duration_list)
trip_duration_list.sort()

if list_size%2 == 0:
    median_trip = (trip_duration_list[list_size // 2 - 1] + trip_duration_list[list_size // 2]) / 2
else:
    median_trip = trip_duration_list[list_size // 2]
min_trip = trip_duration_list[0]
max_trip = trip_duration_list[-1]
mean_trip = sum(trip_duration_list)/list_size

print("\nTASK 9: Printing the min, max, mean and median")
print("Min: ", min_trip, "Max: ", max_trip, "Mean: ", mean_trip, "Median: ", median_trip)

# ------------ DO NOT CHANGE ANY CODE HERE ------------
assert round(min_trip) == 60, "TASK 9: min_trip with wrong result!"
assert round(max_trip) == 86338, "TASK 9: max_trip with wrong result!"
assert round(mean_trip) == 940, "TASK 9: mean_trip with wrong result!"
assert round(median_trip) == 670, "TASK 9: median_trip with wrong result!"
# -----------------------------------------------------
user_types = set(column_to_list(data_list, 3))

print("\nTASK 10: Printing start stations:")
print(len(user_types))
print(user_types)

# ------------ DO NOT CHANGE ANY CODE HERE ------------
assert len(user_types) == 582, "TASK 10: Wrong len of start stations."
# -----------------------------------------------------
print("Will you face it?")
answer = "yes"
print(answer)

def count_items(column_list):
    '''Count how many items of each type a list have

    Args:
        column_list (list): List with all types in a dataset

    Returns:
        [list, list]: First list contain all types names and second list
        contains each item type occurrency.
    '''
    item_types = list(set(column_list))
    count_items = [column_list.count(item_type) for item_type in item_types]

    return item_types, count_items


if answer == "yes":
    # ------------ DO NOT CHANGE ANY CODE HERE ------------
    column_list = column_to_list(data_list, -2)
    types, counts = count_items(column_list)
    print("\nTASK 11: Printing results for count_items()")
    print("Types:", types, "Counts:", counts)
    assert len(types) == 3, "TASK 11: There are 3 types of gender!"
    assert sum(counts) == 1551505, "TASK 11: Returning wrong result!"
    # -----------------------------------------------------
print("Reading data...")
types, quantity = count_items(column_to_list(data_list, 3))
max_type = types[quantity.index(max(quantity))]
min_type = types[quantity.index(min(quantity))]
types = [max_type, min_type]
quantity = [max(quantity), min(quantity)]
y_pos = list(range(len(types)))
plt.bar(y_pos, quantity)
plt.ylabel('Users Quantity')
plt.xlabel('Stations')
plt.xticks(y_pos, types)
plt.title('Quantity by Start Station')
plt.show(block=True)
print("\nTASK 13: Which one is the most popular start station? And less popular?")
print(dict(zip(types, quantity)))
print("How different are the three most popular start stations about users quantity.")
print("Reading data...")
types, quantity = count_items(column_to_list(data_list, 3))
three_max_quantity = []
three_max_types = []
for _ in range(3):
    value_index = quantity.index(max(quantity))
    three_max_quantity.append(quantity.pop(value_index))
    three_max_types.append(types.pop(value_index))

explode = (0.1, 0, 0)  # only "explode" the 2nd slice

fig1, ax1 = plt.subplots()
ax1.pie(three_max_quantity, explode=explode, labels=three_max_types,
        autopct='%1.1f%%', shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()

print(dict(zip(three_max_types, three_max_quantity)))