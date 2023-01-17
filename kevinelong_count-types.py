import sys

if len(sys.argv) < 2:
    print("USAGE: count_types file_path")
    exit(-1)

file_name = "../input/data.csv"
# file_name = sys.argv[1]
print(file_name)

input_file = open(file_name, "r")
line_list = input_file.readlines()

# Assumptions
field_names_in_header_row = True
all_rows_must_have_same_number_of_fields = True
no_field_values_have_commas_in_them = True

# TODO Consider using CSV import library like pandas

count_dict = {}

# line_list = data.split("\n")

DEVICE_MODEL_COLUMN_INDEX = 2  # Third column/field
field_names = []
# Enumerate supplies an index without having to manually create a counter and increment it
for index, line in enumerate(line_list):
    if 0 == index:
        field_names = line.split(",")
    else:
        values = line.split(",")
        if len(values) != len(field_names):
            print(f"Line {index} has different number of fields than header.")
            continue #skip this line
        #good to go
        if values[DEVICE_MODEL_COLUMN_INDEX] not in count_dict:
            count_dict[values[DEVICE_MODEL_COLUMN_INDEX]] = 1
        else:
            count_dict[values[DEVICE_MODEL_COLUMN_INDEX]] += 1
        # OR IN ONE LINE
        # count_dict[values[MODEL_INDEX]] = 1 if values[MODEL_INDEX] not in count_dict else count_dict[values[MODEL_INDEX]] + 1

print(count_dict)

