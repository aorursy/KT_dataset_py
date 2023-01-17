data = [
    [11, 22, 33],
    [44, 55, 66],
    [77, 88, 99],
]

print(data[1][2])  # row index 1 and column index 2 - prints what? 66

# data[1][-1] = 500
data[1][2] = 500

def show_grid(data):
    for row in data:
        for column in row:
            print(column, end=" ")
        print("")  # blank line
# show_grid(data)


def export_grid(data):
    export_rows = []
    for row in data:
        output_row = []
        for column in row:
#             output_row.append("\"" + str(column) + "\"")
            output_row.append(f'"{column}"')
        text = ",".join(output_row)
        export_rows.append(text)
    text_output = "\n".join( export_rows )
    return text_output

# from cool_stuff import export_grid
print(export_grid(data))

########################################
# use a nested for loop and list append to create a grid like the above but a specific
# arbitrary size.

def make_grid(size):
    output_grid = []
    for r in range(size):
        row = []
        for c in range(size):
            row.append([c, r])
        output_grid.append(row)
    return output_grid

result = make_grid(9)
show_grid(result)
print(result)

text_result = export_grid(result)

file_name = "export.csv"
output_file = open(file_name, "w")
output_file.write(text_result)
output_file.close()

# expected_output = [
#     [0,0,0,0,0,0,0,0,0,],
#     [0,0,0,0,0,0,0,0,0,],
#     [0,0,0,0,0,0,0,0,0,],
#     [0,0,0,0,0,0,0,0,0,],
#     [0,0,0,0,0,0,0,0,0,],
#     [0,0,0,0,0,0,0,0,0,],
#     [0,0,0,0,0,0,0,0,0,],
#     [0,0,0,0,0,0,0,0,0,],
#     [0,0,0,0,0,0,0,0,0,],
# ]

